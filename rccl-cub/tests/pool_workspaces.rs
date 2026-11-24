use cuda_core::{CudaContext, DeviceBuffer};
use rccl_memory::{CudaMemoryResource, PoolMemoryResource};

#[test]
fn pooled_radix_and_merge_match_direct_workspaces() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let pool = PoolMemoryResource::new(CudaMemoryResource::new(context.clone()), 1 << 20, 1 << 26)?;
    let program = rccl_cub::cuda::Program::load(&context)?;
    let launch = program.on(&stream)?;
    for len in [0, 17, 20_000] {
        let input: Vec<u64> = (0..len).map(|i| ((i * 7919) % 1009) as u64).collect();
        let mut direct = DeviceBuffer::from_host(&stream, &input)?;
        let mut pooled = DeviceBuffer::from_host(&stream, &input)?;
        let mut old = rccl_cub::device_radix_sort_workspace::<u64>(&launch, len)?;
        let before = pool.stats().allocated_bytes;
        let mut new = rccl_cub::device_radix_sort_workspace_in::<u64, _>(&pool, &launch, len)?;
        if len != 0 { assert!(pool.stats().allocated_bytes > before); }
        rccl_cub::device_radix_sort_keys(&launch, &mut direct, &mut old)?;
        rccl_cub::device_radix_sort_keys(&launch, &mut pooled, &mut new)?;
        assert_eq!(pooled.to_host_vec(&stream)?, direct.to_host_vec(&stream)?);
        let values: Vec<u32> = (0..len as u32).collect();
        let mut direct_keys = DeviceBuffer::from_host(&stream, &input)?;
        let mut pooled_keys = DeviceBuffer::from_host(&stream, &input)?;
        let mut direct_values = DeviceBuffer::from_host(&stream, &values)?;
        let mut pooled_values = DeviceBuffer::from_host(&stream, &values)?;
        let mut old = rccl_cub::device_radix_sort_pairs_workspace::<u64, u32>(&launch, len)?;
        let mut new = rccl_cub::device_radix_sort_pairs_workspace_in::<u64, u32, _>(&pool, &launch, len)?;
        rccl_cub::device_radix_sort_pairs(&launch, &mut direct_keys, &mut direct_values, &mut old)?;
        rccl_cub::device_radix_sort_pairs(&launch, &mut pooled_keys, &mut pooled_values, &mut new)?;
        assert_eq!(pooled_keys.to_host_vec(&stream)?, direct_keys.to_host_vec(&stream)?);
        assert_eq!(pooled_values.to_host_vec(&stream)?, direct_values.to_host_vec(&stream)?);
    }
    let first = DeviceBuffer::from_host(&stream, &[0_u64, 2, 4])?;
    let second = DeviceBuffer::from_host(&stream, &[1_u64, 2, 5])?;
    let mut direct = DeviceBuffer::<u64>::zeroed(&stream, 6)?;
    let mut pooled = DeviceBuffer::<u64>::zeroed(&stream, 6)?;
    let mut old = rccl_cub::device_merge_keys_workspace::<u64>(&launch, 3, 3)?;
    let mut new = rccl_cub::device_merge_keys_workspace_in::<u64, _>(&pool, &launch, 3, 3)?;
    rccl_cub::device_merge_keys(&launch, &first, &second, &mut old, &mut direct, |a: &u64, b: &u64| a.cmp(b))?;
    rccl_cub::device_merge_keys(&launch, &first, &second, &mut new, &mut pooled, |a: &u64, b: &u64| a.cmp(b))?;
    assert_eq!(pooled.to_host_vec(&stream)?, direct.to_host_vec(&stream)?);
    Ok(())
}

#[test]
fn cached_onesweep_lookback_preserves_unused_suffix() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_cub::cuda::Program::load(&context)?;
    let launch = program.on(&stream)?;
    let capacity = 80_000;
    let mut keys_ws = rccl_cub::device_radix_sort_keys_buffer_workspace::<u64>(&launch, capacity)?;
    let mut pairs_ws = rccl_cub::device_radix_sort_pairs_buffer_workspace::<u64, u32>(&launch, capacity)?;
    for len in [20_000, 8_193, 59_999] {
        // Poison stale state before every invocation. Shrinking must not
        // clear the unused suffix; growing again must reset the new prefix.
        for lookback in [keys_ws.__parts_mut().1, pairs_ws.__parts_mut().1] {
            lookback.copy_from_host(&stream, &vec![-1; lookback.len()])?;
        }
        let keys: Vec<_> = (0..len).map(|i| ((i * 7919) % 1009) as u64).collect();
        let ids: Vec<_> = (0..len as u32).collect();
        let mut expected: Vec<_> = keys.iter().copied().zip(ids.iter().copied()).collect();
        expected.sort_by_key(|x| x.0);
        let (sorted, stable_ids): (Vec<_>, Vec<_>) = expected.into_iter().unzip();
        let mut first = DeviceBuffer::from_host(&stream, &keys)?;
        let mut second = DeviceBuffer::zeroed(&stream, len)?;
        let mut buffers = rccl_cub::DoubleBuffer::new(&mut first, &mut second).unwrap();
        rccl_cub::device_radix_sort_keys_buffer(&launch, &mut buffers, 0..10, &mut keys_ws)?;
        assert_eq!(buffers.current().to_host_vec(&stream)?, sorted);

        first.copy_from_host(&stream, &keys)?;
        let mut first_ids = DeviceBuffer::from_host(&stream, &ids)?;
        let mut second_ids = DeviceBuffer::zeroed(&stream, len)?;
        let mut buffers = rccl_cub::DoubleBuffer::paired(
            (&mut first, &mut first_ids), (&mut second, &mut second_ids),
        ).unwrap();
        rccl_cub::device_radix_sort_pairs_buffer(&launch, &mut buffers, 0..10, &mut pairs_ws)?;
        assert_eq!(buffers.current().0.to_host_vec(&stream)?, sorted);
        assert_eq!(buffers.current().1.to_host_vec(&stream)?, stable_ids);
        for (lookback, live) in [
            (keys_ws.__parts_mut().1, rccl_cub::__private::device_radix_sort_workspace_counts::<u64>(len).lookback),
            (pairs_ws.__parts_mut().1, rccl_cub::__private::device_radix_sort_pairs_workspace_counts::<u64, u32>(len).lookback),
        ] {
            assert!(live < lookback.len());
            assert!(lookback.to_host_vec(&stream)?[live..].iter().all(|&word| word == -1));
        }
    }
    Ok(())
}
