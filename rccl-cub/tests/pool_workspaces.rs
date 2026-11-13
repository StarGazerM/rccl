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
