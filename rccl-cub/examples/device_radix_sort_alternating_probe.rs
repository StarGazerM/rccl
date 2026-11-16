//! Histogram and scatter must use the same selected input after an odd pass.
use cuda_core::{CudaContext, DeviceBuffer};
use std::{error::Error, fmt::Debug};

fn check<K>() -> Result<(), Box<dyn Error>>
where
    K: rccl_cub::RadixKey + From<u32> + Ord + Debug + Default,
{
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let cub = rccl_cub::cuda::Program::load(&context)?;
    let launch = cub.on(&stream)?;
    for rows in [2304, 2305, 4864, 4865, 8193] {
        for descending in [false, true] {
            let input: Vec<K> = (0..rows).map(|i| K::from(i as u32 & 1)).collect();
            let replacement = vec![K::from(1); rows];
            let ids: Vec<u32> = (0..rows as u32).collect();
            let mut expected: Vec<_> = input.iter().copied().zip(ids.iter().copied()).collect();
            expected.sort_by(|a, b| if descending { b.0.cmp(&a.0) } else { a.0.cmp(&b.0) });
            let expected_ids: Vec<_> = expected.iter().map(|pair| pair.1).collect();
            let mut keys = DeviceBuffer::from_host(&stream, &input)?;
            let mut other_keys = DeviceBuffer::<K>::zeroed(&stream, rows)?;
            let mut values = DeviceBuffer::from_host(&stream, &ids)?;
            let mut other_values = DeviceBuffer::<u32>::zeroed(&stream, rows)?;
            let mut workspace = rccl_cub::device_radix_sort_pairs_buffer_workspace::<K, u32>(&launch, rows)?;
            let mut buffers = rccl_cub::DoubleBuffer::paired((&mut keys, &mut values), (&mut other_keys, &mut other_values)).unwrap();
            for pass in 0..2 {
                if pass == 1 {
                    assert!(!buffers.current_is_first());
                    buffers.current_mut().0.copy_from_host(&stream, &replacement)?;
                }
                if descending {
                    rccl_cub::device_radix_sort_pairs_buffer_descending(&launch, &mut buffers, 0..1, &mut workspace)?;
                } else {
                    rccl_cub::device_radix_sort_pairs_buffer(&launch, &mut buffers, 0..1, &mut workspace)?;
                }
                assert_eq!(buffers.current().1.to_host_vec(&stream)?, expected_ids);
                let expected_keys: Vec<_> = if pass == 0 { expected.iter().map(|p| p.0).collect() } else { replacement.clone() };
                assert_eq!(buffers.current().0.to_host_vec(&stream)?, expected_keys);
            }
            drop(buffers);
            let mut workspace = rccl_cub::device_radix_sort_keys_buffer_workspace::<K>(&launch, rows)?;
            keys.copy_from_host(&stream, &input)?;
            let mut buffers = rccl_cub::DoubleBuffer::new(&mut keys, &mut other_keys).unwrap();
            for pass in 0..2 {
                if pass == 1 {
                    assert!(!buffers.current_is_first());
                    buffers.current_mut().copy_from_host(&stream, &replacement)?;
                }
                if descending {
                    rccl_cub::device_radix_sort_keys_buffer_descending(&launch, &mut buffers, 0..1, &mut workspace)?;
                } else {
                    rccl_cub::device_radix_sort_keys_buffer(&launch, &mut buffers, 0..1, &mut workspace)?;
                }
                let expected_keys: Vec<_> = if pass == 0 { expected.iter().map(|p| p.0).collect() } else { replacement.clone() };
                assert_eq!(buffers.current().to_host_vec(&stream)?, expected_keys);
            }
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    check::<u32>()?;
    check::<u64>()?;
    println!("alternating radix histogram keys/pairs u32/u64: PASS");
    Ok(())
}
