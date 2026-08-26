use cuda_core::CudaContext;
use rccl_memory::{CudaPoolMemoryResource, DeviceBuffer, PoolConfig, TrackingMemoryResource};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let pool = CudaPoolMemoryResource::new(
        context,
        PoolConfig {
            release_threshold: 64 * 1024 * 1024,
            ..PoolConfig::default()
        },
    )?;
    let tracked = TrackingMemoryResource::new(&pool);

    let input = [3u32, 1, 4, 1, 5, 9];
    let buffer = DeviceBuffer::from_host_in(&tracked, &stream, &input)?;
    assert_eq!(buffer.to_host_vec()?, input);
    drop(buffer);
    stream.synchronize()?;

    let resource_stats = tracked.stats();
    let pool_stats = pool.stats()?;
    println!(
        "{} allocation(s), peak {} bytes, pool reserved {} bytes",
        resource_stats.total_allocations, resource_stats.peak_bytes, pool_stats.reserved_current
    );
    Ok(())
}
