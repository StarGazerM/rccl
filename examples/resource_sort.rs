use cuda_core::CudaContext;
use rccl::memory::{CudaPoolMemoryResource, DeviceBuffer, PoolConfig, TrackingMemoryResource};
use std::error::Error;

mod gpu_support;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let pool = CudaPoolMemoryResource::new(context.clone(), PoolConfig::default())?;
    let tracked = TrackingMemoryResource::new(&pool);

    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par_in(&stream, &tracked)?;
    let mut keys = DeviceBuffer::<u32, _>::zeroed_in(&tracked, &stream, 1 << 20)?;

    support.fill_u32(&stream, &mut keys, 42)?;
    rccl::sort(&exec, &mut keys)?;
    let errors = support.validate_order(&stream, &keys, u32::cmp)?;
    if errors != 0 {
        return Err(format!("device validation found {errors} inversions").into());
    }
    stream.synchronize()?;

    let stats = tracked.stats();
    println!(
        "validation: PASS; {} live allocation(s), {} peak bytes",
        stats.outstanding_allocations, stats.peak_bytes
    );
    Ok(())
}
