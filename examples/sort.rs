use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

mod gpu_support;

use rccl::sort;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par(&stream)?;
    let mut keys = DeviceBuffer::<u32>::zeroed(&stream, 1_000_003)?;

    support.fill_u32(&stream, &mut keys, 123_456)?;
    sort(&exec, &mut keys)?;

    assert_eq!(support.validate_order(&stream, &keys, u32::cmp)?, 0);
    println!("generic GPU sort example: PASS");
    Ok(())
}
