use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

mod gpu_support;

use rccl::{Greater, sort_by};

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par(&stream)?;
    let mut keys = DeviceBuffer::<i64>::zeroed(&stream, 1_000_003)?;

    support.fill_i64(&stream, &mut keys, 0x534d_4152_545f_534f)?;
    sort_by(&exec, &mut keys, Greater)?;
    assert_eq!(support.validate_order(&stream, &keys, Greater)?, 0);
    println!("type-inferred i64 comparator dispatch example: PASS");
    Ok(())
}
