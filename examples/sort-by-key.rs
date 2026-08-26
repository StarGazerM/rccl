use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

mod gpu_support;

use rccl::stable_sort_by_key;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par(&stream)?;
    let mut keys = DeviceBuffer::<u32>::zeroed(&stream, 1_000_003)?;
    let mut values = DeviceBuffer::<u32>::zeroed(&stream, keys.len())?;

    support.fill_low_entropy_u32(&stream, &mut keys, 123_456)?;
    support.fill_iota_u32(&stream, &mut values)?;
    stable_sort_by_key(&exec, &mut keys, &mut values)?;

    assert_eq!(
        support.validate_stable_u32_pairs(&stream, &keys, &values, u32::cmp)?,
        0
    );
    println!("generic GPU stable_sort_by_key example: PASS");
    Ok(())
}
