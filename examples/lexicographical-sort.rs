use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

mod gpu_support;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par(&stream)?;
    let mut records = DeviceBuffer::<gpu_support::I32Triple>::zeroed(&stream, 1_000_003)?;

    support.fill_i32_triples(&stream, &mut records, 0x4c45_5849_434f_4752)?;
    rccl::stable_sort_by(&exec, &mut records, gpu_support::I32TripleLess)?;
    assert_eq!(
        support.validate_order(&stream, &records, gpu_support::I32TripleLess)?,
        0
    );
    println!("lexicographical record sort through the public facade: PASS");
    Ok(())
}
