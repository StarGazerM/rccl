use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

mod gpu_support;

fn main() -> Result<(), Box<dyn Error>> {
    const COUNT: usize = 2 * 1024 * 1024;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par(&stream)?;

    let mut records = DeviceBuffer::<gpu_support::I32Pair>::zeroed(&stream, COUNT)?;
    support.fill_i32_pairs(&stream, &mut records, 0x414f_535f_534f_5254)?;
    let aos_start = stream.record_event(Some(0))?;
    rccl::stable_sort_by(&exec, &mut records, gpu_support::I32PairLess)?;
    let aos_end = stream.record_event(Some(0))?;
    assert_eq!(
        support.validate_order(&stream, &records, gpu_support::I32PairLess)?,
        0
    );

    let mut keys = DeviceBuffer::<u32>::zeroed(&stream, COUNT)?;
    let mut values = DeviceBuffer::<u32>::zeroed(&stream, COUNT)?;
    support.fill_low_entropy_u32(&stream, &mut keys, 0x534f_415f_534f_5254)?;
    support.fill_iota_u32(&stream, &mut values)?;
    let soa_start = stream.record_event(Some(0))?;
    rccl::stable_sort_by_key(&exec, &mut keys, &mut values)?;
    let soa_end = stream.record_event(Some(0))?;
    assert_eq!(
        support.validate_stable_u32_pairs(&stream, &keys, &values, rccl::Less)?,
        0
    );

    println!(
        "AoS {:.3} ms; SoA {:.3} ms; both public-facade GPU verdicts",
        aos_start.elapsed_ms(&aos_end)?,
        soa_start.elapsed_ms(&soa_end)?
    );
    Ok(())
}
