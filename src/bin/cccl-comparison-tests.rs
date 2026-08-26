use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

#[path = "../../examples/gpu_support.rs"]
mod gpu_support;

fn main() -> Result<(), Box<dyn Error>> {
    const COUNT: usize = 1_000_003;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par(&stream)?;

    let mut i64_keys = DeviceBuffer::<i64>::zeroed(&stream, COUNT)?;
    support.fill_i64(&stream, &mut i64_keys, 0x4936_345f_4c45_5353)?;
    rccl::stable_sort(&exec, &mut i64_keys)?;
    assert_eq!(support.validate_order(&stream, &i64_keys, i64::cmp)?, 0);

    support.fill_i64(&stream, &mut i64_keys, 0x4936_345f_4d41_534b)?;
    let mask = 0xffff_i64;
    let masked = move |left: &i64, right: &i64| (left & mask).cmp(&(right & mask));
    rccl::stable_sort_by(&exec, &mut i64_keys, masked)?;
    assert_eq!(
        support.validate_masked_i64_order(&stream, &i64_keys, mask)?,
        0
    );

    let mut pairs = DeviceBuffer::<gpu_support::I32Pair>::zeroed(&stream, COUNT)?;
    support.fill_i32_pairs(&stream, &mut pairs, 0x5041_4952)?;
    rccl::sort_by(&exec, &mut pairs, gpu_support::order_i32_pair)?;
    assert_eq!(
        support.validate_order(&stream, &pairs, gpu_support::order_i32_pair)?,
        0
    );

    let mut triples = DeviceBuffer::<gpu_support::I32Triple>::zeroed(&stream, COUNT)?;
    support.fill_i32_triples(&stream, &mut triples, 0x5452_4950_4c45)?;
    rccl::stable_sort_by(&exec, &mut triples, gpu_support::order_i32_triple)?;
    assert_eq!(
        support.validate_order(&stream, &triples, gpu_support::order_i32_triple)?,
        0
    );

    let mut values = DeviceBuffer::<u32>::zeroed(&stream, COUNT)?;
    support.fill_i32_pairs(&stream, &mut pairs, 0x5041_4952_5641_4c53)?;
    support.fill_iota_u32(&stream, &mut values)?;
    rccl::stable_sort_by_key_with(&exec, &mut pairs, &mut values, gpu_support::order_i32_pair)?;
    assert_eq!(
        support.validate_stable_pairs(&stream, &pairs, &values, gpu_support::order_i32_pair)?,
        0
    );

    println!("typed comparison fallback and captured ordering closures: PASS");
    Ok(())
}
