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

    let mut keys = DeviceBuffer::<u32>::zeroed(&stream, COUNT)?;
    support.fill_u32(&stream, &mut keys, 1)?;
    rccl::sort(&exec, &mut keys)?;
    assert_eq!(support.validate_order(&stream, &keys, u32::cmp)?, 0);

    support.fill_u32(&stream, &mut keys, 2)?;
    rccl::stable_sort(&exec, &mut keys)?;
    assert_eq!(support.validate_order(&stream, &keys, u32::cmp)?, 0);

    support.fill_u32(&stream, &mut keys, 3)?;
    let descending = |left: &u32, right: &u32| right.cmp(left);
    rccl::sort_by(&exec, &mut keys, descending)?;
    assert_eq!(support.validate_order(&stream, &keys, descending)?, 0);

    let mask = 0xffff_u32;
    let masked = move |left: &u32, right: &u32| (left & mask).cmp(&(right & mask));
    support.fill_u32(&stream, &mut keys, 4)?;
    rccl::stable_sort_by(&exec, &mut keys, masked)?;
    assert_eq!(support.validate_masked_u32_order(&stream, &keys, mask)?, 0);

    let mut values = DeviceBuffer::<u32>::zeroed(&stream, COUNT)?;
    support.fill_low_entropy_u32(&stream, &mut keys, 5)?;
    support.fill_iota_u32(&stream, &mut values)?;
    rccl::sort_by_key(&exec, &mut keys, &mut values)?;
    assert_eq!(
        support.validate_stable_pairs(&stream, &keys, &values, u32::cmp)?,
        0
    );

    support.fill_low_entropy_u32(&stream, &mut keys, 6)?;
    support.fill_iota_u32(&stream, &mut values)?;
    rccl::stable_sort_by_key(&exec, &mut keys, &mut values)?;
    assert_eq!(
        support.validate_stable_pairs(&stream, &keys, &values, u32::cmp)?,
        0
    );

    support.fill_low_entropy_u32(&stream, &mut keys, 7)?;
    support.fill_iota_u32(&stream, &mut values)?;
    rccl::sort_by_key_with(&exec, &mut keys, &mut values, descending)?;
    assert_eq!(
        support.validate_stable_pairs(&stream, &keys, &values, descending)?,
        0
    );

    support.fill_u32(&stream, &mut keys, 8)?;
    support.fill_iota_u32(&stream, &mut values)?;
    rccl::stable_sort_by_key_with(&exec, &mut keys, &mut values, masked)?;
    assert_eq!(
        support.validate_masked_u32_stable_pairs(&stream, &keys, &values, mask)?,
        0
    );

    println!("all eight ordinary sort operations: PASS");
    Ok(())
}
