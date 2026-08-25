use cuda_core::{CudaContext, DeviceBuffer};
use rccl::DeviceComparator;
use std::error::Error;

#[path = "../../examples/gpu_support.rs"]
mod gpu_support;

#[derive(Clone, Copy)]
struct MaskedU32Less {
    mask: u32,
}

unsafe impl DeviceComparator<u32> for MaskedU32Less {
    unsafe fn less(&self, left: &u32, right: &u32) -> bool {
        (left & self.mask) < (right & self.mask)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    const COUNT: usize = 1_000_003;
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par(&stream)?;

    let mut keys = DeviceBuffer::<u32>::zeroed(&stream, COUNT)?;
    support.fill_u32(&stream, &mut keys, 1)?;
    rccl::sort(&exec, &mut keys)?;
    assert_eq!(support.validate_order(&stream, &keys, rccl::Less)?, 0);

    support.fill_u32(&stream, &mut keys, 2)?;
    rccl::stable_sort(&exec, &mut keys)?;
    assert_eq!(support.validate_order(&stream, &keys, rccl::Less)?, 0);

    support.fill_u32(&stream, &mut keys, 3)?;
    rccl::sort_by(&exec, &mut keys, rccl::Greater)?;
    assert_eq!(support.validate_order(&stream, &keys, rccl::Greater)?, 0);

    let masked = MaskedU32Less { mask: 0xffff };
    support.fill_u32(&stream, &mut keys, 4)?;
    rccl::stable_sort_by(&exec, &mut keys, masked)?;
    assert_eq!(support.validate_order(&stream, &keys, masked)?, 0);

    let mut values = DeviceBuffer::<u32>::zeroed(&stream, COUNT)?;
    support.fill_low_entropy_u32(&stream, &mut keys, 5)?;
    support.fill_iota_u32(&stream, &mut values)?;
    rccl::sort_by_key(&exec, &mut keys, &mut values)?;
    assert_eq!(
        support.validate_stable_pairs(&stream, &keys, &values, rccl::Less)?,
        0
    );

    support.fill_low_entropy_u32(&stream, &mut keys, 6)?;
    support.fill_iota_u32(&stream, &mut values)?;
    rccl::stable_sort_by_key(&exec, &mut keys, &mut values)?;
    assert_eq!(
        support.validate_stable_pairs(&stream, &keys, &values, rccl::Less)?,
        0
    );

    support.fill_low_entropy_u32(&stream, &mut keys, 7)?;
    support.fill_iota_u32(&stream, &mut values)?;
    rccl::sort_by_key_with(&exec, &mut keys, &mut values, rccl::Greater)?;
    assert_eq!(
        support.validate_stable_pairs(&stream, &keys, &values, rccl::Greater)?,
        0
    );

    support.fill_u32(&stream, &mut keys, 8)?;
    support.fill_iota_u32(&stream, &mut values)?;
    rccl::stable_sort_by_key_with(&exec, &mut keys, &mut values, masked)?;
    assert_eq!(
        support.validate_stable_pairs(&stream, &keys, &values, masked)?,
        0
    );

    println!("all eight ordinary sort operations: PASS");
    Ok(())
}
