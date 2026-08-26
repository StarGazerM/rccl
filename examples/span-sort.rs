use cuda_core::{CudaContext, DeviceBuffer};
use rccl::memory::{DeviceSlice, DeviceSliceMut};
use std::error::Error;

mod gpu_support;

const PREFIX_GUARD: usize = 7;
const SORT_COUNT: usize = 1_000_003;
const SUFFIX_GUARD: usize = 11;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par(&stream)?;
    let mut allocation =
        DeviceBuffer::<u32>::zeroed(&stream, PREFIX_GUARD + SORT_COUNT + SUFFIX_GUARD)?;

    {
        let mut middle = allocation.subspan_mut(PREFIX_GUARD, SORT_COUNT)?;
        support.fill_pattern_u32(&stream, &mut middle, 0, gpu_support::U32Pattern::Descending)?;
        rccl::sort(&exec, &mut middle)?;

        let exact_errors = support.validate_iota_u32(&stream, &middle)?;
        if exact_errors != 0 {
            return Err(format!("sorted subspan has {exact_errors} incorrect values").into());
        }
    }

    let guard_errors = support.validate_zero_outside_u32(
        &stream,
        &allocation,
        PREFIX_GUARD,
        PREFIX_GUARD + SORT_COUNT,
    )?;
    if guard_errors != 0 {
        return Err(format!("subspan sort modified {guard_errors} guard values").into());
    }

    let read_only = allocation.subspan(PREFIX_GUARD, SORT_COUNT)?;
    assert_eq!(read_only.len(), SORT_COUNT);
    println!("device subspan sort with exact output and guard validation: PASS");
    Ok(())
}
