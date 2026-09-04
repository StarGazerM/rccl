use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{
    DeviceBufferExt, DeviceRangeExt, arithmetic_range, discard, for_each, repeat_n, transform_into,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_thrust::cuda::Program::load(&context)?;
    let execution = program.par(&stream)?;

    let mut nested_output = DeviceBuffer::<i32>::zeroed(&stream, 4)?;
    let nested = arithmetic_range(1_i32, 2_i32, 8)
        .map(|value| value * 3)
        .rev()
        .step_by(2)?;
    transform_into(&execution, nested, nested_output.view_mut(), |value| value)?;
    assert_eq!(nested_output.to_host_vec(&stream)?, [45, 33, 21, 9]);

    let mut zipped_output = DeviceBuffer::<i32>::zeroed(&stream, 4)?;
    let zipped = arithmetic_range(0_i32, 1_i32, 4).zip_eq(repeat_n(10_i32, 4))?;
    transform_into(
        &execution,
        zipped,
        zipped_output.view_mut(),
        |(left, right)| left + right,
    )?;
    assert_eq!(zipped_output.to_host_vec(&stream)?, [10, 11, 12, 13]);

    let mut subrange_output = DeviceBuffer::<i32>::zeroed(&stream, 3)?;
    let subrange = arithmetic_range(0_i32, 1_i32, 6).slice(2..5)?;
    transform_into(&execution, subrange, subrange_output.view_mut(), |value| {
        value
    })?;
    assert_eq!(subrange_output.to_host_vec(&stream)?, [2, 3, 4]);

    let mut strided_output = DeviceBuffer::<i32>::zeroed(&stream, 7)?;
    let writable = strided_output.view_mut().rev().step_by(2)?;
    transform_into(
        &execution,
        arithmetic_range(1_i32, 1_i32, 4),
        writable,
        |value| value,
    )?;
    assert_eq!(strided_output.to_host_vec(&stream)?, [4, 0, 3, 0, 2, 0, 1]);

    transform_into(
        &execution,
        arithmetic_range(0_i32, 1_i32, 4),
        discard::<i32>(4),
        |value| value * 2,
    )?;

    let inject_fault = std::env::var_os("RCCL_INJECT_RANGE_FAULT").is_some();
    for_each(
        &execution,
        arithmetic_range(0_i32, 1_i32, 4)
            .map(|value| value * 2)
            .rev(),
        move |value| {
            if inject_fault {
                assert!(value < 0);
            } else {
                assert!(value >= 0 && value <= 6 && value % 2 == 0);
            }
        },
    )?;
    for_each(&execution, repeat_n(7_i32, 0), |_value| {})?;
    for_each(&execution, repeat_n(7_i32, 1), |value| assert!(value == 7))?;

    println!("range lowering: PASS");
    Ok(())
}
