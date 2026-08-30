//! Rust port of the exact simple cases in pinned CCCL
//! `thrust/testing/merge.cu` and `thrust/testing/merge_by_key.cu`.

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::prelude::*;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let thrust = rccl_thrust::thrust_program!(&context)?;
    let exec = thrust.par(&stream)?;

    let first = DeviceBuffer::from_host(&stream, &[0_i32, 2, 4])?;
    let second = DeviceBuffer::from_host(&stream, &[0_i32, 3, 3, 4])?;
    let mut merged = DeviceBuffer::<i32>::zeroed(&stream, 7)?;
    exec.merge_into(&first, &second, &mut merged)?;
    assert_eq!(merged.to_host_vec(&stream)?, [0, 0, 2, 3, 3, 4, 4]);

    let descending_first = DeviceBuffer::from_host(&stream, &[9_i32, 7, 3, 1])?;
    let descending_second = DeviceBuffer::from_host(&stream, &[8_i32, 7, 2, 0])?;
    let mut descending_output = DeviceBuffer::<i32>::zeroed(&stream, 8)?;
    exec.merge_by_into(
        &descending_first,
        &descending_second,
        &mut descending_output,
        |left: &i32, right: &i32| right.cmp(left),
    )?;
    assert_eq!(
        descending_output.to_host_vec(&stream)?,
        [9, 8, 7, 7, 3, 2, 1, 0]
    );

    let first_values = DeviceBuffer::from_host(&stream, &[13_i32, 7, 42])?;
    let second_values = DeviceBuffer::from_host(&stream, &[42_i32, 42, 7, 13])?;
    let mut output_keys = DeviceBuffer::<i32>::zeroed(&stream, 7)?;
    let mut output_values = DeviceBuffer::<i32>::zeroed(&stream, 7)?;
    let mut output = output_keys.with_values_mut(&mut output_values)?;
    exec.merge_into(
        first.with_values(&first_values)?,
        second.with_values(&second_values)?,
        &mut output,
    )?;
    assert_eq!(output_keys.to_host_vec(&stream)?, [0, 0, 2, 3, 3, 4, 4]);
    assert_eq!(
        output_values.to_host_vec(&stream)?,
        [13, 42, 7, 42, 7, 42, 13]
    );

    let descending_first_values = DeviceBuffer::from_host(&stream, &[90_i32, 70, 30, 10])?;
    let descending_second_values = DeviceBuffer::from_host(&stream, &[80_i32, 71, 20, 0])?;
    let mut descending_output_keys = DeviceBuffer::<i32>::zeroed(&stream, 8)?;
    let mut descending_output_values = DeviceBuffer::<i32>::zeroed(&stream, 8)?;
    let mut descending_output =
        descending_output_keys.with_values_mut(&mut descending_output_values)?;
    exec.merge_by_into(
        descending_first.with_values(&descending_first_values)?,
        descending_second.with_values(&descending_second_values)?,
        &mut descending_output,
        |left: &i32, right: &i32| right.cmp(left),
    )?;
    assert_eq!(
        descending_output_keys.to_host_vec(&stream)?,
        [9, 8, 7, 7, 3, 2, 1, 0]
    );
    // Equal keys retain the first input's value before the second input's.
    assert_eq!(
        descending_output_values.to_host_vec(&stream)?,
        [90, 80, 70, 71, 30, 20, 10, 0]
    );

    println!("pinned CCCL Thrust merge simple cases: PASS");
    Ok(())
}
