//! Rust port of pinned CCCL `thrust/examples/set_operations.cu` plus the
//! corresponding key/value source-selection cases.

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::prelude::*;
use std::error::Error;

fn prefix<T: cuda_core::DeviceCopy>(
    buffer: &DeviceBuffer<T>,
    stream: &cuda_core::CudaStream,
    len: usize,
) -> Result<Vec<T>, cuda_core::DriverError> {
    let values = buffer.to_host_vec(stream)?;
    Ok(values[..len].to_vec())
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let thrust = rccl_thrust::cuda::Program::load(&context)?;
    let exec = thrust.par(&stream)?;

    let first_host = [0_i32, 2, 4, 5, 6, 8, 9];
    let second_host = [0_i32, 1, 2, 3, 5, 7, 8];
    let first = DeviceBuffer::from_host(&stream, &first_host)?;
    let second = DeviceBuffer::from_host(&stream, &second_host)?;

    let mut merged = DeviceBuffer::<i32>::zeroed(&stream, first.len() + second.len())?;
    first
        .view()
        .merge(second.view())
        .write_into(merged.view_mut(), &exec)?;
    assert_eq!(
        prefix(&merged, &stream, merged.len())?,
        [0, 0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9]
    );

    let mut union = DeviceBuffer::<i32>::zeroed(&stream, first.len() + second.len())?;
    let union_len = first
        .view()
        .union(second.view())
        .write_into(union.view_mut(), &exec)?;
    assert_eq!(
        prefix(&union, &stream, union_len)?,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );

    let mut intersection = DeviceBuffer::<i32>::zeroed(&stream, first.len().min(second.len()))?;
    let intersection_len = first
        .view()
        .intersection(second.view())
        .write_into(intersection.view_mut(), &exec)?;
    assert_eq!(
        prefix(&intersection, &stream, intersection_len)?,
        [0, 2, 5, 8]
    );

    let mut difference = DeviceBuffer::<i32>::zeroed(&stream, first.len())?;
    let difference_len = first
        .view()
        .difference(second.view())
        .write_into(difference.view_mut(), &exec)?;
    assert_eq!(prefix(&difference, &stream, difference_len)?, [4, 6, 9]);

    let mut symmetric = DeviceBuffer::<i32>::zeroed(&stream, first.len() + second.len())?;
    let symmetric_len = first
        .view()
        .symmetric_difference(second.view())
        .write_into(symmetric.view_mut(), &exec)?;
    assert_eq!(
        prefix(&symmetric, &stream, symmetric_len)?,
        [1, 3, 4, 6, 7, 9]
    );

    let intersection_size = first.view().intersection(second.view()).count(&exec)?;
    assert_eq!(intersection_size, 4);

    let first_values = DeviceBuffer::from_host(&stream, &[10_i32, 20, 40, 50, 60, 80, 90])?;
    let second_values = DeviceBuffer::from_host(&stream, &[100_i32, 110, 120, 130, 150, 170, 180])?;
    let mut pair_keys = DeviceBuffer::<i32>::zeroed(&stream, first.len() + second.len())?;
    let mut pair_values = DeviceBuffer::<i32>::zeroed(&stream, first.len() + second.len())?;
    let pair_len = first
        .view()
        .with_values(first_values.view())?
        .union(second.view().with_values(second_values.view())?)
        .write_into(
            pair_keys.view_mut().with_values(pair_values.view_mut())?,
            &exec,
        )?;
    assert_eq!(
        prefix(&pair_keys, &stream, pair_len)?,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
    assert_eq!(
        prefix(&pair_values, &stream, pair_len)?,
        [10, 110, 20, 130, 40, 50, 60, 170, 80, 90]
    );

    println!("pinned CCCL Thrust set-operation cases: PASS");
    Ok(())
}
