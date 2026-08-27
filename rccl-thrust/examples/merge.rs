//! Rust port of the exact simple cases in pinned CCCL
//! `thrust/testing/merge.cu` and `thrust/testing/merge_by_key.cu`.

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::memory::CudaMemoryResource;
use std::error::Error;

rccl_thrust::thrust_kernels!();

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let allocator = CudaMemoryResource::new(context.clone());
    // SAFETY: this executable owns the one artifact generated above.
    let module = unsafe { __rccl_thrust_kernels::load(&context)? };
    let exec = rccl_thrust::cuda::par(&stream, &allocator, &module)?;

    let first = DeviceBuffer::from_host(&stream, &[0_i32, 2, 4])?;
    let second = DeviceBuffer::from_host(&stream, &[0_i32, 3, 3, 4])?;
    let partition_count = rccl_cub::device_merge_keys_partition_count!(::<i32, sm_89>(
        first.len(),
        second.len(),
    ))?;
    let mut partitions = DeviceBuffer::<usize>::zeroed(&stream, partition_count)?;
    let mut merged = DeviceBuffer::<i32>::zeroed(&stream, 7)?;
    rccl_thrust::merge!(::<i32, sm_89>(
        &exec,
        &first,
        &second,
        &mut partitions,
        &mut merged,
    ))?;
    assert_eq!(merged.to_host_vec(&stream)?, [0, 0, 2, 3, 3, 4, 4]);

    let descending_first = DeviceBuffer::from_host(&stream, &[9_i32, 7, 3, 1])?;
    let descending_second = DeviceBuffer::from_host(&stream, &[8_i32, 7, 2, 0])?;
    let mut descending_output = DeviceBuffer::<i32>::zeroed(&stream, 8)?;
    rccl_thrust::merge_by!(::<i32, sm_89>(
        &exec,
        &descending_first,
        &descending_second,
        &mut partitions,
        &mut descending_output,
        |left: &i32, right: &i32| right.cmp(left),
    ))?;
    assert_eq!(
        descending_output.to_host_vec(&stream)?,
        [9, 8, 7, 7, 3, 2, 1, 0]
    );

    let first_values = DeviceBuffer::from_host(&stream, &[13_i32, 7, 42])?;
    let second_values = DeviceBuffer::from_host(&stream, &[42_i32, 42, 7, 13])?;
    let pair_partition_count = rccl_cub::device_merge_pairs_partition_count!(::<i32, i32, sm_89>(
        first.len(),
        second.len(),
    ))?;
    let mut pair_partitions = DeviceBuffer::<usize>::zeroed(&stream, pair_partition_count)?;
    let mut output_keys = DeviceBuffer::<i32>::zeroed(&stream, 7)?;
    let mut output_values = DeviceBuffer::<i32>::zeroed(&stream, 7)?;
    rccl_thrust::merge_by_key!(::<i32, i32, sm_89>(
        &exec,
        &first,
        &first_values,
        &second,
        &second_values,
        &mut pair_partitions,
        &mut output_keys,
        &mut output_values,
    ))?;
    assert_eq!(output_keys.to_host_vec(&stream)?, [0, 0, 2, 3, 3, 4, 4]);
    assert_eq!(
        output_values.to_host_vec(&stream)?,
        [13, 42, 7, 42, 7, 42, 13]
    );

    let descending_first_values = DeviceBuffer::from_host(&stream, &[90_i32, 70, 30, 10])?;
    let descending_second_values = DeviceBuffer::from_host(&stream, &[80_i32, 71, 20, 0])?;
    let mut descending_output_keys = DeviceBuffer::<i32>::zeroed(&stream, 8)?;
    let mut descending_output_values = DeviceBuffer::<i32>::zeroed(&stream, 8)?;
    rccl_thrust::merge_by_key_with!(::<i32, i32, sm_89>(
        &exec,
        &descending_first,
        &descending_first_values,
        &descending_second,
        &descending_second_values,
        &mut pair_partitions,
        &mut descending_output_keys,
        &mut descending_output_values,
        |left: &i32, right: &i32| right.cmp(left),
    ))?;
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
