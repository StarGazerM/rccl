use core::cmp::Ordering;

use cuda_core::DeviceBuffer;
use rccl_memory::{DeviceAllocator, TemporaryValue};

use crate::{DeviceAlgorithmError, cuda::Execution, memory};

pub(crate) fn radix_sort<K, A>(
    execution: &Execution<'_, '_, A>,
    keys: &mut DeviceBuffer<K>,
) -> Result<(), DeviceAlgorithmError>
where
    K: TemporaryValue + rccl_cub::RadixKey,
    A: DeviceAllocator,
{
    let counts = rccl_cub::thrust_integration::device_radix_sort_workspace_counts::<K>(keys.len());
    let mut temporary = memory::TemporaryWorkspace4::<_, K, u64, i32, i32>::new_in(
        execution.allocator(),
        execution.stream(),
        [
            counts.scratch_keys,
            counts.bins,
            counts.lookback,
            counts.counters,
        ],
        [128, 256, 256, 256],
    )
    .map_err(DeviceAlgorithmError::Memory)?;
    let (scratch_keys, bins, lookback, counters) = temporary.parts_mut();
    let result = execution
        .launch()
        .radix_sort(keys, scratch_keys, bins, lookback, counters);
    // Match Thrust's temporary-array lifetime exactly: the stream-ordered
    // release is enqueued before the synchronous policy completion point.
    drop(temporary);
    execution.synchronize(result)
}

pub(crate) fn merge_sort<K, A, F>(
    execution: &Execution<'_, '_, A>,
    keys: &mut DeviceBuffer<K>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: TemporaryValue,
    A: DeviceAllocator,
    F: Fn(&K, &K) -> Ordering + Copy,
{
    let partition_count =
        rccl_cub::thrust_integration::device_merge_sort_partition_count::<K>(keys.len())?;
    let mut temporary = memory::TemporaryWorkspace2::<_, usize, K>::new_in(
        execution.allocator(),
        execution.stream(),
        [partition_count, keys.len()],
        [256, 256],
    )
    .map_err(DeviceAlgorithmError::Memory)?;
    let (partitions, scratch_keys) = temporary.parts_mut();
    let result = execution
        .launch()
        .merge_sort(keys, scratch_keys, partitions, order);
    execution.synchronize(result)
}

pub(crate) fn radix_sort_pairs<K, V, A>(
    execution: &Execution<'_, '_, A>,
    keys: &mut DeviceBuffer<K>,
    values: &mut DeviceBuffer<V>,
) -> Result<(), DeviceAlgorithmError>
where
    K: TemporaryValue + rccl_cub::RadixKey,
    V: TemporaryValue + rccl_cub::RadixValue,
    A: DeviceAllocator,
{
    if values.len() != keys.len() {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "values",
            expected: keys.len(),
            actual: values.len(),
        });
    }
    let counts =
        rccl_cub::thrust_integration::device_radix_sort_pairs_workspace_counts::<K, V>(keys.len());
    let mut temporary = memory::TemporaryWorkspace5::<_, K, V, u64, i32, i32>::new_in(
        execution.allocator(),
        execution.stream(),
        [
            counts.scratch_keys,
            counts.scratch_values,
            counts.bins,
            counts.lookback,
            counts.counters,
        ],
        [128, 128, 256, 256, 256],
    )
    .map_err(DeviceAlgorithmError::Memory)?;
    let (scratch_keys, scratch_values, bins, lookback, counters) = temporary.parts_mut();
    let result = execution.launch().radix_sort_pairs(
        keys,
        values,
        scratch_keys,
        scratch_values,
        bins,
        lookback,
        counters,
    );
    // Match the key-only smart-sort lifetime above and pinned Thrust.
    drop(temporary);
    execution.synchronize(result)
}

pub(crate) fn merge_sort_pairs<K, V, A, F>(
    execution: &Execution<'_, '_, A>,
    keys: &mut DeviceBuffer<K>,
    values: &mut DeviceBuffer<V>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: TemporaryValue,
    V: TemporaryValue,
    A: DeviceAllocator,
    F: Fn(&K, &K) -> Ordering + Copy,
{
    if values.len() != keys.len() {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "values",
            expected: keys.len(),
            actual: values.len(),
        });
    }
    let partition_count =
        rccl_cub::thrust_integration::device_merge_sort_partition_count::<K>(keys.len())?;
    let mut temporary = memory::TemporaryWorkspace3::<_, usize, K, V>::new_in(
        execution.allocator(),
        execution.stream(),
        [partition_count, keys.len(), values.len()],
        [256, 256, 256],
    )
    .map_err(DeviceAlgorithmError::Memory)?;
    let (partitions, scratch_keys, scratch_values) = temporary.parts_mut();
    let result = execution.launch().merge_sort_pairs(
        keys,
        values,
        scratch_keys,
        scratch_values,
        partitions,
        order,
    );
    execution.synchronize(result)
}

pub(crate) fn range_fill<Output, T, A>(
    execution: &Execution<'_, '_, A>,
    output: &Output,
    value: T,
) -> Result<(), DeviceAlgorithmError>
where
    Output: crate::range::LowerWriteRange<Item = T>,
    T: cuda_core::DeviceCopy,
    A: DeviceAllocator,
{
    let len = rccl_cub::DeviceRange::len(output);
    output
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    if len == 0 {
        return Ok(());
    }
    if core::mem::size_of::<T>() == 4
        && let Ok(len_u32) = u32::try_from(len)
        && let Some(pointer) = output.dense_write_pointer()
        && pointer.addr().is_multiple_of(8)
    {
        return execution
            .launch()
            .range_fill_4b(rccl_cub::DeviceSliceWrite::new(pointer), value, len_u32)
            .map_err(DeviceAlgorithmError::Launch);
    }
    let repeated = crate::range::RepeatN::new(value, len);
    range_transform(execution, &repeated, output, &|item| item)
}

pub(crate) fn range_generate<Input, Output, F, A>(
    execution: &Execution<'_, '_, A>,
    input: &Input,
    output: &Output,
    op: &F,
) -> Result<(), DeviceAlgorithmError>
where
    Input: crate::range::LowerReadRange,
    Output: crate::range::LowerWriteRange,
    F: Fn(Input::Item) -> Output::Item + Copy,
    A: DeviceAllocator,
{
    let len = rccl_cub::DeviceRange::len(input);
    if len != rccl_cub::DeviceRange::len(output) {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "range generation output",
            expected: len,
            actual: rccl_cub::DeviceRange::len(output),
        });
    }
    input
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    output
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    if len == 0 {
        return Ok(());
    }
    if core::mem::size_of::<Output::Item>() == 4
        && let Ok(len_u32) = u32::try_from(len)
        && let Some(pointer) = output.dense_write_pointer()
        && pointer.addr().is_multiple_of(16)
    {
        return execution
            .launch()
            .range_generate_4b(
                input.lower_read(),
                rccl_cub::DeviceSliceWrite::new(pointer),
                *op,
                len_u32,
            )
            .map_err(DeviceAlgorithmError::Launch);
    }
    range_transform(execution, input, output, op)
}

pub(crate) fn range_transform<Input, Output, F, A>(
    execution: &Execution<'_, '_, A>,
    input: &Input,
    output: &Output,
    op: &F,
) -> Result<(), DeviceAlgorithmError>
where
    Input: crate::range::LowerReadRange,
    Output: crate::range::LowerWriteRange,
    F: Fn(Input::Item) -> Output::Item + Copy,
    A: DeviceAllocator,
{
    let input_len = rccl_cub::DeviceRange::len(input);
    let output_len = rccl_cub::DeviceRange::len(output);
    if input_len != output_len {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "range transform output",
            expected: input_len,
            actual: output_len,
        });
    }
    input
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    output
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    execution
        .launch()
        .range_transform(input.lower_read(), output.lower_write(), *op, input_len)
        .map_err(DeviceAlgorithmError::Launch)
}

pub(crate) fn range_for_each<Input, F, A>(
    execution: &Execution<'_, '_, A>,
    input: &Input,
    op: &F,
) -> Result<(), DeviceAlgorithmError>
where
    Input: crate::range::LowerReadRange,
    F: Fn(Input::Item) + Copy,
    A: DeviceAllocator,
{
    input
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    execution
        .launch()
        .range_for_each(input.lower_read(), *op, rccl_cub::DeviceRange::len(input))
        .map_err(DeviceAlgorithmError::Launch)
}
