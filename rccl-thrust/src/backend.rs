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
