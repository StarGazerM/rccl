use core::cmp::Ordering;

use cuda_core::DeviceBuffer;
use rccl_memory::{DeviceAllocator, TemporaryValue};

use crate::{
    DeviceAlgorithmError,
    backend::{MergeBackend, MergePairsBackend},
    cuda::Execution,
    range::{KeyValueRange, KeyValueRangeMut},
};

/// Merge two symmetric sorted inputs into a distinct output range.
///
/// The execution environment is the receiver because neither input is the
/// natural owner of a merge operation.
pub trait DeviceMergeExt<First, Second, Output> {
    type Key;

    fn merge_into(
        &self,
        first: First,
        second: Second,
        output: Output,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Self::Key: PartialOrd;

    fn merge_by_into<F>(
        &self,
        first: First,
        second: Second,
        output: Output,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        F: Fn(&Self::Key, &Self::Key) -> Ordering + Copy;
}

impl<'a, 'b, 'c, K, A, M>
    DeviceMergeExt<&'a DeviceBuffer<K>, &'b DeviceBuffer<K>, &'c mut DeviceBuffer<K>>
    for Execution<'_, '_, A, M>
where
    K: TemporaryValue,
    A: DeviceAllocator,
    M: MergeBackend<K>,
{
    type Key = K;

    fn merge_into(
        &self,
        first: &'a DeviceBuffer<K>,
        second: &'b DeviceBuffer<K>,
        output: &'c mut DeviceBuffer<K>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        K: PartialOrd,
    {
        merge_by(self, first, second, output, |left, right| {
            if left < right {
                Ordering::Less
            } else if right < left {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
    }

    fn merge_by_into<F>(
        &self,
        first: &'a DeviceBuffer<K>,
        second: &'b DeviceBuffer<K>,
        output: &'c mut DeviceBuffer<K>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        merge_by(self, first, second, output, order)
    }
}

impl<'a, 'b, 'view, 'borrow, K, V, A, M>
    DeviceMergeExt<
        KeyValueRange<'a, K, V>,
        KeyValueRange<'b, K, V>,
        &'borrow mut KeyValueRangeMut<'view, K, V>,
    > for Execution<'_, '_, A, M>
where
    K: TemporaryValue,
    V: TemporaryValue,
    A: DeviceAllocator,
    M: MergePairsBackend<K, V>,
{
    type Key = K;

    fn merge_into(
        &self,
        first: KeyValueRange<'a, K, V>,
        second: KeyValueRange<'b, K, V>,
        output: &'borrow mut KeyValueRangeMut<'view, K, V>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        K: PartialOrd,
    {
        merge_pairs_by(self, first, second, output, |left, right| {
            if left < right {
                Ordering::Less
            } else if right < left {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        })
    }

    fn merge_by_into<F>(
        &self,
        first: KeyValueRange<'a, K, V>,
        second: KeyValueRange<'b, K, V>,
        output: &'borrow mut KeyValueRangeMut<'view, K, V>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        merge_pairs_by(self, first, second, output, order)
    }
}

fn merge_by<K, A, M, F>(
    execution: &Execution<'_, '_, A, M>,
    first: &DeviceBuffer<K>,
    second: &DeviceBuffer<K>,
    output: &mut DeviceBuffer<K>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: TemporaryValue,
    A: DeviceAllocator,
    M: MergeBackend<K>,
    F: Fn(&K, &K) -> Ordering + Copy,
{
    let partition_count = execution
        .module()
        .merge_partition_count(first.len(), second.len())?;
    let mut temporary = crate::memory::TemporaryWorkspace1::<_, usize>::new_in(
        execution.allocator(),
        execution.stream(),
        [partition_count],
        [256],
    )
    .map_err(DeviceAlgorithmError::Memory)?;
    let (partitions,) = temporary.parts_mut();
    let result = execution
        .module()
        .merge(execution, first, second, partitions, output, order);
    execution.__synchronize(result)
}

fn merge_pairs_by<K, V, A, M, F>(
    execution: &Execution<'_, '_, A, M>,
    first: KeyValueRange<'_, K, V>,
    second: KeyValueRange<'_, K, V>,
    output: &mut KeyValueRangeMut<'_, K, V>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: TemporaryValue,
    V: TemporaryValue,
    A: DeviceAllocator,
    M: MergePairsBackend<K, V>,
    F: Fn(&K, &K) -> Ordering + Copy,
{
    let (first_keys, first_values) = first.parts();
    let (second_keys, second_values) = second.parts();
    let (output_keys, output_values) = output.parts_mut();
    let partition_count = execution
        .module()
        .merge_pairs_partition_count(first_keys.len(), second_keys.len())?;
    let mut temporary = crate::memory::TemporaryWorkspace1::<_, usize>::new_in(
        execution.allocator(),
        execution.stream(),
        [partition_count],
        [256],
    )
    .map_err(DeviceAlgorithmError::Memory)?;
    let (partitions,) = temporary.parts_mut();
    let result = execution.module().merge_pairs(
        execution,
        first_keys,
        first_values,
        second_keys,
        second_values,
        partitions,
        output_keys,
        output_values,
        order,
    );
    execution.__synchronize(result)
}
