use core::cmp::Ordering;

use cuda_core::DeviceCopy;
use rccl_memory::{DeviceAllocator, TemporaryValue};

use crate::{
    DeviceAlgorithmError,
    backend::{MergeBackend, MergePairsBackend},
    cuda::Execution,
    range::{DeviceView, DeviceViewMut, KeyValueRange, KeyValueRangeMut},
};

/// A lazy merge of two sorted device sequences.
///
/// The expression owns both borrowed views, so the inputs cannot disappear
/// between composition and execution. No CUDA work occurs until `write_into`.
pub struct Merge<First, Second> {
    first: First,
    second: Second,
}

/// A lazy merge carrying a Rust ordering closure.
pub struct MergeBy<First, Second, F> {
    first: First,
    second: Second,
    order: F,
}

impl<'a, K: DeviceCopy> DeviceView<'a, K> {
    /// Compose a naturally ordered merge with another borrowed sequence.
    pub fn merge<'b>(self, second: DeviceView<'b, K>) -> Merge<Self, DeviceView<'b, K>> {
        Merge {
            first: self,
            second,
        }
    }

    /// Compose a merge whose ordering is selected by a Rust closure.
    pub fn merge_by<'b, F>(
        self,
        second: DeviceView<'b, K>,
        order: F,
    ) -> MergeBy<Self, DeviceView<'b, K>, F>
    where
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        MergeBy {
            first: self,
            second,
            order,
        }
    }
}

impl<'a, K: DeviceCopy, V: DeviceCopy> KeyValueRange<'a, K, V> {
    /// Compose a naturally ordered structure-of-arrays merge.
    pub fn merge<'b>(
        self,
        second: KeyValueRange<'b, K, V>,
    ) -> Merge<Self, KeyValueRange<'b, K, V>> {
        Merge {
            first: self,
            second,
        }
    }

    /// Compose a structure-of-arrays merge with a key comparator.
    pub fn merge_by<'b, F>(
        self,
        second: KeyValueRange<'b, K, V>,
        order: F,
    ) -> MergeBy<Self, KeyValueRange<'b, K, V>, F>
    where
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        MergeBy {
            first: self,
            second,
            order,
        }
    }
}

impl<'a, 'b, K> Merge<DeviceView<'a, K>, DeviceView<'b, K>>
where
    K: TemporaryValue + PartialOrd,
{
    pub fn write_into<A, M>(
        self,
        output: DeviceViewMut<'_, K>,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: MergeBackend<K>,
    {
        execute_merge_by(
            execution,
            self.first.into_buffer(),
            self.second.into_buffer(),
            output.into_buffer(),
            natural_order,
        )
    }
}

impl<'a, 'b, K, F> MergeBy<DeviceView<'a, K>, DeviceView<'b, K>, F>
where
    K: TemporaryValue,
    F: Fn(&K, &K) -> Ordering + Copy,
{
    pub fn write_into<A, M>(
        self,
        output: DeviceViewMut<'_, K>,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: MergeBackend<K>,
    {
        execute_merge_by(
            execution,
            self.first.into_buffer(),
            self.second.into_buffer(),
            output.into_buffer(),
            self.order,
        )
    }
}

impl<'a, 'b, K, V> Merge<KeyValueRange<'a, K, V>, KeyValueRange<'b, K, V>>
where
    K: TemporaryValue + PartialOrd,
    V: TemporaryValue,
{
    pub fn write_into<A, M>(
        self,
        mut output: KeyValueRangeMut<'_, K, V>,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: MergePairsBackend<K, V>,
    {
        execute_merge_pairs_by(
            execution,
            self.first,
            self.second,
            &mut output,
            natural_order,
        )
    }
}

impl<'a, 'b, K, V, F> MergeBy<KeyValueRange<'a, K, V>, KeyValueRange<'b, K, V>, F>
where
    K: TemporaryValue,
    V: TemporaryValue,
    F: Fn(&K, &K) -> Ordering + Copy,
{
    pub fn write_into<A, M>(
        self,
        mut output: KeyValueRangeMut<'_, K, V>,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: MergePairsBackend<K, V>,
    {
        execute_merge_pairs_by(execution, self.first, self.second, &mut output, self.order)
    }
}

#[inline]
fn natural_order<K: PartialOrd>(left: &K, right: &K) -> Ordering {
    if left < right {
        Ordering::Less
    } else if right < left {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn execute_merge_by<K, A, M, F>(
    execution: &Execution<'_, '_, A, M>,
    first: &cuda_core::DeviceBuffer<K>,
    second: &cuda_core::DeviceBuffer<K>,
    output: &mut cuda_core::DeviceBuffer<K>,
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

fn execute_merge_pairs_by<K, V, A, M, F>(
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

#[cfg(test)]
mod tests {
    use cuda_core::DeviceBuffer;

    use crate::DeviceBufferExt;

    fn compose_without_launching<'a, 'b>(
        first: &'a DeviceBuffer<i32>,
        second: &'b DeviceBuffer<i32>,
    ) -> super::Merge<crate::DeviceView<'a, i32>, crate::DeviceView<'b, i32>> {
        first.view().merge(second.view())
    }

    #[test]
    fn merge_composition_preserves_both_input_lifetimes() {
        let _ = compose_without_launching;
    }
}
