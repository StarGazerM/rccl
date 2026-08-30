use core::cmp::Ordering;

use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_memory::{DeviceAllocator, TemporaryRegion, TemporaryValue};

use crate::{DeviceAlgorithmError, cuda::Execution, memory};

/// Static compiler interface implemented for a generated cuda-oxide module.
///
/// Declarative program scaffolding implements this trait for concrete
/// primitive key types, while nested CUB proc macros select the policy. The
/// trait carries no runtime policy and is not an application extension point.
#[doc(hidden)]
pub trait RadixSortBackend<K: TemporaryValue>: Sized {
    fn radix_workspace_counts(&self, len: usize) -> rccl_cub::RadixSortWorkspaceCounts;

    fn radix_sort<A: DeviceAllocator>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        keys: &mut DeviceBuffer<K>,
        scratch_keys: &mut TemporaryRegion<'_, K>,
        bins: &mut TemporaryRegion<'_, u64>,
        lookback: &mut TemporaryRegion<'_, i32>,
        counters: &mut TemporaryRegion<'_, i32>,
    ) -> Result<(), DeviceAlgorithmError>;
}

/// Static compiler interface for comparison-based key ordering.
#[doc(hidden)]
pub trait ComparisonSortBackend<K: TemporaryValue>: Sized {
    fn merge_sort_partition_count(&self, len: usize) -> Result<usize, DeviceAlgorithmError>;

    fn merge_sort<A, F>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        keys: &mut DeviceBuffer<K>,
        scratch_keys: &mut TemporaryRegion<'_, K>,
        partitions: &mut TemporaryRegion<'_, usize>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy;
}

/// Static compiler interface for key/value radix ordering.
#[doc(hidden)]
pub trait RadixSortPairsBackend<K: TemporaryValue, V: TemporaryValue>: Sized {
    fn radix_pairs_workspace_counts(&self, len: usize) -> rccl_cub::RadixSortPairsWorkspaceCounts;

    #[allow(clippy::too_many_arguments)]
    fn radix_sort_pairs<A: DeviceAllocator>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        keys: &mut DeviceBuffer<K>,
        values: &mut DeviceBuffer<V>,
        scratch_keys: &mut TemporaryRegion<'_, K>,
        scratch_values: &mut TemporaryRegion<'_, V>,
        bins: &mut TemporaryRegion<'_, u64>,
        lookback: &mut TemporaryRegion<'_, i32>,
        counters: &mut TemporaryRegion<'_, i32>,
    ) -> Result<(), DeviceAlgorithmError>;
}

/// Static compiler interface for comparison-based key/value ordering.
#[doc(hidden)]
pub trait ComparisonSortPairsBackend<K: TemporaryValue, V: TemporaryValue>: Sized {
    fn merge_sort_pairs_partition_count(&self, len: usize) -> Result<usize, DeviceAlgorithmError>;

    #[allow(clippy::too_many_arguments)]
    fn merge_sort_pairs<A, F>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        keys: &mut DeviceBuffer<K>,
        values: &mut DeviceBuffer<V>,
        scratch_keys: &mut TemporaryRegion<'_, K>,
        scratch_values: &mut TemporaryRegion<'_, V>,
        partitions: &mut TemporaryRegion<'_, usize>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy;
}

/// Static compiler interface for merging two sorted key ranges.
#[doc(hidden)]
pub trait MergeBackend<K: TemporaryValue>: Sized {
    fn merge_partition_count(
        &self,
        first_len: usize,
        second_len: usize,
    ) -> Result<usize, DeviceAlgorithmError>;

    #[allow(clippy::too_many_arguments)]
    fn merge<A, F>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first: &DeviceBuffer<K>,
        second: &DeviceBuffer<K>,
        partitions: &mut TemporaryRegion<'_, usize>,
        output: &mut DeviceBuffer<K>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy;
}

/// Static compiler interface for merging key/value ranges.
#[doc(hidden)]
pub trait MergePairsBackend<K: TemporaryValue, V: TemporaryValue>: Sized {
    fn merge_pairs_partition_count(
        &self,
        first_len: usize,
        second_len: usize,
    ) -> Result<usize, DeviceAlgorithmError>;

    #[allow(clippy::too_many_arguments)]
    fn merge_pairs<A, F>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first_keys: &DeviceBuffer<K>,
        first_values: &DeviceBuffer<V>,
        second_keys: &DeviceBuffer<K>,
        second_values: &DeviceBuffer<V>,
        partitions: &mut TemporaryRegion<'_, usize>,
        output_keys: &mut DeviceBuffer<K>,
        output_values: &mut DeviceBuffer<V>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy;
}

/// Static compiler interface for key-only set operations.
#[doc(hidden)]
pub trait SetBackend<K: DeviceCopy>: Sized {
    fn set_workspace_counts<A: DeviceAllocator>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first_len: usize,
        second_len: usize,
    ) -> Result<rccl_cub::SetOperationWorkspaceCounts, DeviceAlgorithmError>;

    #[allow(clippy::too_many_arguments)]
    fn set_i32<A, F, Op>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first: &DeviceBuffer<K>,
        second: &DeviceBuffer<K>,
        workspace: &mut rccl_cub::DeviceSetOperationWorkspace<
            &mut TemporaryRegion<'_, usize>,
            &mut TemporaryRegion<'_, u64>,
            &mut TemporaryRegion<'_, u64>,
            &mut TemporaryRegion<'_, u8>,
        >,
        output: &mut DeviceBuffer<K>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy,
        Op: rccl_cub::__private::SerialSetOperation;

    #[allow(clippy::too_many_arguments)]
    fn set_i64<A, F, Op>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first: &DeviceBuffer<K>,
        second: &DeviceBuffer<K>,
        workspace: &mut rccl_cub::DeviceSetOperationWorkspace<
            &mut TemporaryRegion<'_, usize>,
            &mut TemporaryRegion<'_, u128>,
            &mut TemporaryRegion<'_, u128>,
            &mut TemporaryRegion<'_, u8>,
        >,
        output: &mut DeviceBuffer<K>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy,
        Op: rccl_cub::__private::SerialSetOperation;

    fn set_discard_i32<A, F, Op>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first: &DeviceBuffer<K>,
        second: &DeviceBuffer<K>,
        workspace: &mut rccl_cub::DeviceSetOperationWorkspace<
            &mut TemporaryRegion<'_, usize>,
            &mut TemporaryRegion<'_, u64>,
            &mut TemporaryRegion<'_, u64>,
            &mut TemporaryRegion<'_, u8>,
        >,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy,
        Op: rccl_cub::__private::SerialSetOperation;

    fn set_discard_i64<A, F, Op>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first: &DeviceBuffer<K>,
        second: &DeviceBuffer<K>,
        workspace: &mut rccl_cub::DeviceSetOperationWorkspace<
            &mut TemporaryRegion<'_, usize>,
            &mut TemporaryRegion<'_, u128>,
            &mut TemporaryRegion<'_, u128>,
            &mut TemporaryRegion<'_, u8>,
        >,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy,
        Op: rccl_cub::__private::SerialSetOperation;
}

/// Static compiler interface for arithmetic-range set inputs.
#[doc(hidden)]
pub trait ArithmeticSetBackend<K>: Sized
where
    K: TemporaryValue + TryFrom<usize> + core::ops::Add<Output = K> + core::ops::Mul<Output = K>,
{
    fn set_arithmetic_i32<A, F, Op>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first: crate::range::ArithmeticRange<K>,
        second: crate::range::ArithmeticRange<K>,
        workspace: &mut rccl_cub::DeviceSetOperationWorkspace<
            &mut TemporaryRegion<'_, usize>,
            &mut TemporaryRegion<'_, u64>,
            &mut TemporaryRegion<'_, u64>,
            &mut TemporaryRegion<'_, u8>,
        >,
        output: &mut DeviceBuffer<K>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy,
        Op: rccl_cub::__private::SerialSetOperation;

    fn set_arithmetic_i64<A, F, Op>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first: crate::range::ArithmeticRange<K>,
        second: crate::range::ArithmeticRange<K>,
        workspace: &mut rccl_cub::DeviceSetOperationWorkspace<
            &mut TemporaryRegion<'_, usize>,
            &mut TemporaryRegion<'_, u128>,
            &mut TemporaryRegion<'_, u128>,
            &mut TemporaryRegion<'_, u8>,
        >,
        output: &mut DeviceBuffer<K>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy,
        Op: rccl_cub::__private::SerialSetOperation;
}

/// Static compiler interface for structure-of-arrays set operations.
#[doc(hidden)]
pub trait SetPairsBackend<K: TemporaryValue, V: TemporaryValue>: Sized {
    fn set_pairs_workspace_counts<A: DeviceAllocator>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first_len: usize,
        second_len: usize,
    ) -> Result<rccl_cub::SetOperationWorkspaceCounts, DeviceAlgorithmError>;

    #[allow(clippy::too_many_arguments)]
    fn set_pairs_i32<A, F, Op>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first_keys: &DeviceBuffer<K>,
        first_values: &DeviceBuffer<V>,
        second_keys: &DeviceBuffer<K>,
        second_values: &DeviceBuffer<V>,
        workspace: &mut rccl_cub::DeviceSetOperationWorkspace<
            &mut TemporaryRegion<'_, usize>,
            &mut TemporaryRegion<'_, u64>,
            &mut TemporaryRegion<'_, u64>,
            &mut TemporaryRegion<'_, u8>,
        >,
        output_keys: &mut DeviceBuffer<K>,
        output_values: &mut DeviceBuffer<V>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy,
        Op: rccl_cub::__private::SerialSetOperation;

    #[allow(clippy::too_many_arguments)]
    fn set_pairs_i64<A, F, Op>(
        &self,
        execution: &Execution<'_, '_, A, Self>,
        first_keys: &DeviceBuffer<K>,
        first_values: &DeviceBuffer<V>,
        second_keys: &DeviceBuffer<K>,
        second_values: &DeviceBuffer<V>,
        workspace: &mut rccl_cub::DeviceSetOperationWorkspace<
            &mut TemporaryRegion<'_, usize>,
            &mut TemporaryRegion<'_, u128>,
            &mut TemporaryRegion<'_, u128>,
            &mut TemporaryRegion<'_, u8>,
        >,
        output_keys: &mut DeviceBuffer<K>,
        output_values: &mut DeviceBuffer<V>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        F: Fn(&K, &K) -> Ordering + Copy,
        Op: rccl_cub::__private::SerialSetOperation;
}

pub(crate) fn radix_sort<K, A, M>(
    execution: &Execution<'_, '_, A, M>,
    keys: &mut DeviceBuffer<K>,
) -> Result<(), DeviceAlgorithmError>
where
    K: TemporaryValue,
    A: DeviceAllocator,
    M: RadixSortBackend<K>,
{
    let counts = execution.module().radix_workspace_counts(keys.len());
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
    let result =
        execution
            .module()
            .radix_sort(execution, keys, scratch_keys, bins, lookback, counters);
    // Match Thrust's temporary-array lifetime exactly: the stream-ordered
    // release is enqueued before the synchronous policy completion point.
    drop(temporary);
    execution.__synchronize(result)
}

pub(crate) fn merge_sort<K, A, M, F>(
    execution: &Execution<'_, '_, A, M>,
    keys: &mut DeviceBuffer<K>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: TemporaryValue,
    A: DeviceAllocator,
    M: ComparisonSortBackend<K>,
    F: Fn(&K, &K) -> Ordering + Copy,
{
    let partition_count = execution.module().merge_sort_partition_count(keys.len())?;
    let mut temporary = memory::TemporaryWorkspace2::<_, usize, K>::new_in(
        execution.allocator(),
        execution.stream(),
        [partition_count, keys.len()],
        [256, 256],
    )
    .map_err(DeviceAlgorithmError::Memory)?;
    let (partitions, scratch_keys) = temporary.parts_mut();
    let result = execution
        .module()
        .merge_sort(execution, keys, scratch_keys, partitions, order);
    execution.__synchronize(result)
}

pub(crate) fn radix_sort_pairs<K, V, A, M>(
    execution: &Execution<'_, '_, A, M>,
    keys: &mut DeviceBuffer<K>,
    values: &mut DeviceBuffer<V>,
) -> Result<(), DeviceAlgorithmError>
where
    K: TemporaryValue,
    V: TemporaryValue,
    A: DeviceAllocator,
    M: RadixSortPairsBackend<K, V>,
{
    if values.len() != keys.len() {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "values",
            expected: keys.len(),
            actual: values.len(),
        });
    }
    let counts = execution.module().radix_pairs_workspace_counts(keys.len());
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
    let result = execution.module().radix_sort_pairs(
        execution,
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
    execution.__synchronize(result)
}

pub(crate) fn merge_sort_pairs<K, V, A, M, F>(
    execution: &Execution<'_, '_, A, M>,
    keys: &mut DeviceBuffer<K>,
    values: &mut DeviceBuffer<V>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: TemporaryValue,
    V: TemporaryValue,
    A: DeviceAllocator,
    M: ComparisonSortPairsBackend<K, V>,
    F: Fn(&K, &K) -> Ordering + Copy,
{
    if values.len() != keys.len() {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "values",
            expected: keys.len(),
            actual: values.len(),
        });
    }
    let partition_count = execution
        .module()
        .merge_sort_pairs_partition_count(keys.len())?;
    let mut temporary = memory::TemporaryWorkspace3::<_, usize, K, V>::new_in(
        execution.allocator(),
        execution.stream(),
        [partition_count, keys.len(), values.len()],
        [256, 256, 256],
    )
    .map_err(DeviceAlgorithmError::Memory)?;
    let (partitions, scratch_keys, scratch_values) = temporary.parts_mut();
    let result = execution.module().merge_sort_pairs(
        execution,
        keys,
        values,
        scratch_keys,
        scratch_values,
        partitions,
        order,
    );
    execution.__synchronize(result)
}
