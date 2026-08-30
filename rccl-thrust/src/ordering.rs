use core::cmp::Ordering;

use cuda_core::DeviceBuffer;
use rccl_memory::{DeviceAllocator, TemporaryValue};

use crate::{
    DeviceAlgorithmError,
    backend::{
        ComparisonSortBackend, ComparisonSortPairsBackend, RadixSortBackend, RadixSortPairsBackend,
    },
    cuda::Execution,
    range::KeyValueRangeMut,
};

/// Rust-style in-place ordering for a mutable device buffer.
///
/// Like slices, `sort` is stable and `sort_unstable` does not promise to
/// retain the relative order of equivalent values. The additional execution
/// argument makes the CUDA stream and allocator borrow explicit.
pub trait DeviceSortExt<K: TemporaryValue> {
    fn sort_unstable<A, M>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: RadixSortBackend<K>;

    fn sort_unstable_by<A, M, F>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: ComparisonSortBackend<K>,
        F: Fn(&K, &K) -> Ordering + Copy;

    fn sort<A, M>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: RadixSortBackend<K>;

    fn sort_by<A, M, F>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: ComparisonSortBackend<K>,
        F: Fn(&K, &K) -> Ordering + Copy;
}

impl<K: TemporaryValue> DeviceSortExt<K> for DeviceBuffer<K> {
    fn sort_unstable<A, M>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: RadixSortBackend<K>,
    {
        crate::backend::radix_sort(execution, self)
    }

    fn sort_unstable_by<A, M, F>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: ComparisonSortBackend<K>,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        crate::backend::merge_sort(execution, self, order)
    }

    fn sort<A, M>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: RadixSortBackend<K>,
    {
        crate::backend::radix_sort(execution, self)
    }

    fn sort_by<A, M, F>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: ComparisonSortBackend<K>,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        crate::backend::merge_sort(execution, self, order)
    }
}

/// In-place ordering for a structure-of-arrays key/value range.
///
/// Ordering examines only the key buffer; values follow their corresponding
/// keys. This avoids overloading Rust's `sort_by_key`, which conventionally
/// means key extraction from each element.
pub trait DeviceKeyValueSortExt<K: TemporaryValue, V: TemporaryValue> {
    fn sort_unstable<A, M>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: RadixSortPairsBackend<K, V>;

    fn sort_unstable_by<A, M, F>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: ComparisonSortPairsBackend<K, V>,
        F: Fn(&K, &K) -> Ordering + Copy;

    fn sort<A, M>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: RadixSortPairsBackend<K, V>;

    fn sort_by<A, M, F>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: ComparisonSortPairsBackend<K, V>,
        F: Fn(&K, &K) -> Ordering + Copy;
}

impl<K: TemporaryValue, V: TemporaryValue> DeviceKeyValueSortExt<K, V>
    for KeyValueRangeMut<'_, K, V>
{
    fn sort_unstable<A, M>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: RadixSortPairsBackend<K, V>,
    {
        let (keys, values) = self.parts_mut();
        crate::backend::radix_sort_pairs(execution, keys, values)
    }

    fn sort_unstable_by<A, M, F>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: ComparisonSortPairsBackend<K, V>,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        let (keys, values) = self.parts_mut();
        crate::backend::merge_sort_pairs(execution, keys, values, order)
    }

    fn sort<A, M>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: RadixSortPairsBackend<K, V>,
    {
        let (keys, values) = self.parts_mut();
        crate::backend::radix_sort_pairs(execution, keys, values)
    }

    fn sort_by<A, M, F>(
        &mut self,
        execution: &Execution<'_, '_, A, M>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: ComparisonSortPairsBackend<K, V>,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        let (keys, values) = self.parts_mut();
        crate::backend::merge_sort_pairs(execution, keys, values, order)
    }
}
