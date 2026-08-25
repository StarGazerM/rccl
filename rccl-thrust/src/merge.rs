//! Thrust host merge algorithms backed by CUB device-wide dispatch.

use crate::cuda::{Execution, finish_synchronous};
use cuda_core::DeviceCopy;
use rccl_cub::DeviceComparator;
use rccl_cub::dispatch::{
    merge::{self as cub_merge, MergeError},
    schedule::{
        MERGE_BLOCK_THREADS, MERGE_FALLBACK_BLOCK_THREADS, MERGE_ITEMS_PER_THREAD, MERGE_UNROLL,
    },
};
use rccl_memory::{DeviceAllocator, DeviceSlice, DeviceSliceMut};

/// Stably merge two ascending device ranges.
pub fn merge<K, A1, A2, O, A>(
    exec: &Execution<'_, A>,
    first: &A1,
    second: &A2,
    output: &mut O,
) -> Result<usize, MergeError>
where
    K: DeviceCopy + PartialOrd + 'static,
    A1: DeviceSlice<K> + ?Sized,
    A2: DeviceSlice<K> + ?Sized,
    O: DeviceSliceMut<K> + ?Sized,
    A: DeviceAllocator + Clone,
{
    let operation = unsafe {
        cub_merge::merge::<
            MERGE_BLOCK_THREADS,
            MERGE_ITEMS_PER_THREAD,
            MERGE_UNROLL,
            MERGE_FALLBACK_BLOCK_THREADS,
            K,
            A1,
            A2,
            O,
            A,
        >(
            exec.kernel_context(),
            exec.stream(),
            exec.allocator(),
            first,
            second,
            output,
        )
    };
    finish_synchronous(exec.stream(), operation)
}

/// Stably merge two device ranges using a state-carrying comparator.
pub fn merge_by<C, K, A1, A2, O, A>(
    exec: &Execution<'_, A>,
    first: &A1,
    second: &A2,
    output: &mut O,
    comparator: C,
) -> Result<usize, MergeError>
where
    C: DeviceComparator<K>,
    K: DeviceCopy + 'static,
    A1: DeviceSlice<K> + ?Sized,
    A2: DeviceSlice<K> + ?Sized,
    O: DeviceSliceMut<K> + ?Sized,
    A: DeviceAllocator + Clone,
{
    let operation = unsafe {
        cub_merge::merge_by::<
            MERGE_BLOCK_THREADS,
            MERGE_ITEMS_PER_THREAD,
            MERGE_UNROLL,
            MERGE_FALLBACK_BLOCK_THREADS,
            C,
            K,
            A1,
            A2,
            O,
            A,
        >(
            exec.kernel_context(),
            exec.stream(),
            exec.allocator(),
            first,
            second,
            output,
            comparator,
        )
    };
    finish_synchronous(exec.stream(), operation)
}

/// Stably merge two ascending key/value device ranges.
pub fn merge_by_key<K, V, K1, V1, K2, V2, KO, VO, A>(
    exec: &Execution<'_, A>,
    first_keys: &K1,
    first_values: &V1,
    second_keys: &K2,
    second_values: &V2,
    output_keys: &mut KO,
    output_values: &mut VO,
) -> Result<(usize, usize), MergeError>
where
    K: DeviceCopy + PartialOrd + 'static,
    V: DeviceCopy + 'static,
    K1: DeviceSlice<K> + ?Sized,
    V1: DeviceSlice<V> + ?Sized,
    K2: DeviceSlice<K> + ?Sized,
    V2: DeviceSlice<V> + ?Sized,
    KO: DeviceSliceMut<K> + ?Sized,
    VO: DeviceSliceMut<V> + ?Sized,
    A: DeviceAllocator + Clone,
{
    let operation = unsafe {
        cub_merge::merge_by_key::<
            MERGE_BLOCK_THREADS,
            MERGE_ITEMS_PER_THREAD,
            MERGE_UNROLL,
            MERGE_FALLBACK_BLOCK_THREADS,
            K,
            V,
            K1,
            V1,
            K2,
            V2,
            KO,
            VO,
            A,
        >(
            exec.kernel_context(),
            exec.stream(),
            exec.allocator(),
            first_keys,
            first_values,
            second_keys,
            second_values,
            output_keys,
            output_values,
        )
    };
    finish_synchronous(exec.stream(), operation)
}

/// Stably merge two key/value device ranges using a state-carrying comparator.
#[expect(
    clippy::too_many_arguments,
    reason = "key-value merge keeps four independently borrowed input ranges explicit"
)]
pub fn merge_by_key_with<C, K, V, K1, V1, K2, V2, KO, VO, A>(
    exec: &Execution<'_, A>,
    first_keys: &K1,
    first_values: &V1,
    second_keys: &K2,
    second_values: &V2,
    output_keys: &mut KO,
    output_values: &mut VO,
    comparator: C,
) -> Result<(usize, usize), MergeError>
where
    C: DeviceComparator<K>,
    K: DeviceCopy + 'static,
    V: DeviceCopy + 'static,
    K1: DeviceSlice<K> + ?Sized,
    V1: DeviceSlice<V> + ?Sized,
    K2: DeviceSlice<K> + ?Sized,
    V2: DeviceSlice<V> + ?Sized,
    KO: DeviceSliceMut<K> + ?Sized,
    VO: DeviceSliceMut<V> + ?Sized,
    A: DeviceAllocator + Clone,
{
    let operation = unsafe {
        cub_merge::merge_by_key_with::<
            MERGE_BLOCK_THREADS,
            MERGE_ITEMS_PER_THREAD,
            MERGE_UNROLL,
            MERGE_FALLBACK_BLOCK_THREADS,
            C,
            K,
            V,
            K1,
            V1,
            K2,
            V2,
            KO,
            VO,
            A,
        >(
            exec.kernel_context(),
            exec.stream(),
            exec.allocator(),
            first_keys,
            first_values,
            second_keys,
            second_values,
            output_keys,
            output_values,
            comparator,
        )
    };
    finish_synchronous(exec.stream(), operation)
}
