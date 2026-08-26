//! Thrust-shaped, type-inferred algorithm facade.
//!
//! Compile-target CUB schedules stay internal. Public algorithms keep ordinary
//! generics for elements, orderings, allocators, and borrowed device ranges.

use crate::cuda::{Execution, finish_synchronous};
use cuda_core::{DeviceCopy, DriverError};
use rccl_cub::dispatch::{
    ComparisonSortError, DeviceMergeSort, DeviceRadixSort, RadixSortError,
    schedule::{
        COMPARISON_BLOCK_THREADS, RADIX_BLOCK_THREADS, RADIX_BLOCK_WARPS, RADIX_ITEMS_PER_THREAD,
        RADIX_ITEMS_PER_THREAD_USIZE, RADIX_MIN_BLOCKS_PER_SM, RADIX_UNROLL,
    },
};
use rccl_memory::{DeviceAllocator, DeviceSliceMut};
use std::{cmp::Ordering, error::Error, fmt};

/// Error reported by a Thrust-shaped sort operation.
#[derive(Debug)]
#[non_exhaustive]
pub enum SortError {
    Driver(DriverError),
    Radix(RadixSortError),
    Comparison(ComparisonSortError),
}

impl fmt::Display for SortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => error.fmt(f),
            Self::Radix(error) => error.fmt(f),
            Self::Comparison(error) => error.fmt(f),
        }
    }
}

impl Error for SortError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Radix(error) => Some(error),
            Self::Comparison(error) => Some(error),
        }
    }
}

impl From<DriverError> for SortError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

impl From<RadixSortError> for SortError {
    fn from(value: RadixSortError) -> Self {
        Self::Radix(value)
    }
}

impl From<ComparisonSortError> for SortError {
    fn from(value: ComparisonSortError) -> Self {
        Self::Comparison(value)
    }
}

#[inline(always)]
fn natural_order<K: PartialOrd>(left: &K, right: &K) -> Ordering {
    if left < right {
        Ordering::Less
    } else if right < left {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

trait KeyOrderDispatch<K, A>: Fn(&K, &K) -> Ordering + Copy + 'static
where
    K: DeviceCopy,
    A: DeviceAllocator + Clone,
{
    fn sort_keys<
        const BLOCK_THREADS: u32,
        const ITEMS_PER_THREAD: u32,
        const ITEMS_PER_THREAD_USIZE: usize,
        const BLOCK_WARPS: usize,
        const MIN_BLOCKS_PER_SM: u32,
        const UNROLL: u32,
        const MERGE_BLOCK_THREADS: u32,
        B,
    >(
        self,
        exec: &Execution<'_, A>,
        keys: &mut B,
    ) -> Result<(), SortError>
    where
        B: DeviceSliceMut<K> + ?Sized;
}

impl<C, K, A> KeyOrderDispatch<K, A> for C
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    A: DeviceAllocator + Clone,
{
    fn sort_keys<
        const BLOCK_THREADS: u32,
        const ITEMS_PER_THREAD: u32,
        const ITEMS_PER_THREAD_USIZE: usize,
        const BLOCK_WARPS: usize,
        const MIN_BLOCKS_PER_SM: u32,
        const UNROLL: u32,
        const MERGE_BLOCK_THREADS: u32,
        B,
    >(
        self,
        exec: &Execution<'_, A>,
        keys: &mut B,
    ) -> Result<(), SortError>
    where
        B: DeviceSliceMut<K> + ?Sized,
    {
        let comparison = DeviceMergeSort::new(exec.kernel_context(), exec.allocator().clone())?;
        let stream = exec.stream();
        let mut workspace = comparison.allocate_key_workspace::<K>(stream, keys.len())?;
        unsafe {
            comparison.stable_sort_keys_by::<MERGE_BLOCK_THREADS, _, _, _>(
                stream,
                keys,
                self,
                &mut workspace,
            )?;
        }
        Ok(())
    }
}

trait PairOrderDispatch<K, V, A>: Fn(&K, &K) -> Ordering + Copy + 'static
where
    K: DeviceCopy,
    V: DeviceCopy + 'static,
    A: DeviceAllocator + Clone,
{
    fn sort_pairs<
        const BLOCK_THREADS: u32,
        const ITEMS_PER_THREAD: u32,
        const ITEMS_PER_THREAD_USIZE: usize,
        const BLOCK_WARPS: usize,
        const MIN_BLOCKS_PER_SM: u32,
        const UNROLL: u32,
        const MERGE_BLOCK_THREADS: u32,
        KB,
        VB,
    >(
        self,
        exec: &Execution<'_, A>,
        keys: &mut KB,
        values: &mut VB,
    ) -> Result<(), SortError>
    where
        KB: DeviceSliceMut<K> + ?Sized,
        VB: DeviceSliceMut<V> + ?Sized;
}

impl<C, K, V, A> PairOrderDispatch<K, V, A> for C
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    V: DeviceCopy + 'static,
    A: DeviceAllocator + Clone,
{
    fn sort_pairs<
        const BLOCK_THREADS: u32,
        const ITEMS_PER_THREAD: u32,
        const ITEMS_PER_THREAD_USIZE: usize,
        const BLOCK_WARPS: usize,
        const MIN_BLOCKS_PER_SM: u32,
        const UNROLL: u32,
        const MERGE_BLOCK_THREADS: u32,
        KB,
        VB,
    >(
        self,
        exec: &Execution<'_, A>,
        keys: &mut KB,
        values: &mut VB,
    ) -> Result<(), SortError>
    where
        KB: DeviceSliceMut<K> + ?Sized,
        VB: DeviceSliceMut<V> + ?Sized,
    {
        let comparison = DeviceMergeSort::new(exec.kernel_context(), exec.allocator().clone())?;
        let stream = exec.stream();
        let mut workspace = comparison.allocate_workspace::<K, V>(stream, keys.len())?;
        unsafe {
            comparison.stable_sort_pairs_by::<MERGE_BLOCK_THREADS, _, _, _, _, _>(
                stream,
                keys,
                values,
                self,
                &mut workspace,
            )?;
        }
        Ok(())
    }
}

trait NaturalKeyDispatch<A>: DeviceCopy + PartialOrd + 'static
where
    A: DeviceAllocator + Clone,
{
    fn sort_natural<
        const BLOCK_THREADS: u32,
        const ITEMS_PER_THREAD: u32,
        const ITEMS_PER_THREAD_USIZE: usize,
        const BLOCK_WARPS: usize,
        const MIN_BLOCKS_PER_SM: u32,
        const UNROLL: u32,
        const MERGE_BLOCK_THREADS: u32,
        B,
    >(
        exec: &Execution<'_, A>,
        keys: &mut B,
    ) -> Result<(), SortError>
    where
        B: DeviceSliceMut<Self> + ?Sized;
}

impl<K, A> NaturalKeyDispatch<A> for K
where
    K: DeviceCopy + PartialOrd + 'static,
    A: DeviceAllocator + Clone,
{
    default fn sort_natural<
        const BLOCK_THREADS: u32,
        const ITEMS_PER_THREAD: u32,
        const ITEMS_PER_THREAD_USIZE: usize,
        const BLOCK_WARPS: usize,
        const MIN_BLOCKS_PER_SM: u32,
        const UNROLL: u32,
        const MERGE_BLOCK_THREADS: u32,
        B,
    >(
        exec: &Execution<'_, A>,
        keys: &mut B,
    ) -> Result<(), SortError>
    where
        B: DeviceSliceMut<K> + ?Sized,
    {
        let comparison = DeviceMergeSort::new(exec.kernel_context(), exec.allocator().clone())?;
        let stream = exec.stream();
        let mut workspace = comparison.allocate_key_workspace::<K>(stream, keys.len())?;
        unsafe {
            comparison.stable_sort_keys_by::<MERGE_BLOCK_THREADS, _, _, _>(
                stream,
                keys,
                natural_order::<K>,
                &mut workspace,
            )?;
        }
        Ok(())
    }
}

impl<A> NaturalKeyDispatch<A> for u32
where
    A: DeviceAllocator + Clone,
{
    fn sort_natural<
        const BLOCK_THREADS: u32,
        const ITEMS_PER_THREAD: u32,
        const ITEMS_PER_THREAD_USIZE: usize,
        const BLOCK_WARPS: usize,
        const MIN_BLOCKS_PER_SM: u32,
        const UNROLL: u32,
        const MERGE_BLOCK_THREADS: u32,
        B,
    >(
        exec: &Execution<'_, A>,
        keys: &mut B,
    ) -> Result<(), SortError>
    where
        B: DeviceSliceMut<u32> + ?Sized,
    {
        let radix = DeviceRadixSort::new(exec.kernel_context(), exec.allocator().clone())?;
        let stream = exec.stream();
        let mut workspace =
            radix.allocate_u32_workspace::<BLOCK_THREADS, ITEMS_PER_THREAD>(stream, keys.len())?;
        unsafe {
            radix.sort_u32::<
                BLOCK_THREADS,
                ITEMS_PER_THREAD,
                ITEMS_PER_THREAD_USIZE,
                BLOCK_WARPS,
                MIN_BLOCKS_PER_SM,
                UNROLL,
                false,
                _,
            >(stream, keys, &mut workspace)?;
        }
        Ok(())
    }
}

trait NaturalPairDispatch<V, A>: DeviceCopy + PartialOrd + 'static
where
    V: DeviceCopy + 'static,
    A: DeviceAllocator + Clone,
{
    fn sort_natural_pairs<
        const BLOCK_THREADS: u32,
        const ITEMS_PER_THREAD: u32,
        const ITEMS_PER_THREAD_USIZE: usize,
        const BLOCK_WARPS: usize,
        const MIN_BLOCKS_PER_SM: u32,
        const UNROLL: u32,
        const MERGE_BLOCK_THREADS: u32,
        KB,
        VB,
    >(
        exec: &Execution<'_, A>,
        keys: &mut KB,
        values: &mut VB,
    ) -> Result<(), SortError>
    where
        KB: DeviceSliceMut<Self> + ?Sized,
        VB: DeviceSliceMut<V> + ?Sized;
}

impl<K, V, A> NaturalPairDispatch<V, A> for K
where
    K: DeviceCopy + PartialOrd + 'static,
    V: DeviceCopy + 'static,
    A: DeviceAllocator + Clone,
{
    default fn sort_natural_pairs<
        const BLOCK_THREADS: u32,
        const ITEMS_PER_THREAD: u32,
        const ITEMS_PER_THREAD_USIZE: usize,
        const BLOCK_WARPS: usize,
        const MIN_BLOCKS_PER_SM: u32,
        const UNROLL: u32,
        const MERGE_BLOCK_THREADS: u32,
        KB,
        VB,
    >(
        exec: &Execution<'_, A>,
        keys: &mut KB,
        values: &mut VB,
    ) -> Result<(), SortError>
    where
        KB: DeviceSliceMut<K> + ?Sized,
        VB: DeviceSliceMut<V> + ?Sized,
    {
        let comparison = DeviceMergeSort::new(exec.kernel_context(), exec.allocator().clone())?;
        let stream = exec.stream();
        let mut workspace = comparison.allocate_workspace::<K, V>(stream, keys.len())?;
        unsafe {
            comparison.stable_sort_pairs_by::<MERGE_BLOCK_THREADS, _, _, _, _, _>(
                stream,
                keys,
                values,
                natural_order::<K>,
                &mut workspace,
            )?;
        }
        Ok(())
    }
}

impl<A> NaturalPairDispatch<u32, A> for u32
where
    A: DeviceAllocator + Clone,
{
    fn sort_natural_pairs<
        const BLOCK_THREADS: u32,
        const ITEMS_PER_THREAD: u32,
        const ITEMS_PER_THREAD_USIZE: usize,
        const BLOCK_WARPS: usize,
        const MIN_BLOCKS_PER_SM: u32,
        const UNROLL: u32,
        const MERGE_BLOCK_THREADS: u32,
        KB,
        VB,
    >(
        exec: &Execution<'_, A>,
        keys: &mut KB,
        values: &mut VB,
    ) -> Result<(), SortError>
    where
        KB: DeviceSliceMut<u32> + ?Sized,
        VB: DeviceSliceMut<u32> + ?Sized,
    {
        let radix = DeviceRadixSort::new(exec.kernel_context(), exec.allocator().clone())?;
        let stream = exec.stream();
        let mut workspace = radix
            .allocate_u32_pair_workspace::<BLOCK_THREADS, ITEMS_PER_THREAD>(stream, keys.len())?;
        unsafe {
            radix.stable_sort_by_key_u32::<
                BLOCK_THREADS,
                ITEMS_PER_THREAD,
                ITEMS_PER_THREAD_USIZE,
                BLOCK_WARPS,
                MIN_BLOCKS_PER_SM,
                UNROLL,
                false,
                _,
                _,
            >(stream, keys, values, &mut workspace)?;
        }
        Ok(())
    }
}

fn sort_with_schedule<
    const BLOCK_THREADS: u32,
    const ITEMS_PER_THREAD: u32,
    const ITEMS_PER_THREAD_USIZE: usize,
    const BLOCK_WARPS: usize,
    const MIN_BLOCKS_PER_SM: u32,
    const UNROLL: u32,
    const MERGE_BLOCK_THREADS: u32,
    K,
    B,
    A,
>(
    exec: &Execution<'_, A>,
    keys: &mut B,
) -> Result<(), SortError>
where
    K: DeviceCopy + PartialOrd + 'static,
    B: DeviceSliceMut<K> + ?Sized,
    A: DeviceAllocator + Clone,
{
    let operation = <K as NaturalKeyDispatch<A>>::sort_natural::<
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        ITEMS_PER_THREAD_USIZE,
        BLOCK_WARPS,
        MIN_BLOCKS_PER_SM,
        UNROLL,
        MERGE_BLOCK_THREADS,
        B,
    >(exec, keys);
    finish_synchronous(exec.stream(), operation)
}

fn sort_by_with_schedule<
    const BLOCK_THREADS: u32,
    const ITEMS_PER_THREAD: u32,
    const ITEMS_PER_THREAD_USIZE: usize,
    const BLOCK_WARPS: usize,
    const MIN_BLOCKS_PER_SM: u32,
    const UNROLL: u32,
    const MERGE_BLOCK_THREADS: u32,
    C,
    K,
    B,
    A,
>(
    exec: &Execution<'_, A>,
    keys: &mut B,
    order: C,
) -> Result<(), SortError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    B: DeviceSliceMut<K> + ?Sized,
    A: DeviceAllocator + Clone,
{
    let operation = <C as KeyOrderDispatch<K, A>>::sort_keys::<
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        ITEMS_PER_THREAD_USIZE,
        BLOCK_WARPS,
        MIN_BLOCKS_PER_SM,
        UNROLL,
        MERGE_BLOCK_THREADS,
        B,
    >(order, exec, keys);
    finish_synchronous(exec.stream(), operation)
}

fn sort_by_key_with_schedule_defaults<
    const BLOCK_THREADS: u32,
    const ITEMS_PER_THREAD: u32,
    const ITEMS_PER_THREAD_USIZE: usize,
    const BLOCK_WARPS: usize,
    const MIN_BLOCKS_PER_SM: u32,
    const UNROLL: u32,
    const MERGE_BLOCK_THREADS: u32,
    K,
    V,
    KB,
    VB,
    A,
>(
    exec: &Execution<'_, A>,
    keys: &mut KB,
    values: &mut VB,
) -> Result<(), SortError>
where
    K: DeviceCopy + PartialOrd + 'static,
    V: DeviceCopy + 'static,
    KB: DeviceSliceMut<K> + ?Sized,
    VB: DeviceSliceMut<V> + ?Sized,
    A: DeviceAllocator + Clone,
{
    let operation = <K as NaturalPairDispatch<V, A>>::sort_natural_pairs::<
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        ITEMS_PER_THREAD_USIZE,
        BLOCK_WARPS,
        MIN_BLOCKS_PER_SM,
        UNROLL,
        MERGE_BLOCK_THREADS,
        KB,
        VB,
    >(exec, keys, values);
    finish_synchronous(exec.stream(), operation)
}

/// Ordering-closure form for key/value sorting. Rust has no overload by arity,
/// so the explicit ordering form consistently uses `_with`.
fn sort_by_key_with_schedule<
    const BLOCK_THREADS: u32,
    const ITEMS_PER_THREAD: u32,
    const ITEMS_PER_THREAD_USIZE: usize,
    const BLOCK_WARPS: usize,
    const MIN_BLOCKS_PER_SM: u32,
    const UNROLL: u32,
    const MERGE_BLOCK_THREADS: u32,
    C,
    K,
    V,
    KB,
    VB,
    A,
>(
    exec: &Execution<'_, A>,
    keys: &mut KB,
    values: &mut VB,
    order: C,
) -> Result<(), SortError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    V: DeviceCopy + 'static,
    KB: DeviceSliceMut<K> + ?Sized,
    VB: DeviceSliceMut<V> + ?Sized,
    A: DeviceAllocator + Clone,
{
    let operation = <C as PairOrderDispatch<K, V, A>>::sort_pairs::<
        BLOCK_THREADS,
        ITEMS_PER_THREAD,
        ITEMS_PER_THREAD_USIZE,
        BLOCK_WARPS,
        MIN_BLOCKS_PER_SM,
        UNROLL,
        MERGE_BLOCK_THREADS,
        KB,
        VB,
    >(order, exec, keys, values);
    finish_synchronous(exec.stream(), operation)
}

/// Sort a device range in ascending order.
pub fn sort<K, B, A>(exec: &Execution<'_, A>, keys: &mut B) -> Result<(), SortError>
where
    K: DeviceCopy + PartialOrd + 'static,
    B: DeviceSliceMut<K> + ?Sized,
    A: DeviceAllocator + Clone,
{
    sort_with_schedule::<
        RADIX_BLOCK_THREADS,
        RADIX_ITEMS_PER_THREAD,
        RADIX_ITEMS_PER_THREAD_USIZE,
        RADIX_BLOCK_WARPS,
        RADIX_MIN_BLOCKS_PER_SM,
        RADIX_UNROLL,
        COMPARISON_BLOCK_THREADS,
        K,
        B,
        A,
    >(exec, keys)
}

/// Sort a device range using a state-carrying device order.
pub fn sort_by<C, K, B, A>(exec: &Execution<'_, A>, keys: &mut B, order: C) -> Result<(), SortError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    B: DeviceSliceMut<K> + ?Sized,
    A: DeviceAllocator + Clone,
{
    sort_by_with_schedule::<
        RADIX_BLOCK_THREADS,
        RADIX_ITEMS_PER_THREAD,
        RADIX_ITEMS_PER_THREAD_USIZE,
        RADIX_BLOCK_WARPS,
        RADIX_MIN_BLOCKS_PER_SM,
        RADIX_UNROLL,
        COMPARISON_BLOCK_THREADS,
        C,
        K,
        B,
        A,
    >(exec, keys, order)
}

/// Stably sort a device range in ascending order.
pub fn stable_sort<K, B, A>(exec: &Execution<'_, A>, keys: &mut B) -> Result<(), SortError>
where
    K: DeviceCopy + PartialOrd + 'static,
    B: DeviceSliceMut<K> + ?Sized,
    A: DeviceAllocator + Clone,
{
    sort(exec, keys)
}

/// Stably sort a device range using a state-carrying device order.
pub fn stable_sort_by<C, K, B, A>(
    exec: &Execution<'_, A>,
    keys: &mut B,
    order: C,
) -> Result<(), SortError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    B: DeviceSliceMut<K> + ?Sized,
    A: DeviceAllocator + Clone,
{
    sort_by(exec, keys, order)
}

/// Stably sort key/value pairs by ascending key.
pub fn sort_by_key<K, V, KB, VB, A>(
    exec: &Execution<'_, A>,
    keys: &mut KB,
    values: &mut VB,
) -> Result<(), SortError>
where
    K: DeviceCopy + PartialOrd + 'static,
    V: DeviceCopy + 'static,
    KB: DeviceSliceMut<K> + ?Sized,
    VB: DeviceSliceMut<V> + ?Sized,
    A: DeviceAllocator + Clone,
{
    sort_by_key_with_schedule_defaults::<
        RADIX_BLOCK_THREADS,
        RADIX_ITEMS_PER_THREAD,
        RADIX_ITEMS_PER_THREAD_USIZE,
        RADIX_BLOCK_WARPS,
        RADIX_MIN_BLOCKS_PER_SM,
        RADIX_UNROLL,
        COMPARISON_BLOCK_THREADS,
        K,
        V,
        KB,
        VB,
        A,
    >(exec, keys, values)
}

/// Stably sort key/value pairs using a state-carrying key order.
pub fn sort_by_key_with<C, K, V, KB, VB, A>(
    exec: &Execution<'_, A>,
    keys: &mut KB,
    values: &mut VB,
    order: C,
) -> Result<(), SortError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    V: DeviceCopy + 'static,
    KB: DeviceSliceMut<K> + ?Sized,
    VB: DeviceSliceMut<V> + ?Sized,
    A: DeviceAllocator + Clone,
{
    sort_by_key_with_schedule::<
        RADIX_BLOCK_THREADS,
        RADIX_ITEMS_PER_THREAD,
        RADIX_ITEMS_PER_THREAD_USIZE,
        RADIX_BLOCK_WARPS,
        RADIX_MIN_BLOCKS_PER_SM,
        RADIX_UNROLL,
        COMPARISON_BLOCK_THREADS,
        C,
        K,
        V,
        KB,
        VB,
        A,
    >(exec, keys, values, order)
}

/// Alias emphasizing that key/value sorting is stable.
pub fn stable_sort_by_key<K, V, KB, VB, A>(
    exec: &Execution<'_, A>,
    keys: &mut KB,
    values: &mut VB,
) -> Result<(), SortError>
where
    K: DeviceCopy + PartialOrd + 'static,
    V: DeviceCopy + 'static,
    KB: DeviceSliceMut<K> + ?Sized,
    VB: DeviceSliceMut<V> + ?Sized,
    A: DeviceAllocator + Clone,
{
    sort_by_key(exec, keys, values)
}

/// Alias emphasizing that order key/value sorting is stable.
pub fn stable_sort_by_key_with<C, K, V, KB, VB, A>(
    exec: &Execution<'_, A>,
    keys: &mut KB,
    values: &mut VB,
    order: C,
) -> Result<(), SortError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    V: DeviceCopy + 'static,
    KB: DeviceSliceMut<K> + ?Sized,
    VB: DeviceSliceMut<V> + ?Sized,
    A: DeviceAllocator + Clone,
{
    sort_by_key_with(exec, keys, values, order)
}
