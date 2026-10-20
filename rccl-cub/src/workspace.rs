use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_memory::TemporaryValue;

use crate::{DeviceAlgorithmError, cuda::Launch};

const MERGE_THREADS: usize = 512;
const MERGE_NOMINAL_4B_ITEMS: usize = 15;
const MERGE_SORT_THREADS: usize = 256;
const MERGE_SORT_NOMINAL_4B_ITEMS: usize = 17;
const SET_THREADS: usize = 512;
const SET_NOMINAL_4B_ITEMS: usize = 19;
const RADIX_BINS: usize = 256;

use crate::radix::{RadixKey, RadixValue};

#[derive(Clone, Copy)]
pub(crate) struct RadixPolicy {
    pub(crate) passes: usize,
    pub(crate) single_tile_items: usize,
    pub(crate) tile_items: usize,
}

pub(crate) const fn radix_reg_bound(
    nominal_threads: usize,
    nominal_items: usize,
    type_size: usize,
) -> (usize, usize) {
    let divisor = if type_size > 4 { type_size } else { 4 };
    let candidate = nominal_items * 4 / divisor;
    let items = if candidate > 1 { candidate } else { 1 };
    let static_shared_words = (48 * 1024) / (type_size * items);
    let shared_limited_threads = static_shared_words.div_ceil(32) * 32;
    (
        if nominal_threads < shared_limited_threads {
            nominal_threads
        } else {
            shared_limited_threads
        },
        items,
    )
}

pub(crate) fn radix_keys_policy<K: RadixKey>() -> RadixPolicy {
    let key_size = core::mem::size_of::<K>();
    let (_, single_items) = radix_reg_bound(256, 19, key_size);
    let (onesweep_threads, onesweep_items) = radix_reg_bound(384, 21, key_size);
    RadixPolicy {
        passes: K::BITS.div_ceil(8) as usize,
        single_tile_items: 256 * single_items,
        tile_items: onesweep_threads * onesweep_items,
    }
}

pub(crate) fn radix_pairs_policy<K: RadixKey, V: RadixValue>() -> RadixPolicy {
    let key_size = core::mem::size_of::<K>();
    let value_size = core::mem::size_of::<V>();
    let item_size = key_size.max(value_size);
    let nominal_items = if key_size == 4 { 17 } else { 21 };
    let (_, single_items) = radix_reg_bound(256, 19, item_size);
    let (onesweep_threads, onesweep_items) = radix_reg_bound(384, nominal_items, item_size);
    RadixPolicy {
        passes: K::BITS.div_ceil(8) as usize,
        single_tile_items: 256 * single_items,
        tile_items: onesweep_threads * onesweep_items,
    }
}

fn radix_workspace_counts(len: usize, policy: RadixPolicy) -> RadixSortWorkspaceCounts {
    RadixSortWorkspaceCounts {
        scratch_keys: len,
        bins: policy.passes * RADIX_BINS,
        lookback: len.div_ceil(policy.tile_items) * RADIX_BINS,
        counters: policy.passes,
    }
}

#[doc(hidden)]
pub fn device_radix_sort_workspace_counts<K: RadixKey>(len: usize) -> RadixSortWorkspaceCounts {
    radix_workspace_counts(len, radix_keys_policy::<K>())
}

#[doc(hidden)]
pub fn device_radix_sort_pairs_workspace_counts<K: RadixKey, V: RadixValue>(
    len: usize,
) -> RadixSortPairsWorkspaceCounts {
    let counts = radix_workspace_counts(len, radix_pairs_policy::<K, V>());
    RadixSortPairsWorkspaceCounts {
        scratch_keys: counts.scratch_keys,
        scratch_values: len,
        bins: counts.bins,
        lookback: counts.lookback,
        counters: counts.counters,
    }
}

fn radix_control_lengths(len: usize, policy: RadixPolicy) -> (usize, usize, usize) {
    if len <= policy.single_tile_items {
        (0, 0, 0)
    } else {
        (
            policy.passes * RADIX_BINS,
            len.div_ceil(policy.tile_items) * RADIX_BINS,
            policy.passes,
        )
    }
}

fn device_radix_sort_buffer_workspace_for(
    execution: &Launch<'_, '_>,
    len: usize,
    policy: RadixPolicy,
) -> Result<DeviceRadixSortBufferWorkspace, DeviceAlgorithmError> {
    let (bins, lookback, counters) = radix_control_lengths(len, policy);
    let bins = DeviceBuffer::<u64>::zeroed(execution.stream(), bins)
        .map_err(DeviceAlgorithmError::Driver)?;
    let lookback = DeviceBuffer::<i32>::zeroed(execution.stream(), lookback)
        .map_err(DeviceAlgorithmError::Driver)?;
    let counters = DeviceBuffer::<i32>::zeroed(execution.stream(), counters)
        .map_err(DeviceAlgorithmError::Driver)?;
    Ok(DeviceRadixSortBufferWorkspace::__from_parts(
        bins, lookback, counters,
    ))
}

/// Allocate reusable control storage for a key-only radix double buffer.
pub fn device_radix_sort_keys_buffer_workspace<K: RadixKey>(
    execution: &Launch<'_, '_>,
    len: usize,
) -> Result<DeviceRadixSortBufferWorkspace, DeviceAlgorithmError> {
    device_radix_sort_buffer_workspace_for(execution, len, radix_keys_policy::<K>())
}

/// Allocate reusable control storage for a key/value radix double buffer.
pub fn device_radix_sort_pairs_buffer_workspace<K: RadixKey, V: RadixValue>(
    execution: &Launch<'_, '_>,
    len: usize,
) -> Result<DeviceRadixSortBufferWorkspace, DeviceAlgorithmError> {
    device_radix_sort_buffer_workspace_for(execution, len, radix_pairs_policy::<K, V>())
}

/// Allocate reusable scratch and control storage for an in-place key radix sort.
pub fn device_radix_sort_keys_workspace<K: RadixKey>(
    execution: &Launch<'_, '_>,
    len: usize,
) -> Result<DeviceRadixSortKeysWorkspace<K>, DeviceAlgorithmError> {
    let scratch_keys =
        DeviceBuffer::<K>::zeroed(execution.stream(), len).map_err(DeviceAlgorithmError::Driver)?;
    let controls = device_radix_sort_keys_buffer_workspace::<K>(execution, len)?;
    Ok(DeviceRadixSortKeysWorkspace::__from_parts(
        scratch_keys,
        controls.bins,
        controls.lookback,
        controls.counters,
    ))
}

/// Allocate reusable scratch and control storage for an in-place key/value radix sort.
pub fn device_radix_sort_pairs_workspace<K: RadixKey, V: RadixValue>(
    execution: &Launch<'_, '_>,
    len: usize,
) -> Result<DeviceRadixSortPairsWorkspace<K, V>, DeviceAlgorithmError> {
    let scratch_keys =
        DeviceBuffer::<K>::zeroed(execution.stream(), len).map_err(DeviceAlgorithmError::Driver)?;
    let scratch_values =
        DeviceBuffer::<V>::zeroed(execution.stream(), len).map_err(DeviceAlgorithmError::Driver)?;
    let controls = device_radix_sort_pairs_buffer_workspace::<K, V>(execution, len)?;
    Ok(DeviceRadixSortPairsWorkspace::__from_parts(
        scratch_keys,
        scratch_values,
        controls.bins,
        controls.lookback,
        controls.counters,
    ))
}

fn nominal_4b_items<T>(nominal: usize) -> Result<usize, DeviceAlgorithmError> {
    let size = core::mem::size_of::<T>();
    if size == 0 {
        return Err(DeviceAlgorithmError::SizeOverflow);
    }
    Ok((nominal * 4 / size).clamp(1, nominal))
}

fn partition_count(total: usize, tile_items: usize) -> usize {
    if total == 0 {
        0
    } else {
        total.div_ceil(tile_items) + 1
    }
}

#[doc(hidden)]
pub fn device_merge_partition_count<K>(
    first_len: usize,
    second_len: usize,
) -> Result<usize, DeviceAlgorithmError> {
    let total = first_len
        .checked_add(second_len)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let items = nominal_4b_items::<K>(MERGE_NOMINAL_4B_ITEMS)?;
    Ok(partition_count(total, MERGE_THREADS * items))
}

#[doc(hidden)]
pub fn device_merge_pairs_partition_count<K, V>(
    first_len: usize,
    second_len: usize,
) -> Result<usize, DeviceAlgorithmError> {
    let total = first_len
        .checked_add(second_len)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let size = core::mem::size_of::<K>()
        .checked_add(core::mem::size_of::<V>())
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if size == 0 {
        return Err(DeviceAlgorithmError::SizeOverflow);
    }
    let items = (MERGE_NOMINAL_4B_ITEMS * 4 / size).clamp(1, MERGE_NOMINAL_4B_ITEMS);
    Ok(partition_count(total, MERGE_THREADS * items))
}

/// Allocate reusable partition storage for merging two key ranges.
pub fn device_merge_workspace<K>(
    execution: &Launch<'_, '_>,
    first_len: usize,
    second_len: usize,
) -> Result<DeviceMergeWorkspace, DeviceAlgorithmError> {
    let count = device_merge_partition_count::<K>(first_len, second_len)?;
    let partitions = DeviceBuffer::<usize>::zeroed(execution.stream(), count)
        .map_err(DeviceAlgorithmError::Driver)?;
    Ok(DeviceMergeWorkspace::__from_partitions(partitions))
}

/// Allocate reusable partition storage for merging two key/value ranges.
pub fn device_merge_pairs_workspace<K, V>(
    execution: &Launch<'_, '_>,
    first_len: usize,
    second_len: usize,
) -> Result<DeviceMergeWorkspace, DeviceAlgorithmError> {
    let count = device_merge_pairs_partition_count::<K, V>(first_len, second_len)?;
    let partitions = DeviceBuffer::<usize>::zeroed(execution.stream(), count)
        .map_err(DeviceAlgorithmError::Driver)?;
    Ok(DeviceMergeWorkspace::__from_partitions(partitions))
}

#[doc(hidden)]
pub fn device_merge_sort_partition_count<K>(len: usize) -> Result<usize, DeviceAlgorithmError> {
    let items = nominal_4b_items::<K>(MERGE_SORT_NOMINAL_4B_ITEMS)?;
    Ok(partition_count(len, MERGE_SORT_THREADS * items))
}

/// Allocate reusable scratch and partition storage for merge-sorting keys.
pub fn device_merge_sort_keys_workspace<K: DeviceCopy>(
    execution: &Launch<'_, '_>,
    len: usize,
) -> Result<DeviceMergeSortKeysWorkspace<K>, DeviceAlgorithmError> {
    let count = device_merge_sort_partition_count::<K>(len)?;
    let scratch_keys =
        DeviceBuffer::<K>::zeroed(execution.stream(), len).map_err(DeviceAlgorithmError::Driver)?;
    let partitions = DeviceBuffer::<usize>::zeroed(execution.stream(), count)
        .map_err(DeviceAlgorithmError::Driver)?;
    Ok(DeviceMergeSortKeysWorkspace::__from_parts(
        scratch_keys,
        partitions,
    ))
}

/// Allocate reusable scratch and partition storage for merge-sorting key/value pairs.
pub fn device_merge_sort_pairs_workspace<K: DeviceCopy, V: DeviceCopy>(
    execution: &Launch<'_, '_>,
    len: usize,
) -> Result<DeviceMergeSortPairsWorkspace<K, V>, DeviceAlgorithmError> {
    let count = device_merge_sort_partition_count::<K>(len)?;
    let scratch_keys =
        DeviceBuffer::<K>::zeroed(execution.stream(), len).map_err(DeviceAlgorithmError::Driver)?;
    let scratch_values =
        DeviceBuffer::<V>::zeroed(execution.stream(), len).map_err(DeviceAlgorithmError::Driver)?;
    let partitions = DeviceBuffer::<usize>::zeroed(execution.stream(), count)
        .map_err(DeviceAlgorithmError::Driver)?;
    Ok(DeviceMergeSortPairsWorkspace::__from_parts(
        scratch_keys,
        scratch_values,
        partitions,
    ))
}

#[doc(hidden)]
pub fn device_set_workspace_counts<K, V>(
    stream: &cuda_core::CudaStream,
    first_len: usize,
    second_len: usize,
) -> Result<SetOperationWorkspaceCounts, DeviceAlgorithmError> {
    let max_shared = stream
        .context()
        .launch_limits()
        .map_err(DeviceAlgorithmError::Driver)?
        .max_shared_memory_per_block() as usize;
    set_workspace_counts_for_limit::<K, V>(first_len, second_len, max_shared)
}

fn set_workspace_counts_for_limit<K, V>(
    first_len: usize,
    second_len: usize,
    max_shared: usize,
) -> Result<SetOperationWorkspaceCounts, DeviceAlgorithmError> {
    let total = first_len
        .checked_add(second_len)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if total == 0 {
        return Ok(SetOperationWorkspaceCounts {
            num_tiles: 0,
            tile_descriptors: 0,
            partitions: 0,
            virtual_shared_bytes: 0,
            wide_indices: false,
        });
    }

    let key_size = core::mem::size_of::<K>();
    let value_size = core::mem::size_of::<V>();
    if key_size == 0 || value_size == 0 {
        return Err(DeviceAlgorithmError::SizeOverflow);
    }
    let items = (SET_NOMINAL_4B_ITEMS * 4)
        .div_ceil(key_size)
        .clamp(1, SET_NOMINAL_4B_ITEMS);
    let tile_items = SET_THREADS * items - 1;
    let shared_items = tile_items
        .checked_add(SET_THREADS)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let union_bytes = shared_items
        .checked_mul(key_size.max(value_size))
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let shared_bytes = SET_THREADS
        .checked_mul(core::mem::size_of::<i32>())
        .and_then(|offsets| offsets.checked_add(union_bytes))
        .ok_or(DeviceAlgorithmError::SizeOverflow)?
        .checked_add(15)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?
        / 16
        * 16;
    let num_tiles = total.div_ceil(tile_items);
    let virtual_shared_bytes = if shared_bytes > max_shared {
        shared_bytes
            .checked_mul(num_tiles)
            .ok_or(DeviceAlgorithmError::SizeOverflow)?
    } else {
        0
    };
    Ok(SetOperationWorkspaceCounts {
        num_tiles,
        tile_descriptors: num_tiles + 32,
        partitions: num_tiles + 1,
        virtual_shared_bytes,
        wide_indices: total > i32::MAX as usize,
    })
}

#[doc(hidden)]
pub struct RadixSortWorkspaceCounts {
    pub scratch_keys: usize,
    pub bins: usize,
    pub lookback: usize,
    pub counters: usize,
}

#[doc(hidden)]
pub struct RadixSortPairsWorkspaceCounts {
    pub scratch_keys: usize,
    pub scratch_values: usize,
    pub bins: usize,
    pub lookback: usize,
    pub counters: usize,
}

/// Runtime extents selected by the compile-time set-operation policy.
#[doc(hidden)]
pub struct SetOperationWorkspaceCounts {
    pub num_tiles: usize,
    pub tile_descriptors: usize,
    pub partitions: usize,
    pub virtual_shared_bytes: usize,
    pub wide_indices: bool,
}
/// Normalize an owned direct buffer and a borrowed aliased temporary region to
/// the same mutable range surface.
#[doc(hidden)]
pub trait WorkspaceBuffer<T: DeviceCopy> {
    type Range;

    fn range_mut(&mut self) -> &mut Self::Range;
}

impl<T: DeviceCopy> WorkspaceBuffer<T> for DeviceBuffer<T> {
    type Range = DeviceBuffer<T>;

    fn range_mut(&mut self) -> &mut Self::Range {
        self
    }
}

/// A caller-owned buffer lent to one workspace, e.g. stream-ordered scratch
/// the caller releases with `drop_async` after the operation.
impl<T: DeviceCopy> WorkspaceBuffer<T> for &mut DeviceBuffer<T> {
    type Range = DeviceBuffer<T>;

    fn range_mut(&mut self) -> &mut Self::Range {
        self
    }
}

impl<'stream, T: TemporaryValue> WorkspaceBuffer<T>
    for &mut rccl_memory::TemporaryRegion<'stream, T>
{
    type Range = rccl_memory::TemporaryRegion<'stream, T>;

    fn range_mut(&mut self) -> &mut Self::Range {
        self
    }
}

/// Opaque temporary storage for one direct DeviceMerge operation.
pub struct DeviceMergeWorkspace<P = DeviceBuffer<usize>> {
    partitions: P,
}

impl<P: WorkspaceBuffer<usize>> DeviceMergeWorkspace<P> {
    #[doc(hidden)]
    pub fn __from_partitions(partitions: P) -> Self {
        Self { partitions }
    }

    #[doc(hidden)]
    pub fn __partitions_mut(&mut self) -> &mut P::Range {
        self.partitions.range_mut()
    }
}

/// Opaque regions inside Thrust's one nested set-operation allocation.
#[doc(hidden)]
pub struct DeviceSetOperationWorkspace<O, D, P, V> {
    output_count: O,
    tile_descriptors: D,
    partitions: P,
    virtual_shared: V,
}

impl<O, D, P, V> DeviceSetOperationWorkspace<O, D, P, V> {
    #[doc(hidden)]
    pub fn __from_parts(
        output_count: O,
        tile_descriptors: D,
        partitions: P,
        virtual_shared: V,
    ) -> Self {
        Self {
            output_count,
            tile_descriptors,
            partitions,
            virtual_shared,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut<I>(&mut self) -> (&mut O::Range, &mut D::Range, &mut P::Range, &mut V::Range)
    where
        I: crate::ordering_backend::SetIndex,
        I::Transaction: DeviceCopy,
        O: WorkspaceBuffer<usize>,
        D: WorkspaceBuffer<I::Transaction>,
        P: WorkspaceBuffer<I::Transaction>,
        V: WorkspaceBuffer<u8>,
    {
        (
            self.output_count.range_mut(),
            self.tile_descriptors.range_mut(),
            self.partitions.range_mut(),
            self.virtual_shared.range_mut(),
        )
    }
}

impl<O, D, P, V> DeviceSetOperationWorkspace<O, D, P, V>
where
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<u64>,
    P: WorkspaceBuffer<u64>,
    V: WorkspaceBuffer<u8>,
{
    #[doc(hidden)]
    pub fn __narrow_parts_mut(
        &mut self,
    ) -> (&mut O::Range, &mut D::Range, &mut P::Range, &mut V::Range) {
        (
            self.output_count.range_mut(),
            self.tile_descriptors.range_mut(),
            self.partitions.range_mut(),
            self.virtual_shared.range_mut(),
        )
    }
}

impl<O, D, P, V> DeviceSetOperationWorkspace<O, D, P, V>
where
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<u128>,
    P: WorkspaceBuffer<u128>,
    V: WorkspaceBuffer<u8>,
{
    #[doc(hidden)]
    pub fn __wide_parts_mut(
        &mut self,
    ) -> (&mut O::Range, &mut D::Range, &mut P::Range, &mut V::Range) {
        (
            self.output_count.range_mut(),
            self.tile_descriptors.range_mut(),
            self.partitions.range_mut(),
            self.virtual_shared.range_mut(),
        )
    }
}

/// Opaque temporary storage for direct key-only DeviceMergeSort operations.
pub struct DeviceMergeSortKeysWorkspace<K: DeviceCopy, S = DeviceBuffer<K>, P = DeviceBuffer<usize>>
{
    scratch_keys: S,
    partitions: P,
    key: core::marker::PhantomData<fn() -> K>,
}

impl<K: DeviceCopy, S: WorkspaceBuffer<K>, P: WorkspaceBuffer<usize>>
    DeviceMergeSortKeysWorkspace<K, S, P>
{
    #[doc(hidden)]
    pub fn __from_parts(scratch_keys: S, partitions: P) -> Self {
        Self {
            scratch_keys,
            partitions,
            key: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut S::Range, &mut P::Range) {
        (self.scratch_keys.range_mut(), self.partitions.range_mut())
    }
}

/// Opaque temporary storage for direct key/value DeviceMergeSort operations.
pub struct DeviceMergeSortPairsWorkspace<
    K: DeviceCopy,
    V: DeviceCopy,
    S = DeviceBuffer<K>,
    U = DeviceBuffer<V>,
    P = DeviceBuffer<usize>,
> {
    scratch_keys: S,
    scratch_values: U,
    partitions: P,
    types: core::marker::PhantomData<fn() -> (K, V)>,
}

impl<
    K: DeviceCopy,
    V: DeviceCopy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    P: WorkspaceBuffer<usize>,
> DeviceMergeSortPairsWorkspace<K, V, S, U, P>
{
    #[doc(hidden)]
    pub fn __from_parts(scratch_keys: S, scratch_values: U, partitions: P) -> Self {
        Self {
            scratch_keys,
            scratch_values,
            partitions,
            types: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut S::Range, &mut U::Range, &mut P::Range) {
        (
            self.scratch_keys.range_mut(),
            self.scratch_values.range_mut(),
            self.partitions.range_mut(),
        )
    }
}

/// Opaque CUB-owned state used with caller-owned radix double buffers.
pub struct DeviceRadixSortBufferWorkspace<
    B = DeviceBuffer<u64>,
    L = DeviceBuffer<i32>,
    C = DeviceBuffer<i32>,
> {
    bins: B,
    lookback: L,
    counters: C,
}

impl<B: WorkspaceBuffer<u64>, L: WorkspaceBuffer<i32>, C: WorkspaceBuffer<i32>>
    DeviceRadixSortBufferWorkspace<B, L, C>
{
    #[doc(hidden)]
    pub fn __from_parts(bins: B, lookback: L, counters: C) -> Self {
        Self {
            bins,
            lookback,
            counters,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut B::Range, &mut L::Range, &mut C::Range) {
        (
            self.bins.range_mut(),
            self.lookback.range_mut(),
            self.counters.range_mut(),
        )
    }
}

/// Opaque temporary storage for direct key-only DeviceRadixSort operations.
pub struct DeviceRadixSortKeysWorkspace<
    K: DeviceCopy,
    S = DeviceBuffer<K>,
    B = DeviceBuffer<u64>,
    L = DeviceBuffer<i32>,
    C = DeviceBuffer<i32>,
> {
    scratch_keys: S,
    controls: DeviceRadixSortBufferWorkspace<B, L, C>,
    key: core::marker::PhantomData<fn() -> K>,
}

impl<
    K: DeviceCopy,
    S: WorkspaceBuffer<K>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
> DeviceRadixSortKeysWorkspace<K, S, B, L, C>
{
    #[doc(hidden)]
    pub fn __from_parts(scratch_keys: S, bins: B, lookback: L, counters: C) -> Self {
        Self {
            scratch_keys,
            controls: DeviceRadixSortBufferWorkspace::__from_parts(bins, lookback, counters),
            key: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut S::Range, &mut DeviceRadixSortBufferWorkspace<B, L, C>) {
        (self.scratch_keys.range_mut(), &mut self.controls)
    }
}

/// Opaque temporary storage for direct key/value DeviceRadixSort operations.
pub struct DeviceRadixSortPairsWorkspace<
    K: DeviceCopy,
    V: DeviceCopy,
    S = DeviceBuffer<K>,
    U = DeviceBuffer<V>,
    B = DeviceBuffer<u64>,
    L = DeviceBuffer<i32>,
    C = DeviceBuffer<i32>,
> {
    scratch_keys: S,
    scratch_values: U,
    controls: DeviceRadixSortBufferWorkspace<B, L, C>,
    types: core::marker::PhantomData<fn() -> (K, V)>,
}

impl<
    K: DeviceCopy,
    V: DeviceCopy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
> DeviceRadixSortPairsWorkspace<K, V, S, U, B, L, C>
{
    #[doc(hidden)]
    pub fn __from_parts(
        scratch_keys: S,
        scratch_values: U,
        bins: B,
        lookback: L,
        counters: C,
    ) -> Self {
        Self {
            scratch_keys,
            scratch_values,
            controls: DeviceRadixSortBufferWorkspace::__from_parts(bins, lookback, counters),
            types: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(
        &mut self,
    ) -> (
        &mut S::Range,
        &mut U::Range,
        &mut DeviceRadixSortBufferWorkspace<B, L, C>,
    ) {
        (
            self.scratch_keys.range_mut(),
            self.scratch_values.range_mut(),
            &mut self.controls,
        )
    }
}

#[cfg(test)]
mod policy_tests {
    use super::*;

    type KeyAlias = u32;

    #[repr(C)]
    struct TwelveByteKey {
        words: [u32; 3],
    }

    #[test]
    fn merge_policy_uses_rust_layout_for_aliases_and_user_types() {
        assert_eq!(
            device_merge_partition_count::<KeyAlias>(7_680, 0).unwrap(),
            2
        );
        assert_eq!(device_merge_partition_count::<u64>(3_584, 0).unwrap(), 2);
        assert_eq!(
            device_merge_partition_count::<TwelveByteKey>(2_560, 0).unwrap(),
            2
        );
        assert_eq!(
            device_merge_pairs_partition_count::<u32, u32>(3_584, 0).unwrap(),
            2
        );
    }

    #[test]
    fn merge_sort_policy_uses_rust_layout_without_macro_type_parsing() {
        assert_eq!(
            device_merge_sort_partition_count::<KeyAlias>(4_352).unwrap(),
            2
        );
        assert_eq!(device_merge_sort_partition_count::<u64>(2_048).unwrap(), 2);
        assert_eq!(
            device_merge_sort_partition_count::<TwelveByteKey>(1_280).unwrap(),
            2
        );
    }

    #[test]
    fn zero_sized_and_overflowing_extents_are_rejected() {
        assert!(matches!(
            device_merge_partition_count::<()>(1, 0),
            Err(DeviceAlgorithmError::SizeOverflow)
        ));
        assert!(matches!(
            device_merge_partition_count::<u32>(usize::MAX, 1),
            Err(DeviceAlgorithmError::SizeOverflow)
        ));
    }

    #[test]
    fn set_policy_matches_the_previous_sm80_plus_workspace_formula() {
        let resident = set_workspace_counts_for_limit::<u32, u32>(9_727, 0, usize::MAX)
            .expect("ordinary Rust policy calculation must succeed");
        assert_eq!(resident.num_tiles, 1);
        assert_eq!(resident.tile_descriptors, 33);
        assert_eq!(resident.partitions, 2);
        assert_eq!(resident.virtual_shared_bytes, 0);
        assert!(!resident.wide_indices);

        let virtualized = set_workspace_counts_for_limit::<u64, u64>(4_607, 0, 0)
            .expect("shared-memory fallback calculation must succeed");
        assert_eq!(virtualized.num_tiles, 1);
        assert_eq!(virtualized.virtual_shared_bytes, 47_104);
    }

    #[test]
    fn radix_workspace_policy_is_ordinary_closed_trait_dispatch() {
        type Alias = u64;

        let keys = device_radix_sort_workspace_counts::<Alias>(3_841);
        assert_eq!(keys.scratch_keys, 3_841);
        assert_eq!(keys.bins, 2_048);
        assert_eq!(keys.lookback, 512);
        assert_eq!(keys.counters, 8);

        let pairs = device_radix_sort_pairs_workspace_counts::<f32, i32>(6_529);
        assert_eq!(pairs.scratch_keys, 6_529);
        assert_eq!(pairs.scratch_values, 6_529);
        assert_eq!(pairs.bins, 1_024);
        assert_eq!(pairs.lookback, 512);
        assert_eq!(pairs.counters, 4);
    }

    #[test]
    fn direct_radix_workspace_omits_unused_onesweep_controls() {
        let u32_policy = radix_keys_policy::<u32>();
        assert_eq!(u32_policy.single_tile_items, 4_864);
        assert_eq!(u32_policy.tile_items, 8_064);
        assert_eq!(radix_control_lengths(4_864, u32_policy), (0, 0, 0));
        assert_eq!(radix_control_lengths(4_865, u32_policy), (1_024, 256, 4));

        let pair_policy = radix_pairs_policy::<u32, u64>();
        assert_eq!(pair_policy.single_tile_items, 2_304);
        assert_eq!(pair_policy.tile_items, 3_072);
    }
}
