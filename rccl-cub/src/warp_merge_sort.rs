//! CUB-compatible, device-callable warp merge-sort collective.
//!
//! Each participating lane owns a blocked register array of `ITEMS_PER_THREAD`
//! keys (and optionally values).  The logical warp first sorts those local
//! arrays, then performs the same shared-memory Merge Path rounds as CUB's
//! `WarpMergeSort`.

use crate::DeviceComparator;
use core::marker::PhantomData;
use cuda_core::DeviceCopy;

/// Compile-time description exposed by a [`WarpMergeSort`] specialization.
///
/// Comparator behavior deliberately is not part of this trait: as in CUB, the
/// comparator is a value supplied to each sort call, while tile geometry and
/// temporary-storage layout belong to the specialization.
pub trait WarpMergeSortConfig {
    type Key: DeviceCopy;
    type Value: DeviceCopy;

    const ITEMS_PER_THREAD: usize;
    const LOGICAL_WARP_THREADS: usize;
    const ITEMS_PER_TILE: usize;
    const TEMP_STORAGE_BYTES: usize;
    const TEMP_STORAGE_ALIGN: usize;
}

/// A typed view of one logical warp's caller-owned temporary storage.
///
/// Values of this type are created by
/// [`warp_merge_sort_temp_storage!`](crate::warp_merge_sort_temp_storage).
/// The key/value types and tile geometry are encoded in the type so storage
/// for a different specialization cannot be passed to [`WarpMergeSort::new`].
pub struct WarpMergeSortTempStorage<
    K,
    const ITEMS_PER_THREAD: usize,
    const LOGICAL_WARP_THREADS: usize = 32,
    V = (),
> {
    pointer: *mut u8,
    _types: PhantomData<fn() -> (K, V)>,
}

impl<K, const ITEMS_PER_THREAD: usize, const LOGICAL_WARP_THREADS: usize, V>
    WarpMergeSortTempStorage<K, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, V>
{
    /// Construct the typed view for `warp_index` in a correctly sized shared
    /// allocation.
    ///
    /// This is public for macro expansion. Direct callers should use
    /// [`warp_merge_sort_temp_storage!`], which computes both allocation size
    /// and alignment from this specialization.
    #[doc(hidden)]
    #[inline(always)]
    pub unsafe fn from_raw_warp(base: *mut u8, warp_index: usize) -> Self {
        Self {
            pointer: unsafe {
                base.add(
                    warp_index
                        * warp_merge_sort_temp_storage_bytes::<
                            K,
                            V,
                            ITEMS_PER_THREAD,
                            LOGICAL_WARP_THREADS,
                        >(),
                )
            },
            _types: PhantomData,
        }
    }
}

/// A CUB-shaped logical-warp merge-sort specialization.
///
/// The generic parameter order mirrors CUB:
/// `Key`, `ITEMS_PER_THREAD`, `LOGICAL_WARP_THREADS`, then optional `Value`.
/// Use `V = ()` for keys-only sorting.
pub struct WarpMergeSort<
    K,
    const ITEMS_PER_THREAD: usize,
    const LOGICAL_WARP_THREADS: usize = 32,
    V = (),
> {
    storage: *mut u8,
    linear_tid: u32,
    member_mask: u32,
    _types: PhantomData<fn() -> (K, V)>,
}

impl<K, const ITEMS_PER_THREAD: usize, const LOGICAL_WARP_THREADS: usize, V>
    WarpMergeSort<K, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, V>
where
    K: DeviceCopy,
    V: DeviceCopy,
{
    const CONFIG_VALID: () = assert!(
        ITEMS_PER_THREAD != 0
            && LOGICAL_WARP_THREADS != 0
            && LOGICAL_WARP_THREADS <= 32
            && LOGICAL_WARP_THREADS.is_power_of_two(),
        "WarpMergeSort requires ITEMS_PER_THREAD > 0 and a power-of-two logical warp in 1..=32",
    );

    /// Bind a CUB-style specialization to one logical warp's temporary
    /// storage. All logical-warp members must construct the handle from the
    /// same storage view and call the same collective operation.
    #[inline(always)]
    pub fn new(
        storage: WarpMergeSortTempStorage<K, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, V>,
    ) -> Self {
        let () = Self::CONFIG_VALID;
        let lane = cuda_device::warp::lane_id();
        let logical = LOGICAL_WARP_THREADS as u32;
        let subwarp = lane / logical;
        let member_mask = logical_warp_mask(logical, subwarp);
        Self {
            storage: storage.pointer,
            linear_tid: lane % logical,
            member_mask,
            _types: PhantomData,
        }
    }

    /// Lane index within the logical warp.
    #[inline(always)]
    pub const fn get_linear_tid(&self) -> u32 {
        self.linear_tid
    }

    /// Physical-warp participation mask for this logical warp.
    #[inline(always)]
    pub const fn get_member_mask(&self) -> u32 {
        self.member_mask
    }

    #[inline(always)]
    unsafe fn sort_impl<C, const PAIRS: bool, const CLAMPED: bool, const HAS_OOB: bool>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        values: *mut V,
        comparator: C,
        valid_items: u32,
        oob_default: K,
    ) where
        C: DeviceComparator<K>,
    {
        let tile_items = (ITEMS_PER_THREAD * LOGICAL_WARP_THREADS) as u32;
        let valid_items = valid_items.min(tile_items);

        if CLAMPED {
            let thread_begin = self.linear_tid * ITEMS_PER_THREAD as u32;
            if thread_begin < valid_items {
                let mut maximum = keys[0];
                if HAS_OOB && unsafe { comparator.less(&maximum, &oob_default) } {
                    maximum = oob_default;
                }
                let mut item = 1usize;
                while item < ITEMS_PER_THREAD {
                    if thread_begin + (item as u32) < valid_items {
                        if unsafe { comparator.less(&maximum, &keys[item]) } {
                            maximum = keys[item];
                        }
                    } else {
                        keys[item] = maximum;
                    }
                    item += 1;
                }
                unsafe {
                    stable_odd_even_sort::<K, V, C, ITEMS_PER_THREAD, PAIRS>(
                        keys, values, comparator,
                    )
                };
            }
        } else {
            unsafe {
                stable_odd_even_sort::<K, V, C, ITEMS_PER_THREAD, PAIRS>(keys, values, comparator)
            };
        }

        unsafe { self.merge_rounds::<C, PAIRS, CLAMPED>(keys, values, comparator, valid_items) };
    }

    #[inline(always)]
    unsafe fn merge_rounds<C, const PAIRS: bool, const CLAMPED: bool>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        values: *mut V,
        comparator: C,
        valid_items: u32,
    ) where
        C: DeviceComparator<K>,
    {
        let keys_shared = self.storage as *mut K;
        let mut target_threads = 2u32;
        while target_threads <= LOGICAL_WARP_THREADS as u32 {
            let merged_threads = target_threads / 2;
            let mask = target_threads - 1;
            let run_size = ITEMS_PER_THREAD as u32 * merged_threads;

            if CLAMPED && run_size >= valid_items {
                break;
            }

            cuda_device::warp::sync_mask(self.member_mask);
            let mut item = 0usize;
            while item < ITEMS_PER_THREAD {
                unsafe {
                    keys_shared
                        .add((ITEMS_PER_THREAD as u32 * self.linear_tid + item as u32) as usize)
                        .write(keys[item])
                };
                item += 1;
            }
            if self.linear_tid == 0 {
                unsafe {
                    keys_shared
                        .add(ITEMS_PER_THREAD * LOGICAL_WARP_THREADS)
                        .write(keys[0])
                };
            }
            cuda_device::warp::sync_mask(self.member_mask);

            let first_thread = self.linear_tid & !mask;
            let start = ITEMS_PER_THREAD as u32 * first_thread;
            let thread_in_group = self.linear_tid & mask;

            let (keys1_begin, keys1_end, keys2_end, diagonal) = if CLAMPED {
                let keys1_begin = start.min(valid_items);
                let keys1_end = (keys1_begin + run_size).min(valid_items);
                let keys2_end = (keys1_end + run_size).min(valid_items);
                let diagonal = (keys2_end - keys1_begin)
                    .min(valid_items)
                    .min(ITEMS_PER_THREAD as u32 * thread_in_group);
                (keys1_begin, keys1_end, keys2_end, diagonal)
            } else {
                (
                    start,
                    start + run_size,
                    start + 2 * run_size,
                    ITEMS_PER_THREAD as u32 * thread_in_group,
                )
            };

            let keys2_begin = keys1_end;
            let keys1_count = keys1_end - keys1_begin;
            let keys2_count = keys2_end - keys2_begin;
            let partition = unsafe {
                merge_path(
                    keys_shared.add(keys1_begin as usize),
                    keys_shared.add(keys2_begin as usize),
                    keys1_count,
                    keys2_count,
                    diagonal,
                    comparator,
                )
            };
            let keys1_local = keys1_begin + partition;
            let keys2_local = keys2_begin + diagonal - partition;
            let mut indices = [0u32; ITEMS_PER_THREAD];
            unsafe {
                serial_merge(
                    keys_shared,
                    keys1_local,
                    keys2_local,
                    keys1_end - keys1_local,
                    keys2_end - keys2_local,
                    keys,
                    &mut indices,
                    comparator,
                )
            };

            if PAIRS {
                cuda_device::warp::sync_mask(self.member_mask);
                let items_shared = self.storage as *mut V;
                let mut value_item = 0usize;
                while value_item < ITEMS_PER_THREAD {
                    unsafe {
                        items_shared
                            .add(
                                (ITEMS_PER_THREAD as u32 * self.linear_tid + value_item as u32)
                                    as usize,
                            )
                            .write(values.add(value_item).read())
                    };
                    value_item += 1;
                }
                cuda_device::warp::sync_mask(self.member_mask);
                let mut output_item = 0usize;
                while output_item < ITEMS_PER_THREAD {
                    unsafe {
                        values
                            .add(output_item)
                            .write(items_shared.add(indices[output_item] as usize).read())
                    };
                    output_item += 1;
                }
            }

            target_threads *= 2;
        }
    }
}

impl<K, const ITEMS_PER_THREAD: usize, const LOGICAL_WARP_THREADS: usize, V> WarpMergeSortConfig
    for WarpMergeSort<K, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, V>
where
    K: DeviceCopy,
    V: DeviceCopy,
{
    type Key = K;
    type Value = V;

    const ITEMS_PER_THREAD: usize = ITEMS_PER_THREAD;
    const LOGICAL_WARP_THREADS: usize = LOGICAL_WARP_THREADS;
    const ITEMS_PER_TILE: usize = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS;
    const TEMP_STORAGE_BYTES: usize =
        warp_merge_sort_temp_storage_bytes::<K, V, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS>();
    const TEMP_STORAGE_ALIGN: usize = warp_merge_sort_temp_storage_align::<K, V>();
}

impl<K, const ITEMS_PER_THREAD: usize, const LOGICAL_WARP_THREADS: usize>
    WarpMergeSort<K, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, ()>
where
    K: DeviceCopy,
{
    /// Sort a full blocked logical-warp tile. CUB's `Sort` is currently stable
    /// in practice, but—as upstream documents—callers should only rely on the
    /// ordering guarantee of this method.
    ///
    /// # Safety
    ///
    /// Every lane in the logical warp must call this operation convergently
    /// with storage assigned to that warp and the same comparator state.
    #[inline(always)]
    pub unsafe fn sort<C>(&self, keys: &mut [K; ITEMS_PER_THREAD], comparator: C)
    where
        C: DeviceComparator<K>,
    {
        unsafe {
            self.sort_impl::<C, false, false, false>(
                keys,
                core::ptr::null_mut(),
                comparator,
                (ITEMS_PER_THREAD * LOGICAL_WARP_THREADS) as u32,
                keys[0],
            )
        };
    }

    /// Stable full-tile sort.
    ///
    /// # Safety
    ///
    /// Every lane in the logical warp must call this operation convergently
    /// with storage assigned to that warp and the same comparator state.
    #[inline(always)]
    pub unsafe fn stable_sort<C>(&self, keys: &mut [K; ITEMS_PER_THREAD], comparator: C)
    where
        C: DeviceComparator<K>,
    {
        unsafe { self.sort(keys, comparator) };
    }

    /// Sort the valid tile prefix with an out-of-bounds default, matching the
    /// corresponding CUB overload. Positions after `valid_items` are
    /// unspecified by the primitive.
    ///
    /// # Safety
    ///
    /// Every lane must call convergently with the same comparator,
    /// `valid_items`, and sentinel. `valid_items` must not exceed the tile size,
    /// and the temporary storage must belong to this logical warp.
    #[inline(always)]
    pub unsafe fn sort_partial<C>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        comparator: C,
        valid_items: u32,
        oob_default: K,
    ) where
        C: DeviceComparator<K>,
    {
        unsafe {
            self.sort_impl::<C, false, true, true>(
                keys,
                core::ptr::null_mut(),
                comparator,
                valid_items,
                oob_default,
            )
        };
    }

    /// Stable form of [`Self::sort_partial`].
    ///
    /// # Safety
    ///
    /// Every lane must call convergently with the same comparator,
    /// `valid_items`, and sentinel. `valid_items` must not exceed the tile size,
    /// and the temporary storage must belong to this logical warp.
    #[inline(always)]
    pub unsafe fn stable_sort_partial<C>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        comparator: C,
        valid_items: u32,
        oob_default: K,
    ) where
        C: DeviceComparator<K>,
    {
        unsafe { self.sort_partial(keys, comparator, valid_items, oob_default) };
    }

    /// Sort a valid prefix without requiring a sentinel. This maps to the CUB
    /// 3.6 `Sort(keys, compare_op, valid_items)` overload.
    ///
    /// # Safety
    ///
    /// Every lane must call convergently with the same comparator and
    /// `valid_items`. `valid_items` must not exceed the tile size, and the
    /// temporary storage must belong to this logical warp.
    #[inline(always)]
    pub unsafe fn sort_partial_without_sentinel<C>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        comparator: C,
        valid_items: u32,
    ) where
        C: DeviceComparator<K>,
    {
        unsafe {
            self.sort_impl::<C, false, true, false>(
                keys,
                core::ptr::null_mut(),
                comparator,
                valid_items,
                keys[0],
            )
        };
    }

    /// Stable form of [`Self::sort_partial_without_sentinel`].
    ///
    /// # Safety
    ///
    /// Every lane must call convergently with the same comparator and
    /// `valid_items`. `valid_items` must not exceed the tile size, and the
    /// temporary storage must belong to this logical warp.
    #[inline(always)]
    pub unsafe fn stable_sort_partial_without_sentinel<C>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        comparator: C,
        valid_items: u32,
    ) where
        C: DeviceComparator<K>,
    {
        unsafe { self.sort_partial_without_sentinel(keys, comparator, valid_items) };
    }
}

impl<K, const ITEMS_PER_THREAD: usize, const LOGICAL_WARP_THREADS: usize, V>
    WarpMergeSort<K, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, V>
where
    K: DeviceCopy,
    V: DeviceCopy,
{
    /// Sort a full blocked key/value tile.
    ///
    /// # Safety
    ///
    /// Every lane in the logical warp must call this operation convergently
    /// with storage assigned to that warp and the same comparator state.
    #[inline(always)]
    pub unsafe fn sort_pairs<C>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        values: &mut [V; ITEMS_PER_THREAD],
        comparator: C,
    ) where
        C: DeviceComparator<K>,
    {
        unsafe {
            self.sort_impl::<C, true, false, false>(
                keys,
                values.as_mut_ptr(),
                comparator,
                (ITEMS_PER_THREAD * LOGICAL_WARP_THREADS) as u32,
                keys[0],
            )
        };
    }

    /// Stable full-tile key/value sort.
    ///
    /// # Safety
    ///
    /// Every lane in the logical warp must call this operation convergently
    /// with storage assigned to that warp and the same comparator state.
    #[inline(always)]
    pub unsafe fn stable_sort_pairs<C>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        values: &mut [V; ITEMS_PER_THREAD],
        comparator: C,
    ) where
        C: DeviceComparator<K>,
    {
        unsafe { self.sort_pairs(keys, values, comparator) };
    }

    /// Sort the valid key/value prefix with an out-of-bounds key.
    ///
    /// # Safety
    ///
    /// Every lane must call convergently with the same comparator,
    /// `valid_items`, and sentinel. `valid_items` must not exceed the tile size,
    /// and the temporary storage must belong to this logical warp.
    #[inline(always)]
    pub unsafe fn sort_pairs_partial<C>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        values: &mut [V; ITEMS_PER_THREAD],
        comparator: C,
        valid_items: u32,
        oob_default: K,
    ) where
        C: DeviceComparator<K>,
    {
        unsafe {
            self.sort_impl::<C, true, true, true>(
                keys,
                values.as_mut_ptr(),
                comparator,
                valid_items,
                oob_default,
            )
        };
    }

    /// Stable form of [`Self::sort_pairs_partial`].
    ///
    /// # Safety
    ///
    /// Every lane must call convergently with the same comparator,
    /// `valid_items`, and sentinel. `valid_items` must not exceed the tile size,
    /// and the temporary storage must belong to this logical warp.
    #[inline(always)]
    pub unsafe fn stable_sort_pairs_partial<C>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        values: &mut [V; ITEMS_PER_THREAD],
        comparator: C,
        valid_items: u32,
        oob_default: K,
    ) where
        C: DeviceComparator<K>,
    {
        unsafe { self.sort_pairs_partial(keys, values, comparator, valid_items, oob_default) };
    }

    /// Sort a valid key/value prefix without requiring a sentinel.
    ///
    /// # Safety
    ///
    /// Every lane must call convergently with the same comparator and
    /// `valid_items`. `valid_items` must not exceed the tile size, and the
    /// temporary storage must belong to this logical warp.
    #[inline(always)]
    pub unsafe fn sort_pairs_partial_without_sentinel<C>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        values: &mut [V; ITEMS_PER_THREAD],
        comparator: C,
        valid_items: u32,
    ) where
        C: DeviceComparator<K>,
    {
        unsafe {
            self.sort_impl::<C, true, true, false>(
                keys,
                values.as_mut_ptr(),
                comparator,
                valid_items,
                keys[0],
            )
        };
    }

    /// Stable form of [`Self::sort_pairs_partial_without_sentinel`].
    ///
    /// # Safety
    ///
    /// Every lane must call convergently with the same comparator and
    /// `valid_items`. `valid_items` must not exceed the tile size, and the
    /// temporary storage must belong to this logical warp.
    #[inline(always)]
    pub unsafe fn stable_sort_pairs_partial_without_sentinel<C>(
        &self,
        keys: &mut [K; ITEMS_PER_THREAD],
        values: &mut [V; ITEMS_PER_THREAD],
        comparator: C,
        valid_items: u32,
    ) where
        C: DeviceComparator<K>,
    {
        unsafe { self.sort_pairs_partial_without_sentinel(keys, values, comparator, valid_items) };
    }
}

#[inline(always)]
unsafe fn stable_odd_even_sort<K, V, C, const ITEMS_PER_THREAD: usize, const PAIRS: bool>(
    keys: &mut [K; ITEMS_PER_THREAD],
    values: *mut V,
    comparator: C,
) where
    K: DeviceCopy,
    V: DeviceCopy,
    C: DeviceComparator<K>,
{
    let mut pass = 0usize;
    while pass < ITEMS_PER_THREAD {
        let mut item = pass & 1;
        while item + 1 < ITEMS_PER_THREAD {
            if unsafe { comparator.less(&keys[item + 1], &keys[item]) } {
                keys.swap(item, item + 1);
                if PAIRS {
                    unsafe { core::ptr::swap(values.add(item), values.add(item + 1)) };
                }
            }
            item += 2;
        }
        pass += 1;
    }
}

#[inline(always)]
unsafe fn merge_path<K, C>(
    keys1: *const K,
    keys2: *const K,
    keys1_count: u32,
    keys2_count: u32,
    diagonal: u32,
    comparator: C,
) -> u32
where
    K: DeviceCopy,
    C: DeviceComparator<K>,
{
    let mut begin = diagonal.saturating_sub(keys2_count);
    let mut end = diagonal.min(keys1_count);
    while begin < end {
        let middle = begin + (end - begin) / 2;
        let key1 = unsafe { keys1.add(middle as usize).read() };
        let key2 = unsafe { keys2.add((diagonal - 1 - middle) as usize).read() };
        if unsafe { comparator.less(&key2, &key1) } {
            end = middle;
        } else {
            begin = middle + 1;
        }
    }
    begin
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "device merge primitive passes register windows and shared storage explicitly"
)]
unsafe fn serial_merge<K, C, const ITEMS_PER_THREAD: usize>(
    shared: *const K,
    mut keys1_begin: u32,
    mut keys2_begin: u32,
    keys1_count: u32,
    keys2_count: u32,
    output: &mut [K; ITEMS_PER_THREAD],
    indices: &mut [u32; ITEMS_PER_THREAD],
    comparator: C,
) where
    K: DeviceCopy,
    C: DeviceComparator<K>,
{
    let keys1_end = keys1_begin + keys1_count;
    let keys2_end = keys2_begin + keys2_count;
    let mut key1 = if keys1_count != 0 {
        unsafe { shared.add(keys1_begin as usize).read() }
    } else {
        output[0]
    };
    let mut key2 = if keys2_count != 0 {
        unsafe { shared.add(keys2_begin as usize).read() }
    } else {
        output[0]
    };

    let mut item = 0usize;
    while item < ITEMS_PER_THREAD {
        let take_second = keys2_begin < keys2_end
            && (keys1_begin >= keys1_end || unsafe { comparator.less(&key2, &key1) });
        if take_second {
            output[item] = key2;
            indices[item] = keys2_begin;
            keys2_begin += 1;
            key2 = unsafe { shared.add(keys2_begin as usize).read() };
        } else {
            output[item] = key1;
            indices[item] = keys1_begin;
            keys1_begin += 1;
            key1 = unsafe { shared.add(keys1_begin as usize).read() };
        }
        item += 1;
    }
}

#[inline(always)]
const fn logical_warp_mask(logical_threads: u32, subwarp: u32) -> u32 {
    if logical_threads == 32 {
        u32::MAX
    } else {
        ((1u32 << logical_threads) - 1) << (subwarp * logical_threads)
    }
}

const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Required alignment of one logical warp's temporary storage.
pub const fn warp_merge_sort_temp_storage_align<K, V>() -> usize {
    let key = core::mem::align_of::<K>();
    let value = core::mem::align_of::<V>();
    if key > value { key } else { value }
}

/// Required byte stride of one logical warp's temporary storage.
///
/// This matches CUB's union of `Key[ITEMS_PER_TILE + 1]` and
/// `Value[ITEMS_PER_TILE + 1]`, including the padding element read by serial
/// merge and tail padding required by an array of `TempStorage` objects.
pub const fn warp_merge_sort_temp_storage_bytes<
    K,
    V,
    const ITEMS_PER_THREAD: usize,
    const LOGICAL_WARP_THREADS: usize,
>() -> usize {
    let elements = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS + 1;
    let key_bytes = core::mem::size_of::<K>() * elements;
    let value_bytes = core::mem::size_of::<V>() * elements;
    let union_bytes = if key_bytes > value_bytes {
        key_bytes
    } else {
        value_bytes
    };
    align_up(
        if union_bytes == 0 { 1 } else { union_bytes },
        warp_merge_sort_temp_storage_align::<K, V>(),
    )
}

/// Allocate exact CUB-style warp temporary storage in the caller's kernel and
/// return the typed view for one logical warp.
///
/// The macro is intentionally required: static shared-memory dimensions must
/// be concrete at the CUDA compilation call site, while Rust generic statics
/// cannot capture type or const parameters. The allocation remains explicit,
/// just like CUB's `__shared__ WarpMergeSortT::TempStorage storage[warps]`.
///
/// # Safety
///
/// `warp_index` must be less than `warps`; the kernel must launch no more
/// participating logical warps than the allocation declares; and every member
/// of a logical warp must use the same `warp_index` and collective call.
#[macro_export]
macro_rules! warp_merge_sort_temp_storage {
    (
        $storage:ident,
        key = $key:ty,
        value = $value:ty,
        items_per_thread = $items:expr,
        logical_warp_threads = $logical:expr,
        warps = $warps:expr,
        warp_index = $warp_index:expr $(,)?
    ) => {{
        const _: () = assert!(
            $items > 0
                && $logical > 0
                && $logical <= 32
                && $logical.is_power_of_two()
                && $warps > 0,
            "invalid WarpMergeSort temporary-storage geometry",
        );
        // cuda-oxide recognizes shared-memory intrinsics by their canonical
        // crate path. Routing this through an RCCL re-export compiles as
        // ordinary Rust, but produces an unresolved rccl-prefixed device
        // symbol once the macro is expanded in another crate.
        static mut $storage: ::cuda_device::SharedArray<
            u8,
            {
                $crate::warp_merge_sort_temp_storage_bytes::<
                    $key,
                    $value,
                    { $items },
                    { $logical },
                >() * ($warps as usize)
            },
            { $crate::warp_merge_sort_temp_storage_align::<$key, $value>() },
        > = ::cuda_device::SharedArray::UNINIT;
        let __rccl_storage_base =
            ::cuda_device::SharedArray::as_raw_mut_ptr(&raw mut $storage);
        $crate::WarpMergeSortTempStorage::<$key, { $items }, { $logical }, $value>::from_raw_warp(
            __rccl_storage_base,
            $warp_index as usize,
        )
    }};
    (
        $storage:ident,
        key = $key:ty,
        items_per_thread = $items:expr,
        logical_warp_threads = $logical:expr,
        warps = $warps:expr,
        warp_index = $warp_index:expr $(,)?
    ) => {
        $crate::warp_merge_sort_temp_storage!(
            $storage,
            key = $key,
            value = (),
            items_per_thread = $items,
            logical_warp_threads = $logical,
            warps = $warps,
            warp_index = $warp_index,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_layout_matches_cub_union() {
        assert_eq!(warp_merge_sort_temp_storage_bytes::<u8, (), 1, 4>(), 5);
        assert_eq!(warp_merge_sort_temp_storage_bytes::<i32, (), 4, 16>(), 260);
        assert_eq!(warp_merge_sort_temp_storage_bytes::<u8, u64, 7, 32>(), 1800);
        assert_eq!(warp_merge_sort_temp_storage_align::<u8, u64>(), 8);
    }

    #[test]
    fn logical_masks_match_cub_subwarps() {
        assert_eq!(logical_warp_mask(32, 0), u32::MAX);
        assert_eq!(logical_warp_mask(4, 0), 0x0000_000f);
        assert_eq!(logical_warp_mask(4, 3), 0x0000_f000);
        assert_eq!(logical_warp_mask(16, 1), 0xffff_0000);
    }
}
