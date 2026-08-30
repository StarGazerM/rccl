//! Pinned Thrust CUDA set-operation agent, expressed through Rust ownership.
//!
//! Policy selection and kernel construction stay in `rccl-cub-macros`; this
//! module is the safe device algorithm selected by those compile-time facts.

use core::{
    cmp::Ordering,
    mem::MaybeUninit,
    ops::{Add, Mul, Shr, Sub},
};

use cuda_device::{DisjointSlice, thread, warp};
use rccl_memory::device::{
    BlockSharedMemory, DeviceRegion, SharedRegion, set_lookback_initial_delay,
    set_lookback_retry_delay,
};

const SCAN_TILE_OOB: u64 = 0;
const SCAN_TILE_INVALID: u64 = 99;
const SCAN_TILE_PARTIAL: u64 = 100;
const SCAN_TILE_INCLUSIVE: u64 = 101;
const TILE_STATUS_PADDING: usize = 32;

/// Device-readable sorted range used after host/kernel argument lowering.
#[doc(hidden)]
pub trait SetReadRange<K>: Copy {
    fn len(self) -> usize;
    fn read(self, index: usize) -> K;
}

impl<K: Copy> SetReadRange<K> for &[K] {
    #[inline(always)]
    fn len(self) -> usize {
        <[K]>::len(self)
    }

    #[inline(always)]
    fn read(self, index: usize) -> K {
        self[index]
    }
}

/// Device-side value form of Rust's finite arithmetic range descriptor.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct ArithmeticSetRange<K> {
    start: K,
    step: K,
    len: usize,
}

impl<K> ArithmeticSetRange<K> {
    #[inline(always)]
    pub fn new(start: K, step: K, len: usize) -> Self {
        Self { start, step, len }
    }
}

impl<K> SetReadRange<K> for ArithmeticSetRange<K>
where
    K: Copy + TryFrom<usize> + Add<Output = K> + Mul<Output = K>,
{
    #[inline(always)]
    fn len(self) -> usize {
        self.len
    }

    #[inline(always)]
    fn read(self, index: usize) -> K {
        let Ok(index) = K::try_from(index) else {
            cuda_device::debug::trap();
        };
        self.start + self.step * index
    }
}

/// Signed offset word selected by Thrust's double-index dispatch.
#[doc(hidden)]
pub trait SetIndex:
    Copy + Ord + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Shr<u32, Output = Self>
{
    type Transaction: Copy;

    const ZERO: Self;
    const ONE: Self;

    fn from_usize(value: usize) -> Self;
    fn to_usize(self) -> usize;
    fn midpoint(begin: Self, end: Self) -> Self;
    fn biased_middle<const WIDE_SCALE: bool>(begin: Self, end: Self, shift: u32) -> Self;
    fn shuffle_up(value: Self, offset: u32) -> Self;
    fn shuffle_down(value: Self, offset: u32) -> Self;
    fn pack_pair(first: Self, second: Self) -> Self::Transaction;
    fn unpack_pair(word: Self::Transaction) -> (Self, Self);
    fn descriptor(status: u64, value: Self) -> Self::Transaction;
    fn descriptor_status(word: Self::Transaction) -> u64;
    fn descriptor_value(word: Self::Transaction) -> Self;
}

impl SetIndex for i32 {
    type Transaction = u64;

    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as i32
    }

    #[inline(always)]
    fn to_usize(self) -> usize {
        if self < 0 {
            cuda_device::debug::trap();
        }
        self as usize
    }

    #[inline(always)]
    fn midpoint(begin: Self, end: Self) -> Self {
        begin + ((end - begin) >> 1)
    }

    #[inline(always)]
    fn biased_middle<const WIDE_SCALE: bool>(begin: Self, end: Self, shift: u32) -> Self {
        if shift == 1 {
            return Self::midpoint(begin, end);
        }
        if WIDE_SCALE {
            let scale = (1_i64 << shift) - 1;
            ((begin as i64 + scale * end as i64) >> shift) as i32
        } else {
            let scale = (1_i32 << shift) - 1;
            (begin + scale * end) >> shift
        }
    }

    #[inline(always)]
    fn shuffle_up(value: Self, offset: u32) -> Self {
        warp::shuffle_up_sync(u32::MAX, value as u32, offset) as i32
    }

    #[inline(always)]
    fn shuffle_down(value: Self, offset: u32) -> Self {
        warp::shuffle_down_sync(u32::MAX, value as u32, offset) as i32
    }

    #[inline(always)]
    fn pack_pair(first: Self, second: Self) -> Self::Transaction {
        first as u32 as u64 | ((second as u32 as u64) << 32)
    }

    #[inline(always)]
    fn unpack_pair(word: Self::Transaction) -> (Self, Self) {
        (word as u32 as i32, (word >> 32) as u32 as i32)
    }

    #[inline(always)]
    fn descriptor(status: u64, value: Self) -> Self::Transaction {
        status as u32 as u64 | ((value as u32 as u64) << 32)
    }

    #[inline(always)]
    fn descriptor_status(word: Self::Transaction) -> u64 {
        word as u32 as u64
    }

    #[inline(always)]
    fn descriptor_value(word: Self::Transaction) -> Self {
        (word >> 32) as u32 as i32
    }
}

impl SetIndex for i64 {
    type Transaction = u128;

    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as i64
    }

    #[inline(always)]
    fn to_usize(self) -> usize {
        if self < 0 {
            cuda_device::debug::trap();
        }
        self as usize
    }

    #[inline(always)]
    fn midpoint(begin: Self, end: Self) -> Self {
        begin + ((end - begin) >> 1)
    }

    #[inline(always)]
    fn biased_middle<const WIDE_SCALE: bool>(begin: Self, end: Self, shift: u32) -> Self {
        if shift == 1 {
            return Self::midpoint(begin, end);
        }
        if WIDE_SCALE {
            // Pinned CCCL spells the level literal `4ll`, making the weighted
            // intermediate at least 64-bit. Express the same nonnegative
            // quotient without a potentially overflowing `scale * end` when
            // `Size` itself is already i64.
            let scale = (1_i64 << shift) - 1;
            end - ((end - begin + scale) >> shift)
        } else {
            let scale = (1_i64 << shift) - 1;
            (begin + scale * end) >> shift
        }
    }

    #[inline(always)]
    fn shuffle_up(value: Self, offset: u32) -> Self {
        warp::shuffle_up_u64_sync(u32::MAX, value as u64, offset) as i64
    }

    #[inline(always)]
    fn shuffle_down(value: Self, offset: u32) -> Self {
        warp::shuffle_down_u64_sync(u32::MAX, value as u64, offset) as i64
    }

    #[inline(always)]
    fn pack_pair(first: Self, second: Self) -> Self::Transaction {
        first as u64 as u128 | ((second as u64 as u128) << 64)
    }

    #[inline(always)]
    fn unpack_pair(word: Self::Transaction) -> (Self, Self) {
        (word as u64 as i64, (word >> 64) as u64 as i64)
    }

    #[inline(always)]
    fn descriptor(status: u64, value: Self) -> Self::Transaction {
        status as u128 | ((value as u64 as u128) << 64)
    }

    #[inline(always)]
    fn descriptor_status(word: Self::Transaction) -> u64 {
        word as u64
    }

    #[inline(always)]
    fn descriptor_value(word: Self::Transaction) -> Self {
        (word >> 64) as u64 as i64
    }
}

#[inline(always)]
fn ordered_before<K, F>(order: &F, left: &K, right: &K) -> bool
where
    F: Fn(&K, &K) -> Ordering + Copy,
{
    order(left, right) == Ordering::Less
}

#[inline(always)]
fn binary_search_iteration<const WIDE_SCALE: bool, I, K, F, R>(
    read: R,
    begin: &mut I,
    end: &mut I,
    key: K,
    shift: u32,
    upper_bound: bool,
    order: F,
) where
    I: SetIndex,
    K: Copy,
    F: Fn(&K, &K) -> Ordering + Copy,
    R: Fn(I) -> K + Copy,
{
    let middle = I::biased_middle::<WIDE_SCALE>(*begin, *end, shift);
    let probe = read(middle);
    let predicate = if upper_bound {
        !ordered_before(&order, &key, &probe)
    } else {
        ordered_before(&order, &probe, &key)
    };
    if predicate {
        *begin = middle + I::ONE;
    } else {
        *end = middle;
    }
}

#[inline(always)]
fn binary_search<I, K, F, R>(read: R, count: I, key: K, upper_bound: bool, order: F) -> I
where
    I: SetIndex,
    K: Copy,
    F: Fn(&K, &K) -> Ordering + Copy,
    R: Fn(I) -> K + Copy,
{
    let mut begin = I::ZERO;
    let mut end = count;
    while begin < end {
        binary_search_iteration::<false, _, _, _, _>(
            read,
            &mut begin,
            &mut end,
            key,
            1,
            upper_bound,
            order,
        );
    }
    begin
}

#[inline(always)]
fn biased_binary_search<const WIDE_SCALE: bool, I, K, F, R>(
    read: R,
    count: I,
    key: K,
    levels: i32,
    upper_bound: bool,
    order: F,
) -> I
where
    I: SetIndex,
    K: Copy,
    F: Fn(&K, &K) -> Ordering + Copy,
    R: Fn(I) -> K + Copy,
{
    let mut begin = I::ZERO;
    let mut end = count;
    if levels >= 4 && begin < end {
        binary_search_iteration::<WIDE_SCALE, _, _, _, _>(
            read,
            &mut begin,
            &mut end,
            key,
            9,
            upper_bound,
            order,
        );
    }
    if levels >= 3 && begin < end {
        binary_search_iteration::<WIDE_SCALE, _, _, _, _>(
            read,
            &mut begin,
            &mut end,
            key,
            7,
            upper_bound,
            order,
        );
    }
    if levels >= 2 && begin < end {
        binary_search_iteration::<WIDE_SCALE, _, _, _, _>(
            read,
            &mut begin,
            &mut end,
            key,
            5,
            upper_bound,
            order,
        );
    }
    if levels >= 1 && begin < end {
        binary_search_iteration::<WIDE_SCALE, _, _, _, _>(
            read,
            &mut begin,
            &mut end,
            key,
            4,
            upper_bound,
            order,
        );
    }
    while begin < end {
        binary_search_iteration::<WIDE_SCALE, _, _, _, _>(
            read,
            &mut begin,
            &mut end,
            key,
            1,
            upper_bound,
            order,
        );
    }
    begin
}

#[inline(always)]
fn merge_path<I, K, F, A, B>(
    read_first: A,
    first_count: I,
    read_second: B,
    second_count: I,
    diagonal: I,
    order: F,
) -> I
where
    I: SetIndex,
    K: Copy,
    F: Fn(&K, &K) -> Ordering + Copy,
    A: Fn(I) -> K + Copy,
    B: Fn(I) -> K + Copy,
{
    let mut begin = if diagonal > second_count {
        diagonal - second_count
    } else {
        I::ZERO
    };
    let mut end = diagonal.min(first_count);
    while begin < end {
        let middle = I::midpoint(begin, end);
        let first_key = read_first(middle);
        let second_key = read_second(diagonal - I::ONE - middle);
        if !ordered_before(&order, &second_key, &first_key) {
            begin = middle + I::ONE;
        } else {
            end = middle;
        }
    }
    begin
}

#[inline(always)]
fn balanced_path<const WIDE_SCALE: bool, I, K, F, A, B>(
    read_first: A,
    read_second: B,
    first_count: I,
    second_count: I,
    diagonal: I,
    levels: i32,
    order: F,
) -> (I, I)
where
    I: SetIndex,
    K: Copy,
    F: Fn(&K, &K) -> Ordering + Copy,
    A: Fn(I) -> K + Copy,
    B: Fn(I) -> K + Copy,
{
    let mut first_index = merge_path(
        read_first,
        first_count,
        read_second,
        second_count,
        diagonal,
        order,
    );
    let second_index = diagonal - first_index;
    let mut star = false;
    if second_index < second_count {
        let key = read_second(second_index);
        let first_start = biased_binary_search::<WIDE_SCALE, _, _, _, _>(
            read_first,
            first_index,
            key,
            levels,
            false,
            order,
        );
        let second_start = biased_binary_search::<WIDE_SCALE, _, _, _, _>(
            read_second,
            second_index,
            key,
            levels,
            false,
            order,
        );
        let first_run = first_index - first_start;
        let mut second_run = second_index - second_start;
        let total_run = first_run + second_run;
        let half = total_run >> 1;
        let mut second_advance = half.max(total_run - first_run);
        let second_end = second_count.min(second_start + second_advance + I::ONE);
        let second_run_end = second_index
            + binary_search(
                |offset| read_second(second_index + offset),
                second_end - second_index,
                key,
                true,
                order,
            );
        second_run = second_run_end - second_start;
        second_advance = second_advance.min(second_run);
        let first_advance = total_run - second_advance;
        if first_advance == second_advance + I::ONE && second_advance < second_run {
            star = true;
        }
        first_index = first_start + first_advance;
    }
    (
        first_index,
        diagonal - first_index + if star { I::ONE } else { I::ZERO },
    )
}

/// Exact balanced partition agent used before all four set operations.
#[doc(hidden)]
#[inline(always)]
pub fn set_partition_agent<I, K, F, R1, R2, const ITEMS_PER_TILE: usize>(
    first: R1,
    second: R2,
    order: F,
    mut partitions: DisjointSlice<I::Transaction>,
    launch: rccl_memory::device::OneDimensionalLaunch<'_>,
) where
    I: SetIndex,
    K: Copy,
    F: Fn(&K, &K) -> Ordering + Copy,
    R1: SetReadRange<K>,
    R2: SetReadRange<K>,
{
    let index = thread::index_1d_u32(launch);
    let raw = index.get() as usize;
    if let Some(mut output) = partitions.element_thread32(index) {
        let first_count = I::from_usize(first.len());
        let second_count = I::from_usize(second.len());
        let total = first_count + second_count;
        let diagonal = I::from_usize(raw.saturating_mul(ITEMS_PER_TILE)).min(total);
        let partition = balanced_path::<true, _, _, _, _, _>(
            |i| first.read(i.to_usize()),
            |i| second.read(i.to_usize()),
            first_count,
            second_count,
            diagonal,
            4,
            order,
        );
        output.write(I::pack_pair(partition.0, partition.1));
    }
}

/// Initialize `ScanTileState`'s 32 OOB descriptors and live invalid states.
#[doc(hidden)]
#[inline(always)]
pub fn set_init_agent<I: SetIndex>(tile_state: DisjointSlice<I::Transaction>, num_tiles: usize) {
    let tile_index = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
        + thread::threadIdx_x() as usize;
    let mut tile_state = DeviceRegion::new(tile_state);
    if tile_index < num_tiles {
        tile_state.write_volatile(
            TILE_STATUS_PADDING + tile_index,
            I::descriptor(SCAN_TILE_INVALID, I::ZERO),
        );
    }
    if thread::blockIdx_x() == 0 && tile_index < TILE_STATUS_PADDING {
        tile_state.write_volatile(tile_index, I::descriptor(SCAN_TILE_OOB, I::ZERO));
    }
}

/// Static Rust equivalents of CCCL's four serial set-operation functors.
#[doc(hidden)]
pub trait SerialSetOperation: Copy {
    const READS_SECOND_VALUES: bool;

    fn apply<K, F, const ITEMS: usize>(
        keys: &SharedRegion<MaybeUninit<K>>,
        first_begin: i32,
        second_begin: i32,
        first_count: i32,
        second_count: i32,
        sentinel: K,
        order: F,
    ) -> ([K; ITEMS], [i32; ITEMS], u32)
    where
        K: Copy,
        F: Fn(&K, &K) -> Ordering + Copy;
}

#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct SetIntersection;

#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct SetSymmetricDifference;

#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct SetDifference;

#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct SetUnion;

impl SerialSetOperation for SetIntersection {
    const READS_SECOND_VALUES: bool = false;

    #[inline(always)]
    #[allow(unused_assignments)]
    fn apply<K, F, const ITEMS: usize>(
        keys: &SharedRegion<MaybeUninit<K>>,
        first_begin: i32,
        second_begin: i32,
        first_count: i32,
        second_count: i32,
        sentinel: K,
        order: F,
    ) -> ([K; ITEMS], [i32; ITEMS], u32)
    where
        K: Copy,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        let mut output = [sentinel; ITEMS];
        let mut indices = [0i32; ITEMS];
        let mut active = 0u32;
        let mut first = first_begin;
        let mut second = second_begin;
        let first_end = first_begin + first_count;
        let second_end = second_begin + second_count;
        let mut first_key = keys.read_initialized(first as usize);
        let mut second_key = keys.read_initialized(second as usize);
        for_each_thread_item!(ITEMS, item, {
            let first_before = ordered_before(&order, &first_key, &second_key);
            let second_before = ordered_before(&order, &second_key, &first_key);
            output[item] = first_key;
            indices[item] = first;
            if first < first_end && second < second_end && first_before == second_before {
                active |= 1u32 << item;
            }
            if !second_before {
                first += 1;
                first_key = keys.read_initialized(first as usize);
            }
            if !first_before {
                second += 1;
                second_key = keys.read_initialized(second as usize);
            }
        });
        (output, indices, active)
    }
}

impl SerialSetOperation for SetSymmetricDifference {
    const READS_SECOND_VALUES: bool = true;

    #[inline(always)]
    #[allow(unused_assignments)]
    fn apply<K, F, const ITEMS: usize>(
        keys: &SharedRegion<MaybeUninit<K>>,
        first_begin: i32,
        second_begin: i32,
        first_count: i32,
        second_count: i32,
        sentinel: K,
        order: F,
    ) -> ([K; ITEMS], [i32; ITEMS], u32)
    where
        K: Copy,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        let mut output = [sentinel; ITEMS];
        let mut indices = [0i32; ITEMS];
        let mut active = 0u32;
        let mut first = first_begin;
        let mut second = second_begin;
        let first_end = first_begin + first_count;
        let second_end = second_begin + second_count;
        let end = first_end + second_end;
        let mut first_key = keys.read_initialized(first as usize);
        let mut second_key = keys.read_initialized(second as usize);
        for_each_thread_item!(ITEMS, item, {
            let mut second_selected = first >= first_end;
            let mut first_selected = !second_selected && second >= second_end;
            if !first_selected && !second_selected {
                first_selected = ordered_before(&order, &first_key, &second_key);
                second_selected =
                    !first_selected && ordered_before(&order, &second_key, &first_key);
            }
            output[item] = if first_selected {
                first_key
            } else {
                second_key
            };
            indices[item] = if first_selected { first } else { second };
            if first + second < end && first_selected != second_selected {
                active |= 1u32 << item;
            }
            if !second_selected {
                first += 1;
                first_key = keys.read_initialized(first as usize);
            }
            if !first_selected {
                second += 1;
                second_key = keys.read_initialized(second as usize);
            }
        });
        (output, indices, active)
    }
}

impl SerialSetOperation for SetDifference {
    // `set_difference_by_key` accepts and loads both value ranges even though
    // the serial selector can emit values only from the first range.
    const READS_SECOND_VALUES: bool = true;

    #[inline(always)]
    #[allow(unused_assignments)]
    fn apply<K, F, const ITEMS: usize>(
        keys: &SharedRegion<MaybeUninit<K>>,
        first_begin: i32,
        second_begin: i32,
        first_count: i32,
        second_count: i32,
        sentinel: K,
        order: F,
    ) -> ([K; ITEMS], [i32; ITEMS], u32)
    where
        K: Copy,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        let mut output = [sentinel; ITEMS];
        let mut indices = [0i32; ITEMS];
        let mut active = 0u32;
        let mut first = first_begin;
        let mut second = second_begin;
        let first_end = first_begin + first_count;
        let second_end = second_begin + second_count;
        let end = first_end + second_end;
        let mut first_key = keys.read_initialized(first as usize);
        let mut second_key = keys.read_initialized(second as usize);
        for_each_thread_item!(ITEMS, item, {
            let mut second_selected = first >= first_end;
            let mut first_selected = !second_selected && second >= second_end;
            if !first_selected && !second_selected {
                first_selected = ordered_before(&order, &first_key, &second_key);
                second_selected =
                    !first_selected && ordered_before(&order, &second_key, &first_key);
            }
            output[item] = first_key;
            indices[item] = first;
            if first + second < end && first_selected {
                active |= 1u32 << item;
            }
            if !second_selected {
                first += 1;
                first_key = keys.read_initialized(first as usize);
            }
            if !first_selected {
                second += 1;
                second_key = keys.read_initialized(second as usize);
            }
        });
        (output, indices, active)
    }
}

impl SerialSetOperation for SetUnion {
    const READS_SECOND_VALUES: bool = true;

    #[inline(always)]
    #[allow(unused_assignments)]
    fn apply<K, F, const ITEMS: usize>(
        keys: &SharedRegion<MaybeUninit<K>>,
        first_begin: i32,
        second_begin: i32,
        first_count: i32,
        second_count: i32,
        sentinel: K,
        order: F,
    ) -> ([K; ITEMS], [i32; ITEMS], u32)
    where
        K: Copy,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        let mut output = [sentinel; ITEMS];
        let mut indices = [0i32; ITEMS];
        let mut active = 0u32;
        let mut first = first_begin;
        let mut second = second_begin;
        let first_end = first_begin + first_count;
        let second_end = second_begin + second_count;
        let end = first_end + second_end;
        let mut first_key = keys.read_initialized(first as usize);
        let mut second_key = keys.read_initialized(second as usize);
        for_each_thread_item!(ITEMS, item, {
            let mut second_selected = first >= first_end;
            let mut first_selected = !second_selected && second >= second_end;
            if !first_selected && !second_selected {
                first_selected = ordered_before(&order, &first_key, &second_key);
                second_selected =
                    !first_selected && ordered_before(&order, &second_key, &first_key);
            }
            output[item] = if second_selected {
                second_key
            } else {
                first_key
            };
            indices[item] = if second_selected { second } else { first };
            if first + second < end {
                active |= 1u32 << item;
            }
            if !second_selected {
                first += 1;
                first_key = keys.read_initialized(first as usize);
            }
            if !first_selected {
                second += 1;
                second_key = keys.read_initialized(second as usize);
            }
        });
        (output, indices, active)
    }
}

#[inline(always)]
fn warp_exclusive_sum<I: SetIndex>(value: I, lane: u32) -> (I, I) {
    let mut inclusive = value;
    for_each_warp_scan_offset!(offset, {
        let preceding = I::shuffle_up(inclusive, offset);
        if lane >= offset {
            inclusive = preceding + inclusive;
        }
    });
    // Every lane named by the full warp mask must execute the `_sync`
    // intrinsic.  Select lane zero's identity only after the shuffle; placing
    // the shuffle in the nonzero-lane branch deadlocks at WARPSYNC.
    let preceding = I::shuffle_up(inclusive, 1);
    let exclusive = if lane == 0 { I::ZERO } else { preceding };
    (exclusive, inclusive)
}

#[inline(always)]
fn tail_segmented_sum<I: SetIndex>(value: I, inclusive_status: bool, lane: u32) -> I {
    let mut flags = warp::ballot_sync(u32::MAX, inclusive_status);
    flags &= warp::lanemask_ge();
    flags |= 1u32 << 31;
    let last_lane = flags.trailing_zeros();
    let mut output = value;
    macro_rules! step {
        ($offset:expr) => {{
            let other = I::shuffle_down(output, $offset);
            if lane + $offset <= last_lane {
                output = output + other;
            }
        }};
    }
    step!(1u32);
    step!(2u32);
    step!(4u32);
    step!(8u32);
    step!(16u32);
    output
}

#[inline(always)]
fn set_state<I: SetIndex>(
    tile_state: &mut DeviceRegion<'_, I::Transaction>,
    tile_index: I,
    status: u64,
    value: I,
) {
    tile_state.write_volatile(
        TILE_STATUS_PADDING + tile_index.to_usize(),
        I::descriptor(status, value),
    );
}

#[inline(always)]
fn wait_for_state<I: SetIndex, const SM70: bool>(
    tile_state: &mut DeviceRegion<'_, I::Transaction>,
    tile_index: I,
) -> (u64, I) {
    let physical = (tile_index + I::from_usize(TILE_STATUS_PADDING)).to_usize();
    let mut descriptor = tile_state.read_volatile(physical);
    while warp::any_sync(
        u32::MAX,
        I::descriptor_status(descriptor) == SCAN_TILE_INVALID,
    ) {
        set_lookback_retry_delay::<SM70>();
        descriptor = tile_state.read_volatile(physical);
    }
    (
        I::descriptor_status(descriptor),
        I::descriptor_value(descriptor),
    )
}

#[inline(always)]
fn tile_lookback<I: SetIndex, const SM70: bool>(
    tile_state: &mut DeviceRegion<'_, I::Transaction>,
    tile_index: I,
    block_aggregate: I,
    shared: &BlockSharedMemory,
) -> I {
    let thread_index = thread::threadIdx_x();
    let lane = warp::lane_id();
    let block_scan_bytes = if core::mem::size_of::<I>() == 4 {
        96usize
    } else {
        160usize
    };
    let callback_value_offset = block_scan_bytes + core::mem::align_of::<I>();
    if thread_index == 0 {
        let mut callback = shared.region::<I>(callback_value_offset, 3);
        callback.write(2, block_aggregate);
        set_state(tile_state, tile_index, SCAN_TILE_PARTIAL, block_aggregate);
    }

    let mut predecessor = tile_index - I::from_usize(thread_index as usize) - I::ONE;
    set_lookback_initial_delay::<SM70>();
    let (mut status, value) = wait_for_state::<I, SM70>(tile_state, predecessor);
    let mut exclusive = tail_segmented_sum(value, status == SCAN_TILE_INCLUSIVE, lane);
    while warp::all_sync(u32::MAX, status != SCAN_TILE_INCLUSIVE) {
        predecessor = predecessor - I::from_usize(32);
        let (next_status, next_value) = wait_for_state::<I, SM70>(tile_state, predecessor);
        status = next_status;
        let window = tail_segmented_sum(next_value, status == SCAN_TILE_INCLUSIVE, lane);
        exclusive = window + exclusive;
    }

    if thread_index == 0 {
        let inclusive = exclusive + block_aggregate;
        set_state(tile_state, tile_index, SCAN_TILE_INCLUSIVE, inclusive);
        let mut callback = shared.region::<I>(callback_value_offset, 3);
        callback.write(0, exclusive);
        callback.write(1, inclusive);
    }
    exclusive
}

#[inline(always)]
fn block_exclusive_sum<I: SetIndex, const THREADS: usize, const SM70: bool>(
    value: I,
    tile_index: I,
    tile_state: &mut DeviceRegion<'_, I::Transaction>,
    shared: &BlockSharedMemory,
) -> (I, I, I) {
    let thread_index = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_index = thread_index >> 5;
    let warps = THREADS / 32;
    let (warp_exclusive, warp_inclusive) = warp_exclusive_sum(value, lane);
    let mut aggregates = shared.region::<I>(0, warps);
    if lane == 31 {
        aggregates.write(warp_index as usize, warp_inclusive);
    }
    thread::sync_threads();

    let mut warp_prefix = I::ZERO;
    let mut block_aggregate = aggregates.read(0);
    for_each_set_warp!(warp_number, {
        if warp_number < warps {
            if warp_index as usize == warp_number {
                warp_prefix = block_aggregate;
            }
            block_aggregate = block_aggregate + aggregates.read(warp_number);
        }
    });
    let mut local_exclusive = warp_exclusive;
    if warp_index != 0 {
        local_exclusive = warp_prefix + local_exclusive;
    }

    let block_prefix_offset = warps * core::mem::size_of::<I>() + warps;
    let block_prefix_offset = block_prefix_offset.next_multiple_of(core::mem::align_of::<I>());
    let mut block_prefix_storage = shared.region::<I>(block_prefix_offset, 1);
    let mut tile_prefix = I::ZERO;
    if tile_index == I::ZERO {
        if thread_index == 0 {
            local_exclusive = I::ZERO;
        }
    } else {
        if warp_index == 0 {
            let prefix = tile_lookback::<I, SM70>(tile_state, tile_index, block_aggregate, shared);
            if lane == 0 {
                block_prefix_storage.write(0, prefix);
                local_exclusive = prefix;
            }
        }
        thread::sync_threads();
        tile_prefix = block_prefix_storage.read(0);
        if thread_index > 0 {
            local_exclusive = tile_prefix + local_exclusive;
        }
    }
    (local_exclusive, tile_prefix, block_aggregate)
}

struct TileResult<K, I, const ITEMS: usize> {
    keys: [K; ITEMS],
    indices: [i32; ITEMS],
    active: u32,
    thread_prefix: I,
    tile_prefix: I,
    tile_count: I,
    first_begin: usize,
    second_begin: usize,
    first_count: usize,
    second_count: usize,
}

trait SetOutput<T> {
    fn write(&mut self, index: usize, value: T);
}

impl<T> SetOutput<T> for DeviceRegion<'_, T> {
    #[inline(always)]
    fn write(&mut self, index: usize, value: T) {
        DeviceRegion::write(self, index, value);
    }
}

struct DiscardOutput;

impl<T> SetOutput<T> for DiscardOutput {
    #[inline(always)]
    fn write(&mut self, _index: usize, _value: T) {}
}

#[inline(always)]
fn scatter<T, I, O, const THREADS: usize, const ITEMS: usize>(
    output: &mut O,
    values: &[T; ITEMS],
    active: u32,
    thread_prefix: I,
    tile_prefix: I,
    tile_count: I,
    shared: &BlockSharedMemory,
    shared_offset: usize,
) where
    T: Copy,
    I: SetIndex,
    O: SetOutput<T>,
{
    let mut shared_values = shared.region::<MaybeUninit<T>>(
        shared_offset,
        shared.bytes().saturating_sub(shared_offset) / core::mem::size_of::<T>(),
    );
    let mut local = (thread_prefix - tile_prefix).to_usize();
    for_each_thread_item!(ITEMS, item, {
        if active & (1u32 << item) != 0 {
            shared_values.write(local, MaybeUninit::new(values[item]));
            local += 1;
        }
    });
    let _ = local;
    thread::sync_threads();
    let mut item = thread::threadIdx_x() as usize;
    while item < tile_count.to_usize() {
        output.write(
            tile_prefix.to_usize() + item,
            shared_values.read_initialized(item),
        );
        item += THREADS;
    }
}

#[inline(always)]
fn scatter_initialized<T, I, const THREADS: usize, const ITEMS: usize>(
    output: &mut DeviceRegion<'_, T>,
    values: &[MaybeUninit<T>; ITEMS],
    active: u32,
    thread_prefix: I,
    tile_prefix: I,
    tile_count: I,
    shared: &BlockSharedMemory,
    shared_offset: usize,
) where
    T: Copy,
    I: SetIndex,
{
    let mut shared_values = shared.region::<MaybeUninit<T>>(
        shared_offset,
        shared.bytes().saturating_sub(shared_offset) / core::mem::size_of::<T>(),
    );
    let mut local = (thread_prefix - tile_prefix).to_usize();
    for_each_thread_item!(ITEMS, item, {
        if active & (1u32 << item) != 0 {
            // The serial set operation marks a slot active only when its
            // selected source element was loaded in the value phase.
            shared_values.write(
                local,
                MaybeUninit::new(rccl_memory::device::initialized_value(&values[item])),
            );
            local += 1;
        }
    });
    let _ = local;
    thread::sync_threads();
    let mut item = thread::threadIdx_x() as usize;
    while item < tile_count.to_usize() {
        output.write(
            tile_prefix.to_usize() + item,
            shared_values.read_initialized(item),
        );
        item += THREADS;
    }
}

#[inline(always)]
fn consume_key_tile<
    K,
    I,
    F,
    Op,
    R1,
    R2,
    O,
    const THREADS: usize,
    const ITEMS: usize,
    const SM70: bool,
    const LAST: bool,
>(
    first: R1,
    second: R2,
    partitions: &[I::Transaction],
    order: F,
    tile_state: &mut DeviceRegion<'_, I::Transaction>,
    output: &mut O,
    shared: &BlockSharedMemory,
) -> TileResult<K, I, ITEMS>
where
    K: Copy,
    I: SetIndex,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: SerialSetOperation,
    R1: SetReadRange<K>,
    R2: SetReadRange<K>,
    O: SetOutput<K>,
{
    let tile_index = thread::blockIdx_x() as usize;
    let (first_begin_i, second_begin_i) = I::unpack_pair(partitions[tile_index]);
    let (first_end_i, second_end_i) = I::unpack_pair(partitions[tile_index + 1]);
    let first_begin = first_begin_i.to_usize();
    let second_begin = second_begin_i.to_usize();
    let first_count = (first_end_i - first_begin_i).to_usize();
    let second_count = (second_end_i - second_begin_i).to_usize();
    let total = first_count + second_count;
    let sentinel = if first_count != 0 {
        first.read(first_begin)
    } else {
        second.read(second_begin)
    };
    let mut local = [sentinel; ITEMS];
    let thread_index = thread::threadIdx_x() as usize;
    for_each_thread_item!(ITEMS, item, {
        let index = THREADS * item + thread_index;
        let should_load = if LAST {
            index < total
        } else {
            item + 1 < ITEMS || index < total
        };
        if should_load {
            local[item] = if index < first_count {
                first.read(first_begin + index)
            } else {
                second.read(second_begin + index - first_count)
            };
        }
    });

    let union_offset = THREADS * core::mem::size_of::<i32>();
    let shared_len = THREADS * ITEMS - 1 + THREADS;
    let mut shared_keys = shared.region::<MaybeUninit<K>>(union_offset, shared_len);
    for_each_thread_item!(ITEMS, item, {
        shared_keys.write(THREADS * item + thread_index, MaybeUninit::new(local[item]));
    });
    // Rust cannot read C++'s speculative uninitialized tail. The register
    // transpose initializes `THREADS * ITEMS` slots; CCCL overallocates
    // another `THREADS - 1`. A duplicate-run star partition can make the
    // final register slot valid, so only threads 0..THREADS-1 initialize the
    // true tail and the final thread deliberately performs no write.
    if thread_index + 1 < THREADS {
        shared_keys.write(THREADS * ITEMS + thread_index, MaybeUninit::new(sentinel));
    }
    thread::sync_threads();

    let local_diagonal = (ITEMS * thread_index).min(total) as i32;
    let partition = balanced_path::<false, _, _, _, _, _>(
        |i: i32| shared_keys.read_initialized(i as usize),
        |i: i32| shared_keys.read_initialized(first_count + i as usize),
        first_count as i32,
        second_count as i32,
        local_diagonal,
        4,
        order,
    );
    let packed = if thread_index == 0 {
        ((first_count as i32) << 16) | second_count as i32
    } else {
        (partition.0 << 16) | partition.1
    };
    let destination = if thread_index == 0 {
        THREADS - 1
    } else {
        thread_index - 1
    };
    let mut offsets = shared.region::<i32>(0, THREADS);
    offsets.write(destination, packed);
    thread::sync_threads();
    let next = offsets.read(thread_index);
    let first_local_end = next >> 16;
    let second_local_end = next & 0xffff;
    let (keys, indices, active) = Op::apply(
        &shared_keys,
        partition.0,
        partition.1 + first_count as i32,
        first_local_end - partition.0,
        second_local_end - partition.1,
        sentinel,
        order,
    );
    thread::sync_threads();

    let thread_count = I::from_usize(active.count_ones() as usize);
    let (thread_prefix, tile_prefix, tile_count) = block_exclusive_sum::<I, THREADS, SM70>(
        thread_count,
        I::from_usize(tile_index),
        tile_state,
        shared,
    );
    if tile_index == 0 && thread_index == 0 && !LAST {
        set_state(tile_state, I::ZERO, SCAN_TILE_INCLUSIVE, tile_count);
    }
    thread::sync_threads();
    scatter::<K, I, O, THREADS, ITEMS>(
        output,
        &keys,
        active,
        thread_prefix,
        tile_prefix,
        tile_count,
        shared,
        union_offset,
    );
    TileResult {
        keys,
        indices,
        active,
        thread_prefix,
        tile_prefix,
        tile_count,
        first_begin,
        second_begin,
        first_count,
        second_count,
    }
}

#[inline(always)]
fn run_set_keys_agent<
    K,
    I,
    F,
    Op,
    R1,
    R2,
    O,
    const THREADS: usize,
    const ITEMS: usize,
    const SM70: bool,
>(
    first: R1,
    second: R2,
    partitions: &[I::Transaction],
    order: F,
    tile_state: DisjointSlice<I::Transaction>,
    output_count: DisjointSlice<usize>,
    output: &mut O,
    shared: BlockSharedMemory,
) where
    K: Copy,
    I: SetIndex,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: SerialSetOperation,
    R1: SetReadRange<K>,
    R2: SetReadRange<K>,
    O: SetOutput<K>,
{
    let tile_index = thread::blockIdx_x() as usize;
    let last = tile_index + 1 == thread::gridDim_x() as usize;
    let mut tile_state = DeviceRegion::new(tile_state);
    let result = if last {
        consume_key_tile::<K, I, F, Op, R1, R2, O, THREADS, ITEMS, SM70, true>(
            first,
            second,
            partitions,
            order,
            &mut tile_state,
            output,
            &shared,
        )
    } else {
        consume_key_tile::<K, I, F, Op, R1, R2, O, THREADS, ITEMS, SM70, false>(
            first,
            second,
            partitions,
            order,
            &mut tile_state,
            output,
            &shared,
        )
    };
    if last && thread::threadIdx_x() == 0 {
        let mut output_count = DeviceRegion::new(output_count);
        output_count.write(0, (result.tile_prefix + result.tile_count).to_usize());
    }
}

/// Key-only set-operation block agent.
#[doc(hidden)]
#[inline(always)]
pub fn set_keys_agent<K, I, F, Op, const THREADS: usize, const ITEMS: usize, const SM70: bool>(
    first: &[K],
    second: &[K],
    partitions: &[I::Transaction],
    order: F,
    tile_state: DisjointSlice<I::Transaction>,
    output_count: DisjointSlice<usize>,
    output: DisjointSlice<K>,
    shared: BlockSharedMemory,
) where
    K: Copy,
    I: SetIndex,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: SerialSetOperation,
{
    let mut output = DeviceRegion::new(output);
    run_set_keys_agent::<K, I, F, Op, _, _, _, THREADS, ITEMS, SM70>(
        first,
        second,
        partitions,
        order,
        tile_state,
        output_count,
        &mut output,
        shared,
    );
}

/// Set-operation agent over two finite arithmetic input ranges.
#[doc(hidden)]
#[inline(always)]
pub fn set_arithmetic_keys_agent<
    K,
    I,
    F,
    Op,
    const THREADS: usize,
    const ITEMS: usize,
    const SM70: bool,
>(
    first: ArithmeticSetRange<K>,
    second: ArithmeticSetRange<K>,
    partitions: &[I::Transaction],
    order: F,
    tile_state: DisjointSlice<I::Transaction>,
    output_count: DisjointSlice<usize>,
    output: DisjointSlice<K>,
    shared: BlockSharedMemory,
) where
    K: Copy + TryFrom<usize> + Add<Output = K> + Mul<Output = K>,
    I: SetIndex,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: SerialSetOperation,
{
    let mut output = DeviceRegion::new(output);
    run_set_keys_agent::<K, I, F, Op, _, _, _, THREADS, ITEMS, SM70>(
        first,
        second,
        partitions,
        order,
        tile_state,
        output_count,
        &mut output,
        shared,
    );
}

/// Key-only set operation whose output is a Rust discard range.
#[doc(hidden)]
#[inline(always)]
pub fn set_keys_discard_agent<
    K,
    I,
    F,
    Op,
    const THREADS: usize,
    const ITEMS: usize,
    const SM70: bool,
>(
    first: &[K],
    second: &[K],
    partitions: &[I::Transaction],
    order: F,
    tile_state: DisjointSlice<I::Transaction>,
    output_count: DisjointSlice<usize>,
    shared: BlockSharedMemory,
) where
    K: Copy,
    I: SetIndex,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: SerialSetOperation,
{
    let mut output = DiscardOutput;
    run_set_keys_agent::<K, I, F, Op, _, _, _, THREADS, ITEMS, SM70>(
        first,
        second,
        partitions,
        order,
        tile_state,
        output_count,
        &mut output,
        shared,
    );
}

/// Key/value set-operation block agent.
#[doc(hidden)]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn set_pairs_agent<K, V, I, F, Op, const THREADS: usize, const ITEMS: usize, const SM70: bool>(
    first_keys: &[K],
    second_keys: &[K],
    first_values: &[V],
    second_values: &[V],
    partitions: &[I::Transaction],
    order: F,
    tile_state: DisjointSlice<I::Transaction>,
    output_count: DisjointSlice<usize>,
    output_keys: DisjointSlice<K>,
    output_values: DisjointSlice<V>,
    shared: BlockSharedMemory,
) where
    K: Copy,
    V: Copy,
    I: SetIndex,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: SerialSetOperation,
{
    let tile_index = thread::blockIdx_x() as usize;
    let last = tile_index + 1 == thread::gridDim_x() as usize;
    let mut tile_state = DeviceRegion::new(tile_state);
    let mut output_keys = DeviceRegion::new(output_keys);
    let result = if last {
        consume_key_tile::<K, I, F, Op, _, _, _, THREADS, ITEMS, SM70, true>(
            first_keys,
            second_keys,
            partitions,
            order,
            &mut tile_state,
            &mut output_keys,
            &shared,
        )
    } else {
        consume_key_tile::<K, I, F, Op, _, _, _, THREADS, ITEMS, SM70, false>(
            first_keys,
            second_keys,
            partitions,
            order,
            &mut tile_state,
            &mut output_keys,
            &shared,
        )
    };

    // CCCL always executes the value phase, including tiles that produce no
    // output. `MaybeUninit` is the Rust representation of its ignored
    // register/shared slots: active bits prove initialization before gather
    // and scatter, without inventing a value or skipping the phase.
    let mut local = [MaybeUninit::<V>::uninit(); ITEMS];
    let thread_index = thread::threadIdx_x() as usize;
    for_each_thread_item!(ITEMS, item, {
        let index = THREADS * item + thread_index;
        let total = result.first_count + result.second_count;
        let should_load = if last {
            index < total
        } else {
            item + 1 < ITEMS || index < total
        };
        if should_load {
            if index < result.first_count {
                local[item] = MaybeUninit::new(first_values[result.first_begin + index]);
            } else if Op::READS_SECOND_VALUES {
                local[item] = MaybeUninit::new(
                    second_values[result.second_begin + index - result.first_count],
                );
            }
        }
    });
    thread::sync_threads();
    let union_offset = THREADS * core::mem::size_of::<i32>();
    let shared_len = THREADS * ITEMS - 1 + THREADS;
    let mut shared_values = shared.region::<MaybeUninit<V>>(union_offset, shared_len);
    for_each_thread_item!(ITEMS, item, {
        shared_values.write(THREADS * item + thread_index, local[item]);
    });
    if thread_index + 1 < THREADS {
        shared_values.write(THREADS * ITEMS + thread_index, MaybeUninit::uninit());
    }
    thread::sync_threads();
    for_each_thread_item!(ITEMS, item, {
        if result.active & (1u32 << item) != 0 {
            local[item] =
                MaybeUninit::new(shared_values.read_initialized(result.indices[item] as usize));
        }
    });
    thread::sync_threads();
    let mut output_values = DeviceRegion::new(output_values);
    scatter_initialized::<V, I, THREADS, ITEMS>(
        &mut output_values,
        &local,
        result.active,
        result.thread_prefix,
        result.tile_prefix,
        result.tile_count,
        &shared,
        union_offset,
    );

    if last && thread::threadIdx_x() == 0 {
        let mut output_count = DeviceRegion::new(output_count);
        output_count.write(0, (result.tile_prefix + result.tile_count).to_usize());
    }
    let _ = result.keys;
}
