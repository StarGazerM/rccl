//! Audited CUDA ownership boundary for block-wide ordering collectives.
//!
//! cuda-oxide intentionally exposes shared memory and arbitrary scatter
//! stores through raw pointers: its sealed `SpaceLayout` vocabulary cannot
//! describe CUB's phase-changing ownership (blocked registers, shared merge
//! tile, then warp-striped stores).  Generated RCCL code must not reproduce
//! that escape hatch.  It calls the safe functions in this module instead.
//!
//! The safety proof is local to each collective:
//! - every block owns one output tile and distinct blocks own disjoint tiles;
//! - every shared-memory phase is separated by the same CUB barrier;
//! - uninitialized partial-tile lanes stay `MaybeUninit<K>` until a range
//!   guard proves that the corresponding output is valid;
//! - the warp-transpose store formula is injective for the 32 lanes of each
//!   warp and disjoint between warps.

use core::mem::MaybeUninit;

use cuda_device::{DisjointSlice, LinearTiles, thread, warp};
use rccl_memory::device::{
    BlockSharedMemory, DeviceRegion, MergePathSearch, OneDimensionalLaunch, SharedRegion,
    WarpStripedBlockInput, WarpStripedBlockRead, WarpStripedBlockTile, WarpStripedBlockWrite,
    commit_and_wait_async_copy, commit_async_copy, initialized_ref, initialized_value,
    threadfence_block, wait_async_copy, warp_striped_pair_into_write,
};

// CUB marks each per-thread item loop with `_CCCL_PRAGMA_UNROLL_FULL()`. A
// const-generic Rust loop is not enough: LLVM is permitted to leave it rolled,
// which changes both instruction shape and throughput. The proc macro selects
// `ITEMS`; this shallow declarative macro materializes those compile-time
// repetitions as ordinary Rust statements. Ordering policies currently cap at
// 21 items per thread, and the extra entries leave room for subsequent pinned
// policies while failing closed at the call-site assertion below.
macro_rules! for_each_thread_item {
    ($count:expr, $item:ident, $body:block) => {{
        macro_rules! one {
            ($index:expr) => {
                if $index < $count {
                    let $item = $index;
                    $body
                }
            };
        }
        one!(0usize);
        one!(1usize);
        one!(2usize);
        one!(3usize);
        one!(4usize);
        one!(5usize);
        one!(6usize);
        one!(7usize);
        one!(8usize);
        one!(9usize);
        one!(10usize);
        one!(11usize);
        one!(12usize);
        one!(13usize);
        one!(14usize);
        one!(15usize);
        one!(16usize);
        one!(17usize);
        one!(18usize);
        one!(19usize);
        one!(20usize);
        one!(21usize);
        one!(22usize);
        one!(23usize);
    }};
}

// Policy800 has twelve warps and each warp owns eight histogram bins per
// lane. CUB fully unrolls both fixed-extent loops; materialize the same Rust
// statements so LLVM cannot retain loop-control instructions in the hot
// one-sweep rank path.
macro_rules! for_each_policy800_warp {
    ($warp:ident, $body:block) => {{
        macro_rules! one {
            ($index:expr) => {{
                let $warp = $index;
                $body
            }};
        }
        one!(0usize);
        one!(1usize);
        one!(2usize);
        one!(3usize);
        one!(4usize);
        one!(5usize);
        one!(6usize);
        one!(7usize);
        one!(8usize);
        one!(9usize);
        one!(10usize);
        one!(11usize);
    }};
}

macro_rules! for_each_radix_warp_bin {
    ($slot:ident, $body:block) => {{
        macro_rules! one {
            ($index:expr) => {{
                let $slot = $index;
                $body
            }};
        }
        one!(0u32);
        one!(1u32);
        one!(2u32);
        one!(3u32);
        one!(4u32);
        one!(5u32);
        one!(6u32);
        one!(7u32);
    }};
}

macro_rules! for_each_raking_item {
    ($count:expr, $item:ident, $body:block) => {{
        macro_rules! one {
            ($index:expr) => {
                if $index < $count {
                    let $item = $index;
                    $body
                }
            };
        }
        one!(0usize);
        one!(1usize);
        one!(2usize);
        one!(3usize);
        one!(4usize);
        one!(5usize);
        one!(6usize);
        one!(7usize);
        one!(8usize);
        one!(9usize);
        one!(10usize);
        one!(11usize);
    }};
}

/// CUB `BlockExchange` inserts one item after every 32 logical items when a
/// thread owns a power-of-two number of items greater than four.  `ITEMS` is a
/// proc-macro-selected policy constant, so this branch disappears in each
/// kernel instantiation.
#[inline(always)]
const fn block_exchange_index<const ITEMS: usize>(index: u32) -> usize {
    if ITEMS > 4 && ITEMS.is_power_of_two() {
        (index + (index >> 5)) as usize
    } else {
        index as usize
    }
}

#[inline(always)]
const fn block_exchange_storage_items<const THREADS: usize, const ITEMS: usize>() -> usize {
    let tile_items = THREADS * ITEMS;
    if ITEMS > 4 && ITEMS.is_power_of_two() {
        tile_items + tile_items / 32
    } else {
        tile_items
    }
}

#[inline(always)]
const fn load_to_shared_buffer_bytes(items: u32, item_size: u32) -> u32 {
    let bytes = items * item_size;
    if item_size >= 16 || bytes == 0 {
        bytes
    } else {
        bytes.div_ceil(16) * 16 + 16
    }
}

#[inline(always)]
fn ordered_before<K, F>(order: &F, left: &K, right: &K) -> bool
where
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    order(left, right).is_lt()
}

/// Exact CUB `MergePath` over two complete global input ranges.
///
/// The range capability maintains CUB's binary-search interval invariant while
/// this function supplies the algorithm's stable comparison decision.
#[inline(always)]
pub fn global_merge_path<K, F>(first: &[K], second: &[K], diagonal: usize, order: &F) -> usize
where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let mut search = MergePathSearch::new(first, second, diagonal);
    while search.step(|first_key, second_key| ordered_before(order, second_key, first_key)) {}
    search.partition()
}

/// CUB `MergePath` over the two adjacent ranges already staged in shared
/// memory through the checked block-shared capability.
#[inline(always)]
fn local_merge_path_begin(diagonal: i32, second_count: i32) -> i32 {
    (diagonal - second_count).max(0)
}

#[inline(always)]
fn shared_merge_path_region<K, F>(
    shared: &SharedRegion<MaybeUninit<K>>,
    first_offset: i32,
    second_offset: i32,
    first_count: i32,
    second_count: i32,
    diagonal: i32,
    order: &F,
) -> i32
where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    // CUB's signed local MergePath coordinate uses
    // `max(diagonal - second_count, 0)`. Rust's signed `saturating_sub` would
    // instead retain a negative result, so spell out the upstream clamp.
    let mut first_begin = local_merge_path_begin(diagonal, second_count);
    let mut first_end = diagonal.min(first_count);
    while first_begin < first_end {
        let middle = first_begin + (first_end - first_begin) / 2;
        let first = shared.read_initialized((first_offset + middle) as usize);
        let second_index = first_offset + second_offset + diagonal - 1 - middle;
        let second = shared.read_initialized(second_index as usize);
        if ordered_before(order, &second, &first) {
            first_end = middle;
        } else {
            first_begin = middle + 1;
        }
    }
    first_begin
}

/// Exact `AgentMerge::consume_tile` key path, including the SM80 asynchronous
/// load-to-shared policy selected by the proc macro.
///
/// This function is safe to generated code. Its launch is branded as one
/// 512-thread block per merge tile, `partitions` contains the preceding
/// partition-kernel result, and `output` exclusively borrows the destination.
#[inline(always)]
#[allow(unused_assignments)] // CUB deliberately prefetches after the last serial-merge item.
pub fn merge_keys_agent<
    K,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const K_SIZE: usize,
    const USE_LOAD_TO_SHARED: bool,
    const SHARED_BYTES: usize,
>(
    first: &[K],
    second: &[K],
    partitions: &[usize],
    order: F,
    output: DisjointSlice<K, LinearTiles<ITEMS>>,
    launch: OneDimensionalLaunch<'_>,
) where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let tile_index = thread::blockIdx_x() as usize;
    let tile_items = THREADS * ITEMS;
    let tile_base = tile_index * tile_items;
    let total = first.len() + second.len();
    let items_in_tile = (total - tile_base).min(tile_items);
    if items_in_tile == tile_items {
        let output = WarpStripedBlockWrite::<K, THREADS, ITEMS>::new(output, launch, tile_items);
        merge_keys_tile::<K, F, THREADS, ITEMS, K_SIZE, USE_LOAD_TO_SHARED, SHARED_BYTES, true>(
            first,
            second,
            partitions,
            order,
            (tile_index, tile_base, tile_items),
            output,
        );
    } else {
        let output = WarpStripedBlockWrite::<K, THREADS, ITEMS>::new(output, launch, items_in_tile);
        merge_keys_tile::<K, F, THREADS, ITEMS, K_SIZE, USE_LOAD_TO_SHARED, SHARED_BYTES, false>(
            first,
            second,
            partitions,
            order,
            (tile_index, tile_base, items_in_tile),
            output,
        );
    }
}

#[inline(always)]
#[allow(unused_assignments)]
fn merge_keys_tile<
    K,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const K_SIZE: usize,
    const USE_LOAD_TO_SHARED: bool,
    const SHARED_BYTES: usize,
    const FULL: bool,
>(
    first: &[K],
    second: &[K],
    partitions: &[usize],
    order: F,
    tile_position: (usize, usize, usize),
    mut output: WarpStripedBlockWrite<'_, K, THREADS, ITEMS>,
) where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let (tile_index, tile_base, items_in_tile) = tile_position;
    let thread_index = thread::threadIdx_x();
    let tile_items = THREADS * ITEMS;
    let total = first.len() + second.len();

    let diagonal_first = tile_base;
    let diagonal_last = if FULL { tile_base + tile_items } else { total };
    // SAFETY: the caller-owned workspace contract requires exactly one more
    // partition than launched tiles.
    let first_begin = partitions[tile_index];
    // SAFETY: same workspace proof for the tile's closing partition.
    let first_end = partitions[tile_index + 1];
    let second_begin = diagonal_first - first_begin;
    let second_end = diagonal_last - first_end;
    // Like CUB, global offsets use the caller's offset type while every
    // tile-local shared-memory index narrows to 32-bit after the bounding box
    // proves it is at most `THREADS * ITEMS`.
    let first_count = (first_end - first_begin) as i32;
    let second_count = (second_end - second_begin) as i32;

    let exchange_items = block_exchange_storage_items::<THREADS, ITEMS>();
    let whole_buffer_bytes = load_to_shared_buffer_bytes((tile_items + 1) as u32, K_SIZE as u32)
        + if K_SIZE < 16 { 32 } else { 0 };
    let shared_memory = BlockSharedMemory::new(SHARED_BYTES);
    let mut storage_shared = shared_memory
        .region::<MaybeUninit<K>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<K>>());
    let mut local = [MaybeUninit::<K>::uninit(); ITEMS];
    let shared;
    let second_offset;
    if USE_LOAD_TO_SHARED {
        let second_buffer_bytes = load_to_shared_buffer_bytes(second_count as u32, K_SIZE as u32);
        let second_buffer_offset = whole_buffer_bytes - second_buffer_bytes;
        let first_copy = shared_memory.copy_async_from::<K, THREADS, K_SIZE>(
            0,
            first,
            first_begin,
            first_count as u32,
            thread_index,
        );
        let second_copy = shared_memory.copy_async_from::<K, THREADS, K_SIZE>(
            second_buffer_offset as usize,
            second,
            second_begin,
            second_count as u32,
            thread_index,
        );
        commit_and_wait_async_copy();
        let first_padding = first_copy.head_padding();
        second_offset = ((second_buffer_offset + second_copy.head_padding() - first_padding)
            / K_SIZE as u32) as i32;
        shared = first_copy.into_tail_region();
    } else {
        shared = shared_memory.region::<MaybeUninit<K>>(0, exchange_items);
        second_offset = first_count;
        let input = output.blocked_concatenated_input::<K, FULL>(
            first,
            first_begin,
            first_count,
            second,
            second_begin,
            second_count,
        );
        for_each_thread_item!(ITEMS, item, {
            input.load_into(item, &mut local[item]);
        });

        for_each_thread_item!(ITEMS, item, {
            let shared_index = (THREADS * item) as u32 + thread_index;
            storage_shared.write(shared_index as usize, local[item]);
        });
        thread::sync_threads();
    }

    // One local MergePath and serial merge per thread, exactly as AgentMerge.
    let thread_diagonal = if FULL {
        ITEMS as i32 * thread_index as i32
    } else {
        (ITEMS as i32 * thread_index as i32).min(items_in_tile as i32)
    };
    let local_first = shared_merge_path_region(
        &shared,
        0,
        second_offset,
        first_count,
        second_count,
        thread_diagonal,
        &order,
    );
    let mut first_cursor = local_first;
    let mut second_cursor = second_offset + thread_diagonal - local_first;
    let first_limit = first_count;
    let second_limit = second_offset + second_count;
    let mut first_key = shared.read(first_cursor as usize);
    let mut second_key = shared.read(second_cursor as usize);
    for_each_thread_item!(ITEMS, item, {
        let first_available = first_cursor < first_limit;
        let second_available = second_cursor < second_limit;
        let take_second = second_available
            && (!first_available
                || ordered_before(
                    &order,
                    initialized_ref(&second_key),
                    initialized_ref(&first_key),
                ));
        local[item] = if take_second { second_key } else { first_key };
        if take_second {
            second_cursor += 1;
            // CUB intentionally prefetches one element beyond the logical
            // range. The macro-sized shared allocation includes that padding
            // slot, and `MaybeUninit` lets Rust preserve the load without
            // interpreting the inactive value.
            second_key = shared.read(second_cursor as usize);
        } else {
            first_cursor += 1;
            first_key = shared.read(first_cursor as usize);
        }
    });

    // BlockStore aliases the merge tile storage, so all shared reads must be
    // complete before the blocked-to-warp-striped exchange begins.
    thread::sync_threads();
    let lane = thread_index & 31;
    let warp_index = thread_index >> 5;
    let warp_offset = warp_index * 32 * ITEMS as u32;
    for_each_thread_item!(ITEMS, item, {
        let exchange_index = warp_offset + item as u32 + lane * ITEMS as u32;
        storage_shared.write(block_exchange_index::<ITEMS>(exchange_index), local[item]);
    });
    warp::sync_mask(u32::MAX);

    if FULL {
        for_each_thread_item!(ITEMS, item, {
            let warp_striped = warp_offset + item as u32 * 32 + lane;
            let value =
                storage_shared.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
            output.write_if_valid(item, value);
        });
    } else {
        let mut valid_slot =
            shared_memory.region::<u32>(exchange_items * core::mem::size_of::<MaybeUninit<K>>(), 1);
        if thread_index == 0 {
            valid_slot.write_volatile(0, items_in_tile as u32);
        }
        thread::sync_threads();
        let store_items = valid_slot.read_volatile(0);
        for_each_thread_item!(ITEMS, item, {
            let warp_striped = warp_offset + item as u32 * 32 + lane;
            if warp_striped < store_items {
                let value =
                    storage_shared.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
                output.write_if_valid(item, value);
            }
        });
    }
}

/// Exact no-`BlockLoadToShared` key/value `AgentMerge::consume_tile` path.
///
/// The key merge records shared-memory source indices, then the aliased value
/// phase stages both value ranges and gathers by those indices. Keys and
/// values each use CUB's warp-transpose block store. This is the SM80/SM89
/// path for policies whose key/value layout does not select asynchronous
/// load-to-shared.
#[inline(always)]
#[allow(unused_assignments)] // CUB deliberately prefetches after the last serial-merge item.
pub fn merge_pairs_agent<
    K,
    V,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const K_SIZE: usize,
    const V_SIZE: usize,
    const USE_LOAD_TO_SHARED_KEYS: bool,
    const USE_LOAD_TO_SHARED_VALUES: bool,
    const SHARED_BYTES: usize,
>(
    first: (&[K], &[V]),
    second: (&[K], &[V]),
    partitions: &[usize],
    order: F,
    output: (
        DisjointSlice<K, LinearTiles<ITEMS>>,
        DisjointSlice<V, LinearTiles<ITEMS>>,
    ),
    launch: OneDimensionalLaunch<'_>,
) where
    K: Copy,
    V: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let (first_keys, first_values) = first;
    let (second_keys, second_values) = second;
    let (output_keys, output_values) = output;
    let tile_index = thread::blockIdx_x() as usize;
    let tile_items = THREADS * ITEMS;
    let tile_base = tile_index * tile_items;
    let total = first_keys.len() + second_keys.len();
    let items_in_tile = (total - tile_base).min(tile_items);
    if items_in_tile == tile_items {
        let tile =
            WarpStripedBlockTile::<THREADS, ITEMS>::new(launch, output_keys.len(), tile_items);
        merge_pairs_tile::<
            K,
            V,
            F,
            THREADS,
            ITEMS,
            K_SIZE,
            V_SIZE,
            USE_LOAD_TO_SHARED_KEYS,
            USE_LOAD_TO_SHARED_VALUES,
            SHARED_BYTES,
            true,
        >(
            (first_keys, first_values),
            (second_keys, second_values),
            partitions,
            order,
            (tile_index, tile_base, tile_items),
            (output_keys, output_values),
            tile,
        );
    } else {
        let tile =
            WarpStripedBlockTile::<THREADS, ITEMS>::new(launch, output_keys.len(), items_in_tile);
        merge_pairs_tile::<
            K,
            V,
            F,
            THREADS,
            ITEMS,
            K_SIZE,
            V_SIZE,
            USE_LOAD_TO_SHARED_KEYS,
            USE_LOAD_TO_SHARED_VALUES,
            SHARED_BYTES,
            false,
        >(
            (first_keys, first_values),
            (second_keys, second_values),
            partitions,
            order,
            (tile_index, tile_base, items_in_tile),
            (output_keys, output_values),
            tile,
        );
    }
}

/// CUB `AgentBlockSort` keys-only path used by DeviceMergeSort.
///
/// The proc macro fixes the architecture policy (`THREADS`, `ITEMS`) and the
/// launch chooses the borrowed destination, replacing CUB's runtime `ping`
/// pointer selection without changing the tile algorithm.
#[inline(always)]
pub fn merge_sort_block_keys_agent<
    K,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
>(
    input: &[K],
    order: F,
    output: DisjointSlice<K, LinearTiles<ITEMS>>,
    launch: OneDimensionalLaunch<'_>,
) where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let tile_index = thread::blockIdx_x() as usize;
    let tile_items = THREADS * ITEMS;
    let tile_base = tile_index * tile_items;
    let items_in_tile = input.len().saturating_sub(tile_base).min(tile_items);
    if tile_index + 1 < thread::gridDim_x() as usize {
        let input = WarpStripedBlockInput::<K, THREADS, ITEMS>::new(input, launch, tile_items);
        let output = WarpStripedBlockWrite::<K, THREADS, ITEMS>::new(output, launch, tile_items);
        merge_sort_block_keys_tile::<K, F, THREADS, ITEMS, SHARED_BYTES, false>(
            input,
            order,
            tile_items as u32,
            output,
        );
    } else {
        let input = WarpStripedBlockInput::<K, THREADS, ITEMS>::new(input, launch, items_in_tile);
        let output = WarpStripedBlockWrite::<K, THREADS, ITEMS>::new(output, launch, items_in_tile);
        merge_sort_block_keys_tile::<K, F, THREADS, ITEMS, SHARED_BYTES, true>(
            input,
            order,
            items_in_tile as u32,
            output,
        );
    }
}

/// In-place block-sort launch represented by one exclusive Rust borrow.
#[inline(always)]
pub fn merge_sort_block_keys_in_place_agent<
    K,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
>(
    order: F,
    output: DisjointSlice<K, LinearTiles<ITEMS>>,
    launch: OneDimensionalLaunch<'_>,
) where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let tile_index = thread::blockIdx_x() as usize;
    let tile_items = THREADS * ITEMS;
    let tile_base = tile_index * tile_items;
    let len = output.len();
    let items_in_tile = len.saturating_sub(tile_base).min(tile_items);
    if tile_index + 1 < thread::gridDim_x() as usize {
        let input = WarpStripedBlockRead::<K, THREADS, ITEMS>::new(output, launch, tile_items);
        merge_sort_block_keys_in_place_tile::<K, F, THREADS, ITEMS, SHARED_BYTES, false>(
            input,
            order,
            tile_items as u32,
        );
    } else {
        let input = WarpStripedBlockRead::<K, THREADS, ITEMS>::new(output, launch, items_in_tile);
        merge_sort_block_keys_in_place_tile::<K, F, THREADS, ITEMS, SHARED_BYTES, true>(
            input,
            order,
            items_in_tile as u32,
        );
    }
}

/// CUB `AgentBlockSort` key/value path used by DeviceMergeSort.
///
/// Values are loaded before keys and follow the stable key permutation through
/// every register sort and shared-memory merge round, matching the pinned
/// agent's storage-alias order.
#[inline(always)]
pub fn merge_sort_block_pairs_agent<
    K,
    V,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
>(
    input_keys: &[K],
    input_values: &[V],
    order: F,
    output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
    output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
    launch: OneDimensionalLaunch<'_>,
) where
    K: Copy,
    V: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let tile_index = thread::blockIdx_x() as usize;
    let tile_items = THREADS * ITEMS;
    let tile_base = tile_index * tile_items;
    let items_in_tile = input_keys.len().saturating_sub(tile_base).min(tile_items);
    if tile_index + 1 < thread::gridDim_x() as usize {
        let input_keys =
            WarpStripedBlockInput::<K, THREADS, ITEMS>::new(input_keys, launch, tile_items);
        let input_values =
            WarpStripedBlockInput::<V, THREADS, ITEMS>::new(input_values, launch, tile_items);
        let output_keys =
            WarpStripedBlockWrite::<K, THREADS, ITEMS>::new(output_keys, launch, tile_items);
        let output_values =
            WarpStripedBlockWrite::<V, THREADS, ITEMS>::new(output_values, launch, tile_items);
        merge_sort_block_pairs_tile::<K, V, F, _, _, THREADS, ITEMS, SHARED_BYTES, false>(
            input_keys,
            input_values,
            order,
            tile_items as u32,
            output_keys,
            output_values,
        );
    } else {
        let input_keys =
            WarpStripedBlockInput::<K, THREADS, ITEMS>::new(input_keys, launch, items_in_tile);
        let input_values =
            WarpStripedBlockInput::<V, THREADS, ITEMS>::new(input_values, launch, items_in_tile);
        let output_keys =
            WarpStripedBlockWrite::<K, THREADS, ITEMS>::new(output_keys, launch, items_in_tile);
        let output_values =
            WarpStripedBlockWrite::<V, THREADS, ITEMS>::new(output_values, launch, items_in_tile);
        merge_sort_block_pairs_tile::<K, V, F, _, _, THREADS, ITEMS, SHARED_BYTES, true>(
            input_keys,
            input_values,
            order,
            items_in_tile as u32,
            output_keys,
            output_values,
        );
    }
}

/// In-place key/value block-sort launch represented by two exclusive borrows.
#[inline(always)]
pub fn merge_sort_block_pairs_in_place_agent<
    K,
    V,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
>(
    order: F,
    output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
    output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
    launch: OneDimensionalLaunch<'_>,
) where
    K: Copy,
    V: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let tile_index = thread::blockIdx_x() as usize;
    let tile_items = THREADS * ITEMS;
    let tile_base = tile_index * tile_items;
    let len = output_keys.len();
    let items_in_tile = len.saturating_sub(tile_base).min(tile_items);
    if tile_index + 1 < thread::gridDim_x() as usize {
        let input_keys =
            WarpStripedBlockRead::<K, THREADS, ITEMS>::new(output_keys, launch, tile_items);
        let input_values =
            WarpStripedBlockRead::<V, THREADS, ITEMS>::new(output_values, launch, tile_items);
        merge_sort_block_pairs_in_place_tile::<K, V, F, THREADS, ITEMS, SHARED_BYTES, false>(
            input_keys,
            input_values,
            order,
            tile_items as u32,
        );
    } else {
        let input_keys =
            WarpStripedBlockRead::<K, THREADS, ITEMS>::new(output_keys, launch, items_in_tile);
        let input_values =
            WarpStripedBlockRead::<V, THREADS, ITEMS>::new(output_values, launch, items_in_tile);
        merge_sort_block_pairs_in_place_tile::<K, V, F, THREADS, ITEMS, SHARED_BYTES, true>(
            input_keys,
            input_values,
            order,
            items_in_tile as u32,
        );
    }
}

trait WarpStripedSource<T: Copy> {
    fn first(&mut self) -> T;
    fn read_or(&mut self, item: usize, default: T) -> T;
}

impl<T: Copy, const THREADS: usize, const ITEMS: usize> WarpStripedSource<T>
    for WarpStripedBlockInput<'_, T, THREADS, ITEMS>
{
    #[inline(always)]
    fn first(&mut self) -> T {
        WarpStripedBlockInput::first(self)
    }

    #[inline(always)]
    fn read_or(&mut self, item: usize, default: T) -> T {
        WarpStripedBlockInput::read_or(self, item, default)
    }
}

impl<T: Copy, const THREADS: usize, const ITEMS: usize> WarpStripedSource<T>
    for WarpStripedBlockRead<'_, T, THREADS, ITEMS>
{
    #[inline(always)]
    fn first(&mut self) -> T {
        WarpStripedBlockRead::first(self)
    }

    #[inline(always)]
    fn read_or(&mut self, item: usize, default: T) -> T {
        WarpStripedBlockRead::read_or(self, item, default)
    }
}

#[inline(always)]
fn merge_sort_block_pairs_tile<
    K,
    V,
    F,
    KeySource,
    ValueSource,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
    const LAST: bool,
>(
    mut input_keys: KeySource,
    mut input_values: ValueSource,
    order: F,
    valid_items: u32,
    output_keys: WarpStripedBlockWrite<'_, K, THREADS, ITEMS>,
    output_values: WarpStripedBlockWrite<'_, V, THREADS, ITEMS>,
) where
    K: Copy,
    V: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
    KeySource: WarpStripedSource<K>,
    ValueSource: WarpStripedSource<V>,
{
    let (local_keys, local_values) =
        merge_sort_block_pairs_load::<K, V, KeySource, ValueSource, THREADS, ITEMS, SHARED_BYTES>(
            &mut input_keys,
            &mut input_values,
        );
    thread::sync_threads();
    merge_sort_block_pairs_loaded::<K, V, F, THREADS, ITEMS, SHARED_BYTES, LAST>(
        local_keys,
        local_values,
        order,
        valid_items,
        output_keys,
        output_values,
    );
}

#[inline(always)]
fn merge_sort_block_pairs_in_place_tile<
    K,
    V,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
    const LAST: bool,
>(
    mut input_keys: WarpStripedBlockRead<'_, K, THREADS, ITEMS>,
    mut input_values: WarpStripedBlockRead<'_, V, THREADS, ITEMS>,
    order: F,
    valid_items: u32,
) where
    K: Copy,
    V: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let (local_keys, local_values) =
        merge_sort_block_pairs_load::<K, V, _, _, THREADS, ITEMS, SHARED_BYTES>(
            &mut input_keys,
            &mut input_values,
        );
    let (output_keys, output_values) = warp_striped_pair_into_write(input_keys, input_values);
    merge_sort_block_pairs_loaded::<K, V, F, THREADS, ITEMS, SHARED_BYTES, LAST>(
        local_keys,
        local_values,
        order,
        valid_items,
        output_keys,
        output_values,
    );
}

#[inline(always)]
fn merge_sort_block_pairs_load<
    K,
    V,
    KeySource,
    ValueSource,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
>(
    input_keys: &mut KeySource,
    input_values: &mut ValueSource,
) -> ([K; ITEMS], [V; ITEMS])
where
    K: Copy,
    V: Copy,
    KeySource: WarpStripedSource<K>,
    ValueSource: WarpStripedSource<V>,
{
    let thread_index = thread::threadIdx_x();
    let lane = thread_index & 31;
    let warp_index = thread_index >> 5;
    let warp_offset = warp_index * 32 * ITEMS as u32;
    let shared_memory = BlockSharedMemory::new(SHARED_BYTES);

    // AgentBlockSort deliberately loads values first, then aliases the same
    // temporary storage for keys after a block barrier.
    let value_fallback = input_values.first();
    let mut local_values = [value_fallback; ITEMS];
    let mut shared_values = shared_memory
        .region::<MaybeUninit<V>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<V>>());
    for_each_thread_item!(ITEMS, item, {
        let warp_striped = warp_offset + item as u32 * 32 + lane;
        local_values[item] = input_values.read_or(item, value_fallback);
        shared_values.write(
            block_exchange_index::<ITEMS>(warp_striped),
            MaybeUninit::new(local_values[item]),
        );
    });
    warp::sync_mask(u32::MAX);
    for_each_thread_item!(ITEMS, item, {
        let blocked = warp_offset + lane * ITEMS as u32 + item as u32;
        local_values[item] = shared_values.read_initialized(block_exchange_index::<ITEMS>(blocked));
    });
    thread::sync_threads();

    let key_fallback = input_keys.first();
    let mut local_keys = [key_fallback; ITEMS];
    let mut shared_keys = shared_memory
        .region::<MaybeUninit<K>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<K>>());
    for_each_thread_item!(ITEMS, item, {
        let warp_striped = warp_offset + item as u32 * 32 + lane;
        local_keys[item] = input_keys.read_or(item, key_fallback);
        shared_keys.write(
            block_exchange_index::<ITEMS>(warp_striped),
            MaybeUninit::new(local_keys[item]),
        );
    });
    warp::sync_mask(u32::MAX);
    for_each_thread_item!(ITEMS, item, {
        let blocked = warp_offset + lane * ITEMS as u32 + item as u32;
        local_keys[item] = shared_keys.read_initialized(block_exchange_index::<ITEMS>(blocked));
    });
    (local_keys, local_values)
}

#[inline(always)]
#[allow(unused_assignments)]
fn merge_sort_block_pairs_loaded<
    K,
    V,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
    const LAST: bool,
>(
    mut local_keys: [K; ITEMS],
    mut local_values: [V; ITEMS],
    order: F,
    valid_items: u32,
    mut output_keys: WarpStripedBlockWrite<'_, K, THREADS, ITEMS>,
    mut output_values: WarpStripedBlockWrite<'_, V, THREADS, ITEMS>,
) where
    K: Copy,
    V: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let thread_index = thread::threadIdx_x();
    let lane = thread_index & 31;
    let warp_index = thread_index >> 5;
    let warp_offset = warp_index * 32 * ITEMS as u32;
    let tile_items = THREADS * ITEMS;
    let exchange_items = block_exchange_storage_items::<THREADS, ITEMS>();
    let shared_memory = BlockSharedMemory::new(SHARED_BYTES);
    let mut shared_keys = shared_memory
        .region::<MaybeUninit<K>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<K>>());
    let mut shared_values = shared_memory
        .region::<MaybeUninit<V>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<V>>());

    if LAST {
        let blocked_begin = ITEMS as u32 * thread_index;
        if blocked_begin < valid_items {
            let mut maximum = local_keys[0];
            for_each_thread_item!(ITEMS, item, {
                if item != 0 {
                    if (blocked_begin + item as u32) < valid_items {
                        if ordered_before(&order, &maximum, &local_keys[item]) {
                            maximum = local_keys[item];
                        }
                    } else {
                        local_keys[item] = maximum;
                    }
                }
            });
            stable_odd_even_sort_pairs::<K, V, F, ITEMS>(
                &mut local_keys,
                &mut local_values,
                &order,
            );
        }
        // CUB deliberately leaves MergeRounds as a runtime doubling loop; it
        // carries no unroll pragma and is not template policy selection.
        let mut target_merged_threads = 2u32;
        while target_merged_threads <= THREADS as u32 {
            let merged_size = ITEMS as u32 * (target_merged_threads / 2);
            if merged_size >= valid_items {
                break;
            }
            block_merge_round_pairs::<K, V, F, THREADS, ITEMS, true>(
                &mut shared_keys,
                &shared_memory,
                &mut local_keys,
                &mut local_values,
                valid_items,
                target_merged_threads,
                &order,
            );
            target_merged_threads *= 2;
        }
    } else {
        stable_odd_even_sort_pairs::<K, V, F, ITEMS>(&mut local_keys, &mut local_values, &order);
        let mut target_merged_threads = 2u32;
        while target_merged_threads <= THREADS as u32 {
            block_merge_round_pairs::<K, V, F, THREADS, ITEMS, false>(
                &mut shared_keys,
                &shared_memory,
                &mut local_keys,
                &mut local_values,
                tile_items as u32,
                target_merged_threads,
                &order,
            );
            target_merged_threads *= 2;
        }
    }

    thread::sync_threads();
    for_each_thread_item!(ITEMS, item, {
        let blocked = warp_offset + lane * ITEMS as u32 + item as u32;
        shared_keys.write(
            block_exchange_index::<ITEMS>(blocked),
            MaybeUninit::new(local_keys[item]),
        );
    });
    warp::sync_mask(u32::MAX);
    let key_store_items = if LAST {
        let mut valid_slot =
            shared_memory.region::<u32>(exchange_items * core::mem::size_of::<MaybeUninit<K>>(), 1);
        if thread_index == 0 {
            valid_slot.write_volatile(0, valid_items);
        }
        thread::sync_threads();
        valid_slot.read_volatile(0)
    } else {
        valid_items
    };
    for_each_thread_item!(ITEMS, item, {
        let warp_striped = warp_offset + item as u32 * 32 + lane;
        if !LAST || warp_striped < key_store_items {
            let value = shared_keys.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
            output_keys.write_if_valid(item, value);
        }
    });

    thread::sync_threads();
    for_each_thread_item!(ITEMS, item, {
        let blocked = warp_offset + lane * ITEMS as u32 + item as u32;
        shared_values.write(
            block_exchange_index::<ITEMS>(blocked),
            MaybeUninit::new(local_values[item]),
        );
    });
    warp::sync_mask(u32::MAX);
    let value_store_items = if LAST {
        let mut valid_slot =
            shared_memory.region::<u32>(exchange_items * core::mem::size_of::<MaybeUninit<V>>(), 1);
        if thread_index == 0 {
            valid_slot.write_volatile(0, valid_items);
        }
        thread::sync_threads();
        valid_slot.read_volatile(0)
    } else {
        valid_items
    };
    for_each_thread_item!(ITEMS, item, {
        let warp_striped = warp_offset + item as u32 * 32 + lane;
        if !LAST || warp_striped < value_store_items {
            let value = shared_values.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
            output_values.write_if_valid(item, value);
        }
    });
}

#[inline(always)]
#[allow(unused_assignments)]
fn merge_sort_block_keys_tile<
    K,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
    const LAST: bool,
>(
    input: WarpStripedBlockInput<'_, K, THREADS, ITEMS>,
    order: F,
    valid_items: u32,
    output: WarpStripedBlockWrite<'_, K, THREADS, ITEMS>,
) where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let thread_index = thread::threadIdx_x();
    let lane = thread_index & 31;
    let warp_index = thread_index >> 5;
    let warp_offset = warp_index * 32 * ITEMS as u32;
    // The host lowering launches exactly ceil(len / tile_items) blocks for a
    // nonempty input, so every block has a first key. CUB uses that key as the
    // guarded BlockLoad default for the final partial tile.
    let fallback = input.first();
    let mut local = [fallback; ITEMS];
    let shared_memory = BlockSharedMemory::new(SHARED_BYTES);
    let mut shared = shared_memory
        .region::<MaybeUninit<K>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<K>>());

    // BLOCK_LOAD_WARP_TRANSPOSE: coalesced warp-striped loads followed by a
    // warp-local exchange into blocked per-thread registers.
    for_each_thread_item!(ITEMS, item, {
        let warp_striped = warp_offset + item as u32 * 32 + lane;
        local[item] = input.read_or(item, fallback);
        shared.write(
            block_exchange_index::<ITEMS>(warp_striped),
            MaybeUninit::new(local[item]),
        );
    });
    warp::sync_mask(u32::MAX);
    for_each_thread_item!(ITEMS, item, {
        let blocked = warp_offset + lane * ITEMS as u32 + item as u32;
        local[item] = shared.read_initialized(block_exchange_index::<ITEMS>(blocked));
    });
    thread::sync_threads();

    merge_sort_block_keys_loaded::<K, F, THREADS, ITEMS, SHARED_BYTES, LAST>(
        shared,
        local,
        order,
        valid_items,
        output,
    );
}

#[inline(always)]
#[allow(unused_assignments)]
fn merge_sort_block_keys_in_place_tile<
    K,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
    const LAST: bool,
>(
    mut input: WarpStripedBlockRead<'_, K, THREADS, ITEMS>,
    order: F,
    valid_items: u32,
) where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let thread_index = thread::threadIdx_x();
    let lane = thread_index & 31;
    let warp_index = thread_index >> 5;
    let warp_offset = warp_index * 32 * ITEMS as u32;
    let fallback = input.first();
    let mut local = [fallback; ITEMS];
    let shared_memory = BlockSharedMemory::new(SHARED_BYTES);
    let mut shared = shared_memory
        .region::<MaybeUninit<K>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<K>>());

    for_each_thread_item!(ITEMS, item, {
        let warp_striped = warp_offset + item as u32 * 32 + lane;
        local[item] = input.read_or(item, fallback);
        shared.write(
            block_exchange_index::<ITEMS>(warp_striped),
            MaybeUninit::new(local[item]),
        );
    });
    warp::sync_mask(u32::MAX);
    for_each_thread_item!(ITEMS, item, {
        let blocked = warp_offset + lane * ITEMS as u32 + item as u32;
        local[item] = shared.read_initialized(block_exchange_index::<ITEMS>(blocked));
    });
    // Consuming the exclusive read phase performs CUB's block barrier before
    // the same allocation becomes writable again.
    let output = input.into_write();
    merge_sort_block_keys_loaded::<K, F, THREADS, ITEMS, SHARED_BYTES, LAST>(
        shared,
        local,
        order,
        valid_items,
        output,
    );
}

#[inline(always)]
#[allow(unused_assignments)]
fn merge_sort_block_keys_loaded<
    K,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
    const LAST: bool,
>(
    mut shared: SharedRegion<MaybeUninit<K>>,
    mut local: [K; ITEMS],
    order: F,
    valid_items: u32,
    mut output: WarpStripedBlockWrite<'_, K, THREADS, ITEMS>,
) where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let thread_index = thread::threadIdx_x();
    let lane = thread_index & 31;
    let warp_index = thread_index >> 5;
    let warp_offset = warp_index * 32 * ITEMS as u32;
    let tile_items = THREADS * ITEMS;

    if LAST {
        let blocked_begin = ITEMS as u32 * thread_index;
        if blocked_begin < valid_items {
            let mut maximum = local[0];
            for_each_thread_item!(ITEMS, item, {
                if item != 0 {
                    if (blocked_begin + item as u32) < valid_items {
                        if ordered_before(&order, &maximum, &local[item]) {
                            maximum = local[item];
                        }
                    } else {
                        local[item] = maximum;
                    }
                }
            });
            stable_odd_even_sort::<K, F, ITEMS>(&mut local, &order);
        }
        // CUB deliberately leaves MergeRounds as a runtime doubling loop; it
        // carries no unroll pragma and is not template policy selection.
        let mut target_merged_threads = 2u32;
        while target_merged_threads <= THREADS as u32 {
            let merged_size = ITEMS as u32 * (target_merged_threads / 2);
            if merged_size >= valid_items {
                break;
            }
            block_merge_round::<K, F, THREADS, ITEMS, true>(
                &mut shared,
                &mut local,
                valid_items,
                target_merged_threads,
                &order,
            );
            target_merged_threads *= 2;
        }
    } else {
        stable_odd_even_sort::<K, F, ITEMS>(&mut local, &order);
        let mut target_merged_threads = 2u32;
        while target_merged_threads <= THREADS as u32 {
            block_merge_round::<K, F, THREADS, ITEMS, false>(
                &mut shared,
                &mut local,
                tile_items as u32,
                target_merged_threads,
                &order,
            );
            target_merged_threads *= 2;
        }
    }

    // AgentBlockSort synchronizes before reusing BlockMergeSort storage for
    // BLOCK_STORE_WARP_TRANSPOSE.
    thread::sync_threads();
    for_each_thread_item!(ITEMS, item, {
        let blocked = warp_offset + lane * ITEMS as u32 + item as u32;
        shared.write(
            block_exchange_index::<ITEMS>(blocked),
            MaybeUninit::new(local[item]),
        );
    });
    warp::sync_mask(u32::MAX);
    let store_items = if LAST {
        let shared_memory = BlockSharedMemory::new(SHARED_BYTES);
        let mut valid_slot = shared_memory.region::<u32>(
            block_exchange_storage_items::<THREADS, ITEMS>()
                * core::mem::size_of::<MaybeUninit<K>>(),
            1,
        );
        if thread_index == 0 {
            valid_slot.write_volatile(0, valid_items);
        }
        thread::sync_threads();
        valid_slot.read_volatile(0)
    } else {
        valid_items
    };
    for_each_thread_item!(ITEMS, item, {
        let warp_striped = warp_offset + item as u32 * 32 + lane;
        if !LAST || warp_striped < store_items {
            let value = shared.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
            output.write_if_valid(item, value);
        }
    });
}

#[inline(always)]
fn stable_odd_even_sort<K, F, const ITEMS: usize>(local: &mut [K; ITEMS], order: &F)
where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    for_each_thread_item!(ITEMS, phase, {
        for_each_thread_item!(ITEMS, left, {
            if left + 1 < ITEMS && (left & 1) == (phase & 1) {
                let right = left + 1;
                if ordered_before(order, &local[right], &local[left]) {
                    local.swap(left, right);
                }
            }
        });
    });
}

#[inline(always)]
fn stable_odd_even_sort_pairs<K, V, F, const ITEMS: usize>(
    keys: &mut [K; ITEMS],
    values: &mut [V; ITEMS],
    order: &F,
) where
    K: Copy,
    V: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    for_each_thread_item!(ITEMS, phase, {
        for_each_thread_item!(ITEMS, left, {
            if left + 1 < ITEMS && (left & 1) == (phase & 1) {
                let right = left + 1;
                let left_key = keys[left];
                let right_key = keys[right];
                let left_value = values[left];
                let right_value = values[right];
                let exchange = ordered_before(order, &right_key, &left_key);
                keys[left] = if exchange { right_key } else { left_key };
                keys[right] = if exchange { left_key } else { right_key };
                values[left] = if exchange { right_value } else { left_value };
                values[right] = if exchange { left_value } else { right_value };
            }
        });
    });
}

#[inline(always)]
#[allow(unused_assignments)]
fn block_merge_round<K, F, const THREADS: usize, const ITEMS: usize, const CLAMPED: bool>(
    shared: &mut SharedRegion<MaybeUninit<K>>,
    local: &mut [K; ITEMS],
    valid_items: u32,
    target_merged_threads: u32,
    order: &F,
) where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let thread_index = thread::threadIdx_x() as i32;
    let valid_items = valid_items as i32;
    let target_merged_threads = target_merged_threads as i32;
    let tile_items = (THREADS * ITEMS) as i32;
    thread::sync_threads();
    for_each_thread_item!(ITEMS, item, {
        let blocked = ITEMS as i32 * thread_index + item as i32;
        shared.write(blocked as usize, MaybeUninit::new(local[item]));
    });
    if thread_index == 0 {
        // Exact BlockMergeSort padding contract from NVIDIA/cccl#5327.
        shared.write(tile_items as usize, MaybeUninit::new(local[0]));
    }
    thread::sync_threads();

    let merged_threads = target_merged_threads / 2;
    let mask = target_merged_threads - 1;
    let size = ITEMS as i32 * merged_threads;
    let start = ITEMS as i32 * ((!mask) & thread_index);
    let thread_in_group = mask & thread_index;
    let (first_begin, first_end, second_end, diagonal) = if CLAMPED {
        let first_begin = valid_items.min(start);
        let first_end = valid_items.min(first_begin + size);
        let second_end = valid_items.min(first_end + size);
        let diagonal =
            (second_end - first_begin).min(valid_items.min(ITEMS as i32 * thread_in_group));
        (first_begin, first_end, second_end, diagonal)
    } else {
        (
            start,
            start + size,
            start + 2 * size,
            ITEMS as i32 * thread_in_group,
        )
    };
    let second_begin = first_end;
    let first_count = first_end - first_begin;
    let second_count = second_end - second_begin;
    let partition = shared_merge_path_region(
        shared,
        first_begin,
        second_begin - first_begin,
        first_count,
        second_count,
        diagonal,
        order,
    );
    let mut first_cursor = first_begin + partition;
    let mut second_cursor = second_begin + diagonal - partition;
    let first_limit = first_end;
    let second_limit = second_end;
    // CUB's no-explicit-default serial merge passes output[0] as its
    // out-of-bounds default.  Keep typed heads here: an empty run must not
    // manufacture an initialized key by reading an inactive shared slot.
    let oob_default = local[0];
    let mut first_key = if first_count != 0 {
        shared.read_initialized(first_cursor as usize)
    } else {
        oob_default
    };
    let mut second_key = if second_count != 0 {
        shared.read_initialized(second_cursor as usize)
    } else {
        oob_default
    };
    for_each_thread_item!(ITEMS, item, {
        let take_second = second_cursor < second_limit
            && (first_cursor >= first_limit || ordered_before(order, &second_key, &first_key));
        local[item] = if take_second { second_key } else { first_key };
        if take_second {
            second_cursor += 1;
            second_key = shared.read_initialized(second_cursor as usize);
        } else {
            first_cursor += 1;
            first_key = shared.read_initialized(first_cursor as usize);
        }
    });
}

#[inline(always)]
#[allow(unused_assignments)]
fn block_merge_round_pairs<K, V, F, const THREADS: usize, const ITEMS: usize, const CLAMPED: bool>(
    shared_keys: &mut SharedRegion<MaybeUninit<K>>,
    shared_memory: &BlockSharedMemory,
    local_keys: &mut [K; ITEMS],
    local_values: &mut [V; ITEMS],
    valid_items: u32,
    target_merged_threads: u32,
    order: &F,
) where
    K: Copy,
    V: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let thread_index = thread::threadIdx_x() as i32;
    let valid_items = valid_items as i32;
    let target_merged_threads = target_merged_threads as i32;
    let tile_items = (THREADS * ITEMS) as i32;
    thread::sync_threads();
    for_each_thread_item!(ITEMS, item, {
        let blocked = ITEMS as i32 * thread_index + item as i32;
        shared_keys.write(blocked as usize, MaybeUninit::new(local_keys[item]));
    });
    if thread_index == 0 {
        shared_keys.write(tile_items as usize, MaybeUninit::new(local_keys[0]));
    }
    thread::sync_threads();

    let merged_threads = target_merged_threads / 2;
    let mask = target_merged_threads - 1;
    let size = ITEMS as i32 * merged_threads;
    let start = ITEMS as i32 * ((!mask) & thread_index);
    let thread_in_group = mask & thread_index;
    let (first_begin, first_end, second_end, diagonal) = if CLAMPED {
        let first_begin = valid_items.min(start);
        let first_end = valid_items.min(first_begin + size);
        let second_end = valid_items.min(first_end + size);
        let diagonal =
            (second_end - first_begin).min(valid_items.min(ITEMS as i32 * thread_in_group));
        (first_begin, first_end, second_end, diagonal)
    } else {
        (
            start,
            start + size,
            start + 2 * size,
            ITEMS as i32 * thread_in_group,
        )
    };
    let second_begin = first_end;
    let first_count = first_end - first_begin;
    let second_count = second_end - second_begin;
    let partition = shared_merge_path_region(
        shared_keys,
        first_begin,
        second_begin - first_begin,
        first_count,
        second_count,
        diagonal,
        order,
    );
    let mut first_cursor = first_begin + partition;
    let mut second_cursor = second_begin + diagonal - partition;
    let first_limit = first_end;
    let second_limit = second_end;
    let oob_default = local_keys[0];
    let mut first_key = if first_count != 0 {
        shared_keys.read_initialized(first_cursor as usize)
    } else {
        oob_default
    };
    let mut second_key = if second_count != 0 {
        shared_keys.read_initialized(second_cursor as usize)
    } else {
        oob_default
    };
    let mut indices = [0i32; ITEMS];
    for_each_thread_item!(ITEMS, item, {
        let take_second = second_cursor < second_limit
            && (first_cursor >= first_limit || ordered_before(order, &second_key, &first_key));
        if take_second {
            local_keys[item] = second_key;
            indices[item] = second_cursor;
            second_cursor += 1;
            second_key = shared_keys.read_initialized(second_cursor as usize);
        } else {
            local_keys[item] = first_key;
            indices[item] = first_cursor;
            first_cursor += 1;
            first_key = shared_keys.read_initialized(first_cursor as usize);
        }
    });

    // CUB aliases the key union as values only after every thread has
    // completed serial key merge, then gathers the recorded source indices.
    thread::sync_threads();
    let mut shared_values = shared_memory.region::<MaybeUninit<V>>(
        0,
        shared_memory.bytes() / core::mem::size_of::<MaybeUninit<V>>(),
    );
    for_each_thread_item!(ITEMS, item, {
        let blocked = ITEMS as i32 * thread_index + item as i32;
        shared_values.write(blocked as usize, MaybeUninit::new(local_values[item]));
    });
    thread::sync_threads();
    for_each_thread_item!(ITEMS, item, {
        local_values[item] = shared_values.read_initialized(indices[item] as usize);
    });
}

/// Exact DeviceMergeSort `AgentPartition::Process` point computation over the
/// currently borrowed ping range.
#[inline(always)]
pub fn merge_sort_partition_point<K, F, const THREADS: usize, const ITEMS: usize>(
    source: &[K],
    partition_index: usize,
    partition_count: usize,
    target_merged_tiles: usize,
    order: &F,
) -> usize
where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let tile_items = THREADS * ITEMS;
    let merged_tiles = target_merged_tiles / 2;
    let mask = target_merged_tiles - 1;
    let list = (!mask) & partition_index;
    let start = tile_items * list;
    let size = tile_items * merged_tiles;
    let first_begin = source.len().min(start);
    let first_end = source.len().min(start.saturating_add(size));
    let second_begin = first_end;
    let second_end = source.len().min(second_begin.saturating_add(size));
    if partition_index + 1 == partition_count {
        first_end
    } else {
        let local_tile = mask & partition_index;
        let diagonal = (second_end - first_begin).min(tile_items * local_tile);
        first_begin
            + global_merge_path(
                &source[first_begin..first_end],
                &source[second_begin..second_end],
                diagonal,
                order,
            )
    }
}

/// Exact keys-only DeviceMergeSort merge agent for one ping-pong pass.
#[inline(always)]
pub fn merge_sort_merge_keys_agent<
    K,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
>(
    source: &[K],
    partitions: &[usize],
    target_merged_tiles: usize,
    order: F,
    output: DisjointSlice<K, LinearTiles<ITEMS>>,
    launch: OneDimensionalLaunch<'_>,
) where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let tile_index = thread::blockIdx_x() as usize;
    let tile_items = THREADS * ITEMS;
    let tile_base = tile_index * tile_items;
    let items_in_tile = (source.len() - tile_base).min(tile_items);
    if items_in_tile == tile_items {
        let output = WarpStripedBlockWrite::<K, THREADS, ITEMS>::new(output, launch, tile_items);
        merge_sort_merge_keys_tile::<K, F, THREADS, ITEMS, SHARED_BYTES, true>(
            source,
            partitions,
            target_merged_tiles,
            order,
            tile_index,
            tile_items,
            output,
        );
    } else {
        let output = WarpStripedBlockWrite::<K, THREADS, ITEMS>::new(output, launch, items_in_tile);
        merge_sort_merge_keys_tile::<K, F, THREADS, ITEMS, SHARED_BYTES, false>(
            source,
            partitions,
            target_merged_tiles,
            order,
            tile_index,
            items_in_tile,
            output,
        );
    }
}

#[inline(always)]
#[allow(unused_assignments)]
fn merge_sort_merge_keys_tile<
    K,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
    const FULL: bool,
>(
    source: &[K],
    partitions: &[usize],
    target_merged_tiles: usize,
    order: F,
    tile_index: usize,
    _items_in_tile: usize,
    mut output: WarpStripedBlockWrite<'_, K, THREADS, ITEMS>,
) where
    K: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let thread_index = thread::threadIdx_x();
    let tile_items = THREADS * ITEMS;
    let partition_begin = partitions[tile_index];
    let partition_end = partitions[tile_index + 1];
    let merged_tiles = target_merged_tiles / 2;
    let mask = target_merged_tiles - 1;
    let list = (!mask) & tile_index;
    let start = tile_items * list;
    let size = tile_items * merged_tiles;
    let diagonal = tile_items * tile_index - start;
    let first_begin = partition_begin - start;
    let mut first_end = partition_end - start;
    let distance = source.len() - start;
    let max_second = distance.saturating_sub(size);
    let second_begin = max_second.min(diagonal - first_begin);
    let mut second_end = max_second.min(diagonal.saturating_add(tile_items) - first_end);
    if mask == (mask & tile_index) {
        first_end = distance.min(size);
        second_end = max_second.min(size);
    }
    let first_count = (first_end - first_begin) as i32;
    let second_count = (second_end - second_begin) as i32;
    let input = output.blocked_concatenated_input::<K, FULL>(
        source,
        start + first_begin,
        first_count,
        source,
        start + size + second_begin,
        second_count,
    );
    let mut local = [MaybeUninit::<K>::uninit(); ITEMS];
    for_each_thread_item!(ITEMS, item, {
        input.load_into(item, &mut local[item]);
    });
    let exchange_items = block_exchange_storage_items::<THREADS, ITEMS>();
    let shared_memory = BlockSharedMemory::new(SHARED_BYTES);
    let mut shared = shared_memory
        .region::<MaybeUninit<K>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<K>>());
    for_each_thread_item!(ITEMS, item, {
        let index = THREADS as i32 * item as i32 + thread_index as i32;
        shared.write(index as usize, local[item]);
    });
    thread::sync_threads();

    let local_diagonal = (first_count + second_count).min(ITEMS as i32 * thread_index as i32);
    let partition = shared_merge_path_region(
        &shared,
        0,
        first_count,
        first_count,
        second_count,
        local_diagonal,
        &order,
    );
    let mut first_cursor = partition;
    let mut second_cursor = first_count + local_diagonal - partition;
    let first_limit = first_count;
    let second_limit = first_count + second_count;
    let mut first_key = shared.read(first_cursor as usize);
    let mut second_key = shared.read(second_cursor as usize);
    for_each_thread_item!(ITEMS, item, {
        let take_second = second_cursor < second_limit
            && (first_cursor >= first_limit
                || ordered_before(
                    &order,
                    initialized_ref(&second_key),
                    initialized_ref(&first_key),
                ));
        local[item] = if take_second { second_key } else { first_key };
        if take_second {
            second_cursor += 1;
            second_key = shared.read(second_cursor as usize);
        } else {
            first_cursor += 1;
            first_key = shared.read(first_cursor as usize);
        }
    });

    thread::sync_threads();
    let lane = thread_index & 31;
    let warp_index = thread_index >> 5;
    let warp_offset = warp_index * 32 * ITEMS as u32;
    for_each_thread_item!(ITEMS, item, {
        let blocked = warp_offset + item as u32 + lane * ITEMS as u32;
        shared.write(block_exchange_index::<ITEMS>(blocked), local[item]);
    });
    warp::sync_mask(u32::MAX);
    let store_items = if FULL {
        (first_count + second_count) as u32
    } else {
        let mut valid_slot =
            shared_memory.region::<u32>(exchange_items * core::mem::size_of::<MaybeUninit<K>>(), 1);
        if thread_index == 0 {
            valid_slot.write_volatile(0, (first_count + second_count) as u32);
        }
        thread::sync_threads();
        valid_slot.read_volatile(0)
    };
    for_each_thread_item!(ITEMS, item, {
        let warp_striped = warp_offset + item as u32 * 32 + lane;
        if FULL || warp_striped < store_items {
            let value = shared.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
            output.write_if_valid(item, value);
        }
    });
}

/// Exact key/value DeviceMergeSort merge agent for one ping-pong pass.
#[inline(always)]
pub fn merge_sort_merge_pairs_agent<
    K,
    V,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
>(
    source: (&[K], &[V]),
    partitions: &[usize],
    target_merged_tiles: usize,
    order: F,
    output: (
        DisjointSlice<K, LinearTiles<ITEMS>>,
        DisjointSlice<V, LinearTiles<ITEMS>>,
    ),
    launch: OneDimensionalLaunch<'_>,
) where
    K: Copy,
    V: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let (source_keys, source_values) = source;
    let (output_keys, output_values) = output;
    let tile_index = thread::blockIdx_x() as usize;
    let tile_items = THREADS * ITEMS;
    let tile_base = tile_index * tile_items;
    let items_in_tile = (source_keys.len() - tile_base).min(tile_items);
    if items_in_tile == tile_items {
        let output_keys =
            WarpStripedBlockWrite::<K, THREADS, ITEMS>::new(output_keys, launch, tile_items);
        let output_values =
            WarpStripedBlockWrite::<V, THREADS, ITEMS>::new(output_values, launch, tile_items);
        merge_sort_merge_pairs_tile::<K, V, F, THREADS, ITEMS, SHARED_BYTES, true>(
            (source_keys, source_values),
            partitions,
            target_merged_tiles,
            order,
            (tile_index, tile_items),
            (output_keys, output_values),
        );
    } else {
        let output_keys =
            WarpStripedBlockWrite::<K, THREADS, ITEMS>::new(output_keys, launch, items_in_tile);
        let output_values =
            WarpStripedBlockWrite::<V, THREADS, ITEMS>::new(output_values, launch, items_in_tile);
        merge_sort_merge_pairs_tile::<K, V, F, THREADS, ITEMS, SHARED_BYTES, false>(
            (source_keys, source_values),
            partitions,
            target_merged_tiles,
            order,
            (tile_index, items_in_tile),
            (output_keys, output_values),
        );
    }
}

#[inline(always)]
#[allow(unused_assignments)]
fn merge_sort_merge_pairs_tile<
    K,
    V,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const SHARED_BYTES: usize,
    const FULL: bool,
>(
    source: (&[K], &[V]),
    partitions: &[usize],
    target_merged_tiles: usize,
    order: F,
    tile_position: (usize, usize),
    output: (
        WarpStripedBlockWrite<'_, K, THREADS, ITEMS>,
        WarpStripedBlockWrite<'_, V, THREADS, ITEMS>,
    ),
) where
    K: Copy,
    V: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let (source_keys, source_values) = source;
    let (tile_index, items_in_tile) = tile_position;
    let (mut output_keys, mut output_values) = output;
    let thread_index = thread::threadIdx_x();
    let tile_items = THREADS * ITEMS;
    let partition_begin = partitions[tile_index];
    let partition_end = partitions[tile_index + 1];
    let merged_tiles = target_merged_tiles / 2;
    let mask = target_merged_tiles - 1;
    let list = (!mask) & tile_index;
    let start = tile_items * list;
    let size = tile_items * merged_tiles;
    let diagonal = tile_items * tile_index - start;
    let first_begin = partition_begin - start;
    let mut first_end = partition_end - start;
    let distance = source_keys.len() - start;
    let max_second = distance.saturating_sub(size);
    let second_begin = max_second.min(diagonal - first_begin);
    let mut second_end = max_second.min(diagonal.saturating_add(tile_items) - first_end);
    if mask == (mask & tile_index) {
        first_end = distance.min(size);
        second_end = max_second.min(size);
    }
    let first_count = (first_end - first_begin) as i32;
    let second_count = (second_end - second_begin) as i32;
    let local_count = first_count + second_count;
    let keys = output_keys.blocked_concatenated_input::<K, FULL>(
        source_keys,
        start + first_begin,
        first_count,
        source_keys,
        start + size + second_begin,
        second_count,
    );
    let values = output_keys.blocked_concatenated_input::<V, FULL>(
        source_values,
        start + first_begin,
        first_count,
        source_values,
        start + size + second_begin,
        second_count,
    );
    let mut local_keys = [MaybeUninit::<K>::uninit(); ITEMS];
    let mut local_values = [MaybeUninit::<V>::uninit(); ITEMS];
    for_each_thread_item!(ITEMS, item, {
        keys.load_into(item, &mut local_keys[item]);
        values.load_into(item, &mut local_values[item]);
    });
    let exchange_items = block_exchange_storage_items::<THREADS, ITEMS>();
    let shared_memory = BlockSharedMemory::new(SHARED_BYTES);
    let mut shared_keys = shared_memory
        .region::<MaybeUninit<K>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<K>>());
    for_each_thread_item!(ITEMS, item, {
        let index = THREADS as i32 * item as i32 + thread_index as i32;
        shared_keys.write(index as usize, local_keys[item]);
    });
    thread::sync_threads();

    let local_diagonal = local_count.min(ITEMS as i32 * thread_index as i32);
    let partition = shared_merge_path_region(
        &shared_keys,
        0,
        first_count,
        first_count,
        second_count,
        local_diagonal,
        &order,
    );
    let mut first_cursor = partition;
    let mut second_cursor = first_count + local_diagonal - partition;
    let first_limit = first_count;
    let second_limit = local_count;
    let mut first_key = shared_keys.read(first_cursor as usize);
    let mut second_key = shared_keys.read(second_cursor as usize);
    let mut indices = [0i32; ITEMS];
    for_each_thread_item!(ITEMS, item, {
        let take_second = second_cursor < second_limit
            && (first_cursor >= first_limit
                || ordered_before(
                    &order,
                    initialized_ref(&second_key),
                    initialized_ref(&first_key),
                ));
        if take_second {
            local_keys[item] = second_key;
            indices[item] = second_cursor;
            second_cursor += 1;
            second_key = shared_keys.read(second_cursor as usize);
        } else {
            local_keys[item] = first_key;
            indices[item] = first_cursor;
            first_cursor += 1;
            first_key = shared_keys.read(first_cursor as usize);
        }
    });

    thread::sync_threads();
    let lane = thread_index & 31;
    let warp_index = thread_index >> 5;
    let warp_offset = warp_index * 32 * ITEMS as u32;
    for_each_thread_item!(ITEMS, item, {
        let blocked = warp_offset + item as u32 + lane * ITEMS as u32;
        shared_keys.write(block_exchange_index::<ITEMS>(blocked), local_keys[item]);
    });
    warp::sync_mask(u32::MAX);
    let key_store_items = if FULL {
        local_count as u32
    } else {
        let mut valid_slot =
            shared_memory.region::<u32>(exchange_items * core::mem::size_of::<MaybeUninit<K>>(), 1);
        if thread_index == 0 {
            valid_slot.write_volatile(0, local_count as u32);
        }
        thread::sync_threads();
        valid_slot.read_volatile(0)
    };
    for_each_thread_item!(ITEMS, item, {
        let warp_striped = warp_offset + item as u32 * 32 + lane;
        if FULL || warp_striped < key_store_items {
            let value = shared_keys.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
            output_keys.write_if_valid(item, value);
        }
    });

    thread::sync_threads();
    let mut shared_values = shared_memory
        .region::<MaybeUninit<V>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<V>>());
    for_each_thread_item!(ITEMS, item, {
        let index = THREADS as u32 * item as u32 + thread_index;
        shared_values.write(index as usize, local_values[item]);
    });
    thread::sync_threads();
    for_each_thread_item!(ITEMS, item, {
        local_values[item] = shared_values.read(indices[item] as usize);
    });

    thread::sync_threads();
    for_each_thread_item!(ITEMS, item, {
        let blocked = warp_offset + item as u32 + lane * ITEMS as u32;
        shared_values.write(block_exchange_index::<ITEMS>(blocked), local_values[item]);
    });
    warp::sync_mask(u32::MAX);
    let value_store_items = if FULL {
        items_in_tile as u32
    } else {
        let mut valid_slot =
            shared_memory.region::<u32>(exchange_items * core::mem::size_of::<MaybeUninit<V>>(), 1);
        if thread_index == 0 {
            valid_slot.write_volatile(0, items_in_tile as u32);
        }
        thread::sync_threads();
        valid_slot.read_volatile(0)
    };
    for_each_thread_item!(ITEMS, item, {
        let warp_striped = warp_offset + item as u32 * 32 + lane;
        if FULL || warp_striped < value_store_items {
            let value = shared_values.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
            output_values.write_if_valid(item, value);
        }
    });
}

#[inline(always)]
#[allow(unused_assignments)]
fn merge_pairs_tile<
    K,
    V,
    F,
    const THREADS: usize,
    const ITEMS: usize,
    const K_SIZE: usize,
    const V_SIZE: usize,
    const USE_LOAD_TO_SHARED_KEYS: bool,
    const USE_LOAD_TO_SHARED_VALUES: bool,
    const SHARED_BYTES: usize,
    const FULL: bool,
>(
    first: (&[K], &[V]),
    second: (&[K], &[V]),
    partitions: &[usize],
    order: F,
    tile_position: (usize, usize, usize),
    output: (
        DisjointSlice<K, LinearTiles<ITEMS>>,
        DisjointSlice<V, LinearTiles<ITEMS>>,
    ),
    tile: WarpStripedBlockTile<THREADS, ITEMS>,
) where
    K: Copy,
    V: Copy,
    F: Fn(&K, &K) -> core::cmp::Ordering + Copy,
{
    let (first_keys, first_values) = first;
    let (second_keys, second_values) = second;
    let (tile_index, tile_base, items_in_tile) = tile_position;
    let (output_keys, output_values) = output;
    let thread_index = thread::threadIdx_x();
    let tile_items = THREADS * ITEMS;
    let total = first_keys.len() + second_keys.len();
    let diagonal_first = tile_base;
    let diagonal_last = if FULL { tile_base + tile_items } else { total };
    // SAFETY: the caller-owned workspace contains one more partition than
    // launched tiles.
    let first_begin = partitions[tile_index];
    // SAFETY: same workspace proof for the closing partition.
    let first_end = partitions[tile_index + 1];
    let second_begin = diagonal_first - first_begin;
    let second_end = diagonal_last - first_end;
    let first_count = (first_end - first_begin) as i32;
    let second_count = (second_end - second_begin) as i32;

    let exchange_items = block_exchange_storage_items::<THREADS, ITEMS>();
    let shared_memory = BlockSharedMemory::new(SHARED_BYTES);
    let mut storage_keys = shared_memory
        .region::<MaybeUninit<K>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<K>>());
    let mut local_keys = [MaybeUninit::<K>::uninit(); ITEMS];
    let shared_keys;
    let keys2_offset;
    if USE_LOAD_TO_SHARED_KEYS {
        let whole_buffer_bytes =
            load_to_shared_buffer_bytes((tile_items + 1) as u32, K_SIZE as u32)
                + if K_SIZE < 16 { 32 } else { 0 };
        let second_buffer_bytes = load_to_shared_buffer_bytes(second_count as u32, K_SIZE as u32);
        let second_buffer_offset = whole_buffer_bytes - second_buffer_bytes;
        let first_copy = shared_memory.copy_async_from::<K, THREADS, K_SIZE>(
            0,
            first_keys,
            first_begin,
            first_count as u32,
            thread_index,
        );
        let second_copy = shared_memory.copy_async_from::<K, THREADS, K_SIZE>(
            second_buffer_offset as usize,
            second_keys,
            second_begin,
            second_count as u32,
            thread_index,
        );
        commit_and_wait_async_copy();
        let first_padding = first_copy.head_padding();
        keys2_offset = ((second_buffer_offset + second_copy.head_padding() - first_padding)
            / K_SIZE as u32) as i32;
        shared_keys = first_copy.into_tail_region();
    } else {
        shared_keys = shared_memory
            .region::<MaybeUninit<K>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<K>>());
        keys2_offset = first_count;
        let input = tile.blocked_concatenated_input::<K, FULL>(
            first_keys,
            first_begin,
            first_count,
            second_keys,
            second_begin,
            second_count,
        );
        for_each_thread_item!(ITEMS, item, {
            input.load_into(item, &mut local_keys[item]);
        });

        for_each_thread_item!(ITEMS, item, {
            let shared_index = (THREADS * item) as u32 + thread_index;
            storage_keys.write(shared_index as usize, local_keys[item]);
        });
        thread::sync_threads();
    }

    let thread_diagonal = if FULL {
        ITEMS as i32 * thread_index as i32
    } else {
        (ITEMS as i32 * thread_index as i32).min(items_in_tile as i32)
    };
    let local_first = shared_merge_path_region(
        &shared_keys,
        0,
        keys2_offset,
        first_count,
        second_count,
        thread_diagonal,
        &order,
    );
    let mut first_cursor = local_first;
    let mut second_cursor = keys2_offset + thread_diagonal - local_first;
    let first_limit = first_count;
    let second_limit = keys2_offset + second_count;
    let mut source_indices = [0i32; ITEMS];
    let mut first_key = shared_keys.read(first_cursor as usize);
    let mut second_key = shared_keys.read(second_cursor as usize);
    for_each_thread_item!(ITEMS, item, {
        let first_available = first_cursor < first_limit;
        let second_available = second_cursor < second_limit;
        let take_second = second_available
            && (!first_available
                || ordered_before(
                    &order,
                    initialized_ref(&second_key),
                    initialized_ref(&first_key),
                ));
        if take_second {
            local_keys[item] = second_key;
            source_indices[item] = second_cursor;
            second_cursor += 1;
            second_key = shared_keys.read(second_cursor as usize);
        } else {
            local_keys[item] = first_key;
            source_indices[item] = first_cursor;
            first_cursor += 1;
            first_key = shared_keys.read(first_cursor as usize);
        }
    });

    // Key BlockStore aliases the key merge tile.
    thread::sync_threads();
    let lane = thread_index & 31;
    let warp_index = thread_index >> 5;
    let warp_offset = warp_index * 32 * ITEMS as u32;
    for_each_thread_item!(ITEMS, item, {
        let exchange_index = warp_offset + item as u32 + lane * ITEMS as u32;
        storage_keys.write(
            block_exchange_index::<ITEMS>(exchange_index),
            local_keys[item],
        );
    });
    warp::sync_mask(u32::MAX);
    let mut output_keys = tile.output(output_keys);
    if FULL {
        for_each_thread_item!(ITEMS, item, {
            let warp_striped = warp_offset + item as u32 * 32 + lane;
            let value = storage_keys.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
            output_keys.write_if_valid(item, value);
        });
    } else {
        let mut valid_slot =
            shared_memory.region::<u32>(exchange_items * core::mem::size_of::<MaybeUninit<K>>(), 1);
        if thread_index == 0 {
            valid_slot.write_volatile(0, items_in_tile as u32);
        }
        thread::sync_threads();
        let store_items = valid_slot.read_volatile(0);
        for_each_thread_item!(ITEMS, item, {
            let warp_striped = warp_offset + item as u32 * 32 + lane;
            if warp_striped < store_items {
                let value =
                    storage_keys.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
                output_keys.write_if_valid(item, value);
            }
        });
    }

    let mut storage_values = shared_memory
        .region::<MaybeUninit<V>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<V>>());
    let mut local_values = [MaybeUninit::<V>::uninit(); ITEMS];
    let shared_values;
    if USE_LOAD_TO_SHARED_VALUES {
        // Key BlockStore aliases this storage; CUB synchronizes before issuing
        // the value copies into the same bytes.
        thread::sync_threads();
        let whole_buffer_bytes =
            load_to_shared_buffer_bytes((tile_items + 1) as u32, V_SIZE as u32)
                + if V_SIZE < 16 { 32 } else { 0 };
        let second_buffer_bytes = load_to_shared_buffer_bytes(second_count as u32, V_SIZE as u32);
        let second_buffer_offset = whole_buffer_bytes - second_buffer_bytes;
        let first_copy = shared_memory.copy_async_from::<V, THREADS, V_SIZE>(
            0,
            first_values,
            first_begin,
            first_count as u32,
            thread_index,
        );
        let second_copy = shared_memory.copy_async_from::<V, THREADS, V_SIZE>(
            second_buffer_offset as usize,
            second_values,
            second_begin,
            second_count as u32,
            thread_index,
        );
        commit_async_copy();
        let first_padding = first_copy.head_padding();
        let values2_offset = ((second_buffer_offset + second_copy.head_padding() - first_padding)
            / V_SIZE as u32) as i32;
        let offset_difference = values2_offset - keys2_offset;
        for_each_thread_item!(ITEMS, item, {
            if source_indices[item] >= keys2_offset {
                source_indices[item] += offset_difference;
            }
        });
        wait_async_copy();
        shared_values = first_copy.into_tail_region();
    } else {
        // CUB overlaps independent global value loads with completion of the
        // key stores, then synchronizes before reusing shared bytes.
        let input = tile.blocked_concatenated_input::<V, FULL>(
            first_values,
            first_begin,
            first_count,
            second_values,
            second_begin,
            second_count,
        );
        for_each_thread_item!(ITEMS, item, {
            input.load_into(item, &mut local_values[item]);
        });

        thread::sync_threads();
        shared_values = shared_memory
            .region::<MaybeUninit<V>>(0, SHARED_BYTES / core::mem::size_of::<MaybeUninit<V>>());
        if USE_LOAD_TO_SHARED_KEYS {
            let offset_difference = first_count - keys2_offset;
            for_each_thread_item!(ITEMS, item, {
                if source_indices[item] >= keys2_offset {
                    source_indices[item] += offset_difference;
                }
            });
        }
        for_each_thread_item!(ITEMS, item, {
            let shared_index = (THREADS * item) as u32 + thread_index;
            storage_values.write(shared_index as usize, local_values[item]);
        });
        thread::sync_threads();
    }

    for_each_thread_item!(ITEMS, item, {
        // Keep inactive partial outputs as MaybeUninit; the later store guard
        // is the proof before any value is interpreted.
        local_values[item] = shared_values.read(source_indices[item] as usize);
    });
    thread::sync_threads();

    for_each_thread_item!(ITEMS, item, {
        let exchange_index = warp_offset + item as u32 + lane * ITEMS as u32;
        storage_values.write(
            block_exchange_index::<ITEMS>(exchange_index),
            local_values[item],
        );
    });
    warp::sync_mask(u32::MAX);
    let mut output_values = tile.output(output_values);
    if FULL {
        for_each_thread_item!(ITEMS, item, {
            let warp_striped = warp_offset + item as u32 * 32 + lane;
            let value =
                storage_values.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
            output_values.write_if_valid(item, value);
        });
    } else {
        let mut valid_slot =
            shared_memory.region::<u32>(exchange_items * core::mem::size_of::<MaybeUninit<V>>(), 1);
        if thread_index == 0 {
            valid_slot.write_volatile(0, items_in_tile as u32);
        }
        thread::sync_threads();
        let store_items = valid_slot.read_volatile(0);
        for_each_thread_item!(ITEMS, item, {
            let warp_striped = warp_offset + item as u32 * 32 + lane;
            if warp_striped < store_items {
                let value =
                    storage_values.read_initialized(block_exchange_index::<ITEMS>(warp_striped));
                output_values.write_if_valid(item, value);
            }
        });
    }
}

/// Exact pinned `DeviceRadixSortInitKernel` body.
///
/// The two exclusive borrows are the Rust form of CUB's two disjoint temporary
/// allocations.  The grid-stride schedule is retained even though the current
/// full-width primitive policies need only one block.
#[inline(always)]
pub fn radix_init_bins_and_counters_agent(
    mut counters: DisjointSlice<i32, LinearTiles<1>>,
    mut bins: DisjointSlice<u64, LinearTiles<1>>,
    counter_thread: cuda_device::ThreadIndex32<'_>,
    bin_thread: cuda_device::ThreadIndex32<'_>,
) {
    let mut counter_runs = counters.grid_stride_runs32(counter_thread);
    let mut bin_runs = bins.grid_stride_runs32(bin_thread);
    loop {
        let mut advanced = false;
        if let Some(mut run) = counter_runs.next_run() {
            if let Some(mut slot) = run.at(0) {
                slot.write(0);
            }
            advanced = true;
        }
        if let Some(mut run) = bin_runs.next_run() {
            if let Some(mut slot) = run.at(0) {
                slot.write(0);
            }
            advanced = true;
        }
        if !advanced {
            break;
        }
    }
}

/// Exact pinned SM80+ histogram agent used before one-sweep radix passes.
///
/// `to_ordered` and `digit` are proc-macro-emitted primitive operations. This
/// keeps key representation metadata out of the runtime support crate while
/// retaining CUB's striped loads, per-block shared histogram, and device-wide
/// atomic accumulation.
#[inline(always)]
pub fn radix_histogram_agent<
    K,
    B,
    DefaultKey,
    ToOrdered,
    Digit,
    const ITEMS: usize,
    const MAX_PASSES: usize,
>(
    keys: &[K],
    begin_bit: u32,
    end_bit: u32,
    default_key: DefaultKey,
    to_ordered: ToOrdered,
    digit: Digit,
    bins: DisjointSlice<u64>,
) where
    K: Copy,
    B: Copy,
    DefaultKey: Fn() -> B + Copy,
    ToOrdered: Fn(K) -> B + Copy,
    Digit: Fn(B, u32, u32) -> u32 + Copy,
{
    const RADIX_DIGITS: usize = 256;
    let thread_index = thread::threadIdx_x() as usize;
    let block_threads = thread::blockDim_x() as usize;
    let tile_items = block_threads * ITEMS;
    let shared_memory =
        BlockSharedMemory::new(MAX_PASSES * RADIX_DIGITS * core::mem::size_of::<u32>());
    let mut shared = shared_memory.region::<u32>(0, MAX_PASSES * RADIX_DIGITS);
    let mut bins = DeviceRegion::new(bins);
    let num_passes = (end_bit - begin_bit).div_ceil(8) as usize;
    let sentinel = default_key();

    for bin in (thread_index..RADIX_DIGITS).step_by(block_threads) {
        let mut pass = 0usize;
        while pass < num_passes {
            shared.write(pass * RADIX_DIGITS + bin, 0);
            pass += 1;
        }
    }
    thread::sync_threads();

    let mut tile_offset = thread::blockIdx_x() as usize * tile_items;
    let tile_stride = thread::gridDim_x() as usize * tile_items;
    while tile_offset < keys.len() {
        let mut ordered_keys = [MaybeUninit::<B>::uninit(); ITEMS];
        for_each_thread_item!(ITEMS, item, {
            let index = tile_offset + thread_index + item * block_threads;
            // CUB fills the final striped tile with its transformed maximum
            // sentinel, then twiddles every register before any histogram
            // pass begins.
            ordered_keys[item] = MaybeUninit::new(if index < keys.len() {
                to_ordered(keys[index])
            } else {
                sentinel
            });
        });
        // `current_bit` and `pass` are ordinary agent state in CUB. Only the
        // per-thread item array is compile-time and fully unrolled.
        let mut current_bit = begin_bit;
        let mut pass = 0usize;
        while current_bit < end_bit {
            let num_bits = (end_bit - current_bit).min(8);
            for_each_thread_item!(ITEMS, item, {
                let bin = digit(
                    initialized_value(&ordered_keys[item]),
                    current_bit,
                    num_bits,
                ) as usize;
                shared.fetch_add_block(pass * RADIX_DIGITS + bin, 1);
            });
            current_bit += 8;
            pass += 1;
        }
        tile_offset += tile_stride;
    }
    thread::sync_threads();

    for bin in (thread_index..RADIX_DIGITS).step_by(block_threads) {
        let mut pass = 0usize;
        while pass < num_passes {
            let count = shared.read(pass * RADIX_DIGITS + bin);
            if count != 0 {
                bins.fetch_add_device(pass * RADIX_DIGITS + bin, count as u64);
            }
            pass += 1;
        }
    }
    thread::sync_threads();
}

#[inline(always)]
fn warp_exclusive_sum_u64(value: u64, lane: u32) -> (u64, u64) {
    let mut inclusive = value;
    let mut offset = 1u32;
    while offset < 32 {
        let preceding = warp::shuffle_up_u64_sync(u32::MAX, inclusive, offset);
        if lane >= offset {
            inclusive += preceding;
        }
        offset <<= 1;
    }
    (inclusive - value, inclusive)
}

#[inline(always)]
fn block_exclusive_sum_raking_256_u64(value: u64, grid: &mut SharedRegion<u64>) -> u64 {
    const SEGMENT_LENGTH: usize = 8;
    const RAKING_THREADS: usize = 32;
    let thread_index = thread::threadIdx_x() as usize;
    // `BlockRakingLayout<u64, 256>` pads every eight-value segment. CUB's
    // `BlockScan` temp storage starts with a one-byte NullType warp-scan
    // member, then aligns the raking grid to 16 bytes.
    let placement = thread_index + thread_index / SEGMENT_LENGTH;
    grid.write(placement, value);
    thread::sync_threads();

    if thread_index < RAKING_THREADS {
        let segment = thread_index * (SEGMENT_LENGTH + 1);
        let mut cached = [0u64; SEGMENT_LENGTH];
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            cached[item] = grid.read(segment + item);
        });
        let mut aggregate = cached[0];
        for_each_raking_item!(SEGMENT_LENGTH - 1, offset, {
            aggregate += cached[offset + 1];
        });
        let (segment_prefix, _) = warp_exclusive_sum_u64(aggregate, thread_index as u32);

        // Policy default is BLOCK_SCAN_RAKING, not memoize: the downsweep
        // deliberately reloads the segment from shared memory.
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            cached[item] = grid.read(segment + item);
        });
        let mut running = segment_prefix;
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            let current = cached[item];
            cached[item] = running;
            running += current;
        });
        let _ = running;
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            grid.write(segment + item, cached[item]);
        });
    }
    thread::sync_threads();
    grid.read(placement)
}

/// One 256-thread block per radix pass, matching
/// `DeviceRadixSortExclusiveSumKernel`'s default `BLOCK_SCAN_RAKING` launch,
/// padding, reload, and observable scan result.
#[inline(always)]
pub fn radix_exclusive_sum_agent<const PASSES: usize>(
    mut bins: DisjointSlice<u64>,
    bin_thread: cuda_device::ThreadIndex32<'_>,
) {
    let pass = thread::blockIdx_x() as usize;
    if pass >= PASSES {
        return;
    }
    const SHARED_BYTES: usize = 2_336;
    const GRID_OFFSET_BYTES: usize = 16;
    let shared_memory = BlockSharedMemory::new(SHARED_BYTES);
    let mut grid = shared_memory.region::<u64>(
        GRID_OFFSET_BYTES,
        (SHARED_BYTES - GRID_OFFSET_BYTES) / core::mem::size_of::<u64>(),
    );
    let Some(mut slot) = bins.element_thread32(bin_thread) else {
        return;
    };
    let value = slot.read();
    let block_prefix = block_exclusive_sum_raking_256_u64(value, &mut grid);
    slot.write(block_prefix);
}

#[inline(always)]
fn warp_exclusive_sum_i32(value: i32, lane: u32) -> (i32, i32) {
    let mut inclusive = value;
    let mut offset = 1u32;
    while offset < 32 {
        // cuda-oxide's typed 32-bit shuffle transports bits as `u32`; the
        // pinned CUB scan value is nevertheless C++ `int` throughout.
        let preceding = warp::shuffle_up_sync(u32::MAX, inclusive as u32, offset) as i32;
        if lane >= offset {
            inclusive += preceding;
        }
        offset <<= 1;
    }
    (inclusive - value, inclusive)
}

/// Pinned CUB `MatchAny<8>` lowering.
///
/// CUB deliberately uses one ballot per label bit because `match.any.sync`
/// is slower for this eight-bit rank policy. Express the same predicate
/// network through cuda-oxide intrinsics; no handwritten device assembly belongs in
/// RCCL's Rust implementation.
#[inline(always)]
fn radix_match_any_8(label: u32) -> u32 {
    macro_rules! matching_lanes_for_bit {
        ($bit:expr) => {{
            let bit_is_set = label & (1u32 << $bit) != 0;
            let set_lanes = warp::ballot_sync(u32::MAX, bit_is_set);
            if bit_is_set { set_lanes } else { !set_lanes }
        }};
    }
    // CUB assigns the first ballot result to `retval`, then intersects the
    // remaining seven.  Starting from `u32::MAX` is functionally equivalent,
    // but it is not the pinned instruction dependency graph.
    let mut peers = matching_lanes_for_bit!(0u32);
    peers &= matching_lanes_for_bit!(1u32);
    peers &= matching_lanes_for_bit!(2u32);
    peers &= matching_lanes_for_bit!(3u32);
    peers &= matching_lanes_for_bit!(4u32);
    peers &= matching_lanes_for_bit!(5u32);
    peers &= matching_lanes_for_bit!(6u32);
    peers &= matching_lanes_for_bit!(7u32);
    peers
}

#[inline(always)]
fn block_exclusive_sum_raking_memoize_384_i32(value: i32, grid: &mut SharedRegion<i32>) -> i32 {
    const SEGMENT_LENGTH: usize = 12;
    const RAKING_THREADS: usize = 32;
    let thread_index = thread::threadIdx_x() as usize;
    let placement = thread_index + thread_index / SEGMENT_LENGTH;
    grid.write(placement, value);
    thread::sync_threads();

    if thread_index < RAKING_THREADS {
        let segment = thread_index * (SEGMENT_LENGTH + 1);
        let mut cached = [0i32; SEGMENT_LENGTH];
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            cached[item] = grid.read(segment + item);
        });
        let mut aggregate = cached[0];
        for_each_raking_item!(SEGMENT_LENGTH - 1, offset, {
            aggregate += cached[offset + 1];
        });
        let (segment_prefix, _) = warp_exclusive_sum_i32(aggregate, thread_index as u32);

        // Policy800 selects BLOCK_SCAN_RAKING_MEMOIZE: reuse the register
        // segment loaded during the upsweep instead of rereading shared.
        let mut running = segment_prefix;
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            let current = cached[item];
            cached[item] = running;
            running += current;
        });
        let _ = running;
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            grid.write(segment + item, cached[item]);
        });
    }
    thread::sync_threads();
    grid.read(placement)
}

#[inline(always)]
fn radix_lookback_global(
    bin: usize,
    block_index: i32,
    local_count: i32,
    lookback: &mut DeviceRegion<'_, i32>,
    global_offsets: &mut SharedRegion<u64>,
) {
    const RADIX_DIGITS: usize = 256;
    const GLOBAL_MASK: i32 = 0x8000_0000u32 as i32;
    const VALUE_MASK: i32 = 0x3fff_ffff;
    let mut inclusive = local_count;
    let mut wanted = u32::MAX;
    let mut previous = block_index - 1;
    while previous >= 0 {
        let location = previous as usize * RADIX_DIGITS + bin;
        let mut value;
        loop {
            // This is CUB's anti-hoisting fence immediately before its
            // volatile lookback load. Inter-block visibility is provided by
            // the volatile global publication protocol, not CTA ownership.
            threadfence_block();
            value = lookback.read_volatile(location);
            if value != 0 {
                break;
            }
        }
        inclusive += value & VALUE_MASK;
        wanted = warp::ballot_sync(wanted, value & GLOBAL_MASK == 0);
        if value & GLOBAL_MASK != 0 {
            break;
        }
        previous -= 1;
    }
    let publication = inclusive | GLOBAL_MASK;
    lookback.write_volatile(block_index as usize * RADIX_DIGITS + bin, publication);
    let current = global_offsets.read(bin);
    global_offsets.write(bin, current + (inclusive - local_count) as u64);
    let _ = wanted;
}

/// Exact pinned SM80 `AgentRadixSortOnesweep` keys-only path.
///
/// The proc macro fixes the policy (`THREADS`, `ITEMS`) and supplies primitive
/// bit conversions. The backend owns the only raw-memory boundary required for
/// CUB's phase-aliased shared storage and device-wide atomic lookback protocol.
#[inline(always)]
pub fn radix_onesweep_keys_agent<
    K,
    B,
    DefaultKey,
    ToOrdered,
    FromOrdered,
    Digit,
    RankDigit,
    const THREADS: usize,
    const ITEMS: usize,
    const UNION_BYTES: usize,
>(
    storage: (DisjointSlice<i32>, DisjointSlice<i32>, &[u64]),
    buffers: (&[K], DisjointSlice<K>),
    pass: (usize, usize, u32, u32),
    transforms: (DefaultKey, ToOrdered, FromOrdered, Digit, RankDigit),
) where
    K: Copy,
    B: Copy,
    DefaultKey: Fn() -> B + Copy,
    ToOrdered: Fn(K) -> B + Copy,
    FromOrdered: Fn(B) -> K + Copy,
    Digit: FnMut(B, u32, u32) -> u32 + Copy,
    RankDigit: FnMut(B, u32, u32) -> u32 + Copy,
{
    let (lookback, counter, bins) = storage;
    let (input, output) = buffers;
    let (bin_offset, counter_offset, current_bit, num_bits) = pass;
    let (default_key, to_ordered, from_ordered, mut digit, mut rank_digit) = transforms;
    const RADIX_DIGITS: usize = 256;
    let thread_index = thread::threadIdx_x();
    let lane = thread_index & 31;
    let warp_index = thread_index >> 5;
    let tile_items = (THREADS * ITEMS) as u32;
    let num_items = input.len() as u32;
    let shared_bytes = UNION_BYTES + RADIX_DIGITS * core::mem::size_of::<u64>();
    let warp_offsets_bytes = (THREADS / 32) * RADIX_DIGITS * core::mem::size_of::<u32>();
    let prefix_storage_offset = warp_offsets_bytes + RADIX_DIGITS * core::mem::size_of::<u32>();
    let prefix_grid_offset = prefix_storage_offset + 16;
    let shared_memory = BlockSharedMemory::new(shared_bytes);
    let mut warp_offsets = shared_memory.region::<i32>(0, (THREADS / 32) * RADIX_DIGITS);
    // CUB retains one guard allocation of `match_masks[1][RADIX_DIGITS]`
    // even for WARP_MATCH_ANY. Use that otherwise-idle region to lower the
    // block-vote result without aliasing the following `prefix_tmp` storage.
    let mut short_flags = shared_memory.region::<u32>(warp_offsets_bytes, RADIX_DIGITS);
    let mut prefix_grid = shared_memory.region::<i32>(
        prefix_grid_offset,
        (UNION_BYTES - prefix_grid_offset) / core::mem::size_of::<i32>(),
    );
    let mut keys_shared = shared_memory
        .region::<MaybeUninit<B>>(0, UNION_BYTES / core::mem::size_of::<MaybeUninit<B>>());
    let mut global_offsets = shared_memory.region::<u64>(UNION_BYTES, RADIX_DIGITS);
    let mut block_index_slot = shared_memory.region::<i32>(UNION_BYTES, 1);
    let mut lookback = DeviceRegion::new(lookback);
    let mut counter = DeviceRegion::new(counter);
    let mut output = DeviceRegion::new(output);
    let sentinel = default_key();
    // The pass offset is uniform for the launch. Prove its complete 256-bin
    // band once instead of retaining a bounds check at each callback read.
    let Some(pass_bins) = bins.get(bin_offset..bin_offset + RADIX_DIGITS) else {
        return;
    };

    if thread_index == 0 {
        let dynamic_block = counter.fetch_add_device(counter_offset, 1);
        // CUB aliases `block_idx` with `global_offsets` and broadcasts it with
        // a block barrier. Retain that phase change instead of assuming CUDA's
        // physical block order is the one-sweep tile order.
        block_index_slot.write(0, dynamic_block);
    }
    thread::sync_threads();
    let block_index = block_index_slot.read(0);
    if thread_index == 0 {
        // cuda-oxide does not yet expose CUDA's `__syncthreads_or`. Reserve
        // one cell in CUB's guard match-mask allocation; the histogram's
        // later block barrier publishes this initialization before the vote.
        short_flags.write(0, 0);
    }
    let tile_base = block_index as u32 * tile_items;
    let full_block = tile_base + tile_items <= num_items;
    let valid_items = if full_block {
        tile_items
    } else {
        num_items - tile_base
    };

    let mut keys = [sentinel; ITEMS];
    let warp_base = warp_index * 32 * ITEMS as u32 + lane;
    let warp_group_base = tile_base as usize + warp_index as usize * 32usize * ITEMS;
    if full_block {
        // Prove the complete warp-striped input band once.  The prior
        // full-tile predicate guarantees this range for every warp; matching
        // the checked `Option` keeps the proof in safe Rust and avoids 21
        // duplicated slice bounds checks in the unrolled load.
        let Some(warp_group) = input.get(warp_group_base..warp_group_base + 32usize * ITEMS) else {
            return;
        };
        for_each_thread_item!(ITEMS, item, {
            let source_index = lane as usize + item * 32usize;
            keys[item] = to_ordered(warp_group[source_index]);
        });
    } else {
        // Only the final tile is partial. A warp whose band starts beyond the
        // input receives an empty slice; `get` is then exactly CUB's guarded
        // warp-striped load and supplies the default key for invalid lanes.
        let empty_input = &input[..0];
        let warp_group = input.get(warp_group_base..).unwrap_or(empty_input);
        for_each_thread_item!(ITEMS, item, {
            let source_index = lane as usize + item * 32usize;
            keys[item] = warp_group
                .get(source_index)
                .copied()
                .map(to_ordered)
                .unwrap_or(sentinel);
        });
    }

    // One private warp histogram, exactly `NUM_PARTS == 1` for the pinned
    // SM80 policy. The warp-strided input arrangement preserves source order.
    for_each_radix_warp_bin!(slot, {
        let bin = (lane + slot * 32) as usize;
        warp_offsets.write(warp_index as usize * RADIX_DIGITS + bin, 0);
    });
    warp::sync_mask(u32::MAX);
    for_each_thread_item!(ITEMS, item, {
        warp_offsets.fetch_add_block(
            warp_index as usize * RADIX_DIGITS + digit(keys[item], current_bit, num_bits) as usize,
            1,
        );
    });
    thread::sync_threads();

    let active_bin = thread_index < RADIX_DIGITS as u32;
    let bin = thread_index.min(RADIX_DIGITS as u32 - 1) as usize;
    let mut local_count = 0i32;
    if active_bin {
        for_each_policy800_warp!(source_warp, {
            let location = source_warp * RADIX_DIGITS + bin;
            let count = warp_offsets.read(location);
            warp_offsets.write(location, local_count);
            local_count += count;
        });
        let partial = local_count | 0x4000_0000;
        lookback.write_volatile(block_index as usize * RADIX_DIGITS + bin, partial);
    }

    // CUB invokes `__syncthreads_or` in the lookback callback. At most one
    // digit can own the whole tile, so its bin thread uniquely publishes the
    // true case into the otherwise-idle guard match-mask cell.
    if active_bin && local_count == tile_items as i32 {
        short_flags.write(0, 1);
    }
    thread::sync_threads();
    // A same-address shared load is a warp multicast. Every thread consumes
    // the block-wide result directly, which is the safe cuda-oxide capability
    // replacement for CUB's currently unavailable `__syncthreads_or` result.
    let short_circuit = short_flags.read(0) != 0;

    // CUB performs the short-circuit from its counts callback, before the
    // block-wide exclusive scan and rank downsweep. A uniform digit tile must
    // therefore never execute those later phases merely because the result
    // would still be correct.
    if short_circuit {
        let common_bin = digit(keys[0], current_bit, num_bits) as usize;
        if active_bin {
            global_offsets.write(
                bin,
                pass_bins[bin]
                    - if bin > common_bin {
                        tile_items as u64
                    } else {
                        0
                    },
            );
            radix_lookback_global(
                bin,
                block_index,
                local_count,
                &mut lookback,
                &mut global_offsets,
            );
        }
        thread::sync_threads();
        let output_base = global_offsets.read(common_bin) as usize;
        if full_block {
            for_each_thread_item!(ITEMS, item, {
                let local_index = warp_base + item as u32 * 32;
                let value = from_ordered(keys[item]);
                output.write(output_base + local_index as usize, value);
            });
        } else {
            for_each_thread_item!(ITEMS, item, {
                let local_index = warp_base + item as u32 * 32;
                if local_index < valid_items {
                    let value = from_ordered(keys[item]);
                    output.write(output_base + local_index as usize, value);
                }
            });
        }
        return;
    }

    let exclusive_digit_prefix =
        block_exclusive_sum_raking_memoize_384_i32(local_count, &mut prefix_grid);
    if active_bin {
        for_each_policy800_warp!(source_warp, {
            let location = source_warp * RADIX_DIGITS + bin;
            let prior_warps = warp_offsets.read(location);
            warp_offsets.write(location, prior_warps + exclusive_digit_prefix);
        });
    }
    thread::sync_threads();

    // CUB's register array is uninitialized and every fully-unrolled rank
    // step writes its slot before the scatter reads it. Initializing all 21
    // registers here would hide an unnecessary instruction path behind the
    // same functional result.
    let mut ranks = [0i32; ITEMS];
    for_each_thread_item!(ITEMS, item, {
        let item_bin = rank_digit(keys[item], current_bit, num_bits);
        let matching = radix_match_any_8(item_bin);
        let leader = 31 - matching.leading_zeros();
        let within_match = (matching & warp::lanemask_le()).count_ones() as i32;
        let mut warp_offset = 0i32;
        if lane == leader {
            warp_offset = warp_offsets.fetch_add_block(
                warp_index as usize * RADIX_DIGITS + item_bin as usize,
                within_match,
            );
        }
        warp_offset = warp::shuffle_sync(u32::MAX, warp_offset as u32, leader) as i32;
        ranks[item] = warp_offset + within_match - 1;
    });

    // `rank_temp_storage` and `keys_out` are the same CUB union. Every warp
    // must finish consuming its rank offsets before any thread starts the key
    // scatter phase and overwrites that storage.
    thread::sync_threads();

    for_each_thread_item!(ITEMS, item, {
        let ordered = keys[item];
        let rank = ranks[item];
        keys_shared.write(rank as usize, MaybeUninit::new(ordered));
    });

    // CUB overlaps independent key-scatter and lookback work here. The single
    // barrier after lookback publishes both keys and global offsets before
    // any thread reads either; inserting an earlier barrier serializes the
    // phases and changes the one-sweep schedule.
    if active_bin {
        global_offsets.write(bin, pass_bins[bin] - exclusive_digit_prefix as u64);
        radix_lookback_global(
            bin,
            block_index,
            local_count,
            &mut lookback,
            &mut global_offsets,
        );
    }
    thread::sync_threads();

    if full_block {
        for_each_thread_item!(ITEMS, item, {
            let blocked_index = thread_index + item as u32 * THREADS as u32;
            let ordered = keys_shared.read_initialized(blocked_index as usize);
            let item_bin = digit(ordered, current_bit, num_bits) as usize;
            let output_index = blocked_index as usize + global_offsets.read(item_bin) as usize;
            output.write(output_index, from_ordered(ordered));
            warp::sync_mask(u32::MAX);
        });
    } else {
        for_each_thread_item!(ITEMS, item, {
            let blocked_index = thread_index + item as u32 * THREADS as u32;
            if blocked_index < valid_items {
                let ordered = keys_shared.read_initialized(blocked_index as usize);
                let item_bin = digit(ordered, current_bit, num_bits) as usize;
                let output_index = blocked_index as usize + global_offsets.read(item_bin) as usize;
                output.write(output_index, from_ordered(ordered));
            }
            warp::sync_mask(u32::MAX);
        });
    }
}

/// Exact pinned SM80 `AgentRadixSortOnesweep` key/value path.
///
/// Values remain `MaybeUninit` for lanes outside the final partial tile. This
/// is the Rust representation of CUB's guarded value load: invalid lanes may
/// participate in the shared permutation, but are never read as `V` or stored
/// to the output range.
#[inline(always)]
pub fn radix_onesweep_pairs_agent<
    K,
    V,
    B,
    DefaultKey,
    ToOrdered,
    FromOrdered,
    Digit,
    RankDigit,
    const THREADS: usize,
    const ITEMS: usize,
    const UNION_BYTES: usize,
>(
    storage: (DisjointSlice<i32>, DisjointSlice<i32>, &[u64]),
    buffers: (&[K], &[V], DisjointSlice<K>, DisjointSlice<V>),
    pass: (usize, usize, u32, u32),
    transforms: (DefaultKey, ToOrdered, FromOrdered, Digit, RankDigit),
) where
    K: Copy,
    V: Copy,
    B: Copy,
    DefaultKey: Fn() -> B + Copy,
    ToOrdered: Fn(K) -> B + Copy,
    FromOrdered: Fn(B) -> K + Copy,
    Digit: FnMut(B, u32, u32) -> u32 + Copy,
    RankDigit: FnMut(B, u32, u32) -> u32 + Copy,
{
    let (lookback, counter, bins) = storage;
    let (input_keys, input_values, output_keys, output_values) = buffers;
    let (bin_offset, counter_offset, current_bit, num_bits) = pass;
    let (default_key, to_ordered, from_ordered, mut digit, mut rank_digit) = transforms;
    const RADIX_DIGITS: usize = 256;
    let thread_index = thread::threadIdx_x();
    let lane = thread_index & 31;
    let warp_index = thread_index >> 5;
    let tile_items = (THREADS * ITEMS) as u32;
    let num_items = input_keys.len() as u32;
    let shared_bytes = UNION_BYTES + RADIX_DIGITS * core::mem::size_of::<u64>();
    let warp_offsets_bytes = (THREADS / 32) * RADIX_DIGITS * core::mem::size_of::<u32>();
    let prefix_storage_offset = warp_offsets_bytes + RADIX_DIGITS * core::mem::size_of::<u32>();
    let prefix_grid_offset = prefix_storage_offset + 16;
    let shared_memory = BlockSharedMemory::new(shared_bytes);
    let mut warp_offsets = shared_memory.region::<i32>(0, (THREADS / 32) * RADIX_DIGITS);
    let mut short_flags = shared_memory.region::<u32>(warp_offsets_bytes, RADIX_DIGITS);
    let mut prefix_grid = shared_memory.region::<i32>(
        prefix_grid_offset,
        (UNION_BYTES - prefix_grid_offset) / core::mem::size_of::<i32>(),
    );
    let mut keys_shared = shared_memory
        .region::<MaybeUninit<B>>(0, UNION_BYTES / core::mem::size_of::<MaybeUninit<B>>());
    let mut values_shared = shared_memory
        .region::<MaybeUninit<V>>(0, UNION_BYTES / core::mem::size_of::<MaybeUninit<V>>());
    let mut global_offsets = shared_memory.region::<u64>(UNION_BYTES, RADIX_DIGITS);
    let mut block_index_slot = shared_memory.region::<i32>(UNION_BYTES, 1);
    let mut lookback = DeviceRegion::new(lookback);
    let mut counter = DeviceRegion::new(counter);
    let mut output_keys = DeviceRegion::new(output_keys);
    let mut output_values = DeviceRegion::new(output_values);
    let sentinel = default_key();
    let Some(pass_bins) = bins.get(bin_offset..bin_offset + RADIX_DIGITS) else {
        return;
    };

    if thread_index == 0 {
        let dynamic_block = counter.fetch_add_device(counter_offset, 1);
        block_index_slot.write(0, dynamic_block);
    }
    thread::sync_threads();
    let block_index = block_index_slot.read(0);
    if thread_index == 0 {
        short_flags.write(0, 0);
    }
    let tile_base = block_index as u32 * tile_items;
    let full_block = tile_base + tile_items <= num_items;
    let valid_items = if full_block {
        tile_items
    } else {
        num_items - tile_base
    };

    let mut keys = [sentinel; ITEMS];
    let warp_base = warp_index * 32 * ITEMS as u32 + lane;
    let warp_group_base = tile_base as usize + warp_index as usize * 32usize * ITEMS;
    if full_block {
        let Some(warp_group) = input_keys.get(warp_group_base..warp_group_base + 32usize * ITEMS)
        else {
            return;
        };
        for_each_thread_item!(ITEMS, item, {
            keys[item] = to_ordered(warp_group[lane as usize + item * 32usize]);
        });
    } else {
        let empty_input = &input_keys[..0];
        let warp_group = input_keys.get(warp_group_base..).unwrap_or(empty_input);
        for_each_thread_item!(ITEMS, item, {
            keys[item] = warp_group
                .get(lane as usize + item * 32usize)
                .copied()
                .map(to_ordered)
                .unwrap_or(sentinel);
        });
    }

    for_each_radix_warp_bin!(slot, {
        let bin = (lane + slot * 32) as usize;
        warp_offsets.write(warp_index as usize * RADIX_DIGITS + bin, 0);
    });
    warp::sync_mask(u32::MAX);
    for_each_thread_item!(ITEMS, item, {
        warp_offsets.fetch_add_block(
            warp_index as usize * RADIX_DIGITS + digit(keys[item], current_bit, num_bits) as usize,
            1,
        );
    });
    thread::sync_threads();

    let active_bin = thread_index < RADIX_DIGITS as u32;
    let bin = thread_index.min(RADIX_DIGITS as u32 - 1) as usize;
    let mut local_count = 0i32;
    if active_bin {
        for_each_policy800_warp!(source_warp, {
            let location = source_warp * RADIX_DIGITS + bin;
            let count = warp_offsets.read(location);
            warp_offsets.write(location, local_count);
            local_count += count;
        });
        lookback.write_volatile(
            block_index as usize * RADIX_DIGITS + bin,
            local_count | 0x4000_0000,
        );
    }
    if active_bin && local_count == tile_items as i32 {
        short_flags.write(0, 1);
    }
    thread::sync_threads();
    let short_circuit = short_flags.read(0) != 0;

    if short_circuit {
        let common_bin = digit(keys[0], current_bit, num_bits) as usize;
        if active_bin {
            global_offsets.write(
                bin,
                pass_bins[bin]
                    - if bin > common_bin {
                        tile_items as u64
                    } else {
                        0
                    },
            );
            radix_lookback_global(
                bin,
                block_index,
                local_count,
                &mut lookback,
                &mut global_offsets,
            );
        }
        thread::sync_threads();
        let output_base = global_offsets.read(common_bin) as usize;
        for_each_thread_item!(ITEMS, item, {
            let local_index = warp_base + item as u32 * 32;
            if full_block || local_index < valid_items {
                output_keys.write(output_base + local_index as usize, from_ordered(keys[item]));
            }
        });

        let mut values = [MaybeUninit::<V>::uninit(); ITEMS];
        let empty_values = &input_values[..0];
        let value_group = if full_block {
            let Some(group) = input_values.get(warp_group_base..warp_group_base + 32usize * ITEMS)
            else {
                return;
            };
            group
        } else {
            input_values.get(warp_group_base..).unwrap_or(empty_values)
        };
        for_each_thread_item!(ITEMS, item, {
            if let Some(value) = value_group.get(lane as usize + item * 32usize).copied() {
                values[item].write(value);
            }
        });
        for_each_thread_item!(ITEMS, item, {
            let local_index = warp_base + item as u32 * 32;
            if full_block || local_index < valid_items {
                output_values.write(
                    output_base + local_index as usize,
                    initialized_value(&values[item]),
                );
            }
        });
        return;
    }

    let exclusive_digit_prefix =
        block_exclusive_sum_raking_memoize_384_i32(local_count, &mut prefix_grid);
    if active_bin {
        for_each_policy800_warp!(source_warp, {
            let location = source_warp * RADIX_DIGITS + bin;
            let prior_warps = warp_offsets.read(location);
            warp_offsets.write(location, prior_warps + exclusive_digit_prefix);
        });
    }
    thread::sync_threads();

    let mut ranks = [0i32; ITEMS];
    for_each_thread_item!(ITEMS, item, {
        let item_bin = rank_digit(keys[item], current_bit, num_bits);
        let matching = radix_match_any_8(item_bin);
        let leader = 31 - matching.leading_zeros();
        let within_match = (matching & warp::lanemask_le()).count_ones() as i32;
        let mut warp_offset = 0i32;
        if lane == leader {
            warp_offset = warp_offsets.fetch_add_block(
                warp_index as usize * RADIX_DIGITS + item_bin as usize,
                within_match,
            );
        }
        warp_offset = warp::shuffle_sync(u32::MAX, warp_offset as u32, leader) as i32;
        ranks[item] = warp_offset + within_match - 1;
    });
    thread::sync_threads();

    for_each_thread_item!(ITEMS, item, {
        keys_shared.write(ranks[item] as usize, MaybeUninit::new(keys[item]));
    });

    // Match CUB's overlap: bin threads may perform lookback after their key
    // scatter while the rest of the block finishes scattering. The following
    // barrier is the sole join before global key reads.
    if active_bin {
        global_offsets.write(bin, pass_bins[bin] - exclusive_digit_prefix as u64);
        radix_lookback_global(
            bin,
            block_index,
            local_count,
            &mut lookback,
            &mut global_offsets,
        );
    }
    thread::sync_threads();

    if full_block {
        for_each_thread_item!(ITEMS, item, {
            let blocked_index = thread_index + item as u32 * THREADS as u32;
            let ordered = keys_shared.read_initialized(blocked_index as usize);
            let item_bin = digit(ordered, current_bit, num_bits) as usize;
            let output_index = blocked_index as usize + global_offsets.read(item_bin) as usize;
            output_keys.write(output_index, from_ordered(ordered));
            warp::sync_mask(u32::MAX);
        });
    } else {
        for_each_thread_item!(ITEMS, item, {
            let blocked_index = thread_index + item as u32 * THREADS as u32;
            if blocked_index < valid_items {
                let ordered = keys_shared.read_initialized(blocked_index as usize);
                let item_bin = digit(ordered, current_bit, num_bits) as usize;
                let output_index = blocked_index as usize + global_offsets.read(item_bin) as usize;
                output_keys.write(output_index, from_ordered(ordered));
            }
            warp::sync_mask(u32::MAX);
        });
    }

    let mut value_digits = [0u32; ITEMS];
    for_each_thread_item!(ITEMS, item, {
        let blocked_index = thread_index + item as u32 * THREADS as u32;
        let ordered = keys_shared.read_initialized(blocked_index as usize);
        value_digits[item] = digit(ordered, current_bit, num_bits);
    });

    let mut values = [MaybeUninit::<V>::uninit(); ITEMS];
    let empty_values = &input_values[..0];
    let value_group = if full_block {
        let Some(group) = input_values.get(warp_group_base..warp_group_base + 32usize * ITEMS)
        else {
            return;
        };
        group
    } else {
        input_values.get(warp_group_base..).unwrap_or(empty_values)
    };
    for_each_thread_item!(ITEMS, item, {
        if let Some(value) = value_group.get(lane as usize + item * 32usize).copied() {
            values[item].write(value);
        }
    });
    thread::sync_threads();
    for_each_thread_item!(ITEMS, item, {
        values_shared.write(ranks[item] as usize, values[item]);
    });
    thread::sync_threads();

    for_each_thread_item!(ITEMS, item, {
        let blocked_index = thread_index + item as u32 * THREADS as u32;
        if full_block || blocked_index < valid_items {
            let value = values_shared.read_initialized(blocked_index as usize);
            let output_index =
                blocked_index as usize + global_offsets.read(value_digits[item] as usize) as usize;
            output_values.write(output_index, value);
        }
        warp::sync_mask(u32::MAX);
    });
}

#[cfg(test)]
mod tests {
    use super::{
        block_exchange_index, block_exchange_storage_items, load_to_shared_buffer_bytes,
        local_merge_path_begin,
    };

    #[test]
    fn signed_local_merge_path_clamps_to_zero_like_cub() {
        assert_eq!(local_merge_path_begin(0, 17), 0);
        assert_eq!(local_merge_path_begin(8, 17), 0);
        assert_eq!(local_merge_path_begin(17, 17), 0);
        assert_eq!(local_merge_path_begin(23, 17), 6);
    }

    #[test]
    fn global_merge_path_delegates_range_proof_without_recovering_indexes() {
        let source = include_str!("ordering_backend.rs");
        let begin = source
            .find("pub fn global_merge_path")
            .expect("global MergePath wrapper must exist");
        let end = source[begin..]
            .find("/// CUB `MergePath` over the two adjacent ranges")
            .map(|offset| begin + offset)
            .expect("local MergePath helper must follow the global wrapper");
        let merge_path = &source[begin..end];

        for required in [
            "MergePathSearch::new(first, second, diagonal)",
            "search.step(|first_key, second_key|",
            "search.partition()",
        ] {
            assert!(
                merge_path.contains(required),
                "global MergePath lost its affine range capability: {required}"
            );
        }
        for forbidden in [
            "first[".to_owned(),
            "second[".to_owned(),
            "get_unchecked".to_owned(),
            ["un", "safe"].concat(),
        ] {
            assert!(
                !merge_path.contains(&forbidden),
                "global MergePath must not recreate C++ iterator arithmetic: {forbidden}"
            );
        }
    }

    #[test]
    fn policy800_rank_keeps_the_pinned_ballot_loop_without_asm_shortcuts() {
        let source = include_str!("ordering_backend.rs");
        let begin = source
            .find("fn radix_match_any_8")
            .expect("Policy800 match-any helper must exist");
        let end = source[begin..]
            .find("fn block_exclusive_sum_raking_memoize_384_i32")
            .map(|offset| begin + offset)
            .expect("Policy800 block scan must follow match-any");
        let match_any = &source[begin..end];
        assert!(match_any.contains("warp::ballot_sync"));
        assert!(match_any.contains("label & (1u32 << $bit) != 0"));
        assert!(match_any.contains("if bit_is_set"));
        assert!(match_any.contains("!set_lanes"));
        assert!(match_any.contains("let mut peers = matching_lanes_for_bit!(0u32)"));
        assert_eq!(
            match_any
                .matches("peers &= matching_lanes_for_bit!")
                .count(),
            7
        );
        for forbidden in [
            ["warp::", "match_any_sync"].concat(),
            "polarity".to_owned(),
            "as i32 >> 31".to_owned(),
            "wrapping_sub(1u32)".to_owned(),
            ["p", "tx_", "a", "sm", "!"].concat(),
            ["a", "sm", "!"].concat(),
        ] {
            assert!(
                !match_any.contains(&forbidden),
                "forbidden shortcut: {forbidden}"
            );
        }
    }

    #[test]
    fn policy800_onesweep_keeps_signed_counts_and_checked_slice_bands() {
        let source = include_str!("ordering_backend.rs");
        let begin = source
            .find("pub fn radix_onesweep_keys_agent")
            .expect("Policy800 one-sweep agent must exist");
        let end = source[begin..]
            .find("#[cfg(test)]")
            .map(|offset| begin + offset)
            .expect("backend tests must follow one-sweep");
        let agent = &source[begin..end];

        for required in [
            "let num_items = input.len() as u32",
            "let mut warp_offsets =",
            "let mut local_count = 0i32",
            "let mut ranks = [0i32; ITEMS]",
            "bins.get(bin_offset..bin_offset + RADIX_DIGITS)",
            "input.get(warp_group_base..warp_group_base + 32usize * ITEMS)",
            ".get(source_index)",
        ] {
            assert!(
                agent.contains(required),
                "missing one-sweep contract: {required}"
            );
        }
        for forbidden in ["num_items: u32", "input[source_index", "bins[bin_offset"] {
            assert!(
                !agent.contains(forbidden),
                "one-sweep must not duplicate a slice extent or its bounds checks: {forbidden}"
            );
        }

        let constructor_begin = agent
            .find("let dynamic_block = counter.fetch_add_device")
            .expect("the dynamic block-index publication must remain explicit");
        let constructor_end = agent[constructor_begin..]
            .find("let block_index =")
            .map(|offset| constructor_begin + offset)
            .expect("the published block index must be consumed");
        assert_eq!(
            agent[constructor_begin..constructor_end]
                .matches("thread::sync_threads()")
                .count(),
            1,
            "the pinned AgentRadixSortOnesweep constructor has exactly one CTA barrier"
        );

        for scatter_marker in ["MaybeUninit::new(ordered)", "MaybeUninit::new(keys[item])"] {
            let scatter = agent
                .find(scatter_marker)
                .expect("both one-sweep agents must scatter keys into their shared union");
            let after_scatter = &agent[scatter + scatter_marker.len()..];
            let lookback = after_scatter
                .find("if active_bin")
                .expect("lookback must follow the shared key scatter");
            assert!(
                !after_scatter[..lookback].contains("thread::sync_threads()"),
                "an extra barrier must not serialize CUB's key-scatter/lookback overlap"
            );
        }
    }

    #[test]
    fn radix_histogram_keeps_runtime_pass_state_and_compile_time_item_extent() {
        let source = include_str!("ordering_backend.rs");
        let begin = source
            .find("pub fn radix_histogram_agent")
            .expect("radix histogram agent must exist");
        let end = source[begin..]
            .find("fn warp_exclusive_sum_u64")
            .map(|offset| begin + offset)
            .expect("exclusive-sum helper must follow the histogram agent");
        let histogram = &source[begin..end];

        for required in [
            "const MAX_PASSES: usize",
            "let num_passes = (end_bit - begin_bit).div_ceil(8) as usize",
            "while current_bit < end_bit",
            "for_each_thread_item!(ITEMS, item",
        ] {
            assert!(
                histogram.contains(required),
                "missing pinned histogram staging contract: {required}"
            );
        }
        assert_eq!(
            histogram.matches("while pass < num_passes").count(),
            2,
            "initialization and global accumulation traverse runtime pass state"
        );
        assert!(
            !histogram.contains("for_each_radix_pass"),
            "the pinned histogram pass traversal is runtime agent state, not TMP expansion"
        );
    }

    #[test]
    fn serial_merge_does_not_duplicate_cursor_state_as_raw_pointers() {
        let source = include_str!("ordering_backend.rs");
        let tests_begin = source
            .find("#[cfg(test)]")
            .expect("backend tests must follow the implementation");
        let implementation = &source[..tests_begin];
        for forbidden in [
            ["let mut first_", "pointer"].concat(),
            ["let mut second_", "pointer"].concat(),
            ["select_", "unpredictable"].concat(),
        ] {
            assert!(
                !implementation.contains(&forbidden),
                "serial merge must keep the measured Rust cursor/control-flow form: {forbidden}"
            );
        }
        for required in [
            "let oob_default = local[0]",
            "let oob_default = local_keys[0]",
            "let mut first_key = if first_count != 0",
            "let mut second_key = if second_count != 0",
            "local_keys[item] = second_key",
            "local_keys[item] = first_key",
            "indices[item] = second_cursor",
            "source_indices[item] = second_cursor",
        ] {
            assert!(
                implementation.contains(required),
                "BlockMergeSort serial merge must retain CUB's typed conditional head: {required}"
            );
        }
        assert!(
            !implementation.contains("local_keys[item] = if take_second"),
            "Rust pair serial merge must not keep selected keys and indices live together"
        );
    }

    #[test]
    fn pair_compare_exchange_uses_one_borrow_safe_selection_predicate() {
        let source = include_str!("ordering_backend.rs");
        let begin = source
            .find("fn stable_odd_even_sort_pairs")
            .expect("pair thread sort must exist");
        let end = source[begin..]
            .find("fn block_merge_round")
            .map(|offset| begin + offset)
            .expect("block merge must follow the thread sort");
        let pair_sort = &source[begin..end];

        for required in [
            "let exchange = ordered_before",
            "keys[left] = if exchange",
            "keys[right] = if exchange",
            "values[left] = if exchange",
            "values[right] = if exchange",
        ] {
            assert!(
                pair_sort.contains(required),
                "stable pair compare-exchange lost its single-predicate selection form: {required}"
            );
        }
        for forbidden in ["temporary_key", "temporary_value"] {
            assert!(
                !pair_sort.contains(forbidden),
                "sequential swap mutation regresses pair block-sort codegen: {forbidden}"
            );
        }
    }

    #[test]
    fn merge_sort_agents_keep_cccl_tile_and_offset_predicates() {
        let source = include_str!("ordering_backend.rs");
        let keys_begin = source
            .find("pub fn merge_sort_merge_keys_agent")
            .expect("keys merge agent must exist");
        let pairs_begin = source
            .find("pub fn merge_sort_merge_pairs_agent")
            .expect("pair merge agent must exist");
        let pairs_end = source[pairs_begin..]
            .find("fn merge_sort_merge_pairs_tile")
            .map(|offset| pairs_begin + offset)
            .expect("pair merge tile must follow its agent");
        let keys_agent = &source[keys_begin..pairs_begin];
        let pairs_agent = &source[pairs_begin..pairs_end];
        for agent in [keys_agent, pairs_agent] {
            assert!(agent.contains("if items_in_tile == tile_items"));
            assert!(
                !agent.contains("gridDim_x"),
                "AgentMerge selects its specialization from the actual remaining count"
            );
        }

        let partition_begin = source
            .find("pub fn merge_sort_partition_point")
            .expect("partition agent must exist");
        let partition = &source[partition_begin..keys_begin];
        assert!(!partition.contains("saturating_mul"));
        assert!(partition.contains("start.saturating_add(size)"));
        assert!(partition.contains("second_begin.saturating_add(size)"));
    }

    #[test]
    fn merge_sort_agents_use_checked_concatenation_without_invalid_empty_ranges() {
        let source = include_str!("ordering_backend.rs");
        let keys_begin = source
            .find("fn merge_sort_merge_keys_tile")
            .expect("keys merge tile must exist");
        let pairs_begin = source
            .find("fn merge_sort_merge_pairs_tile")
            .expect("pair merge tile must exist");
        let pairs_end = source[pairs_begin..]
            .find("fn merge_pairs_tile")
            .map(|offset| pairs_begin + offset)
            .expect("DeviceMerge must follow merge-sort agents");
        let keys = &source[keys_begin..pairs_begin];
        let pairs = &source[pairs_begin..pairs_end];

        for required in [
            "output.blocked_concatenated_input::<K, FULL>(",
            "start + first_begin",
            "start + size + second_begin",
        ] {
            assert!(
                keys.contains(required),
                "keys AgentMerge must preserve CUB's concatenated input boundary: {required}"
            );
        }
        for required in [
            "output_keys.blocked_concatenated_input::<K, FULL>(",
            "output_keys.blocked_concatenated_input::<V, FULL>(",
        ] {
            assert!(
                pairs.contains(required),
                "pair AgentMerge must preserve CUB's concatenated input boundary: {required}"
            );
        }
        for agent in [keys, pairs] {
            assert!(
                agent.contains(".load_into(item, &mut local"),
                "AgentMerge must load directly into its MaybeUninit register slot"
            );
            assert!(
                !agent.contains("let source_index = start +"),
                "an unrolled AgentMerge load must not rebuild the full-slice address for every item"
            );
            assert!(
                !agent.contains("&source[start + size + second_begin..]"),
                "a zero-count CUB iterator must not become an out-of-bounds Rust slice"
            );
            assert!(
                !agent.contains("get_unchecked"),
                "AgentMerge must consume the safe input capability instead of recovering unchecked indexing"
            );
        }
    }

    #[test]
    fn device_merge_register_loads_use_the_same_checked_concatenation() {
        let source = include_str!("ordering_backend.rs");
        let keys_begin = source
            .find("fn merge_keys_tile")
            .expect("DeviceMerge keys tile must exist");
        let pairs_agent = source[keys_begin..]
            .find("pub fn merge_pairs_agent")
            .map(|offset| keys_begin + offset)
            .expect("DeviceMerge pair agent must follow keys");
        let pairs_begin = source[pairs_agent..]
            .find("fn merge_pairs_tile")
            .map(|offset| pairs_agent + offset)
            .expect("DeviceMerge pair tile must exist");
        let pairs_end = source[pairs_begin..]
            .find("pub fn radix_init_bins_and_counters_agent")
            .map(|offset| pairs_begin + offset)
            .expect("radix agents must follow DeviceMerge");
        let keys = &source[keys_begin..pairs_agent];
        let pairs = &source[pairs_begin..pairs_end];

        assert!(keys.contains("output.blocked_concatenated_input::<K, FULL>("));
        assert_eq!(
            pairs
                .matches("tile.blocked_concatenated_input::<K, FULL>(")
                .count(),
            1
        );
        assert_eq!(
            pairs
                .matches("tile.blocked_concatenated_input::<V, FULL>(")
                .count(),
            1
        );
        assert_eq!(pairs.matches("tile.output(").count(), 2);
        for backend in [keys, pairs] {
            assert!(
                backend.contains(".load_into(item, &mut local"),
                "DeviceMerge must not return an Option for every unrolled register load"
            );
            for forbidden in [
                "first[first_begin +",
                "second[second_begin +",
                "first_keys[first_begin +",
                "second_keys[second_begin +",
                "first_values[first_begin +",
                "second_values[second_begin +",
                "get_unchecked",
            ] {
                assert!(
                    !backend.contains(forbidden),
                    "DeviceMerge must not bypass its checked concatenated input: {forbidden}"
                );
            }
        }
    }

    #[test]
    fn block_sort_global_ranges_use_affine_capabilities_without_pointer_recovery() {
        let source = include_str!("ordering_backend.rs");
        let begin = source
            .find("pub fn merge_sort_block_keys_agent")
            .expect("keys block-sort agent must exist");
        let end = source[begin..]
            .find("fn merge_sort_block_keys_tile")
            .map(|offset| begin + offset)
            .expect("keys tile implementation must follow all block agents");
        let agents_and_pair_io = &source[begin..end];

        for required in [
            "OneDimensionalLaunch<'_>",
            "WarpStripedBlockInput::<K",
            "WarpStripedBlockRead::<K",
            "WarpStripedBlockWrite::<K",
            "warp_striped_pair_into_write",
        ] {
            assert!(
                agents_and_pair_io.contains(required),
                "block-sort global range lost its Rust ownership capability: {required}"
            );
        }
        for forbidden in [
            ".as_ptr()".to_owned(),
            ".as_mut_ptr()".to_owned(),
            "cast_const".to_owned(),
            ["*", "const"].concat(),
            ["*", "mut"].concat(),
        ] {
            assert!(
                !agents_and_pair_io.contains(&forbidden),
                "block-sort global range recovered a raw pointer: {forbidden}"
            );
        }
    }

    #[test]
    fn block_sort_guarded_load_has_one_bounds_decision() {
        let source = include_str!("ordering_backend.rs");
        let pair_begin = source
            .find("fn merge_sort_block_pairs_load")
            .expect("pair block load must exist");
        let pair_end = source[pair_begin..]
            .find("fn merge_sort_block_pairs_loaded")
            .map(|offset| pair_begin + offset)
            .expect("pair loaded phase must follow pair block load");
        let pair_load = &source[pair_begin..pair_end];
        assert_eq!(pair_load.matches(".read_or(item,").count(), 2);
        assert!(
            !pair_load.contains("if !LAST || warp_striped < valid_items"),
            "the safe capability is already CUB's single guarded-load decision"
        );

        let keys_begin = source
            .find("fn merge_sort_block_keys_tile")
            .expect("keys block load must exist");
        let keys_end = source[keys_begin..]
            .find("fn merge_sort_block_keys_in_place_tile")
            .map(|offset| keys_begin + offset)
            .expect("in-place keys load must follow copy keys load");
        let keys_load = &source[keys_begin..keys_end];
        assert_eq!(keys_load.matches("input.read_or(item,").count(), 1);
        assert!(!keys_load.contains("if !LAST || warp_striped < valid_items"));
    }

    #[test]
    fn radix_init_uses_affine_grid_stride_capabilities() {
        let source = include_str!("ordering_backend.rs");
        let begin = source
            .find("pub fn radix_init_bins_and_counters_agent")
            .expect("radix init agent must exist");
        let end = source[begin..]
            .find("pub fn radix_histogram_agent")
            .map(|offset| begin + offset)
            .expect("histogram agent must follow radix init");
        let init = &source[begin..end];
        assert_eq!(init.matches("grid_stride_runs32").count(), 2);
        assert!(!init.contains("as_mut_ptr"));
        assert!(!init.contains(&["un", "safe"].concat()));
    }

    #[test]
    fn radix_scan_uses_one_affine_bin_capability() {
        let source = include_str!("ordering_backend.rs");
        let begin = source
            .find("pub fn radix_exclusive_sum_agent")
            .expect("radix scan agent must exist");
        let end = source[begin..]
            .find("fn warp_exclusive_sum_i32")
            .map(|offset| begin + offset)
            .expect("one-sweep helpers must follow radix scan");
        let scan = &source[begin..end];
        assert!(scan.contains("element_thread32"));
        assert!(!scan.contains("as_mut_ptr"));
    }

    #[test]
    fn warp_transpose_store_formula_is_a_tile_bijection() {
        for items in 1usize..=15 {
            let mut seen = vec![false; 512 * items];
            for thread in 0usize..512 {
                let lane = thread & 31;
                let warp = thread >> 5;
                let warp_offset = warp * 32 * items;
                for item in 0usize..items {
                    let output = warp_offset + item * 32 + lane;
                    assert!(!seen[output]);
                    seen[output] = true;
                }
            }
            assert!(seen.into_iter().all(|value| value));
        }
    }

    #[test]
    fn block_exchange_padding_matches_the_pinned_power_of_two_rule() {
        assert_eq!(block_exchange_storage_items::<256, 17>(), 4_352);
        assert_eq!(block_exchange_storage_items::<256, 8>(), 2_112);
        assert_eq!(block_exchange_index::<8>(31), 31);
        assert_eq!(block_exchange_index::<8>(32), 33);

        let mut occupied = vec![false; block_exchange_storage_items::<256, 8>()];
        for logical in 0..(256 * 8) as u32 {
            let physical = block_exchange_index::<8>(logical);
            assert!(
                !occupied[physical],
                "padded exchange index must be injective"
            );
            occupied[physical] = true;
        }
        assert_eq!(occupied.iter().filter(|&&slot| slot).count(), 2_048);
        assert_eq!(occupied.iter().filter(|&&slot| !slot).count(), 64);
    }

    #[test]
    fn block_merge_round_width_remains_runtime_algorithm_state() {
        let source = include_str!("ordering_backend.rs");
        let tests_begin = source
            .find("#[cfg(test)]")
            .expect("backend tests must follow the implementation");
        let implementation = &source[..tests_begin];
        assert_eq!(
            implementation
                .matches("while target_merged_threads <=")
                .count(),
            4
        );
        assert!(implementation.contains("target_merged_threads *= 2"));
        assert!(!implementation.contains("for_each_merge_round"));
        assert!(!implementation.contains("const TARGET"));
    }

    #[test]
    fn load_to_shared_layout_covers_every_alignment_without_overlap() {
        for item_size in [1u32, 2, 4, 8, 16] {
            for first_items in 0u32..=128 {
                for second_items in 0u32..=128 {
                    let tile_items = first_items + second_items;
                    let whole_buffer_bytes = load_to_shared_buffer_bytes(tile_items + 1, item_size)
                        + u32::from(item_size < 16) * 32;
                    let second_buffer_bytes = load_to_shared_buffer_bytes(second_items, item_size);
                    let second_buffer_offset = whole_buffer_bytes - second_buffer_bytes;

                    for first_address_mod_16 in 0u32..16 {
                        for second_address_mod_16 in 0u32..16 {
                            if first_address_mod_16 % item_size.min(16) != 0
                                || second_address_mod_16 % item_size.min(16) != 0
                            {
                                continue;
                            }
                            let first_padding = u32::from(first_items != 0) * first_address_mod_16;
                            let second_padding =
                                u32::from(second_items != 0) * second_address_mod_16;
                            let first_end = first_padding + first_items * item_size;
                            let second_begin = second_buffer_offset + second_padding;
                            let second_end = second_begin + second_items * item_size;
                            assert!(first_end <= second_begin);
                            assert!(second_end <= whole_buffer_bytes);
                        }
                    }
                }
            }
        }
    }
}
