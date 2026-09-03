use super::*;

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
pub(super) fn local_merge_path_begin(diagonal: i32, second_count: i32) -> i32 {
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
/// load-to-shared policy selected by ordinary cfg-aware const policy.
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
    let lane = warp::lane_id();
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
/// Ordinary cfg-aware const policy fixes `THREADS` and `ITEMS`, and the
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
    let lane = warp::lane_id();
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
    let lane = warp::lane_id();
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
    let lane = warp::lane_id();
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
    let lane = warp::lane_id();
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
    let lane = warp::lane_id();
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
    let lane = warp::lane_id();
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
    let lane = warp::lane_id();
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
    let lane = warp::lane_id();
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
