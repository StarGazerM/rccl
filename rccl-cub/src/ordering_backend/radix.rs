use super::*;

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

// These hidden function items are the Rust spelling of CUB's primitive
// `Traits` operations.  The sealed `RadixKey` implementation selects the exact
// item for the key type and direction. Keeping each operation as one named item
// is significant: identical operation calls then have identical function types, so
// cuda-oxide emits one kernel specialization instead of one per call site.
// None of the functions performs runtime policy or type dispatch.

#[doc(hidden)]
#[inline(always)]
pub fn radix_default_u32() -> u32 {
    u32::MAX
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_default_u64() -> u64 {
    u64::MAX
}

// CUB's primitive raw minimum/maximum padding keys become zero/all-ones after
// conversion to bit order. Keep that integer representation directly in
// Rust. Constructing a synthetic floating NaN value would permit payload
// canonicalization that CUB's byte reinterpretation never performs.

#[doc(hidden)]
#[inline(always)]
pub fn radix_min_ordered_u32() -> u32 {
    u32::MIN
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_min_ordered_u64() -> u64 {
    u64::MIN
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_u32_to_ordered_ascending(key: u32) -> u32 {
    key
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_u32_from_ordered_ascending(ordered: u32) -> u32 {
    ordered
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_u32_to_ordered_descending(key: u32) -> u32 {
    !key
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_u32_from_ordered_descending(ordered: u32) -> u32 {
    !ordered
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_i32_to_ordered_ascending(key: i32) -> u32 {
    (key as u32) ^ 0x8000_0000
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_i32_from_ordered_ascending(ordered: u32) -> i32 {
    (ordered ^ 0x8000_0000) as i32
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_i32_to_ordered_descending(key: i32) -> u32 {
    !((key as u32) ^ 0x8000_0000)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_i32_from_ordered_descending(ordered: u32) -> i32 {
    ((!ordered) ^ 0x8000_0000) as i32
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_f32_to_ordered_ascending(key: f32) -> u32 {
    let raw = key.to_bits();
    if raw & 0x8000_0000 != 0 {
        !raw
    } else {
        raw ^ 0x8000_0000
    }
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_f32_from_ordered_ascending(ordered: u32) -> f32 {
    let raw = if ordered & 0x8000_0000 != 0 {
        ordered ^ 0x8000_0000
    } else {
        !ordered
    };
    f32::from_bits(raw)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_f32_to_ordered_descending(key: f32) -> u32 {
    !radix_f32_to_ordered_ascending(key)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_f32_from_ordered_descending(ordered: u32) -> f32 {
    radix_f32_from_ordered_ascending(!ordered)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_u64_to_ordered_ascending(key: u64) -> u64 {
    key
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_u64_from_ordered_ascending(ordered: u64) -> u64 {
    ordered
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_u64_to_ordered_descending(key: u64) -> u64 {
    !key
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_u64_from_ordered_descending(ordered: u64) -> u64 {
    !ordered
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_i64_to_ordered_ascending(key: i64) -> u64 {
    (key as u64) ^ 0x8000_0000_0000_0000
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_i64_from_ordered_ascending(ordered: u64) -> i64 {
    (ordered ^ 0x8000_0000_0000_0000) as i64
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_i64_to_ordered_descending(key: i64) -> u64 {
    !((key as u64) ^ 0x8000_0000_0000_0000)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_i64_from_ordered_descending(ordered: u64) -> i64 {
    ((!ordered) ^ 0x8000_0000_0000_0000) as i64
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_f64_to_ordered_ascending(key: f64) -> u64 {
    let raw = key.to_bits();
    if raw & 0x8000_0000_0000_0000 != 0 {
        !raw
    } else {
        raw ^ 0x8000_0000_0000_0000
    }
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_f64_from_ordered_ascending(ordered: u64) -> f64 {
    let raw = if ordered & 0x8000_0000_0000_0000 != 0 {
        ordered ^ 0x8000_0000_0000_0000
    } else {
        !ordered
    };
    f64::from_bits(raw)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_f64_to_ordered_descending(key: f64) -> u64 {
    !radix_f64_to_ordered_ascending(key)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_f64_from_ordered_descending(ordered: u64) -> f64 {
    radix_f64_from_ordered_ascending(!ordered)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_digit_u32(ordered: u32, bit: u32, bits: u32) -> u32 {
    (ordered >> bit) & ((1u32 << bits) - 1)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_digit_masked_u32(ordered: u32, bit: u32, mask: u32) -> u32 {
    (ordered >> bit) & mask
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_rank_digit_u32(ordered: u32, bit: u32, bits: u32) -> u32 {
    radix_digit_u32(ordered, bit, bits)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_float_digit_u32(ordered: u32, bit: u32, bits: u32) -> u32 {
    let zero_ordered = if ordered == 0x7fff_ffff {
        0x8000_0000
    } else {
        ordered
    };
    radix_digit_u32(zero_ordered, bit, bits)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_float_digit_masked_u32(ordered: u32, bit: u32, mask: u32) -> u32 {
    let zero_ordered = if ordered == 0x7fff_ffff {
        0x8000_0000
    } else {
        ordered
    };
    radix_digit_masked_u32(zero_ordered, bit, mask)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_float_rank_digit_u32(ordered: u32, bit: u32, bits: u32) -> u32 {
    let zero_ordered = if ordered == 0x7fff_ffff {
        0x8000_0000
    } else {
        ordered
    };
    radix_rank_digit_u32(zero_ordered, bit, bits)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_digit_u64(ordered: u64, bit: u32, bits: u32) -> u32 {
    ((ordered >> bit) & ((1u64 << bits) - 1)) as u32
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_digit_masked_u64(ordered: u64, bit: u32, mask: u32) -> u32 {
    ((ordered >> bit) & mask as u64) as u32
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_rank_digit_u64(ordered: u64, bit: u32, bits: u32) -> u32 {
    radix_digit_u64(ordered, bit, bits)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_float_digit_u64(ordered: u64, bit: u32, bits: u32) -> u32 {
    let zero_ordered = if ordered == 0x7fff_ffff_ffff_ffff {
        0x8000_0000_0000_0000
    } else {
        ordered
    };
    radix_digit_u64(zero_ordered, bit, bits)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_float_digit_masked_u64(ordered: u64, bit: u32, mask: u32) -> u32 {
    let zero_ordered = if ordered == 0x7fff_ffff_ffff_ffff {
        0x8000_0000_0000_0000
    } else {
        ordered
    };
    radix_digit_masked_u64(zero_ordered, bit, mask)
}

#[doc(hidden)]
#[inline(always)]
pub fn radix_float_rank_digit_u64(ordered: u64, bit: u32, bits: u32) -> u32 {
    let zero_ordered = if ordered == 0x7fff_ffff_ffff_ffff {
        0x8000_0000_0000_0000
    } else {
        ordered
    };
    radix_rank_digit_u64(zero_ordered, bit, bits)
}

#[inline(always)]
fn warp_exclusive_sum_u32(value: u32, lane: u32) -> (u32, u32) {
    let mut inclusive = value;
    macro_rules! scan_step {
        ($offset:expr) => {{
            let preceding = warp::shuffle_up_sync(u32::MAX, inclusive, $offset);
            if lane >= $offset {
                inclusive = inclusive.wrapping_add(preceding);
            }
        }};
    }
    scan_step!(1u32);
    scan_step!(2u32);
    scan_step!(4u32);
    scan_step!(8u32);
    scan_step!(16u32);
    (inclusive.wrapping_sub(value), inclusive)
}

/// Pinned `BlockRadixRank<256, 6, ..., true, BLOCK_SCAN_WARP_SCANS>`.
///
/// CUB aliases the same 33x256 packed-counter grid in two ways: each thread
/// first owns one packed u16 counter pair for every counter lane, then each
/// thread rakes a contiguous 33-word segment.  Rust keeps the physical shared
/// allocation as checked u32 regions and names both phase mappings directly;
/// no public union, pointer, or type-erased view is required.
#[inline(always)]
fn radix_single_tile_rank<
    B,
    RankDigit,
    const THREADS: usize,
    const ITEMS: usize,
    const DESCENDING: bool,
>(
    keys: &[B; ITEMS],
    current_bit: u32,
    pass_bits: u32,
    rank_digit: RankDigit,
    shared_memory: &AlignedBlockSharedMemory<32>,
) -> [u32; ITEMS]
where
    B: Copy,
    RankDigit: Fn(B, u32, u32) -> u32 + Copy,
{
    const COUNTER_LANES: usize = 32;
    const PACKING_RATIO: usize = 2;
    const RAKING_SEGMENT: usize = COUNTER_LANES + 1;
    const WARPS: usize = 8;

    if THREADS != 256 || ITEMS == 0 || ITEMS > 19 {
        cuda_device::debug::trap();
    }

    let thread_index = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_index = thread_index >> 5;
    let grid_words = THREADS * RAKING_SEGMENT;

    // Reset through the digit-counter view
    // `[counter_lane][thread][packed_sub_counter]`. Only after the following
    // barrier does CUB reinterpret these same bytes as `[thread][33]` raking
    // segments. Keeping the two formulas distinct is the essential union
    // phase boundary.
    {
        let mut reset_words = shared_memory.region::<u32>(0, grid_words);
        for_each_radix_raking_word!(counter_lane, {
            reset_words.write(counter_lane * THREADS + thread_index as usize, 0);
        });
    }

    let mut thread_prefixes = [MaybeUninit::<u16>::uninit(); ITEMS];
    let mut counter_indices = [MaybeUninit::<u32>::uninit(); ITEMS];
    {
        let mut counters = shared_memory.region::<u16>(0, grid_words * PACKING_RATIO);
        for_each_thread_item!(ITEMS, item, {
            let digit = rank_digit(keys[item], current_bit, pass_bits);
            let mut sub_counter = digit >> 5;
            let mut counter_lane = digit & (COUNTER_LANES as u32 - 1);
            if DESCENDING {
                sub_counter = PACKING_RATIO as u32 - 1 - sub_counter;
                counter_lane = COUNTER_LANES as u32 - 1 - counter_lane;
            }
            let counter_index =
                (counter_lane * THREADS as u32 + thread_index) * PACKING_RATIO as u32 + sub_counter;
            let prefix = counters.read(counter_index as usize);
            thread_prefixes[item].write(prefix);
            counter_indices[item].write(counter_index);
            counters.write(counter_index as usize, prefix.wrapping_add(1));
        });
        thread::sync_threads();
    }

    // Memoized outer raking scan: cache all 33 words, reduce them, scan one
    // packed aggregate per thread with the warp-scans block algorithm, then
    // perform the exclusive downsweep in registers.
    {
        let mut cached_segment = [MaybeUninit::<u32>::uninit(); RAKING_SEGMENT];
        let mut grid = shared_memory.region::<u32>(0, grid_words);
        for_each_radix_raking_word!(word, {
            let value = grid.read(thread_index as usize * RAKING_SEGMENT + word);
            cached_segment[word].write(value);
        });
        let mut aggregate = initialized_value(&cached_segment[0]);
        for_each_radix_raking_word!(word, {
            if word != 0 {
                aggregate = aggregate.wrapping_add(initialized_value(&cached_segment[word]));
            }
        });

        let (warp_exclusive, warp_inclusive) = warp_exclusive_sum_u32(aggregate, lane);
        let scan_offset = grid_words * core::mem::size_of::<u32>();
        let mut warp_aggregates = shared_memory.region::<u32>(scan_offset, WARPS);
        // CUB stores eight one-byte `NullType` warp-scan placeholders between
        // the aggregate array and this aligned u32 block prefix.
        let mut block_prefix_storage = shared_memory
            .region::<u32>(scan_offset + WARPS * core::mem::size_of::<u32>() + WARPS, 1);
        if lane == 31 {
            warp_aggregates.write(warp_index as usize, warp_inclusive);
        }
        thread::sync_threads();

        let mut warp_prefix = 0u32;
        let mut block_aggregate = warp_aggregates.read(0);
        for_each_single_tile_warp!(warp, {
            if warp_index as usize == warp {
                warp_prefix = block_aggregate;
            }
            block_aggregate = block_aggregate.wrapping_add(warp_aggregates.read(warp));
        });
        let mut exclusive_partial =
            warp_exclusive.wrapping_add(if warp_index == 0 { 0 } else { warp_prefix });
        if warp_index == 0 {
            let block_prefix = block_aggregate << 16;
            if lane == 0 {
                block_prefix_storage.write(0, block_prefix);
                exclusive_partial = block_prefix;
            }
        }
        thread::sync_threads();
        let block_prefix = block_prefix_storage.read(0);
        if thread_index > 0 {
            exclusive_partial = block_prefix.wrapping_add(exclusive_partial);
        }
        let mut running = exclusive_partial;
        for_each_radix_raking_word!(word, {
            let value = initialized_value(&cached_segment[word]);
            cached_segment[word].write(running);
            running = running.wrapping_add(value);
        });
        let _ = running;
        for_each_radix_raking_word!(word, {
            grid.write(
                thread_index as usize * RAKING_SEGMENT + word,
                initialized_value(&cached_segment[word]),
            );
        });
        thread::sync_threads();
    }

    let mut ranks = [0u32; ITEMS];
    let counters = shared_memory.region::<u16>(0, grid_words * PACKING_RATIO);
    for_each_thread_item!(ITEMS, item, {
        let block_prefix = counters.read(initialized_value(&counter_indices[item]) as usize);
        ranks[item] = initialized_value(&thread_prefixes[item]) as u32 + block_prefix as u32;
    });
    ranks
}

#[inline(always)]
fn radix_single_tile_scatter<T, const THREADS: usize, const ITEMS: usize, const STRIPED: bool>(
    items: &mut [T; ITEMS],
    ranks: &[u32; ITEMS],
    shared_memory: &AlignedBlockSharedMemory<32>,
) where
    T: Copy,
{
    let thread_index = thread::threadIdx_x() as usize;
    let storage_items = block_exchange_storage_items::<THREADS, ITEMS>();
    let mut exchange = shared_memory.region::<MaybeUninit<T>>(0, storage_items);
    for_each_thread_item!(ITEMS, item, {
        exchange.write(
            block_exchange_index::<ITEMS>(ranks[item]),
            MaybeUninit::new(items[item]),
        );
    });
    thread::sync_threads();
    for_each_thread_item!(ITEMS, item, {
        let logical = if STRIPED {
            item * THREADS + thread_index
        } else {
            thread_index * ITEMS + item
        };
        items[item] = exchange.read_initialized(block_exchange_index::<ITEMS>(logical as u32));
    });
}

/// Exact single-block route selected by pinned SM80 `DeviceRadixSort` for a
/// nonempty range no larger than the Policy800 single-tile extent.
#[inline(always)]
pub fn radix_single_tile_keys_agent<
    K,
    B,
    DefaultOrdered,
    ToOrdered,
    FromOrdered,
    RankDigit,
    const THREADS: usize,
    const ITEMS: usize,
    const DESCENDING: bool,
>(
    input: &[K],
    output: DisjointSlice<K, LinearTiles<ITEMS>>,
    bit_range: Range<u32>,
    operations: (DefaultOrdered, ToOrdered, FromOrdered, RankDigit),
    launch_context: OneDimensionalLaunch<'_>,
    shared_memory: AlignedBlockSharedMemory<32>,
) where
    K: Copy,
    B: Copy,
    DefaultOrdered: Fn() -> B + Copy,
    ToOrdered: Fn(K) -> B + Copy,
    FromOrdered: Fn(B) -> K + Copy,
    RankDigit: Fn(B, u32, u32) -> u32 + Copy,
{
    let (default_ordered, to_ordered, from_ordered, rank_digit) = operations;
    let mut output =
        BlockStripedBlockWrite::<K, THREADS, ITEMS>::new(output, launch_context, input.len());
    let thread_index = thread::threadIdx_x() as usize;
    let default = default_ordered();
    let mut keys = [default; ITEMS];
    for_each_thread_item!(ITEMS, item, {
        let blocked = thread_index * ITEMS + item;
        if let Some(key) = input.get(blocked).copied() {
            keys[item] = to_ordered(key);
        }
    });
    thread::sync_threads();

    let mut current_bit = bit_range.start;
    while current_bit < bit_range.end {
        let pass_bits = (bit_range.end - current_bit).min(6);
        let ranks = radix_single_tile_rank::<B, _, THREADS, ITEMS, DESCENDING>(
            &keys,
            current_bit,
            pass_bits,
            rank_digit,
            &shared_memory,
        );
        current_bit += 6;
        thread::sync_threads();
        if current_bit >= bit_range.end {
            radix_single_tile_scatter::<B, THREADS, ITEMS, true>(&mut keys, &ranks, &shared_memory);
            break;
        }
        radix_single_tile_scatter::<B, THREADS, ITEMS, false>(&mut keys, &ranks, &shared_memory);
        thread::sync_threads();
    }

    for_each_thread_item!(ITEMS, item, {
        let striped = item * THREADS + thread_index;
        if striped < input.len() {
            output.write_if_valid(item, from_ordered(keys[item]));
        }
    });
}

/// Key/value form of the pinned single-tile route. Values undergo the same
/// stable shared-memory permutation after every key-rank phase.
#[inline(always)]
pub fn radix_single_tile_pairs_agent<
    K,
    V,
    B,
    DefaultOrdered,
    ToOrdered,
    FromOrdered,
    RankDigit,
    const THREADS: usize,
    const ITEMS: usize,
    const DESCENDING: bool,
>(
    input: (&[K], &[V]),
    output: (
        DisjointSlice<K, LinearTiles<ITEMS>>,
        DisjointSlice<V, LinearTiles<ITEMS>>,
    ),
    bit_range: Range<u32>,
    operations: (DefaultOrdered, ToOrdered, FromOrdered, RankDigit),
    launch_context: OneDimensionalLaunch<'_>,
    shared_memory: AlignedBlockSharedMemory<32>,
) where
    K: Copy,
    V: Copy,
    B: Copy,
    DefaultOrdered: Fn() -> B + Copy,
    ToOrdered: Fn(K) -> B + Copy,
    FromOrdered: Fn(B) -> K + Copy,
    RankDigit: Fn(B, u32, u32) -> u32 + Copy,
{
    let (input_keys, input_values) = input;
    let (output_keys, output_values) = output;
    let (default_ordered, to_ordered, from_ordered, rank_digit) = operations;
    let mut output_keys = BlockStripedBlockWrite::<K, THREADS, ITEMS>::new(
        output_keys,
        launch_context,
        input_keys.len(),
    );
    let mut output_values = BlockStripedBlockWrite::<V, THREADS, ITEMS>::new(
        output_values,
        launch_context,
        input_values.len(),
    );
    let thread_index = thread::threadIdx_x() as usize;
    let default = default_ordered();
    let mut keys = [default; ITEMS];
    let mut values = [MaybeUninit::<V>::uninit(); ITEMS];
    for_each_thread_item!(ITEMS, item, {
        let blocked = thread_index * ITEMS + item;
        if let Some(key) = input_keys.get(blocked).copied() {
            keys[item] = to_ordered(key);
        }
    });
    thread::sync_threads();
    // Pinned CUB deliberately moves `num_items` through lane zero before the
    // second guarded direct load so the backend cannot reuse the key-load
    // predicate/address state and inflate pair-sort register pressure.
    let value_items = warp::shuffle_sync(u32::MAX, input_values.len() as u32, 0) as usize;
    for_each_thread_item!(ITEMS, item, {
        let blocked = thread_index * ITEMS + item;
        if blocked < value_items {
            if let Some(value) = input_values.get(blocked).copied() {
                values[item].write(value);
            }
        }
    });
    thread::sync_threads();

    let mut current_bit = bit_range.start;
    while current_bit < bit_range.end {
        let pass_bits = (bit_range.end - current_bit).min(6);
        let ranks = radix_single_tile_rank::<B, _, THREADS, ITEMS, DESCENDING>(
            &keys,
            current_bit,
            pass_bits,
            rank_digit,
            &shared_memory,
        );
        current_bit += 6;
        thread::sync_threads();
        if current_bit >= bit_range.end {
            radix_single_tile_scatter::<B, THREADS, ITEMS, true>(&mut keys, &ranks, &shared_memory);
            thread::sync_threads();
            radix_single_tile_scatter::<MaybeUninit<V>, THREADS, ITEMS, true>(
                &mut values,
                &ranks,
                &shared_memory,
            );
            break;
        }
        radix_single_tile_scatter::<B, THREADS, ITEMS, false>(&mut keys, &ranks, &shared_memory);
        thread::sync_threads();
        radix_single_tile_scatter::<MaybeUninit<V>, THREADS, ITEMS, false>(
            &mut values,
            &ranks,
            &shared_memory,
        );
        thread::sync_threads();
    }

    for_each_thread_item!(ITEMS, item, {
        let striped = item * THREADS + thread_index;
        if striped < input_keys.len() {
            output_keys.write_if_valid(item, from_ordered(keys[item]));
            output_values.write_if_valid(item, initialized_value(&values[item]));
        }
    });
}

/// Exact pinned SM80+ histogram agent used before one-sweep radix passes.
///
/// `to_ordered` and `digit` are sealed `RadixKey` function items. This keeps
/// representation selection static while
/// retaining CUB's striped loads, per-block shared histogram, and device-wide
/// atomic accumulation.
#[inline(always)]
pub fn radix_histogram_agent<
    K,
    B,
    DefaultKey,
    ToOrdered,
    Digit,
    const THREADS: usize,
    const ITEMS: usize,
    const MAX_PASSES: usize,
>(
    keys: &[K],
    bit_range: core::ops::Range<u32>,
    default_key: DefaultKey,
    to_ordered: ToOrdered,
    digit: Digit,
    bins: DisjointSlice<u64>,
    shared_memory: BlockSharedMemory,
) where
    K: Copy,
    B: Copy,
    DefaultKey: Fn() -> B + Copy,
    ToOrdered: Fn(K) -> B + Copy,
    Digit: Fn(B, u32, u32) -> u32 + Copy,
{
    const RADIX_DIGITS: usize = 256;
    let thread_index = thread::threadIdx_x() as usize;
    let tile_items = THREADS * ITEMS;
    if shared_memory.bytes() != MAX_PASSES * RADIX_DIGITS * core::mem::size_of::<u32>() {
        cuda_device::debug::trap();
    }
    let mut shared = shared_memory.region::<u32>(0, MAX_PASSES * RADIX_DIGITS);
    let mut bins = DeviceRegion::new(bins);
    let num_passes = (bit_range.end - bit_range.start).div_ceil(8) as usize;
    let sentinel = default_key();

    for_each_histogram_thread_bin!(THREADS, thread_index, bin, {
        for_each_histogram_pass!(MAX_PASSES, num_passes, pass, {
            shared.write(pass * RADIX_DIGITS + bin, 0);
        });
    });
    thread::sync_threads();

    let mut tile_offset = thread::blockIdx_x() as usize * tile_items;
    let tile_stride = thread::gridDim_x() as usize * tile_items;
    while tile_offset < keys.len() {
        let mut ordered_keys = [MaybeUninit::<B>::uninit(); ITEMS];
        for_each_thread_item!(ITEMS, item, {
            let index = tile_offset + thread_index + item * THREADS;
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
        for_each_histogram_pass!(MAX_PASSES, num_passes, pass, {
            let current_bit = bit_range.start + pass as u32 * 8;
            let num_bits = (bit_range.end - current_bit).min(8);
            let digit_mask = (1u32 << num_bits) - 1;
            for_each_thread_item!(ITEMS, item, {
                let bin = digit(
                    initialized_value(&ordered_keys[item]),
                    current_bit,
                    digit_mask,
                ) as usize;
                shared.fetch_add_block(pass * RADIX_DIGITS + bin, 1);
            });
        });
        tile_offset += tile_stride;
    }
    thread::sync_threads();

    for_each_histogram_thread_bin!(THREADS, thread_index, bin, {
        for_each_histogram_pass!(MAX_PASSES, num_passes, pass, {
            let count = shared.read(pass * RADIX_DIGITS + bin);
            if count != 0 {
                bins.fetch_add_device(pass * RADIX_DIGITS + bin, count as u64);
            }
        });
    });
    thread::sync_threads();
}

#[inline(always)]
fn warp_exclusive_sum_u64(value: u64, lane: u32) -> (u64, u64) {
    let mut inclusive = value;
    for_each_warp_scan_offset!(offset, {
        let preceding = warp::shuffle_up_u64_sync(u32::MAX, inclusive, offset);
        if lane >= offset {
            inclusive += preceding;
        }
    });
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
        let mut cached = [MaybeUninit::<u64>::uninit(); SEGMENT_LENGTH];
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            cached[item].write(grid.read(segment + item));
        });
        let mut aggregate = initialized_value(&cached[0]);
        for_each_raking_item!(SEGMENT_LENGTH - 1, offset, {
            aggregate += initialized_value(&cached[offset + 1]);
        });
        let (segment_prefix, _) = warp_exclusive_sum_u64(aggregate, thread_index as u32);

        // Policy default is BLOCK_SCAN_RAKING, not memoize: the downsweep
        // deliberately reloads the segment from shared memory.
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            cached[item].write(grid.read(segment + item));
        });
        let mut running = segment_prefix;
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            let current = initialized_value(&cached[item]);
            cached[item].write(running);
            running += current;
        });
        let _ = running;
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            grid.write(segment + item, initialized_value(&cached[item]));
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
    shared_memory: BlockSharedMemory,
) {
    let pass = thread::blockIdx_x() as usize;
    if pass >= PASSES {
        return;
    }
    const SHARED_BYTES: usize = 2_336;
    const GRID_OFFSET_BYTES: usize = 16;
    if shared_memory.bytes() != SHARED_BYTES {
        cuda_device::debug::trap();
    }
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
    for_each_warp_scan_offset!(offset, {
        // cuda-oxide's typed 32-bit shuffle transports bits as `u32`; the
        // pinned CUB scan value is nevertheless C++ `int` throughout.
        let preceding = warp::shuffle_up_sync(u32::MAX, inclusive as u32, offset) as i32;
        if lane >= offset {
            inclusive += preceding;
        }
    });
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
        let mut cached = [MaybeUninit::<i32>::uninit(); SEGMENT_LENGTH];
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            cached[item].write(grid.read(segment + item));
        });
        let mut aggregate = initialized_value(&cached[0]);
        for_each_raking_item!(SEGMENT_LENGTH - 1, offset, {
            aggregate += initialized_value(&cached[offset + 1]);
        });
        let (segment_prefix, _) = warp_exclusive_sum_i32(aggregate, thread_index as u32);

        // Policy800 selects BLOCK_SCAN_RAKING_MEMOIZE: reuse the register
        // segment loaded during the upsweep instead of rereading shared.
        let mut running = segment_prefix;
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            let current = initialized_value(&cached[item]);
            cached[item].write(running);
            running += current;
        });
        let _ = running;
        for_each_raking_item!(SEGMENT_LENGTH, item, {
            grid.write(segment + item, initialized_value(&cached[item]));
        });
    }
    thread::sync_threads();
    grid.read(placement)
}

#[inline(always)]
fn radix_lookback_global(
    bin: u32,
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
        let location = previous as usize * RADIX_DIGITS + bin as usize;
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
    lookback.write_volatile(
        block_index as usize * RADIX_DIGITS + bin as usize,
        publication,
    );
    let current = global_offsets.read(bin as usize);
    global_offsets.write(bin as usize, current + (inclusive - local_count) as u64);
    let _ = wanted;
}

/// Exact pinned SM80 `AgentRadixSortOnesweep` keys-only path.
///
/// Ordinary const policy fixes `THREADS` and `ITEMS`; `RadixKey` supplies the
/// bit conversions. Checked `rccl-memory` capabilities own CUB's phase-aliased
/// shared storage, device-wide atomic lookback, and disjoint output contracts;
/// the algorithm itself receives only affine values and borrowed slices.
#[inline(always)]
pub fn radix_onesweep_keys_agent<
    K,
    B,
    DefaultKey,
    ToOrdered,
    FromOrdered,
    Digit,
    const THREADS: usize,
    const ITEMS: usize,
    const UNION_BYTES: usize,
>(
    storage: (DisjointSlice<i32>, DisjointSlice<i32>, &[u64]),
    buffers: (&[K], DisjointSlice<K>),
    pass: (usize, usize, u32, u32),
    transforms: (DefaultKey, ToOrdered, FromOrdered, Digit),
    shared_memory: BlockSharedMemory,
) where
    K: Copy,
    B: Copy,
    DefaultKey: Fn() -> B + Copy,
    ToOrdered: Fn(K) -> B + Copy,
    FromOrdered: Fn(B) -> K + Copy,
    Digit: Fn(B, u32, u32) -> u32 + Copy,
{
    let (lookback, counter, bins) = storage;
    let (input, output) = buffers;
    let (bin_offset, counter_offset, current_bit, num_bits) = pass;
    let (default_key, to_ordered, from_ordered, digit) = transforms;
    let digit_mask = (1u32 << num_bits) - 1;
    const RADIX_DIGITS: usize = 256;
    let thread_index = thread::threadIdx_x();
    let lane_capability = WarpLane::current();
    let lane = lane_capability.index();
    let warp_index = thread_index >> 5;
    let tile_items = (THREADS * ITEMS) as u32;
    let num_items = input.len() as u32;
    let warp_offsets_bytes = (THREADS / 32) * RADIX_DIGITS * core::mem::size_of::<u32>();
    let prefix_storage_offset = warp_offsets_bytes + RADIX_DIGITS * core::mem::size_of::<u32>();
    let prefix_grid_offset = prefix_storage_offset + 16;
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
        let mut block_index_slot = shared_memory.region::<i32>(UNION_BYTES, 1);
        block_index_slot.write(0, dynamic_block);
    }
    thread::sync_threads();
    let block_index = shared_memory.region::<i32>(UNION_BYTES, 1).read(0);
    if thread_index == 0 {
        // cuda-oxide does not yet expose CUDA's `__syncthreads_or`. Reserve
        // one cell in CUB's guard match-mask allocation; the histogram's
        // later block barrier publishes this initialization before the vote.
        let mut short_flags = shared_memory.region::<u32>(warp_offsets_bytes, RADIX_DIGITS);
        short_flags.write(0, 0);
    }
    let tile_base = block_index as u32 * tile_items;
    let full_block = tile_base + tile_items <= num_items;

    let mut keys = [MaybeUninit::<B>::uninit(); ITEMS];
    let warp_base = warp_index * 32 * ITEMS as u32 + lane;
    let warp_group_base = tile_base as usize + warp_index as usize * 32usize * ITEMS;
    if full_block {
        let Some(warp_group) = input.get(warp_group_base..warp_group_base + 32usize * ITEMS) else {
            return;
        };
        let full_input = FullWarpStripedInput::<K, ITEMS>::new(warp_group, lane_capability);
        for_each_thread_item!(ITEMS, item, {
            keys[item].write(to_ordered(full_input.read(item)));
        });
    } else {
        let partial_input =
            PartialWarpStripedInput::<K, ITEMS>::new(input, warp_group_base, lane_capability);
        for_each_thread_item!(ITEMS, item, {
            keys[item].write(partial_input.map_or(item, sentinel, to_ordered));
        });
    }

    // One private warp histogram, exactly `NUM_PARTS == 1` for the pinned
    // SM80 policy. The warp-strided input arrangement preserves source order.
    {
        let mut warp_offsets = shared_memory.region::<i32>(0, (THREADS / 32) * RADIX_DIGITS);
        for_each_radix_warp_bin!(slot, {
            let bin = (lane + slot * 32) as usize;
            warp_offsets.write((warp_index * RADIX_DIGITS as u32 + bin as u32) as usize, 0);
        });
        warp::sync_mask(u32::MAX);
        for_each_thread_item!(ITEMS, item, {
            let ordered = initialized_value(&keys[item]);
            warp_offsets.fetch_add_block(
                (warp_index * RADIX_DIGITS as u32 + digit(ordered, current_bit, digit_mask))
                    as usize,
                1,
            );
        });
    }
    thread::sync_threads();

    let active_bin = thread_index < RADIX_DIGITS as u32;
    let bin = thread_index;
    let mut local_count = 0i32;
    if active_bin {
        let mut warp_offsets = shared_memory.region::<i32>(0, (THREADS / 32) * RADIX_DIGITS);
        for_each_policy800_warp!(source_warp, {
            let location = source_warp as u32 * RADIX_DIGITS as u32 + bin;
            let count = warp_offsets.read(location as usize);
            warp_offsets.write(location as usize, local_count);
            local_count += count;
        });
        let partial = local_count | 0x4000_0000;
        lookback.write_volatile(block_index as usize * RADIX_DIGITS + bin as usize, partial);
    }

    // CUB invokes `__syncthreads_or` in the lookback callback. At most one
    // digit can own the whole tile, so its bin thread uniquely publishes the
    // true case into the otherwise-idle guard match-mask cell.
    if active_bin && local_count == tile_items as i32 {
        let mut short_flags = shared_memory.region::<u32>(warp_offsets_bytes, RADIX_DIGITS);
        short_flags.write(0, 1);
    }
    thread::sync_threads();
    // A same-address shared load is a warp multicast. Every thread consumes
    // the block-wide result directly, which is the safe cuda-oxide capability
    // replacement for CUB's currently unavailable `__syncthreads_or` result.
    let short_circuit = shared_memory
        .region::<u32>(warp_offsets_bytes, RADIX_DIGITS)
        .read(0)
        != 0;

    // CUB performs the short-circuit from its counts callback, before the
    // block-wide exclusive scan and rank downsweep. A uniform digit tile must
    // therefore never execute those later phases merely because the result
    // would still be correct.
    if short_circuit {
        let common_bin = digit(initialized_value(&keys[0]), current_bit, digit_mask);
        let mut global_offsets = shared_memory.region::<u64>(UNION_BYTES, RADIX_DIGITS);
        if active_bin {
            global_offsets.write(
                bin as usize,
                pass_bins[bin as usize]
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
        let output_base = global_offsets.read(common_bin as usize) as usize;
        if full_block {
            for_each_thread_item!(ITEMS, item, {
                let local_index = warp_base + item as u32 * 32;
                let value = from_ordered(initialized_value(&keys[item]));
                output.write(output_base + local_index as usize, value);
            });
        } else {
            let tile_items = num_items - tile_base;
            for_each_thread_item!(ITEMS, item, {
                let local_index = warp_base + item as u32 * 32;
                if local_index < tile_items {
                    let value = from_ordered(initialized_value(&keys[item]));
                    output.write(output_base + local_index as usize, value);
                }
            });
        }
        return;
    }

    let exclusive_digit_prefix = {
        let mut prefix_grid = shared_memory.region::<i32>(
            prefix_grid_offset,
            (UNION_BYTES - prefix_grid_offset) / core::mem::size_of::<i32>(),
        );
        block_exclusive_sum_raking_memoize_384_i32(local_count, &mut prefix_grid)
    };
    if active_bin {
        let mut warp_offsets = shared_memory.region::<i32>(0, (THREADS / 32) * RADIX_DIGITS);
        for_each_policy800_warp!(source_warp, {
            let location = source_warp as u32 * RADIX_DIGITS as u32 + bin;
            let prior_warps = warp_offsets.read(location as usize);
            warp_offsets.write(location as usize, prior_warps + exclusive_digit_prefix);
        });
    }
    thread::sync_threads();

    // CUB's register array is uninitialized and every fully-unrolled rank
    // step writes its slot before the scatter reads it. Initializing all 21
    // registers here would hide an unnecessary instruction path behind the
    // same functional result.
    let mut ranks = [MaybeUninit::<i32>::uninit(); ITEMS];
    {
        let mut warp_offsets = shared_memory.region::<i32>(0, (THREADS / 32) * RADIX_DIGITS);
        for_each_thread_item!(ITEMS, item, {
            let item_bin = digit(initialized_value(&keys[item]), current_bit, digit_mask);
            let matching = radix_match_any_8(item_bin);
            let leader = 31 - matching.leading_zeros();
            let within_match = (matching & warp::lanemask_le()).count_ones() as i32;
            let mut warp_offset = 0i32;
            if lane == leader {
                warp_offset = warp_offsets.fetch_add_block(
                    (warp_index * RADIX_DIGITS as u32 + item_bin) as usize,
                    within_match,
                );
            }
            warp_offset = warp::shuffle_sync(u32::MAX, warp_offset as u32, leader) as i32;
            ranks[item].write(warp_offset + within_match - 1);
        });
    }

    // `rank_temp_storage` and `keys_out` are the same CUB union. Every warp
    // must finish consuming its rank offsets before any thread starts the key
    // scatter phase and overwrites that storage.
    thread::sync_threads();

    let mut keys_shared = shared_memory
        .region::<MaybeUninit<B>>(0, UNION_BYTES / core::mem::size_of::<MaybeUninit<B>>());
    for_each_thread_item!(ITEMS, item, {
        let ordered = initialized_value(&keys[item]);
        let rank = initialized_value(&ranks[item]);
        keys_shared.write(rank as usize, MaybeUninit::new(ordered));
    });

    // CUB overlaps independent key-scatter and lookback work here. The single
    // barrier after lookback publishes both keys and global offsets before
    // any thread reads either; inserting an earlier barrier serializes the
    // phases and changes the one-sweep schedule.
    let mut global_offsets = shared_memory.region::<u64>(UNION_BYTES, RADIX_DIGITS);
    if active_bin {
        global_offsets.write(
            bin as usize,
            pass_bins[bin as usize] - exclusive_digit_prefix as u64,
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

    if full_block {
        for_each_thread_item!(ITEMS, item, {
            let blocked_index = thread_index + item as u32 * THREADS as u32;
            let ordered = keys_shared.read_initialized(blocked_index as usize);
            let item_bin = digit(ordered, current_bit, digit_mask);
            let output_index =
                blocked_index as usize + global_offsets.read(item_bin as usize) as usize;
            output.write(output_index, from_ordered(ordered));
            warp::sync_mask(u32::MAX);
        });
    } else {
        let tile_items = num_items - tile_base;
        for_each_thread_item!(ITEMS, item, {
            let blocked_index = thread_index + item as u32 * THREADS as u32;
            if blocked_index < tile_items {
                let ordered = keys_shared.read_initialized(blocked_index as usize);
                let item_bin = digit(ordered, current_bit, digit_mask);
                let output_index =
                    blocked_index as usize + global_offsets.read(item_bin as usize) as usize;
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
    const THREADS: usize,
    const ITEMS: usize,
    const UNION_BYTES: usize,
>(
    storage: (DisjointSlice<i32>, DisjointSlice<i32>, &[u64]),
    buffers: (&[K], &[V], DisjointSlice<K>, DisjointSlice<V>),
    pass: (usize, usize, u32, u32),
    transforms: (DefaultKey, ToOrdered, FromOrdered, Digit),
    shared_memory: BlockSharedMemory,
) where
    K: Copy,
    V: Copy,
    B: Copy,
    DefaultKey: Fn() -> B + Copy,
    ToOrdered: Fn(K) -> B + Copy,
    FromOrdered: Fn(B) -> K + Copy,
    Digit: Fn(B, u32, u32) -> u32 + Copy,
{
    let (lookback, counter, bins) = storage;
    let (input_keys, input_values, output_keys, output_values) = buffers;
    let (bin_offset, counter_offset, current_bit, num_bits) = pass;
    let (default_key, to_ordered, from_ordered, digit) = transforms;
    let digit_mask = (1u32 << num_bits) - 1;
    const RADIX_DIGITS: usize = 256;
    let thread_index = thread::threadIdx_x();
    let lane_capability = WarpLane::current();
    let lane = lane_capability.index();
    let warp_index = thread_index >> 5;
    let tile_items = (THREADS * ITEMS) as u32;
    let num_items = input_keys.len() as u32;
    let warp_offsets_bytes = (THREADS / 32) * RADIX_DIGITS * core::mem::size_of::<u32>();
    let prefix_storage_offset = warp_offsets_bytes + RADIX_DIGITS * core::mem::size_of::<u32>();
    let prefix_grid_offset = prefix_storage_offset + 16;
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
        let mut block_index_slot = shared_memory.region::<i32>(UNION_BYTES, 1);
        block_index_slot.write(0, dynamic_block);
    }
    thread::sync_threads();
    let block_index = shared_memory.region::<i32>(UNION_BYTES, 1).read(0);
    if thread_index == 0 {
        let mut short_flags = shared_memory.region::<u32>(warp_offsets_bytes, RADIX_DIGITS);
        short_flags.write(0, 0);
    }
    let tile_base = block_index as u32 * tile_items;
    let full_block = tile_base + tile_items <= num_items;

    let mut keys = [MaybeUninit::<B>::uninit(); ITEMS];
    let warp_base = warp_index * 32 * ITEMS as u32 + lane;
    let warp_group_base = tile_base as usize + warp_index as usize * 32usize * ITEMS;
    if full_block {
        let Some(warp_group) = input_keys.get(warp_group_base..warp_group_base + 32usize * ITEMS)
        else {
            return;
        };
        let full_input = FullWarpStripedInput::<K, ITEMS>::new(warp_group, lane_capability);
        for_each_thread_item!(ITEMS, item, {
            keys[item].write(to_ordered(full_input.read(item)));
        });
    } else {
        let partial_input =
            PartialWarpStripedInput::<K, ITEMS>::new(input_keys, warp_group_base, lane_capability);
        for_each_thread_item!(ITEMS, item, {
            keys[item].write(partial_input.map_or(item, sentinel, to_ordered));
        });
    }

    {
        let mut warp_offsets = shared_memory.region::<i32>(0, (THREADS / 32) * RADIX_DIGITS);
        for_each_radix_warp_bin!(slot, {
            let bin = (lane + slot * 32) as usize;
            warp_offsets.write((warp_index * RADIX_DIGITS as u32 + bin as u32) as usize, 0);
        });
        warp::sync_mask(u32::MAX);
        for_each_thread_item!(ITEMS, item, {
            let ordered = initialized_value(&keys[item]);
            warp_offsets.fetch_add_block(
                (warp_index * RADIX_DIGITS as u32 + digit(ordered, current_bit, digit_mask))
                    as usize,
                1,
            );
        });
    }
    thread::sync_threads();

    let active_bin = thread_index < RADIX_DIGITS as u32;
    let bin = thread_index;
    let mut local_count = 0i32;
    if active_bin {
        let mut warp_offsets = shared_memory.region::<i32>(0, (THREADS / 32) * RADIX_DIGITS);
        for_each_policy800_warp!(source_warp, {
            let location = source_warp as u32 * RADIX_DIGITS as u32 + bin;
            let count = warp_offsets.read(location as usize);
            warp_offsets.write(location as usize, local_count);
            local_count += count;
        });
        lookback.write_volatile(
            block_index as usize * RADIX_DIGITS + bin as usize,
            local_count | 0x4000_0000,
        );
    }
    if active_bin && local_count == tile_items as i32 {
        let mut short_flags = shared_memory.region::<u32>(warp_offsets_bytes, RADIX_DIGITS);
        short_flags.write(0, 1);
    }
    thread::sync_threads();
    let short_circuit = shared_memory
        .region::<u32>(warp_offsets_bytes, RADIX_DIGITS)
        .read(0)
        != 0;

    if short_circuit {
        let common_bin = digit(initialized_value(&keys[0]), current_bit, digit_mask);
        let mut global_offsets = shared_memory.region::<u64>(UNION_BYTES, RADIX_DIGITS);
        if active_bin {
            global_offsets.write(
                bin as usize,
                pass_bins[bin as usize]
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
        let output_base = global_offsets.read(common_bin as usize) as usize;
        if full_block {
            for_each_thread_item!(ITEMS, item, {
                let local_index = warp_base + item as u32 * 32;
                output_keys.write(
                    output_base + local_index as usize,
                    from_ordered(initialized_value(&keys[item])),
                );
            });
        } else {
            let tile_items = num_items - tile_base;
            for_each_thread_item!(ITEMS, item, {
                let local_index = warp_base + item as u32 * 32;
                if local_index < tile_items {
                    output_keys.write(
                        output_base + local_index as usize,
                        from_ordered(initialized_value(&keys[item])),
                    );
                }
            });
        }

        let mut values = [MaybeUninit::<V>::uninit(); ITEMS];
        if full_block {
            let Some(warp_group) =
                input_values.get(warp_group_base..warp_group_base + 32usize * ITEMS)
            else {
                return;
            };
            let full_values = FullWarpStripedInput::<V, ITEMS>::new(warp_group, lane_capability);
            for_each_thread_item!(ITEMS, item, {
                values[item].write(full_values.read(item));
            });
        } else {
            let partial_values = PartialWarpStripedInput::<V, ITEMS>::new(
                input_values,
                warp_group_base,
                lane_capability,
            );
            for_each_thread_item!(ITEMS, item, {
                partial_values.load_into(item, &mut values[item]);
            });
        }
        if full_block {
            for_each_thread_item!(ITEMS, item, {
                let local_index = warp_base + item as u32 * 32;
                output_values.write(
                    output_base + local_index as usize,
                    initialized_value(&values[item]),
                );
            });
        } else {
            let tile_items = num_items - tile_base;
            for_each_thread_item!(ITEMS, item, {
                let local_index = warp_base + item as u32 * 32;
                if local_index < tile_items {
                    output_values.write(
                        output_base + local_index as usize,
                        initialized_value(&values[item]),
                    );
                }
            });
        }
        return;
    }

    let exclusive_digit_prefix = {
        let mut prefix_grid = shared_memory.region::<i32>(
            prefix_grid_offset,
            (UNION_BYTES - prefix_grid_offset) / core::mem::size_of::<i32>(),
        );
        block_exclusive_sum_raking_memoize_384_i32(local_count, &mut prefix_grid)
    };
    if active_bin {
        let mut warp_offsets = shared_memory.region::<i32>(0, (THREADS / 32) * RADIX_DIGITS);
        for_each_policy800_warp!(source_warp, {
            let location = source_warp as u32 * RADIX_DIGITS as u32 + bin;
            let prior_warps = warp_offsets.read(location as usize);
            warp_offsets.write(location as usize, prior_warps + exclusive_digit_prefix);
        });
    }
    thread::sync_threads();

    let mut ranks = [MaybeUninit::<i32>::uninit(); ITEMS];
    {
        let mut warp_offsets = shared_memory.region::<i32>(0, (THREADS / 32) * RADIX_DIGITS);
        for_each_thread_item!(ITEMS, item, {
            let item_bin = digit(initialized_value(&keys[item]), current_bit, digit_mask);
            let matching = radix_match_any_8(item_bin);
            let leader = 31 - matching.leading_zeros();
            let within_match = (matching & warp::lanemask_le()).count_ones() as i32;
            let mut warp_offset = 0i32;
            if lane == leader {
                warp_offset = warp_offsets.fetch_add_block(
                    (warp_index * RADIX_DIGITS as u32 + item_bin) as usize,
                    within_match,
                );
            }
            warp_offset = warp::shuffle_sync(u32::MAX, warp_offset as u32, leader) as i32;
            ranks[item].write(warp_offset + within_match - 1);
        });
    }
    thread::sync_threads();

    let mut keys_shared = shared_memory
        .region::<MaybeUninit<B>>(0, UNION_BYTES / core::mem::size_of::<MaybeUninit<B>>());
    for_each_thread_item!(ITEMS, item, {
        let rank = initialized_value(&ranks[item]);
        keys_shared.write(
            rank as usize,
            MaybeUninit::new(initialized_value(&keys[item])),
        );
    });

    // Match CUB's overlap: bin threads may perform lookback after their key
    // scatter while the rest of the block finishes scattering. The following
    // barrier is the sole join before global key reads.
    let mut global_offsets = shared_memory.region::<u64>(UNION_BYTES, RADIX_DIGITS);
    if active_bin {
        global_offsets.write(
            bin as usize,
            pass_bins[bin as usize] - exclusive_digit_prefix as u64,
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

    if full_block {
        for_each_thread_item!(ITEMS, item, {
            let blocked_index = thread_index + item as u32 * THREADS as u32;
            let ordered = keys_shared.read_initialized(blocked_index as usize);
            let item_bin = digit(ordered, current_bit, digit_mask);
            let output_index =
                blocked_index as usize + global_offsets.read(item_bin as usize) as usize;
            output_keys.write(output_index, from_ordered(ordered));
            warp::sync_mask(u32::MAX);
        });
    } else {
        let tile_items = num_items - tile_base;
        for_each_thread_item!(ITEMS, item, {
            let blocked_index = thread_index + item as u32 * THREADS as u32;
            if blocked_index < tile_items {
                let ordered = keys_shared.read_initialized(blocked_index as usize);
                let item_bin = digit(ordered, current_bit, digit_mask);
                let output_index =
                    blocked_index as usize + global_offsets.read(item_bin as usize) as usize;
                output_keys.write(output_index, from_ordered(ordered));
            }
            warp::sync_mask(u32::MAX);
        });
    }

    let mut value_digits = [0u32; ITEMS];
    for_each_thread_item!(ITEMS, item, {
        let blocked_index = thread_index + item as u32 * THREADS as u32;
        let ordered = keys_shared.read_initialized(blocked_index as usize);
        value_digits[item] = digit(ordered, current_bit, digit_mask);
    });

    let mut values = [MaybeUninit::<V>::uninit(); ITEMS];
    if full_block {
        let Some(warp_group) = input_values.get(warp_group_base..warp_group_base + 32usize * ITEMS)
        else {
            return;
        };
        let full_values = FullWarpStripedInput::<V, ITEMS>::new(warp_group, lane_capability);
        for_each_thread_item!(ITEMS, item, {
            values[item].write(full_values.read(item));
        });
    } else {
        let partial_values = PartialWarpStripedInput::<V, ITEMS>::new(
            input_values,
            warp_group_base,
            lane_capability,
        );
        for_each_thread_item!(ITEMS, item, {
            partial_values.load_into(item, &mut values[item]);
        });
    }
    thread::sync_threads();
    let mut values_shared = shared_memory
        .region::<MaybeUninit<V>>(0, UNION_BYTES / core::mem::size_of::<MaybeUninit<V>>());
    for_each_thread_item!(ITEMS, item, {
        let rank = initialized_value(&ranks[item]);
        values_shared.write(rank as usize, values[item]);
    });
    thread::sync_threads();

    if full_block {
        for_each_thread_item!(ITEMS, item, {
            let blocked_index = thread_index + item as u32 * THREADS as u32;
            let value = values_shared.read_initialized(blocked_index as usize);
            let output_index =
                blocked_index as usize + global_offsets.read(value_digits[item] as usize) as usize;
            output_values.write(output_index, value);
            warp::sync_mask(u32::MAX);
        });
    } else {
        let tile_items = num_items - tile_base;
        for_each_thread_item!(ITEMS, item, {
            let blocked_index = thread_index + item as u32 * THREADS as u32;
            if blocked_index < tile_items {
                let value = values_shared.read_initialized(blocked_index as usize);
                let output_index = blocked_index as usize
                    + global_offsets.read(value_digits[item] as usize) as usize;
                output_values.write(output_index, value);
            }
            warp::sync_mask(u32::MAX);
        });
    }
}
