#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::config::Policy;
use rccl_macros::radix_policy_chain;
use std::{error::Error, fmt, sync::Arc};

pub const RADIX_BITS: u32 = 8;
pub const RADIX_SIZE: usize = 1 << RADIX_BITS;
const RADIX_PASSES: usize = 32usize.div_ceil(RADIX_BITS as usize);
const LOOKBACK_PARTIAL: u32 = 1 << 30;
const LOOKBACK_GLOBAL: u32 = 2 << 30;
const LOOKBACK_VALUE_MASK: u32 = (1 << 30) - 1;
const SCAN_ITEMS_PER_THREAD: u32 = 4;
const SCAN_THREADS: u32 = 256;
const SCAN_TILE: u32 = SCAN_THREADS * SCAN_ITEMS_PER_THREAD;
const SCAN_WARPS: usize = (SCAN_THREADS / 32) as usize;
const fn padded_warps(block_threads: u32) -> usize {
    let warps = (block_threads / 32) as usize;
    if warps.is_multiple_of(2) {
        warps + 1
    } else {
        warps
    }
}

const fn raking_segment(block_threads: u32) -> usize {
    (RADIX_SIZE * padded_warps(block_threads)).div_ceil(block_threads as usize)
}

const fn raking_storage(block_threads: u32) -> usize {
    block_threads as usize * raking_segment(block_threads)
}

/// Compile-time tuning vocabulary consumed by the generic radix kernels.
///
/// Concrete implementations are emitted by [`radix_policy_chain!`]. A policy
/// is a zero-sized type parameter, so cuda-oxide folds these values into a
/// distinct PTX specialization rather than passing them to the GPU at runtime.
pub trait RadixPolicy: Policy {
    const MIN_SM: u32;
    const BLOCK_THREADS: u32;
    const ITEMS_PER_THREAD: u32;
    const RADIX_BITS: u32;
    const MIN_BLOCKS_PER_SM: u32;
    const UNROLL: u32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyConfig {
    pub min_sm: u32,
    pub block_threads: u32,
    pub items_per_thread: u32,
    pub radix_bits: u32,
    pub min_blocks_per_sm: u32,
    pub unroll: u32,
}

impl PolicyConfig {
    pub const fn tile_items(self) -> u32 {
        self.block_threads * self.items_per_thread
    }
}

// These are initial, deliberately explicit policy points. They are generated
// as types so every entry becomes a separate cuda-oxide monomorphization. The
// values will be replaced by measured per-architecture tuning, not hidden in
// runtime conditionals.
radix_policy_chain! {
    pub enum RadixPolicySelection: crate::RadixPolicy {
        namespace: 0x7263_636c_5f72_6478;
        Sm80 => { min_sm: 80, threads: 256, items: 12, radix_bits: 8, min_blocks: 2, unroll: 16 },
        Sm86 => { min_sm: 86, threads: 256, items: 12, radix_bits: 8, min_blocks: 2, unroll: 16 },
        Sm89 => { min_sm: 89, threads: 256, items: 16, radix_bits: 8, min_blocks: 2, unroll: 16 },
        Sm90 => { min_sm: 90, threads: 384, items: 16, radix_bits: 8, min_blocks: 2, unroll: 16 },
        Sm100 => { min_sm: 100, threads: 512, items: 16, radix_bits: 8, min_blocks: 1, unroll: 16 },
        Sm120 => { min_sm: 120, threads: 512, items: 16, radix_bits: 8, min_blocks: 1, unroll: 16 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SortOrder {
    #[default]
    Ascending,
    Descending,
}

/// Mirrors Thrust's compile-time smart-sort choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortAlgorithm {
    /// Arithmetic/bit-transformable keys with the built-in less/greater order.
    Radix,
    /// User-defined keys or a custom strict-weak-order comparator.
    Merge,
}

impl SortAlgorithm {
    pub const fn auto(radix_key: bool, builtin_order: bool) -> Self {
        if radix_key && builtin_order {
            Self::Radix
        } else {
            Self::Merge
        }
    }
}

#[derive(Debug)]
pub enum SortError {
    Driver(DriverError),
    Module(cuda_host::EmbeddedModuleError),
    UnsupportedArchitecture(u32),
    TooManyItems(usize),
    WorkspaceTooSmall { requested: usize, capacity: usize },
    WorkspacePolicyMismatch,
}

impl fmt::Display for SortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => write!(f, "CUDA driver error: {error}"),
            Self::Module(error) => write!(f, "embedded CUDA module error: {error}"),
            Self::UnsupportedArchitecture(sm) => {
                write!(
                    f,
                    "sm_{sm} is unsupported; cuda-oxide requires Ampere (sm_80) or newer"
                )
            }
            Self::TooManyItems(count) => {
                write!(
                    f,
                    "{count} items exceed the current u32 device-offset limit"
                )
            }
            Self::WorkspaceTooSmall {
                requested,
                capacity,
            } => write!(
                f,
                "workspace capacity is {capacity} keys, but {requested} were requested"
            ),
            Self::WorkspacePolicyMismatch => {
                write!(f, "workspace was built for a different SM policy")
            }
        }
    }
}

impl Error for SortError {}

impl From<DriverError> for SortError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

impl From<cuda_host::EmbeddedModuleError> for SortError {
    fn from(value: cuda_host::EmbeddedModuleError) -> Self {
        Self::Module(value)
    }
}

#[allow(dead_code)]
#[cuda_host::cuda_module]
mod kernels {
    use super::*;
    use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
    use cuda_device::cooperative_groups::{block_scan, ops::Sum, this_thread_block};
    use cuda_device::{DynamicSharedArray, SharedArray, kernel, launch_bounds, thread, warp};

    const FULL_MASK: u32 = u32::MAX;

    #[inline(always)]
    fn radix_digit(key: u32, shift: u32, descending: bool) -> u32 {
        let digit = (key >> shift) & ((RADIX_SIZE as u32) - 1);
        if descending {
            ((RADIX_SIZE as u32) - 1) - digit
        } else {
            digit
        }
    }

    #[inline(always)]
    fn splitmix64(index: u64) -> u64 {
        let mut value = index.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Stateless random construction: every key is born in device memory.
    #[kernel]
    pub unsafe fn fill_random_u32(output: *mut u32, count: u32, seed: u64) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            let value = splitmix64((index as u64) ^ seed);
            unsafe { output.add(index as usize).write(value as u32) };
        }
    }

    #[kernel]
    pub unsafe fn reset_words_u32(output: *mut u32, count: u32) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            unsafe { output.add(index as usize).write(0) };
        }
    }

    /// One input traversal computes global histograms for all byte passes.
    /// The result is invariant under the permutations performed by sorting.
    #[kernel]
    #[launch_bounds(P::BLOCK_THREADS, P::MIN_BLOCKS_PER_SM)]
    pub unsafe fn histogram_all_digits_u32<P: RadixPolicy>(
        input: *const u32,
        histograms: *mut u32,
        count: u32,
        descending: bool,
    ) where
        [(); P::ITEMS_PER_THREAD as usize]:,
    {
        static mut HISTOGRAMS: SharedArray<u32, { RADIX_PASSES * RADIX_SIZE }> =
            SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HISTOGRAMS) };
        let mut slot = tid;
        while slot < (RADIX_PASSES * RADIX_SIZE) as u32 {
            unsafe { shared.add(slot as usize).write(0) };
            slot += P::BLOCK_THREADS;
        }
        thread::sync_threads();

        let tile = P::BLOCK_THREADS * P::ITEMS_PER_THREAD;
        let stride = thread::gridDim_x() * tile;
        let mut base = thread::blockIdx_x() * tile;
        while base < count {
            let mut item = 0;
            #[unroll(P::UNROLL)]
            while item < P::ITEMS_PER_THREAD {
                let index = base + item * P::BLOCK_THREADS + tid;
                let valid = index < count;
                let key = if valid {
                    unsafe { input.add(index as usize).read() }
                } else {
                    0
                };
                if valid {
                    let mut pass = 0u32;
                    while pass < RADIX_PASSES as u32 {
                        let digit = radix_digit(key, pass * RADIX_BITS, descending);
                        let histogram_slot = pass * RADIX_SIZE as u32 + digit;
                        let counter = unsafe {
                            BlockAtomicU32::from_ptr(shared.add(histogram_slot as usize))
                        };
                        counter.fetch_add(1, AtomicOrdering::Relaxed);
                        pass += 1;
                    }
                }
                item += 1;
            }
            base += stride;
        }
        thread::sync_threads();

        slot = tid;
        while slot < (RADIX_PASSES * RADIX_SIZE) as u32 {
            let value = unsafe { shared.add(slot as usize).read() };
            if value != 0 {
                let counter = unsafe { DeviceAtomicU32::from_ptr(histograms.add(slot as usize)) };
                counter.fetch_add(value, AtomicOrdering::Relaxed);
            }
            slot += P::BLOCK_THREADS;
        }
    }

    #[kernel]
    pub unsafe fn global_digit_bases(histograms: *const u32, bases: *mut u32) {
        static mut SCAN_SHARED: SharedArray<u32, SCAN_WARPS> = SharedArray::UNINIT;

        let pass = thread::blockIdx_x();
        let digit = thread::threadIdx_x();
        let index = pass * RADIX_SIZE as u32 + digit;
        let count = unsafe { histograms.add(index as usize).read() };
        let block = this_thread_block();
        let inclusive = block_scan::<u32, Sum, SCAN_WARPS>(&block, count, &raw mut SCAN_SHARED);
        unsafe {
            bases
                .add(index as usize)
                .write(inclusive.wrapping_sub(count))
        };
    }

    /// Per-CTA 256-bin histogram. Warp match aggregation reduces shared-memory
    /// atomics from one per key to one per distinct digit in a warp.
    #[kernel]
    #[launch_bounds(P::BLOCK_THREADS, P::MIN_BLOCKS_PER_SM)]
    pub unsafe fn histogram_u32<P: RadixPolicy>(
        input: *const u32,
        block_histograms: *mut u32,
        count: u32,
        shift: u32,
        num_blocks: u32,
        descending: bool,
    ) where
        [(); P::ITEMS_PER_THREAD as usize]:,
    {
        static mut HISTOGRAM: SharedArray<u32, RADIX_SIZE> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let block = thread::blockIdx_x();
        let mut digit_slot = tid;
        while digit_slot < RADIX_SIZE as u32 {
            unsafe { HISTOGRAM[digit_slot as usize] = 0 };
            digit_slot += P::BLOCK_THREADS;
        }
        thread::sync_threads();

        let tile = P::BLOCK_THREADS * P::ITEMS_PER_THREAD;
        let base = block * tile;
        let lane = warp::lane_id();
        let histogram = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HISTOGRAM) };
        let mut item = 0;
        #[unroll(P::UNROLL)]
        while item < P::ITEMS_PER_THREAD {
            let index = base + item * P::BLOCK_THREADS + tid;
            let valid = index < count;
            let key = if valid {
                unsafe { input.add(index as usize).read() }
            } else {
                0
            };
            let digit = radix_digit(key, shift, descending);
            let valid_mask = warp::ballot_sync(FULL_MASK, valid);
            let peers = warp::match_any_sync(FULL_MASK, digit) & valid_mask;
            if peers != 0 && lane == peers.trailing_zeros() {
                let counter = unsafe { BlockAtomicU32::from_ptr(histogram.add(digit as usize)) };
                counter.fetch_add(peers.count_ones(), AtomicOrdering::Relaxed);
            }
            item += 1;
        }
        thread::sync_threads();

        digit_slot = tid;
        while digit_slot < RADIX_SIZE as u32 {
            let output_index = digit_slot * num_blocks + block;
            unsafe {
                block_histograms
                    .add(output_index as usize)
                    .write(HISTOGRAM[digit_slot as usize]);
            }
            digit_slot += P::BLOCK_THREADS;
        }
    }

    /// First level: exclusive-scan 1024 consecutive CTA counts for each digit.
    #[kernel]
    pub unsafe fn scan_block_histograms(
        histograms: *const u32,
        offsets: *mut u32,
        chunk_sums: *mut u32,
        num_blocks: u32,
        num_chunks: u32,
    ) {
        static mut SCAN_SHARED: SharedArray<u32, SCAN_WARPS> = SharedArray::UNINIT;

        let digit = thread::blockIdx_y();
        let chunk = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let row_base = digit * num_blocks;
        let first = chunk * SCAN_TILE + tid * SCAN_ITEMS_PER_THREAD;
        let mut local = [0u32; SCAN_ITEMS_PER_THREAD as usize];
        let mut running = 0u32;
        let mut item = 0;
        while item < SCAN_ITEMS_PER_THREAD {
            local[item as usize] = running;
            let index = first + item;
            if index < num_blocks {
                running = running
                    .wrapping_add(unsafe { histograms.add((row_base + index) as usize).read() });
            }
            item += 1;
        }

        let block = this_thread_block();
        let inclusive = block_scan::<u32, Sum, SCAN_WARPS>(&block, running, &raw mut SCAN_SHARED);
        let thread_prefix = inclusive.wrapping_sub(running);
        item = 0;
        while item < SCAN_ITEMS_PER_THREAD {
            let index = first + item;
            if index < num_blocks {
                unsafe {
                    offsets
                        .add((row_base + index) as usize)
                        .write(thread_prefix.wrapping_add(local[item as usize]));
                }
            }
            item += 1;
        }
        if tid == SCAN_THREADS - 1 {
            unsafe {
                chunk_sums
                    .add((digit * num_chunks + chunk) as usize)
                    .write(inclusive);
            }
        }
    }

    /// Second level: one CTA scans all first-level sums for one digit.
    #[kernel]
    pub unsafe fn scan_chunk_sums(chunk_sums: *mut u32, num_chunks: u32) {
        static mut SCAN_SHARED: SharedArray<u32, SCAN_WARPS> = SharedArray::UNINIT;

        let digit = thread::blockIdx_y();
        let tid = thread::threadIdx_x();
        let first = tid * SCAN_ITEMS_PER_THREAD;
        let mut local = [0u32; SCAN_ITEMS_PER_THREAD as usize];
        let mut running = 0u32;
        let mut item = 0;
        while item < SCAN_ITEMS_PER_THREAD {
            local[item as usize] = running;
            let index = first + item;
            if index < num_chunks {
                running = running.wrapping_add(unsafe {
                    chunk_sums.add((digit * num_chunks + index) as usize).read()
                });
            }
            item += 1;
        }

        let block = this_thread_block();
        let inclusive = block_scan::<u32, Sum, SCAN_WARPS>(&block, running, &raw mut SCAN_SHARED);
        let thread_prefix = inclusive.wrapping_sub(running);
        item = 0;
        while item < SCAN_ITEMS_PER_THREAD {
            let index = first + item;
            if index < num_chunks {
                unsafe {
                    chunk_sums
                        .add((digit * num_chunks + index) as usize)
                        .write(thread_prefix.wrapping_add(local[item as usize]));
                }
            }
            item += 1;
        }
    }

    #[kernel]
    pub unsafe fn add_chunk_prefixes(
        offsets: *mut u32,
        chunk_prefixes: *const u32,
        num_blocks: u32,
        num_chunks: u32,
    ) {
        let digit = thread::blockIdx_y();
        let chunk = thread::blockIdx_x();
        let tid = thread::threadIdx_x();
        let prefix = unsafe {
            chunk_prefixes
                .add((digit * num_chunks + chunk) as usize)
                .read()
        };
        let first = chunk * SCAN_TILE + tid * SCAN_ITEMS_PER_THREAD;
        let mut item = 0;
        while item < SCAN_ITEMS_PER_THREAD {
            let index = first + item;
            if index < num_blocks {
                let slot = unsafe { offsets.add((digit * num_blocks + index) as usize) };
                unsafe { slot.write(slot.read().wrapping_add(prefix)) };
            }
            item += 1;
        }
    }

    /// Exclusive prefix across the 256 digit totals gives each digit's global
    /// output base for this radix pass.
    #[kernel]
    pub unsafe fn digit_bases(
        histograms: *const u32,
        offsets: *const u32,
        bases: *mut u32,
        num_blocks: u32,
    ) {
        static mut SCAN_SHARED: SharedArray<u32, SCAN_WARPS> = SharedArray::UNINIT;

        let digit = thread::threadIdx_x();
        let last = digit * num_blocks + (num_blocks - 1);
        let total = unsafe {
            offsets
                .add(last as usize)
                .read()
                .wrapping_add(histograms.add(last as usize).read())
        };
        let block = this_thread_block();
        let inclusive = block_scan::<u32, Sum, SCAN_WARPS>(&block, total, &raw mut SCAN_SHARED);
        unsafe {
            bases
                .add(digit as usize)
                .write(inclusive.wrapping_sub(total));
        }
    }

    /// CUB-style stable block-rank scatter. A warp owns a contiguous input
    /// segment and accumulates one counter per digit. An exclusive scan over
    /// the digit-major warp counters seeds each warp rank. Keeping keys and
    /// ranks live across the scan makes this a single global-memory read.
    #[kernel]
    #[launch_bounds(P::BLOCK_THREADS, P::MIN_BLOCKS_PER_SM)]
    pub unsafe fn onesweep_u32<P: RadixPolicy>(
        input: *const u32,
        output: *mut u32,
        digit_bases: *const u32,
        lookback: *mut u32,
        tile_counter: *mut u32,
        count: u32,
        shift: u32,
        num_blocks: u32,
        descending: bool,
    ) where
        [(); P::ITEMS_PER_THREAD as usize]:,
        [(); (P::BLOCK_THREADS / 32) as usize]:,
    {
        let tid = thread::threadIdx_x();
        let warp_id = tid / 32;
        let lane = warp::lane_id();
        let tile = P::BLOCK_THREADS * P::ITEMS_PER_THREAD;
        let padded_warps = padded_warps(P::BLOCK_THREADS) as u32;
        let segment = raking_segment(P::BLOCK_THREADS) as u32;
        let rank_words = raking_storage(P::BLOCK_THREADS);
        let exchange_words = (tile as usize).max(rank_words);
        let counters = DynamicSharedArray::<u32>::get();
        let rake_scan = DynamicSharedArray::<u32>::offset(exchange_words * size_of::<u32>())
            as *mut SharedArray<u32, { (P::BLOCK_THREADS / 32) as usize }>;
        let global_offsets = DynamicSharedArray::<u32>::offset(
            (exchange_words + (P::BLOCK_THREADS / 32) as usize) * size_of::<u32>(),
        );
        let block_index_shared = unsafe { global_offsets.add(RADIX_SIZE) };
        if tid == 0 {
            let counter = unsafe { DeviceAtomicU32::from_ptr(tile_counter) };
            let block = counter.fetch_add(1, AtomicOrdering::Relaxed);
            unsafe { block_index_shared.write(block) };
        }
        thread::sync_threads();
        let block = unsafe { block_index_shared.read() };
        if block >= num_blocks {
            return;
        }
        let base = block * tile;
        let mut rake_item = 0;
        while rake_item < segment {
            unsafe { counters.add((tid * segment + rake_item) as usize).write(0) };
            rake_item += 1;
        }
        thread::sync_threads();

        let mut keys = [0u32; P::ITEMS_PER_THREAD as usize];
        let mut ranks = [0u32; P::ITEMS_PER_THREAD as usize];
        let warp_base = base + warp_id * 32 * P::ITEMS_PER_THREAD;
        let mut item = 0;
        #[unroll(P::UNROLL)]
        while item < P::ITEMS_PER_THREAD {
            let index = warp_base + item * 32 + lane;
            let valid = index < count;
            let key = if valid {
                unsafe { input.add(index as usize).read() }
            } else {
                0
            };
            keys[item as usize] = key;
            let digit = radix_digit(key, shift, descending);
            let valid_mask = warp::ballot_sync(FULL_MASK, valid);
            let peers = warp::match_any_sync(FULL_MASK, digit) & valid_mask;
            let counter_slot = digit * padded_warps + warp_id;
            let counter = unsafe { counters.add(counter_slot as usize) };
            let prior = if valid {
                unsafe { counter.read_volatile() }
            } else {
                0
            };
            warp::sync_mask(FULL_MASK);
            if valid {
                let leader = peers.trailing_zeros();
                if lane == leader {
                    unsafe { counter.write_volatile(prior.wrapping_add(peers.count_ones())) };
                }
                ranks[item as usize] =
                    prior.wrapping_add((peers & warp::lanemask_lt()).count_ones());
            }
            warp::sync_mask(FULL_MASK);
            item += 1;
        }
        thread::sync_threads();

        let first_counter = tid * segment;
        let mut thread_total = 0u32;
        rake_item = 0;
        while rake_item < segment {
            thread_total = thread_total
                .wrapping_add(unsafe { counters.add((first_counter + rake_item) as usize).read() });
            rake_item += 1;
        }
        let cta = this_thread_block();
        let inclusive = block_scan::<u32, Sum, { (P::BLOCK_THREADS / 32) as usize }>(
            &cta,
            thread_total,
            rake_scan,
        );
        let mut thread_prefix = inclusive.wrapping_sub(thread_total);
        rake_item = 0;
        while rake_item < segment {
            let slot = unsafe { counters.add((first_counter + rake_item) as usize) };
            let value = unsafe { slot.read() };
            unsafe { slot.write(thread_prefix) };
            thread_prefix = thread_prefix.wrapping_add(value);
            rake_item += 1;
        }
        thread::sync_threads();

        let tile_items = count.saturating_sub(base).min(tile);
        let has_bin = tid < RADIX_SIZE as u32;
        let local_prefix = if has_bin {
            unsafe { counters.add((tid * padded_warps) as usize).read() }
        } else {
            0
        };
        let digit_end = if has_bin {
            if tid + 1 < RADIX_SIZE as u32 {
                unsafe { counters.add(((tid + 1) * padded_warps) as usize).read() }
            } else {
                tile_items
            }
        } else {
            0
        };
        let local_count = digit_end.wrapping_sub(local_prefix);
        if has_bin {
            let descriptor = unsafe {
                DeviceAtomicU32::from_ptr(lookback.add((block * RADIX_SIZE as u32 + tid) as usize))
            };
            descriptor.store(LOOKBACK_PARTIAL | local_count, AtomicOrdering::Release);
        }
        thread::sync_threads();

        item = 0;
        #[unroll(P::UNROLL)]
        while item < P::ITEMS_PER_THREAD {
            let index = warp_base + item * 32 + lane;
            if index < count {
                let key = keys[item as usize];
                let digit = radix_digit(key, shift, descending);
                let warp_prefix = unsafe {
                    counters
                        .add((digit * padded_warps + warp_id) as usize)
                        .read()
                };
                ranks[item as usize] = ranks[item as usize].wrapping_add(warp_prefix);
            }
            item += 1;
        }
        thread::sync_threads();

        item = 0;
        #[unroll(P::UNROLL)]
        while item < P::ITEMS_PER_THREAD {
            let index = warp_base + item * 32 + lane;
            if index < count {
                unsafe {
                    counters
                        .add(ranks[item as usize] as usize)
                        .write(keys[item as usize])
                };
            }
            item += 1;
        }
        thread::sync_threads();

        if has_bin {
            let mut inclusive_count = local_count;
            if block != 0 {
                let mut prior_block = block - 1;
                loop {
                    let prior_descriptor = unsafe {
                        DeviceAtomicU32::from_ptr(
                            lookback.add((prior_block * RADIX_SIZE as u32 + tid) as usize),
                        )
                    };
                    let mut value = prior_descriptor.load(AtomicOrdering::Acquire);
                    while value == 0 {
                        value = prior_descriptor.load(AtomicOrdering::Acquire);
                    }
                    inclusive_count = inclusive_count.wrapping_add(value & LOOKBACK_VALUE_MASK);
                    if value & LOOKBACK_GLOBAL != 0 || prior_block == 0 {
                        break;
                    }
                    prior_block -= 1;
                }
            }

            let descriptor = unsafe {
                DeviceAtomicU32::from_ptr(lookback.add((block * RADIX_SIZE as u32 + tid) as usize))
            };
            descriptor.store(LOOKBACK_GLOBAL | inclusive_count, AtomicOrdering::Release);
            let pass = shift / RADIX_BITS;
            let prior_count = inclusive_count.wrapping_sub(local_count);
            let global_offset = unsafe {
                digit_bases
                    .add((pass * RADIX_SIZE as u32 + tid) as usize)
                    .read()
                    .wrapping_add(prior_count)
                    .wrapping_sub(local_prefix)
            };
            unsafe { global_offsets.add(tid as usize).write(global_offset) };
        }
        thread::sync_threads();

        item = 0;
        #[unroll(P::UNROLL)]
        while item < P::ITEMS_PER_THREAD {
            let block_rank = tid + item * P::BLOCK_THREADS;
            if block_rank < tile_items {
                let key = unsafe { counters.add(block_rank as usize).read() };
                let digit = radix_digit(key, shift, descending);
                let global_rank = unsafe {
                    global_offsets
                        .add(digit as usize)
                        .read()
                        .wrapping_add(block_rank)
                };
                unsafe { output.add(global_rank as usize).write(key) };
            }
            warp::sync_mask(FULL_MASK);
            item += 1;
        }
    }

    #[kernel]
    pub unsafe fn reset_validation(status: *mut u32) {
        if thread::index_1d().get() == 0 {
            unsafe { status.write(0) };
        }
    }

    #[kernel]
    pub unsafe fn validate_order_u32(
        input: *const u32,
        count: u32,
        descending: bool,
        status: *mut u32,
    ) {
        let index = thread::index_1d().get() as u32;
        if index + 1 < count {
            let left = unsafe { input.add(index as usize).read() };
            let right = unsafe { input.add(index as usize + 1).read() };
            let invalid = if descending {
                left < right
            } else {
                left > right
            };
            if invalid {
                let errors = unsafe { DeviceAtomicU32::from_ptr(status) };
                errors.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }
}

pub struct U32SortWorkspace {
    scratch: DeviceBuffer<u32>,
    global_histograms: DeviceBuffer<u32>,
    all_digit_bases: DeviceBuffer<u32>,
    lookback: DeviceBuffer<u32>,
    tile_counter: DeviceBuffer<u32>,
    validation: DeviceBuffer<u32>,
    capacity: usize,
    max_blocks: u32,
    policy: RadixPolicySelection,
}

impl U32SortWorkspace {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn temporary_bytes(&self) -> usize {
        self.scratch.num_bytes()
            + self.global_histograms.num_bytes()
            + self.all_digit_bases.num_bytes()
            + self.lookback.num_bytes()
            + self.tile_counter.num_bytes()
            + self.validation.num_bytes()
    }
}

pub struct GpuSorter {
    module: kernels::LoadedModule,
    policy: RadixPolicySelection,
    sm: u32,
    multiprocessors: u32,
}

impl GpuSorter {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, SortError> {
        let (major, minor) = context.compute_capability()?;
        let sm = (major * 10 + minor) as u32;
        let policy =
            RadixPolicySelection::select(sm).ok_or(SortError::UnsupportedArchitecture(sm))?;
        let module = kernels::load(context)?;
        let multiprocessors = context.multiprocessor_count()?;
        Ok(Self {
            module,
            policy,
            sm,
            multiprocessors,
        })
    }

    pub fn sm(&self) -> u32 {
        self.sm
    }

    pub fn policy_name(&self) -> &'static str {
        self.policy.name()
    }

    pub fn policy_config(&self) -> PolicyConfig {
        self.policy.config()
    }

    pub fn allocate_u32_workspace(
        &self,
        stream: &Arc<CudaStream>,
        capacity: usize,
    ) -> Result<U32SortWorkspace, SortError> {
        let count = u32::try_from(capacity).map_err(|_| SortError::TooManyItems(capacity))?;
        let tile = self.policy.config().tile_items();
        let max_blocks = count.div_ceil(tile).max(1);
        let histogram_len = RADIX_SIZE
            .checked_mul(max_blocks as usize)
            .ok_or(SortError::TooManyItems(capacity))?;

        Ok(U32SortWorkspace {
            scratch: DeviceBuffer::zeroed(stream, capacity)?,
            global_histograms: DeviceBuffer::zeroed(stream, RADIX_PASSES * RADIX_SIZE)?,
            all_digit_bases: DeviceBuffer::zeroed(stream, RADIX_PASSES * RADIX_SIZE)?,
            lookback: DeviceBuffer::zeroed(stream, histogram_len)?,
            tile_counter: DeviceBuffer::zeroed(stream, 1)?,
            validation: DeviceBuffer::zeroed(stream, 1)?,
            capacity,
            max_blocks,
            policy: self.policy,
        })
    }

    /// Fill a device allocation without constructing or staging keys on the CPU.
    pub fn fill_random_u32(
        &self,
        stream: &Arc<CudaStream>,
        keys: &mut DeviceBuffer<u32>,
        seed: u64,
    ) -> Result<(), SortError> {
        let count = u32::try_from(keys.len()).map_err(|_| SortError::TooManyItems(keys.len()))?;
        if count == 0 {
            return Ok(());
        }
        // SAFETY: one thread is launched per key and the pointer owns `count`
        // writable u32 values until the stream completes.
        unsafe {
            self.module.fill_random_u32(
                stream,
                LaunchConfig::for_num_elems(count),
                keys.cu_deviceptr() as *mut u32,
                count,
                seed,
            )?;
        }
        Ok(())
    }

    pub fn sort_u32(
        &self,
        stream: &Arc<CudaStream>,
        keys: &mut DeviceBuffer<u32>,
        workspace: &mut U32SortWorkspace,
        order: SortOrder,
    ) -> Result<(), SortError> {
        if keys.len() > workspace.capacity {
            return Err(SortError::WorkspaceTooSmall {
                requested: keys.len(),
                capacity: workspace.capacity,
            });
        }
        if workspace.policy != self.policy {
            return Err(SortError::WorkspacePolicyMismatch);
        }
        if keys.is_empty() {
            return Ok(());
        }
        match self.policy {
            RadixPolicySelection::Sm80 => {
                self.sort_u32_with::<Sm80>(stream, keys, workspace, order)
            }
            RadixPolicySelection::Sm86 => {
                self.sort_u32_with::<Sm86>(stream, keys, workspace, order)
            }
            RadixPolicySelection::Sm89 => {
                self.sort_u32_with::<Sm89>(stream, keys, workspace, order)
            }
            RadixPolicySelection::Sm90 => {
                self.sort_u32_with::<Sm90>(stream, keys, workspace, order)
            }
            RadixPolicySelection::Sm100 => {
                self.sort_u32_with::<Sm100>(stream, keys, workspace, order)
            }
            RadixPolicySelection::Sm120 => {
                self.sort_u32_with::<Sm120>(stream, keys, workspace, order)
            }
        }
    }

    fn sort_u32_with<P: RadixPolicy>(
        &self,
        stream: &Arc<CudaStream>,
        keys: &mut DeviceBuffer<u32>,
        workspace: &mut U32SortWorkspace,
        order: SortOrder,
    ) -> Result<(), SortError>
    where
        [(); P::UNROLL as usize]:,
        [(); P::BLOCK_THREADS as usize]:,
        [(); P::MIN_BLOCKS_PER_SM as usize]:,
        [(); P::ITEMS_PER_THREAD as usize]:,
        [(); (P::BLOCK_THREADS / 32) as usize]:,
    {
        let count = u32::try_from(keys.len()).map_err(|_| SortError::TooManyItems(keys.len()))?;
        let tile = P::BLOCK_THREADS * P::ITEMS_PER_THREAD;
        let num_blocks = count.div_ceil(tile).max(1);
        debug_assert!(num_blocks <= workspace.max_blocks);

        let histogram_cfg = LaunchConfig {
            grid_dim: (num_blocks.min(self.multiprocessors * 4).max(1), 1, 1),
            block_dim: (P::BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let scatter_cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (P::BLOCK_THREADS, 1, 1),
            shared_mem_bytes: ((raking_storage(P::BLOCK_THREADS)
                .max((P::BLOCK_THREADS * P::ITEMS_PER_THREAD) as usize)
                + (P::BLOCK_THREADS / 32) as usize
                + RADIX_SIZE
                + 1)
                * size_of::<u32>()) as u32,
        };
        let bases_cfg = LaunchConfig {
            grid_dim: (RADIX_PASSES as u32, 1, 1),
            block_dim: (RADIX_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let descending = order == SortOrder::Descending;
        let mut input = keys.cu_deviceptr() as *const u32;
        let mut output = workspace.scratch.cu_deviceptr() as *mut u32;
        let keys_ptr = keys.cu_deviceptr() as *mut u32;
        let scratch_ptr = workspace.scratch.cu_deviceptr() as *mut u32;

        unsafe {
            self.module.reset_words_u32(
                stream,
                LaunchConfig::for_num_elems((RADIX_PASSES * RADIX_SIZE) as u32),
                workspace.global_histograms.cu_deviceptr() as *mut u32,
                (RADIX_PASSES * RADIX_SIZE) as u32,
            )?;
            self.module.histogram_all_digits_u32::<P>(
                stream,
                histogram_cfg,
                input,
                workspace.global_histograms.cu_deviceptr() as *mut u32,
                count,
                descending,
            )?;
            self.module.global_digit_bases(
                stream,
                bases_cfg,
                workspace.global_histograms.cu_deviceptr() as *const u32,
                workspace.all_digit_bases.cu_deviceptr() as *mut u32,
            )?;
        }

        let mut shift = 0;
        while shift < 32 {
            let descriptor_count = num_blocks * RADIX_SIZE as u32;

            // SAFETY: logical tile IDs are assigned in launch order, every
            // digit publishes a release-ordered lookback descriptor, and the
            // rank exchange establishes a unique destination for every key.
            unsafe {
                self.module.reset_words_u32(
                    stream,
                    LaunchConfig::for_num_elems(descriptor_count),
                    workspace.lookback.cu_deviceptr() as *mut u32,
                    descriptor_count,
                )?;
                self.module.reset_words_u32(
                    stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    workspace.tile_counter.cu_deviceptr() as *mut u32,
                    1,
                )?;
                self.module.onesweep_u32::<P>(
                    stream,
                    scatter_cfg,
                    input,
                    output,
                    workspace.all_digit_bases.cu_deviceptr() as *const u32,
                    workspace.lookback.cu_deviceptr() as *mut u32,
                    workspace.tile_counter.cu_deviceptr() as *mut u32,
                    count,
                    shift,
                    num_blocks,
                    descending,
                )?;
            }
            input = output as *const u32;
            output = if output == scratch_ptr {
                keys_ptr
            } else {
                scratch_ptr
            };
            shift += RADIX_BITS;
        }
        debug_assert_eq!(input, keys_ptr as *const u32);
        Ok(())
    }

    /// Device-side order validation. Only the final 4-byte error count crosses
    /// to the host; no key or value data is copied or materialized on the CPU.
    pub fn validate_u32(
        &self,
        stream: &Arc<CudaStream>,
        keys: &DeviceBuffer<u32>,
        workspace: &mut U32SortWorkspace,
        order: SortOrder,
    ) -> Result<u32, SortError> {
        let count = u32::try_from(keys.len()).map_err(|_| SortError::TooManyItems(keys.len()))?;
        if count < 2 {
            return Ok(0);
        }
        let one_thread = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: validation owns one writable status word, and every launched
        // comparison reads a valid adjacent pair or exits before dereference.
        unsafe {
            self.module.reset_validation(
                stream,
                one_thread,
                workspace.validation.cu_deviceptr() as *mut u32,
            )?;
            self.module.validate_order_u32(
                stream,
                LaunchConfig::for_num_elems(count - 1),
                keys.cu_deviceptr() as *const u32,
                count,
                order == SortOrder::Descending,
                workspace.validation.cu_deviceptr() as *mut u32,
            )?;
        }
        let status = workspace.validation.to_host_vec(stream)?;
        Ok(status[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_policy_chain_selects_latest_compatible_specialization() {
        assert_eq!(RadixPolicySelection::select(79), None);
        assert_eq!(
            RadixPolicySelection::select(80),
            Some(RadixPolicySelection::Sm80)
        );
        assert_eq!(
            RadixPolicySelection::select(87),
            Some(RadixPolicySelection::Sm86)
        );
        assert_eq!(
            RadixPolicySelection::select(89),
            Some(RadixPolicySelection::Sm89)
        );
        assert_eq!(
            RadixPolicySelection::select(95),
            Some(RadixPolicySelection::Sm90)
        );
        assert_eq!(
            RadixPolicySelection::select(120),
            Some(RadixPolicySelection::Sm120)
        );
    }

    #[test]
    fn smart_sort_matches_thrust_dispatch_boundary() {
        assert_eq!(SortAlgorithm::auto(true, true), SortAlgorithm::Radix);
        assert_eq!(SortAlgorithm::auto(true, false), SortAlgorithm::Merge);
        assert_eq!(SortAlgorithm::auto(false, true), SortAlgorithm::Merge);
    }

    #[test]
    fn generated_policy_shapes_are_valid() {
        for policy in [
            RadixPolicySelection::Sm80,
            RadixPolicySelection::Sm86,
            RadixPolicySelection::Sm89,
            RadixPolicySelection::Sm90,
            RadixPolicySelection::Sm100,
            RadixPolicySelection::Sm120,
        ] {
            let config = policy.config();
            assert_eq!(config.block_threads % 32, 0);
            assert!((128..=1024).contains(&config.block_threads));
            assert_eq!(config.radix_bits, RADIX_BITS);
        }
    }
}
