//! CUB-style device-wide radix-sort backend.

use cuda_core::{CudaStream, DriverError, LaunchConfig};
use rccl_memory::{
    CudaAsyncMemoryResource, DeviceAllocator, DeviceBuffer as ResourceBuffer, DeviceSlice,
    DeviceSliceMut, MemoryError,
};
use std::{error::Error, fmt, marker::PhantomData, sync::Arc};

#[doc(hidden)]
pub mod sort_abi {
    use super::*;
    use crate::device::{DeviceRange, DeviceRangeMut};

    const RADIX_BITS: u32 = 8;
    const RADIX_SIZE: usize = 1 << RADIX_BITS;
    const RADIX_PASSES: usize = 32usize.div_ceil(RADIX_BITS as usize);
    const LOOKBACK_PARTIAL: u32 = 1 << 30;
    const LOOKBACK_GLOBAL: u32 = 2 << 30;
    const LOOKBACK_VALUE_MASK: u32 = (1 << 30) - 1;
    const SCAN_WARPS: usize = RADIX_SIZE / 32;
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

    #[doc(hidden)]
    #[derive(Debug)]
    #[non_exhaustive]
    pub enum SortError {
        Driver(DriverError),
        Memory(MemoryError),
        Module(cuda_host::EmbeddedModuleError),
        TooManyItems(usize),
        WorkspaceTooSmall { requested: usize, capacity: usize },
        LengthMismatch { keys: usize, values: usize },
    }

    impl fmt::Display for SortError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Driver(error) => write!(f, "CUDA driver error: {error}"),
                Self::Memory(error) => write!(f, "CUDA memory error: {error}"),
                Self::Module(error) => write!(f, "embedded CUDA module error: {error}"),
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
                Self::LengthMismatch { keys, values } => {
                    write!(f, "key length {keys} does not match value length {values}")
                }
            }
        }
    }

    impl Error for SortError {
        fn source(&self) -> Option<&(dyn Error + 'static)> {
            match self {
                Self::Driver(error) => Some(error),
                Self::Memory(error) => Some(error),
                Self::Module(error) => Some(error),
                _ => None,
            }
        }
    }

    impl From<DriverError> for SortError {
        fn from(value: DriverError) -> Self {
            Self::Driver(value)
        }
    }

    impl From<MemoryError> for SortError {
        fn from(value: MemoryError) -> Self {
            Self::Memory(value)
        }
    }

    impl From<cuda_host::EmbeddedModuleError> for SortError {
        fn from(value: cuda_host::EmbeddedModuleError) -> Self {
            Self::Module(value)
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "kernel macro emits flat CUDA ABI companion functions inside this module"
    )]
    #[cuda_host::cuda_module]
    #[doc(hidden)]
    pub mod kernels {
        use super::*;
        use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
        use cuda_device::cooperative_groups::{block_scan, ops::Sum, this_thread_block};
        use cuda_device::{DynamicSharedArray, SharedArray, kernel, thread, warp};

        // Keep launch-bound calls explicit and absolute so `#[kernel]` forwards
        // them into each generic entry. cuda-oxide's convenience attribute
        // currently adds a u32-to-usize generic-const witness that leaks into
        // the host generic API. Direct unroll markers preserve the same device
        // metadata without that lexical obligation.

        const FULL_MASK: u32 = u32::MAX;

        #[inline(always)]
        fn radix_digit<const DESCENDING: bool>(key: u32, shift: u32) -> u32 {
            let digit = (key >> shift) & ((RADIX_SIZE as u32) - 1);
            if DESCENDING {
                ((RADIX_SIZE as u32) - 1) - digit
            } else {
                digit
            }
        }

        // Keep this helper specialized on the real schedule literal. The
        // cuda-oxide collector emits generic kernel instances reached by the
        // host handle; a nongeneric helper here leaves that handle without a
        // corresponding embedded entry.
        #[kernel]
        pub unsafe fn reset_words_u32<const BLOCK_THREADS: u32>(
            output: DeviceRangeMut<u32>,
            count: u32,
        ) {
            let _ = BLOCK_THREADS;
            let index = thread::index_1d().get() as u32;
            if index < count {
                unsafe { output.write(index as u64, 0) };
            }
        }

        /// One input traversal computes global histograms for all byte passes.
        /// The result is invariant under the permutations performed by sorting.
        #[kernel]
        pub unsafe fn histogram_all_digits_u32<
            const BLOCK_THREADS: u32,
            const ITEMS_PER_THREAD: u32,
            const MIN_BLOCKS_PER_SM: u32,
            const UNROLL: u32,
            const DESCENDING: bool,
        >(
            input: DeviceRange<u32>,
            histograms: DeviceRangeMut<u32>,
            count: u32,
        ) {
            ::cuda_device::thread::__launch_bounds_config::<BLOCK_THREADS, MIN_BLOCKS_PER_SM>();
            static mut HISTOGRAMS: SharedArray<u32, { RADIX_PASSES * RADIX_SIZE }> =
                SharedArray::UNINIT;

            let tid = thread::threadIdx_x();
            let shared = unsafe { SharedArray::as_raw_mut_ptr(&raw mut HISTOGRAMS) };
            let mut slot = tid;
            while slot < (RADIX_PASSES * RADIX_SIZE) as u32 {
                unsafe { shared.add(slot as usize).write(0) };
                slot += BLOCK_THREADS;
            }
            thread::sync_threads();

            let tile = BLOCK_THREADS * ITEMS_PER_THREAD;
            let stride = thread::gridDim_x() * tile;
            let mut base = thread::blockIdx_x() * tile;
            while base < count {
                let mut item = 0;
                while item < ITEMS_PER_THREAD {
                    cuda_device::thread::__unroll_config::<UNROLL>();
                    let index = base + item * BLOCK_THREADS + tid;
                    let valid = index < count;
                    let key = if valid {
                        unsafe { input.read(index as u64) }
                    } else {
                        0
                    };
                    if valid {
                        let mut pass = 0u32;
                        while pass < RADIX_PASSES as u32 {
                            let digit = radix_digit::<DESCENDING>(key, pass * RADIX_BITS);
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
                    let counter = unsafe {
                        DeviceAtomicU32::from_ptr(histograms.as_mut_ptr().add(slot as usize))
                    };
                    counter.fetch_add(value, AtomicOrdering::Relaxed);
                }
                slot += BLOCK_THREADS;
            }
        }

        // Same collector requirement as `reset_words_u32`: specialize the
        // helper on the selected block geometry so its entry is embedded.
        #[kernel]
        pub unsafe fn global_digit_bases<const BLOCK_THREADS: u32>(
            histograms: DeviceRange<u32>,
            bases: DeviceRangeMut<u32>,
        ) {
            let _ = BLOCK_THREADS;
            static mut SCAN_SHARED: SharedArray<u32, SCAN_WARPS> = SharedArray::UNINIT;

            let pass = thread::blockIdx_x();
            let digit = thread::threadIdx_x();
            let index = pass * RADIX_SIZE as u32 + digit;
            let count = unsafe { histograms.read(index as u64) };
            let block = this_thread_block();
            let inclusive = block_scan::<u32, Sum, SCAN_WARPS>(&block, count, &raw mut SCAN_SHARED);
            unsafe { bases.write(index as u64, inclusive.wrapping_sub(count)) };
        }

        /// CUB-style stable block-rank scatter. A warp owns a contiguous input
        /// segment and accumulates one counter per digit. An exclusive scan over
        /// the digit-major warp counters seeds each warp rank. Keeping keys and
        /// ranks live across the scan makes this a single global-memory read.
        #[kernel]
        pub unsafe fn onesweep_u32<
            const BLOCK_THREADS: u32,
            const ITEMS_PER_THREAD: u32,
            const ITEMS_PER_THREAD_USIZE: usize,
            const BLOCK_WARPS: usize,
            const MIN_BLOCKS_PER_SM: u32,
            const UNROLL: u32,
            const DESCENDING: bool,
        >(
            input: DeviceRange<u32>,
            output: DeviceRangeMut<u32>,
            digit_bases: DeviceRange<u32>,
            lookback: DeviceRangeMut<u32>,
            tile_counter: DeviceRangeMut<u32>,
            count: u32,
            shift: u32,
            num_blocks: u32,
        ) {
            ::cuda_device::thread::__launch_bounds_config::<BLOCK_THREADS, MIN_BLOCKS_PER_SM>();
            let tid = thread::threadIdx_x();
            let warp_id = tid / 32;
            let lane = warp::lane_id();
            let tile = BLOCK_THREADS * ITEMS_PER_THREAD;
            let padded_warps = padded_warps(BLOCK_THREADS) as u32;
            let segment = raking_segment(BLOCK_THREADS) as u32;
            let rank_words = raking_storage(BLOCK_THREADS);
            let exchange_words = (tile as usize).max(rank_words);
            let counters = DynamicSharedArray::<u32>::get();
            let rake_scan = DynamicSharedArray::<u32>::offset(exchange_words * size_of::<u32>())
                as *mut SharedArray<u32, BLOCK_WARPS>;
            let global_offsets = DynamicSharedArray::<u32>::offset(
                (exchange_words + BLOCK_WARPS) * size_of::<u32>(),
            );
            let block_index_shared = unsafe { global_offsets.add(RADIX_SIZE) };
            if tid == 0 {
                let counter = unsafe { DeviceAtomicU32::from_ptr(tile_counter.as_mut_ptr()) };
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

            let mut keys = [0u32; ITEMS_PER_THREAD_USIZE];
            let mut ranks = [0u32; ITEMS_PER_THREAD_USIZE];
            let warp_base = base + warp_id * 32 * ITEMS_PER_THREAD;
            let mut item = 0;
            while item < ITEMS_PER_THREAD {
                cuda_device::thread::__unroll_config::<UNROLL>();
                let index = warp_base + item * 32 + lane;
                let valid = index < count;
                let key = if valid {
                    unsafe { input.read(index as u64) }
                } else {
                    0
                };
                keys[item as usize] = key;
                let digit = radix_digit::<DESCENDING>(key, shift);
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
                thread_total = thread_total.wrapping_add(unsafe {
                    counters.add((first_counter + rake_item) as usize).read()
                });
                rake_item += 1;
            }
            let cta = this_thread_block();
            let inclusive = block_scan::<u32, Sum, BLOCK_WARPS>(&cta, thread_total, rake_scan);
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
                    DeviceAtomicU32::from_ptr(
                        lookback
                            .as_mut_ptr()
                            .add((block * RADIX_SIZE as u32 + tid) as usize),
                    )
                };
                descriptor.store(LOOKBACK_PARTIAL | local_count, AtomicOrdering::Release);
            }
            thread::sync_threads();

            item = 0;
            while item < ITEMS_PER_THREAD {
                cuda_device::thread::__unroll_config::<UNROLL>();
                let index = warp_base + item * 32 + lane;
                if index < count {
                    let key = keys[item as usize];
                    let digit = radix_digit::<DESCENDING>(key, shift);
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
            while item < ITEMS_PER_THREAD {
                cuda_device::thread::__unroll_config::<UNROLL>();
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
                                lookback
                                    .as_mut_ptr()
                                    .add((prior_block * RADIX_SIZE as u32 + tid) as usize),
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
                    DeviceAtomicU32::from_ptr(
                        lookback
                            .as_mut_ptr()
                            .add((block * RADIX_SIZE as u32 + tid) as usize),
                    )
                };
                descriptor.store(LOOKBACK_GLOBAL | inclusive_count, AtomicOrdering::Release);
                let pass = shift / RADIX_BITS;
                let prior_count = inclusive_count.wrapping_sub(local_count);
                let global_offset = unsafe {
                    digit_bases
                        .read((pass * RADIX_SIZE as u32 + tid) as u64)
                        .wrapping_add(prior_count)
                        .wrapping_sub(local_prefix)
                };
                unsafe { global_offsets.add(tid as usize).write(global_offset) };
            }
            thread::sync_threads();

            item = 0;
            while item < ITEMS_PER_THREAD {
                cuda_device::thread::__unroll_config::<UNROLL>();
                let block_rank = tid + item * BLOCK_THREADS;
                if block_rank < tile_items {
                    let key = unsafe { counters.add(block_rank as usize).read() };
                    let digit = radix_digit::<DESCENDING>(key, shift);
                    let global_rank = unsafe {
                        global_offsets
                            .add(digit as usize)
                            .read()
                            .wrapping_add(block_rank)
                    };
                    unsafe { output.write(global_rank as u64, key) };
                }
                warp::sync_mask(FULL_MASK);
                item += 1;
            }
        }

        /// Stable OneSweep scatter for `(u32 key, u32 value)` pairs. Values never
        /// participate in ranking; they follow their key through the same stable
        /// rank-order exchange and coalesced global destination.
        #[kernel]
        pub unsafe fn onesweep_pairs_u32<
            const BLOCK_THREADS: u32,
            const ITEMS_PER_THREAD: u32,
            const ITEMS_PER_THREAD_USIZE: usize,
            const BLOCK_WARPS: usize,
            const MIN_BLOCKS_PER_SM: u32,
            const UNROLL: u32,
            const DESCENDING: bool,
        >(
            input_keys: DeviceRange<u32>,
            input_values: DeviceRange<u32>,
            output_keys: DeviceRangeMut<u32>,
            output_values: DeviceRangeMut<u32>,
            digit_bases: DeviceRange<u32>,
            lookback: DeviceRangeMut<u32>,
            tile_counter: DeviceRangeMut<u32>,
            count: u32,
            shift: u32,
            num_blocks: u32,
        ) {
            ::cuda_device::thread::__launch_bounds_config::<BLOCK_THREADS, MIN_BLOCKS_PER_SM>();
            let tid = thread::threadIdx_x();
            let warp_id = tid / 32;
            let lane = warp::lane_id();
            let tile = BLOCK_THREADS * ITEMS_PER_THREAD;
            let padded_warps = padded_warps(BLOCK_THREADS) as u32;
            let segment = raking_segment(BLOCK_THREADS) as u32;
            let rank_words = raking_storage(BLOCK_THREADS);
            let exchange_words = ((tile * 2) as usize).max(rank_words);
            let shared = DynamicSharedArray::<u32>::get();
            let exchange_values = unsafe { shared.add(tile as usize) };
            let rake_scan = DynamicSharedArray::<u32>::offset(exchange_words * size_of::<u32>())
                as *mut SharedArray<u32, BLOCK_WARPS>;
            let global_offsets = DynamicSharedArray::<u32>::offset(
                (exchange_words + BLOCK_WARPS) * size_of::<u32>(),
            );
            let block_index_shared = unsafe { global_offsets.add(RADIX_SIZE) };
            if tid == 0 {
                let counter = unsafe { DeviceAtomicU32::from_ptr(tile_counter.as_mut_ptr()) };
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
                unsafe { shared.add((tid * segment + rake_item) as usize).write(0) };
                rake_item += 1;
            }
            thread::sync_threads();

            let mut keys = [0u32; ITEMS_PER_THREAD_USIZE];
            let mut values = [0u32; ITEMS_PER_THREAD_USIZE];
            let mut ranks = [0u32; ITEMS_PER_THREAD_USIZE];
            let warp_base = base + warp_id * 32 * ITEMS_PER_THREAD;
            let mut item = 0;
            while item < ITEMS_PER_THREAD {
                cuda_device::thread::__unroll_config::<UNROLL>();
                let index = warp_base + item * 32 + lane;
                let valid = index < count;
                let key = if valid {
                    unsafe { input_keys.read(index as u64) }
                } else {
                    0
                };
                let value = if valid {
                    unsafe { input_values.read(index as u64) }
                } else {
                    0
                };
                keys[item as usize] = key;
                values[item as usize] = value;
                let digit = radix_digit::<DESCENDING>(key, shift);
                let valid_mask = warp::ballot_sync(FULL_MASK, valid);
                let peers = warp::match_any_sync(FULL_MASK, digit) & valid_mask;
                let counter_slot = digit * padded_warps + warp_id;
                let counter = unsafe { shared.add(counter_slot as usize) };
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
                thread_total = thread_total.wrapping_add(unsafe {
                    shared.add((first_counter + rake_item) as usize).read()
                });
                rake_item += 1;
            }
            let cta = this_thread_block();
            let inclusive = block_scan::<u32, Sum, BLOCK_WARPS>(&cta, thread_total, rake_scan);
            let mut thread_prefix = inclusive.wrapping_sub(thread_total);
            rake_item = 0;
            while rake_item < segment {
                let slot = unsafe { shared.add((first_counter + rake_item) as usize) };
                let value = unsafe { slot.read() };
                unsafe { slot.write(thread_prefix) };
                thread_prefix = thread_prefix.wrapping_add(value);
                rake_item += 1;
            }
            thread::sync_threads();

            let tile_items = count.saturating_sub(base).min(tile);
            let has_bin = tid < RADIX_SIZE as u32;
            let local_prefix = if has_bin {
                unsafe { shared.add((tid * padded_warps) as usize).read() }
            } else {
                0
            };
            let digit_end = if has_bin {
                if tid + 1 < RADIX_SIZE as u32 {
                    unsafe { shared.add(((tid + 1) * padded_warps) as usize).read() }
                } else {
                    tile_items
                }
            } else {
                0
            };
            let local_count = digit_end.wrapping_sub(local_prefix);
            if has_bin {
                let descriptor = unsafe {
                    DeviceAtomicU32::from_ptr(
                        lookback
                            .as_mut_ptr()
                            .add((block * RADIX_SIZE as u32 + tid) as usize),
                    )
                };
                descriptor.store(LOOKBACK_PARTIAL | local_count, AtomicOrdering::Release);
            }
            thread::sync_threads();

            item = 0;
            while item < ITEMS_PER_THREAD {
                cuda_device::thread::__unroll_config::<UNROLL>();
                let index = warp_base + item * 32 + lane;
                if index < count {
                    let digit = radix_digit::<DESCENDING>(keys[item as usize], shift);
                    let warp_prefix =
                        unsafe { shared.add((digit * padded_warps + warp_id) as usize).read() };
                    ranks[item as usize] = ranks[item as usize].wrapping_add(warp_prefix);
                }
                item += 1;
            }
            thread::sync_threads();

            item = 0;
            while item < ITEMS_PER_THREAD {
                cuda_device::thread::__unroll_config::<UNROLL>();
                let index = warp_base + item * 32 + lane;
                if index < count {
                    let rank = ranks[item as usize] as usize;
                    unsafe {
                        shared.add(rank).write(keys[item as usize]);
                        exchange_values.add(rank).write(values[item as usize]);
                    }
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
                                lookback
                                    .as_mut_ptr()
                                    .add((prior_block * RADIX_SIZE as u32 + tid) as usize),
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
                    DeviceAtomicU32::from_ptr(
                        lookback
                            .as_mut_ptr()
                            .add((block * RADIX_SIZE as u32 + tid) as usize),
                    )
                };
                descriptor.store(LOOKBACK_GLOBAL | inclusive_count, AtomicOrdering::Release);
                let pass = shift / RADIX_BITS;
                let prior_count = inclusive_count.wrapping_sub(local_count);
                let global_offset = unsafe {
                    digit_bases
                        .read((pass * RADIX_SIZE as u32 + tid) as u64)
                        .wrapping_add(prior_count)
                        .wrapping_sub(local_prefix)
                };
                unsafe { global_offsets.add(tid as usize).write(global_offset) };
            }
            thread::sync_threads();

            item = 0;
            while item < ITEMS_PER_THREAD {
                cuda_device::thread::__unroll_config::<UNROLL>();
                let block_rank = tid + item * BLOCK_THREADS;
                if block_rank < tile_items {
                    let key = unsafe { shared.add(block_rank as usize).read() };
                    let value = unsafe { exchange_values.add(block_rank as usize).read() };
                    let digit = radix_digit::<DESCENDING>(key, shift);
                    let global_rank = unsafe {
                        global_offsets
                            .add(digit as usize)
                            .read()
                            .wrapping_add(block_rank)
                    };
                    unsafe {
                        output_keys.write(global_rank as u64, key);
                        output_values.write(global_rank as u64, value);
                    }
                }
                warp::sync_mask(FULL_MASK);
                item += 1;
            }
        }
    }

    /// Affine borrowed workspace capability consumed by one radix dispatch.
    #[doc(hidden)]
    pub struct U32RadixWorkspaceView<'workspace> {
        scratch: DeviceRangeMut<u32>,
        global_histograms: DeviceRangeMut<u32>,
        all_digit_bases: DeviceRangeMut<u32>,
        lookback: DeviceRangeMut<u32>,
        tile_counter: DeviceRangeMut<u32>,
        max_blocks: u32,
        _borrow: PhantomData<&'workspace mut U32SortWorkspaceMarker>,
    }

    /// Affine borrowed key/value workspace consumed by one radix dispatch.
    #[doc(hidden)]
    pub struct U32PairRadixWorkspaceView<'workspace> {
        radix: U32RadixWorkspaceView<'workspace>,
        value_scratch: DeviceRangeMut<u32>,
        _value_borrow: PhantomData<&'workspace mut u32>,
    }

    enum U32SortWorkspaceMarker {}

    #[doc(hidden)]
    pub struct U32SortWorkspace<'stream, A: DeviceAllocator + Clone = CudaAsyncMemoryResource> {
        scratch: ResourceBuffer<'stream, u32, A>,
        global_histograms: ResourceBuffer<'stream, u32, A>,
        all_digit_bases: ResourceBuffer<'stream, u32, A>,
        lookback: ResourceBuffer<'stream, u32, A>,
        tile_counter: ResourceBuffer<'stream, u32, A>,
        capacity: usize,
        max_blocks: u32,
    }

    impl<A: DeviceAllocator + Clone> U32SortWorkspace<'_, A> {
        fn launch_view(&mut self) -> U32RadixWorkspaceView<'_> {
            U32RadixWorkspaceView {
                // SAFETY: each range is bounded by its owning allocation, and
                // the returned affine view cannot outlive this mutable borrow.
                scratch: unsafe {
                    DeviceRangeMut::from_raw_parts(
                        self.scratch.cu_deviceptr() as *mut u32,
                        self.scratch.len(),
                    )
                },
                global_histograms: unsafe {
                    DeviceRangeMut::from_raw_parts(
                        self.global_histograms.cu_deviceptr() as *mut u32,
                        self.global_histograms.len(),
                    )
                },
                all_digit_bases: unsafe {
                    DeviceRangeMut::from_raw_parts(
                        self.all_digit_bases.cu_deviceptr() as *mut u32,
                        self.all_digit_bases.len(),
                    )
                },
                lookback: unsafe {
                    DeviceRangeMut::from_raw_parts(
                        self.lookback.cu_deviceptr() as *mut u32,
                        self.lookback.len(),
                    )
                },
                tile_counter: unsafe {
                    DeviceRangeMut::from_raw_parts(
                        self.tile_counter.cu_deviceptr() as *mut u32,
                        self.tile_counter.len(),
                    )
                },
                max_blocks: self.max_blocks,
                _borrow: PhantomData,
            }
        }

        fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
            self.scratch.ensure_usable_on(stream)?;
            self.global_histograms.ensure_usable_on(stream)?;
            self.all_digit_bases.ensure_usable_on(stream)?;
            self.lookback.ensure_usable_on(stream)?;
            self.tile_counter.ensure_usable_on(stream)
        }
    }

    /// Reusable storage for stable `sort_by_key` on `u32` keys and values.
    #[doc(hidden)]
    pub struct U32PairSortWorkspace<'stream, A: DeviceAllocator + Clone = CudaAsyncMemoryResource> {
        radix: U32SortWorkspace<'stream, A>,
        value_scratch: ResourceBuffer<'stream, u32, A>,
        capacity: usize,
    }

    impl<A: DeviceAllocator + Clone> U32PairSortWorkspace<'_, A> {
        fn launch_view(&mut self) -> U32PairRadixWorkspaceView<'_> {
            // SAFETY: the range is bounded by its owning allocation, and the
            // affine view retains this exclusive borrow until dispatch ends.
            let value_scratch = unsafe {
                DeviceRangeMut::from_raw_parts(
                    self.value_scratch.cu_deviceptr() as *mut u32,
                    self.value_scratch.len(),
                )
            };
            U32PairRadixWorkspaceView {
                radix: self.radix.launch_view(),
                value_scratch,
                _value_borrow: PhantomData,
            }
        }

        fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
            self.radix.ensure_usable_on(stream)?;
            self.value_scratch.ensure_usable_on(stream)
        }
    }

    /// Host launch context used by CUB device-wide radix dispatch.
    ///
    /// CUB supplies resolved target literals, so every kernel call is statically
    /// monomorphized and no launch table survives compilation.
    #[doc(hidden)]
    pub struct RadixLauncher<'a> {
        module: &'a kernels::LoadedModule,
        multiprocessors: u32,
    }

    impl RadixLauncher<'_> {
        #[doc(hidden)]
        #[inline(always)]
        pub unsafe fn __sort_u32_with<
            const BLOCK_THREADS: u32,
            const ITEMS_PER_THREAD: u32,
            const ITEMS_PER_THREAD_USIZE: usize,
            const BLOCK_WARPS: usize,
            const MIN_BLOCKS_PER_SM: u32,
            const UNROLL: u32,
            const DESCENDING: bool,
        >(
            &self,
            stream: &CudaStream,
            keys: DeviceRangeMut<u32>,
            workspace: U32RadixWorkspaceView<'_>,
        ) -> Result<(), SortError> {
            let keys_len = keys.len() as usize;
            let count = u32::try_from(keys_len).map_err(|_| SortError::TooManyItems(keys_len))?;
            let tile = BLOCK_THREADS * ITEMS_PER_THREAD;
            let num_blocks = count.div_ceil(tile).max(1);
            debug_assert!(num_blocks <= workspace.max_blocks);

            let histogram_cfg = LaunchConfig {
                grid_dim: (num_blocks.min(self.multiprocessors * 4).max(1), 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let scatter_cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: ((raking_storage(BLOCK_THREADS)
                    .max((BLOCK_THREADS * ITEMS_PER_THREAD) as usize)
                    + BLOCK_WARPS
                    + RADIX_SIZE
                    + 1)
                    * size_of::<u32>()) as u32,
            };
            let bases_cfg = LaunchConfig {
                grid_dim: (RADIX_PASSES as u32, 1, 1),
                block_dim: (RADIX_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut input = keys.as_read_only();
            let mut output = workspace.scratch;
            let scratch = workspace.scratch;

            unsafe {
                self.module.reset_words_u32::<BLOCK_THREADS>(
                    stream,
                    LaunchConfig::for_num_elems((RADIX_PASSES * RADIX_SIZE) as u32),
                    workspace.global_histograms,
                    (RADIX_PASSES * RADIX_SIZE) as u32,
                )?;
                self.module.histogram_all_digits_u32::<
                    BLOCK_THREADS,
                    ITEMS_PER_THREAD,
                    MIN_BLOCKS_PER_SM,
                    UNROLL,
                    DESCENDING,
                >(
                    stream,
                    histogram_cfg,
                    input,
                    workspace.global_histograms,
                    count,
                )?;
                self.module.global_digit_bases::<BLOCK_THREADS>(
                    stream,
                    bases_cfg,
                    workspace.global_histograms.as_read_only(),
                    workspace.all_digit_bases,
                )?;
            }

            let mut shift = 0;
            while shift < 32 {
                let descriptor_count = num_blocks * RADIX_SIZE as u32;
                unsafe {
                    self.module.reset_words_u32::<BLOCK_THREADS>(
                        stream,
                        LaunchConfig::for_num_elems(descriptor_count),
                        workspace.lookback,
                        descriptor_count,
                    )?;
                    self.module.reset_words_u32::<BLOCK_THREADS>(
                        stream,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        workspace.tile_counter,
                        1,
                    )?;
                    self.module.onesweep_u32::<
                        BLOCK_THREADS,
                        ITEMS_PER_THREAD,
                        ITEMS_PER_THREAD_USIZE,
                        BLOCK_WARPS,
                        MIN_BLOCKS_PER_SM,
                        UNROLL,
                        DESCENDING,
                    >(
                        stream,
                        scatter_cfg,
                        input,
                        output,
                        workspace.all_digit_bases.as_read_only(),
                        workspace.lookback,
                        workspace.tile_counter,
                        count,
                        shift,
                        num_blocks,
                    )?;
                }
                input = output.as_read_only();
                output = if output.same_start(scratch) {
                    keys
                } else {
                    scratch
                };
                shift += RADIX_BITS;
            }
            debug_assert_eq!(input.as_ptr(), keys.as_read_only().as_ptr());
            Ok(())
        }

        #[doc(hidden)]
        #[inline(always)]
        pub unsafe fn __sort_pairs_u32_with<
            const BLOCK_THREADS: u32,
            const ITEMS_PER_THREAD: u32,
            const ITEMS_PER_THREAD_USIZE: usize,
            const BLOCK_WARPS: usize,
            const MIN_BLOCKS_PER_SM: u32,
            const UNROLL: u32,
            const DESCENDING: bool,
        >(
            &self,
            stream: &CudaStream,
            keys: DeviceRangeMut<u32>,
            values: DeviceRangeMut<u32>,
            workspace: U32PairRadixWorkspaceView<'_>,
        ) -> Result<(), SortError> {
            let len = keys.len() as usize;
            let count = u32::try_from(len).map_err(|_| SortError::TooManyItems(len))?;
            let tile = BLOCK_THREADS * ITEMS_PER_THREAD;
            let num_blocks = count.div_ceil(tile).max(1);
            debug_assert!(num_blocks <= workspace.radix.max_blocks);
            let histogram_cfg = LaunchConfig {
                grid_dim: (num_blocks.min(self.multiprocessors * 4).max(1), 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let pair_exchange_words = ((tile * 2) as usize).max(raking_storage(BLOCK_THREADS));
            let scatter_cfg = LaunchConfig {
                grid_dim: (num_blocks, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: ((pair_exchange_words + BLOCK_WARPS + RADIX_SIZE + 1)
                    * size_of::<u32>()) as u32,
            };
            let bases_cfg = LaunchConfig {
                grid_dim: (RADIX_PASSES as u32, 1, 1),
                block_dim: (RADIX_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let key_scratch = workspace.radix.scratch;
            let value_scratch = workspace.value_scratch;
            let mut input_keys = keys.as_read_only();
            let mut input_values = values.as_read_only();
            let mut output_keys = key_scratch;
            let mut output_values = value_scratch;

            unsafe {
                self.module.reset_words_u32::<BLOCK_THREADS>(
                    stream,
                    LaunchConfig::for_num_elems((RADIX_PASSES * RADIX_SIZE) as u32),
                    workspace.radix.global_histograms,
                    (RADIX_PASSES * RADIX_SIZE) as u32,
                )?;
                self.module.histogram_all_digits_u32::<
                    BLOCK_THREADS,
                    ITEMS_PER_THREAD,
                    MIN_BLOCKS_PER_SM,
                    UNROLL,
                    DESCENDING,
                >(
                    stream,
                    histogram_cfg,
                    input_keys,
                    workspace.radix.global_histograms,
                    count,
                )?;
                self.module.global_digit_bases::<BLOCK_THREADS>(
                    stream,
                    bases_cfg,
                    workspace.radix.global_histograms.as_read_only(),
                    workspace.radix.all_digit_bases,
                )?;
            }

            let mut shift = 0;
            while shift < 32 {
                let descriptor_count = num_blocks * RADIX_SIZE as u32;
                unsafe {
                    self.module.reset_words_u32::<BLOCK_THREADS>(
                        stream,
                        LaunchConfig::for_num_elems(descriptor_count),
                        workspace.radix.lookback,
                        descriptor_count,
                    )?;
                    self.module.reset_words_u32::<BLOCK_THREADS>(
                        stream,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        workspace.radix.tile_counter,
                        1,
                    )?;
                    self.module.onesweep_pairs_u32::<
                        BLOCK_THREADS,
                        ITEMS_PER_THREAD,
                        ITEMS_PER_THREAD_USIZE,
                        BLOCK_WARPS,
                        MIN_BLOCKS_PER_SM,
                        UNROLL,
                        DESCENDING,
                    >(
                        stream,
                        scatter_cfg,
                        input_keys,
                        input_values,
                        output_keys,
                        output_values,
                        workspace.radix.all_digit_bases.as_read_only(),
                        workspace.radix.lookback,
                        workspace.radix.tile_counter,
                        count,
                        shift,
                        num_blocks,
                    )?;
                }
                input_keys = output_keys.as_read_only();
                input_values = output_values.as_read_only();
                output_keys = if output_keys.same_start(key_scratch) {
                    keys
                } else {
                    key_scratch
                };
                output_values = if output_values.same_start(value_scratch) {
                    values
                } else {
                    value_scratch
                };
                shift += RADIX_BITS;
            }
            debug_assert_eq!(input_keys.as_ptr(), keys.as_read_only().as_ptr());
            debug_assert_eq!(input_values.as_ptr(), values.as_read_only().as_ptr());
            Ok(())
        }
    }

    #[doc(hidden)]
    pub mod backend {
        use super::*;

        pub struct DeviceRadixSort<A: DeviceAllocator + Clone> {
            module: Arc<kernels::LoadedModule>,
            allocator: A,
            multiprocessors: u32,
        }

        impl<A: DeviceAllocator + Clone> DeviceRadixSort<A> {
            pub fn new(context: &crate::KernelContext, allocator: A) -> Result<Self, SortError> {
                Ok(Self {
                    module: context.loaded_module(|module| kernels::from_module(module.clone()))?,
                    allocator,
                    multiprocessors: context.multiprocessors(),
                })
            }

            pub fn allocate_u32_workspace<
                'stream,
                const BLOCK_THREADS: u32,
                const ITEMS_PER_THREAD: u32,
            >(
                &self,
                stream: &'stream CudaStream,
                capacity: usize,
            ) -> Result<U32SortWorkspace<'stream, A>, SortError> {
                let count =
                    u32::try_from(capacity).map_err(|_| SortError::TooManyItems(capacity))?;
                let tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
                let max_blocks = count.div_ceil(tile_items).max(1);
                let histogram_len = RADIX_SIZE
                    .checked_mul(max_blocks as usize)
                    .ok_or(SortError::TooManyItems(capacity))?;
                Ok(U32SortWorkspace {
                    scratch: ResourceBuffer::zeroed_in(self.allocator.clone(), stream, capacity)?,
                    global_histograms: ResourceBuffer::zeroed_in(
                        self.allocator.clone(),
                        stream,
                        RADIX_PASSES * RADIX_SIZE,
                    )?,
                    all_digit_bases: ResourceBuffer::zeroed_in(
                        self.allocator.clone(),
                        stream,
                        RADIX_PASSES * RADIX_SIZE,
                    )?,
                    lookback: ResourceBuffer::zeroed_in(
                        self.allocator.clone(),
                        stream,
                        histogram_len,
                    )?,
                    tile_counter: ResourceBuffer::zeroed_in(self.allocator.clone(), stream, 1)?,
                    capacity,
                    max_blocks,
                })
            }

            pub fn allocate_u32_pair_workspace<
                'stream,
                const BLOCK_THREADS: u32,
                const ITEMS_PER_THREAD: u32,
            >(
                &self,
                stream: &'stream CudaStream,
                capacity: usize,
            ) -> Result<U32PairSortWorkspace<'stream, A>, SortError> {
                Ok(U32PairSortWorkspace {
                    radix: self.allocate_u32_workspace::<BLOCK_THREADS, ITEMS_PER_THREAD>(
                        stream, capacity,
                    )?,
                    value_scratch: ResourceBuffer::zeroed_in(
                        self.allocator.clone(),
                        stream,
                        capacity,
                    )?,
                    capacity,
                })
            }

            /// Enqueue a radix sort on `stream`.
            ///
            /// # Safety
            ///
            /// `keys` and `workspace` must remain alive and inaccessible until
            /// all work submitted by this call completes, including after an
            /// error returned by a later launch.
            pub unsafe fn sort_u32<
                const BLOCK_THREADS: u32,
                const ITEMS_PER_THREAD: u32,
                const ITEMS_PER_THREAD_USIZE: usize,
                const BLOCK_WARPS: usize,
                const MIN_BLOCKS_PER_SM: u32,
                const UNROLL: u32,
                const DESCENDING: bool,
                B,
            >(
                &self,
                stream: &CudaStream,
                keys: &mut B,
                workspace: &mut U32SortWorkspace<'_, A>,
            ) -> Result<(), SortError>
            where
                B: DeviceSliceMut<u32> + ?Sized,
            {
                keys.ensure_usable_on(stream)?;
                if keys.len() > workspace.capacity {
                    return Err(SortError::WorkspaceTooSmall {
                        requested: keys.len(),
                        capacity: workspace.capacity,
                    });
                }
                if keys.is_empty() {
                    return Ok(());
                }
                workspace.ensure_usable_on(stream)?;
                let launcher = RadixLauncher {
                    module: &self.module,
                    multiprocessors: self.multiprocessors,
                };
                // SAFETY: `keys` owns this initialized device range and the
                // public unsafe contract keeps it exclusively borrowed until
                // the queued kernels complete.
                let keys_range = unsafe {
                    DeviceRangeMut::from_raw_parts(keys.cu_deviceptr() as *mut u32, keys.len())
                };
                unsafe {
                    launcher.__sort_u32_with::<
                        BLOCK_THREADS,
                        ITEMS_PER_THREAD,
                        ITEMS_PER_THREAD_USIZE,
                        BLOCK_WARPS,
                        MIN_BLOCKS_PER_SM,
                        UNROLL,
                        DESCENDING,
                    >(
                        stream,
                        keys_range,
                        workspace.launch_view(),
                    )
                }
            }

            /// Enqueue a stable key/value radix sort on `stream`.
            ///
            /// # Safety
            ///
            /// `keys`, `values`, and `workspace` must remain alive and
            /// inaccessible until all submitted work completes, including after
            /// an error returned by a later launch.
            pub unsafe fn stable_sort_by_key_u32<
                const BLOCK_THREADS: u32,
                const ITEMS_PER_THREAD: u32,
                const ITEMS_PER_THREAD_USIZE: usize,
                const BLOCK_WARPS: usize,
                const MIN_BLOCKS_PER_SM: u32,
                const UNROLL: u32,
                const DESCENDING: bool,
                K,
                V,
            >(
                &self,
                stream: &CudaStream,
                keys: &mut K,
                values: &mut V,
                workspace: &mut U32PairSortWorkspace<'_, A>,
            ) -> Result<(), SortError>
            where
                K: DeviceSliceMut<u32> + ?Sized,
                V: DeviceSliceMut<u32> + ?Sized,
            {
                keys.ensure_usable_on(stream)?;
                values.ensure_usable_on(stream)?;
                if keys.len() != values.len() {
                    return Err(SortError::LengthMismatch {
                        keys: keys.len(),
                        values: values.len(),
                    });
                }
                if keys.len() > workspace.capacity {
                    return Err(SortError::WorkspaceTooSmall {
                        requested: keys.len(),
                        capacity: workspace.capacity,
                    });
                }
                if keys.is_empty() {
                    return Ok(());
                }
                workspace.ensure_usable_on(stream)?;
                let launcher = RadixLauncher {
                    module: &self.module,
                    multiprocessors: self.multiprocessors,
                };
                // SAFETY: both owners cover their initialized ranges, have
                // equal lengths, and remain exclusively borrowed under this
                // method's unsafe enqueue contract.
                let keys_range = unsafe {
                    DeviceRangeMut::from_raw_parts(keys.cu_deviceptr() as *mut u32, keys.len())
                };
                let values_range = unsafe {
                    DeviceRangeMut::from_raw_parts(values.cu_deviceptr() as *mut u32, values.len())
                };
                unsafe {
                    launcher.__sort_pairs_u32_with::<
                        BLOCK_THREADS,
                        ITEMS_PER_THREAD,
                        ITEMS_PER_THREAD_USIZE,
                        BLOCK_WARPS,
                        MIN_BLOCKS_PER_SM,
                        UNROLL,
                        DESCENDING,
                    >(
                        stream,
                        keys_range,
                        values_range,
                        workspace.launch_view(),
                    )
                }
            }
        }
    }
}
