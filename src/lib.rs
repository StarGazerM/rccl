#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::config::Policy;
use rccl_macros::radix_policy_chain;
use std::{error::Error, fmt, sync::Arc};

pub const RADIX_BITS: u32 = 4;
pub const RADIX_SIZE: usize = 1 << RADIX_BITS;
const SCAN_ITEMS_PER_THREAD: u32 = 4;
const SCAN_THREADS: u32 = 256;
const SCAN_TILE: u32 = SCAN_THREADS * SCAN_ITEMS_PER_THREAD;
const SCAN_WARPS: usize = (SCAN_THREADS / 32) as usize;
const MAX_WARPS: usize = 32;

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
        Sm80 => { min_sm: 80, threads: 256, items: 12, radix_bits: 4, min_blocks: 2, unroll: 4 },
        Sm86 => { min_sm: 86, threads: 256, items: 12, radix_bits: 4, min_blocks: 2, unroll: 4 },
        Sm89 => { min_sm: 89, threads: 256, items: 16, radix_bits: 4, min_blocks: 2, unroll: 4 },
        Sm90 => { min_sm: 90, threads: 384, items: 16, radix_bits: 4, min_blocks: 2, unroll: 4 },
        Sm100 => { min_sm: 100, threads: 512, items: 16, radix_bits: 4, min_blocks: 1, unroll: 8 },
        Sm120 => { min_sm: 120, threads: 512, items: 16, radix_bits: 4, min_blocks: 1, unroll: 8 }
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
    ScanHierarchyTooDeep { chunks: u32 },
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
            Self::ScanHierarchyTooDeep { chunks } => write!(
                f,
                "{chunks} histogram chunks exceed the two-level scan limit of {SCAN_TILE}"
            ),
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
    use cuda_device::{SharedArray, kernel, launch_bounds, thread, warp};

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
        if tid < RADIX_SIZE as u32 {
            unsafe { HISTOGRAM[tid as usize] = 0 };
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

        if tid < RADIX_SIZE as u32 {
            let output_index = tid * num_blocks + block;
            unsafe {
                block_histograms
                    .add(output_index as usize)
                    .write(HISTOGRAM[tid as usize]);
            }
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

    /// Stable block-rank scatter. Each item stripe is ranked in thread order:
    /// warp match groups publish one count per digit, earlier warp counts form
    /// the stripe rank, and `DIGIT_PREFIXES` carries totals across stripes.
    /// Keys are read exactly once and no per-thread local array is required.
    #[kernel]
    #[launch_bounds(P::BLOCK_THREADS, P::MIN_BLOCKS_PER_SM)]
    pub unsafe fn scatter_u32<P: RadixPolicy>(
        input: *const u32,
        output: *mut u32,
        block_offsets: *const u32,
        digit_bases: *const u32,
        count: u32,
        shift: u32,
        num_blocks: u32,
        descending: bool,
    ) {
        static mut WARP_HISTOGRAMS: SharedArray<u32, { MAX_WARPS * RADIX_SIZE }> =
            SharedArray::UNINIT;
        static mut DIGIT_PREFIXES: SharedArray<u32, RADIX_SIZE> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let block = thread::blockIdx_x();
        let warp_id = tid / 32;
        let lane = warp::lane_id();
        let num_warps = P::BLOCK_THREADS / 32;
        let tile = P::BLOCK_THREADS * P::ITEMS_PER_THREAD;
        let base = block * tile;
        let warp_histograms = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WARP_HISTOGRAMS) };
        let digit_prefixes = unsafe { SharedArray::as_raw_mut_ptr(&raw mut DIGIT_PREFIXES) };
        let mut slot = tid;
        let active_warp_slots = num_warps * RADIX_SIZE as u32;
        while slot < active_warp_slots {
            unsafe { warp_histograms.add(slot as usize).write(0) };
            slot += P::BLOCK_THREADS;
        }
        if tid < RADIX_SIZE as u32 {
            unsafe { digit_prefixes.add(tid as usize).write(0) };
        }
        thread::sync_threads();

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
            let stripe_prefix = if valid {
                unsafe { digit_prefixes.add(digit as usize).read() }
            } else {
                0
            };
            let valid_mask = warp::ballot_sync(FULL_MASK, valid);
            let peers = warp::match_any_sync(FULL_MASK, digit) & valid_mask;
            if valid {
                let leader = peers.trailing_zeros();
                if lane == leader {
                    let counter = unsafe {
                        warp_histograms.add((warp_id * RADIX_SIZE as u32 + digit) as usize)
                    };
                    unsafe { counter.write(peers.count_ones()) };
                }
            }
            thread::sync_threads();

            if valid {
                let mut rank = stripe_prefix;
                let mut earlier_warp = 0u32;
                while earlier_warp < warp_id {
                    rank = rank.wrapping_add(unsafe {
                        warp_histograms
                            .add((earlier_warp * RADIX_SIZE as u32 + digit) as usize)
                            .read()
                    });
                    earlier_warp += 1;
                }
                rank = rank.wrapping_add((peers & warp::lanemask_lt()).count_ones());
                let global_rank = unsafe {
                    digit_bases
                        .add(digit as usize)
                        .read()
                        .wrapping_add(
                            block_offsets
                                .add((digit * num_blocks + block) as usize)
                                .read(),
                        )
                        .wrapping_add(rank)
                };
                unsafe { output.add(global_rank as usize).write(key) };
            }

            if tid < RADIX_SIZE as u32 {
                let digit = tid;
                let mut stripe_total = 0u32;
                let mut warp_index = 0u32;
                while warp_index < num_warps {
                    stripe_total = stripe_total.wrapping_add(unsafe {
                        warp_histograms
                            .add((warp_index * RADIX_SIZE as u32 + digit) as usize)
                            .read()
                    });
                    warp_index += 1;
                }
                let prefix = unsafe { digit_prefixes.add(digit as usize).read() };
                unsafe {
                    digit_prefixes
                        .add(digit as usize)
                        .write(prefix.wrapping_add(stripe_total));
                }
            }
            thread::sync_threads();

            slot = tid;
            while slot < active_warp_slots {
                unsafe { warp_histograms.add(slot as usize).write(0) };
                slot += P::BLOCK_THREADS;
            }
            thread::sync_threads();
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
    block_histograms: DeviceBuffer<u32>,
    block_offsets: DeviceBuffer<u32>,
    chunk_sums: DeviceBuffer<u32>,
    digit_bases: DeviceBuffer<u32>,
    validation: DeviceBuffer<u32>,
    capacity: usize,
    max_blocks: u32,
    max_chunks: u32,
    policy: RadixPolicySelection,
}

impl U32SortWorkspace {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn temporary_bytes(&self) -> usize {
        self.scratch.num_bytes()
            + self.block_histograms.num_bytes()
            + self.block_offsets.num_bytes()
            + self.chunk_sums.num_bytes()
            + self.digit_bases.num_bytes()
            + self.validation.num_bytes()
    }
}

pub struct GpuSorter {
    module: kernels::LoadedModule,
    policy: RadixPolicySelection,
    sm: u32,
}

impl GpuSorter {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, SortError> {
        let (major, minor) = context.compute_capability()?;
        let sm = (major * 10 + minor) as u32;
        let policy =
            RadixPolicySelection::select(sm).ok_or(SortError::UnsupportedArchitecture(sm))?;
        let module = kernels::load(context)?;
        Ok(Self { module, policy, sm })
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
        let max_chunks = max_blocks.div_ceil(SCAN_TILE).max(1);
        if max_chunks > SCAN_TILE {
            return Err(SortError::ScanHierarchyTooDeep { chunks: max_chunks });
        }
        let histogram_len = RADIX_SIZE
            .checked_mul(max_blocks as usize)
            .ok_or(SortError::TooManyItems(capacity))?;
        let chunk_len = RADIX_SIZE
            .checked_mul(max_chunks as usize)
            .ok_or(SortError::TooManyItems(capacity))?;

        Ok(U32SortWorkspace {
            scratch: DeviceBuffer::zeroed(stream, capacity)?,
            block_histograms: DeviceBuffer::zeroed(stream, histogram_len)?,
            block_offsets: DeviceBuffer::zeroed(stream, histogram_len)?,
            chunk_sums: DeviceBuffer::zeroed(stream, chunk_len)?,
            digit_bases: DeviceBuffer::zeroed(stream, RADIX_SIZE)?,
            validation: DeviceBuffer::zeroed(stream, 1)?,
            capacity,
            max_blocks,
            max_chunks,
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
    {
        let count = u32::try_from(keys.len()).map_err(|_| SortError::TooManyItems(keys.len()))?;
        let tile = P::BLOCK_THREADS * P::ITEMS_PER_THREAD;
        let num_blocks = count.div_ceil(tile).max(1);
        let num_chunks = num_blocks.div_ceil(SCAN_TILE).max(1);
        debug_assert!(num_blocks <= workspace.max_blocks);
        debug_assert!(num_chunks <= workspace.max_chunks);
        if num_chunks > SCAN_TILE {
            return Err(SortError::ScanHierarchyTooDeep { chunks: num_chunks });
        }

        let histogram_cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (P::BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let scan_cfg = LaunchConfig {
            grid_dim: (num_chunks, RADIX_SIZE as u32, 1),
            block_dim: (SCAN_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let scan_sums_cfg = LaunchConfig {
            grid_dim: (1, RADIX_SIZE as u32, 1),
            block_dim: (SCAN_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let bases_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (RADIX_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let descending = order == SortOrder::Descending;
        let mut input = keys.cu_deviceptr() as *const u32;
        let mut output = workspace.scratch.cu_deviceptr() as *mut u32;
        let keys_ptr = keys.cu_deviceptr() as *mut u32;
        let scratch_ptr = workspace.scratch.cu_deviceptr() as *mut u32;

        let mut shift = 0;
        while shift < 32 {
            // SAFETY: workspace sizes cover all digit/block products; scatter
            // destinations are a bijection established by scanned histograms
            // and stable per-CTA ranks. All allocations outlive synchronization.
            unsafe {
                self.module.histogram_u32::<P>(
                    stream,
                    histogram_cfg,
                    input,
                    workspace.block_histograms.cu_deviceptr() as *mut u32,
                    count,
                    shift,
                    num_blocks,
                    descending,
                )?;
                self.module.scan_block_histograms(
                    stream,
                    scan_cfg,
                    workspace.block_histograms.cu_deviceptr() as *const u32,
                    workspace.block_offsets.cu_deviceptr() as *mut u32,
                    workspace.chunk_sums.cu_deviceptr() as *mut u32,
                    num_blocks,
                    num_chunks,
                )?;
                self.module.scan_chunk_sums(
                    stream,
                    scan_sums_cfg,
                    workspace.chunk_sums.cu_deviceptr() as *mut u32,
                    num_chunks,
                )?;
                self.module.add_chunk_prefixes(
                    stream,
                    scan_cfg,
                    workspace.block_offsets.cu_deviceptr() as *mut u32,
                    workspace.chunk_sums.cu_deviceptr() as *const u32,
                    num_blocks,
                    num_chunks,
                )?;
                self.module.digit_bases(
                    stream,
                    bases_cfg,
                    workspace.block_histograms.cu_deviceptr() as *const u32,
                    workspace.block_offsets.cu_deviceptr() as *const u32,
                    workspace.digit_bases.cu_deviceptr() as *mut u32,
                    num_blocks,
                )?;
                self.module.scatter_u32::<P>(
                    stream,
                    histogram_cfg,
                    input,
                    output,
                    workspace.block_offsets.cu_deviceptr() as *const u32,
                    workspace.digit_bases.cu_deviceptr() as *const u32,
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
