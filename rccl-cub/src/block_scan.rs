//! Private fixed-policy block scan used by workload-facing device scans.

use rccl_memory::device::BlockSharedMemory;

const WARP_THREADS: u32 = 32;

/// Scalar supported by the workload's block-wide prefix sums.
///
/// SRDatalog uses `u32` tuple/write counts and `u64` fanout counts. Signed and
/// floating variants stay absent until a real caller requires their overflow
/// or ordering semantics.
#[doc(hidden)]
pub trait BlockScanValue: Copy {
    fn zero() -> Self;
    fn add(left: Self, right: Self) -> Self;
    fn shuffle_up(value: Self, offset: u32) -> Self;
}

impl BlockScanValue for u32 {
    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn add(left: Self, right: Self) -> Self {
        left.wrapping_add(right)
    }

    #[inline(always)]
    fn shuffle_up(value: Self, offset: u32) -> Self {
        cuda_device::warp::shuffle_up_sync(u32::MAX, value, offset)
    }
}

impl BlockScanValue for u64 {
    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn add(left: Self, right: Self) -> Self {
        left.wrapping_add(right)
    }

    #[inline(always)]
    fn shuffle_up(value: Self, offset: u32) -> Self {
        cuda_device::warp::shuffle_up_u64_sync(u32::MAX, value, offset)
    }
}

#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct BlockScanResult<T> {
    pub inclusive: T,
    pub exclusive: T,
    pub aggregate: T,
}

#[inline(always)]
fn warp_inclusive_sum<T: BlockScanValue>(value: T, lane: u32) -> T {
    let mut inclusive = value;
    macro_rules! step {
        ($offset:expr) => {{
            let preceding = T::shuffle_up(inclusive, $offset);
            if lane >= $offset {
                inclusive = T::add(preceding, inclusive);
            }
        }};
    }
    step!(1u32);
    step!(2u32);
    step!(4u32);
    step!(8u32);
    step!(16u32);
    inclusive
}

/// Scan one contiguous logical prefix of a physical 256-thread block.
///
/// This is the low-latency CUB `BLOCK_SCAN_WARP_SCANS` shape needed by the
/// device-wide workload scan: every physical thread contributes one item,
/// invalid tail threads contribute the additive identity, and every thread
/// receives inclusive, exclusive, and aggregate values. Arithmetic wraps for
/// the unsigned workload counters, matching CUDA integer addition.
///
/// # Divergence contract
///
/// Every physical thread in the block must call this function in converged
/// control flow, including threads beyond `valid_items`, because it executes
/// two block barriers and full-warp shuffles. `valid_items` must be uniform and
/// at most 256. Callers may ignore the per-thread result outside that logical
/// prefix. There is no allocation or grid-wide synchronization.
#[doc(hidden)]
#[inline(always)]
pub fn block_scan_sum_256<T: BlockScanValue>(
    value: T,
    valid_items: u32,
    shared: &BlockSharedMemory,
) -> BlockScanResult<T> {
    block_scan_sum::<T, 256>(value, valid_items, shared)
}

/// Scan one contiguous logical prefix of a physical 128-thread block.
///
/// This is the fixed SM86+ `BLOCK_SCAN_WARP_SCANS` shape used by the
/// decoupled-lookback device scan.
#[doc(hidden)]
#[inline(always)]
pub fn block_scan_sum_128<T: BlockScanValue>(
    value: T,
    valid_items: u32,
    shared: &BlockSharedMemory,
) -> BlockScanResult<T> {
    block_scan_sum::<T, 128>(value, valid_items, shared)
}

#[inline(always)]
fn block_scan_sum<T: BlockScanValue, const THREADS: u32>(
    value: T,
    valid_items: u32,
    shared: &BlockSharedMemory,
) -> BlockScanResult<T> {
    if valid_items > THREADS || !THREADS.is_multiple_of(WARP_THREADS) {
        cuda_device::debug::trap();
    }
    let warps = (THREADS / WARP_THREADS) as usize;
    let thread_index = cuda_device::thread::threadIdx_x();
    let lane = cuda_device::warp::lane_id();
    let warp_index = thread_index / WARP_THREADS;
    let logical = if thread_index < valid_items {
        value
    } else {
        T::zero()
    };
    let warp_inclusive = warp_inclusive_sum(logical, lane);
    let mut warp_aggregates = shared.region::<T>(0, warps);
    if lane == WARP_THREADS - 1 {
        warp_aggregates.write(warp_index as usize, warp_inclusive);
    }
    cuda_device::thread::sync_threads();

    if warp_index == 0 {
        let aggregate = if (lane as usize) < warps {
            warp_aggregates.read(lane as usize)
        } else {
            T::zero()
        };
        let scanned = warp_inclusive_sum(aggregate, lane);
        if (lane as usize) < warps {
            warp_aggregates.write(lane as usize, scanned);
        }
    }
    cuda_device::thread::sync_threads();

    let warp_prefix = if warp_index == 0 {
        T::zero()
    } else {
        warp_aggregates.read(warp_index as usize - 1)
    };
    let preceding = T::shuffle_up(warp_inclusive, 1);
    let exclusive = if lane == 0 {
        warp_prefix
    } else {
        T::add(warp_prefix, preceding)
    };
    BlockScanResult {
        inclusive: T::add(warp_prefix, warp_inclusive),
        exclusive,
        aggregate: warp_aggregates.read(warps - 1),
    }
}
