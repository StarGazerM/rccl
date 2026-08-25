//! Compile-target CUB policy constants.

include!(concat!(env!("OUT_DIR"), "/target_schedule.rs"));

pub const RADIX_ITEMS_PER_THREAD_USIZE: usize = RADIX_ITEMS_PER_THREAD as usize;
pub const RADIX_BLOCK_WARPS: usize = (RADIX_BLOCK_THREADS / 32) as usize;
pub const RADIX_UNROLL: u32 = 16;
pub const COMPARISON_BLOCK_THREADS: u32 = 256;

pub const MERGE_BLOCK_THREADS: u32 = 512;
pub const MERGE_ITEMS_PER_THREAD: u32 = 15;
pub const MERGE_UNROLL: u32 = 15;
pub const MERGE_FALLBACK_BLOCK_THREADS: u32 = 64;
