//! CUDA ownership boundary for block-wide ordering collectives.
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

use core::{mem::MaybeUninit, ops::Range};

use cuda_device::{DisjointSlice, LinearTiles, thread, warp};
use rccl_memory::device::{
    AlignedBlockSharedMemory, BlockSharedMemory, BlockStripedBlockWrite, DeviceRegion,
    FullWarpStripedInput, MergePathSearch, OneDimensionalLaunch, PartialWarpStripedInput,
    SharedRegion, WarpLane, WarpStripedBlockInput, WarpStripedBlockRead, WarpStripedBlockTile,
    WarpStripedBlockWrite, commit_and_wait_async_copy, commit_async_copy, initialized_ref,
    initialized_value, threadfence_block, wait_async_copy, warp_striped_pair_into_write,
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

// Policy800's single-tile agent has 256 threads: exactly eight warps. Keep
// this distinct from the 384-thread one-sweep repetition so the Rust source
// has the same compile-time recursion extent as `BlockScanWarpScans`.
macro_rules! for_each_single_tile_warp {
    ($warp:ident, $body:block) => {{
        macro_rules! one {
            ($index:expr) => {{
                let $warp = $index;
                $body
            }};
        }
        one!(1usize);
        one!(2usize);
        one!(3usize);
        one!(4usize);
        one!(5usize);
        one!(6usize);
        one!(7usize);
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

// Policy800's histogram agent has 128 threads, so each thread owns exactly
// two of the 256 digit counters. CUB fully unrolls this fixed bin loop.
macro_rules! for_each_histogram_thread_bin {
    ($threads:expr, $thread:expr, $bin:ident, $body:block) => {{
        macro_rules! one {
            ($offset:expr) => {{
                let $bin = $thread + $offset * $threads;
                $body
            }};
        }
        one!(0usize);
        one!(1usize);
    }};
}

// The primitive radix surface tops out at eight 8-bit passes for 64-bit
// keys. CUB marks every histogram pass loop as fully unrolled even though the
// selected bit range is runtime state. Materialize that same bounded shape in
// Rust; `MAX_PASSES` remains a proc-macro-selected type-level policy.
macro_rules! for_each_histogram_pass {
    ($maximum:expr, $active:expr, $pass:ident, $body:block) => {{
        macro_rules! one {
            ($index:expr) => {
                if $index < $maximum && $index < $active {
                    let $pass = $index;
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

// CUB's WarpScan specialization fully unrolls the five shuffle steps for a
// 32-lane logical warp. Keep that fixed instruction dependency graph visible
// to Rust instead of relying on a later optimizer to discover a rolled loop.
macro_rules! for_each_warp_scan_offset {
    ($offset:ident, $body:block) => {{
        macro_rules! one {
            ($value:expr) => {{
                let $offset = $value;
                $body
            }};
        }
        one!(1u32);
        one!(2u32);
        one!(4u32);
        one!(8u32);
        one!(16u32);
    }};
}

// Set-operation Policy600 has sixteen warps. `BlockScanWarpScans` template
// recurses over warp aggregates 1..16; preserve that fixed dependency graph.
macro_rules! for_each_set_warp {
    ($warp:ident, $body:block) => {{
        macro_rules! one {
            ($index:expr) => {{
                let $warp = $index;
                $body
            }};
        }
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
    }};
}

// `BlockRadixRank` fully unrolls its 33-word memoized raking segment. Keep the
// exact compile-time repetition visible to rustc/cuda-oxide instead of relying
// on an optimizer loop heuristic.
macro_rules! for_each_radix_raking_word {
    ($word:ident, $body:block) => {{
        macro_rules! one {
            ($index:expr) => {{
                let $word = $index;
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
        one!(24usize);
        one!(25usize);
        one!(26usize);
        one!(27usize);
        one!(28usize);
        one!(29usize);
        one!(30usize);
        one!(31usize);
        one!(32usize);
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

mod merge;
mod radix;
mod set;

pub use merge::*;
pub use radix::*;
pub use set::*;

#[cfg(test)]
use merge::local_merge_path_begin;
#[cfg(test)]
mod tests;
