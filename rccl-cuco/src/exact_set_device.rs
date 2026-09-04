//! Device implementations for the workload-specific exact HISA row set.

use cuda_device::{
    DisjointSlice,
    atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64},
    cooperative_groups::{ThreadGroup, WarpCollective, this_thread_block},
    thread,
};

use crate::{ExactInsertOutcome, ExactRowSetReadRef, ExactRowSetRef, Row, XxHash64};

#[inline(always)]
pub fn clear(index: thread::ThreadIndex<'_>, mut states: DisjointSlice<u32>) {
    if let Some(state) = states.get_mut(index) {
        *state = 0;
    }
}

#[inline(always)]
pub fn clear_direct(
    index: thread::ThreadIndex<'_>,
    mut slots: DisjointSlice<u64>,
    empty_slot: u64,
) {
    if let Some(slot) = slots.get_mut(index) {
        *slot = empty_slot;
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn insert<const N: usize, const CHECK_SENTINEL: bool, const COUNT: bool>(
    global_thread: usize,
    states: &[DeviceAtomicU32],
    hashes: &[DeviceAtomicU64],
    words: &[DeviceAtomicU32],
    metadata: &[DeviceAtomicU64],
    rows: &[Row<N>],
    empty_row: Row<N>,
    seed: u64,
    force_hash: u32,
    forced_hash: u64,
) {
    let input_index = global_thread / 4;
    if input_index >= rows.len() {
        return;
    }
    let row = rows[input_index];
    let tile_rank = global_thread % 4;
    let outcome = if CHECK_SENTINEL && row.same(empty_row) {
        None
    } else {
        // SAFETY: the host validates non-empty prime-group storage, and the
        // launch assigns one convergent four-lane tile to every input row.
        let set = unsafe { ExactRowSetRef::new(states, hashes, words, seed) };
        Some(if force_hash != 0 {
            set.insert_hash(row, forced_hash)
        } else {
            set.insert(row)
        })
    };
    if COUNT && tile_rank == 0 {
        let counter = match outcome {
            Some(ExactInsertOutcome::Inserted) => 0,
            Some(ExactInsertOutcome::Duplicate) => 1,
            Some(ExactInsertOutcome::CapacityExhausted) => 2,
            None => 3,
        };
        metadata[counter].fetch_add(1, AtomicOrdering::Relaxed);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn contains<const N: usize>(
    global_thread: usize,
    states: &[DeviceAtomicU32],
    hashes: &[DeviceAtomicU64],
    words: &[DeviceAtomicU32],
    rows: &[Row<N>],
    output: &[DeviceAtomicU32],
    seed: u64,
    force_hash: u32,
    forced_hash: u64,
) {
    let input_index = global_thread / 4;
    if input_index >= rows.len() {
        return;
    }
    // SAFETY: the owning set supplies a matching immutable storage triple and
    // retains it through completion.
    let set = unsafe { ExactRowSetReadRef::new(states, hashes, words, seed) };
    let present = if force_hash != 0 {
        set.contains_hash(rows[input_index], forced_hash)
    } else {
        set.contains(rows[input_index])
    };
    if global_thread % 4 == 0 {
        output[input_index].store(present as u32, AtomicOrdering::Relaxed);
    }
}

#[inline(always)]
fn row2_bits<const N: usize>(row: Row<N>) -> u64 {
    // SAFETY: this helper is instantiated on the direct path only for Row<2>,
    // whose repr(C) object is exactly one initialized, four-byte-aligned u64.
    unsafe { core::ptr::read_unaligned(core::ptr::from_ref(&row).cast::<u64>()) }
}

#[inline(always)]
fn direct_insert_one<const N: usize>(
    slots: &[DeviceAtomicU64],
    row: Row<N>,
    hash: u64,
    seed: u64,
    empty_slot: u64,
) -> ExactInsertOutcome {
    let groups = slots.len() / 4;
    let start = hash as usize % groups;
    let second = XxHash64::new(seed.wrapping_add(1)).hash_value(&row);
    let step = second as usize % (groups - 1) + 1;
    let desired = row2_bits(row);
    let tile = this_thread_block().tiled_partition::<4>();
    let rank = tile.thread_rank() as usize;
    loop {
        let mut group = start;
        let mut visited = 0usize;
        while visited < groups {
            let index = group * 4 + rank;
            let observed = slots[index].cached_snapshot();
            if tile.ballot(observed == desired) != 0 {
                return ExactInsertOutcome::Duplicate;
            }
            let empty = tile.ballot(observed == empty_slot);
            if empty != 0 {
                let owner = empty.trailing_zeros();
                let status = if rank == owner as usize {
                    match slots[index].compare_exchange_relaxed(empty_slot, desired) {
                        Ok(_) => 1u32,
                        Err(actual) if actual == desired => 2u32,
                        Err(_) => 0u32,
                    }
                } else {
                    0u32
                };
                match tile.shfl(status, owner) {
                    1 => return ExactInsertOutcome::Inserted,
                    2 => return ExactInsertOutcome::Duplicate,
                    _ => break,
                }
            }
            visited += 1;
            group += step;
            if group >= groups {
                group -= groups;
            }
        }
        if visited == groups {
            return ExactInsertOutcome::CapacityExhausted;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn insert_direct<const N: usize, const CHECK_SENTINEL: bool, const COUNT: bool>(
    global_thread: usize,
    slots: &[DeviceAtomicU64],
    metadata: &[DeviceAtomicU64],
    rows: &[Row<N>],
    empty_slot: u64,
    seed: u64,
    force_hash: u32,
    forced_hash: u64,
) {
    let input_index = global_thread / 4;
    if input_index >= rows.len() {
        return;
    }
    let row = rows[input_index];
    let desired = row2_bits(row);
    let outcome = if CHECK_SENTINEL && desired == empty_slot {
        None
    } else {
        let hash = if force_hash != 0 {
            forced_hash
        } else {
            XxHash64::new(seed).hash_value(&row)
        };
        Some(direct_insert_one(slots, row, hash, seed, empty_slot))
    };
    if COUNT && global_thread % 4 == 0 {
        let counter = match outcome {
            Some(ExactInsertOutcome::Inserted) => 0,
            Some(ExactInsertOutcome::Duplicate) => 1,
            Some(ExactInsertOutcome::CapacityExhausted) => 2,
            None => 3,
        };
        metadata[counter].fetch_add(1, AtomicOrdering::Relaxed);
    }
}

#[inline(always)]
fn direct_contains_one<const N: usize>(
    slots: &[DeviceAtomicU64],
    row: Row<N>,
    hash: u64,
    seed: u64,
    empty_slot: u64,
) -> bool {
    let groups = slots.len() / 4;
    let start = hash as usize % groups;
    let second = XxHash64::new(seed.wrapping_add(1)).hash_value(&row);
    let step = second as usize % (groups - 1) + 1;
    let desired = row2_bits(row);
    let tile = this_thread_block().tiled_partition::<4>();
    let rank = tile.thread_rank() as usize;
    let mut group = start;
    let mut visited = 0usize;
    while visited < groups {
        let observed = slots[group * 4 + rank].cached_snapshot();
        if tile.ballot(observed == desired) != 0 {
            return true;
        }
        if tile.ballot(observed == empty_slot) != 0 {
            return false;
        }
        visited += 1;
        group += step;
        if group >= groups {
            group -= groups;
        }
    }
    false
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn contains_direct<const N: usize>(
    global_thread: usize,
    slots: &[DeviceAtomicU64],
    rows: &[Row<N>],
    output: &[DeviceAtomicU32],
    empty_slot: u64,
    seed: u64,
    force_hash: u32,
    forced_hash: u64,
) {
    let input_index = global_thread / 4;
    if input_index >= rows.len() {
        return;
    }
    let row = rows[input_index];
    let hash = if force_hash != 0 {
        forced_hash
    } else {
        XxHash64::new(seed).hash_value(&row)
    };
    let present = direct_contains_one(slots, row, hash, seed, empty_slot);
    if global_thread % 4 == 0 {
        output[input_index].store(present as u32, AtomicOrdering::Relaxed);
    }
}
