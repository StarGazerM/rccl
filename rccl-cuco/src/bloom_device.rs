//! Device implementations for the HISA-compatible blocked Bloom filter.

use cuda_device::{
    DisjointSlice,
    atomic::{AtomicOrdering, DeviceAtomicU32},
    thread, warp,
};

use crate::{
    BLOOM_WORDS_PER_BLOCK, BloomFilterReadRef, BloomFilterRef, StableHashBytes, XxHash64,
    bloom::{bloom_block_index, bloom_word_pattern},
};

#[inline(always)]
fn publish_source<const SOURCE: u32>(
    valid_lanes: u32,
    tile_base: u32,
    word: usize,
    block: u32,
    lower_hash: u32,
    words: &[DeviceAtomicU32],
) {
    // All lanes execute every full-warp shuffle. Publication is predicated
    // afterwards so a partial final tile cannot create a divergent collective.
    let source_lane = tile_base + SOURCE;
    let source_block = warp::shuffle(block, source_lane) as usize;
    let source_lower = warp::shuffle(lower_hash, source_lane);
    if valid_lanes & (1u32 << source_lane) == 0 {
        return;
    }
    let pattern = bloom_word_pattern(source_lower as u64, word);
    // SAFETY: multiply-shift bounds the block and the tile rank is in 0..8.
    unsafe { words.get_unchecked(source_block * BLOOM_WORDS_PER_BLOCK + word) }
        .fetch_or(pattern, AtomicOrdering::Relaxed);
}
#[inline(always)]
pub fn clear(index: thread::ThreadIndex<'_>, mut words: DisjointSlice<u32>) {
    if let Some(word) = words.get_mut(index) {
        *word = 0;
    }
}

#[inline(always)]
pub fn add<K: StableHashBytes>(
    global_thread: usize,
    words: &[DeviceAtomicU32],
    keys: &[K],
    num_blocks: usize,
    seed: u64,
) {
    // Pinned cuCollections uses an eight-lane horizontal add layout. Every
    // lane hashes one input, then each tile transposes eight fingerprints via
    // full-warp shuffles so lane `r` publishes word `r` for all eight keys.
    // Invalid tail lanes still participate in the collectives, avoiding both
    // divergent masks and generated shuffle/ballot helper paths.
    let lane = (global_thread as u32) & 31;
    let tile_base = lane & !7;
    let word = (lane & 7) as usize;
    let valid = global_thread < keys.len();
    let mut hash = 0u64;
    let mut block = 0u32;
    if valid {
        // SAFETY: `valid` proves this input index is in bounds.
        hash = XxHash64::new(seed).hash_value(unsafe { keys.get_unchecked(global_thread) });
        block = bloom_block_index(hash, num_blocks) as u32;
    }
    let valid_lanes = warp::ballot(valid);
    publish_source::<0>(valid_lanes, tile_base, word, block, hash as u32, words);
    publish_source::<1>(valid_lanes, tile_base, word, block, hash as u32, words);
    publish_source::<2>(valid_lanes, tile_base, word, block, hash as u32, words);
    publish_source::<3>(valid_lanes, tile_base, word, block, hash as u32, words);
    publish_source::<4>(valid_lanes, tile_base, word, block, hash as u32, words);
    publish_source::<5>(valid_lanes, tile_base, word, block, hash as u32, words);
    publish_source::<6>(valid_lanes, tile_base, word, block, hash as u32, words);
    publish_source::<7>(valid_lanes, tile_base, word, block, hash as u32, words);
}

#[inline(always)]
pub fn add_fault<K: StableHashBytes>(
    global_thread: usize,
    words: &[DeviceAtomicU32],
    keys: &[K],
    num_blocks: usize,
    seed: u64,
    omitted_word: u32,
) {
    if global_thread >= keys.len() {
        return;
    }
    // SAFETY: same invariant as `add`; this test-only path intentionally skips
    // one otherwise-required publication to exercise the no-false-negative gate.
    let filter = unsafe { BloomFilterRef::<K>::new(words, num_blocks, seed) };
    // SAFETY: the preceding range check proves the index.
    filter.add_omitting(unsafe { keys.get_unchecked(global_thread) }, omitted_word);
}

#[inline(always)]
pub fn contains<K: StableHashBytes>(
    index: thread::ThreadIndex<'_>,
    words: &[u32],
    keys: &[K],
    mut output: DisjointSlice<u8>,
    num_blocks: usize,
    seed: u64,
) {
    let global_thread = index.get();
    if global_thread >= keys.len() {
        return;
    }
    let filter = BloomFilterReadRef::<K>::new(words, num_blocks, seed);
    if let Some(result) = output.get_mut(index) {
        // SAFETY: the range check proves the key index. The owning filter
        // constructor validates the storage base alignment once, and every
        // eight-word filter block preserves it.
        *result = u8::from(unsafe { filter.contains_aligned(keys.get_unchecked(global_thread)) });
    }
}
