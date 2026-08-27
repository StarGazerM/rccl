//! Affine device-memory capabilities used by RCCL collectives.
//!
//! CUDA kernels sometimes use a layout whose ownership unit is wider than one
//! hardware thread.  This module contains the small trusted boundary that
//! maps those layouts to cuda-oxide's exclusive [`DisjointSlice`] packet.  CUB
//! algorithms consume the safe capabilities; raw pointer extraction stays in
//! this memory layer and is never emitted by an operation macro.

use core::mem::MaybeUninit;
use cuda_device::{
    DisjointSlice, LinearTiles,
    thread::{self, __internal, LaunchContextRef},
};

/// Bounds-preserving state for one MergePath binary search over two borrowed
/// ranges.
pub struct MergePathSearch<'a, T> {
    first: &'a [T],
    second: &'a [T],
    diagonal: usize,
    first_begin: usize,
    first_end: usize,
}

impl<'a, T> MergePathSearch<'a, T> {
    #[inline(always)]
    pub fn new(first: &'a [T], second: &'a [T], diagonal: usize) -> Self {
        let diagonal = diagonal.min(first.len().saturating_add(second.len()));
        Self {
            first,
            second,
            diagonal,
            first_begin: diagonal.saturating_sub(second.len()),
            first_end: diagonal.min(first.len()),
        }
    }

    /// Probe and narrow one interval step. `second_before_first` returns the
    /// same stable ordering decision used by CUB's `MergePath`.
    #[inline(always)]
    pub fn step(&mut self, second_before_first: impl FnOnce(&T, &T) -> bool) -> bool {
        if self.first_begin >= self.first_end {
            return false;
        }
        let middle = self.first_begin + (self.first_end - self.first_begin) / 2;
        let second_index = self.diagonal - 1 - middle;
        // SAFETY: the maintained MergePath interval proves `middle` names a
        // first element and its complementary coordinate names a second.
        let first = unsafe { self.first.get_unchecked(middle) };
        let second = unsafe { self.second.get_unchecked(second_index) };
        if second_before_first(first, second) {
            self.first_end = middle;
        } else {
            self.first_begin = middle + 1;
        }
        true
    }

    #[inline(always)]
    pub fn partition(&self) -> usize {
        self.first_begin
    }
}

/// Unforgeable proof that the kernel was prepared with a one-dimensional,
/// 32-bit coordinate launch contract.
pub type OneDimensionalLaunch<'kernel> =
    LaunchContextRef<'kernel, __internal::Domain1, __internal::U32Coordinates>;

#[inline(always)]
const fn warp_striped_local<const THREADS: usize, const ITEMS: usize>(
    thread_index: u32,
    item: usize,
) -> Option<usize> {
    if THREADS == 0
        || ITEMS == 0
        || THREADS % 32 != 0
        || thread_index as usize >= THREADS
        || item >= ITEMS
    {
        return None;
    }
    let lane = thread_index & 31;
    let warp = thread_index >> 5;
    Some(warp as usize * 32 * ITEMS + item * 32 + lane as usize)
}

#[inline(always)]
const fn validated_warp_striped_local<const ITEMS: usize>(thread_index: u32, item: usize) -> usize {
    let lane = thread_index & 31;
    let warp = thread_index >> 5;
    warp as usize * 32 * ITEMS + item * 32 + lane as usize
}

#[inline(always)]
fn block_tile<const THREADS: usize, const ITEMS: usize>(
    _launch: OneDimensionalLaunch<'_>,
    len: usize,
    valid_items: usize,
) -> (usize, u32) {
    if THREADS == 0
        || ITEMS == 0
        || THREADS % 32 != 0
        || THREADS > usize::MAX / ITEMS
        || thread::blockDim_x() as usize != THREADS
    {
        cuda_device::debug::trap();
    }
    let tile_items = THREADS * ITEMS;
    if valid_items == 0 || valid_items > tile_items {
        cuda_device::debug::trap();
    }
    let tile_base = thread::blockIdx_x() as usize * tile_items;
    if tile_base > len || valid_items > len - tile_base {
        cuda_device::debug::trap();
    }
    let thread_index = thread::threadIdx_x();
    if thread_index as usize >= THREADS {
        cuda_device::debug::trap();
    }
    (tile_base, thread_index)
}

/// Blocked-register view of two logical input ranges concatenated by CUB's
/// `gmem_to_reg` operation.
///
/// `FULL` is selected by the generated kernel from the output tile shape. A
/// full tile proves that every blocked `(item, thread)` coordinate names an
/// element. A partial tile returns `None` for coordinates beyond the logical
/// concatenation. Empty ranges remain valid even when their nominal CUB
/// iterator base would be outside the backing allocation.
pub struct BlockedConcatenatedInput<
    'a,
    T,
    const THREADS: usize,
    const ITEMS: usize,
    const FULL: bool,
> {
    first: Option<&'a T>,
    second: Option<&'a T>,
    first_count: i32,
    count: i32,
    thread_index: i32,
}

/// Reusable proof of one block's warp-striped tile geometry.
///
/// Pair collectives use one proof for their key and value phases so Rust does
/// not keep two derived writers live across aliased shared-memory phases.
#[derive(Clone, Copy)]
pub struct WarpStripedBlockTile<const THREADS: usize, const ITEMS: usize> {
    tile_base: usize,
    thread_index: u32,
    valid_items: usize,
    output_len: usize,
}

impl<const THREADS: usize, const ITEMS: usize> WarpStripedBlockTile<THREADS, ITEMS> {
    #[inline(always)]
    pub fn new(launch: OneDimensionalLaunch<'_>, output_len: usize, valid_items: usize) -> Self {
        let (tile_base, thread_index) =
            block_tile::<THREADS, ITEMS>(launch, output_len, valid_items);
        Self {
            tile_base,
            thread_index,
            valid_items,
            output_len,
        }
    }

    #[inline(always)]
    pub fn blocked_concatenated_input<'input, U: Copy, const FULL: bool>(
        &self,
        first: &'input [U],
        first_begin: usize,
        first_count: i32,
        second: &'input [U],
        second_begin: usize,
        second_count: i32,
    ) -> BlockedConcatenatedInput<'input, U, THREADS, ITEMS, FULL> {
        BlockedConcatenatedInput::new(
            first,
            first_begin,
            first_count,
            second,
            second_begin,
            second_count,
            self.thread_index,
            self.valid_items,
        )
    }

    #[inline(always)]
    pub fn output<'output, T>(
        &self,
        output: DisjointSlice<'output, T, LinearTiles<ITEMS>>,
    ) -> WarpStripedBlockWrite<'output, T, THREADS, ITEMS> {
        if output.len() != self.output_len {
            cuda_device::debug::trap();
        }
        WarpStripedBlockWrite::from_tile(output, *self)
    }
}

impl<'a, T: Copy, const THREADS: usize, const ITEMS: usize, const FULL: bool>
    BlockedConcatenatedInput<'a, T, THREADS, ITEMS, FULL>
{
    #[inline(always)]
    fn new(
        first: &'a [T],
        first_begin: usize,
        first_count: i32,
        second: &'a [T],
        second_begin: usize,
        second_count: i32,
        thread_index: u32,
        valid_items: usize,
    ) -> Self {
        if first_count < 0 || second_count < 0 {
            cuda_device::debug::trap();
        }
        let first_count_usize = first_count as usize;
        let second_count_usize = second_count as usize;
        let capacity = THREADS * ITEMS;
        if capacity > i32::MAX as usize
            || first_count_usize > capacity
            || second_count_usize > capacity - first_count_usize
        {
            cuda_device::debug::trap();
        }
        let count = first_count + second_count;
        if count as usize != valid_items || (FULL && valid_items != capacity) {
            cuda_device::debug::trap();
        }
        let first = if first_count == 0 {
            None
        } else {
            if first_begin >= first.len() || first_count_usize > first.len() - first_begin {
                cuda_device::debug::trap();
            }
            // SAFETY: the preceding range proof established a nonempty head.
            Some(unsafe { first.get_unchecked(first_begin) })
        };
        let second = if second_count == 0 {
            None
        } else {
            if second_begin >= second.len() || second_count_usize > second.len() - second_begin {
                cuda_device::debug::trap();
            }
            // SAFETY: the preceding range proof established a nonempty head.
            Some(unsafe { second.get_unchecked(second_begin) })
        };
        Self {
            first,
            second,
            first_count,
            count,
            thread_index: thread_index as i32,
        }
    }

    /// Read this thread's blocked item, or report an inactive partial-tile
    /// coordinate. The only unchecked operations are downstream of the range
    /// and concatenation proofs established by `new`.
    #[inline(always)]
    pub fn read(&self, item: usize) -> Option<T> {
        if item >= ITEMS {
            return None;
        }
        let index = item as i32 * THREADS as i32 + self.thread_index;
        if !FULL && index >= self.count {
            return None;
        }
        if index < self.first_count {
            // SAFETY: a positive first count constructed `Some(head)`, and
            // the branch keeps the offset inside the validated range.
            let head = unsafe { self.first.unwrap_unchecked() };
            Some(unsafe { core::ptr::from_ref(head).add(index as usize).read() })
        } else {
            let second_index = index - self.first_count;
            // SAFETY: `FULL` was proved to fill the block capacity, while the
            // partial path checked `index < count`; either proof places the
            // translated index inside `second`.
            let head = unsafe { self.second.unwrap_unchecked() };
            Some(unsafe { core::ptr::from_ref(head).add(second_index as usize).read() })
        }
    }

    /// Copy this thread's blocked item directly into a register slot.
    ///
    /// Unlike an `Option<T>` return, the destination keeps CUB's inactive
    /// partial-tile element uninitialized and lets the `FULL` specialization
    /// erase the activity branch completely.
    #[inline(always)]
    pub fn load_into(&self, item: usize, output: &mut MaybeUninit<T>) {
        if item >= ITEMS {
            return;
        }
        let index = item as i32 * THREADS as i32 + self.thread_index;
        if !FULL && index >= self.count {
            return;
        }
        let value = if index < self.first_count {
            // SAFETY: a positive first count constructed `Some(head)`, and
            // the branch keeps the offset inside the validated range.
            let head = unsafe { self.first.unwrap_unchecked() };
            unsafe { core::ptr::from_ref(head).add(index as usize).read() }
        } else {
            let second_index = index - self.first_count;
            // SAFETY: the full or partial activity proof places the translated
            // coordinate inside the validated second range.
            let head = unsafe { self.second.unwrap_unchecked() };
            unsafe { core::ptr::from_ref(head).add(second_index as usize).read() }
        };
        output.write(value);
    }
}

/// Read-only input for one block-wide warp-striped tile.
///
/// Unlike an output packet, a shared Rust slice permits overlapping reads, so
/// no unsafe ownership translation is needed. The constructor narrows the
/// borrow to the exact valid prefix selected by the macro-generated launch.
pub struct WarpStripedBlockInput<'a, T, const THREADS: usize, const ITEMS: usize> {
    tile: &'a [T],
    thread_index: u32,
}

impl<'a, T: Copy, const THREADS: usize, const ITEMS: usize>
    WarpStripedBlockInput<'a, T, THREADS, ITEMS>
{
    #[inline(always)]
    pub fn new(input: &'a [T], launch: OneDimensionalLaunch<'_>, valid_items: usize) -> Self {
        let (tile_base, thread_index) =
            block_tile::<THREADS, ITEMS>(launch, input.len(), valid_items);
        Self {
            // SAFETY: `block_tile` proved this complete prefix in bounds.
            tile: unsafe { input.get_unchecked(tile_base..tile_base + valid_items) },
            thread_index,
        }
    }

    /// First initialized item, used as CUB's guarded-load default.
    #[inline(always)]
    pub fn first(&self) -> T {
        self.tile[0]
    }

    /// Copy this thread's item when it lies in the valid prefix.
    #[inline(always)]
    pub fn read(&self, item: usize) -> Option<T> {
        let local = warp_striped_local::<THREADS, ITEMS>(self.thread_index, item)?;
        self.tile.get(local).copied()
    }

    /// Copy this thread's valid item, otherwise preserve the caller's explicit
    /// CUB guarded-load default.
    #[inline(always)]
    pub fn read_or(&self, item: usize, default: T) -> T {
        if item >= ITEMS {
            return default;
        }
        let local = validated_warp_striped_local::<ITEMS>(self.thread_index, item);
        if local >= self.tile.len() {
            return default;
        }
        // SAFETY: the explicit prefix check proved this index in the tile.
        unsafe { *self.tile.get_unchecked(local) }
    }
}

/// Exclusive output for one block-wide warp-striped tile.
///
/// For thread `t`, item `i` maps to
/// `warp * (32 * ITEMS) + i * 32 + lane`.  Over every thread and item in a
/// block this is a bijection over `0..THREADS * ITEMS`, so safe writes cannot
/// alias. Threads outside the selected complete-warp layout trap during
/// construction; the macro-generated launch separately fixes the exact block
/// width required by the CUB algorithm and its barriers.
pub struct WarpStripedBlockWrite<'a, T, const THREADS: usize, const ITEMS: usize> {
    output: DisjointSlice<'a, T, LinearTiles<ITEMS>>,
    tile_base: usize,
    thread_index: u32,
    valid_items: usize,
}

impl<'a, T, const THREADS: usize, const ITEMS: usize> WarpStripedBlockWrite<'a, T, THREADS, ITEMS> {
    #[inline(always)]
    pub fn new(
        output: DisjointSlice<'a, T, LinearTiles<ITEMS>>,
        launch: OneDimensionalLaunch<'_>,
        valid_items: usize,
    ) -> Self {
        let tile = WarpStripedBlockTile::<THREADS, ITEMS>::new(launch, output.len(), valid_items);
        Self::from_tile(output, tile)
    }

    #[inline(always)]
    fn from_tile(
        output: DisjointSlice<'a, T, LinearTiles<ITEMS>>,
        tile: WarpStripedBlockTile<THREADS, ITEMS>,
    ) -> Self {
        Self {
            output,
            tile_base: tile.tile_base,
            thread_index: tile.thread_index,
            valid_items: tile.valid_items,
        }
    }

    /// Borrow two checked input ranges using the block geometry already
    /// proven by this exclusive output capability.
    #[inline(always)]
    pub fn blocked_concatenated_input<'input, U: Copy, const FULL: bool>(
        &self,
        first: &'input [U],
        first_begin: usize,
        first_count: i32,
        second: &'input [U],
        second_begin: usize,
        second_count: i32,
    ) -> BlockedConcatenatedInput<'input, U, THREADS, ITEMS, FULL> {
        WarpStripedBlockTile {
            tile_base: self.tile_base,
            thread_index: self.thread_index,
            valid_items: self.valid_items,
            output_len: self.output.len(),
        }
        .blocked_concatenated_input::<U, FULL>(
            first,
            first_begin,
            first_count,
            second,
            second_begin,
            second_count,
        )
    }

    /// Write this thread's item when it lies in the valid prefix.
    #[inline(always)]
    pub fn write_if_valid(&mut self, item: usize, value: T) {
        if item >= ITEMS {
            return;
        }
        let local = validated_warp_striped_local::<ITEMS>(self.thread_index, item);
        if local >= self.valid_items {
            return;
        }
        let global = self.tile_base + local;
        // SAFETY: `local_index` is injective over (thread, item), the tile
        // bases of different blocks do not overlap, and both prefix and parent
        // bounds were checked by `new`. No reference escapes this call.
        unsafe { *self.output.get_unchecked_mut(global) = value };
    }
}

/// Read phase for an in-place block tile.
///
/// The capability owns the exclusive output packet while exposing only copy
/// reads.  Consuming it with [`into_write`](Self::into_write) performs the CTA
/// phase barrier and is the only way to regain write access.
pub struct WarpStripedBlockRead<'a, T, const THREADS: usize, const ITEMS: usize> {
    output: DisjointSlice<'a, T, LinearTiles<ITEMS>>,
    tile_base: usize,
    thread_index: u32,
    valid_items: usize,
}

impl<'a, T: Copy, const THREADS: usize, const ITEMS: usize>
    WarpStripedBlockRead<'a, T, THREADS, ITEMS>
{
    #[inline(always)]
    pub fn new(
        output: DisjointSlice<'a, T, LinearTiles<ITEMS>>,
        launch: OneDimensionalLaunch<'_>,
        valid_items: usize,
    ) -> Self {
        let (tile_base, thread_index) =
            block_tile::<THREADS, ITEMS>(launch, output.len(), valid_items);
        Self {
            output,
            tile_base,
            thread_index,
            valid_items,
        }
    }

    #[inline(always)]
    fn local_index(&self, item: usize) -> Option<usize> {
        warp_striped_local::<THREADS, ITEMS>(self.thread_index, item)
    }

    /// Copy the first item in the block tile, used by CUB as the guarded-load
    /// default for the final partial tile.
    #[inline(always)]
    pub fn first(&mut self) -> T {
        let ptr = self.output.as_mut_ptr();
        // SAFETY: construction proved `tile_base < output.len()`. Concurrent
        // calls are reads only; the write phase is unavailable until `self`
        // is consumed.
        unsafe { ptr.add(self.tile_base).read() }
    }

    /// Copy this thread's warp-striped item when it lies in the valid prefix.
    #[inline(always)]
    pub fn read(&mut self, item: usize) -> Option<T> {
        let local = self.local_index(item)?;
        if local >= self.valid_items {
            return None;
        }
        let global = self.tile_base + local;
        let ptr = self.output.as_mut_ptr();
        // SAFETY: parent bounds were checked and this phase exposes copy reads
        // only. The exclusive packet remains owned by the capability.
        Some(unsafe { ptr.add(global).read() })
    }

    /// Copy this thread's valid item, otherwise preserve the caller's explicit
    /// CUB guarded-load default.
    #[inline(always)]
    pub fn read_or(&mut self, item: usize, default: T) -> T {
        if item >= ITEMS {
            return default;
        }
        let local = validated_warp_striped_local::<ITEMS>(self.thread_index, item);
        if local >= self.valid_items {
            return default;
        }
        let ptr = self.output.as_mut_ptr();
        // SAFETY: construction proved the parent prefix and this method keeps
        // the capability in its read-only phase.
        unsafe { ptr.add(self.tile_base + local).read() }
    }

    /// End all block reads and return the unique warp-striped writer.
    #[inline(always)]
    pub fn into_write(self) -> WarpStripedBlockWrite<'a, T, THREADS, ITEMS> {
        thread::sync_threads();
        self.into_write_after_barrier()
    }

    #[inline(always)]
    fn into_write_after_barrier(self) -> WarpStripedBlockWrite<'a, T, THREADS, ITEMS> {
        WarpStripedBlockWrite {
            output: self.output,
            tile_base: self.tile_base,
            thread_index: self.thread_index,
            valid_items: self.valid_items,
        }
    }
}

/// Finish the read phase of two independent key/value tiles with one CTA
/// barrier, then return both exclusive writers.
#[inline(always)]
pub fn warp_striped_pair_into_write<
    'keys,
    'values,
    K: Copy,
    V: Copy,
    const THREADS: usize,
    const ITEMS: usize,
>(
    keys: WarpStripedBlockRead<'keys, K, THREADS, ITEMS>,
    values: WarpStripedBlockRead<'values, V, THREADS, ITEMS>,
) -> (
    WarpStripedBlockWrite<'keys, K, THREADS, ITEMS>,
    WarpStripedBlockWrite<'values, V, THREADS, ITEMS>,
) {
    thread::sync_threads();
    (
        keys.into_write_after_barrier(),
        values.into_write_after_barrier(),
    )
}

#[cfg(test)]
mod tests {
    use super::{MergePathSearch, warp_striped_local};

    fn merge_path(first: &[i32], second: &[i32], diagonal: usize) -> usize {
        let mut search = MergePathSearch::new(first, second, diagonal);
        while search.step(|first, second| second < first) {}
        search.partition()
    }

    #[test]
    fn merge_path_search_preserves_stable_diagonal_partitions() {
        let first = [0, 2, 4, 4];
        let second = [0, 3, 4];
        let merged = [0, 0, 2, 3, 4, 4, 4];
        for diagonal in 0..=merged.len() {
            let first_count = merge_path(&first, &second, diagonal);
            let second_count = diagonal - first_count;
            assert!(first_count <= first.len());
            assert!(second_count <= second.len());
            if first_count != 0 && second_count != second.len() {
                assert!(first[first_count - 1] <= second[second_count]);
            }
            if second_count != 0 && first_count != first.len() {
                assert!(second[second_count - 1] < first[first_count]);
            }
        }
    }

    #[test]
    fn warp_striped_mapping_is_a_tile_bijection() {
        const THREADS: usize = 256;
        const ITEMS: usize = 17;
        let mut seen = vec![false; THREADS * ITEMS];
        for thread in 0..THREADS as u32 {
            for item in 0..ITEMS {
                let index = warp_striped_local::<THREADS, ITEMS>(thread, item)
                    .expect("valid thread item must map into the tile");
                assert!(index < seen.len());
                assert!(!seen[index], "duplicate warp-striped index {index}");
                seen[index] = true;
            }
        }
        assert!(seen.into_iter().all(|value| value));
    }

    #[test]
    fn warp_striped_mapping_rejects_invalid_geometry_and_coordinates() {
        assert_eq!(warp_striped_local::<0, 17>(0, 0), None);
        assert_eq!(warp_striped_local::<33, 17>(0, 0), None);
        assert_eq!(warp_striped_local::<256, 0>(0, 0), None);
        assert_eq!(warp_striped_local::<256, 17>(256, 0), None);
        assert_eq!(warp_striped_local::<256, 17>(0, 17), None);
    }

    #[test]
    fn warp_striped_capabilities_fail_closed_and_pair_transition_syncs_once() {
        let source = include_str!("device.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("implementation precedes tests");
        assert!(implementation.contains("OneDimensionalLaunch<'_>"));
        assert!(implementation.contains("thread::blockDim_x() as usize != THREADS"));
        assert!(implementation.contains("cuda_device::debug::trap()"));
        assert!(!implementation.contains("return None; // invalid launch"));

        let transition = implementation
            .split("pub fn warp_striped_pair_into_write")
            .nth(1)
            .expect("pair phase transition must exist");
        assert_eq!(transition.matches("thread::sync_threads()").count(), 1);
        assert_eq!(transition.matches("into_write_after_barrier()").count(), 2);
    }

    #[test]
    fn reusable_tile_rechecks_only_the_parent_extent() {
        let source = include_str!("device.rs");
        let begin = source
            .find("pub struct WarpStripedBlockTile")
            .expect("reusable tile proof must exist");
        let end = source[begin..]
            .find("impl<'a, T: Copy")
            .map(|offset| begin + offset)
            .expect("concatenated input implementation must follow the tile proof");
        let tile = &source[begin..end];
        assert!(tile.contains("output_len: usize"));
        assert!(tile.contains("if output.len() != self.output_len"));
        assert!(!tile.contains("self.tile_base > output.len()"));
    }

    #[test]
    fn concatenated_input_models_zero_count_without_an_invalid_reference() {
        let source = include_str!("device.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("implementation precedes tests");
        let begin = implementation
            .find("pub struct BlockedConcatenatedInput")
            .expect("concatenated input capability must exist");
        let end = implementation[begin..]
            .find("/// Read-only input for one block-wide warp-striped tile")
            .map(|offset| begin + offset)
            .expect("warp-striped input must follow concatenated input");
        let concatenation = &implementation[begin..end];
        assert!(concatenation.contains("first: Option<&'a T>"));
        assert!(concatenation.contains("if first_count == 0"));
        assert!(concatenation.contains("if second_count == 0"));
        assert!(concatenation.contains("None"));
        assert!(concatenation.contains("pub fn load_into"));
        assert!(!concatenation.contains("as_ptr"));
        assert!(!concatenation.contains("from_raw_parts"));
    }
}
