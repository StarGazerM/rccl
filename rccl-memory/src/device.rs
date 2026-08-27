//! Affine device-memory capabilities used by RCCL collectives.
//!
//! CUDA kernels sometimes use a layout whose ownership unit is wider than one
//! hardware thread.  This module contains the small trusted boundary that
//! maps those layouts to cuda-oxide's exclusive [`DisjointSlice`] packet.  CUB
//! algorithms consume the safe capabilities; raw pointer extraction stays in
//! this memory layer and is never emitted by an operation macro.

use core::{
    marker::PhantomData,
    mem::{MaybeUninit, align_of, size_of},
};
use cuda_device::{
    DisjointSlice, DynamicSharedArray, LinearTiles,
    atomic::{AtomicOrdering, BlockAtomicI32, BlockAtomicU32, DeviceAtomicI32, DeviceAtomicU64},
    fence,
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

/// One block's dynamically-sized shared-memory allocation.
///
/// cuda-oxide exposes dynamic shared storage as an address because the launch
/// determines its extent. RCCL's proc macros already determine that extent;
/// this capability records it once and hands algorithms checked typed regions
/// instead of making every collective recover and cast the address.
pub struct BlockSharedMemory {
    base: *mut u8,
    bytes: usize,
    _not_send_or_sync: PhantomData<*mut ()>,
}

impl BlockSharedMemory {
    /// Bind the macro-declared dynamic shared extent for this kernel block.
    #[inline(always)]
    pub fn new(bytes: usize) -> Self {
        if bytes == 0 {
            cuda_device::debug::trap();
        }
        Self {
            base: DynamicSharedArray::<u8, 16>::get_raw(),
            bytes,
            _not_send_or_sync: PhantomData,
        }
    }

    #[inline(always)]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Create a typed region after validating its complete byte range and
    /// alignment against the macro-declared allocation.
    #[inline(always)]
    pub fn region<T>(&self, byte_offset: usize, len: usize) -> SharedRegion<T> {
        let element_bytes = size_of::<T>();
        if element_bytes == 0
            || !byte_offset.is_multiple_of(align_of::<T>())
            || byte_offset > self.bytes
            || len > (self.bytes - byte_offset) / element_bytes
        {
            cuda_device::debug::trap();
        }
        SharedRegion {
            // SAFETY: the checks above prove the complete region lies in the
            // launch-provided allocation with the alignment required by T.
            base: unsafe { self.base.add(byte_offset).cast::<T>() },
            len,
            _not_send_or_sync: PhantomData,
        }
    }

    /// Copy one checked global range into this block's byte storage using the
    /// pinned SM80 16-byte asynchronous-copy schedule.
    #[inline(always)]
    pub fn copy_async_from<T, const THREADS: usize, const SIZE: usize>(
        &self,
        destination_offset: usize,
        source: &[T],
        source_begin: usize,
        count: u32,
        thread_index: u32,
    ) -> AsyncSharedCopy<T> {
        let count = count as usize;
        if SIZE != size_of::<T>()
            || align_of::<T>() > 16
            || destination_offset & 15 != 0
            || destination_offset > self.bytes
            || source_begin > source.len()
            || count > source.len() - source_begin
        {
            cuda_device::debug::trap();
        }
        let num_bytes = count * SIZE;
        if num_bytes == 0 {
            return AsyncSharedCopy {
                region: SharedRegion {
                    // SAFETY: the destination check above proves the empty
                    // tail starts within the 16-byte-aligned allocation.
                    base: unsafe { self.base.add(destination_offset).cast::<MaybeUninit<T>>() },
                    len: (self.bytes - destination_offset) / SIZE,
                    _not_send_or_sync: PhantomData,
                },
                head_padding: 0,
            };
        }

        // SAFETY: the nonempty checked range has a first element, and the
        // returned address remains borrowed from `source` for this call.
        let global = unsafe { source.as_ptr().add(source_begin) };
        let address_mod_16 = global.addr() & 15;
        let align_difference = if SIZE >= 16 {
            0
        } else {
            (16 - address_mod_16) & 15
        };
        let head_padding = if SIZE >= 16 {
            0
        } else {
            (16 - align_difference) & 15
        };
        let destination_begin = destination_offset + head_padding;
        if destination_begin > self.bytes || num_bytes > self.bytes - destination_begin {
            cuda_device::debug::trap();
        }

        // SAFETY: the preceding allocation check covers the destination, and
        // the source range check covers the same number of bytes.
        let actual_shared = unsafe { self.base.add(destination_begin) };
        let head_bytes = align_difference.min(num_bytes);
        let bulk_bytes = (num_bytes - head_bytes) / 16 * 16;
        let aligned_global = unsafe { global.cast::<u8>().add(head_bytes) };

        let mut offset = thread_index as usize * 16;
        while offset < bulk_bytes {
            // SAFETY: both addresses are 16-byte aligned and each block thread
            // owns a distinct 16-byte chunk in this grid-stride loop.
            unsafe {
                cuda_device::async_copy::cp_async_cg_16(
                    actual_shared.add(head_bytes + offset).cast::<u32>(),
                    aligned_global.add(offset).cast::<u32>(),
                )
            };
            offset += THREADS * 16;
        }

        let begin_offset = if (thread_index as usize) < head_bytes {
            0
        } else {
            bulk_bytes
        };
        let byte_index = begin_offset + thread_index as usize;
        if byte_index < num_bytes {
            // SAFETY: the branch proves one byte in each checked range.
            unsafe {
                actual_shared
                    .add(byte_index)
                    .write(global.cast::<u8>().add(byte_index).read())
            };
        }
        AsyncSharedCopy {
            region: SharedRegion {
                // SAFETY: `destination_offset` is 16-byte aligned and the
                // computed padding gives this address the same modulo-16
                // alignment as the borrowed `T` source address. The checked
                // destination range starts here and lies in the allocation.
                base: actual_shared.cast::<MaybeUninit<T>>(),
                len: (self.bytes - destination_begin) / SIZE,
                _not_send_or_sync: PhantomData,
            },
            head_padding: head_padding as u32,
        }
    }
}

/// Proof produced by one checked asynchronous global-to-shared copy.
///
/// Besides the copied prefix, CUB's merge agents address a second copied range
/// relative to the first range's aligned start. The tail region preserves that
/// coordinate system without asking the algorithm to reconstruct or recheck a
/// raw shared-memory address.
pub struct AsyncSharedCopy<T> {
    region: SharedRegion<MaybeUninit<T>>,
    head_padding: u32,
}

impl<T> AsyncSharedCopy<T> {
    #[inline(always)]
    pub fn head_padding(&self) -> u32 {
        self.head_padding
    }

    #[inline(always)]
    pub fn into_tail_region(self) -> SharedRegion<MaybeUninit<T>> {
        self.region
    }
}

/// Typed, bounds-checked view into one block's dynamic shared allocation.
///
/// The view deliberately does not yield references: CUDA barriers and warp
/// collectives define phase ownership, which ordinary Rust borrows cannot
/// express across hardware threads. Loads and stores remain small reviewed
/// operations in this memory layer.
pub struct SharedRegion<T> {
    base: *mut T,
    len: usize,
    _not_send_or_sync: PhantomData<*mut ()>,
}

impl<T> SharedRegion<T> {
    #[inline(always)]
    fn checked(&self, index: usize) -> *mut T {
        debug_assert!(index < self.len);
        // SAFETY: the branch proves the requested element lies in the region
        // that BlockSharedMemory validated as a whole. Release kernels rely on
        // the collective's reviewed phase/index invariant, just as Rust's
        // standard-library sort internals rely on their partition invariant.
        unsafe { self.base.add(index) }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn read(&self, index: usize) -> T
    where
        T: Copy,
    {
        // SAFETY: `checked` proves the address; the collective's phase
        // protocol establishes initialization before this read.
        unsafe { self.checked(index).read() }
    }

    #[inline(always)]
    pub fn write(&mut self, index: usize, value: T) {
        // SAFETY: `checked` proves the address; each collective supplies an
        // injective lane/index mapping or an atomic operation for collisions.
        unsafe { self.checked(index).write(value) };
    }

    #[inline(always)]
    pub fn read_volatile(&self, index: usize) -> T
    where
        T: Copy,
    {
        // SAFETY: same checked-region proof as `read`; volatile preserves the
        // CUDA publication protocol required by the caller.
        unsafe { self.checked(index).read_volatile() }
    }

    #[inline(always)]
    pub fn write_volatile(&mut self, index: usize, value: T) {
        // SAFETY: same checked-region proof as `write`; volatile preserves the
        // CUDA publication protocol required by the caller.
        unsafe { self.checked(index).write_volatile(value) };
    }
}

impl<T: Copy> SharedRegion<MaybeUninit<T>> {
    /// Load a slot after the collective's barrier/range proof establishes
    /// that it was initialized in the current shared-memory phase.
    #[inline(always)]
    pub fn read_initialized(&self, index: usize) -> T {
        // SAFETY: the caller-visible operation is tied to a CUB phase whose
        // initialized subset is established before it selects this index.
        unsafe { self.read(index).assume_init() }
    }
}

impl SharedRegion<u32> {
    #[inline(always)]
    pub fn fetch_add_block(&mut self, index: usize, value: u32) -> u32 {
        // SAFETY: `checked` returns an aligned u32 in shared memory.
        let atomic = unsafe { BlockAtomicU32::from_ptr(self.checked(index)) };
        atomic.fetch_add(value, AtomicOrdering::Relaxed)
    }
}

impl SharedRegion<i32> {
    #[inline(always)]
    pub fn fetch_add_block(&mut self, index: usize, value: i32) -> i32 {
        // SAFETY: `checked` returns an aligned i32 in shared memory.
        let atomic = unsafe { BlockAtomicI32::from_ptr(self.checked(index)) };
        atomic.fetch_add(value, AtomicOrdering::Relaxed)
    }
}

/// Commit all asynchronous shared-memory copies issued by this block thread.
#[inline(always)]
pub fn commit_async_copy() {
    // SAFETY: collectives call this uniformly after issuing their copy set.
    unsafe { cuda_device::async_copy::cp_async_commit_group() };
}

/// Wait for the committed asynchronous group and publish it to the block.
#[inline(always)]
pub fn wait_async_copy() {
    // SAFETY: a preceding uniform commit created the group being awaited.
    unsafe { cuda_device::async_copy::cp_async_wait_group(0) };
    thread::sync_threads();
}

/// Commit, await, and publish one block asynchronous-copy phase.
#[inline(always)]
pub fn commit_and_wait_async_copy() {
    commit_async_copy();
    wait_async_copy();
}

/// Borrow a register/shared slot after the surrounding collective has proved
/// that the selected lane is active in the current phase.
#[inline(always)]
pub fn initialized_ref<T>(slot: &MaybeUninit<T>) -> &T {
    // SAFETY: RCCL collectives call this only on the active side of their
    // guarded-load or serial-merge availability predicate.
    unsafe { slot.assume_init_ref() }
}

/// Copy a register slot after the surrounding active-lane predicate has
/// established initialization.
#[inline(always)]
pub fn initialized_value<T: Copy>(slot: &MaybeUninit<T>) -> T {
    // SAFETY: same active-lane invariant as `initialized_ref`.
    unsafe { slot.assume_init() }
}

/// Exclusive device allocation viewed through one algorithm-defined index
/// protocol.
///
/// cuda-oxide's `DisjointSlice` intentionally cannot describe a destination
/// index produced by a data-dependent permutation. This wrapper keeps that
/// escape hatch in the memory audit layer while retaining the exclusive
/// allocation borrow and bounds extent as ordinary Rust state.
pub struct DeviceRegion<'a, T> {
    storage: DisjointSlice<'a, T>,
}

impl<'a, T> DeviceRegion<'a, T> {
    #[inline(always)]
    pub fn new(storage: DisjointSlice<'a, T>) -> Self {
        Self { storage }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    #[inline(always)]
    fn checked(&mut self, index: usize) -> *mut T {
        debug_assert!(index < self.storage.len());
        // SAFETY: release collectives derive the index from their reviewed
        // disjoint/permutation protocol; debug kernels retain the extent check.
        unsafe { self.storage.as_mut_ptr().add(index) }
    }

    #[inline(always)]
    pub fn read(&mut self, index: usize) -> T
    where
        T: Copy,
    {
        // SAFETY: `checked` supplies the allocation proof and the collective
        // phase establishes initialization.
        unsafe { self.checked(index).read() }
    }

    #[inline(always)]
    pub fn write(&mut self, index: usize, value: T) {
        // SAFETY: the algorithm-defined index protocol is injective for
        // ordinary writes; colliding protocols use the atomic methods below.
        unsafe { self.checked(index).write(value) };
    }

    #[inline(always)]
    pub fn read_volatile(&mut self, index: usize) -> T
    where
        T: Copy,
    {
        // SAFETY: same allocation proof as `read`; volatile retains the
        // inter-block publication load.
        unsafe { self.checked(index).read_volatile() }
    }

    #[inline(always)]
    pub fn write_volatile(&mut self, index: usize, value: T) {
        // SAFETY: same allocation proof as `write`; volatile retains the
        // inter-block publication store.
        unsafe { self.checked(index).write_volatile(value) };
    }
}

impl DeviceRegion<'_, i32> {
    #[inline(always)]
    pub fn fetch_add_device(&mut self, index: usize, value: i32) -> i32 {
        // SAFETY: `checked` returns one aligned element in device memory.
        let atomic = unsafe { DeviceAtomicI32::from_ptr(self.checked(index)) };
        atomic.fetch_add(value, AtomicOrdering::Relaxed)
    }
}

impl DeviceRegion<'_, u64> {
    #[inline(always)]
    pub fn fetch_add_device(&mut self, index: usize, value: u64) -> u64 {
        // SAFETY: `checked` returns one aligned element in device memory.
        let atomic = unsafe { DeviceAtomicU64::from_ptr(self.checked(index)) };
        atomic.fetch_add(value, AtomicOrdering::Relaxed)
    }
}

/// Preserve CUB's block-scoped anti-hoisting fence in a reviewed device-memory
/// publication protocol.
#[inline(always)]
pub fn threadfence_block() {
    fence::threadfence_block();
}

#[inline(always)]
const fn warp_striped_local<const THREADS: usize, const ITEMS: usize>(
    thread_index: u32,
    item: usize,
) -> Option<usize> {
    if THREADS == 0
        || ITEMS == 0
        || !THREADS.is_multiple_of(32)
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
        || !THREADS.is_multiple_of(32)
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
            (first, first_begin, first_count),
            (second, second_begin, second_count),
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
        first: (&'a [T], usize, i32),
        second: (&'a [T], usize, i32),
        thread_index: u32,
        valid_items: usize,
    ) -> Self {
        let (first, first_begin, first_count) = first;
        let (second, second_begin, second_count) = second;
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

/// Exclusive output for CUB's block-striped arrangement.
///
/// For thread `t`, item `i` maps to `i * THREADS + t`. This is a bijection
/// over one block tile, and distinct CUDA blocks own disjoint tile bases. The
/// wrapper keeps that collective ownership proof in the memory layer instead
/// of making an algorithm recover an output pointer.
pub struct BlockStripedBlockWrite<'a, T, const THREADS: usize, const ITEMS: usize> {
    output: DisjointSlice<'a, T, LinearTiles<ITEMS>>,
    tile_base: usize,
    thread_index: u32,
    valid_items: usize,
}

#[inline(always)]
const fn block_striped_local<const THREADS: usize, const ITEMS: usize>(
    thread_index: u32,
    item: usize,
) -> Option<usize> {
    if THREADS == 0 || ITEMS == 0 || thread_index as usize >= THREADS || item >= ITEMS {
        return None;
    }
    Some(item * THREADS + thread_index as usize)
}

impl<'a, T, const THREADS: usize, const ITEMS: usize>
    BlockStripedBlockWrite<'a, T, THREADS, ITEMS>
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
    pub fn write_if_valid(&mut self, item: usize, value: T) {
        if item >= ITEMS {
            return;
        }
        let Some(local) = block_striped_local::<THREADS, ITEMS>(self.thread_index, item) else {
            return;
        };
        if local >= self.valid_items {
            return;
        }
        let global = self.tile_base + local;
        // SAFETY: `(item, thread_index)` is a bijection over the block tile;
        // `block_tile` proved both the valid prefix and parent allocation.
        unsafe { *self.output.get_unchecked_mut(global) = value };
    }
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
    use super::{MergePathSearch, block_striped_local, warp_striped_local};

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
    fn block_striped_mapping_is_a_tile_bijection_and_fails_closed() {
        const THREADS: usize = 256;
        const ITEMS: usize = 19;
        let mut seen = vec![false; THREADS * ITEMS];
        for thread in 0..THREADS as u32 {
            for item in 0..ITEMS {
                let index = block_striped_local::<THREADS, ITEMS>(thread, item)
                    .expect("valid coordinate must map into the block tile");
                assert!(!seen[index], "duplicate block-striped index {index}");
                seen[index] = true;
            }
        }
        assert!(seen.into_iter().all(|value| value));
        assert_eq!(block_striped_local::<0, 19>(0, 0), None);
        assert_eq!(block_striped_local::<256, 0>(0, 0), None);
        assert_eq!(block_striped_local::<256, 19>(256, 0), None);
        assert_eq!(block_striped_local::<256, 19>(0, 19), None);
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
