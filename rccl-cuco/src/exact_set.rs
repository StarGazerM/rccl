use core::marker::PhantomData;
use std::{error::Error, fmt};

use cuda_core::{
    CudaStream, DeviceCopy, DriverError, LaunchContractError,
    memory::{memcpy_dtoh_async, memset_d8_async},
};
use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64},
    cooperative_groups::{ThreadGroup, WarpCollective},
};
use rccl_memory::{
    BorrowedCudaAsyncMemoryResource, ContiguousDeviceRange, ContiguousDeviceRangeMut,
    CudaCompletion, DeviceAllocator, DeviceBuffer, MemoryError,
};

use crate::{Launch, StableHashBytes, XxHash64, exact_set_backend::ExactSetBackend};

pub const EXACT_SET_COOPERATIVE_GROUP: usize = 4;
const METADATA_WORDS: usize = 4;
const EMPTY: u32 = 0;
const LOCKED: u32 = 1;
const OCCUPIED: u32 = 2;

/// Padding-free HISA row key: `N` contiguous little-endian `u32` columns.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Row<const N: usize> {
    pub data: [u32; N],
}

impl<const N: usize> Row<N> {
    pub const fn new(data: [u32; N]) -> Self {
        Self { data }
    }

    #[inline(always)]
    pub fn same(self, other: Self) -> bool {
        let mut column = 0usize;
        while column < N {
            if self.data[column] != other.data[column] {
                return false;
            }
            column += 1;
        }
        true
    }
}
// SAFETY: every Row value is Copy, contains only DeviceCopy u32 elements, and
// has no padding or invalid bit patterns.
unsafe impl<const N: usize> DeviceCopy for Row<N> {}


impl<const N: usize> From<[u32; N]> for Row<N> {
    fn from(data: [u32; N]) -> Self {
        Self::new(data)
    }
}

// SAFETY: repr(C) preserves the sole array field, whose initialized u32
// elements are contiguous and have the required stable little-endian bytes.
unsafe impl<const N: usize> StableHashBytes for Row<N> {}

#[derive(Debug)]
pub enum ExactSetError {
    Memory(MemoryError),
    Launch(DriverError),
    Contract(LaunchContractError),
    EmptyRow,
    InvalidLoadFactor(f64),
    CapacityOverflow,
    ReservedRows { count: usize },
    OutputTooSmall { needed: usize, available: usize },
    ContextMismatch,
}

impl fmt::Display for ExactSetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory(error) => error.fmt(formatter),
            Self::Launch(error) => write!(formatter, "CUDA exact-set kernel launch failed: {error}"),
            Self::Contract(error) => write!(formatter, "CUDA exact-set launch contract failed: {error}"),
            Self::EmptyRow => formatter.write_str("HISA row keys must contain at least one u32 column"),
            Self::InvalidLoadFactor(value) => write!(
                formatter,
                "exact-set load factor must be finite and in (0, 1], received {value}"
            ),
            Self::CapacityOverflow => formatter.write_str("exact-set capacity overflowed"),
            Self::ReservedRows { count } => write!(
                formatter,
                "input contains {count} row(s) equal to the reserved empty-row sentinel; other rows may already be inserted"
            ),
            Self::OutputTooSmall { needed, available } => write!(
                formatter,
                "exact-set output needs {needed} elements but only {available} are available"
            ),
            Self::ContextMismatch => formatter.write_str(
                "exact-set operands, storage, and execution must share one CUDA context",
            ),
        }
    }
}

impl Error for ExactSetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Memory(error) => Some(error),
            Self::Launch(error) => Some(error),
            Self::Contract(error) => Some(error),
            _ => None,
        }
    }
}

impl From<MemoryError> for ExactSetError {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

impl From<DriverError> for ExactSetError {
    fn from(value: DriverError) -> Self {
        Self::Launch(value)
    }
}

impl From<LaunchContractError> for ExactSetError {
    fn from(value: LaunchContractError) -> Self {
        Self::Contract(value)
    }
}

/// Per-call insertion result. Capacity exhaustion is reported explicitly and
/// never converted into membership, so callers can fail open to an exact
/// fallback without losing tuples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExactSetInsertReport {
    pub inserted: usize,
    pub duplicates: usize,
    pub capacity_exhausted: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CapacitySpec {
    Slots(usize),
    InsertCount { elements: usize, load_factor: f64 },
}

/// Configuration for the workload-specific exact `Row<N>` set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactRowSetConfig<const N: usize> {
    capacity: CapacitySpec,
    empty_row: Row<N>,
    seed: u64,
}

impl<const N: usize> ExactRowSetConfig<N> {
    pub const fn with_capacity(capacity: usize, empty_row: Row<N>) -> Self {
        Self { capacity: CapacitySpec::Slots(capacity), empty_row, seed: 0 }
    }

    pub const fn for_insert_count(elements: usize, load_factor: f64, empty_row: Row<N>) -> Self {
        Self {
            capacity: CapacitySpec::InsertCount { elements, load_factor },
            empty_row,
            seed: 0,
        }
    }

    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn build<'stream, M>(
        self,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<ExactRowSet<'stream, N, BorrowedCudaAsyncMemoryResource<'stream>>, ExactSetError>
    where
        M: ExactSetBackend<N>,
    {
        self.build_in(BorrowedCudaAsyncMemoryResource::new(launch.stream().context()), launch)
    }

    pub fn build_in<'stream, A, M>(
        self,
        allocator: A,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<ExactRowSet<'stream, N, A>, ExactSetError>
    where
        A: DeviceAllocator + Clone,
        M: ExactSetBackend<N>,
    {
        if N == 0 {
            return Err(ExactSetError::EmptyRow);
        }
        let requested = match self.capacity {
            CapacitySpec::Slots(capacity) => capacity,
            CapacitySpec::InsertCount { elements, load_factor } => {
                if !load_factor.is_finite() || load_factor <= 0.0 || load_factor > 1.0 {
                    return Err(ExactSetError::InvalidLoadFactor(load_factor));
                }
                let requested = ((elements as f64) / load_factor).ceil();
                if requested > usize::MAX as f64 {
                    return Err(ExactSetError::CapacityOverflow);
                }
                requested as usize
            }
        };
        ExactRowSet::with_capacity_in(requested, self.empty_row, self.seed, allocator, launch)
    }
}

/// Owning, stream-bound exact set for HISA rows.
pub struct ExactRowSet<'stream, const N: usize, A: DeviceAllocator> {
    slots: DeviceBuffer<'stream, u64, A>,
    states: DeviceBuffer<'stream, u32, A>,
    hashes: DeviceBuffer<'stream, u64, A>,
    words: DeviceBuffer<'stream, u32, A>,
    metadata: DeviceBuffer<'stream, u64, A>,
    capacity: usize,
    empty_row: Row<N>,
    seed: u64,
}

impl<'stream, const N: usize>
    ExactRowSet<'stream, N, BorrowedCudaAsyncMemoryResource<'stream>>
{
    pub fn with_capacity<M>(
        requested_capacity: usize,
        empty_row: Row<N>,
        seed: u64,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<Self, ExactSetError>
    where
        M: ExactSetBackend<N>,
    {
        ExactRowSetConfig::with_capacity(requested_capacity, empty_row)
            .with_seed(seed)
            .build(launch)
    }
}

impl<'stream, const N: usize, A> ExactRowSet<'stream, N, A>
where
    A: DeviceAllocator + Clone,
{
    fn with_capacity_in<M>(
        requested_capacity: usize,
        empty_row: Row<N>,
        seed: u64,
        allocator: A,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<Self, ExactSetError>
    where
        M: ExactSetBackend<N>,
    {
        if N == 0 {
            return Err(ExactSetError::EmptyRow);
        }
        let capacity = valid_capacity(requested_capacity).ok_or(ExactSetError::CapacityOverflow)?;
        let _ = u32::try_from(capacity).map_err(|_| ExactSetError::CapacityOverflow)?;
        let word_count = capacity.checked_mul(N).ok_or(ExactSetError::CapacityOverflow)?;
        let mut slots = DeviceBuffer::zeroed_in(allocator.clone(), launch.stream(), capacity)?;
        let mut states = DeviceBuffer::zeroed_in(allocator.clone(), launch.stream(), capacity)?;
        let hashes = DeviceBuffer::zeroed_in(allocator.clone(), launch.stream(), capacity)?;
        let words = DeviceBuffer::zeroed_in(allocator.clone(), launch.stream(), word_count)?;
        let metadata = DeviceBuffer::zeroed_in(allocator, launch.stream(), METADATA_WORDS)?;
        let submitted = launch.module().exact_set_clear(
            launch.stream(), &mut slots, &mut states, empty_row,
        );
        finish_submission(submitted, launch.stream())?;
        Ok(Self { slots, states, hashes, words, metadata, capacity, empty_row, seed })
    }

    pub const fn capacity(&self) -> usize { self.capacity }
    pub const fn empty_row(&self) -> Row<N> { self.empty_row }
    pub const fn seed(&self) -> u64 { self.seed }
    pub fn stream(&self) -> &'stream CudaStream { self.states.stream() }

    pub fn clear<M>(&mut self, launch: &Launch<'stream, '_, M>) -> Result<(), ExactSetError>
    where
        M: ExactSetBackend<N>,
    {
        self.ensure_launch(launch)?;
        let submitted = launch.module().exact_set_clear(
            launch.stream(), &mut self.slots, &mut self.states, self.empty_row,
        );
        finish_submission(submitted, launch.stream())
    }

    /// Insert rows exactly. Duplicate rows occupy one slot; rows equal to the
    /// configured empty sentinel are rejected, and full-table misses are
    /// counted in `capacity_exhausted` for an exact fail-open fallback.
    pub fn insert<M, R>(
        &mut self,
        rows: &R,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<ExactSetInsertReport, ExactSetError>
    where
        M: ExactSetBackend<N>,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
    {
        self.insert_impl(rows, launch, None)
    }

    #[doc(hidden)]
    pub fn insert_forced_hash<M, R>(
        &mut self,
        rows: &R,
        forced_hash: u64,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<ExactSetInsertReport, ExactSetError>
    where
        M: ExactSetBackend<N>,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
    {
        self.insert_impl(rows, launch, Some(forced_hash))
    }

    fn insert_impl<M, R>(
        &mut self,
        rows: &R,
        launch: &Launch<'stream, '_, M>,
        forced_hash: Option<u64>,
    ) -> Result<ExactSetInsertReport, ExactSetError>
    where
        M: ExactSetBackend<N>,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
    {
        self.ensure_launch(launch)?;
        rows.ensure_usable_on(launch.stream())?;
        if rows.is_empty() { return Ok(ExactSetInsertReport::default()); }
        self.reset_metadata()?;
        let submitted = launch.module().exact_set_insert(
            launch.stream(), &mut self.slots, &mut self.states, &mut self.hashes,
            &mut self.words, &mut self.metadata, rows, self.capacity, self.empty_row,
            self.seed, forced_hash,
        );
        if let Err(error) = submitted {
            let _ = launch.stream().synchronize();
            return Err(error);
        }
        let [inserted, duplicates, exhausted, reserved] = self.read_metadata()?;
        if reserved != 0 {
            return Err(ExactSetError::ReservedRows { count: reserved as usize });
        }
        Ok(ExactSetInsertReport {
            inserted: usize::try_from(inserted).map_err(|_| ExactSetError::CapacityOverflow)?,
            duplicates: usize::try_from(duplicates).map_err(|_| ExactSetError::CapacityOverflow)?,
            capacity_exhausted: usize::try_from(exhausted).map_err(|_| ExactSetError::CapacityOverflow)?,
        })
    }

    pub fn insert_prevalidated<M, R>(
        &mut self,
        rows: &R,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), ExactSetError>
    where
        M: ExactSetBackend<N>,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
    {
        self.insert_prevalidated_async(rows, launch)?.wait()
    }

    pub fn insert_prevalidated_async<'borrow, M, R>(
        &'borrow mut self,
        rows: &'borrow R,
        launch: &'borrow Launch<'stream, '_, M>,
    ) -> Result<PendingExactSetOperation<'borrow>, ExactSetError>
    where
        M: ExactSetBackend<N>,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
    {
        self.ensure_launch(launch)?;
        rows.ensure_usable_on(launch.stream())?;
        let resources = PendingExactSetResources(PhantomData);
        if rows.is_empty() {
            return Ok(PendingExactSetOperation(unsafe {
                CudaCompletion::ready(launch.stream(), resources, Ok(()))
            }));
        }
        launch.module().exact_set_insert_prevalidated(
            launch.stream(), &mut self.slots, &mut self.states, &mut self.hashes,
            &mut self.words, rows, self.capacity, self.empty_row, self.seed,
        )?;
        Ok(PendingExactSetOperation(unsafe {
            CudaCompletion::from_submission(launch.stream(), resources, |_| Ok(()))
        }))
    }

    pub fn contains<M, R, O>(
        &self,
        rows: &R,
        output: &mut O,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), ExactSetError>
    where
        M: ExactSetBackend<N>,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
        O: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        self.contains_async_impl(rows, output, launch, None)?.wait()
    }
    #[doc(hidden)]
    pub fn contains_forced_hash<M, R, O>(
        &self,
        rows: &R,
        output: &mut O,
        forced_hash: u64,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), ExactSetError>
    where
        M: ExactSetBackend<N>,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
        O: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        self.contains_async_impl(rows, output, launch, Some(forced_hash))?.wait()
    }

    pub fn contains_async<'borrow, M, R, O>(
        &'borrow self,
        rows: &'borrow R,
        output: &'borrow mut O,
        launch: &'borrow Launch<'stream, '_, M>,
    ) -> Result<PendingExactSetOperation<'borrow>, ExactSetError>
    where
        M: ExactSetBackend<N>,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
        O: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        self.contains_async_impl(rows, output, launch, None)
    }

    fn contains_async_impl<'borrow, M, R, O>(
        &'borrow self,
        rows: &'borrow R,
        output: &'borrow mut O,
        launch: &'borrow Launch<'stream, '_, M>,
        forced_hash: Option<u64>,
    ) -> Result<PendingExactSetOperation<'borrow>, ExactSetError>
    where
        M: ExactSetBackend<N>,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
        O: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        self.ensure_launch(launch)?;
        rows.ensure_usable_on(launch.stream())?;
        output.ensure_usable_on(launch.stream())?;
        if output.len() < rows.len() {
            return Err(ExactSetError::OutputTooSmall { needed: rows.len(), available: output.len() });
        }
        let resources = PendingExactSetResources(PhantomData);
        if rows.is_empty() {
            return Ok(PendingExactSetOperation(unsafe {
                CudaCompletion::ready(launch.stream(), resources, Ok(()))
            }));
        }
        launch.module().exact_set_contains(
            launch.stream(), &self.slots, &self.states, &self.hashes, &self.words,
            rows, output, self.capacity, self.empty_row, self.seed, forced_hash,
        )?;
        Ok(PendingExactSetOperation(unsafe {
            CudaCompletion::from_submission(launch.stream(), resources, |_| Ok(()))
        }))
    }

    fn ensure_launch<M>(&self, launch: &Launch<'stream, '_, M>) -> Result<(), ExactSetError> {
        if self.states.context() != launch.stream().context() {
            return Err(ExactSetError::ContextMismatch);
        }
        Ok(())
    }

    fn reset_metadata(&mut self) -> Result<(), ExactSetError> {
        let bytes = METADATA_WORDS.checked_mul(core::mem::size_of::<u64>())
            .ok_or(ExactSetError::CapacityOverflow)?;
        unsafe { memset_d8_async(self.metadata.cu_deviceptr(), 0, bytes, self.stream().cu_stream())?; }
        Ok(())
    }

    fn read_metadata(&self) -> Result<[u64; METADATA_WORDS], ExactSetError> {
        let mut values = [0u64; METADATA_WORDS];
        let bytes = METADATA_WORDS.checked_mul(core::mem::size_of::<u64>())
            .ok_or(ExactSetError::CapacityOverflow)?;
        unsafe {
            memcpy_dtoh_async(values.as_mut_ptr(), self.metadata.cu_deviceptr(), bytes, self.stream().cu_stream())?;
        }
        self.stream().synchronize()?;
        Ok(values)
    }
}

struct PendingExactSetResources<'borrow>(PhantomData<&'borrow mut ()>);

/// Completion guard retaining set, input, output, and program borrows.
pub struct PendingExactSetOperation<'borrow>(
    CudaCompletion<'borrow, (), ExactSetError, PendingExactSetResources<'borrow>>,
);

impl PendingExactSetOperation<'_> {
    pub fn wait(self) -> Result<(), ExactSetError> {
        self.0.wait()
    }

    pub fn is_complete(&self) -> Result<bool, DriverError> {
        self.0.is_complete()
    }
}

fn finish_submission(
    submitted: Result<(), ExactSetError>,
    stream: &CudaStream,
) -> Result<(), ExactSetError> {
    let synchronized = stream.synchronize().map_err(ExactSetError::Launch);
    submitted.and(synchronized)
}

fn valid_capacity(requested: usize) -> Option<usize> {
    let groups = requested.max(1).div_ceil(EXACT_SET_COOPERATIVE_GROUP);
    next_prime(groups)?.checked_mul(EXACT_SET_COOPERATIVE_GROUP)
}

fn next_prime(mut value: usize) -> Option<usize> {
    if value <= 2 { return Some(2); }
    value |= 1;
    loop {
        if is_prime(value) { return Some(value); }
        value = value.checked_add(2)?;
    }
}

fn is_prime(value: usize) -> bool {
    if value < 2 { return false; }
    let mut divisor = 2usize;
    while divisor <= value / divisor {
        if value.is_multiple_of(divisor) { return false; }
        divisor += 1;
    }
    true
}

/// Device-side exact read capability. `contains` always verifies every row word
/// after the xxHash64 filter matches.
#[derive(Clone, Copy)]
pub struct ExactRowSetReadRef<'a, const N: usize> {
    states: &'a [DeviceAtomicU32],
    hashes: &'a [DeviceAtomicU64],
    words: &'a [DeviceAtomicU32],
    seed: u64,
}

impl<'a, const N: usize> ExactRowSetReadRef<'a, N> {
    /// # Safety
    /// Buffers must describe one valid set with `states.len() * N == words.len()`.
    pub const unsafe fn new(
        states: &'a [DeviceAtomicU32],
        hashes: &'a [DeviceAtomicU64],
        words: &'a [DeviceAtomicU32],
        seed: u64,
    ) -> Self { Self { states, hashes, words, seed } }

    #[inline(always)]
    pub fn contains(&self, row: Row<N>) -> bool {
        let hash = XxHash64::new(self.seed).hash_value(&row);
        self.contains_hash(row, hash)
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn contains_hash(&self, row: Row<N>, hash: u64) -> bool {
        let groups = self.states.len() / EXACT_SET_COOPERATIVE_GROUP;
        let start = hash as usize % groups;
        let second = XxHash64::new(self.seed.wrapping_add(1)).hash_value(&row);
        let step = second as usize % (groups - 1) + 1;
        let tile = cuda_device::cooperative_groups::this_thread_block()
            .tiled_partition::<{ EXACT_SET_COOPERATIVE_GROUP as u32 }>();
        let rank = tile.thread_rank() as usize;
        let mut group = start;
        let mut visited = 0usize;
        while visited < groups {
            let index = group * EXACT_SET_COOPERATIVE_GROUP + rank;
            let mut state = self.states[index].load(AtomicOrdering::Acquire);
            while state == LOCKED {
                state = self.states[index].load(AtomicOrdering::Acquire);
            }
            let matching = state == OCCUPIED
                && self.hashes[index].load(AtomicOrdering::Relaxed) == hash
                && row_equal(self.words, index, row);
            if tile.ballot(matching) != 0 { return true; }
            if tile.ballot(state == EMPTY) != 0 { return false; }
            visited += 1;
            group += step;
            if group >= groups { group -= groups; }
        }
        false
    }
}

/// Device-side exact mutation capability used by the bulk insertion kernel.
#[derive(Clone, Copy)]
pub struct ExactRowSetRef<'a, const N: usize> {
    read: ExactRowSetReadRef<'a, N>,
}

impl<'a, const N: usize> ExactRowSetRef<'a, N> {
    /// # Safety
    /// Same storage invariants as `ExactRowSetReadRef::new`; calls must be
    /// convergent across four-lane tiles and no non-atomic access may race.
    pub const unsafe fn new(
        states: &'a [DeviceAtomicU32], hashes: &'a [DeviceAtomicU64],
        words: &'a [DeviceAtomicU32], seed: u64,
    ) -> Self {
        Self { read: unsafe { ExactRowSetReadRef::new(states, hashes, words, seed) } }
    }

    #[inline(always)]
    pub fn insert(&self, row: Row<N>) -> ExactInsertOutcome {
        let hash = XxHash64::new(self.read.seed).hash_value(&row);
        self.insert_hash(row, hash)
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn insert_hash(&self, row: Row<N>, hash: u64) -> ExactInsertOutcome {
        let groups = self.read.states.len() / EXACT_SET_COOPERATIVE_GROUP;
        let start = hash as usize % groups;
        let second = XxHash64::new(self.read.seed.wrapping_add(1)).hash_value(&row);
        let step = second as usize % (groups - 1) + 1;
        let tile = cuda_device::cooperative_groups::this_thread_block()
            .tiled_partition::<{ EXACT_SET_COOPERATIVE_GROUP as u32 }>();
        let rank = tile.thread_rank() as usize;
        loop {
            let mut group = start;
            let mut visited = 0usize;
            while visited < groups {
                let index = group * EXACT_SET_COOPERATIVE_GROUP + rank;
                let mut state = self.read.states[index].load(AtomicOrdering::Acquire);
                while state == LOCKED {
                    state = self.read.states[index].load(AtomicOrdering::Acquire);
                }
                let matching = state == OCCUPIED
                    && self.read.hashes[index].load(AtomicOrdering::Relaxed) == hash
                    && row_equal(self.read.words, index, row);
                if tile.ballot(matching) != 0 { return ExactInsertOutcome::Duplicate; }
                let empty = tile.ballot(state == EMPTY);
                if empty != 0 {
                    let owner = empty.trailing_zeros();
                    let status = if rank == owner as usize {
                        match self.read.states[index].compare_exchange(
                            EMPTY, LOCKED, AtomicOrdering::AcqRel, AtomicOrdering::Acquire,
                        ) {
                            Ok(_) => {
                                self.read.hashes[index].store(hash, AtomicOrdering::Relaxed);
                                write_row(self.read.words, index, row);
                                self.read.states[index].store(OCCUPIED, AtomicOrdering::Release);
                                1u32
                            }
                            Err(_) => 0u32,
                        }
                    } else { 0u32 };
                    if tile.shfl(status, owner) != 0 { return ExactInsertOutcome::Inserted; }
                    break;
                }
                visited += 1;
                group += step;
                if group >= groups { group -= groups; }
            }
            if visited == groups { return ExactInsertOutcome::CapacityExhausted; }
        }
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactInsertOutcome { Inserted, Duplicate, CapacityExhausted }

#[inline(always)]
fn row_equal<const N: usize>(words: &[DeviceAtomicU32], slot: usize, row: Row<N>) -> bool {
    let base = slot * N;
    let mut column = 0usize;
    while column < N {
        if words[base + column].load(AtomicOrdering::Relaxed) != row.data[column] { return false; }
        column += 1;
    }
    true
}

#[inline(always)]
fn write_row<const N: usize>(words: &[DeviceAtomicU32], slot: usize, row: Row<N>) {
    let base = slot * N;
    let mut column = 0usize;
    while column < N {
        words[base + column].store(row.data[column], AtomicOrdering::Relaxed);
        column += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_layout_is_hisa_compatible() {
        assert_eq!(core::mem::size_of::<Row<1>>(), 4);
        assert_eq!(core::mem::size_of::<Row<7>>(), 28);
        assert_eq!(core::mem::align_of::<Row<7>>(), 4);
        let row = Row::new([0u32, 1, 0x1234_5678, u32::MAX]);
        assert_eq!(XxHash64::new(0x9e37_79b9_7f4a_7c15).hash_value(&row), 0xbd1c_e3d4_a883_afb8);
    }

    #[test]
    fn capacity_matches_pinned_double_hash_group_rounding() {
        assert_eq!(valid_capacity(0), Some(8));
        assert_eq!(valid_capacity(1), Some(8));
        assert_eq!(valid_capacity(8), Some(8));
        assert_eq!(valid_capacity(9), Some(12));
        assert_eq!(valid_capacity(20), Some(20));
        assert_eq!(valid_capacity(21), Some(28));
    }
}
