//! CUB-level Rust support.
//!
//! These are ordinary runtime values and Rust capability traits. Closed CUB
//! policy selection, kernel construction, and launch lowering belong to
//! `rccl-cub-macros`; support vocabulary does not.

use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_memory::{DeviceAllocator, TemporaryValue};

mod ordering_backend;

#[doc(hidden)]
pub struct RadixSortWorkspaceCounts {
    pub scratch_keys: usize,
    pub bins: usize,
    pub lookback: usize,
    pub counters: usize,
}

#[doc(hidden)]
pub struct RadixSortPairsWorkspaceCounts {
    pub scratch_keys: usize,
    pub scratch_values: usize,
    pub bins: usize,
    pub lookback: usize,
    pub counters: usize,
}

pub use rccl_cub_macros::{
    cub_program, device_merge_keys, device_merge_keys_workspace, device_merge_pairs,
    device_merge_pairs_workspace, device_merge_sort_keys, device_merge_sort_keys_copy,
    device_merge_sort_keys_workspace, device_merge_sort_pairs, device_merge_sort_pairs_copy,
    device_merge_sort_pairs_workspace, device_radix_sort_keys, device_radix_sort_keys_buffer,
    device_radix_sort_keys_buffer_descending, device_radix_sort_keys_buffer_workspace,
    device_radix_sort_keys_descending, device_radix_sort_pairs, device_radix_sort_pairs_buffer,
    device_radix_sort_pairs_buffer_descending, device_radix_sort_pairs_buffer_workspace,
    device_radix_sort_pairs_descending, device_radix_sort_pairs_workspace,
    device_radix_sort_workspace, device_stable_merge_sort_keys, device_stable_merge_sort_keys_copy,
    device_stable_merge_sort_pairs, program, warp_reduce, warp_reduce_items,
};

/// Normalize an owned direct buffer and a borrowed aliased temporary region to
/// the same mutable range surface.
#[doc(hidden)]
pub trait WorkspaceBuffer<T: DeviceCopy> {
    type Range;

    fn range_mut(&mut self) -> &mut Self::Range;
}

impl<T: DeviceCopy> WorkspaceBuffer<T> for DeviceBuffer<T> {
    type Range = DeviceBuffer<T>;

    fn range_mut(&mut self) -> &mut Self::Range {
        self
    }
}

impl<'stream, T: TemporaryValue> WorkspaceBuffer<T>
    for &mut rccl_memory::TemporaryRegion<'stream, T>
{
    type Range = rccl_memory::TemporaryRegion<'stream, T>;

    fn range_mut(&mut self) -> &mut Self::Range {
        self
    }
}

/// Opaque temporary storage for one direct DeviceMerge operation.
pub struct DeviceMergeWorkspace<P = DeviceBuffer<usize>> {
    partitions: P,
}

impl<P: WorkspaceBuffer<usize>> DeviceMergeWorkspace<P> {
    #[doc(hidden)]
    pub fn __from_partitions(partitions: P) -> Self {
        Self { partitions }
    }

    #[doc(hidden)]
    pub fn __partitions_mut(&mut self) -> &mut P::Range {
        self.partitions.range_mut()
    }
}

/// Opaque temporary storage for direct key-only DeviceMergeSort operations.
pub struct DeviceMergeSortKeysWorkspace<K: DeviceCopy, S = DeviceBuffer<K>, P = DeviceBuffer<usize>>
{
    scratch_keys: S,
    partitions: P,
    key: core::marker::PhantomData<fn() -> K>,
}

impl<K: DeviceCopy, S: WorkspaceBuffer<K>, P: WorkspaceBuffer<usize>>
    DeviceMergeSortKeysWorkspace<K, S, P>
{
    #[doc(hidden)]
    pub fn __from_parts(scratch_keys: S, partitions: P) -> Self {
        Self {
            scratch_keys,
            partitions,
            key: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut S::Range, &mut P::Range) {
        (self.scratch_keys.range_mut(), self.partitions.range_mut())
    }
}

/// Opaque temporary storage for direct key/value DeviceMergeSort operations.
pub struct DeviceMergeSortPairsWorkspace<
    K: DeviceCopy,
    V: DeviceCopy,
    S = DeviceBuffer<K>,
    U = DeviceBuffer<V>,
    P = DeviceBuffer<usize>,
> {
    scratch_keys: S,
    scratch_values: U,
    partitions: P,
    types: core::marker::PhantomData<fn() -> (K, V)>,
}

impl<
    K: DeviceCopy,
    V: DeviceCopy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    P: WorkspaceBuffer<usize>,
> DeviceMergeSortPairsWorkspace<K, V, S, U, P>
{
    #[doc(hidden)]
    pub fn __from_parts(scratch_keys: S, scratch_values: U, partitions: P) -> Self {
        Self {
            scratch_keys,
            scratch_values,
            partitions,
            types: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut S::Range, &mut U::Range, &mut P::Range) {
        (
            self.scratch_keys.range_mut(),
            self.scratch_values.range_mut(),
            self.partitions.range_mut(),
        )
    }
}

/// Opaque CUB-owned state used with caller-owned radix double buffers.
pub struct DeviceRadixSortBufferWorkspace<
    B = DeviceBuffer<u64>,
    L = DeviceBuffer<i32>,
    C = DeviceBuffer<i32>,
> {
    bins: B,
    lookback: L,
    counters: C,
}

impl<B: WorkspaceBuffer<u64>, L: WorkspaceBuffer<i32>, C: WorkspaceBuffer<i32>>
    DeviceRadixSortBufferWorkspace<B, L, C>
{
    #[doc(hidden)]
    pub fn __from_parts(bins: B, lookback: L, counters: C) -> Self {
        Self {
            bins,
            lookback,
            counters,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut B::Range, &mut L::Range, &mut C::Range) {
        (
            self.bins.range_mut(),
            self.lookback.range_mut(),
            self.counters.range_mut(),
        )
    }
}

/// Opaque temporary storage for direct key-only DeviceRadixSort operations.
pub struct DeviceRadixSortKeysWorkspace<
    K: DeviceCopy,
    S = DeviceBuffer<K>,
    B = DeviceBuffer<u64>,
    L = DeviceBuffer<i32>,
    C = DeviceBuffer<i32>,
> {
    scratch_keys: S,
    controls: DeviceRadixSortBufferWorkspace<B, L, C>,
    key: core::marker::PhantomData<fn() -> K>,
}

impl<
    K: DeviceCopy,
    S: WorkspaceBuffer<K>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
> DeviceRadixSortKeysWorkspace<K, S, B, L, C>
{
    #[doc(hidden)]
    pub fn __from_parts(scratch_keys: S, bins: B, lookback: L, counters: C) -> Self {
        Self {
            scratch_keys,
            controls: DeviceRadixSortBufferWorkspace::__from_parts(bins, lookback, counters),
            key: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut S::Range, &mut DeviceRadixSortBufferWorkspace<B, L, C>) {
        (self.scratch_keys.range_mut(), &mut self.controls)
    }
}

/// Opaque temporary storage for direct key/value DeviceRadixSort operations.
pub struct DeviceRadixSortPairsWorkspace<
    K: DeviceCopy,
    V: DeviceCopy,
    S = DeviceBuffer<K>,
    U = DeviceBuffer<V>,
    B = DeviceBuffer<u64>,
    L = DeviceBuffer<i32>,
    C = DeviceBuffer<i32>,
> {
    scratch_keys: S,
    scratch_values: U,
    controls: DeviceRadixSortBufferWorkspace<B, L, C>,
    types: core::marker::PhantomData<fn() -> (K, V)>,
}

impl<
    K: DeviceCopy,
    V: DeviceCopy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
> DeviceRadixSortPairsWorkspace<K, V, S, U, B, L, C>
{
    #[doc(hidden)]
    pub fn __from_parts(
        scratch_keys: S,
        scratch_values: U,
        bins: B,
        lookback: L,
        counters: C,
    ) -> Self {
        Self {
            scratch_keys,
            scratch_values,
            controls: DeviceRadixSortBufferWorkspace::__from_parts(bins, lookback, counters),
            types: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(
        &mut self,
    ) -> (
        &mut S::Range,
        &mut U::Range,
        &mut DeviceRadixSortBufferWorkspace<B, L, C>,
    ) {
        (
            self.scratch_keys.range_mut(),
            self.scratch_values.range_mut(),
            &mut self.controls,
        )
    }
}

/// CUDA program ownership and stream binding for direct CUB operations.
///
/// Architecture policy and kernel selection remain entirely in the proc
/// macro. Neither [`cuda::Program`] nor [`cuda::Launch`] stores compiler
/// metadata or a runtime registry.
pub mod cuda {
    use cuda_core::{CudaContext, CudaStream};
    use rccl_memory::MemoryError;

    #[derive(Clone, Copy)]
    pub struct Launch<'stream, 'program, M> {
        stream: &'stream CudaStream,
        module: &'program M,
    }

    impl<M> Launch<'_, '_, M> {
        pub fn stream(&self) -> &CudaStream {
            self.stream
        }

        #[doc(hidden)]
        pub fn module(&self) -> &M {
            self.module
        }
    }

    /// Runtime owner for one macro-generated CUB program.
    pub struct Program<'context, M> {
        context: &'context CudaContext,
        module: M,
    }

    impl<'context, M> Program<'context, M> {
        #[doc(hidden)]
        pub fn __from_module(context: &'context CudaContext, module: M) -> Self {
            Self { context, module }
        }

        /// Bind this program to the stream used by a direct CUB call.
        pub fn on<'program, 'stream>(
            &'program self,
            stream: &'stream CudaStream,
        ) -> Result<Launch<'stream, 'program, M>, MemoryError> {
            if self.context != stream.context().as_ref() {
                return Err(MemoryError::ContextMismatch);
            }
            Ok(Launch {
                stream,
                module: &self.module,
            })
        }
    }
}

/// Failure reported by a generated device-wide ordering operation.
#[derive(Debug)]
pub enum DeviceAlgorithmError {
    LengthMismatch {
        role: &'static str,
        expected: usize,
        actual: usize,
    },
    WorkspaceTooSmall {
        role: &'static str,
        required: usize,
        actual: usize,
    },
    InvalidBitRange {
        begin: u32,
        end: u32,
        key_bits: u32,
    },
    SizeOverflow,
    Memory(rccl_memory::MemoryError),
    Driver(cuda_core::DriverError),
    Launch(cuda_core::LaunchContractError),
}

impl core::fmt::Display for DeviceAlgorithmError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "{role} length {actual} does not match required length {expected}"
            ),
            Self::WorkspaceTooSmall {
                role,
                required,
                actual,
            } => write!(
                formatter,
                "{role} length {actual} is smaller than required length {required}"
            ),
            Self::InvalidBitRange {
                begin,
                end,
                key_bits,
            } => write!(
                formatter,
                "radix bit range {begin}..{end} is outside 0..={key_bits} or reversed"
            ),
            Self::SizeOverflow => formatter.write_str("combined device range length overflowed"),
            Self::Memory(error) => error.fmt(formatter),
            Self::Driver(error) => error.fmt(formatter),
            Self::Launch(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DeviceAlgorithmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Memory(error) => Some(error),
            Self::Launch(error) => Some(error),
            Self::LengthMismatch { .. }
            | Self::WorkspaceTooSmall { .. }
            | Self::InvalidBitRange { .. }
            | Self::SizeOverflow => None,
        }
    }
}

/// Implementation details used by RCCL's operation macros.
///
/// This is public only because a proc-macro expansion is compiled in the
/// caller's crate. It is not a compatibility API.
#[doc(hidden)]
pub mod __private {
    #[doc(hidden)]
    pub use crate::ordering_backend::{
        global_merge_path, merge_keys_agent, merge_pairs_agent, merge_sort_block_keys_agent,
        merge_sort_block_keys_in_place_agent, merge_sort_block_pairs_agent,
        merge_sort_block_pairs_in_place_agent, merge_sort_merge_keys_agent,
        merge_sort_merge_pairs_agent, merge_sort_partition_point, radix_default_u32,
        radix_default_u64, radix_digit_masked_u32, radix_digit_masked_u64, radix_digit_u32,
        radix_digit_u64, radix_exclusive_sum_agent, radix_f32_from_ordered_ascending,
        radix_f32_from_ordered_descending, radix_f32_to_ordered_ascending,
        radix_f32_to_ordered_descending, radix_f64_from_ordered_ascending,
        radix_f64_from_ordered_descending, radix_f64_to_ordered_ascending,
        radix_f64_to_ordered_descending, radix_float_digit_masked_u32,
        radix_float_digit_masked_u64, radix_float_digit_u32, radix_float_digit_u64,
        radix_float_rank_digit_u32, radix_float_rank_digit_u64, radix_histogram_agent,
        radix_i32_from_ordered_ascending, radix_i32_from_ordered_descending,
        radix_i32_to_ordered_ascending, radix_i32_to_ordered_descending,
        radix_i64_from_ordered_ascending, radix_i64_from_ordered_descending,
        radix_i64_to_ordered_ascending, radix_i64_to_ordered_descending,
        radix_init_bins_and_counters_agent, radix_min_ordered_u32, radix_min_ordered_u64,
        radix_onesweep_keys_agent, radix_onesweep_pairs_agent, radix_rank_digit_u32,
        radix_rank_digit_u64, radix_single_tile_keys_agent, radix_single_tile_pairs_agent,
        radix_u32_from_ordered_ascending, radix_u32_from_ordered_descending,
        radix_u32_to_ordered_ascending, radix_u32_to_ordered_descending,
        radix_u64_from_ordered_ascending, radix_u64_from_ordered_descending,
        radix_u64_to_ordered_ascending, radix_u64_to_ordered_descending,
    };
    pub use cuda_core;
    pub use cuda_device;
    pub use cuda_host;
    #[doc(hidden)]
    pub use rccl_cub_macros::{
        __device_kernels, __device_merge_keys_partition_count,
        __device_merge_pairs_partition_count, __device_merge_sort_partition_count,
        __device_radix_sort_pairs_workspace_counts, __device_radix_sort_workspace_counts,
        __device_transform_repeat_in_place,
    };
    pub use rccl_memory::{copy_device_range, static_block_shared_memory};
    /// Normalize cuda-oxide and RCCL allocator buffers for CUB's required
    /// stream-ordered lookback memset. Implemented only for the signed
    /// 32-bit protocol word, so byte-zeroing cannot create an invalid value.
    pub trait ZeroRadixLookback {
        fn zero_radix_lookback(
            &mut self,
            stream: &cuda_core::CudaStream,
        ) -> Result<(), crate::DeviceAlgorithmError>;
    }

    impl ZeroRadixLookback for cuda_core::DeviceBuffer<i32> {
        fn zero_radix_lookback(
            &mut self,
            stream: &cuda_core::CudaStream,
        ) -> Result<(), crate::DeviceAlgorithmError> {
            self.zero_async(stream)
                .map_err(crate::DeviceAlgorithmError::Driver)
        }
    }

    impl<A: rccl_memory::DeviceAllocator> ZeroRadixLookback for rccl_memory::DeviceBuffer<'_, i32, A> {
        fn zero_radix_lookback(
            &mut self,
            stream: &cuda_core::CudaStream,
        ) -> Result<(), crate::DeviceAlgorithmError> {
            rccl_memory::ContiguousDeviceRange::ensure_usable_on(self, stream)
                .map_err(crate::DeviceAlgorithmError::Memory)?;
            self.zero_async()
                .map_err(crate::DeviceAlgorithmError::Memory)
        }
    }

    impl ZeroRadixLookback for rccl_memory::TemporaryRegion<'_, i32> {
        fn zero_radix_lookback(
            &mut self,
            stream: &cuda_core::CudaStream,
        ) -> Result<(), crate::DeviceAlgorithmError> {
            rccl_memory::ContiguousDeviceRange::ensure_usable_on(self, stream)
                .map_err(crate::DeviceAlgorithmError::Memory)?;
            self.zero_async()
                .map_err(crate::DeviceAlgorithmError::Memory)
        }
    }
}

/// A key/value product used when field names are semantically useful.
/// Ordinary products remain Rust tuples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct KeyValue<K, V> {
    pub key: K,
    pub value: V,
}

/// Runtime extent shared by device-range expressions.
///
/// This trait contains no host-side access operation. An algorithm macro
/// lowers a concrete range expression to device access after Rust has inferred
/// its type.
pub trait DeviceRange {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        DeviceRange::len(self) == 0
    }
}

/// A range that generated device code may read.
pub trait ReadRange: DeviceRange {
    type Item: DeviceCopy;
}

/// A range that generated device code may write.
pub trait WriteRange: DeviceRange {
    type Item: DeviceCopy;
}

/// A range supporting constant-time indexed device access.
pub trait RandomAccessRange: DeviceRange {}

/// A random-access range with contiguous device representation.
pub trait ContiguousRange: RandomAccessRange {}

impl<R: DeviceRange + ?Sized> DeviceRange for &R {
    fn len(&self) -> usize {
        DeviceRange::len(*self)
    }
}

impl<R: DeviceRange + ?Sized> DeviceRange for &mut R {
    fn len(&self) -> usize {
        DeviceRange::len(*self)
    }
}

impl<R: ReadRange + ?Sized> ReadRange for &R {
    type Item = R::Item;
}

impl<R: ReadRange + ?Sized> ReadRange for &mut R {
    type Item = R::Item;
}

impl<R: WriteRange + ?Sized> WriteRange for &mut R {
    type Item = R::Item;
}

impl<R: RandomAccessRange + ?Sized> RandomAccessRange for &R {}
impl<R: RandomAccessRange + ?Sized> RandomAccessRange for &mut R {}
impl<R: ContiguousRange + ?Sized> ContiguousRange for &R {}
impl<R: ContiguousRange + ?Sized> ContiguousRange for &mut R {}

impl<T: DeviceCopy> DeviceRange for cuda_core::DeviceBuffer<T> {
    fn len(&self) -> usize {
        cuda_core::DeviceBuffer::<T>::len(self)
    }
}

impl<T: DeviceCopy> ReadRange for cuda_core::DeviceBuffer<T> {
    type Item = T;
}

impl<T: DeviceCopy> WriteRange for cuda_core::DeviceBuffer<T> {
    type Item = T;
}

impl<T: DeviceCopy> RandomAccessRange for cuda_core::DeviceBuffer<T> {}
impl<T: DeviceCopy> ContiguousRange for cuda_core::DeviceBuffer<T> {}

impl<T, A> DeviceRange for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    fn len(&self) -> usize {
        rccl_memory::DeviceBuffer::<T, A>::len(self)
    }
}

impl<T, A> ReadRange for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    type Item = T;
}

impl<T, A> WriteRange for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    type Item = T;
}

impl<T, A> RandomAccessRange for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
}

impl<T, A> ContiguousRange for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
}

impl<T: TemporaryValue> DeviceRange for rccl_memory::TemporaryRegion<'_, T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: TemporaryValue> ReadRange for rccl_memory::TemporaryRegion<'_, T> {
    type Item = T;
}

impl<T: TemporaryValue> WriteRange for rccl_memory::TemporaryRegion<'_, T> {
    type Item = T;
}

impl<T: TemporaryValue> RandomAccessRange for rccl_memory::TemporaryRegion<'_, T> {}
impl<T: TemporaryValue> ContiguousRange for rccl_memory::TemporaryRegion<'_, T> {}

/// Two same-shaped ranges used by ping-pong algorithms.
///
/// Passing two mutable borrows makes exclusivity a normal Rust property.
pub struct DoubleBuffer<First, Second = First> {
    first: First,
    second: Second,
    current_is_first: bool,
    len: usize,
}

impl<First: DeviceRange, Second: DeviceRange> DoubleBuffer<First, Second> {
    pub fn new(first: First, second: Second) -> Option<Self> {
        let len = first.len();
        (second.len() == len).then_some(Self {
            first,
            second,
            current_is_first: true,
            len,
        })
    }
}

impl<FirstKey, FirstValue, SecondKey, SecondValue>
    DoubleBuffer<(FirstKey, FirstValue), (SecondKey, SecondValue)>
where
    FirstKey: DeviceRange,
    FirstValue: DeviceRange,
    SecondKey: DeviceRange,
    SecondValue: DeviceRange,
{
    /// Construct one selector over paired key/value ranges.
    ///
    /// A single selector makes key/value synchronization structural rather
    /// than a convention between independently mutable selector integers.
    pub fn paired(first: (FirstKey, FirstValue), second: (SecondKey, SecondValue)) -> Option<Self> {
        let len = first.0.len();
        (first.1.len() == len && second.0.len() == len && second.1.len() == len).then_some(Self {
            first,
            second,
            current_is_first: true,
            len,
        })
    }
}

impl<First, Second> DoubleBuffer<First, Second> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn current_is_first(&self) -> bool {
        self.current_is_first
    }

    pub fn first(&self) -> &First {
        &self.first
    }

    pub fn second(&self) -> &Second {
        &self.second
    }

    pub fn first_and_second_mut(&mut self) -> (&mut First, &mut Second) {
        (&mut self.first, &mut self.second)
    }

    pub fn swap(&mut self) {
        self.current_is_first = !self.current_is_first;
    }

    pub fn into_parts(self) -> (First, Second) {
        (self.first, self.second)
    }
}

impl<R> DoubleBuffer<R, R> {
    pub fn current(&self) -> &R {
        if self.current_is_first {
            &self.first
        } else {
            &self.second
        }
    }

    pub fn current_mut(&mut self) -> &mut R {
        if self.current_is_first {
            &mut self.first
        } else {
            &mut self.second
        }
    }

    pub fn alternate_mut(&mut self) -> &mut R {
        if self.current_is_first {
            &mut self.second
        } else {
            &mut self.first
        }
    }

    /// Borrow the selected input and alternate output together.
    ///
    /// The two references are disjoint because they come from different
    /// fields. This is the Rust replacement for CUB's pointer/selector pair:
    /// generated algorithms never recover either address or manufacture an
    /// alias.
    pub fn current_and_alternate_mut(&mut self) -> (&mut R, &mut R) {
        if self.current_is_first {
            (&mut self.first, &mut self.second)
        } else {
            (&mut self.second, &mut self.first)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::marker::PhantomData;

    struct TestRange<T> {
        len: usize,
        tag: usize,
        item: PhantomData<T>,
    }

    impl<T> TestRange<T> {
        fn new(len: usize) -> Self {
            Self {
                len,
                tag: 0,
                item: PhantomData,
            }
        }
    }

    impl<T> DeviceRange for TestRange<T> {
        fn len(&self) -> usize {
            self.len
        }
    }

    impl<T: DeviceCopy> ReadRange for TestRange<T> {
        type Item = T;
    }

    impl<T: DeviceCopy> WriteRange for TestRange<T> {
        type Item = T;
    }

    impl<T> RandomAccessRange for TestRange<T> {}
    impl<T> ContiguousRange for TestRange<T> {}

    fn read_i32<R: ReadRange<Item = i32> + RandomAccessRange>(_range: R) {}
    fn write_i32<R: WriteRange<Item = i32> + RandomAccessRange>(_range: R) {}

    #[test]
    fn rust_borrows_supply_range_capabilities() {
        let mut values = TestRange::<i32>::new(3);
        read_i32(&values);
        read_i32(&mut values);
        write_i32(&mut values);
        assert_eq!(DeviceRange::len(&values), 3);
    }

    #[test]
    fn double_buffer_tracks_shape_without_host_data_access() {
        let mut buffers = DoubleBuffer::new(TestRange::<i32>::new(3), TestRange::new(3))
            .expect("equal ranges must construct");
        assert_eq!(buffers.len(), 3);
        assert_eq!(DeviceRange::len(buffers.current()), 3);
        buffers.swap();
        assert_eq!(DeviceRange::len(buffers.alternate_mut()), 3);
        let _parts = buffers.into_parts();
        assert!(DoubleBuffer::new(TestRange::<i32>::new(1), TestRange::<i32>::new(2)).is_none());
    }

    #[test]
    fn double_buffer_split_borrow_follows_the_selected_side() {
        let mut buffers = DoubleBuffer::new(TestRange::<u32>::new(1), TestRange::new(1))
            .expect("equal ranges must construct");
        let (current, alternate) = buffers.current_and_alternate_mut();
        current.tag = 3;
        alternate.tag = 4;
        assert_eq!(buffers.current().tag, 3);

        buffers.swap();
        let (current, alternate) = buffers.current_and_alternate_mut();
        current.tag = 5;
        alternate.tag = 6;
        assert_eq!(buffers.current().tag, 5);
        let (first, second) = buffers.into_parts();
        assert_eq!((first.tag, second.tag), (6, 5));
    }

    #[test]
    fn paired_double_buffer_rejects_every_shape_mismatch() {
        assert!(
            DoubleBuffer::paired(
                (TestRange::<u32>::new(3), TestRange::<i32>::new(3)),
                (TestRange::<u32>::new(3), TestRange::<i32>::new(3)),
            )
            .is_some()
        );
        assert!(
            DoubleBuffer::paired(
                (TestRange::<u32>::new(3), TestRange::<i32>::new(2)),
                (TestRange::<u32>::new(3), TestRange::<i32>::new(3)),
            )
            .is_none()
        );
        assert!(
            DoubleBuffer::paired(
                (TestRange::<u32>::new(3), TestRange::<i32>::new(3)),
                (TestRange::<u32>::new(2), TestRange::<i32>::new(3)),
            )
            .is_none()
        );
    }
}
