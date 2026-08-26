//! CUB-level Rust support.
//!
//! These are ordinary runtime values and Rust capability traits. Closed CUB
//! policy selection, kernel construction, and launch lowering belong to
//! `rccl-cub-macros`; support vocabulary does not.

use core::alloc::{Layout, LayoutError};

use cuda_core::DeviceCopy;
use rccl_memory::{DeviceAllocator, DeviceBlock};

pub use rccl_cub_macros::{warp_reduce, warp_reduce_items};

/// Implementation details used by RCCL's operation macros.
///
/// This is public only because a proc-macro expansion is compiled in the
/// caller's crate. It is not a compatibility API.
#[doc(hidden)]
pub mod __private {
    pub use cuda_core;
    pub use cuda_device;
    pub use cuda_host;
    #[doc(hidden)]
    pub use rccl_cub_macros::{__device_transform_kernels, __device_transform_repeat_in_place};

    use cuda_device::cooperative_groups::{
        ThreadGroup, WarpCollective, WarpTile, this_thread_block,
    };
    use cuda_device::warp;

    /// A scalar that can be exchanged between lanes without exposing a raw
    /// pointer or a C++-style operator policy.
    pub trait WarpReduceValue: Copy {
        fn shuffle_down<const N: u32>(self, tile: &WarpTile<N>, delta: u32) -> Self;

        fn sum_full<const N: u32>(self) -> Self;
    }

    #[inline(always)]
    fn redux_mask<const N: u32>() -> u32 {
        if N == 32 {
            u32::MAX
        } else {
            let first_lane = warp::lane_id() & !(N - 1);
            ((1u32 << N) - 1) << first_lane
        }
    }

    macro_rules! impl_redux_sum_unsigned {
        ($type:ty) => {
            #[inline(always)]
            fn sum_full<const N: u32>(self) -> Self {
                if N == 1 {
                    self
                } else {
                    warp::redux_sync_add(redux_mask::<N>(), self as u32) as Self
                }
            }
        };
    }

    macro_rules! impl_redux_sum_signed {
        ($type:ty) => {
            #[inline(always)]
            fn sum_full<const N: u32>(self) -> Self {
                if N == 1 {
                    self
                } else {
                    warp::redux_sync_add(redux_mask::<N>(), self as i32 as u32) as i32 as Self
                }
            }
        };
    }

    impl WarpReduceValue for u8 {
        #[inline(always)]
        fn shuffle_down<const N: u32>(self, tile: &WarpTile<N>, delta: u32) -> Self {
            tile.shfl_down(self as u32, delta) as Self
        }

        impl_redux_sum_unsigned!(u8);
    }

    impl WarpReduceValue for u16 {
        #[inline(always)]
        fn shuffle_down<const N: u32>(self, tile: &WarpTile<N>, delta: u32) -> Self {
            tile.shfl_down(self as u32, delta) as Self
        }

        impl_redux_sum_unsigned!(u16);
    }

    impl WarpReduceValue for u32 {
        #[inline(always)]
        fn shuffle_down<const N: u32>(self, tile: &WarpTile<N>, delta: u32) -> Self {
            tile.shfl_down(self, delta)
        }

        impl_redux_sum_unsigned!(u32);
    }

    impl WarpReduceValue for i8 {
        #[inline(always)]
        fn shuffle_down<const N: u32>(self, tile: &WarpTile<N>, delta: u32) -> Self {
            tile.shfl_down(self as u32, delta) as Self
        }

        impl_redux_sum_signed!(i8);
    }

    impl WarpReduceValue for i16 {
        #[inline(always)]
        fn shuffle_down<const N: u32>(self, tile: &WarpTile<N>, delta: u32) -> Self {
            tile.shfl_down(self as u32, delta) as Self
        }

        impl_redux_sum_signed!(i16);
    }

    impl WarpReduceValue for i32 {
        #[inline(always)]
        fn shuffle_down<const N: u32>(self, tile: &WarpTile<N>, delta: u32) -> Self {
            tile.shfl_down(self as u32, delta) as Self
        }

        impl_redux_sum_signed!(i32);
    }

    #[inline(always)]
    fn shuffle_u64<const N: u32>(bits: u64, tile: &WarpTile<N>, delta: u32) -> u64 {
        let low = tile.shfl_down(bits as u32, delta) as u64;
        let high = tile.shfl_down((bits >> 32) as u32, delta) as u64;
        low | (high << 32)
    }

    impl WarpReduceValue for u64 {
        #[inline(always)]
        fn shuffle_down<const N: u32>(self, tile: &WarpTile<N>, delta: u32) -> Self {
            shuffle_u64(self, tile, delta)
        }

        #[inline(always)]
        fn sum_full<const N: u32>(self) -> Self {
            warp_reduce_full::<N, _, _>(self, |left, right| left.wrapping_add(right))
        }
    }

    impl WarpReduceValue for i64 {
        #[inline(always)]
        fn shuffle_down<const N: u32>(self, tile: &WarpTile<N>, delta: u32) -> Self {
            shuffle_u64(self as u64, tile, delta) as Self
        }

        #[inline(always)]
        fn sum_full<const N: u32>(self) -> Self {
            warp_reduce_full::<N, _, _>(self, |left, right| left.wrapping_add(right))
        }
    }

    impl WarpReduceValue for f32 {
        #[inline(always)]
        fn shuffle_down<const N: u32>(self, tile: &WarpTile<N>, delta: u32) -> Self {
            tile.shfl_down_f32(self, delta)
        }

        #[inline(always)]
        fn sum_full<const N: u32>(self) -> Self {
            warp_reduce_full::<N, _, _>(self, |left, right| left + right)
        }
    }

    impl WarpReduceValue for f64 {
        #[inline(always)]
        fn shuffle_down<const N: u32>(self, tile: &WarpTile<N>, delta: u32) -> Self {
            f64::from_bits(shuffle_u64(self.to_bits(), tile, delta))
        }

        #[inline(always)]
        fn sum_full<const N: u32>(self) -> Self {
            warp_reduce_full::<N, _, _>(self, |left, right| left + right)
        }
    }

    /// Reduce one value per lane in a complete power-of-two logical warp.
    ///
    /// This is deliberately separate from the runtime-valid-count overload.
    /// The complete-warp form has no lane predicates in its shuffle tree,
    /// matching CUB's full `WarpReduce::Reduce`/`Sum` path.
    #[inline(always)]
    pub fn warp_reduce_full<const N: u32, T, F>(value: T, op: F) -> T
    where
        T: WarpReduceValue,
        F: Fn(T, T) -> T + Copy,
    {
        let tile = this_thread_block().tiled_partition::<N>();
        let mut aggregate = value;
        let mut offset = 1;
        while offset < N {
            aggregate = op(aggregate, aggregate.shuffle_down(&tile, offset));
            offset <<= 1;
        }
        aggregate
    }

    /// Full-warp addition selected from an addition closure by the operation
    /// macro. Integer values up to 32 bits use CUDA's single-instruction
    /// `redux.sync.add`, exactly as current CUB does on sm_80+.
    #[inline(always)]
    pub fn warp_reduce_sum_full<const N: u32, T>(value: T) -> T
    where
        T: WarpReduceValue,
    {
        value.sum_full::<N>()
    }

    /// Reduce one value per lane in a partially populated logical warp.
    ///
    /// Only lane zero is promised to hold the aggregate, matching CUB. The
    /// down-shuffle tree retains left-to-right operand order, so associative
    /// non-commutative closures are supported as well.
    #[inline(always)]
    pub fn warp_reduce<const N: u32, T, F>(value: T, valid_items: u32, op: F) -> T
    where
        T: WarpReduceValue,
        F: Fn(T, T) -> T + Copy,
    {
        let tile = this_thread_block().tiled_partition::<N>();
        let rank = tile.thread_rank();
        let mut aggregate = value;
        let mut offset = 1;
        while offset < N {
            let other = aggregate.shuffle_down(&tile, offset);
            if rank < valid_items && rank + offset < valid_items {
                aggregate = op(aggregate, other);
            }
            offset <<= 1;
        }
        aggregate
    }

    /// Fold a fixed-size per-thread Rust array before the warp-wide phase.
    #[inline(always)]
    pub fn warp_reduce_items<const N: u32, T, F, const ITEMS: usize>(values: [T; ITEMS], op: F) -> T
    where
        T: WarpReduceValue,
        F: Fn(T, T) -> T + Copy,
    {
        const {
            assert!(ITEMS > 0, "warp_reduce_items! requires a non-empty array");
        }
        let mut thread_aggregate = values[0];
        let mut item = 1;
        while item < ITEMS {
            thread_aggregate = op(thread_aggregate, values[item]);
            item += 1;
        }
        warp_reduce_full::<N, _, _>(thread_aggregate, op)
    }

    /// Fixed-array addition followed by the same builtin full-warp sum path
    /// used by CUB after its per-thread fold.
    #[inline(always)]
    pub fn warp_reduce_items_sum<const N: u32, T, F, const ITEMS: usize>(
        values: [T; ITEMS],
        op: F,
    ) -> T
    where
        T: WarpReduceValue,
        F: Fn(T, T) -> T + Copy,
    {
        const {
            assert!(ITEMS > 0, "warp_reduce_items! requires a non-empty array");
        }
        let mut thread_aggregate = values[0];
        let mut item = 1;
        while item < ITEMS {
            thread_aggregate = op(thread_aggregate, values[item]);
            item += 1;
        }
        warp_reduce_sum_full::<N, _>(thread_aggregate)
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

/// Two same-shaped ranges used by ping-pong algorithms.
///
/// Passing two mutable borrows makes exclusivity a normal Rust property.
pub struct DoubleBuffer<R> {
    first: R,
    second: R,
    current_is_first: bool,
    len: usize,
}

impl<R: DeviceRange> DoubleBuffer<R> {
    pub fn new(first: R, second: R) -> Option<Self> {
        let len = first.len();
        (second.len() == len).then_some(Self {
            first,
            second,
            current_is_first: true,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

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

    pub fn swap(&mut self) {
        self.current_is_first = !self.current_is_first;
    }

    pub fn into_parts(self) -> (R, R) {
        (self.first, self.second)
    }
}

/// Size and alignment computed by one macro-generated operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TempStorageLayout {
    layout: Layout,
}

impl TempStorageLayout {
    pub fn new(size: usize, align: usize) -> Result<Self, LayoutError> {
        Ok(Self {
            layout: Layout::from_size_align(size, align)?,
        })
    }

    pub fn layout(self) -> Layout {
        self.layout
    }

    pub fn size(self) -> usize {
        self.layout.size()
    }

    pub fn align(self) -> usize {
        self.layout.align()
    }
}

/// Why an owned device block cannot satisfy an operation layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TempStorageError {
    TooSmall { required: usize, actual: usize },
    InsufficientAlignment { required: usize, actual: usize },
}

/// A checked borrow of caller-owned device temporary storage.
///
/// The allocation owner retains deallocation responsibility. Generated launch
/// code receives this value through a safe backend boundary.
pub struct TempStorage<'a> {
    block: &'a mut DeviceBlock,
    required: TempStorageLayout,
}

impl<'a> TempStorage<'a> {
    pub fn new(
        block: &'a mut DeviceBlock,
        required: TempStorageLayout,
    ) -> Result<Self, TempStorageError> {
        if block.size() < required.size() {
            return Err(TempStorageError::TooSmall {
                required: required.size(),
                actual: block.size(),
            });
        }
        if block.align() < required.align() {
            return Err(TempStorageError::InsufficientAlignment {
                required: required.align(),
                actual: block.align(),
            });
        }
        Ok(Self { block, required })
    }

    pub fn layout(&self) -> TempStorageLayout {
        self.required
    }

    pub fn block(&self) -> &DeviceBlock {
        self.block
    }

    pub fn block_mut(&mut self) -> &mut DeviceBlock {
        self.block
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::marker::PhantomData;

    struct TestRange<T> {
        len: usize,
        item: PhantomData<T>,
    }

    impl<T> TestRange<T> {
        fn new(len: usize) -> Self {
            Self {
                len,
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
        assert!(DoubleBuffer::new(TestRange::<i32>::new(1), TestRange::new(2)).is_none());
    }

    #[test]
    fn temporary_storage_layout_preserves_size_and_alignment() {
        let layout = TempStorageLayout::new(4097, 256).expect("valid layout must construct");
        assert_eq!(layout.size(), 4097);
        assert_eq!(layout.align(), 256);
        assert!(TempStorageLayout::new(1, 3).is_err());
    }
}
