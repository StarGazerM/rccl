//! CUB-level Rust support.
//!
//! These are ordinary runtime values and Rust capability traits. Closed CUB
//! policy selection, kernel construction, and launch lowering belong to
//! `rccl-cub-macros`; support vocabulary does not.

use core::alloc::{Layout, LayoutError};

use cuda_core::DeviceCopy;
use rccl_memory::{DeviceAllocator, DeviceBlock};

pub use rccl_cub_macros::{
    cub_kernels, device_merge_keys, device_merge_pairs, device_merge_sort_keys,
    device_merge_sort_keys_copy, device_merge_sort_pairs, device_merge_sort_pairs_copy,
    device_radix_sort_keys, device_radix_sort_keys_descending, device_radix_sort_pairs,
    device_radix_sort_pairs_descending, device_stable_merge_sort_keys,
    device_stable_merge_sort_keys_copy, device_stable_merge_sort_pairs, warp_reduce,
    warp_reduce_items,
};

/// Borrowed launch context for direct CUB device-wide operations.
///
/// It contains runtime values only. Kernel families and schedules are emitted
/// by `cub_kernels!`; this value does not own a registry, cache, policy, target,
/// or compiler handle.
pub mod cuda {
    use cuda_core::CudaStream;

    #[derive(Clone, Copy)]
    pub struct Execution<'stream, 'module, M> {
        stream: &'stream CudaStream,
        module: &'module M,
    }

    impl<M> Execution<'_, '_, M> {
        pub fn stream(&self) -> &CudaStream {
            self.stream
        }

        #[doc(hidden)]
        pub fn module(&self) -> &M {
            self.module
        }
    }

    pub fn par<'stream, 'module, M>(
        stream: &'stream CudaStream,
        module: &'module M,
    ) -> Execution<'stream, 'module, M> {
        Execution { stream, module }
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
    SizeOverflow,
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
            Self::SizeOverflow => formatter.write_str("combined device range length overflowed"),
            Self::Driver(error) => error.fmt(formatter),
            Self::Launch(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DeviceAlgorithmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Launch(error) => Some(error),
            Self::LengthMismatch { .. } | Self::SizeOverflow => None,
        }
    }
}

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
    pub use rccl_cub_macros::{__device_kernels, __device_transform_repeat_in_place};
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
