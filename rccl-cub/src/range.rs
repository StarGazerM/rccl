use cuda_core::DeviceCopy;
use rccl_memory::{DeviceAllocator, TemporaryValue};

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
