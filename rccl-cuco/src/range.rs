use core::marker::PhantomData;

use cuda_core::{CudaStream, DeviceCopy};
use rccl_memory::{ContiguousDeviceRange, DeviceAllocator};

use crate::{MapError, MapKey, MapValue};

pub use rccl_cub::{ArithmeticRange, KeyValue, arithmetic_range};

/// Device-side checked indexed read used by static-map range kernels.
///
/// `None` reports an invalid composed range access (currently an out-of-bounds
/// permutation index) without trapping the CUDA context.
#[doc(hidden)]
pub trait DeviceMapRead: Copy {
    type Item: Copy;

    /// # Safety
    ///
    /// `index` must be inside the host-validated logical range, and every
    /// captured allocation must remain alive for the kernel operation.
    unsafe fn read_at(self, index: usize) -> Option<Self::Item>;
}

/// Device-side indexed write used by static-map range kernels.
#[doc(hidden)]
pub trait DeviceMapWrite: Copy {
    type Item: Copy;

    /// # Safety
    ///
    /// `index` must be inside the logical output range and distinct launched
    /// indices must not alias.
    unsafe fn write_at(self, index: usize, value: Self::Item);
}

#[derive(Clone, Copy)]
pub struct DeviceContiguousRead<T> {
    pointer: *const T,
}

impl<T> DeviceContiguousRead<T> {
    const fn new(pointer: *const T) -> Self {
        Self { pointer }
    }
}

impl<T: Copy> DeviceMapRead for DeviceContiguousRead<T> {
    type Item = T;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> Option<T> {
        // SAFETY: guaranteed by DeviceMapRead::read_at's caller contract.
        Some(unsafe { self.pointer.add(index).read() })
    }
}

#[derive(Clone, Copy)]
pub struct DeviceContiguousWrite<T> {
    pointer: *mut T,
}

impl<T> DeviceContiguousWrite<T> {
    const fn new(pointer: *mut T) -> Self {
        Self { pointer }
    }
}

impl<T: Copy> DeviceMapWrite for DeviceContiguousWrite<T> {
    type Item = T;

    #[inline(always)]
    unsafe fn write_at(self, index: usize, value: T) {
        // SAFETY: guaranteed by DeviceMapWrite::write_at's caller contract.
        unsafe { self.pointer.add(index).write(value) };
    }
}

#[derive(Clone, Copy)]
pub struct DeviceArithmeticRead<T> {
    start: T,
    step: T,
}

impl<T> DeviceArithmeticRead<T> {
    const fn new(start: T, step: T) -> Self {
        Self { start, step }
    }
}

impl<T: rccl_cub::DeviceArithmetic> DeviceMapRead for DeviceArithmeticRead<T> {
    type Item = T;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> Option<T> {
        Some(T::value_at(self.start, self.step, index))
    }
}

#[derive(Clone, Copy)]
pub struct DeviceTransformRead<R, F> {
    inner: R,
    transform: F,
}

impl<R, F, T> DeviceMapRead for DeviceTransformRead<R, F>
where
    R: DeviceMapRead,
    F: Fn(R::Item) -> T + Copy,
    T: Copy,
{
    type Item = T;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> Option<T> {
        // SAFETY: the adaptor preserves the validated logical index.
        unsafe { self.inner.read_at(index) }.map(self.transform)
    }
}

#[derive(Clone, Copy)]
pub struct DevicePermutationRead<R, I> {
    source: R,
    indices: I,
    source_len: usize,
}

impl<R, I> DeviceMapRead for DevicePermutationRead<R, I>
where
    R: DeviceMapRead,
    I: DeviceMapRead<Item = u32>,
{
    type Item = R::Item;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> Option<Self::Item> {
        // SAFETY: the outer kernel bounds index by the index-range extent.
        let selected = unsafe { self.indices.read_at(index) }? as usize;
        if selected >= self.source_len {
            return None;
        }
        // SAFETY: selected was checked against the source extent above.
        unsafe { self.source.read_at(selected) }
    }
}

#[derive(Clone, Copy)]
pub struct DeviceEntriesRead<KR, VR> {
    keys: KR,
    values: VR,
}

impl<KR, VR, K, V> DeviceMapRead for DeviceEntriesRead<KR, VR>
where
    KR: DeviceMapRead<Item = K>,
    VR: DeviceMapRead<Item = V>,
    K: Copy,
    V: Copy,
{
    type Item = KeyValue<K, V>;

    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> Option<Self::Item> {
        // SAFETY: construction proves equal extents and the kernel bounds index.
        Some(KeyValue {
            key: unsafe { self.keys.read_at(index) }?,
            value: unsafe { self.values.read_at(index) }?,
        })
    }
}

/// Finite random-access input accepted by static-map bulk range terminals.
///
/// Implementations lower to a copyable device descriptor. The host range is
/// still borrowed by the terminal or its completion guard, so descriptor raw
/// pointers cannot outlive their allocations.
pub trait MapReadRange<T: DeviceCopy> {
    type Device: DeviceMapRead<Item = T>;

    fn len(&self) -> usize;
    fn lower(&self) -> Self::Device;
    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MapError>;

    /// Source extent to report when a checked device access fails.
    fn failure_source_len(&self) -> Option<usize> {
        None
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Affine random-access output accepted by static-map bulk range terminals.
pub trait MapWriteRange<T: DeviceCopy> {
    type Device: DeviceMapWrite<Item = T>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lower_write(&mut self) -> Self::Device;
    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MapError>;
}
impl<T, R> MapReadRange<T> for &R
where
    T: DeviceCopy,
    R: MapReadRange<T> + ?Sized,
{
    type Device = R::Device;

    fn len(&self) -> usize {
        (**self).len()
    }

    fn lower(&self) -> Self::Device {
        (**self).lower()
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MapError> {
        (**self).ensure_usable_on(stream)
    }

    fn failure_source_len(&self) -> Option<usize> {
        (**self).failure_source_len()
    }
}

impl<T: DeviceCopy> MapReadRange<T> for cuda_core::DeviceBuffer<T> {
    type Device = DeviceContiguousRead<T>;

    fn len(&self) -> usize {
        self.len()
    }

    fn lower(&self) -> Self::Device {
        DeviceContiguousRead::new(self.cu_deviceptr() as usize as *const T)
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MapError> {
        ContiguousDeviceRange::<T>::ensure_usable_on(self, stream).map_err(Into::into)
    }
}

impl<T: DeviceCopy> MapWriteRange<T> for cuda_core::DeviceBuffer<T> {
    type Device = DeviceContiguousWrite<T>;

    fn len(&self) -> usize {
        self.len()
    }

    fn lower_write(&mut self) -> Self::Device {
        DeviceContiguousWrite::new(self.cu_deviceptr() as usize as *mut T)
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MapError> {
        ContiguousDeviceRange::<T>::ensure_usable_on(self, stream).map_err(Into::into)
    }
}

impl<T, A> MapReadRange<T> for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    type Device = DeviceContiguousRead<T>;

    fn len(&self) -> usize {
        self.len()
    }

    fn lower(&self) -> Self::Device {
        DeviceContiguousRead::new(self.cu_deviceptr() as usize as *const T)
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MapError> {
        ContiguousDeviceRange::<T>::ensure_usable_on(self, stream).map_err(Into::into)
    }
}

impl<T, A> MapWriteRange<T> for rccl_memory::DeviceBuffer<'_, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    type Device = DeviceContiguousWrite<T>;

    fn len(&self) -> usize {
        self.len()
    }

    fn lower_write(&mut self) -> Self::Device {
        DeviceContiguousWrite::new(self.cu_deviceptr() as usize as *mut T)
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MapError> {
        ContiguousDeviceRange::<T>::ensure_usable_on(self, stream).map_err(Into::into)
    }
}

impl<T> MapReadRange<T> for ArithmeticRange<T>
where
    T: DeviceCopy
        + rccl_cub::DeviceArithmetic
        + TryFrom<usize>
        + core::ops::Add<Output = T>
        + core::ops::Mul<Output = T>,
{
    type Device = DeviceArithmeticRead<T>;

    fn len(&self) -> usize {
        self.len()
    }

    fn lower(&self) -> Self::Device {
        DeviceArithmeticRead::new(*self.start(), *self.step())
    }

    fn ensure_usable_on(&self, _stream: &CudaStream) -> Result<(), MapError> {
        if self.is_empty() || T::index_fits(self.len() - 1) {
            Ok(())
        } else {
            Err(MapError::CapacityOverflow)
        }
    }
}

/// Zero-materialization transform over a finite device range.
#[derive(Clone, Copy, Debug)]
pub struct Transform<R, F, Input, Output> {
    inner: R,
    transform: F,
    types: PhantomData<fn(Input) -> Output>,
}

impl<R, F, Input, Output> Transform<R, F, Input, Output> {
    pub fn new(inner: R, transform: F) -> Self {
        Self {
            inner,
            transform,
            types: PhantomData,
        }
    }
}

impl<R, F, Input, Output> MapReadRange<Output> for Transform<R, F, Input, Output>
where
    R: MapReadRange<Input>,
    F: Fn(Input) -> Output + Copy,
    Input: DeviceCopy,
    Output: DeviceCopy,
{
    type Device = DeviceTransformRead<R::Device, F>;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn lower(&self) -> Self::Device {
        DeviceTransformRead {
            inner: self.inner.lower(),
            transform: self.transform,
        }
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MapError> {
        self.inner.ensure_usable_on(stream)
    }

    fn failure_source_len(&self) -> Option<usize> {
        self.inner.failure_source_len()
    }
}

/// Zero-materialization selection through a parallel `u32` index range.
#[derive(Clone, Copy, Debug)]
pub struct Permutation<R, I, T> {
    source: R,
    indices: I,
    item: PhantomData<fn() -> T>,
}

impl<R, I, T> Permutation<R, I, T> {
    pub fn new(source: R, indices: I) -> Self {
        Self {
            source,
            indices,
            item: PhantomData,
        }
    }
}

impl<R, I, T> MapReadRange<T> for Permutation<R, I, T>
where
    R: MapReadRange<T>,
    I: MapReadRange<u32>,
    T: DeviceCopy,
{
    type Device = DevicePermutationRead<R::Device, I::Device>;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn lower(&self) -> Self::Device {
        DevicePermutationRead {
            source: self.source.lower(),
            indices: self.indices.lower(),
            source_len: self.source.len(),
        }
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MapError> {
        self.source.ensure_usable_on(stream)?;
        self.indices.ensure_usable_on(stream)
    }

    fn failure_source_len(&self) -> Option<usize> {
        Some(self.source.len())
    }
}

/// Composition methods for static-map input ranges.
pub trait MapRangeExt<T: DeviceCopy>: MapReadRange<T> + Sized {
    fn transform<F, U>(self, transform: F) -> Transform<Self, F, T, U>
    where
        F: Fn(T) -> U + Copy,
        U: DeviceCopy,
    {
        Transform::new(self, transform)
    }

    fn permutation<I>(self, indices: I) -> Permutation<Self, I, T>
    where
        I: MapReadRange<u32>,
    {
        Permutation::new(self, indices)
    }
}

impl<R, T> MapRangeExt<T> for R
where
    R: MapReadRange<T>,
    T: DeviceCopy,
{
}

/// Borrowed structure-of-arrays key/value input for a bulk map operation.
pub struct Entries<'a, K, V, KR: ?Sized, VR: ?Sized> {
    pub(crate) keys: &'a KR,
    pub(crate) values: &'a VR,
    marker: PhantomData<fn() -> (K, V)>,
}

pub fn entries<'a, K, V, KR, VR>(
    keys: &'a KR,
    values: &'a VR,
) -> Result<Entries<'a, K, V, KR, VR>, MapError>
where
    K: MapKey,
    V: MapValue,
    KR: ContiguousDeviceRange<K> + ?Sized,
    VR: ContiguousDeviceRange<V> + ?Sized,
{
    if keys.len() != values.len() {
        return Err(MapError::LengthMismatch {
            keys: keys.len(),
            values: values.len(),
        });
    }
    Ok(Entries {
        keys,
        values,
        marker: PhantomData,
    })
}

impl<K, V, KR: ?Sized, VR: ?Sized> Entries<'_, K, V, KR, VR> {
    pub fn len(&self) -> usize
    where
        KR: ContiguousDeviceRange<K>,
    {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool
    where
        KR: ContiguousDeviceRange<K>,
    {
        self.keys.len() == 0
    }
}

impl<K, V, KR, VR> MapReadRange<KeyValue<K, V>> for Entries<'_, K, V, KR, VR>
where
    K: MapKey,
    V: MapValue,
    KR: ContiguousDeviceRange<K> + ?Sized,
    VR: ContiguousDeviceRange<V> + ?Sized,
{
    type Device = DeviceEntriesRead<DeviceContiguousRead<K>, DeviceContiguousRead<V>>;

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn lower(&self) -> Self::Device {
        DeviceEntriesRead {
            keys: DeviceContiguousRead::new(self.keys.cu_deviceptr() as usize as *const K),
            values: DeviceContiguousRead::new(self.values.cu_deviceptr() as usize as *const V),
        }
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MapError> {
        self.keys.ensure_usable_on(stream)?;
        self.values.ensure_usable_on(stream)?;
        Ok(())
    }
}
