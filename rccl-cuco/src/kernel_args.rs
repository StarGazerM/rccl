//! Host-to-kernel adapters private to rccl-cuco.
//!
//! These keep raw pointer reinterpretation at the container boundary while the
//! public API uses Rust borrows and `rccl-memory` contiguous ranges.

use core::marker::PhantomData;
use cuda_host::{KernelSliceArg, KernelSliceArgMut};
use rccl_memory::{ContiguousDeviceRange, ContiguousDeviceRangeMut};

use crate::MapValue;

#[doc(hidden)]
pub struct RangeArg<'a, T, R: ?Sized> {
    range: &'a R,
    marker: PhantomData<fn() -> T>,
}

impl<'a, T, R: ContiguousDeviceRange<T> + ?Sized> RangeArg<'a, T, R> {
    pub fn new(range: &'a R) -> Self {
        Self {
            range,
            marker: PhantomData,
        }
    }
}

unsafe impl<T, R: ContiguousDeviceRange<T> + ?Sized> KernelSliceArg for RangeArg<'_, T, R> {
    type Elem = T;

    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.range.cu_deviceptr()
    }

    fn len(&self) -> usize {
        self.range.len()
    }
}

#[doc(hidden)]
pub struct RangeMutArg<'a, T, R: ?Sized> {
    range: &'a mut R,
    offset: usize,
    len: usize,
    marker: PhantomData<fn() -> T>,
}

impl<'a, T, R: ContiguousDeviceRangeMut<T> + ?Sized> RangeMutArg<'a, T, R> {
    pub fn subrange(range: &'a mut R, offset: usize, len: usize) -> Self {
        assert!(offset <= range.len() && len <= range.len() - offset);
        Self {
            range,
            offset,
            len,
            marker: PhantomData,
        }
    }
}

unsafe impl<T, R: ContiguousDeviceRangeMut<T> + ?Sized> KernelSliceArg for RangeMutArg<'_, T, R> {
    type Elem = T;

    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.range.cu_deviceptr() + (self.offset * core::mem::size_of::<T>()) as u64
    }

    fn len(&self) -> usize {
        self.len
    }
}

unsafe impl<T, R: ContiguousDeviceRangeMut<T> + ?Sized> KernelSliceArgMut
    for RangeMutArg<'_, T, R>
{
}

#[doc(hidden)]
pub struct AtomicU64StorageArg<'a, R: ?Sized> {
    range: &'a mut R,
}

impl<'a, R: ContiguousDeviceRangeMut<u64> + ?Sized> AtomicU64StorageArg<'a, R> {
    pub fn new(range: &'a mut R) -> Self {
        Self { range }
    }

    pub fn subrange(&self, offset: usize, len: usize) -> AtomicU64SubrangeArg<'_> {
        assert!(offset <= self.range.len() && len <= self.range.len() - offset);
        AtomicU64SubrangeArg {
            ptr: self.range.cu_deviceptr()
                + (offset * core::mem::size_of::<u64>()) as cuda_core::sys::CUdeviceptr,
            len,
            owner: PhantomData,
        }
    }

    pub fn plain_subrange(&self, offset: usize, len: usize) -> U64SubrangeArg<'_> {
        assert!(offset <= self.range.len() && len <= self.range.len() - offset);
        U64SubrangeArg {
            ptr: self.range.cu_deviceptr()
                + (offset * core::mem::size_of::<u64>()) as cuda_core::sys::CUdeviceptr,
            len,
            owner: PhantomData,
        }
    }
}

#[doc(hidden)]
pub struct AtomicU64SubrangeArg<'a> {
    ptr: cuda_core::sys::CUdeviceptr,
    len: usize,
    owner: PhantomData<&'a AtomicU64StorageArg<'a, ()>>,
}

unsafe impl KernelSliceArg for AtomicU64SubrangeArg<'_> {
    type Elem = cuda_device::atomic::DeviceAtomicU64;

    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[doc(hidden)]
pub struct U64SubrangeArg<'a> {
    ptr: cuda_core::sys::CUdeviceptr,
    len: usize,
    owner: PhantomData<&'a AtomicU64StorageArg<'a, ()>>,
}

unsafe impl KernelSliceArg for U64SubrangeArg<'_> {
    type Elem = u64;

    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[doc(hidden)]
pub struct AtomicU32RangeMutArg<'a, T, R: ?Sized> {
    range: &'a mut R,
    marker: PhantomData<fn() -> T>,
}

impl<'a, T, R: ContiguousDeviceRangeMut<T> + ?Sized> AtomicU32RangeMutArg<'a, T, R> {
    pub fn new(range: &'a mut R) -> Self {
        assert_eq!(core::mem::size_of::<T>(), core::mem::size_of::<u32>());
        assert!(core::mem::align_of::<T>() >= core::mem::align_of::<u32>());
        Self {
            range,
            marker: PhantomData,
        }
    }
}

unsafe impl<T, R: ContiguousDeviceRangeMut<T> + ?Sized> KernelSliceArg
    for AtomicU32RangeMutArg<'_, T, R>
{
    type Elem = cuda_device::atomic::DeviceAtomicU32;

    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.range.cu_deviceptr()
    }

    fn len(&self) -> usize {
        self.range.len()
    }
}

#[doc(hidden)]
pub struct ValueAtomicRangeMutArg<'a, V, R: ?Sized>
where
    V: MapValue,
{
    range: &'a mut R,
    marker: PhantomData<fn() -> V>,
}

impl<'a, V, R> ValueAtomicRangeMutArg<'a, V, R>
where
    V: MapValue,
    R: ContiguousDeviceRangeMut<V> + ?Sized,
{
    pub fn new(range: &'a mut R) -> Self {
        assert!(core::mem::align_of::<V>() >= core::mem::align_of::<V::Atomic>());
        Self {
            range,
            marker: PhantomData,
        }
    }
}

unsafe impl<V, R> KernelSliceArg for ValueAtomicRangeMutArg<'_, V, R>
where
    V: MapValue,
    R: ContiguousDeviceRangeMut<V> + ?Sized,
{
    type Elem = V::Atomic;

    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.range.cu_deviceptr()
    }

    fn len(&self) -> usize {
        self.range.len()
    }
}

#[doc(hidden)]
pub struct AtomicU32RangeArg<'a, R: ?Sized> {
    range: &'a R,
}

impl<'a, R: ContiguousDeviceRange<u32> + ?Sized> AtomicU32RangeArg<'a, R> {
    pub fn new(range: &'a R) -> Self {
        Self { range }
    }
}

unsafe impl<R: ContiguousDeviceRange<u32> + ?Sized> KernelSliceArg for AtomicU32RangeArg<'_, R> {
    type Elem = cuda_device::atomic::DeviceAtomicU32;

    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.range.cu_deviceptr()
    }

    fn len(&self) -> usize {
        self.range.len()
    }
}

#[doc(hidden)]
pub struct AtomicU64RangeArg<'a, R: ?Sized> {
    range: &'a R,
}

impl<'a, R: ContiguousDeviceRange<u64> + ?Sized> AtomicU64RangeArg<'a, R> {
    pub fn new(range: &'a R) -> Self {
        Self { range }
    }
}

unsafe impl<R: ContiguousDeviceRange<u64> + ?Sized> KernelSliceArg for AtomicU64RangeArg<'_, R> {
    type Elem = cuda_device::atomic::DeviceAtomicU64;

    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.range.cu_deviceptr()
    }

    fn len(&self) -> usize {
        self.range.len()
    }
}

#[doc(hidden)]
pub struct Prefix<'a, T, R: ?Sized> {
    range: &'a R,
    len: usize,
    marker: PhantomData<fn() -> T>,
}

impl<'a, T, R: ContiguousDeviceRange<T> + ?Sized> Prefix<'a, T, R> {
    pub fn new(range: &'a R, len: usize) -> Self {
        assert!(len <= range.len());
        Self {
            range,
            len,
            marker: PhantomData,
        }
    }
}

unsafe impl<T, R: ContiguousDeviceRange<T> + ?Sized> ContiguousDeviceRange<T> for Prefix<'_, T, R> {
    fn len(&self) -> usize {
        self.len
    }

    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.range.cu_deviceptr()
    }

    fn context(&self) -> &std::sync::Arc<cuda_core::CudaContext> {
        self.range.context()
    }

    fn ensure_usable_on(
        &self,
        stream: &cuda_core::CudaStream,
    ) -> Result<(), rccl_memory::MemoryError> {
        self.range.ensure_usable_on(stream)
    }
}
