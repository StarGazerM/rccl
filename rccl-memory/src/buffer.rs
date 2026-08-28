use super::*;

/// Fixed-length typed device allocation owned with an allocator handle and
/// bound to one borrowed CUDA stream.
///
/// `A` follows the same model as `Vec<T, A>`: it is stored by value. Passing
/// `&'a PoolMemoryResource<_>` makes the buffer borrow the pool for `'a`;
/// passing `Arc<_>` explicitly selects shared ownership.
///
/// The borrow checker rejects destroying either a borrowed pool or the stream
/// handle before its buffer:
///
/// ```compile_fail
/// use cuda_core::CudaContext;
/// use rccl_memory::{CudaAsyncMemoryResource, DeviceBuffer, PoolMemoryResource};
///
/// let context = CudaContext::new(0).unwrap();
/// let stream = context.default_stream();
/// let pool = PoolMemoryResource::new(
///     CudaAsyncMemoryResource::new(context),
///     0,
///     0,
/// )
/// .unwrap();
/// let buffer = DeviceBuffer::<u8, _>::zeroed_in(&pool, &stream, 16).unwrap();
/// drop(pool); // cannot move out of `pool` because `buffer` borrows it
/// drop(buffer);
/// ```
///
/// The same rule applies to the stream owner:
///
/// ```compile_fail
/// use cuda_core::CudaContext;
/// use rccl_memory::{CudaAsyncMemoryResource, DeviceBuffer};
///
/// let context = CudaContext::new(0).unwrap();
/// let stream = context.default_stream();
/// let allocator = CudaAsyncMemoryResource::new(context.clone());
/// let buffer = DeviceBuffer::<u8, _>::zeroed_in(allocator, &stream, 16).unwrap();
/// drop(stream); // cannot move out of `stream` because `buffer` borrows it
/// drop(buffer);
/// ```
pub struct DeviceBuffer<'stream, T: DeviceCopy, A: DeviceAllocator> {
    block: Option<DeviceBlock>,
    len: usize,
    capacity: usize,
    allocator: A,
    stream: &'stream CudaStream,
    _marker: PhantomData<T>,
}

impl<T: DeviceCopy, A: DeviceAllocator> fmt::Debug for DeviceBuffer<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("ptr", &self.cu_deviceptr())
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("num_bytes", &self.num_bytes())
            .field("device", &self.context().ordinal())
            .finish()
    }
}

impl<'stream, T: DeviceCopy, A: DeviceAllocator> DeviceBuffer<'stream, T, A> {
    pub(crate) fn layout_for(len: usize) -> Result<Layout, MemoryError> {
        if std::mem::size_of::<T>() == 0 {
            return Err(MemoryError::ZeroSizedType);
        }
        let size = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(MemoryError::SizeOverflow)?;
        Ok(Layout::from_size_align(size, std::mem::align_of::<T>())?)
    }

    fn allocate_in(
        allocator: A,
        stream: &'stream CudaStream,
        len: usize,
    ) -> Result<Self, MemoryError> {
        validate_stream(allocator.context(), stream)?;
        let block = allocator.allocate(Self::layout_for(len)?, stream)?;
        Ok(Self {
            block: Some(block),
            len,
            capacity: len,
            allocator,
            stream,
            _marker: PhantomData,
        })
    }

    pub fn zeroed_in(
        allocator: A,
        stream: &'stream CudaStream,
        len: usize,
    ) -> Result<Self, MemoryError> {
        let buffer = Self::allocate_in(allocator, stream, len)?;
        if buffer.num_bytes() > 0 {
            unsafe {
                memset_d8_async(
                    buffer.cu_deviceptr(),
                    0,
                    buffer.num_bytes(),
                    buffer.stream.cu_stream(),
                )?;
            }
        }
        Ok(buffer)
    }

    pub fn from_host_in(
        allocator: A,
        stream: &'stream CudaStream,
        values: &[T],
    ) -> Result<Self, MemoryError> {
        let buffer = Self::allocate_in(allocator, stream, values.len())?;
        if buffer.num_bytes() > 0 {
            unsafe {
                memcpy_htod_async(
                    buffer.cu_deviceptr(),
                    values.as_ptr(),
                    buffer.num_bytes(),
                    buffer.stream.cu_stream(),
                )?;
            }
            buffer.stream.synchronize()?;
        }
        Ok(buffer)
    }

    pub fn copy_from_host(&mut self, values: &[T]) -> Result<(), MemoryError> {
        assert_eq!(
            values.len(),
            self.len,
            "source length must equal device buffer length"
        );
        if self.num_bytes() > 0 {
            unsafe {
                memcpy_htod_async(
                    self.cu_deviceptr(),
                    values.as_ptr(),
                    self.num_bytes(),
                    self.stream.cu_stream(),
                )?;
            }
            self.stream.synchronize()?;
        }
        Ok(())
    }

    /// Allocate at least `new_capacity` elements while preserving the current
    /// contents. Capacity growth and the old allocation's release are ordered
    /// in the buffer's stream and do not synchronize the device.
    pub fn reserve(&mut self, new_capacity: usize) -> Result<(), MemoryError> {
        if new_capacity <= self.capacity {
            return Ok(());
        }
        self.reallocate(new_capacity)
    }

    /// Change the logical length and zero-initialize any newly exposed tail.
    pub fn resize_zeroed(&mut self, new_len: usize) -> Result<(), MemoryError> {
        let old_len = self.len;
        self.reserve(new_len)?;
        if new_len > old_len {
            let element_size = std::mem::size_of::<T>();
            let offset = old_len
                .checked_mul(element_size)
                .ok_or(MemoryError::SizeOverflow)?;
            let tail_bytes = (new_len - old_len)
                .checked_mul(element_size)
                .ok_or(MemoryError::SizeOverflow)?;
            unsafe {
                memset_d8_async(
                    self.cu_deviceptr() + offset as sys::CUdeviceptr,
                    0,
                    tail_bytes,
                    self.stream.cu_stream(),
                )?;
            }
        }
        self.len = new_len;
        Ok(())
    }

    /// Reallocate so capacity equals length while preserving current contents.
    pub fn shrink_to_fit(&mut self) -> Result<(), MemoryError> {
        if self.capacity != self.len {
            self.reallocate(self.len)?;
        }
        Ok(())
    }

    /// Make a stream-ordered deep copy using the same memory resource.
    pub fn try_clone(&self) -> Result<Self, MemoryError>
    where
        A: Clone,
    {
        let copy = Self::allocate_in(self.allocator.clone(), self.stream, self.len)?;
        if self.num_bytes() > 0 {
            unsafe {
                memcpy_dtod_async(
                    copy.cu_deviceptr(),
                    self.cu_deviceptr(),
                    self.num_bytes(),
                    self.stream.cu_stream(),
                )?;
            }
        }
        Ok(copy)
    }

    fn reallocate(&mut self, new_capacity: usize) -> Result<(), MemoryError> {
        let new_layout = Self::layout_for(new_capacity)?;
        let new_block = self.allocator.allocate(new_layout, self.stream)?;
        let bytes_to_copy = self
            .len
            .min(new_capacity)
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(MemoryError::SizeOverflow)?;
        if bytes_to_copy > 0
            && let Err(error) = unsafe {
                memcpy_dtod_async(
                    new_block.cu_deviceptr(),
                    self.cu_deviceptr(),
                    bytes_to_copy,
                    self.stream.cu_stream(),
                )
            }
        {
            let _ = unsafe { self.allocator.deallocate(new_block, self.stream) };
            return Err(error.into());
        }

        let old_block = self
            .block
            .take()
            .expect("device buffer always owns a block");
        if let Err(error) = unsafe { self.allocator.deallocate(old_block, self.stream) } {
            // The old affine token was consumed even though its release may
            // have leaked. Keep the successfully copied replacement so this
            // buffer remains internally valid when the error is observed.
            self.block = Some(new_block);
            self.capacity = new_capacity;
            self.len = self.len.min(new_capacity);
            return Err(error);
        }
        self.block = Some(new_block);
        self.capacity = new_capacity;
        self.len = self.len.min(new_capacity);
        Ok(())
    }

    pub fn to_host_vec(&self) -> Result<Vec<T>, MemoryError> {
        let mut host = Vec::with_capacity(self.len);
        if self.num_bytes() > 0 {
            unsafe {
                memcpy_dtoh_async(
                    host.as_mut_ptr(),
                    self.cu_deviceptr(),
                    self.num_bytes(),
                    self.stream.cu_stream(),
                )?;
            }
            self.stream.synchronize()?;
        }
        unsafe { host.set_len(self.len) };
        Ok(host)
    }

    pub const fn stream(&self) -> &'stream CudaStream {
        self.stream
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn num_bytes(&self) -> usize {
        self.len.saturating_mul(std::mem::size_of::<T>())
    }

    /// Enqueue a byte-zero fill on this buffer's statically bound stream.
    ///
    /// Ordering workspaces use this for CUB algorithms whose published agent
    /// graph specifies `cudaMemsetAsync` between kernel launches. The method
    /// performs no allocation and does not synchronize the stream.
    pub fn zero_async(&mut self) -> Result<(), MemoryError> {
        if self.num_bytes() != 0 {
            unsafe {
                memset_d8_async(
                    self.cu_deviceptr(),
                    0,
                    self.num_bytes(),
                    self.stream.cu_stream(),
                )?;
            }
        }
        Ok(())
    }

    pub fn allocation_bytes(&self) -> usize {
        self.block.as_ref().map_or(0, DeviceBlock::size)
    }

    pub fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.block.as_ref().map_or(0, DeviceBlock::cu_deviceptr)
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        self.allocator.context()
    }

    /// Deallocate on the statically bound stream and surface submission errors
    /// instead of recording them during `Drop`.
    pub fn release(mut self) -> Result<(), MemoryError> {
        if let Some(block) = self.block.take() {
            unsafe { self.allocator.deallocate(block, self.stream)? };
        }
        Ok(())
    }
}

impl<T: DeviceCopy, A: DeviceAllocator> Drop for DeviceBuffer<'_, T, A> {
    fn drop(&mut self) {
        let Some(block) = self.block.take() else {
            return;
        };
        match unsafe { self.allocator.deallocate(block, self.stream) } {
            Ok(()) => {}
            Err(MemoryError::Driver(error)) => self.context().record_err::<()>(Err(error)),
            Err(_) => {}
        }
    }
}

unsafe impl<T: DeviceCopy, A: DeviceAllocator> ContiguousDeviceRange<T> for DeviceBuffer<'_, T, A> {
    fn len(&self) -> usize {
        self.len
    }

    fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.cu_deviceptr()
    }

    fn context(&self) -> &Arc<CudaContext> {
        self.context()
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
        validate_stream(self.context(), stream)?;
        if self.stream.cu_stream() != stream.cu_stream() {
            return Err(MemoryError::StreamMismatch);
        }
        Ok(())
    }
}

unsafe impl<T: DeviceCopy, A: DeviceAllocator> ContiguousDeviceRangeMut<T>
    for DeviceBuffer<'_, T, A>
{
}

/// Typed device storage whose elements may not yet be initialized.
///
/// This type deliberately does not implement `ContiguousDeviceRange<T>` and cannot copy
/// values to the host as initialized `T`.
///
/// ```compile_fail
/// use rccl_memory::{DeviceAllocator, ContiguousDeviceRangeMut, DeviceUninitBuffer};
///
/// fn needs_initialized<T, B: ContiguousDeviceRangeMut<T>>(_: &mut B) {}
/// fn rejected<T, A: DeviceAllocator>(buffer: &mut DeviceUninitBuffer<'_, T, A>)
/// where
///     T: cuda_core::DeviceCopy,
/// {
///     needs_initialized(buffer);
/// }
/// ```
pub struct DeviceUninitBuffer<'stream, T: DeviceCopy, A: DeviceAllocator> {
    storage: DeviceBuffer<'stream, MaybeUninit<T>, A>,
}

impl<T: DeviceCopy, A: DeviceAllocator> fmt::Debug for DeviceUninitBuffer<'_, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceUninitBuffer")
            .field("ptr", &self.cu_deviceptr())
            .field("len", &self.len())
            .field("device", &self.context().ordinal())
            .finish()
    }
}

impl<'stream, T: DeviceCopy, A: DeviceAllocator> DeviceUninitBuffer<'stream, T, A> {
    /// Allocate storage for `len` values without initializing them.
    pub fn new_in(
        allocator: A,
        stream: &'stream CudaStream,
        len: usize,
    ) -> Result<Self, MemoryError> {
        Ok(Self {
            storage: DeviceBuffer::allocate_in(allocator, stream, len)?,
        })
    }

    pub fn len(&self) -> usize {
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.storage.cu_deviceptr()
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        self.storage.context()
    }

    pub fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
        self.storage.ensure_usable_on(stream)
    }

    /// Treat every element as initialized.
    ///
    /// # Safety
    ///
    /// Every element must contain a valid `T`, and the writes establishing
    /// initialization must be ordered before subsequent reads.
    pub unsafe fn assume_init(self) -> DeviceBuffer<'stream, T, A> {
        let storage = std::mem::ManuallyDrop::new(self.storage);
        DeviceBuffer {
            block: unsafe { std::ptr::read(&storage.block) },
            len: storage.len,
            capacity: storage.capacity,
            allocator: unsafe { std::ptr::read(&storage.allocator) },
            stream: storage.stream,
            _marker: PhantomData,
        }
    }
}

/// One typed value in device memory with explicit stream ordering.
#[derive(Debug)]
pub struct DeviceScalar<'stream, T: DeviceCopy, A: DeviceAllocator> {
    storage: DeviceBuffer<'stream, T, A>,
}

impl<'stream, T: DeviceCopy, A: DeviceAllocator> DeviceScalar<'stream, T, A> {
    pub fn new_in(
        allocator: A,
        stream: &'stream CudaStream,
        value: T,
    ) -> Result<Self, MemoryError> {
        Ok(Self {
            storage: DeviceBuffer::from_host_in(allocator, stream, std::slice::from_ref(&value))?,
        })
    }

    pub fn set(&mut self, value: T) -> Result<(), MemoryError> {
        self.storage.copy_from_host(std::slice::from_ref(&value))
    }

    pub fn get(&self) -> Result<T, MemoryError> {
        Ok(self.storage.to_host_vec()?.remove(0))
    }

    pub fn as_buffer(&self) -> &DeviceBuffer<'stream, T, A> {
        &self.storage
    }

    pub fn as_buffer_mut(&mut self) -> &mut DeviceBuffer<'stream, T, A> {
        &mut self.storage
    }
}

unsafe impl<T: DeviceCopy, A: DeviceAllocator> ContiguousDeviceRange<T> for DeviceScalar<'_, T, A> {
    fn len(&self) -> usize {
        1
    }

    fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.storage.cu_deviceptr()
    }

    fn context(&self) -> &Arc<CudaContext> {
        self.storage.context()
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
        self.storage.ensure_usable_on(stream)
    }
}

unsafe impl<T: DeviceCopy, A: DeviceAllocator> ContiguousDeviceRangeMut<T>
    for DeviceScalar<'_, T, A>
{
}
