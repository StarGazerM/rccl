use super::*;

/// CUDA device allocations are at least 256-byte aligned.
///
/// Higher alignments require an over-allocation/VMM resource because CUDA's
/// ordinary allocation APIs do not expose an alignment argument.
pub const DEVICE_ALLOCATION_ALIGNMENT: usize = 256;

/// Address-space capabilities carried through resource adaptors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryProperties {
    pub device_accessible: bool,
    pub host_accessible: bool,
    pub prefetchable: bool,
}

impl MemoryProperties {
    pub const DEVICE: Self = Self {
        device_accessible: true,
        host_accessible: false,
        prefetchable: false,
    };
    pub const HOST_DEVICE: Self = Self {
        device_accessible: true,
        host_accessible: true,
        prefetchable: false,
    };
    pub const MANAGED: Self = Self {
        device_accessible: true,
        host_accessible: true,
        prefetchable: true,
    };

    pub(crate) fn intersection(self, other: Self) -> Self {
        Self {
            device_accessible: self.device_accessible && other.device_accessible,
            host_accessible: self.host_accessible && other.host_accessible,
            prefetchable: self.prefetchable && other.prefetchable,
        }
    }
}

/// Errors produced by memory resources and resource-owned buffers.
#[derive(Debug)]
#[non_exhaustive]
pub enum MemoryError {
    Driver(DriverError),
    CudaRuntime(i32),
    Layout(std::alloc::LayoutError),
    SizeOverflow,
    ZeroSizedType,
    UnsupportedAlignment {
        requested: usize,
        supported: usize,
    },
    ContextMismatch,
    StreamMismatch,
    AllocationLimitExceeded {
        requested: usize,
        allocated: usize,
        limit: usize,
    },
    InvalidPoolConfiguration {
        initial_size: usize,
        maximum_size: usize,
    },
    UnknownAllocation {
        ptr: sys::CUdeviceptr,
    },
    AllocationSizeMismatch {
        expected: usize,
        actual: usize,
    },
    LengthMismatch {
        source: usize,
        destination: usize,
    },
    InvalidAlignment {
        requested: usize,
    },
    SystemMemoryUnsupported {
        device: usize,
    },
    HostAllocationFailed {
        requested: usize,
    },
    CounterStackUnderflow,
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => write!(f, "CUDA memory operation failed: {error}"),
            Self::CudaRuntime(status) => {
                write!(
                    f,
                    "CUDA runtime memory operation failed with status {status}"
                )
            }
            Self::Layout(error) => write!(f, "invalid allocation layout: {error}"),
            Self::SizeOverflow => write!(f, "device allocation size overflow"),
            Self::ZeroSizedType => write!(f, "zero-sized device element types are unsupported"),
            Self::UnsupportedAlignment {
                requested,
                supported,
            } => write!(
                f,
                "requested device alignment {requested} exceeds the guaranteed {supported}-byte alignment"
            ),
            Self::ContextMismatch => {
                write!(
                    f,
                    "memory resource, buffer, and stream must share one CUDA context"
                )
            }
            Self::StreamMismatch => write!(f, "buffer is bound to a different CUDA stream"),
            Self::AllocationLimitExceeded {
                requested,
                allocated,
                limit,
            } => write!(
                f,
                "allocation of {requested} bytes would exceed the resource limit: {allocated} of {limit} bytes are already allocated"
            ),
            Self::InvalidPoolConfiguration {
                initial_size,
                maximum_size,
            } => write!(
                f,
                "initial pool size {initial_size} exceeds maximum pool size {maximum_size}"
            ),
            Self::UnknownAllocation { ptr } => {
                write!(f, "pointer 0x{ptr:x} is not allocated by this resource")
            }
            Self::AllocationSizeMismatch { expected, actual } => write!(
                f,
                "allocation descriptor size mismatch: expected {expected} bytes, got {actual}"
            ),
            Self::LengthMismatch {
                source,
                destination,
            } => write!(
                f,
                "device copy source length {source} differs from destination length {destination}"
            ),
            Self::InvalidAlignment { requested } => write!(
                f,
                "allocation alignment {requested} must be a non-zero power of two"
            ),
            Self::SystemMemoryUnsupported { device } => write!(
                f,
                "CUDA device {device} cannot access ordinary pageable system memory"
            ),
            Self::HostAllocationFailed { requested } => {
                write!(f, "host allocation of {requested} bytes failed")
            }
            Self::CounterStackUnderflow => {
                write!(f, "cannot pop the root statistics counter scope")
            }
        }
    }
}

impl Error for MemoryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Layout(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DriverError> for MemoryError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

impl From<std::alloc::LayoutError> for MemoryError {
    fn from(value: std::alloc::LayoutError) -> Self {
        Self::Layout(value)
    }
}

/// A byte allocation returned by a [`DeviceAllocator`].
///
/// This is an affine ownership token. It is intentionally neither `Copy` nor
/// `Clone`: ownership is transferred back to the resource by exactly one
/// `deallocate` call.
///
/// ```compile_fail
/// use rccl_memory::DeviceBlock;
///
/// fn duplicate(block: DeviceBlock) {
///     let first_owner = block;
///     let second_owner = block; // use of moved value
///     drop((first_owner, second_owner));
/// }
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct DeviceBlock {
    ptr: sys::CUdeviceptr,
    size: usize,
    align: usize,
}

impl DeviceBlock {
    /// Construct a resource block from a raw allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must identify `size` live bytes in the resource's CUDA context,
    /// with at least `align` alignment. The allocation must not already be
    /// owned by another RAII value.
    pub const unsafe fn from_raw_parts(ptr: sys::CUdeviceptr, size: usize, align: usize) -> Self {
        Self { ptr, size, align }
    }

    pub const fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }

    pub const fn size(&self) -> usize {
        self.size
    }

    pub const fn align(&self) -> usize {
        self.align
    }

    /// Interpret a host-accessible allocation's unified address as a host
    /// pointer. Dereferencing is only valid when the allocating resource
    /// reports `properties().host_accessible` and CUDA's coherence rules permit
    /// host access at that point.
    pub const fn host_ptr(&self) -> *mut c_void {
        self.ptr as usize as *mut c_void
    }

    pub fn layout(&self) -> Result<Layout, MemoryError> {
        Ok(Layout::from_size_align(self.size, self.align)?)
    }
}

/// Page-locked host memory that is also device accessible under CUDA unified
/// virtual addressing.
///
/// CUDA's pinned-host allocation/free APIs are synchronous. To preserve the
/// crate's stronger stream-order contract, deallocation synchronizes the
/// supplied stream before releasing the host range.
#[derive(Clone, Debug)]
pub struct PinnedHostMemoryResource {
    context: Arc<CudaContext>,
}

impl PinnedHostMemoryResource {
    pub fn new(context: Arc<CudaContext>) -> Self {
        Self { context }
    }
}

unsafe impl DeviceAllocator for PinnedHostMemoryResource {
    fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    fn properties(&self) -> MemoryProperties {
        MemoryProperties::HOST_DEVICE
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        validate_layout(layout)?;
        validate_stream(&self.context, stream)?;
        if layout.size() == 0 {
            return Ok(unsafe { DeviceBlock::from_raw_parts(0, 0, layout.align()) });
        }
        let host_ptr = unsafe { malloc_host(layout.size())? };
        Ok(unsafe {
            DeviceBlock::from_raw_parts(
                host_ptr as usize as sys::CUdeviceptr,
                layout.size(),
                layout.align(),
            )
        })
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        validate_stream(&self.context, stream)?;
        if block.cu_deviceptr() != 0 {
            stream.synchronize()?;
            unsafe { free_host(block.host_ptr())? };
        }
        Ok(())
    }
}

/// Ordinary pageable system memory accessible by both the CPU and CUDA.
///
/// This resource is only available on systems where CUDA reports pageable
/// memory access (for example HMM or ATS). Because host `free` is immediate,
/// deallocation first synchronizes the supplied CUDA stream.
#[derive(Clone, Debug)]
pub struct SystemMemoryResource {
    context: Arc<CudaContext>,
}

impl SystemMemoryResource {
    pub fn is_supported(context: &CudaContext) -> Result<bool, MemoryError> {
        context.bind_to_thread()?;
        let mut supported = MaybeUninit::<i32>::uninit();
        unsafe {
            sys::cuDeviceGetAttribute(
                supported.as_mut_ptr(),
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
                context.cu_device(),
            )
            .result()?;
            Ok(supported.assume_init() != 0)
        }
    }

    pub fn new(context: Arc<CudaContext>) -> Result<Self, MemoryError> {
        if !Self::is_supported(&context)? {
            return Err(MemoryError::SystemMemoryUnsupported {
                device: context.ordinal(),
            });
        }
        Ok(Self { context })
    }
}

unsafe impl DeviceAllocator for SystemMemoryResource {
    fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    fn properties(&self) -> MemoryProperties {
        MemoryProperties::MANAGED
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        validate_layout(layout)?;
        validate_stream(&self.context, stream)?;
        if layout.size() == 0 {
            return Ok(unsafe { DeviceBlock::from_raw_parts(0, 0, layout.align()) });
        }
        let host_layout = Layout::from_size_align(layout.size(), DEVICE_ALLOCATION_ALIGNMENT)?;
        let ptr = unsafe { std::alloc::alloc(host_layout) };
        if ptr.is_null() {
            return Err(MemoryError::HostAllocationFailed {
                requested: layout.size(),
            });
        }
        Ok(unsafe {
            DeviceBlock::from_raw_parts(
                ptr as usize as sys::CUdeviceptr,
                layout.size(),
                layout.align(),
            )
        })
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        validate_stream(&self.context, stream)?;
        if block.cu_deviceptr() != 0 {
            stream.synchronize()?;
            let host_layout = Layout::from_size_align(block.size(), DEVICE_ALLOCATION_ALIGNMENT)?;
            unsafe { std::alloc::dealloc(block.host_ptr().cast(), host_layout) };
        }
        Ok(())
    }
}

/// Pageable system memory with a reserved amount of free GPU memory.
///
/// The portion that fits above `headroom` receives a device preferred-location
/// hint; the remainder receives a host hint. This mirrors RMM's SAM headroom
/// resource and avoids consuming every free device byte through HMM migration.
#[derive(Debug)]
pub struct SamHeadroomMemoryResource {
    system: SystemMemoryResource,
    headroom: usize,
}

impl SamHeadroomMemoryResource {
    pub fn new(context: Arc<CudaContext>, headroom: usize) -> Result<Self, MemoryError> {
        Ok(Self {
            system: SystemMemoryResource::new(context)?,
            headroom,
        })
    }

    pub fn headroom(&self) -> usize {
        self.headroom
    }

    fn advise(
        &self,
        ptr: sys::CUdeviceptr,
        bytes: usize,
        device: sys::CUdevice,
    ) -> Result<(), MemoryError> {
        if bytes == 0 {
            return Ok(());
        }
        unsafe {
            sys::cuMemAdvise(
                ptr,
                bytes,
                sys::CUmem_advise_enum_CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                device,
            )
            .result()?;
        }
        Ok(())
    }
}

unsafe impl DeviceAllocator for SamHeadroomMemoryResource {
    fn context(&self) -> &Arc<CudaContext> {
        self.system.context()
    }

    fn properties(&self) -> MemoryProperties {
        MemoryProperties::MANAGED
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        let block = self.system.allocate(layout, stream)?;
        if block.size() == 0 {
            return Ok(block);
        }
        let advice = (|| {
            let mut free = MaybeUninit::<usize>::uninit();
            let mut total = MaybeUninit::<usize>::uninit();
            unsafe { sys::cuMemGetInfo_v2(free.as_mut_ptr(), total.as_mut_ptr()).result()? };
            let allocatable = unsafe { free.assume_init() }.saturating_sub(self.headroom);
            let gpu_bytes = allocatable.min(block.size()) / DEVICE_ALLOCATION_ALIGNMENT
                * DEVICE_ALLOCATION_ALIGNMENT;
            let host_bytes = block.size() - gpu_bytes;
            self.advise(block.cu_deviceptr(), gpu_bytes, self.context().cu_device())?;
            // CUDA's driver API uses -1 for CU_DEVICE_CPU in pre-CUDA-13
            // preferred-location advice.
            self.advise(
                block.cu_deviceptr() + gpu_bytes as sys::CUdeviceptr,
                host_bytes,
                -1,
            )
        })();
        if let Err(error) = advice {
            let _ = unsafe { self.system.deallocate(block, stream) };
            return Err(error);
        }
        Ok(block)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        unsafe { self.system.deallocate(block, stream) }
    }
}

pub(crate) fn validate_layout(layout: Layout) -> Result<(), MemoryError> {
    if layout.align() > DEVICE_ALLOCATION_ALIGNMENT {
        return Err(MemoryError::UnsupportedAlignment {
            requested: layout.align(),
            supported: DEVICE_ALLOCATION_ALIGNMENT,
        });
    }
    Ok(())
}

pub(crate) fn validate_stream(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
) -> Result<(), MemoryError> {
    // CudaContext::new retains the device's primary context. Separate Arc
    // wrappers can therefore denote the same CUDA context and must interoperate.
    if context.as_ref() != stream.context().as_ref() {
        return Err(MemoryError::ContextMismatch);
    }
    context.bind_to_thread()?;
    Ok(())
}

/// Direct CUDA Unified Memory resource.
///
/// Allocations are accessible from the host and CUDA devices under CUDA's
/// Unified Memory coherence rules. Allocation and deallocation are synchronous,
/// matching RMM's `managed_memory_resource` contract.
#[derive(Clone, Debug)]
pub struct ManagedMemoryResource {
    context: Arc<CudaContext>,
}

impl ManagedMemoryResource {
    pub fn new(context: Arc<CudaContext>) -> Self {
        Self { context }
    }
}

unsafe impl DeviceAllocator for ManagedMemoryResource {
    fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    fn properties(&self) -> MemoryProperties {
        MemoryProperties::MANAGED
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        validate_layout(layout)?;
        validate_stream(&self.context, stream)?;
        if layout.size() == 0 {
            return Ok(unsafe { DeviceBlock::from_raw_parts(0, 0, layout.align()) });
        }
        let mut ptr = MaybeUninit::uninit();
        unsafe {
            sys::cuMemAllocManaged(
                ptr.as_mut_ptr(),
                layout.size(),
                sys::CUmemAttach_flags_enum_CU_MEM_ATTACH_GLOBAL,
            )
            .result()?;
            Ok(DeviceBlock::from_raw_parts(
                ptr.assume_init(),
                layout.size(),
                layout.align(),
            ))
        }
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        validate_stream(&self.context, stream)?;
        if block.ptr != 0 {
            unsafe { free_sync(block.ptr)? };
        }
        Ok(())
    }
}

/// Stream-ordered device allocation policy.
///
/// Unlike Rust's host [`std::alloc::Allocator`], every operation receives a
/// CUDA stream. Like that trait, this trait is designed to be implemented by
/// concrete allocator values, references, and smart pointers. Containers are
/// generic over their allocator handle; dynamic dispatch is not required.
///
/// # Safety
///
/// Implementations must return a unique allocation satisfying `layout`, and
/// must accept exactly that block once in `deallocate`. An allocation must
/// remain usable until its ordered deallocation executes. Moving an allocator
/// handle, or cloning a handle that supports cloning, must not invalidate its
/// outstanding allocations. As with [`std::alloc::Allocator`], an allocation's
/// validity is bounded by the lifetime of the allocator handle (and any
/// allocator-specific shared state) retained by its owning container.
pub unsafe trait DeviceAllocator {
    fn context(&self) -> &Arc<CudaContext>;

    fn properties(&self) -> MemoryProperties {
        MemoryProperties::DEVICE
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError>;

    /// Return a block previously produced by this resource.
    ///
    /// # Safety
    ///
    /// `block` must be live, owned by the caller, allocated by `self`, and no
    /// use may be ordered after this deallocation on any stream. Ownership is
    /// consumed regardless of whether this method returns `Ok` or `Err`; an
    /// implementation that cannot complete deallocation may therefore leak the
    /// allocation, but it must never fabricate a replacement ownership token.
    unsafe fn deallocate(&self, block: DeviceBlock, stream: &CudaStream)
    -> Result<(), MemoryError>;

    /// Borrow this allocator as a container allocator handle.
    fn by_ref(&self) -> &Self
    where
        Self: Sized,
    {
        self
    }
}

// Match `std::alloc::Allocator`: ownership and dispatch are properties of the
// handle chosen by the caller, not requirements imposed by the allocator
// trait. A borrowed handle carries the allocator lifetime in its type, while
// Rc/Arc are explicit opt-ins to shared ownership.
unsafe impl<A: DeviceAllocator + ?Sized> DeviceAllocator for &A {
    fn context(&self) -> &Arc<CudaContext> {
        (**self).context()
    }

    fn properties(&self) -> MemoryProperties {
        (**self).properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        (**self).allocate(layout, stream)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        unsafe { (**self).deallocate(block, stream) }
    }
}

unsafe impl<A: DeviceAllocator + ?Sized> DeviceAllocator for &mut A {
    fn context(&self) -> &Arc<CudaContext> {
        (**self).context()
    }

    fn properties(&self) -> MemoryProperties {
        (**self).properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        (**self).allocate(layout, stream)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        unsafe { (**self).deallocate(block, stream) }
    }
}

unsafe impl<A: DeviceAllocator + ?Sized> DeviceAllocator for Box<A> {
    fn context(&self) -> &Arc<CudaContext> {
        (**self).context()
    }

    fn properties(&self) -> MemoryProperties {
        (**self).properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        (**self).allocate(layout, stream)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        unsafe { (**self).deallocate(block, stream) }
    }
}

unsafe impl<A: DeviceAllocator + ?Sized> DeviceAllocator for Rc<A> {
    fn context(&self) -> &Arc<CudaContext> {
        (**self).context()
    }

    fn properties(&self) -> MemoryProperties {
        (**self).properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        (**self).allocate(layout, stream)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        unsafe { (**self).deallocate(block, stream) }
    }
}

unsafe impl<A: DeviceAllocator + ?Sized> DeviceAllocator for Arc<A> {
    fn context(&self) -> &Arc<CudaContext> {
        (**self).context()
    }

    fn properties(&self) -> MemoryProperties {
        (**self).properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        (**self).allocate(layout, stream)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        unsafe { (**self).deallocate(block, stream) }
    }
}

/// Direct `cuMemAlloc`/`cuMemFree` resource.
///
/// This is useful as a conservative upstream and for devices without CUDA
/// stream-ordered allocation support. Allocation and free may synchronize.
#[derive(Clone, Debug)]
pub struct CudaMemoryResource {
    context: Arc<CudaContext>,
}

/// Direct CUDA allocation resource borrowing an existing context owner.
///
/// This is the allocation policy used by synchronous Thrust `par`: pinned
/// CCCL's `execute_on_stream` temporary-buffer path calls `cudaMalloc` and
/// `cudaFree`. The distinct `par_nosync` policy is the one that selects
/// stream-ordered allocation.
#[derive(Debug)]
pub struct BorrowedCudaMemoryResource<'context> {
    context: &'context Arc<CudaContext>,
    runtime_anchor: Option<DeviceBlock>,
}

/// Exact-size CUDA allocation cache restricted to one stream.
///
/// This is the Rust ownership equivalent of the pinned CCCL benchmark's
/// `caching_allocator_t`: deallocation returns an affine [`DeviceBlock`] to a
/// host-side exact-size cache, and the next request on the same stream may
/// reuse it without recording CUDA work. The single-stream lifetime and
/// `RefCell` make the intended non-concurrent use explicit; callers that need
/// cross-stream or cross-thread reuse should select a stream-ordered pool.
pub struct SingleStreamCachingMemoryResource<'stream> {
    stream: &'stream CudaStream,
    upstream: BorrowedCudaMemoryResource<'stream>,
    free_blocks: RefCell<Vec<DeviceBlock>>,
}

impl fmt::Debug for SingleStreamCachingMemoryResource<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SingleStreamCachingMemoryResource")
            .field("device", &self.stream.context().ordinal())
            .field("free_blocks", &self.free_blocks.borrow().len())
            .finish()
    }
}

impl<'stream> SingleStreamCachingMemoryResource<'stream> {
    pub fn new(stream: &'stream CudaStream) -> Result<Self, MemoryError> {
        Ok(Self {
            stream,
            upstream: BorrowedCudaMemoryResource::new(stream.context())?,
            free_blocks: RefCell::new(Vec::new()),
        })
    }

    pub fn free_blocks(&self) -> usize {
        self.free_blocks.borrow().len()
    }

    fn validate_stream(&self, stream: &CudaStream) -> Result<(), MemoryError> {
        validate_stream(self.upstream.context(), stream)?;
        if self.stream.cu_stream() != stream.cu_stream() {
            return Err(MemoryError::StreamMismatch);
        }
        Ok(())
    }
}

unsafe impl DeviceAllocator for SingleStreamCachingMemoryResource<'_> {
    fn context(&self) -> &Arc<CudaContext> {
        self.upstream.context()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        validate_layout(layout)?;
        self.validate_stream(stream)?;
        let mut free_blocks = self.free_blocks.borrow_mut();
        if let Some(index) = free_blocks
            .iter()
            .position(|block| block.size() == layout.size())
        {
            let DeviceBlock { ptr, size, .. } = free_blocks.swap_remove(index);
            return Ok(DeviceBlock {
                ptr,
                size,
                align: layout.align(),
            });
        }
        drop(free_blocks);
        self.upstream.allocate(layout, stream)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        self.validate_stream(stream)?;
        self.free_blocks.borrow_mut().push(block);
        Ok(())
    }
}

impl Drop for SingleStreamCachingMemoryResource<'_> {
    fn drop(&mut self) {
        self.context().record_err(self.stream.synchronize());
        for block in self.free_blocks.get_mut().drain(..) {
            let _ = unsafe { self.upstream.deallocate(block, self.stream) };
        }
    }
}

mod cuda_runtime {
    use std::ffi::c_void;

    #[link(name = "cudart_static", kind = "static")]
    unsafe extern "C" {
        fn cudaSetDevice(device: i32) -> i32;
        fn cudaMalloc(pointer: *mut *mut c_void, bytes: usize) -> i32;
        fn cudaFree(pointer: *mut c_void) -> i32;
    }

    pub fn set_device(device: usize) -> Result<(), super::MemoryError> {
        let device = i32::try_from(device).map_err(|_| super::MemoryError::SizeOverflow)?;
        let status = unsafe { cudaSetDevice(device) };
        if status == 0 {
            Ok(())
        } else {
            Err(super::MemoryError::CudaRuntime(status))
        }
    }

    pub fn malloc(bytes: usize) -> Result<cuda_core::sys::CUdeviceptr, super::MemoryError> {
        let mut pointer = std::ptr::null_mut();
        let status = unsafe { cudaMalloc(&mut pointer, bytes) };
        if status == 0 {
            Ok(pointer as cuda_core::sys::CUdeviceptr)
        } else {
            Err(super::MemoryError::CudaRuntime(status))
        }
    }

    pub unsafe fn free(pointer: cuda_core::sys::CUdeviceptr) -> Result<(), super::MemoryError> {
        let status = unsafe { cudaFree(pointer as *mut c_void) };
        if status == 0 {
            Ok(())
        } else {
            Err(super::MemoryError::CudaRuntime(status))
        }
    }
}

impl<'context> BorrowedCudaMemoryResource<'context> {
    pub fn new(context: &'context Arc<CudaContext>) -> Result<Self, MemoryError> {
        // cuda-core initializes the CUDA driver directly. Initialize the
        // runtime side of the same retained primary context before timed
        // allocations, just as a CCCL program's first cudaMalloc does.
        cuda_runtime::set_device(context.ordinal())?;
        let pointer = cuda_runtime::malloc(1)?;
        Ok(Self {
            context,
            runtime_anchor: Some(unsafe { DeviceBlock::from_raw_parts(pointer, 1, 1) }),
        })
    }
}

impl Drop for BorrowedCudaMemoryResource<'_> {
    fn drop(&mut self) {
        if let Some(block) = self.runtime_anchor.take() {
            let _ = unsafe { cuda_runtime::free(block.cu_deviceptr()) };
        }
    }
}

unsafe impl DeviceAllocator for BorrowedCudaMemoryResource<'_> {
    fn context(&self) -> &Arc<CudaContext> {
        self.context
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        validate_layout(layout)?;
        validate_stream(self.context, stream)?;
        if layout.size() == 0 {
            return Ok(unsafe { DeviceBlock::from_raw_parts(0, 0, layout.align()) });
        }
        let ptr = cuda_runtime::malloc(layout.size())?;
        Ok(unsafe { DeviceBlock::from_raw_parts(ptr, layout.size(), layout.align()) })
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        validate_stream(self.context, stream)?;
        if block.ptr != 0 {
            unsafe { cuda_runtime::free(block.ptr)? };
        }
        Ok(())
    }
}

impl CudaMemoryResource {
    pub fn new(context: Arc<CudaContext>) -> Self {
        Self { context }
    }
}

unsafe impl DeviceAllocator for CudaMemoryResource {
    fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        validate_layout(layout)?;
        validate_stream(&self.context, stream)?;
        if layout.size() == 0 {
            return Ok(unsafe { DeviceBlock::from_raw_parts(0, 0, layout.align()) });
        }
        let ptr = unsafe { malloc_sync(layout.size())? };
        Ok(unsafe { DeviceBlock::from_raw_parts(ptr, layout.size(), layout.align()) })
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        validate_stream(&self.context, stream)?;
        if block.ptr != 0 {
            unsafe { free_sync(block.ptr)? };
        }
        Ok(())
    }
}

/// Stream-ordered resource backed by the CUDA device's current memory pool.
///
/// `cuMemAllocAsync` already pools allocations in the driver. This should be
/// the default resource when an application does not need a private pool.
#[derive(Clone, Debug)]
pub struct CudaAsyncMemoryResource {
    context: Arc<CudaContext>,
}

/// A stream-ordered CUDA resource that borrows an existing context owner.
///
/// This is the default temporary-allocation handle used by a borrowed
/// execution policy. Unlike [`CudaAsyncMemoryResource`], constructing it does
/// not increment the context's `Arc` count: the stream already keeps that
/// context alive for the complete borrow.
#[derive(Clone, Copy, Debug)]
pub struct BorrowedCudaAsyncMemoryResource<'context> {
    context: &'context Arc<CudaContext>,
}

impl<'context> BorrowedCudaAsyncMemoryResource<'context> {
    pub fn new(context: &'context Arc<CudaContext>) -> Self {
        Self { context }
    }
}

unsafe impl DeviceAllocator for BorrowedCudaAsyncMemoryResource<'_> {
    fn context(&self) -> &Arc<CudaContext> {
        self.context
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        validate_layout(layout)?;
        validate_stream(self.context, stream)?;
        if layout.size() == 0 {
            return Ok(unsafe { DeviceBlock::from_raw_parts(0, 0, layout.align()) });
        }
        let ptr = unsafe { malloc_async(stream.cu_stream(), layout.size())? };
        Ok(unsafe { DeviceBlock::from_raw_parts(ptr, layout.size(), layout.align()) })
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        validate_stream(self.context, stream)?;
        if block.ptr != 0 {
            unsafe { free_async(block.ptr, stream.cu_stream())? };
        }
        Ok(())
    }
}

impl CudaAsyncMemoryResource {
    pub fn new(context: Arc<CudaContext>) -> Self {
        Self { context }
    }
}

unsafe impl DeviceAllocator for CudaAsyncMemoryResource {
    fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        validate_layout(layout)?;
        validate_stream(&self.context, stream)?;
        if layout.size() == 0 {
            return Ok(unsafe { DeviceBlock::from_raw_parts(0, 0, layout.align()) });
        }
        let ptr = unsafe { malloc_async(stream.cu_stream(), layout.size())? };
        Ok(unsafe { DeviceBlock::from_raw_parts(ptr, layout.size(), layout.align()) })
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        validate_stream(&self.context, stream)?;
        if block.ptr != 0 {
            unsafe { free_async(block.ptr, stream.cu_stream())? };
        }
        Ok(())
    }
}

/// Configuration for a private CUDA driver memory pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolConfig {
    /// Bytes retained by CUDA before a synchronization may release backing
    /// allocations to the operating system.
    pub release_threshold: u64,
    /// Permit reuse when an explicit CUDA event establishes dependency.
    pub follow_event_dependencies: bool,
    /// Permit reuse of frees already observed complete by the driver.
    pub opportunistic_reuse: bool,
    /// Permit CUDA to insert dependencies needed for safe cross-stream reuse.
    pub internal_dependencies: bool,
    /// Hard pool capacity. Zero lets CUDA choose a system-dependent maximum.
    pub maximum_size: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            release_threshold: u64::MAX,
            follow_event_dependencies: true,
            opportunistic_reuse: true,
            internal_dependencies: true,
            maximum_size: 0,
        }
    }
}

/// Current and high-watermark statistics reported by a CUDA memory pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    pub reserved_current: u64,
    pub reserved_high: u64,
    pub used_current: u64,
    pub used_high: u64,
}

/// Stream-ordered resource backed by a private CUDA driver memory pool.
pub struct CudaPoolMemoryResource {
    context: Arc<CudaContext>,
    pool: sys::CUmemoryPool,
}

impl fmt::Debug for CudaPoolMemoryResource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaPoolMemoryResource")
            .field("device", &self.context.ordinal())
            .field("pool", &self.pool)
            .finish()
    }
}

// CUDA memory-pool handles are context-owned driver handles and may be used
// from any thread after binding the owning context.
unsafe impl Send for CudaPoolMemoryResource {}
unsafe impl Sync for CudaPoolMemoryResource {}

impl CudaPoolMemoryResource {
    pub fn new(context: Arc<CudaContext>, config: PoolConfig) -> Result<Self, MemoryError> {
        context.bind_to_thread()?;
        let props = sys::CUmemPoolProps {
            allocType: sys::CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_PINNED,
            handleTypes: sys::CUmemAllocationHandleType_enum_CU_MEM_HANDLE_TYPE_NONE,
            location: sys::CUmemLocation {
                type_: sys::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE,
                id: context.ordinal() as i32,
            },
            win32SecurityAttributes: std::ptr::null_mut(),
            maxSize: config.maximum_size,
            usage: 0,
            reserved: [0; 54],
        };
        let mut pool = MaybeUninit::uninit();
        unsafe { sys::cuMemPoolCreate(pool.as_mut_ptr(), &props).result()? };
        let resource = Self {
            context,
            pool: unsafe { pool.assume_init() },
        };
        resource.configure(config)?;
        Ok(resource)
    }

    fn configure(&self, config: PoolConfig) -> Result<(), MemoryError> {
        self.set_u64_attribute(
            sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            config.release_threshold,
        )?;
        self.set_bool_attribute(
            sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES,
            config.follow_event_dependencies,
        )?;
        self.set_bool_attribute(
            sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
            config.opportunistic_reuse,
        )?;
        self.set_bool_attribute(
            sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
            config.internal_dependencies,
        )
    }

    fn set_u64_attribute(
        &self,
        attribute: sys::CUmemPool_attribute,
        mut value: u64,
    ) -> Result<(), MemoryError> {
        self.context.bind_to_thread()?;
        unsafe {
            sys::cuMemPoolSetAttribute(
                self.pool,
                attribute,
                (&mut value as *mut u64).cast::<c_void>(),
            )
            .result()?;
        }
        Ok(())
    }

    fn set_bool_attribute(
        &self,
        attribute: sys::CUmemPool_attribute,
        enabled: bool,
    ) -> Result<(), MemoryError> {
        let mut value: i32 = enabled.into();
        self.context.bind_to_thread()?;
        unsafe {
            sys::cuMemPoolSetAttribute(
                self.pool,
                attribute,
                (&mut value as *mut i32).cast::<c_void>(),
            )
            .result()?;
        }
        Ok(())
    }

    fn get_u64_attribute(&self, attribute: sys::CUmemPool_attribute) -> Result<u64, MemoryError> {
        let mut value = 0u64;
        self.context.bind_to_thread()?;
        unsafe {
            sys::cuMemPoolGetAttribute(
                self.pool,
                attribute,
                (&mut value as *mut u64).cast::<c_void>(),
            )
            .result()?;
        }
        Ok(value)
    }

    pub fn stats(&self) -> Result<PoolStats, MemoryError> {
        Ok(PoolStats {
            reserved_current: self.get_u64_attribute(
                sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            )?,
            reserved_high: self.get_u64_attribute(
                sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            )?,
            used_current: self.get_u64_attribute(
                sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            )?,
            used_high: self
                .get_u64_attribute(sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
        })
    }

    pub fn reset_high_watermarks(&self) -> Result<(), MemoryError> {
        self.set_u64_attribute(
            sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            0,
        )?;
        self.set_u64_attribute(
            sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_USED_MEM_HIGH,
            0,
        )
    }

    pub fn trim_to(&self, minimum_bytes_to_keep: usize) -> Result<(), MemoryError> {
        self.context.bind_to_thread()?;
        unsafe { sys::cuMemPoolTrimTo(self.pool, minimum_bytes_to_keep).result()? };
        Ok(())
    }

    pub fn cu_memory_pool(&self) -> sys::CUmemoryPool {
        self.pool
    }
}

impl Drop for CudaPoolMemoryResource {
    fn drop(&mut self) {
        self.context.record_err(self.context.bind_to_thread());
        self.context
            .record_err(unsafe { sys::cuMemPoolDestroy(self.pool).result() });
    }
}

unsafe impl DeviceAllocator for CudaPoolMemoryResource {
    fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        validate_layout(layout)?;
        validate_stream(&self.context, stream)?;
        if layout.size() == 0 {
            return Ok(unsafe { DeviceBlock::from_raw_parts(0, 0, layout.align()) });
        }
        let mut ptr = MaybeUninit::uninit();
        unsafe {
            sys::cuMemAllocFromPoolAsync(
                ptr.as_mut_ptr(),
                layout.size(),
                self.pool,
                stream.cu_stream(),
            )
            .result()?;
        }
        Ok(
            unsafe {
                DeviceBlock::from_raw_parts(ptr.assume_init(), layout.size(), layout.align())
            },
        )
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        validate_stream(&self.context, stream)?;
        if block.ptr != 0 {
            unsafe { free_async(block.ptr, stream.cu_stream())? };
        }
        Ok(())
    }
}

/// Host-side statistics for a [`TrackingMemoryResource`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceStats {
    pub current_bytes: usize,
    pub peak_bytes: usize,
    pub total_bytes: u64,
    pub outstanding_allocations: usize,
    pub peak_allocations: usize,
    pub total_allocations: u64,
}

/// RMM-style adaptor that instruments any upstream resource.
#[derive(Debug)]
pub struct TrackingMemoryResource<U> {
    upstream: U,
    current_bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
    total_bytes: AtomicU64,
    outstanding_allocations: AtomicUsize,
    peak_allocations: AtomicUsize,
    total_allocations: AtomicU64,
}

impl<U> TrackingMemoryResource<U> {
    pub fn new(upstream: U) -> Self {
        Self {
            upstream,
            current_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            total_bytes: AtomicU64::new(0),
            outstanding_allocations: AtomicUsize::new(0),
            peak_allocations: AtomicUsize::new(0),
            total_allocations: AtomicU64::new(0),
        }
    }

    pub fn stats(&self) -> ResourceStats {
        ResourceStats {
            current_bytes: self.current_bytes.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
            total_bytes: self.total_bytes.load(Ordering::Relaxed),
            outstanding_allocations: self.outstanding_allocations.load(Ordering::Relaxed),
            peak_allocations: self.peak_allocations.load(Ordering::Relaxed),
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
        }
    }

    pub fn upstream(&self) -> &U {
        &self.upstream
    }
}

unsafe impl<U: DeviceAllocator> DeviceAllocator for TrackingMemoryResource<U> {
    fn context(&self) -> &Arc<CudaContext> {
        self.upstream.context()
    }

    fn properties(&self) -> MemoryProperties {
        self.upstream.properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        let block = self.upstream.allocate(layout, stream)?;
        if block.size > 0 {
            let current = self.current_bytes.fetch_add(block.size, Ordering::Relaxed) + block.size;
            self.peak_bytes.fetch_max(current, Ordering::Relaxed);
            self.total_bytes
                .fetch_add(block.size as u64, Ordering::Relaxed);
            let outstanding = self.outstanding_allocations.fetch_add(1, Ordering::Relaxed) + 1;
            self.peak_allocations
                .fetch_max(outstanding, Ordering::Relaxed);
            self.total_allocations.fetch_add(1, Ordering::Relaxed);
        }
        Ok(block)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        let size = block.size;
        unsafe { self.upstream.deallocate(block, stream)? };
        if size > 0 {
            self.current_bytes.fetch_sub(size, Ordering::Relaxed);
            self.outstanding_allocations.fetch_sub(1, Ordering::Relaxed);
        }
        Ok(())
    }
}
