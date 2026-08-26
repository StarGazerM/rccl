//! RMM-style, stream-ordered device memory management for Rust CUDA code.
//!
//! The crate separates three concerns:
//!
//! - [`DeviceAllocator`] is the stream-ordered allocation policy boundary.
//! - resource implementations select synchronous, CUDA-default-pool, explicit
//!   pool, or instrumented allocation behavior.
//! - [`DeviceBuffer`] provides typed RAII ownership over an allocator-backed
//!   allocation.
//!
//! `std::alloc::Allocator` is intentionally not implemented. It describes
//! host-addressable memory and has no CUDA stream parameter, while device
//! allocation and deallocation are ordered operations in a stream DAG.

use cuda_core::{
    CudaContext, CudaStream, DeviceCopy, DriverError, IntoResult,
    memory::{
        free_async, free_host, free_sync, malloc_async, malloc_host, malloc_sync,
        memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async, memset_d8_async,
    },
    sys,
};
use std::{
    alloc::Layout,
    error::Error,
    ffi::c_void,
    fmt,
    marker::PhantomData,
    mem::MaybeUninit,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

mod adaptors;
mod fixed;
mod pool;
pub mod registry;

pub use adaptors::{
    AlignedMemoryResource, CallbackMemoryResource, FailureCallbackMemoryResource,
    LimitingMemoryResource, LoggingMemoryResource, PrefetchMemoryResource, StatisticsCounter,
    StatisticsMemoryResource, StatisticsSnapshot, ThreadSafeMemoryResource,
};
pub use fixed::{BinningMemoryResource, FixedSizeMemoryResource};
pub use pool::{PoolMemoryResource, SuballocatorStats};

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

    fn intersection(self, other: Self) -> Self {
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
    InvalidAlignment {
        requested: usize,
    },
    DeviceRangeOutOfBounds {
        offset: usize,
        len: usize,
        source_len: usize,
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
            Self::InvalidAlignment { requested } => write!(
                f,
                "allocation alignment {requested} must be a non-zero power of two"
            ),
            Self::DeviceRangeOutOfBounds {
                offset,
                len,
                source_len,
            } => write!(
                f,
                "device subrange [{offset}..{end}) is outside a {source_len}-element range",
                end = offset.saturating_add(*len),
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

fn validate_layout(layout: Layout) -> Result<(), MemoryError> {
    if layout.align() > DEVICE_ALLOCATION_ALIGNMENT {
        return Err(MemoryError::UnsupportedAlignment {
            requested: layout.align(),
            supported: DEVICE_ALLOCATION_ALIGNMENT,
        });
    }
    Ok(())
}

fn validate_stream(context: &Arc<CudaContext>, stream: &CudaStream) -> Result<(), MemoryError> {
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

/// RMM-compatible name for [`DeviceAllocator`].
pub use DeviceAllocator as DeviceMemoryResource;

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

/// Borrowed view of an existing private CUDA pool.
///
/// RMM exposes a non-owning raw-handle view. Rust represents that relationship
/// as a borrow, preventing the source pool from being destroyed while a view
/// or one of its resource-owned buffers is still live.
///
/// ```compile_fail
/// use cuda_core::CudaContext;
/// use rccl_memory::{CudaPoolMemoryResource, CudaPoolViewMemoryResource, DeviceBuffer, PoolConfig};
///
/// let context = CudaContext::new(0).unwrap();
/// let stream = context.default_stream();
/// let pool = CudaPoolMemoryResource::new(context, PoolConfig::default()).unwrap();
/// let view = CudaPoolViewMemoryResource::new(&pool);
/// let buffer = DeviceBuffer::<u8, _>::zeroed_in(view, &stream, 16).unwrap();
/// drop(pool); // cannot move the pool while `buffer` owns a borrowed view
/// drop(buffer);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CudaPoolViewMemoryResource<'pool> {
    pool: &'pool CudaPoolMemoryResource,
}

impl<'pool> CudaPoolViewMemoryResource<'pool> {
    pub const fn new(pool: &'pool CudaPoolMemoryResource) -> Self {
        Self { pool }
    }

    pub const fn pool(&self) -> &'pool CudaPoolMemoryResource {
        self.pool
    }

    pub fn cu_memory_pool(&self) -> sys::CUmemoryPool {
        self.pool.cu_memory_pool()
    }
}

unsafe impl DeviceAllocator for CudaPoolViewMemoryResource<'_> {
    fn context(&self) -> &Arc<CudaContext> {
        self.pool.context()
    }

    fn properties(&self) -> MemoryProperties {
        self.pool.properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        self.pool.allocate(layout, stream)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        unsafe { self.pool.deallocate(block, stream) }
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

/// Common read-only device allocation surface used by RCCL algorithms.
///
/// # Safety
///
/// The pointer must remain valid for `len()` contiguous `T` values for the
/// lifetime of `self`, and it must belong to `context()`.
pub unsafe trait DeviceSlice<T> {
    fn len(&self) -> usize;
    fn cu_deviceptr(&self) -> sys::CUdeviceptr;
    fn context(&self) -> &Arc<CudaContext>;

    /// Borrow the complete device range as a lifetime-bearing span.
    fn span(&self) -> DeviceSpan<'_, T>
    where
        Self: Sized,
    {
        DeviceSpan::new(self)
    }

    /// Borrow one checked device subrange.
    fn subspan(&self, offset: usize, len: usize) -> Result<DeviceSpan<'_, T>, MemoryError>
    where
        Self: Sized,
    {
        DeviceSpan::new(self).subspan(offset, len)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn num_bytes(&self) -> usize {
        self.len().saturating_mul(std::mem::size_of::<T>())
    }

    /// Check that this allocation may be used by work submitted to `stream`.
    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
        validate_stream(self.context(), stream)
    }
}

/// Writable extension of [`DeviceSlice`].
///
/// # Safety
///
/// Mutable access to the implementor must grant exclusive write access to all
/// `len()` device elements for the submitted operation.
pub unsafe trait DeviceSliceMut<T>: DeviceSlice<T> {
    /// Exclusively borrow the complete device range as a mutable span.
    fn span_mut(&mut self) -> DeviceSpanMut<'_, T>
    where
        Self: Sized,
    {
        DeviceSpanMut::new(self)
    }

    /// Exclusively borrow one checked device subrange.
    fn subspan_mut(
        &mut self,
        offset: usize,
        len: usize,
    ) -> Result<DeviceSpanMut<'_, T>, MemoryError>
    where
        Self: Sized,
    {
        DeviceSpanMut::new(self).into_subspan(offset, len)
    }
}

fn checked_device_subrange<T>(
    pointer: sys::CUdeviceptr,
    source_len: usize,
    offset: usize,
    len: usize,
) -> Result<sys::CUdeviceptr, MemoryError> {
    let end = offset.checked_add(len).ok_or(MemoryError::SizeOverflow)?;
    if end > source_len {
        return Err(MemoryError::DeviceRangeOutOfBounds {
            offset,
            len,
            source_len,
        });
    }
    let byte_offset = offset
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(MemoryError::SizeOverflow)?;
    let byte_offset =
        sys::CUdeviceptr::try_from(byte_offset).map_err(|_| MemoryError::SizeOverflow)?;
    pointer
        .checked_add(byte_offset)
        .ok_or(MemoryError::SizeOverflow)
}

/// A checked, read-only borrow of a contiguous device range.
///
/// Unlike a host `&[T]`, this value is not dereferenceable by the CPU. It
/// carries the source allocation borrow and delegates context/stream-order
/// validation to that source. CUB dispatch lowers it to CUDA pointer/length
/// arguments; cuda-oxide kernels may then receive an ordinary device slice.
#[derive(Clone, Copy)]
pub struct DeviceSpan<'a, T> {
    source: &'a dyn DeviceSlice<T>,
    pointer: sys::CUdeviceptr,
    len: usize,
    _borrow: PhantomData<&'a T>,
}

impl<T> fmt::Debug for DeviceSpan<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceSpan")
            .field("ptr", &self.pointer)
            .field("len", &self.len)
            .finish()
    }
}

impl<'a, T> DeviceSpan<'a, T> {
    pub fn new<B: DeviceSlice<T>>(source: &'a B) -> Self {
        Self {
            pointer: source.cu_deviceptr(),
            len: source.len(),
            source,
            _borrow: PhantomData,
        }
    }

    fn from_parts(source: &'a dyn DeviceSlice<T>, pointer: sys::CUdeviceptr, len: usize) -> Self {
        Self {
            source,
            pointer,
            len,
            _borrow: PhantomData,
        }
    }

    pub fn subspan(self, offset: usize, len: usize) -> Result<Self, MemoryError> {
        let pointer = checked_device_subrange::<T>(self.pointer, self.len, offset, len)?;
        Ok(Self::from_parts(self.source, pointer, len))
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.pointer
    }
}

// SAFETY: construction borrows a DeviceSlice source and only admits checked
// subranges within it. The source remains live and supplies context/ordering.
unsafe impl<T> DeviceSlice<T> for DeviceSpan<'_, T> {
    fn len(&self) -> usize {
        self.len
    }

    fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.pointer
    }

    fn context(&self) -> &Arc<CudaContext> {
        self.source.context()
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
        self.source.ensure_usable_on(stream)
    }
}

/// A checked, exclusive borrow of a contiguous device range.
///
/// This value is intentionally neither `Copy` nor `Clone`. Consuming
/// [`split_at`](Self::split_at) is the safe way to produce two simultaneous
/// mutable spans.
///
/// ```compile_fail
/// use rccl_memory::DeviceSliceMut;
///
/// fn overlapping<T, B: DeviceSliceMut<T>>(buffer: &mut B) {
///     let first = buffer.span_mut();
///     let second = buffer.span_mut();
///     drop((first, second));
/// }
/// ```
pub struct DeviceSpanMut<'a, T> {
    source: &'a dyn DeviceSlice<T>,
    pointer: sys::CUdeviceptr,
    len: usize,
    _borrow: PhantomData<&'a mut T>,
}

impl<T> fmt::Debug for DeviceSpanMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceSpanMut")
            .field("ptr", &self.pointer)
            .field("len", &self.len)
            .finish()
    }
}

impl<'a, T> DeviceSpanMut<'a, T> {
    pub fn new<B: DeviceSliceMut<T>>(source: &'a mut B) -> Self {
        let pointer = source.cu_deviceptr();
        let len = source.len();
        let source: &'a dyn DeviceSlice<T> = source;
        Self::from_parts(source, pointer, len)
    }

    fn from_parts(source: &'a dyn DeviceSlice<T>, pointer: sys::CUdeviceptr, len: usize) -> Self {
        Self {
            source,
            pointer,
            len,
            _borrow: PhantomData,
        }
    }

    pub fn into_subspan(self, offset: usize, len: usize) -> Result<Self, MemoryError> {
        let pointer = checked_device_subrange::<T>(self.pointer, self.len, offset, len)?;
        Ok(Self::from_parts(self.source, pointer, len))
    }

    pub fn split_at(self, mid: usize) -> Result<(Self, Self), MemoryError> {
        let right_pointer = checked_device_subrange::<T>(self.pointer, self.len, mid, 0)?;
        let right_len = self.len - mid;
        Ok((
            Self::from_parts(self.source, self.pointer, mid),
            Self::from_parts(self.source, right_pointer, right_len),
        ))
    }

    pub fn as_read_only(&self) -> DeviceSpan<'_, T> {
        DeviceSpan::from_parts(self.source, self.pointer, self.len)
    }

    pub fn into_read_only(self) -> DeviceSpan<'a, T> {
        DeviceSpan::from_parts(self.source, self.pointer, self.len)
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.pointer
    }
}

// SAFETY: as DeviceSpan, with the additional exclusive source borrow retained
// by DeviceSpanMut for its complete lifetime.
unsafe impl<T> DeviceSlice<T> for DeviceSpanMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }

    fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.pointer
    }

    fn context(&self) -> &Arc<CudaContext> {
        self.source.context()
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
        self.source.ensure_usable_on(stream)
    }
}

// SAFETY: safe constructors require an exclusive source borrow, and split_at
// consumes that borrow while returning only disjoint subranges.
unsafe impl<T> DeviceSliceMut<T> for DeviceSpanMut<'_, T> {}

unsafe impl<T> DeviceSlice<T> for cuda_core::DeviceBuffer<T> {
    fn len(&self) -> usize {
        self.len()
    }

    fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.cu_deviceptr()
    }

    fn context(&self) -> &Arc<CudaContext> {
        self.context()
    }
}

unsafe impl<T> DeviceSliceMut<T> for cuda_core::DeviceBuffer<T> {}

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
    fn layout_for(len: usize) -> Result<Layout, MemoryError> {
        if std::mem::size_of::<T>() == 0 {
            return Err(MemoryError::ZeroSizedType);
        }
        let size = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(MemoryError::SizeOverflow)?;
        Ok(Layout::from_size_align(size, std::mem::align_of::<T>())?)
    }

    /// Allocate storage without initializing its elements.
    ///
    /// # Safety
    ///
    /// The caller must initialize every element before any operation reads it
    /// or copies it to the host as `T`.
    pub unsafe fn uninitialized_in(
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
        let buffer = unsafe { Self::uninitialized_in(allocator, stream, len)? };
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
        let buffer = unsafe { Self::uninitialized_in(allocator, stream, values.len())? };
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

    /// Change the logical length without initializing newly exposed elements.
    ///
    /// # Safety
    ///
    /// If `new_len` is greater than the old length, the caller must initialize
    /// the new elements before an operation reads them or copies them to host
    /// memory as `T`.
    pub unsafe fn resize_uninitialized(&mut self, new_len: usize) -> Result<(), MemoryError> {
        self.reserve(new_len)?;
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
        let copy =
            unsafe { Self::uninitialized_in(self.allocator.clone(), self.stream, self.len)? };
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

unsafe impl<T: DeviceCopy, A: DeviceAllocator> DeviceSlice<T> for DeviceBuffer<'_, T, A> {
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

unsafe impl<T: DeviceCopy, A: DeviceAllocator> DeviceSliceMut<T> for DeviceBuffer<'_, T, A> {}

/// RMM-compatible name for an uninitialized typed device vector.
pub type DeviceUVector<'stream, T, A> = DeviceBuffer<'stream, T, A>;

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

    /// Allocate one value without initializing it.
    ///
    /// # Safety
    ///
    /// The caller must initialize the value before reading it.
    pub unsafe fn uninitialized_in(
        allocator: A,
        stream: &'stream CudaStream,
    ) -> Result<Self, MemoryError> {
        Ok(Self {
            storage: unsafe { DeviceBuffer::uninitialized_in(allocator, stream, 1)? },
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

unsafe impl<T: DeviceCopy, A: DeviceAllocator> DeviceSlice<T> for DeviceScalar<'_, T, A> {
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

unsafe impl<T: DeviceCopy, A: DeviceAllocator> DeviceSliceMut<T> for DeviceScalar<'_, T, A> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    struct FakeDeviceRange {
        pointer: sys::CUdeviceptr,
        len: usize,
    }

    // SAFETY: tests use only pointer/length arithmetic and never dereference
    // the synthetic device address or ask for its CUDA context.
    unsafe impl DeviceSlice<u32> for FakeDeviceRange {
        fn len(&self) -> usize {
            self.len
        }

        fn cu_deviceptr(&self) -> sys::CUdeviceptr {
            self.pointer
        }

        fn context(&self) -> &Arc<CudaContext> {
            panic!("synthetic span tests do not have a CUDA context")
        }
    }

    // SAFETY: each test creates at most one exclusive borrow of the synthetic
    // range, then uses split_at to produce only non-overlapping spans.
    unsafe impl DeviceSliceMut<u32> for FakeDeviceRange {}

    #[test]
    fn typed_layout_uses_element_size_and_alignment() {
        let layout = DeviceBuffer::<u64, CudaMemoryResource>::layout_for(17).unwrap();
        assert_eq!(layout.size(), 17 * 8);
        assert_eq!(layout.align(), 8);
    }

    #[test]
    fn typed_layout_detects_size_overflow() {
        assert!(matches!(
            DeviceBuffer::<u64, CudaMemoryResource>::layout_for(usize::MAX),
            Err(MemoryError::SizeOverflow)
        ));
    }

    #[test]
    fn immutable_subspan_offsets_pointer_and_checks_nested_ranges() {
        let source = FakeDeviceRange {
            pointer: 0x1000,
            len: 16,
        };
        let middle = source.subspan(3, 8).expect("valid subspan");
        assert_eq!(middle.cu_deviceptr(), 0x1000 + 3 * size_of::<u32>() as u64);
        assert_eq!(middle.len(), 8);

        let nested = middle.subspan(2, 3).expect("valid nested subspan");
        assert_eq!(nested.cu_deviceptr(), 0x1000 + 5 * size_of::<u32>() as u64);
        assert_eq!(nested.len(), 3);
    }

    #[test]
    fn subspan_rejects_bounds_and_address_overflow() {
        let source = FakeDeviceRange {
            pointer: 0x1000,
            len: 8,
        };
        let empty_end = source.subspan(8, 0).expect("empty end subspan");
        assert_eq!(
            empty_end.cu_deviceptr(),
            0x1000 + 8 * size_of::<u32>() as u64
        );
        assert!(empty_end.is_empty());
        assert!(matches!(
            source.subspan(7, 2),
            Err(MemoryError::DeviceRangeOutOfBounds {
                offset: 7,
                len: 2,
                source_len: 8,
            })
        ));
        assert!(matches!(
            source.subspan(usize::MAX, 2),
            Err(MemoryError::SizeOverflow)
        ));

        let address_edge = FakeDeviceRange {
            pointer: sys::CUdeviceptr::MAX - 1,
            len: 2,
        };
        assert!(matches!(
            address_edge.subspan(1, 1),
            Err(MemoryError::SizeOverflow)
        ));
    }

    #[test]
    fn mutable_span_split_is_disjoint_and_preserves_element_offsets() {
        let mut source = FakeDeviceRange {
            pointer: 0x2000,
            len: 9,
        };
        let span = source.span_mut();
        let (left, right) = span.split_at(4).expect("valid split point");
        assert_eq!(left.cu_deviceptr(), 0x2000);
        assert_eq!(left.len(), 4);
        assert_eq!(right.cu_deviceptr(), 0x2000 + 4 * size_of::<u32>() as u64);
        assert_eq!(right.len(), 5);
    }

    #[test]
    fn mutable_span_split_rejects_out_of_bounds_midpoint() {
        let mut source = FakeDeviceRange {
            pointer: 0x2000,
            len: 9,
        };
        assert!(matches!(
            source.span_mut().split_at(10),
            Err(MemoryError::DeviceRangeOutOfBounds {
                offset: 10,
                len: 0,
                source_len: 9,
            })
        ));
    }

    #[test]
    fn default_pool_policy_retains_and_reuses_memory() {
        let config = PoolConfig::default();
        assert_eq!(config.release_threshold, u64::MAX);
        assert!(config.follow_event_dependencies);
        assert!(config.opportunistic_reuse);
        assert!(config.internal_dependencies);
    }
}
