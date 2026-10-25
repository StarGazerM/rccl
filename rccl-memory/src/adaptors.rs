use crate::{DeviceAllocator, DeviceBlock, MemoryError, MemoryProperties};
use cuda_core::{CudaContext, CudaStream, DriverError, IntoResult};
use std::{
    alloc::Layout,
    collections::HashMap,
    fmt,
    fs::File,
    io::{self, Write},
    mem::MaybeUninit,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

/// Resource implemented by captured Rust allocation and deallocation closures.
///
/// This is RMM's callback resource expressed without a raw `void*` argument:
/// Rust closures capture their typed state directly.
pub struct CallbackMemoryResource<A, D> {
    context: Arc<CudaContext>,
    properties: MemoryProperties,
    allocate_callback: A,
    deallocate_callback: D,
}

impl<A, D> fmt::Debug for CallbackMemoryResource<A, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallbackMemoryResource")
            .field("context", &self.context)
            .field("properties", &self.properties)
            .finish_non_exhaustive()
    }
}

impl<A, D> CallbackMemoryResource<A, D>
where
    A: Fn(Layout, &CudaStream) -> Result<DeviceBlock, MemoryError>,
    D: Fn(DeviceBlock, &CudaStream) -> Result<(), MemoryError>,
{
    /// Constructs a device-accessible callback resource.
    ///
    /// # Safety
    ///
    /// The callbacks must satisfy all invariants of [`DeviceAllocator`]:
    /// allocations must be unique, live, correctly sized and aligned, usable
    /// on `stream`, and accepted exactly once by `deallocate_callback`.
    pub unsafe fn new(
        context: Arc<CudaContext>,
        allocate_callback: A,
        deallocate_callback: D,
    ) -> Self {
        unsafe {
            Self::with_properties(
                context,
                MemoryProperties::DEVICE,
                allocate_callback,
                deallocate_callback,
            )
        }
    }

    /// Constructs a callback resource with explicit address-space properties.
    ///
    /// # Safety
    ///
    /// In addition to [`Self::new`]'s requirements, `properties` must describe
    /// every allocation returned by the allocation callback.
    pub unsafe fn with_properties(
        context: Arc<CudaContext>,
        properties: MemoryProperties,
        allocate_callback: A,
        deallocate_callback: D,
    ) -> Self {
        Self {
            context,
            properties,
            allocate_callback,
            deallocate_callback,
        }
    }
}

unsafe impl<A, D> DeviceAllocator for CallbackMemoryResource<A, D>
where
    A: Fn(Layout, &CudaStream) -> Result<DeviceBlock, MemoryError>,
    D: Fn(DeviceBlock, &CudaStream) -> Result<(), MemoryError>,
{
    fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    fn properties(&self) -> MemoryProperties {
        self.properties
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        crate::validate_layout(layout)?;
        crate::validate_stream(&self.context, stream)?;
        (self.allocate_callback)(layout, stream)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        crate::validate_stream(&self.context, stream)?;
        (self.deallocate_callback)(block, stream)
    }
}

/// Serializes calls into an upstream resource.
///
/// All built-in resources already implement `Send + Sync`; this adaptor exists
/// for behavioral parity when wrapping an FFI resource whose internal state
/// requires one-call-at-a-time access.
#[derive(Debug)]
pub struct ThreadSafeMemoryResource<U> {
    context: Arc<CudaContext>,
    properties: MemoryProperties,
    upstream: Mutex<U>,
}

impl<U: DeviceAllocator> ThreadSafeMemoryResource<U> {
    pub fn new(upstream: U) -> Self {
        Self {
            context: upstream.context().clone(),
            properties: upstream.properties(),
            upstream: Mutex::new(upstream),
        }
    }

    pub fn upstream_mut(&mut self) -> &mut U {
        self.upstream
            .get_mut()
            .expect("thread-safe resource poisoned")
    }

    pub fn into_upstream(self) -> U {
        self.upstream
            .into_inner()
            .expect("thread-safe resource poisoned")
    }
}

unsafe impl<U: DeviceAllocator + Send> DeviceAllocator for ThreadSafeMemoryResource<U> {
    fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    fn properties(&self) -> MemoryProperties {
        self.properties
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        self.upstream
            .lock()
            .expect("thread-safe resource poisoned")
            .allocate(layout, stream)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        unsafe {
            self.upstream
                .lock()
                .expect("thread-safe resource poisoned")
                .deallocate(block, stream)
        }
    }
}

/// Prefetches managed allocations to the resource's device on the allocation
/// stream. Device-only allocations and devices without concurrent managed
/// access are intentionally treated as no-ops, matching RMM.
#[derive(Debug)]
pub struct PrefetchMemoryResource<U> {
    upstream: U,
    concurrent_managed_access: bool,
}

impl<U: DeviceAllocator> PrefetchMemoryResource<U> {
    pub fn new(upstream: U) -> Result<Self, MemoryError> {
        upstream.context().bind_to_thread()?;
        let mut supported = MaybeUninit::<i32>::uninit();
        unsafe {
            cuda_core::sys::cuDeviceGetAttribute(
                supported.as_mut_ptr(),
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
                upstream.context().cu_device(),
            )
            .result()?;
            Ok(Self {
                upstream,
                concurrent_managed_access: supported.assume_init() != 0,
            })
        }
    }

    pub fn upstream(&self) -> &U {
        &self.upstream
    }

    pub fn is_supported(&self) -> bool {
        self.concurrent_managed_access
    }

    fn prefetch(&self, block: &DeviceBlock, stream: &CudaStream) -> Result<(), MemoryError> {
        if block.size() == 0
            || !self.concurrent_managed_access
            || !self.upstream.properties().prefetchable
        {
            return Ok(());
        }
        let result = unsafe {
            cuda_core::sys::cuMemPrefetchAsync_v2(
                block.cu_deviceptr(),
                block.size(),
                cuda_core::sys::CUmemLocation {
                    type_: cuda_core::sys::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE,
                    id: self.context().cu_device(),
                },
                0,
                stream.cu_stream(),
            )
        };
        if result == cuda_core::sys::cudaError_enum_CUDA_SUCCESS
            || result == cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE
        {
            Ok(())
        } else {
            Err(MemoryError::Driver(DriverError(result)))
        }
    }
}

unsafe impl<U: DeviceAllocator> DeviceAllocator for PrefetchMemoryResource<U> {
    fn context(&self) -> &Arc<CudaContext> {
        self.upstream.context()
    }

    fn properties(&self) -> MemoryProperties {
        self.upstream.properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        let block = self.upstream.allocate(layout, stream)?;
        if let Err(error) = self.prefetch(&block, stream) {
            let _ = unsafe { self.upstream.deallocate(block, stream) };
            return Err(error);
        }
        Ok(block)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        unsafe { self.upstream.deallocate(block, stream) }
    }
}

/// CSV allocation logger compatible with RMM's action vocabulary and header.
///
/// The sink is a normal Rust [`Write`] value. Logging errors are remembered and
/// returned by [`Self::flush`] but never turn a successful upstream allocation
/// into a leaked allocation.
pub struct LoggingMemoryResource<U, W> {
    upstream: U,
    writer: Mutex<W>,
    auto_flush: bool,
    log_error: Mutex<Option<io::Error>>,
}

impl<U, W> fmt::Debug for LoggingMemoryResource<U, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoggingMemoryResource")
            .field("upstream_type", &std::any::type_name::<U>())
            .field("writer_type", &std::any::type_name::<W>())
            .field("auto_flush", &self.auto_flush)
            .field(
                "has_log_error",
                &self
                    .log_error
                    .lock()
                    .expect("logging resource error state poisoned")
                    .is_some(),
            )
            .finish_non_exhaustive()
    }
}

impl<U: DeviceAllocator, W: Write> LoggingMemoryResource<U, W> {
    pub const HEADER: &'static str = "Thread,Time,Action,Pointer,Size,Stream";

    pub fn new(upstream: U, writer: W, auto_flush: bool) -> io::Result<Self> {
        let resource = Self {
            upstream,
            writer: Mutex::new(writer),
            auto_flush,
            log_error: Mutex::new(None),
        };
        resource.write_header()?;
        Ok(resource)
    }

    pub fn upstream(&self) -> &U {
        &self.upstream
    }

    pub fn flush(&self) -> io::Result<()> {
        if let Some(error) = self
            .log_error
            .lock()
            .expect("logging resource error state poisoned")
            .take()
        {
            return Err(error);
        }
        self.writer
            .lock()
            .expect("logging resource writer poisoned")
            .flush()
    }

    fn write_header(&self) -> io::Result<()> {
        let mut writer = self
            .writer
            .lock()
            .expect("logging resource writer poisoned");
        writeln!(writer, "{}", Self::HEADER)?;
        if self.auto_flush {
            writer.flush()?;
        }
        Ok(())
    }

    fn log(&self, action: &str, block: &DeviceBlock, stream: &CudaStream) {
        let micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros();
        let result = {
            let mut writer = self
                .writer
                .lock()
                .expect("logging resource writer poisoned");
            writeln!(
                writer,
                "{:?},{micros},{action},0x{:x},{},0x{:x}",
                std::thread::current().id(),
                block.cu_deviceptr(),
                block.size(),
                stream.cu_stream() as usize,
            )
            .and_then(|()| {
                if self.auto_flush {
                    writer.flush()
                } else {
                    Ok(())
                }
            })
        };
        if let Err(error) = result {
            let mut stored = self
                .log_error
                .lock()
                .expect("logging resource error state poisoned");
            if stored.is_none() {
                *stored = Some(error);
            }
        }
    }
}

impl<U: DeviceAllocator> LoggingMemoryResource<U, File> {
    pub fn to_file(upstream: U, path: impl AsRef<Path>, auto_flush: bool) -> io::Result<Self> {
        Self::new(upstream, File::create(path)?, auto_flush)
    }
}

unsafe impl<U: DeviceAllocator, W: Write> DeviceAllocator for LoggingMemoryResource<U, W> {
    fn context(&self) -> &Arc<CudaContext> {
        self.upstream.context()
    }

    fn properties(&self) -> MemoryProperties {
        self.upstream.properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        match self.upstream.allocate(layout, stream) {
            Ok(block) => {
                self.log("allocate", &block, stream);
                Ok(block)
            }
            Err(error) => {
                let failed =
                    unsafe { DeviceBlock::from_raw_parts(0, layout.size(), layout.align()) };
                self.log("allocate failure", &failed, stream);
                Err(error)
            }
        }
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        self.log("free", &block, stream);
        unsafe { self.upstream.deallocate(block, stream) }
    }
}

#[derive(Debug)]
struct AlignedAllocation {
    upstream: DeviceBlock,
    returned_size: usize,
}

/// Over-allocating adaptor for device alignments greater than CUDA's ordinary
/// 256-byte guarantee.
///
/// Rust `Layout` alignment is honored even when it is stronger than the
/// adaptor's configured alignment. Requests below `alignment_threshold` pass
/// through when CUDA's natural alignment is sufficient.
pub struct AlignedMemoryResource<U> {
    upstream: U,
    alignment: usize,
    alignment_threshold: usize,
    adjusted: Mutex<HashMap<cuda_core::sys::CUdeviceptr, AlignedAllocation>>,
}

impl<U> fmt::Debug for AlignedMemoryResource<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedMemoryResource")
            .field("alignment", &self.alignment)
            .field("alignment_threshold", &self.alignment_threshold)
            .field(
                "adjusted_allocations",
                &self
                    .adjusted
                    .lock()
                    .expect("aligned resource poisoned")
                    .len(),
            )
            .field("upstream_type", &std::any::type_name::<U>())
            .finish()
    }
}

impl<U: DeviceAllocator> AlignedMemoryResource<U> {
    pub fn new(
        upstream: U,
        alignment: usize,
        alignment_threshold: usize,
    ) -> Result<Self, MemoryError> {
        if !alignment.is_power_of_two() {
            return Err(MemoryError::InvalidAlignment {
                requested: alignment,
            });
        }
        Ok(Self {
            upstream,
            alignment: alignment.max(crate::DEVICE_ALLOCATION_ALIGNMENT),
            alignment_threshold,
            adjusted: Mutex::new(HashMap::new()),
        })
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn alignment_threshold(&self) -> usize {
        self.alignment_threshold
    }

    pub fn upstream(&self) -> &U {
        &self.upstream
    }

    fn upstream_size(&self, bytes: usize, alignment: usize) -> Result<usize, MemoryError> {
        bytes
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
            .and_then(|aligned| aligned.checked_add(alignment - crate::DEVICE_ALLOCATION_ALIGNMENT))
            .ok_or(MemoryError::SizeOverflow)
    }
}

unsafe impl<U: DeviceAllocator> DeviceAllocator for AlignedMemoryResource<U> {
    fn context(&self) -> &Arc<CudaContext> {
        self.upstream.context()
    }

    fn properties(&self) -> MemoryProperties {
        self.upstream.properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        crate::validate_stream(self.context(), stream)?;
        let alignment = self.alignment.max(layout.align());
        if layout.size() == 0
            || alignment <= crate::DEVICE_ALLOCATION_ALIGNMENT
            || (layout.align() <= crate::DEVICE_ALLOCATION_ALIGNMENT
                && layout.size() < self.alignment_threshold)
        {
            return self.upstream.allocate(layout, stream);
        }

        let upstream_size = self.upstream_size(layout.size(), alignment)?;
        let upstream_layout = Layout::from_size_align(upstream_size, 1)?;
        let upstream_block = self.upstream.allocate(upstream_layout, stream)?;
        let Some(aligned_ptr) = upstream_block
            .cu_deviceptr()
            .checked_add((alignment - 1) as cuda_core::sys::CUdeviceptr)
            .map(|value| value & !((alignment - 1) as cuda_core::sys::CUdeviceptr))
        else {
            let _ = unsafe { self.upstream.deallocate(upstream_block, stream) };
            return Err(MemoryError::SizeOverflow);
        };
        let mut adjusted = self.adjusted.lock().expect("aligned resource poisoned");
        if adjusted.contains_key(&aligned_ptr) {
            drop(adjusted);
            let _ = unsafe { self.upstream.deallocate(upstream_block, stream) };
            return Err(MemoryError::UnknownAllocation { ptr: aligned_ptr });
        }
        let allocation = AlignedAllocation {
            upstream: upstream_block,
            returned_size: layout.size(),
        };
        adjusted.insert(aligned_ptr, allocation);
        Ok(unsafe { DeviceBlock::from_raw_parts(aligned_ptr, layout.size(), alignment) })
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        let allocation = self
            .adjusted
            .lock()
            .expect("aligned resource poisoned")
            .remove(&block.cu_deviceptr());
        if let Some(allocation) = allocation {
            if allocation.returned_size != block.size() {
                return Err(MemoryError::AllocationSizeMismatch {
                    expected: allocation.returned_size,
                    actual: block.size(),
                });
            }
            unsafe { self.upstream.deallocate(allocation.upstream, stream)? };
            return Ok(());
        }
        unsafe { self.upstream.deallocate(block, stream) }
    }
}

/// Caps the total live bytes delegated to an upstream resource.
///
/// The reservation is made atomically before calling the upstream allocator
/// and rolled back if that allocation fails. This makes the cap reliable when
/// the resource is shared by multiple host threads.
pub struct LimitingMemoryResource<U> {
    upstream: U,
    limit: usize,
    allocated: AtomicUsize,
}

impl<U: DeviceAllocator> fmt::Debug for LimitingMemoryResource<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LimitingMemoryResource")
            .field("limit", &self.limit)
            .field("allocated", &self.allocated_bytes())
            .field("upstream_type", &std::any::type_name::<U>())
            .finish()
    }
}

impl<U: DeviceAllocator> LimitingMemoryResource<U> {
    pub fn new(upstream: U, limit: usize) -> Self {
        Self {
            upstream,
            limit,
            allocated: AtomicUsize::new(0),
        }
    }

    pub fn allocation_limit(&self) -> usize {
        self.limit
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated.load(Ordering::Acquire)
    }

    pub fn upstream(&self) -> &U {
        &self.upstream
    }

    fn reserve(&self, requested: usize) -> Result<(), MemoryError> {
        let mut current = self.allocated.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(requested) else {
                return Err(MemoryError::AllocationLimitExceeded {
                    requested,
                    allocated: current,
                    limit: self.limit,
                });
            };
            if next > self.limit {
                return Err(MemoryError::AllocationLimitExceeded {
                    requested,
                    allocated: current,
                    limit: self.limit,
                });
            }
            match self.allocated.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => current = observed,
            }
        }
    }
}

unsafe impl<U: DeviceAllocator> DeviceAllocator for LimitingMemoryResource<U> {
    fn context(&self) -> &Arc<CudaContext> {
        self.upstream.context()
    }

    fn properties(&self) -> MemoryProperties {
        self.upstream.properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        self.reserve(layout.size())?;
        match self.upstream.allocate(layout, stream) {
            Ok(block) => Ok(block),
            Err(error) => {
                self.allocated.fetch_sub(layout.size(), Ordering::AcqRel);
                Err(error)
            }
        }
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        let size = block.size();
        unsafe { self.upstream.deallocate(block, stream)? };
        self.allocated.fetch_sub(size, Ordering::AcqRel);
        Ok(())
    }
}

/// Retries failed upstream allocations while a callback requests another try.
///
/// The callback receives the requested byte count and the concrete Rust error.
/// Returning `true` retries the allocation; returning `false` propagates the
/// error. The callback is expected to release or reclaim memory before retrying.
pub struct FailureCallbackMemoryResource<U, F> {
    upstream: U,
    callback: F,
}

impl<U, F> fmt::Debug for FailureCallbackMemoryResource<U, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FailureCallbackMemoryResource")
            .field("upstream_type", &std::any::type_name::<U>())
            .field("callback_type", &std::any::type_name::<F>())
            .finish_non_exhaustive()
    }
}

impl<U: DeviceAllocator, F> FailureCallbackMemoryResource<U, F>
where
    F: Fn(usize, &MemoryError) -> bool,
{
    pub fn new(upstream: U, callback: F) -> Self
    where
        F: Fn(usize, &MemoryError) -> bool,
    {
        Self { upstream, callback }
    }

    pub fn upstream(&self) -> &U {
        &self.upstream
    }
}

unsafe impl<U: DeviceAllocator, F> DeviceAllocator for FailureCallbackMemoryResource<U, F>
where
    F: Fn(usize, &MemoryError) -> bool,
{
    fn context(&self) -> &Arc<CudaContext> {
        self.upstream.context()
    }

    fn properties(&self) -> MemoryProperties {
        self.upstream.properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        loop {
            match self.upstream.allocate(layout, stream) {
                Ok(block) => return Ok(block),
                Err(error) if (self.callback)(layout.size(), &error) => {}
                Err(error) => return Err(error),
            }
        }
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        unsafe { self.upstream.deallocate(block, stream) }
    }
}

/// One RMM-style current/peak/total counter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StatisticsCounter {
    pub value: i64,
    pub peak: i64,
    pub total: u64,
}

impl StatisticsCounter {
    fn add(&mut self, amount: usize) {
        self.value = self.value.saturating_add(amount as i64);
        self.peak = self.peak.max(self.value);
        self.total = self.total.saturating_add(amount as u64);
    }

    fn subtract(&mut self, amount: usize) {
        self.value = self.value.saturating_sub(amount as i64);
    }
}

/// Byte and allocation counters for one statistics scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StatisticsSnapshot {
    pub bytes: StatisticsCounter,
    pub allocations: StatisticsCounter,
}

/// Debug statistics adaptor with nested counter scopes.
///
/// `push_counters` starts a fresh scope and `pop_counters` returns the completed
/// inner scope. Allocation and deallocation affect only the current top scope,
/// matching RMM's nested-statistics semantics (including negative values when
/// an inner scope frees an outer allocation).
#[derive(Debug)]
pub struct StatisticsMemoryResource<U> {
    upstream: U,
    counters: Mutex<Vec<StatisticsSnapshot>>,
}

impl<U: DeviceAllocator> StatisticsMemoryResource<U> {
    pub fn new(upstream: U) -> Self {
        Self {
            upstream,
            counters: Mutex::new(vec![StatisticsSnapshot::default()]),
        }
    }

    pub fn snapshot(&self) -> StatisticsSnapshot {
        *self
            .counters
            .lock()
            .expect("statistics resource poisoned")
            .last()
            .expect("root counter scope always exists")
    }

    pub fn push_counters(&self) -> StatisticsSnapshot {
        let mut counters = self.counters.lock().expect("statistics resource poisoned");
        let previous = *counters.last().expect("root counter scope always exists");
        counters.push(StatisticsSnapshot::default());
        previous
    }

    pub fn pop_counters(&self) -> Result<StatisticsSnapshot, MemoryError> {
        let mut counters = self.counters.lock().expect("statistics resource poisoned");
        if counters.len() == 1 {
            return Err(MemoryError::CounterStackUnderflow);
        }
        Ok(counters.pop().expect("length checked above"))
    }

    pub fn scope_depth(&self) -> usize {
        self.counters
            .lock()
            .expect("statistics resource poisoned")
            .len()
    }

    pub fn upstream(&self) -> &U {
        &self.upstream
    }
}

unsafe impl<U: DeviceAllocator> DeviceAllocator for StatisticsMemoryResource<U> {
    fn context(&self) -> &Arc<CudaContext> {
        self.upstream.context()
    }

    fn properties(&self) -> MemoryProperties {
        self.upstream.properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        let block = self.upstream.allocate(layout, stream)?;
        let mut counters = self.counters.lock().expect("statistics resource poisoned");
        let current = counters
            .last_mut()
            .expect("root counter scope always exists");
        current.bytes.add(block.size());
        if block.size() > 0 {
            current.allocations.add(1);
        }
        Ok(block)
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        let size = block.size();
        unsafe { self.upstream.deallocate(block, stream)? };
        let mut counters = self.counters.lock().expect("statistics resource poisoned");
        let current = counters
            .last_mut()
            .expect("root counter scope always exists");
        current.bytes.subtract(size);
        if size > 0 {
            current.allocations.subtract(1);
        }
        Ok(())
    }
}
