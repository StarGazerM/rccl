use crate::{
    DEVICE_ALLOCATION_ALIGNMENT, DeviceAllocator, DeviceBlock, MemoryError, MemoryProperties,
    validate_layout, validate_stream,
};
use cuda_core::{CudaContext, CudaEvent, CudaStream, IntoResult, sys};
use std::{
    alloc::Layout,
    collections::{BTreeMap, HashMap},
    fmt,
    sync::{Arc, Mutex},
};

pub const DEFAULT_FIXED_BLOCK_SIZE: usize = 1 << 20;
pub const DEFAULT_BLOCKS_TO_PREALLOCATE: usize = 128;

#[derive(Debug)]
struct FixedFreeBlock {
    ptr: sys::CUdeviceptr,
    ready: Arc<CudaEvent>,
    stream_id: u64,
}

#[derive(Debug)]
struct FixedState {
    free: Vec<FixedFreeBlock>,
    live: HashMap<sys::CUdeviceptr, usize>,
    stream_events: HashMap<u64, Arc<CudaEvent>>,
    upstream_chunks: Vec<DeviceBlock>,
}

/// Stream-ordered allocator for requests no larger than one fixed block.
///
/// Upstream allocations are made in chunks and split into equal blocks. This
/// mirrors RMM's fixed-size resource and avoids free-list search or external
/// fragmentation for uniform allocation workloads.
pub struct FixedSizeMemoryResource<U: DeviceAllocator> {
    upstream: U,
    block_size: usize,
    blocks_per_chunk: usize,
    state: Mutex<FixedState>,
}

impl<U: DeviceAllocator> fmt::Debug for FixedSizeMemoryResource<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.lock().expect("fixed-size pool poisoned");
        f.debug_struct("FixedSizeMemoryResource")
            .field("block_size", &self.block_size)
            .field("blocks_per_chunk", &self.blocks_per_chunk)
            .field("free_blocks", &state.free.len())
            .field("live_allocations", &state.live.len())
            .field("upstream_chunks", &state.upstream_chunks.len())
            .finish()
    }
}

impl<U: DeviceAllocator> FixedSizeMemoryResource<U> {
    pub fn new(
        upstream: U,
        block_size: usize,
        blocks_to_preallocate: usize,
    ) -> Result<Self, MemoryError> {
        let block_size = round_up(block_size)?;
        if block_size == 0 || blocks_to_preallocate == 0 {
            return Err(MemoryError::InvalidPoolConfiguration {
                initial_size: block_size,
                maximum_size: blocks_to_preallocate,
            });
        }
        let resource = Self {
            upstream,
            block_size,
            blocks_per_chunk: blocks_to_preallocate,
            state: Mutex::new(FixedState {
                free: Vec::new(),
                live: HashMap::new(),
                stream_events: HashMap::new(),
                upstream_chunks: Vec::new(),
            }),
        };
        let stream = resource.context().default_stream();
        let mut state = resource.state.lock().expect("fixed-size pool poisoned");
        resource.grow(&mut state, &stream)?;
        drop(state);
        Ok(resource)
    }

    pub fn with_defaults(upstream: U) -> Result<Self, MemoryError> {
        Self::new(
            upstream,
            DEFAULT_FIXED_BLOCK_SIZE,
            DEFAULT_BLOCKS_TO_PREALLOCATE,
        )
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn upstream(&self) -> &U {
        &self.upstream
    }

    pub fn free_blocks(&self) -> usize {
        self.state
            .lock()
            .expect("fixed-size pool poisoned")
            .free
            .len()
    }

    fn grow(&self, state: &mut FixedState, stream: &CudaStream) -> Result<(), MemoryError> {
        let chunk_size = self
            .block_size
            .checked_mul(self.blocks_per_chunk)
            .ok_or(MemoryError::SizeOverflow)?;
        let layout = Layout::from_size_align(chunk_size, DEVICE_ALLOCATION_ALIGNMENT)?;
        let chunk = self.upstream.allocate(layout, stream)?;
        let stream_id = stream_id(stream)?;
        let ready = match record_ready(state, stream, stream_id) {
            Ok(event) => event,
            Err(error) => {
                let _ = unsafe { self.upstream.deallocate(chunk, stream) };
                return Err(error);
            }
        };
        for index in 0..self.blocks_per_chunk {
            state.free.push(FixedFreeBlock {
                ptr: chunk.cu_deviceptr() + (index * self.block_size) as sys::CUdeviceptr,
                ready: ready.clone(),
                stream_id,
            });
        }
        state.upstream_chunks.push(chunk);
        Ok(())
    }
}

unsafe impl<U: DeviceAllocator> DeviceAllocator for FixedSizeMemoryResource<U> {
    fn context(&self) -> &Arc<CudaContext> {
        self.upstream.context()
    }

    fn properties(&self) -> MemoryProperties {
        self.upstream.properties()
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        validate_layout(layout)?;
        validate_stream(self.context(), stream)?;
        if layout.size() == 0 {
            return Ok(unsafe { DeviceBlock::from_raw_parts(0, 0, layout.align()) });
        }
        if round_up(layout.size())? > self.block_size {
            return Err(MemoryError::AllocationLimitExceeded {
                requested: layout.size(),
                allocated: 0,
                limit: self.block_size,
            });
        }
        let stream_id = stream_id(stream)?;
        let mut state = self.state.lock().expect("fixed-size pool poisoned");
        if state.free.is_empty() {
            self.grow(&mut state, stream)?;
        }
        let free = state.free.pop().expect("pool was grown above");
        if free.stream_id != stream_id {
            stream.wait(&free.ready)?;
        }
        state.live.insert(free.ptr, layout.size());
        Ok(unsafe {
            DeviceBlock::from_raw_parts(free.ptr, layout.size(), DEVICE_ALLOCATION_ALIGNMENT)
        })
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        validate_stream(self.context(), stream)?;
        if block.cu_deviceptr() == 0 {
            return Ok(());
        }
        let stream_id = stream_id(stream)?;
        let mut state = self.state.lock().expect("fixed-size pool poisoned");
        let ready = record_ready(&mut state, stream, stream_id)?;
        let Some(expected_size) = state.live.remove(&block.cu_deviceptr()) else {
            return Err(MemoryError::UnknownAllocation {
                ptr: block.cu_deviceptr(),
            });
        };
        if expected_size != block.size() {
            state.live.insert(block.cu_deviceptr(), expected_size);
            return Err(MemoryError::AllocationSizeMismatch {
                expected: expected_size,
                actual: block.size(),
            });
        }
        state.free.push(FixedFreeBlock {
            ptr: block.cu_deviceptr(),
            ready,
            stream_id,
        });
        Ok(())
    }
}

impl<U: DeviceAllocator> Drop for FixedSizeMemoryResource<U> {
    fn drop(&mut self) {
        let context = self.upstream.context().clone();
        let stream = context.default_stream();
        let state = self.state.get_mut().expect("fixed-size pool poisoned");
        for free in &state.free {
            context.record_err(free.ready.synchronize());
        }
        for chunk in state.upstream_chunks.drain(..) {
            match unsafe { self.upstream.deallocate(chunk, &stream) } {
                Ok(()) => {}
                Err(MemoryError::Driver(error)) => context.record_err::<()>(Err(error)),
                Err(_) => {}
            }
        }
        context.record_err(stream.synchronize());
    }
}

/// Size-class dispatcher equivalent to RMM's binning memory resource.
///
/// Each request uses the smallest registered bin at least as large as the
/// request; requests larger than every bin go directly to the upstream
/// resource. Deallocation repeats the same deterministic size lookup.
pub struct BinningMemoryResource<
    U: DeviceAllocator,
    B: DeviceAllocator = FixedSizeMemoryResource<U>,
> {
    upstream: U,
    bins: BTreeMap<usize, B>,
}

impl<U: DeviceAllocator, B: DeviceAllocator> fmt::Debug for BinningMemoryResource<U, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BinningMemoryResource")
            .field("bin_sizes", &self.bins.keys().collect::<Vec<_>>())
            .field("upstream_type", &std::any::type_name::<U>())
            .field("bin_type", &std::any::type_name::<B>())
            .finish()
    }
}

impl<U: DeviceAllocator, B: DeviceAllocator> BinningMemoryResource<U, B> {
    /// Create an empty dispatcher for explicitly supplied, homogeneous bins.
    /// Use an enum as `B` when a closed set of different bin policies is
    /// required.
    pub fn with_upstream(upstream: U) -> Self {
        Self {
            upstream,
            bins: BTreeMap::new(),
        }
    }

    pub fn add_bin(&mut self, allocation_size: usize, bin: B) -> Result<(), MemoryError> {
        let allocation_size = round_up(allocation_size)?;
        if bin.context().as_ref() != self.upstream.context().as_ref() {
            return Err(MemoryError::ContextMismatch);
        }
        self.bins.entry(allocation_size).or_insert(bin);
        Ok(())
    }

    pub fn bin_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        self.bins.keys().copied()
    }

    pub fn upstream(&self) -> &U {
        &self.upstream
    }
}

impl<U: DeviceAllocator + Clone> BinningMemoryResource<U, FixedSizeMemoryResource<U>> {
    /// Create an empty dispatcher whose bins are fixed-size allocators sharing
    /// clones of the upstream handle. Borrowed handles (`&U`) clone by copying
    /// the borrow; `Arc<U>` remains an explicit caller choice.
    pub fn new(upstream: U) -> Self {
        Self::with_upstream(upstream)
    }

    pub fn with_exponents(
        upstream: U,
        minimum_exponent: u8,
        maximum_exponent: u8,
    ) -> Result<Self, MemoryError> {
        let mut resource = Self::new(upstream);
        for exponent in minimum_exponent..=maximum_exponent {
            let size = 1usize
                .checked_shl(exponent.into())
                .ok_or(MemoryError::SizeOverflow)?;
            resource.add_fixed_bin(size)?;
        }
        Ok(resource)
    }

    pub fn add_fixed_bin(&mut self, allocation_size: usize) -> Result<(), MemoryError> {
        let allocation_size = round_up(allocation_size)?;
        let bin = FixedSizeMemoryResource::new(
            self.upstream.clone(),
            allocation_size,
            DEFAULT_BLOCKS_TO_PREALLOCATE,
        )?;
        self.bins.entry(allocation_size).or_insert(bin);
        Ok(())
    }
}

unsafe impl<U: DeviceAllocator, B: DeviceAllocator> DeviceAllocator
    for BinningMemoryResource<U, B>
{
    fn context(&self) -> &Arc<CudaContext> {
        self.upstream.context()
    }

    fn properties(&self) -> MemoryProperties {
        self.bins
            .values()
            .fold(self.upstream.properties(), |properties, resource| {
                properties.intersection(resource.properties())
            })
    }

    fn allocate(&self, layout: Layout, stream: &CudaStream) -> Result<DeviceBlock, MemoryError> {
        match self.bins.range(layout.size()..).next() {
            Some((_, bin)) => bin.allocate(layout, stream),
            None => self.upstream.allocate(layout, stream),
        }
    }

    unsafe fn deallocate(
        &self,
        block: DeviceBlock,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        match self.bins.range(block.size()..).next() {
            Some((_, bin)) => unsafe { bin.deallocate(block, stream) },
            None => unsafe { self.upstream.deallocate(block, stream) },
        }
    }
}

fn round_up(size: usize) -> Result<usize, MemoryError> {
    if size == 0 {
        return Ok(0);
    }
    size.checked_add(DEVICE_ALLOCATION_ALIGNMENT - 1)
        .map(|value| value & !(DEVICE_ALLOCATION_ALIGNMENT - 1))
        .ok_or(MemoryError::SizeOverflow)
}

fn stream_id(stream: &CudaStream) -> Result<u64, MemoryError> {
    if stream.cu_stream().is_null() {
        return Ok(u64::MAX);
    }
    stream.context().bind_to_thread()?;
    let mut id = 0u64;
    unsafe { sys::cuStreamGetId(stream.cu_stream(), &mut id).result()? };
    Ok(id)
}

fn record_ready(
    state: &mut FixedState,
    stream: &CudaStream,
    stream_id: u64,
) -> Result<Arc<CudaEvent>, MemoryError> {
    if let Some(event) = state.stream_events.get(&stream_id) {
        unsafe { sys::cuEventRecord(event.cu_event(), stream.cu_stream()).result()? };
        return Ok(event.clone());
    }
    let event = Arc::new(stream.record_event(None)?);
    state.stream_events.insert(stream_id, event.clone());
    Ok(event)
}
