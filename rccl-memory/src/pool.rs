use crate::{
    DEVICE_ALLOCATION_ALIGNMENT, DeviceAllocator, DeviceBlock, MemoryError, MemoryProperties,
    validate_layout, validate_stream,
};
use cuda_core::{CudaContext, CudaEvent, CudaStream, IntoResult, sys};
use std::{
    alloc::Layout,
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    sync::{Arc, Mutex},
};

const DEFAULT_GROWTH_SIZE: usize = 1 << 20;

#[derive(Debug)]
struct FreeBlock {
    size: usize,
    ready: Arc<CudaEvent>,
    stream_id: u64,
    upstream_head: bool,
}

#[derive(Clone, Copy, Debug)]
struct LiveAllocation {
    rounded_size: usize,
    requested_size: usize,
    upstream_head: bool,
}

#[derive(Debug)]
struct PoolState {
    free: BTreeMap<sys::CUdeviceptr, FreeBlock>,
    free_by_size: BTreeSet<(usize, sys::CUdeviceptr)>,
    live: HashMap<sys::CUdeviceptr, LiveAllocation>,
    stream_events: HashMap<u64, Arc<CudaEvent>>,
    upstream_blocks: Vec<DeviceBlock>,
    pool_size: usize,
    next_growth: usize,
    allocated_bytes: usize,
    peak_allocated_bytes: usize,
    total_allocations: u64,
}

/// Host-side statistics for [`PoolMemoryResource`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SuballocatorStats {
    pub pool_size: usize,
    pub free_bytes: usize,
    pub allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub upstream_allocations: usize,
    pub live_allocations: usize,
    pub total_allocations: u64,
}

/// Stream-ordered coalescing best-fit suballocator.
///
/// This is the Rust counterpart of RMM's `pool_memory_resource`: large chunks
/// come from an arbitrary upstream resource and small allocations are served
/// from an address-ordered free list. A CUDA event accompanies every free
/// range, so reuse on another stream inserts a dependency instead of forcing a
/// device-wide synchronization.
pub struct PoolMemoryResource<U: DeviceAllocator> {
    upstream: U,
    maximum_size: usize,
    state: Mutex<PoolState>,
}

impl<U: DeviceAllocator> fmt::Debug for PoolMemoryResource<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolMemoryResource")
            .field("device", &self.context().ordinal())
            .field("maximum_size", &self.maximum_size)
            .field("stats", &self.stats())
            .field("upstream_type", &std::any::type_name::<U>())
            .finish()
    }
}

impl<U: DeviceAllocator> PoolMemoryResource<U> {
    /// Create a pool and optionally prime it with `initial_size` bytes.
    ///
    /// A `maximum_size` of zero means unlimited. Sizes are rounded up to CUDA's
    /// 256-byte allocation alignment.
    pub fn new(upstream: U, initial_size: usize, maximum_size: usize) -> Result<Self, MemoryError> {
        let initial_size = round_up(initial_size)?;
        let maximum_size = if maximum_size == 0 {
            usize::MAX
        } else {
            round_up(maximum_size)?
        };
        if initial_size > maximum_size {
            return Err(MemoryError::InvalidPoolConfiguration {
                initial_size,
                maximum_size,
            });
        }
        let resource = Self {
            upstream,
            maximum_size,
            state: Mutex::new(PoolState {
                free: BTreeMap::new(),
                free_by_size: BTreeSet::new(),
                live: HashMap::new(),
                stream_events: HashMap::new(),
                upstream_blocks: Vec::new(),
                pool_size: 0,
                next_growth: initial_size.max(DEFAULT_GROWTH_SIZE),
                allocated_bytes: 0,
                peak_allocated_bytes: 0,
                total_allocations: 0,
            }),
        };
        if initial_size > 0 {
            let stream = resource.context().default_stream();
            let mut state = resource.state.lock().expect("memory pool poisoned");
            resource.grow_free(&mut state, initial_size, &stream)?;
        }
        Ok(resource)
    }

    pub fn upstream(&self) -> &U {
        &self.upstream
    }

    pub fn maximum_size(&self) -> Option<usize> {
        (self.maximum_size != usize::MAX).then_some(self.maximum_size)
    }

    pub fn stats(&self) -> SuballocatorStats {
        let state = self.state.lock().expect("memory pool poisoned");
        SuballocatorStats {
            pool_size: state.pool_size,
            free_bytes: state.free.values().map(|block| block.size).sum(),
            allocated_bytes: state.allocated_bytes,
            peak_allocated_bytes: state.peak_allocated_bytes,
            upstream_allocations: state.upstream_blocks.len(),
            live_allocations: state.live.len(),
            total_allocations: state.total_allocations,
        }
    }

    /// Return every wholly free upstream chunk.
    ///
    /// Dependencies are inserted into `stream`; no device-wide synchronization
    /// is performed. Fragmented or partially allocated chunks remain pooled.
    pub fn release_unused(&self, stream: &CudaStream) -> Result<usize, MemoryError> {
        validate_stream(self.context(), stream)?;
        let stream_id = stream_id(stream)?;
        let mut state = self.state.lock().expect("memory pool poisoned");
        self.coalesce_all(&mut state, stream, stream_id)?;

        let mut released = 0usize;
        let mut index = 0usize;
        while index < state.upstream_blocks.len() {
            let upstream_block = state.upstream_blocks[index];
            let is_wholly_free = state
                .free
                .get(&upstream_block.cu_deviceptr())
                .is_some_and(|free| free.upstream_head && free.size == upstream_block.size());
            if !is_wholly_free {
                index += 1;
                continue;
            }

            {
                let free = state
                    .free
                    .get(&upstream_block.cu_deviceptr())
                    .expect("free block checked above");
                order_after(stream, stream_id, free)?;
            }
            let free = remove_free(&mut state, upstream_block.cu_deviceptr())
                .expect("free block checked above");
            if let Err(error) = unsafe { self.upstream.deallocate(upstream_block, stream) } {
                insert_free(&mut state, upstream_block.cu_deviceptr(), free);
                return Err(error);
            }
            state.upstream_blocks.swap_remove(index);
            state.pool_size -= upstream_block.size();
            released += upstream_block.size();
        }
        Ok(released)
    }

    fn allocate_from_free(
        &self,
        state: &mut PoolState,
        requested_size: usize,
        rounded_size: usize,
        stream: &CudaStream,
        stream_id: u64,
    ) -> Result<Option<DeviceBlock>, MemoryError> {
        let best = state
            .free_by_size
            .range((rounded_size, 0)..)
            .next()
            .map(|(_, ptr)| *ptr);
        let Some(ptr) = best else {
            return Ok(None);
        };

        {
            let free = state.free.get(&ptr).expect("selected free block exists");
            order_after(stream, stream_id, free)?;
        }
        let free = remove_free(state, ptr).expect("selected free block exists");
        let remainder_size = free.size - rounded_size;
        if remainder_size > 0 {
            insert_free(
                state,
                ptr + rounded_size as sys::CUdeviceptr,
                FreeBlock {
                    size: remainder_size,
                    ready: free.ready.clone(),
                    stream_id,
                    upstream_head: false,
                },
            );
        }
        self.record_live(state, ptr, rounded_size, requested_size, free.upstream_head);
        Ok(Some(unsafe {
            DeviceBlock::from_raw_parts(ptr, requested_size, DEVICE_ALLOCATION_ALIGNMENT)
        }))
    }

    fn record_live(
        &self,
        state: &mut PoolState,
        ptr: sys::CUdeviceptr,
        rounded_size: usize,
        requested_size: usize,
        upstream_head: bool,
    ) {
        state.live.insert(
            ptr,
            LiveAllocation {
                rounded_size,
                requested_size,
                upstream_head,
            },
        );
        state.allocated_bytes += rounded_size;
        state.peak_allocated_bytes = state.peak_allocated_bytes.max(state.allocated_bytes);
        state.total_allocations += 1;
    }

    fn grow_for_allocation(
        &self,
        state: &mut PoolState,
        requested_size: usize,
        rounded_size: usize,
        stream: &CudaStream,
        stream_id: u64,
    ) -> Result<DeviceBlock, MemoryError> {
        let remaining = self.maximum_size.saturating_sub(state.pool_size);
        if rounded_size > remaining {
            return Err(MemoryError::AllocationLimitExceeded {
                requested: rounded_size,
                allocated: state.pool_size,
                limit: self.maximum_size,
            });
        }
        let growth = rounded_size.max(state.next_growth.min(remaining));
        let layout = Layout::from_size_align(growth, DEVICE_ALLOCATION_ALIGNMENT)?;
        let upstream_block = self.upstream.allocate(layout, stream)?;
        let remainder_size = growth - rounded_size;
        let remainder_event = if remainder_size > 0 {
            match record_ready(state, stream, stream_id) {
                Ok(event) => Some(event),
                Err(error) => {
                    let _ = unsafe { self.upstream.deallocate(upstream_block, stream) };
                    return Err(error);
                }
            }
        } else {
            None
        };
        let ptr = upstream_block.cu_deviceptr();
        if remainder_size > 0 {
            let remainder_ptr = ptr + rounded_size as sys::CUdeviceptr;
            insert_free(
                state,
                remainder_ptr,
                FreeBlock {
                    size: remainder_size,
                    ready: remainder_event.expect("remainder records an event"),
                    stream_id,
                    upstream_head: false,
                },
            );
        }
        state.upstream_blocks.push(upstream_block);
        state.pool_size += growth;
        state.next_growth = growth.saturating_mul(2).min(self.maximum_size);
        self.record_live(state, ptr, rounded_size, requested_size, true);
        Ok(
            unsafe {
                DeviceBlock::from_raw_parts(ptr, requested_size, DEVICE_ALLOCATION_ALIGNMENT)
            },
        )
    }

    fn grow_free(
        &self,
        state: &mut PoolState,
        size: usize,
        stream: &CudaStream,
    ) -> Result<(), MemoryError> {
        let layout = Layout::from_size_align(size, DEVICE_ALLOCATION_ALIGNMENT)?;
        let upstream_block = self.upstream.allocate(layout, stream)?;
        let stream_id = stream_id(stream)?;
        let ready = match record_ready(state, stream, stream_id) {
            Ok(event) => event,
            Err(error) => {
                let _ = unsafe { self.upstream.deallocate(upstream_block, stream) };
                return Err(error);
            }
        };
        let ptr = upstream_block.cu_deviceptr();
        insert_free(
            state,
            ptr,
            FreeBlock {
                size,
                ready,
                stream_id,
                upstream_head: true,
            },
        );
        state.upstream_blocks.push(upstream_block);
        state.pool_size += size;
        Ok(())
    }

    fn coalesce_all(
        &self,
        state: &mut PoolState,
        stream: &CudaStream,
        stream_id: u64,
    ) -> Result<(), MemoryError> {
        loop {
            let mut adjacent = None;
            let mut previous = None;
            for (&ptr, block) in &state.free {
                if let Some((previous_ptr, previous_size)) = previous
                    && previous_ptr + previous_size as sys::CUdeviceptr == ptr
                    && !block.upstream_head
                {
                    adjacent = Some((previous_ptr, ptr));
                    break;
                }
                previous = Some((ptr, block.size));
            }
            let Some((left_ptr, right_ptr)) = adjacent else {
                return Ok(());
            };

            order_after(
                stream,
                stream_id,
                state.free.get(&left_ptr).expect("left free block exists"),
            )?;
            order_after(
                stream,
                stream_id,
                state.free.get(&right_ptr).expect("right free block exists"),
            )?;
            let ready = record_ready(state, stream, stream_id)?;
            let left = remove_free(state, left_ptr).expect("left block exists");
            let right = remove_free(state, right_ptr).expect("right block exists");
            insert_free(
                state,
                left_ptr,
                FreeBlock {
                    size: left.size + right.size,
                    ready,
                    stream_id,
                    upstream_head: left.upstream_head,
                },
            );
        }
    }
}

unsafe impl<U: DeviceAllocator> DeviceAllocator for PoolMemoryResource<U> {
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
        let rounded_size = round_up(layout.size())?;
        let stream_id = stream_id(stream)?;
        let mut state = self.state.lock().expect("memory pool poisoned");

        if let Some(block) =
            self.allocate_from_free(&mut state, layout.size(), rounded_size, stream, stream_id)?
        {
            return Ok(block);
        }
        self.coalesce_all(&mut state, stream, stream_id)?;
        if let Some(block) =
            self.allocate_from_free(&mut state, layout.size(), rounded_size, stream, stream_id)?
        {
            return Ok(block);
        }
        self.grow_for_allocation(&mut state, layout.size(), rounded_size, stream, stream_id)
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
        let mut state = self.state.lock().expect("memory pool poisoned");
        let stream_id = stream_id(stream)?;
        let ready = record_ready(&mut state, stream, stream_id)?;
        let Some(allocation) = state.live.remove(&block.cu_deviceptr()) else {
            return Err(MemoryError::UnknownAllocation {
                ptr: block.cu_deviceptr(),
            });
        };
        if allocation.requested_size != block.size() {
            state.live.insert(block.cu_deviceptr(), allocation);
            return Err(MemoryError::AllocationSizeMismatch {
                expected: allocation.requested_size,
                actual: block.size(),
            });
        }
        state.allocated_bytes -= allocation.rounded_size;
        insert_free(
            &mut state,
            block.cu_deviceptr(),
            FreeBlock {
                size: allocation.rounded_size,
                ready,
                stream_id,
                upstream_head: allocation.upstream_head,
            },
        );
        Ok(())
    }
}

impl<U: DeviceAllocator> Drop for PoolMemoryResource<U> {
    fn drop(&mut self) {
        let context = self.upstream.context().clone();
        let stream = context.default_stream();
        let state = self.state.get_mut().expect("memory pool poisoned");
        for free in state.free.values() {
            context.record_err(free.ready.synchronize());
        }
        for block in state.upstream_blocks.drain(..) {
            match unsafe { self.upstream.deallocate(block, &stream) } {
                Ok(()) => {}
                Err(MemoryError::Driver(error)) => context.record_err::<()>(Err(error)),
                Err(_) => {}
            }
        }
        context.record_err(stream.synchronize());
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

fn order_after(stream: &CudaStream, stream_id: u64, free: &FreeBlock) -> Result<(), MemoryError> {
    if free.stream_id != stream_id {
        stream.wait(&free.ready)?;
    }
    Ok(())
}

fn record_ready(
    state: &mut PoolState,
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

/// Insert a free range and coalesce address neighbors already ordered on the
/// same stream. Cross-stream neighbors remain separate until `coalesce_all`
/// inserts their event dependencies. An upstream head is a hard boundary.
fn insert_free(state: &mut PoolState, mut ptr: sys::CUdeviceptr, mut block: FreeBlock) {
    let previous = state
        .free
        .range(..ptr)
        .next_back()
        .map(|(&previous_ptr, previous)| {
            (
                previous_ptr,
                previous.size,
                previous.stream_id,
                previous.upstream_head,
            )
        });
    if let Some((previous_ptr, previous_size, previous_stream, previous_head)) = previous
        && !block.upstream_head
        && previous_ptr + previous_size as sys::CUdeviceptr == ptr
        && previous_stream == block.stream_id
    {
        remove_free(state, previous_ptr).expect("previous block was observed above");
        ptr = previous_ptr;
        block.size += previous_size;
        block.upstream_head = previous_head;
    }

    let next_ptr = ptr + block.size as sys::CUdeviceptr;
    let next = state
        .free
        .get(&next_ptr)
        .map(|next| (next.size, next.stream_id, next.upstream_head));
    if let Some((next_size, next_stream, next_head)) = next
        && !next_head
        && next_stream == block.stream_id
    {
        remove_free(state, next_ptr).expect("next block was observed above");
        block.size += next_size;
    }
    state.free_by_size.insert((block.size, ptr));
    state.free.insert(ptr, block);
}

fn remove_free(state: &mut PoolState, ptr: sys::CUdeviceptr) -> Option<FreeBlock> {
    let block = state.free.remove(&ptr)?;
    let removed = state.free_by_size.remove(&(block.size, ptr));
    debug_assert!(removed, "size index must contain every free block");
    Some(block)
}
