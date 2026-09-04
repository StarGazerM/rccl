use cuda_core::IntoResult;
use rccl_memory::{
    AlignedMemoryResource, BinningMemoryResource, CallbackMemoryResource, ContiguousDeviceRange,
    CudaAsyncMemoryResource, CudaCompletion, CudaMemoryResource, CudaPoolMemoryResource,
    DeviceAllocator, DeviceBuffer, DeviceScalar, DeviceUninitBuffer, FailureCallbackMemoryResource,
    FixedSizeMemoryResource, LimitingMemoryResource, LoggingMemoryResource, ManagedMemoryResource,
    MemoryError, PinnedHostMemoryResource, PoolConfig, PoolMemoryResource, PrefetchMemoryResource,
    SamHeadroomMemoryResource, StatisticsMemoryResource, SystemMemoryResource,
    ThreadSafeMemoryResource, TrackingMemoryResource, cross_stream_dependency,
};
use std::{
    alloc::Layout,
    cell::RefCell,
    error::Error,
    io::{self, Write},
    mem::{MaybeUninit, size_of},
    rc::Rc,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

#[derive(Clone)]
struct SharedWriter(Rc<RefCell<Vec<u8>>>);

impl Write for SharedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("GPU test lock poisoned")
}

fn context_or_skip() -> Option<Arc<cuda_core::CudaContext>> {
    match cuda_core::CudaContext::new(0) {
        Ok(context) => Some(context),
        Err(error) => {
            eprintln!("SKIP: CUDA device 0 is unavailable: {error}");
            None
        }
    }
}

fn assert_round_trip<A: DeviceAllocator>(
    allocator: A,
    stream: &cuda_core::CudaStream,
    values: &[u64],
) -> Result<(), Box<dyn Error>> {
    let buffer = DeviceBuffer::from_host_in(allocator, stream, values)?;
    assert_eq!(buffer.to_host_vec()?, values);
    drop(buffer);
    stream.synchronize()?;
    Ok(())
}

#[test]
fn completion_wait_error_and_drop_runtime_contract() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.new_stream()?;

    let waited = Arc::new(AtomicUsize::new(0));
    // SAFETY: the queued callback owns its Arc and is enqueued on `stream`;
    // the retained Arc names the same state until the completion is released.
    let completion = unsafe {
        CudaCompletion::from_submission(&stream, waited.clone(), |retained| {
            let callback_state = retained.clone();
            stream
                .launch_host_function(move || callback_state.store(1, Ordering::Release))
                .map_err(MemoryError::Driver)?;
            Ok::<_, MemoryError>(137)
        })
    };
    assert_eq!(completion.wait()?, 137);
    assert_eq!(waited.load(Ordering::Acquire), 1);

    // SAFETY: this injected submission error enqueues no GPU work.
    let failed = unsafe {
        CudaCompletion::from_submission(&stream, (), |_| {
            Err::<(), MemoryError>(MemoryError::SizeOverflow)
        })
    };
    assert!(matches!(failed.wait(), Err(MemoryError::SizeOverflow)));

    let dropped = Arc::new(AtomicUsize::new(0));
    // SAFETY: as above, the callback owns its Arc and uses only `stream`.
    let completion = unsafe {
        CudaCompletion::from_submission(&stream, dropped.clone(), |retained| {
            let callback_state = retained.clone();
            stream
                .launch_host_function(move || {
                    thread::sleep(Duration::from_millis(10));
                    callback_state.store(1, Ordering::Release);
                })
                .map_err(MemoryError::Driver)
        })
    };
    drop(completion);
    assert_eq!(dropped.load(Ordering::Acquire), 1);
    assert!(stream.query()?);
    Ok(())
}

#[test]
fn cross_stream_dependency_orders_work_after_event_drop() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let producer = context.new_stream()?;
    let consumer = context.new_stream()?;
    let ordering = Arc::new(AtomicUsize::new(0));

    let producer_ordering = ordering.clone();
    producer.launch_host_function(move || {
        thread::sleep(Duration::from_millis(10));
        producer_ordering.store(1, Ordering::Release);
    })?;
    let event = cross_stream_dependency(&producer, &consumer)?;
    let consumer_ordering = ordering.clone();
    consumer.launch_host_function(move || {
        let observed = consumer_ordering.load(Ordering::Acquire);
        consumer_ordering.store(if observed == 1 { 2 } else { 3 }, Ordering::Release);
    })?;

    drop(event);
    consumer.synchronize()?;
    assert_eq!(ordering.load(Ordering::Acquire), 2);
    Ok(())
}

#[test]
fn direct_async_and_managed_resources_round_trip() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let values = [3_u64, 1, 4, 1, 5, 9];

    assert_round_trip(CudaMemoryResource::new(context.clone()), &stream, &values)?;
    assert_round_trip(
        CudaAsyncMemoryResource::new(context.clone()),
        &stream,
        &values,
    )?;
    assert_round_trip(
        ManagedMemoryResource::new(context.clone()),
        &stream,
        &values,
    )?;
    Ok(())
}

#[test]
fn managed_memory_is_host_accessible() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let resource = ManagedMemoryResource::new(context.clone());
    assert!(resource.properties().device_accessible);
    assert!(resource.properties().host_accessible);
    let buffer = DeviceBuffer::from_host_in(resource, &stream, &[11_u32, 22, 33])?;
    stream.synchronize()?;

    let host_pointer = buffer.cu_deviceptr() as *mut u32;
    unsafe { host_pointer.add(1).write(77) };
    assert_eq!(buffer.to_host_vec()?, [11, 77, 33]);
    drop(buffer);
    stream.synchronize()?;
    Ok(())
}

#[test]
fn limiting_resource_enforces_and_reclaims_exact_limit() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let limiting = LimitingMemoryResource::new(CudaAsyncMemoryResource::new(context.clone()), 64);

    let first = DeviceBuffer::<u8, _>::zeroed_in(&limiting, &stream, 64)?;
    assert_eq!(limiting.allocated_bytes(), 64);
    assert!(matches!(
        DeviceBuffer::<u8, _>::zeroed_in(&limiting, &stream, 1),
        Err(MemoryError::AllocationLimitExceeded { .. })
    ));
    drop(first);
    assert_eq!(limiting.allocated_bytes(), 0);
    let replacement = DeviceBuffer::<u8, _>::zeroed_in(&limiting, &stream, 64)?;
    drop(replacement);
    stream.synchronize()?;
    Ok(())
}

#[test]
fn failure_callback_controls_allocation_retries() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let failing = LimitingMemoryResource::new(CudaMemoryResource::new(context.clone()), 0);
    let attempts = AtomicUsize::new(0);
    let retrying = FailureCallbackMemoryResource::new(failing, |_, _| {
        attempts.fetch_add(1, Ordering::SeqCst) == 0
    });

    let result = retrying.allocate(Layout::from_size_align(1, 1)?, &stream);
    assert!(matches!(
        result,
        Err(MemoryError::AllocationLimitExceeded { .. })
    ));
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn callback_resource_and_thread_safe_adaptor_serialize_calls() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let upstream = CudaAsyncMemoryResource::new(context.clone());
    let active = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);
    let callback = unsafe {
        CallbackMemoryResource::new(
            context.clone(),
            |layout, stream| {
                let simultaneous = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(simultaneous, Ordering::SeqCst);
                std::thread::yield_now();
                let result = upstream.allocate(layout, stream);
                active.fetch_sub(1, Ordering::SeqCst);
                result
            },
            |block, stream| upstream.deallocate(block, stream),
        )
    };
    let serialized = ThreadSafeMemoryResource::new(callback);
    let streams = (0..4)
        .map(|_| context.new_stream())
        .collect::<Result<Vec<_>, _>>()?;
    std::thread::scope(|scope| -> Result<(), MemoryError> {
        let threads = streams
            .into_iter()
            .map(|stream| {
                let resource = &serialized;
                scope.spawn(move || -> Result<(), MemoryError> {
                    for _ in 0..32 {
                        let block = resource.allocate(
                            Layout::from_size_align(4096, 256).expect("valid layout"),
                            &stream,
                        )?;
                        unsafe { resource.deallocate(block, &stream)? };
                    }
                    stream.synchronize()?;
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().expect("callback worker panicked")?;
        }
        Ok(())
    })?;
    assert_eq!(peak.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn prefetch_adaptor_prefetches_managed_memory() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let upstream = ManagedMemoryResource::new(context.clone());
    let prefetch = PrefetchMemoryResource::new(upstream)?;
    let supported = prefetch.is_supported();
    let buffer = DeviceBuffer::<u8, _>::zeroed_in(prefetch, &stream, 64 << 10)?;
    stream.synchronize()?;

    if supported {
        let mut location = MaybeUninit::<i32>::uninit();
        unsafe {
            cuda_core::sys::cuMemRangeGetAttribute(
                location.as_mut_ptr().cast(),
                size_of::<i32>(),
                cuda_core::sys::CUmem_range_attribute_enum_CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION,
                buffer.cu_deviceptr(),
                buffer.allocation_bytes(),
            )
            .result()?;
            assert_eq!(location.assume_init(), context.ordinal() as i32);
        }
    }
    drop(buffer);
    stream.synchronize()?;

    let direct_prefetch = PrefetchMemoryResource::new(CudaMemoryResource::new(context))?;
    let direct = DeviceBuffer::from_host_in(direct_prefetch, &stream, &[1_u32, 2, 3])?;
    assert_eq!(direct.to_host_vec()?, [1, 2, 3]);
    drop(direct);
    stream.synchronize()?;
    Ok(())
}

#[test]
fn logging_adaptor_emits_rmm_compatible_csv_actions() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let upstream = CudaAsyncMemoryResource::new(context);
    let bytes = Rc::new(RefCell::new(Vec::new()));
    let logger = LoggingMemoryResource::new(upstream, SharedWriter(bytes.clone()), true)?;
    let block = logger.allocate(Layout::from_size_align(1234, 256)?, &stream)?;
    unsafe { logger.deallocate(block, &stream)? };
    logger.flush()?;
    stream.synchronize()?;

    let output = String::from_utf8(bytes.borrow().clone())?;
    let lines = output.lines().collect::<Vec<_>>();
    assert_eq!(
        lines[0],
        LoggingMemoryResource::<CudaAsyncMemoryResource, SharedWriter>::HEADER
    );
    assert!(lines[1].contains(",allocate,"));
    assert!(lines[1].contains(",1234,"));
    assert!(lines[2].contains(",free,"));
    assert!(lines[2].contains(",1234,"));
    Ok(())
}

#[test]
fn vector_capacity_clone_resize_and_scalar_contracts() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let resource = CudaAsyncMemoryResource::new(context.clone());

    let mut vector = DeviceBuffer::from_host_in(&resource, &stream, &[1_u32, 2, 3])?;
    vector.reserve(16)?;
    assert_eq!(vector.len(), 3);
    assert!(vector.capacity() >= 16);
    assert_eq!(vector.to_host_vec()?, [1, 2, 3]);
    vector.resize_zeroed(6)?;
    assert_eq!(vector.to_host_vec()?, [1, 2, 3, 0, 0, 0]);
    let copy = vector.try_clone()?;
    assert_eq!(copy.to_host_vec()?, [1, 2, 3, 0, 0, 0]);
    vector.shrink_to_fit()?;
    assert_eq!(vector.capacity(), vector.len());

    let mut scalar = DeviceScalar::new_in(&resource, &stream, 42_u64)?;
    assert_eq!(scalar.get()?, 42);
    scalar.set(99)?;
    assert_eq!(scalar.get()?, 99);
    drop((scalar, copy, vector));
    stream.synchronize()?;
    Ok(())
}

#[test]
fn uninitialized_buffer_requires_explicit_initialization_transition() -> Result<(), Box<dyn Error>>
{
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let storage =
        DeviceUninitBuffer::<u32, _>::new_in(CudaAsyncMemoryResource::new(context), &stream, 17)?;
    unsafe {
        cuda_core::memory::memset_d8_async(
            storage.cu_deviceptr(),
            0,
            storage.len() * size_of::<u32>(),
            stream.cu_stream(),
        )?;
    }
    stream.synchronize()?;
    let initialized = unsafe { storage.assume_init() };
    assert_eq!(initialized.to_host_vec()?, vec![0; 17]);
    Ok(())
}

#[test]
fn private_cuda_pool_reports_usage_and_trims() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let pool = CudaPoolMemoryResource::new(context, PoolConfig::default())?;
    pool.reset_high_watermarks()?;
    let allocation = DeviceBuffer::<u8, _>::zeroed_in(&pool, &stream, 1 << 20)?;
    stream.synchronize()?;
    let live = pool.stats()?;
    assert!(live.used_current >= 1 << 20);
    assert!(live.used_high >= live.used_current);
    drop(allocation);
    stream.synchronize()?;
    assert_eq!(pool.stats()?.used_current, 0);
    pool.trim_to(0)?;
    Ok(())
}

#[test]
fn borrowed_cuda_pool_is_an_allocator() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let pool = CudaPoolMemoryResource::new(context, PoolConfig::default())?;
    let buffer = DeviceBuffer::from_host_in(&pool, &stream, &[8_u32, 6, 7, 5, 3, 0, 9])?;
    assert_eq!(buffer.to_host_vec()?, [8, 6, 7, 5, 3, 0, 9]);
    drop(buffer);
    stream.synchronize()?;
    Ok(())
}

#[test]
fn buffer_rejects_a_different_stream() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let first_stream = context.new_stream()?;
    let second_stream = context.new_stream()?;
    let resource = CudaAsyncMemoryResource::new(context);
    let buffer = DeviceBuffer::from_host_in(resource, &first_stream, &[8_u32, 6, 7, 5, 3, 0, 9])?;
    assert!(matches!(
        buffer.ensure_usable_on(&second_stream),
        Err(MemoryError::StreamMismatch)
    ));
    assert_eq!(buffer.to_host_vec()?, [8, 6, 7, 5, 3, 0, 9]);
    drop(buffer);
    first_stream.synchronize()?;
    Ok(())
}

#[test]
fn coalescing_pool_reuses_and_merges_adjacent_blocks() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let upstream = CudaAsyncMemoryResource::new(context);
    let pool = PoolMemoryResource::new(upstream, 1024, 1024)?;
    let quarter = Layout::from_size_align(256, 8)?;
    let half = Layout::from_size_align(512, 8)?;

    let first = pool.allocate(quarter, &stream)?;
    let first_ptr = first.cu_deviceptr();
    let second = pool.allocate(quarter, &stream)?;
    let third = pool.allocate(quarter, &stream)?;
    unsafe {
        pool.deallocate(first, &stream)?;
        pool.deallocate(second, &stream)?;
    }
    let merged = pool.allocate(half, &stream)?;
    assert_eq!(merged.cu_deviceptr(), first_ptr);
    assert_eq!(pool.stats().pool_size, 1024);

    unsafe {
        pool.deallocate(merged, &stream)?;
        pool.deallocate(third, &stream)?;
    }
    assert_eq!(pool.release_unused(&stream)?, 1024);
    assert_eq!(pool.stats().pool_size, 0);
    stream.synchronize()?;
    Ok(())
}

#[test]
fn coalescing_pool_honors_hard_maximum() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let upstream = CudaMemoryResource::new(context);
    let pool = PoolMemoryResource::new(upstream, 512, 512)?;
    let full = pool.allocate(Layout::from_size_align(512, 8)?, &stream)?;
    assert!(matches!(
        pool.allocate(Layout::from_size_align(1, 1)?, &stream),
        Err(MemoryError::AllocationLimitExceeded { .. })
    ));
    unsafe { pool.deallocate(full, &stream)? };
    Ok(())
}

#[test]
fn coalescing_pool_reuses_across_streams_with_dependency() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let first_stream = context.new_stream()?;
    let second_stream = context.new_stream()?;
    let upstream = CudaAsyncMemoryResource::new(context);
    let pool = PoolMemoryResource::new(upstream, 1024, 1024)?;
    let layout = Layout::from_size_align(256, 8)?;
    let first = pool.allocate(layout, &first_stream)?;
    let first_ptr = first.cu_deviceptr();
    unsafe {
        cuda_core::memory::memset_d8_async(
            first.cu_deviceptr(),
            0xA5,
            first.size(),
            first_stream.cu_stream(),
        )?;
        pool.deallocate(first, &first_stream)?;
    }

    let reused = pool.allocate(layout, &second_stream)?;
    assert_eq!(reused.cu_deviceptr(), first_ptr);
    unsafe {
        cuda_core::memory::memset_d8_async(
            reused.cu_deviceptr(),
            0x5A,
            reused.size(),
            second_stream.cu_stream(),
        )?;
        pool.deallocate(reused, &second_stream)?;
    }
    second_stream.synchronize()?;
    Ok(())
}

#[test]
fn fixed_size_pool_preallocates_grows_and_reuses() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let tracked = TrackingMemoryResource::new(CudaAsyncMemoryResource::new(context));
    let fixed = FixedSizeMemoryResource::new(&tracked, 256, 2)?;
    assert_eq!(tracked.stats().total_allocations, 1);

    let layout = Layout::from_size_align(100, 4)?;
    let first = fixed.allocate(layout, &stream)?;
    let first_ptr = first.cu_deviceptr();
    let second = fixed.allocate(layout, &stream)?;
    let third = fixed.allocate(layout, &stream)?;
    assert_eq!(tracked.stats().total_allocations, 2);
    unsafe { fixed.deallocate(first, &stream)? };
    let reused = fixed.allocate(layout, &stream)?;
    assert_eq!(reused.cu_deviceptr(), first_ptr);
    unsafe {
        fixed.deallocate(reused, &stream)?;
        fixed.deallocate(second, &stream)?;
        fixed.deallocate(third, &stream)?;
    }
    stream.synchronize()?;
    Ok(())
}

#[test]
fn binning_resource_selects_bin_and_upstream_fallback() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let direct = CudaAsyncMemoryResource::new(context);
    let upstream_tracking = TrackingMemoryResource::new(&direct);
    let bin_tracking = TrackingMemoryResource::new(&direct);
    let mut binned = BinningMemoryResource::with_upstream(&upstream_tracking);
    binned.add_bin(256, &bin_tracking)?;

    let small = binned.allocate(Layout::from_size_align(100, 4)?, &stream)?;
    assert_eq!(bin_tracking.stats().outstanding_allocations, 1);
    assert_eq!(upstream_tracking.stats().outstanding_allocations, 0);
    unsafe { binned.deallocate(small, &stream)? };

    let large = binned.allocate(Layout::from_size_align(300, 4)?, &stream)?;
    assert_eq!(upstream_tracking.stats().outstanding_allocations, 1);
    unsafe { binned.deallocate(large, &stream)? };
    stream.synchronize()?;
    Ok(())
}

#[test]
fn aligned_resource_honors_large_layout_and_threshold() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let tracked = TrackingMemoryResource::new(CudaAsyncMemoryResource::new(context));
    let aligned = AlignedMemoryResource::new(&tracked, 4096, 0)?;
    assert!(!aligned.properties().host_accessible);
    let block = aligned.allocate(Layout::from_size_align(65_536, 8192)?, &stream)?;
    assert_eq!(block.cu_deviceptr() as usize % 8192, 0);
    assert_eq!(block.align(), 8192);
    assert!(tracked.stats().current_bytes > block.size());
    unsafe { aligned.deallocate(block, &stream)? };
    assert_eq!(tracked.stats().current_bytes, 0);

    let thresholded = AlignedMemoryResource::new(&tracked, 4096, 65_536)?;
    let small = thresholded.allocate(Layout::from_size_align(64, 8)?, &stream)?;
    assert_eq!(tracked.stats().current_bytes, 64);
    unsafe { thresholded.deallocate(small, &stream)? };
    stream.synchronize()?;
    Ok(())
}

#[test]
fn aligned_resource_is_thread_safe() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let upstream = CudaAsyncMemoryResource::new(context.clone());
    let aligned = AlignedMemoryResource::new(upstream, 4096, 0)?;
    let streams = (0..4)
        .map(|_| context.new_stream())
        .collect::<Result<Vec<_>, _>>()?;
    std::thread::scope(|scope| -> Result<(), MemoryError> {
        let threads = streams
            .into_iter()
            .map(|stream| {
                let resource = &aligned;
                scope.spawn(move || -> Result<(), MemoryError> {
                    for size in 1..=64 {
                        let block = resource.allocate(
                            Layout::from_size_align(size * 37, 4096).expect("valid layout"),
                            &stream,
                        )?;
                        if !(block.cu_deviceptr() as usize).is_multiple_of(4096) {
                            return Err(MemoryError::InvalidAlignment { requested: 4096 });
                        }
                        unsafe { resource.deallocate(block, &stream)? };
                    }
                    stream.synchronize()?;
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().expect("allocation thread panicked")?;
        }
        Ok(())
    })?;
    Ok(())
}

#[test]
fn statistics_resource_tracks_nested_scopes() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let upstream = CudaAsyncMemoryResource::new(context);
    let statistics = StatisticsMemoryResource::new(upstream);
    let layout = Layout::from_size_align(100, 4)?;

    let outer = statistics.allocate(layout, &stream)?;
    assert_eq!(statistics.snapshot().bytes.value, 100);
    assert_eq!(statistics.push_counters().bytes.value, 100);
    let inner = statistics.allocate(layout, &stream)?;
    statistics.push_counters();
    let temporary = statistics.allocate(layout, &stream)?;
    unsafe { statistics.deallocate(temporary, &stream)? };
    let temporary_scope = statistics.pop_counters()?;
    assert_eq!(temporary_scope.bytes.value, 0);
    assert_eq!(temporary_scope.bytes.peak, 100);
    assert_eq!(temporary_scope.bytes.total, 100);
    assert_eq!(temporary_scope.allocations.peak, 1);

    unsafe { statistics.deallocate(outer, &stream)? };
    let inner_scope = statistics.pop_counters()?;
    assert_eq!(inner_scope.bytes.value, 0);
    assert_eq!(inner_scope.bytes.peak, 100);
    unsafe { statistics.deallocate(inner, &stream)? };
    assert_eq!(statistics.snapshot().bytes.value, 0);
    assert!(matches!(
        statistics.pop_counters(),
        Err(MemoryError::CounterStackUnderflow)
    ));
    stream.synchronize()?;
    Ok(())
}

#[test]
fn coalescing_pool_keeps_upstream_chunks_separate() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let upstream = CudaAsyncMemoryResource::new(context);
    let pool = PoolMemoryResource::new(upstream, 0, 3 << 20)?;
    let first = pool.allocate(Layout::from_size_align(1 << 20, 256)?, &stream)?;
    let second = pool.allocate(Layout::from_size_align(2 << 20, 256)?, &stream)?;
    assert_eq!(pool.stats().upstream_allocations, 2);
    unsafe {
        pool.deallocate(first, &stream)?;
        pool.deallocate(second, &stream)?;
    }
    assert_eq!(pool.release_unused(&stream)?, 3 << 20);
    assert_eq!(pool.stats().upstream_allocations, 0);
    stream.synchronize()?;
    Ok(())
}

#[test]
fn coalescing_pool_reuses_after_deallocation_stream_is_destroyed() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let upstream = CudaAsyncMemoryResource::new(context.clone());
    let pool = PoolMemoryResource::new(upstream, 1024, 1024)?;
    let layout = Layout::from_size_align(256, 8)?;
    let first_stream = context.new_stream()?;
    let first = pool.allocate(layout, &first_stream)?;
    let first_ptr = first.cu_deviceptr();
    unsafe { pool.deallocate(first, &first_stream)? };
    drop(first_stream);

    let second_stream = context.new_stream()?;
    let reused = pool.allocate(layout, &second_stream)?;
    assert_eq!(reused.cu_deviceptr(), first_ptr);
    unsafe { pool.deallocate(reused, &second_stream)? };
    second_stream.synchronize()?;
    Ok(())
}

#[test]
fn buffer_teardown_rebinds_its_device_context() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(first_context) = context_or_skip() else {
        return Ok(());
    };
    let Ok(second_context) = cuda_core::CudaContext::new(1) else {
        eprintln!("SKIP: CUDA device 1 is unavailable");
        return Ok(());
    };
    let first_stream = first_context.default_stream();
    let resource = CudaAsyncMemoryResource::new(first_context);
    let buffer = DeviceBuffer::<u32, _>::zeroed_in(resource, &first_stream, 1024)?;
    second_context.bind_to_thread()?;
    drop(buffer);
    first_stream.synchronize()?;
    Ok(())
}

#[test]
fn coalescing_pool_is_thread_safe_under_multi_stream_load() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let upstream = CudaAsyncMemoryResource::new(context.clone());
    let pool = PoolMemoryResource::new(upstream, 4 << 20, 32 << 20)?;
    let streams = (0..4)
        .map(|_| context.new_stream())
        .collect::<Result<Vec<_>, _>>()?;
    std::thread::scope(|scope| -> Result<(), MemoryError> {
        let threads = streams
            .into_iter()
            .enumerate()
            .map(|(thread_index, stream)| {
                let pool = &pool;
                scope.spawn(move || -> Result<(), MemoryError> {
                    for iteration in 0..500 {
                        let size = 1 + ((iteration * 7919 + thread_index * 104729) % 65_536);
                        let block = pool.allocate(
                            Layout::from_size_align(size, 8).expect("valid layout"),
                            &stream,
                        )?;
                        unsafe {
                            cuda_core::memory::memset_d8_async(
                                block.cu_deviceptr(),
                                thread_index as u8,
                                block.size(),
                                stream.cu_stream(),
                            )?;
                            pool.deallocate(block, &stream)?;
                        }
                    }
                    stream.synchronize()?;
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().expect("pool stress thread panicked")?;
        }
        Ok(())
    })?;
    assert_eq!(pool.stats().live_allocations, 0);
    Ok(())
}

#[test]
fn fixed_size_pool_reuses_across_destroyed_stream() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let upstream = CudaAsyncMemoryResource::new(context.clone());
    let fixed = FixedSizeMemoryResource::new(upstream, 256, 1)?;
    let layout = Layout::from_size_align(100, 4)?;
    let first_stream = context.new_stream()?;
    let first = fixed.allocate(layout, &first_stream)?;
    let first_ptr = first.cu_deviceptr();
    unsafe { fixed.deallocate(first, &first_stream)? };
    drop(first_stream);

    let second_stream = context.new_stream()?;
    let reused = fixed.allocate(layout, &second_stream)?;
    assert_eq!(reused.cu_deviceptr(), first_ptr);
    unsafe { fixed.deallocate(reused, &second_stream)? };
    second_stream.synchronize()?;
    Ok(())
}

#[test]
fn pinned_host_resource_is_visible_to_host_and_device() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let pinned = PinnedHostMemoryResource::new(context);
    assert!(pinned.properties().host_accessible);
    assert!(pinned.properties().device_accessible);
    let block = pinned.allocate(Layout::from_size_align(4096, 256)?, &stream)?;
    unsafe {
        std::ptr::write_bytes(block.host_ptr(), 0x11, block.size());
        cuda_core::memory::memset_d8_async(
            block.cu_deviceptr(),
            0x2A,
            block.size(),
            stream.cu_stream(),
        )?;
    }
    stream.synchronize()?;
    let bytes = unsafe { std::slice::from_raw_parts(block.host_ptr().cast::<u8>(), block.size()) };
    assert!(bytes.iter().all(|byte| *byte == 0x2A));
    unsafe { pinned.deallocate(block, &stream)? };
    Ok(())
}

#[test]
fn system_memory_resource_is_visible_to_host_and_device_when_supported()
-> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    if !SystemMemoryResource::is_supported(&context)? {
        eprintln!("SKIP: CUDA pageable system memory access is unavailable");
        return Ok(());
    }
    let stream = context.default_stream();
    let system = SystemMemoryResource::new(context)?;
    assert!(system.properties().host_accessible);
    assert!(system.properties().device_accessible);
    let block = system.allocate(Layout::from_size_align(4096, 256)?, &stream)?;
    assert_eq!(block.cu_deviceptr() as usize % 256, 0);
    unsafe {
        std::ptr::write_bytes(block.host_ptr(), 0x11, block.size());
        cuda_core::memory::memset_d8_async(
            block.cu_deviceptr(),
            0x3A,
            block.size(),
            stream.cu_stream(),
        )?;
    }
    stream.synchronize()?;
    let bytes = unsafe { std::slice::from_raw_parts(block.host_ptr().cast::<u8>(), block.size()) };
    assert!(bytes.iter().all(|byte| *byte == 0x3A));
    unsafe { system.deallocate(block, &stream)? };
    Ok(())
}

#[test]
fn sam_headroom_resource_preserves_reserved_device_capacity() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    if !SystemMemoryResource::is_supported(&context)? {
        eprintln!("SKIP: CUDA pageable system memory access is unavailable");
        return Ok(());
    }
    let stream = context.default_stream();
    let mut free = MaybeUninit::<usize>::uninit();
    let mut total = MaybeUninit::<usize>::uninit();
    unsafe { cuda_core::sys::cuMemGetInfo_v2(free.as_mut_ptr(), total.as_mut_ptr()).result()? };
    let headroom = unsafe { free.assume_init() }.saturating_add(1 << 20);
    let system = SamHeadroomMemoryResource::new(context, headroom)?;
    assert_eq!(system.headroom(), headroom);
    let block = system.allocate(Layout::from_size_align(4096, 256)?, &stream)?;
    unsafe {
        std::ptr::write_bytes(block.host_ptr(), 0x27, block.size());
        cuda_core::memory::memset_d8_async(
            block.cu_deviceptr(),
            0x4B,
            block.size(),
            stream.cu_stream(),
        )?;
    }
    stream.synchronize()?;
    let bytes = unsafe { std::slice::from_raw_parts(block.host_ptr().cast::<u8>(), block.size()) };
    assert!(bytes.iter().all(|byte| *byte == 0x4B));
    unsafe { system.deallocate(block, &stream)? };
    Ok(())
}

#[test]
fn coalescing_pool_supports_pinned_host_upstream() -> Result<(), Box<dyn Error>> {
    let _guard = gpu_test_lock();
    let Some(context) = context_or_skip() else {
        return Ok(());
    };
    let stream = context.default_stream();
    let upstream = PinnedHostMemoryResource::new(context);
    let pool = PoolMemoryResource::new(upstream, 4096, 4096)?;
    assert!(pool.properties().host_accessible);
    let block = pool.allocate(Layout::from_size_align(512, 8)?, &stream)?;
    unsafe { std::ptr::write_bytes(block.host_ptr(), 0x5C, block.size()) };
    let bytes = unsafe { std::slice::from_raw_parts(block.host_ptr().cast::<u8>(), block.size()) };
    assert!(bytes.iter().all(|byte| *byte == 0x5C));
    unsafe { pool.deallocate(block, &stream)? };
    assert_eq!(pool.release_unused(&stream)?, 4096);
    Ok(())
}
