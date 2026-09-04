//! Focused runtime probe for pending completion and cross-stream dependencies.

use cuda_core::{CudaContext, DeviceBuffer, LaunchContractError};
use rccl_memory::{CudaCompletion, cross_stream_dependency};
use rccl_thrust::{DeviceAlgorithmError, DeviceGenerateExt, cuda::Program};
use std::{
    error::Error,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    thread,
    time::Duration,
};

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let producer = context.new_stream()?;
    let consumer = context.new_stream()?;
    let program = Program::load(&context)?;
    let execution = program.par(&producer)?;

    let mut waited = DeviceBuffer::<u32>::zeroed(&producer, 1 << 20)?;
    waited.fill_device_async(&execution, 0x1234_5678).wait()?;
    assert_eq!(waited.to_host_vec(&producer)?, vec![0x1234_5678; 1 << 20]);

    let mut dropped = DeviceBuffer::<u32>::zeroed(&producer, 1 << 20)?;
    drop(dropped.fill_device_async(&execution, 0xa5a5_5a5a));
    assert!(
        producer.query()?,
        "dropping a pending fill must synchronize"
    );
    assert_eq!(dropped.to_host_vec(&producer)?, vec![0xa5a5_5a5a; 1 << 20]);

    // SAFETY: the injected launch-contract error enqueues no GPU work and
    // therefore borrows no resource beyond the retained unit value.
    let failed = unsafe {
        CudaCompletion::from_submission(&producer, (), |_| {
            Err::<(), DeviceAlgorithmError>(DeviceAlgorithmError::Launch(
                LaunchContractError::EmptyKernelName,
            ))
        })
    };
    assert!(matches!(
        failed.wait(),
        Err(DeviceAlgorithmError::Launch(
            LaunchContractError::EmptyKernelName
        ))
    ));

    let ordering = Arc::new(AtomicU32::new(0));
    let producer_ordering = ordering.clone();
    producer.launch_host_function(move || {
        thread::sleep(Duration::from_millis(25));
        producer_ordering.store(1, Ordering::Release);
    })?;

    let event = cross_stream_dependency(&producer, &consumer)?;
    assert!(!event.is_complete()? || ordering.load(Ordering::Acquire) == 1);
    let consumer_ordering = ordering.clone();
    consumer.launch_host_function(move || {
        let observed = consumer_ordering.load(Ordering::Acquire);
        consumer_ordering.store(if observed == 1 { 2 } else { 3 }, Ordering::Release);
    })?;

    // CUDA keeps an already-enqueued wait valid after event destruction.
    drop(event);
    consumer.synchronize()?;
    assert_eq!(ordering.load(Ordering::Acquire), 2);

    println!("runtime safety: PASS");
    Ok(())
}
