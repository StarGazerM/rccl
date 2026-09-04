use cuda_core::{CudaEvent as CoreCudaEvent, CudaStream, DriverError};

use crate::MemoryError;

/// In-flight CUDA work together with every resource borrowed by its submission.
///
/// `CudaCompletion` is a completion guard, not a Rust [`Future`]. The `resources`
/// value is retained until the stream has synchronized, so operation surfaces
/// can put mutable outputs, inputs, callable state, programs, and workspaces in
/// one affine guard without allocation or type erasure.
///
/// Submission errors are retained too: [`wait`](Self::wait) first makes the
/// stream safe to release and then returns the original operation result. If
/// the operation was submitted successfully, a synchronization error is
/// returned instead.
#[must_use = "pending CUDA work synchronizes when dropped; call wait to observe its result"]
pub struct CudaCompletion<'stream, T, E, Resources> {
    stream: &'stream CudaStream,
    result: Option<Result<T, E>>,
    resources: Option<Resources>,
    synchronize: bool,
}

impl<'stream, T, E, Resources> CudaCompletion<'stream, T, E, Resources> {
    /// Build a tracked completion while taking ownership of all resources
    /// borrowed by an already-defined submission.
    ///
    /// # Safety
    ///
    /// `submit` must enqueue all GPU access through `stream`, and every
    /// allocation or host-side state the enqueued work can access must be
    /// retained by `resources`. The closure must not capture an omitted
    /// resource used by the GPU work.
    pub unsafe fn from_submission<F>(
        stream: &'stream CudaStream,
        mut resources: Resources,
        submit: F,
    ) -> Self
    where
        F: FnOnce(&mut Resources) -> Result<T, E>,
    {
        let result = submit(&mut resources);
        Self {
            stream,
            result: Some(result),
            resources: Some(resources),
            synchronize: true,
        }
    }

    /// Construct an already-completed guard while preserving the same resource
    /// and result shape as a submitted operation.
    ///
    /// # Safety
    ///
    /// No outstanding GPU work may access `resources`.
    pub unsafe fn ready(
        stream: &'stream CudaStream,
        resources: Resources,
        result: Result<T, E>,
    ) -> Self {
        Self {
            stream,
            result: Some(result),
            resources: Some(resources),
            synchronize: false,
        }
    }

    /// Return the stream on which this operation is pending.
    pub fn stream(&self) -> &'stream CudaStream {
        self.stream
    }

    /// Query whether all work currently preceding the completion point has
    /// finished without releasing any retained resource.
    pub fn is_complete(&self) -> Result<bool, DriverError> {
        if self.synchronize {
            self.stream.query()
        } else {
            Ok(true)
        }
    }
}

impl<T, E, Resources> CudaCompletion<'_, T, E, Resources>
where
    E: From<DriverError>,
{
    /// Synchronize the operation stream, release retained resources, and return
    /// the operation result.
    ///
    /// An error produced while submitting the operation takes precedence over
    /// a later synchronization error, matching ordinary `Result` sequencing.
    pub fn wait(mut self) -> Result<T, E> {
        let synchronized = if self.synchronize {
            self.stream.synchronize().map_err(E::from)
        } else {
            Ok(())
        };
        self.synchronize = false;
        let result = self
            .result
            .take()
            .expect("CUDA completion result is present until wait");
        self.resources.take();
        match result {
            Ok(value) => synchronized.map(|()| value),
            Err(error) => {
                let _ = synchronized;
                Err(error)
            }
        }
    }
}

impl<T, E, Resources> Drop for CudaCompletion<'_, T, E, Resources> {
    fn drop(&mut self) {
        if self.synchronize {
            self.stream.context().record_err(self.stream.synchronize());
        }
    }
}

/// An owned, already-recorded CUDA event with no raw-handle escape hatch.
///
/// The wrapped cuda-core event keeps its CUDA context alive. Dropping this
/// value after [`wait_on`](Self::wait_on) is safe: CUDA retains the event state
/// needed by an already-enqueued stream wait.
///
/// Raw CUDA handles are intentionally unavailable:
///
/// ```compile_fail
/// use rccl_memory::RecordedCudaEvent;
///
/// fn raw_handle(event: &RecordedCudaEvent) {
///     let _ = event.cu_event();
/// }
/// ```
#[derive(Debug)]
pub struct RecordedCudaEvent {
    event: CoreCudaEvent,
}

impl RecordedCudaEvent {
    /// Record a new timing-disabled event after all work currently enqueued on
    /// `stream`.
    pub fn record(stream: &CudaStream) -> Result<Self, DriverError> {
        Ok(Self {
            event: stream.record_event(None)?,
        })
    }

    /// Enqueue a dependency on this event in `stream`.
    ///
    /// Subsequent work in `stream` observes all work preceding this event's
    /// record point. Both streams must belong to the same CUDA context.
    pub fn wait_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
        if self.event.context().as_ref() != stream.context().as_ref() {
            return Err(MemoryError::ContextMismatch);
        }
        stream.wait(&self.event).map_err(MemoryError::Driver)
    }

    /// Block the calling thread until the event's record point completes.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.event.synchronize()
    }

    /// Query the record point without blocking.
    pub fn is_complete(&self) -> Result<bool, DriverError> {
        self.event.query()
    }
}

/// Record prior work in `producer` and make subsequent `consumer` work depend
/// on it.
///
/// The returned owned event may be retained for observation or dropped as soon
/// as the wait has been enqueued. No stream-wide host synchronization occurs.
pub fn cross_stream_dependency(
    producer: &CudaStream,
    consumer: &CudaStream,
) -> Result<RecordedCudaEvent, MemoryError> {
    if producer.context().as_ref() != consumer.context().as_ref() {
        return Err(MemoryError::ContextMismatch);
    }
    let event = RecordedCudaEvent::record(producer).map_err(MemoryError::Driver)?;
    event.wait_on(consumer)?;
    Ok(event)
}
