//! CUDA execution policies for Thrust host algorithms.

use cuda_core::{CudaStream, DriverError};
use rccl_cub::{KernelContext, KernelContextError};
use rccl_memory::{CudaAsyncMemoryResource, DeviceAllocator, MemoryError};
use std::{error::Error, fmt};

/// Failure while binding a Thrust execution policy to a CUDA stream.
#[derive(Debug)]
#[non_exhaustive]
pub enum ExecutionError {
    Memory(MemoryError),
    Kernel(KernelContextError),
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory(error) => write!(f, "CUDA memory error: {error}"),
            Self::Kernel(error) => error.fmt(f),
        }
    }
}

impl Error for ExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Memory(error) => Some(error),
            Self::Kernel(error) => Some(error),
        }
    }
}

impl From<MemoryError> for ExecutionError {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

impl From<KernelContextError> for ExecutionError {
    fn from(value: KernelContextError) -> Self {
        Self::Kernel(value)
    }
}

/// Thrust host execution policy bound to a borrowed CUDA stream.
///
/// The policy owns no algorithm schedule. CUB owns kernel modules and target
/// policy; Thrust retains only the CUB context handle needed by its algorithms.
pub struct Execution<'stream, A: DeviceAllocator + Clone = CudaAsyncMemoryResource> {
    stream: &'stream CudaStream,
    allocator: A,
    kernels: KernelContext,
}

impl<A: DeviceAllocator + Clone> Execution<'_, A> {
    pub(crate) fn stream(&self) -> &CudaStream {
        self.stream
    }

    pub(crate) fn allocator(&self) -> &A {
        &self.allocator
    }

    pub(crate) fn kernel_context(&self) -> &KernelContext {
        &self.kernels
    }
}

/// Bind Thrust execution to a borrowed stream and the default stream-ordered
/// allocator.
pub fn par(stream: &CudaStream) -> Result<Execution<'_>, ExecutionError> {
    par_in(
        stream,
        CudaAsyncMemoryResource::new(stream.context().clone()),
    )
}

/// Bind Thrust execution to a borrowed stream and allocator handle.
pub fn par_in<A: DeviceAllocator + Clone>(
    stream: &CudaStream,
    allocator: A,
) -> Result<Execution<'_, A>, ExecutionError> {
    if allocator.context().as_ref() != stream.context().as_ref() {
        return Err(MemoryError::ContextMismatch.into());
    }
    Ok(Execution {
        stream,
        allocator,
        kernels: KernelContext::load(stream.context())?,
    })
}

pub(crate) fn finish_synchronous<T, E>(stream: &CudaStream, operation: Result<T, E>) -> Result<T, E>
where
    E: From<DriverError>,
{
    // Always synchronize, including after a later launch reports an error:
    // earlier queued kernels may still access the borrowed ranges.
    let synchronized = stream.synchronize().map_err(E::from);
    match operation {
        Err(error) => {
            let _ = synchronized;
            Err(error)
        }
        Ok(value) => synchronized.map(|()| value),
    }
}
