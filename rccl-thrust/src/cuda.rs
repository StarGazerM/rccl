use cuda_core::CudaStream;
use rccl_memory::{DeviceAllocator, MemoryError};
use std::{error::Error, fmt};

/// A borrowed CUDA stream and allocator with no compiler policy.
#[derive(Clone, Copy)]
pub struct Execution<'stream, 'allocator, A: DeviceAllocator + ?Sized> {
    stream: &'stream CudaStream,
    allocator: &'allocator A,
}

impl<A: DeviceAllocator + ?Sized> Execution<'_, '_, A> {
    pub fn stream(&self) -> &CudaStream {
        self.stream
    }

    pub fn allocator(&self) -> &A {
        self.allocator
    }
}

#[derive(Debug)]
pub struct ExecutionError(MemoryError);

impl fmt::Display for ExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Error for ExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.0)
    }
}

pub fn par<'stream, 'allocator, A: DeviceAllocator + ?Sized>(
    stream: &'stream CudaStream,
    allocator: &'allocator A,
) -> Result<Execution<'stream, 'allocator, A>, ExecutionError> {
    if allocator.context().as_ref() != stream.context().as_ref() {
        return Err(ExecutionError(MemoryError::ContextMismatch));
    }
    Ok(Execution { stream, allocator })
}
