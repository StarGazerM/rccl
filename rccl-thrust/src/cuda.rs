use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
use rccl_memory::{CudaMemoryResource, DeviceAllocator, MemoryError};
use std::{error::Error, fmt};

/// Allocates the concrete temporary buffers accepted by cuda-oxide's checked
/// synchronous kernel launch surface.
///
/// This is a runtime allocation capability, not an algorithm policy or a TMP
/// selection point. The current cuda-oxide launch API accepts its own owning
/// `DeviceBuffer` for synchronous slice arguments, so only the matching direct
/// CUDA resource implements this capability.
pub trait TemporaryAllocator: DeviceAllocator {
    #[doc(hidden)]
    fn temporary<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        len: usize,
    ) -> Result<DeviceBuffer<T>, MemoryError>;
}

impl TemporaryAllocator for CudaMemoryResource {
    fn temporary<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        len: usize,
    ) -> Result<DeviceBuffer<T>, MemoryError> {
        DeviceBuffer::zeroed(stream, len).map_err(MemoryError::Driver)
    }
}

/// A borrowed CUDA stream and allocator with no compiler policy.
#[derive(Clone, Copy)]
pub struct Execution<'stream, 'allocator, 'module, A: TemporaryAllocator + ?Sized, M> {
    stream: &'stream CudaStream,
    allocator: &'allocator A,
    module: &'module M,
}

impl<A: TemporaryAllocator + ?Sized, M> Execution<'_, '_, '_, A, M> {
    pub fn stream(&self) -> &CudaStream {
        self.stream
    }

    pub fn allocator(&self) -> &A {
        self.allocator
    }

    #[doc(hidden)]
    pub fn temporary<T: DeviceCopy>(&self, len: usize) -> Result<DeviceBuffer<T>, MemoryError> {
        self.allocator.temporary(self.stream, len)
    }

    #[doc(hidden)]
    pub fn module(&self) -> &M {
        self.module
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

pub fn par<'stream, 'allocator, 'module, A: TemporaryAllocator + ?Sized, M>(
    stream: &'stream CudaStream,
    allocator: &'allocator A,
    module: &'module M,
) -> Result<Execution<'stream, 'allocator, 'module, A, M>, ExecutionError> {
    if allocator.context().as_ref() != stream.context().as_ref() {
        return Err(ExecutionError(MemoryError::ContextMismatch));
    }
    Ok(Execution {
        stream,
        allocator,
        module,
    })
}
