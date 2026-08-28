use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use rccl_memory::{CudaMemoryResource, DeviceAllocator, MemoryError};
use std::{error::Error, fmt};

/// Allocates the concrete temporary buffers accepted by cuda-oxide's checked
/// synchronous kernel launch surface.
///
/// This is a runtime allocation capability, not an algorithm policy or a TMP
/// selection point. The current cuda-oxide launch API accepts its own owning
/// `DeviceBuffer` for synchronous slice arguments, so only the matching direct
/// CUDA resource implements the explicit-resource form.
pub trait TemporaryAllocator {
    #[doc(hidden)]
    fn usable_on(&self, stream: &CudaStream) -> bool;

    #[doc(hidden)]
    fn temporary<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        len: usize,
    ) -> Result<DeviceBuffer<T>, MemoryError>;
}

impl TemporaryAllocator for CudaMemoryResource {
    fn usable_on(&self, stream: &CudaStream) -> bool {
        self.context().as_ref() == stream.context().as_ref()
    }

    fn temporary<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        len: usize,
    ) -> Result<DeviceBuffer<T>, MemoryError> {
        DeviceBuffer::zeroed(stream, len).map_err(MemoryError::Driver)
    }
}

impl<T: TemporaryAllocator + ?Sized> TemporaryAllocator for &T {
    fn usable_on(&self, stream: &CudaStream) -> bool {
        (**self).usable_on(stream)
    }

    fn temporary<U: DeviceCopy>(
        &self,
        stream: &CudaStream,
        len: usize,
    ) -> Result<DeviceBuffer<U>, MemoryError> {
        (**self).temporary(stream, len)
    }
}

/// The allocation behavior of ordinary `cuda::par`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultTemporaryAllocator;

impl TemporaryAllocator for DefaultTemporaryAllocator {
    fn usable_on(&self, _stream: &CudaStream) -> bool {
        true
    }

    fn temporary<T: DeviceCopy>(
        &self,
        stream: &CudaStream,
        len: usize,
    ) -> Result<DeviceBuffer<T>, MemoryError> {
        DeviceBuffer::zeroed(stream, len).map_err(MemoryError::Driver)
    }
}

/// A synchronous Thrust execution policy borrowing one stream and program.
#[derive(Clone, Copy)]
pub struct Execution<'stream, 'program, A: TemporaryAllocator, M> {
    stream: &'stream CudaStream,
    allocator: A,
    module: &'program M,
}

impl<A: TemporaryAllocator, M> Execution<'_, '_, A, M> {
    pub fn stream(&self) -> &CudaStream {
        self.stream
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
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

/// Runtime owner for one macro-generated Thrust/CUB program.
pub struct Program<'context, M> {
    context: &'context CudaContext,
    module: M,
}

impl<'context, M> Program<'context, M> {
    #[doc(hidden)]
    pub fn __from_module(context: &'context CudaContext, module: M) -> Self {
        Self { context, module }
    }

    /// Construct the ordinary synchronous CUDA execution policy.
    pub fn par<'stream, 'program>(
        &'program self,
        stream: &'stream CudaStream,
    ) -> Result<Execution<'stream, 'program, DefaultTemporaryAllocator, M>, ExecutionError> {
        self.par_with(stream, DefaultTemporaryAllocator)
    }

    /// Construct a synchronous CUDA execution policy using a caller resource.
    pub fn par_in<'stream, 'program, 'allocator, A: TemporaryAllocator + ?Sized>(
        &'program self,
        stream: &'stream CudaStream,
        allocator: &'allocator A,
    ) -> Result<Execution<'stream, 'program, &'allocator A, M>, ExecutionError> {
        self.par_with(stream, allocator)
    }

    fn par_with<'stream, 'program, A: TemporaryAllocator>(
        &'program self,
        stream: &'stream CudaStream,
        allocator: A,
    ) -> Result<Execution<'stream, 'program, A, M>, ExecutionError> {
        if self.context != stream.context().as_ref() || !allocator.usable_on(stream) {
            return Err(ExecutionError(MemoryError::ContextMismatch));
        }
        Ok(Execution {
            stream,
            allocator,
            module: &self.module,
        })
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
