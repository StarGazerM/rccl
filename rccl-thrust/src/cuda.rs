use cuda_core::{CudaContext, CudaStream};
use rccl_memory::{BorrowedCudaMemoryResource, DeviceAllocator, MemoryError};
use std::{error::Error, fmt};

/// A synchronous Thrust execution policy borrowing one stream and program.
pub struct Execution<'stream, 'program, A: DeviceAllocator, M> {
    stream: &'stream CudaStream,
    allocator: A,
    module: &'program M,
}

impl<A: DeviceAllocator, M> Execution<'_, '_, A, M> {
    pub fn stream(&self) -> &CudaStream {
        self.stream
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    #[doc(hidden)]
    pub fn module(&self) -> &M {
        self.module
    }

    /// Apply the completion point required by synchronous Thrust `par`.
    ///
    /// Each lowering places this call at the same temporary-array lifetime
    /// boundary as pinned Thrust: merge and merge-sort synchronize before
    /// releasing storage, while radix sort releases its inner temporary array
    /// before the enclosing smart-sort synchronization.
    #[doc(hidden)]
    pub fn __synchronize(
        &self,
        result: Result<(), crate::DeviceAlgorithmError>,
    ) -> Result<(), crate::DeviceAlgorithmError> {
        result?;
        self.stream
            .synchronize()
            .map_err(crate::DeviceAlgorithmError::Driver)
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
    ) -> Result<Execution<'stream, 'program, BorrowedCudaMemoryResource<'stream>, M>, ExecutionError>
    {
        let allocator =
            BorrowedCudaMemoryResource::new(stream.context()).map_err(ExecutionError)?;
        self.par_with(stream, allocator)
    }

    /// Construct a synchronous CUDA execution policy using a caller resource.
    pub fn par_in<'stream, 'program, 'allocator, A: DeviceAllocator + ?Sized>(
        &'program self,
        stream: &'stream CudaStream,
        allocator: &'allocator A,
    ) -> Result<Execution<'stream, 'program, &'allocator A, M>, ExecutionError> {
        self.par_with(stream, allocator)
    }

    fn par_with<'stream, 'program, A: DeviceAllocator>(
        &'program self,
        stream: &'stream CudaStream,
        allocator: A,
    ) -> Result<Execution<'stream, 'program, A, M>, ExecutionError> {
        if self.context != stream.context().as_ref()
            || allocator.context().as_ref() != stream.context().as_ref()
        {
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
