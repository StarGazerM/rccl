use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use rccl_memory::{BorrowedCudaMemoryResource, DeviceAllocator, MemoryError};
use std::{error::Error, fmt, sync::Arc};

/// A synchronous Thrust execution policy borrowing one stream and the fixed
/// RCCL kernel program.
pub struct Execution<'stream, 'program, A: DeviceAllocator> {
    launch: rccl_cub::cuda::Launch<'stream, 'program>,
    allocator: A,
}

impl<A: DeviceAllocator> Execution<'_, '_, A> {
    pub fn stream(&self) -> &CudaStream {
        self.launch.stream()
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    pub(crate) fn launch(&self) -> &rccl_cub::cuda::Launch<'_, '_> {
        &self.launch
    }

    pub(crate) fn transform_repeat_in_place<T, F>(
        &self,
        output: &mut DeviceBuffer<T>,
        repeated: T,
        op: F,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        T: DeviceCopy,
        F: Fn(T, T) -> T + Copy,
    {
        rccl_cub::thrust_integration::transform_repeat_in_place(&self.launch, output, repeated, op)
    }

    pub(crate) fn synchronize(
        &self,
        result: Result<(), crate::DeviceAlgorithmError>,
    ) -> Result<(), crate::DeviceAlgorithmError> {
        match result {
            Ok(()) => self
                .stream()
                .synchronize()
                .map_err(crate::DeviceAlgorithmError::Driver),
            Err(error) => {
                let _ = self.stream().synchronize();
                Err(error)
            }
        }
    }
}

/// Runtime owner for the fixed Thrust/CUB kernel program.
pub struct Program<'context> {
    cub: rccl_cub::cuda::Program<'context>,
}

impl<'context> Program<'context> {
    pub fn load(
        context: &'context Arc<CudaContext>,
    ) -> Result<Self, cuda_host::EmbeddedModuleError> {
        Ok(Self {
            cub: rccl_cub::cuda::Program::load(context)?,
        })
    }

    pub fn par<'stream, 'program>(
        &'program self,
        stream: &'stream CudaStream,
    ) -> Result<Execution<'stream, 'program, BorrowedCudaMemoryResource<'stream>>, ExecutionError>
    {
        let allocator =
            BorrowedCudaMemoryResource::new(stream.context()).map_err(ExecutionError)?;
        self.par_with(stream, allocator)
    }

    pub fn par_in<'stream, 'program, 'allocator, A: DeviceAllocator + ?Sized>(
        &'program self,
        stream: &'stream CudaStream,
        allocator: &'allocator A,
    ) -> Result<Execution<'stream, 'program, &'allocator A>, ExecutionError> {
        self.par_with(stream, allocator)
    }

    fn par_with<'stream, 'program, A: DeviceAllocator>(
        &'program self,
        stream: &'stream CudaStream,
        allocator: A,
    ) -> Result<Execution<'stream, 'program, A>, ExecutionError> {
        if allocator.context().as_ref() != stream.context().as_ref() {
            return Err(ExecutionError(MemoryError::ContextMismatch));
        }
        let launch = self.cub.on(stream).map_err(ExecutionError)?;
        Ok(Execution { launch, allocator })
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
