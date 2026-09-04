use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use rccl_memory::{BorrowedCudaMemoryResource, CudaCompletion, DeviceAllocator, MemoryError};
use std::{error::Error, fmt, sync::Arc};

/// A stream-bound Thrust execution context borrowing the fixed RCCL kernel
/// program.
pub struct Execution<'stream, 'program, A: DeviceAllocator> {
    launch: rccl_cub::cuda::Launch<'stream, 'program>,
    allocator: A,
}

impl<'stream, 'program, A: DeviceAllocator> Execution<'stream, 'program, A> {
    pub fn stream(&self) -> &'stream CudaStream {
        self.launch.stream()
    }

    pub fn allocator(&self) -> &A {
        &self.allocator
    }

    /// Submit CUDA work while retaining this execution context (including its
    /// program and allocator) together with every operation resource.
    ///
    /// # Safety
    ///
    /// Every resource accessed by the submitted GPU work must be present in
    /// `resources`, and the work must be enqueued only on this execution's
    /// stream.
    pub unsafe fn submit<'borrow, Resources, T, E, F>(
        &'borrow self,
        resources: Resources,
        submit: F,
    ) -> CudaCompletion<'borrow, T, E, (&'borrow Self, Resources)>
    where
        F: FnOnce(&Self, &mut Resources) -> Result<T, E>,
    {
        // SAFETY: the caller supplies the resource-retention contract.
        unsafe {
            CudaCompletion::from_submission(self.stream(), (self, resources), |retained| {
                submit(retained.0, &mut retained.1)
            })
        }
    }

    pub(crate) fn launch(&self) -> &rccl_cub::cuda::Launch<'stream, 'program> {
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
