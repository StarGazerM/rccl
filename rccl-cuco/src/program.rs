use cuda_core::{CudaContext, CudaStream};
use rccl_memory::MemoryError;

/// One macro-generated cuCollections kernel program bound to a CUDA stream.
#[derive(Clone, Copy)]
pub struct Launch<'stream, 'program, M> {
    stream: &'stream CudaStream,
    module: &'program M,
}

impl<'stream, M> Launch<'stream, '_, M> {
    pub fn stream(&self) -> &'stream CudaStream {
        self.stream
    }

    #[doc(hidden)]
    pub fn module(&self) -> &M {
        self.module
    }
}

/// Runtime owner for one macro-generated cuCollections program.
pub struct Program<'context, M> {
    context: &'context CudaContext,
    module: M,
}

impl<'context, M> Program<'context, M> {
    #[doc(hidden)]
    pub fn __from_module(context: &'context CudaContext, module: M) -> Self {
        Self { context, module }
    }

    pub fn on<'stream, 'program>(
        &'program self,
        stream: &'stream CudaStream,
    ) -> Result<Launch<'stream, 'program, M>, MemoryError> {
        if self.context != stream.context().as_ref() {
            return Err(MemoryError::ContextMismatch);
        }
        Ok(Launch {
            stream,
            module: &self.module,
        })
    }
}
