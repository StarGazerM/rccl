/// CUDA program ownership and stream binding for direct CUB operations.
///
/// Architecture policy and kernel selection remain entirely in the proc
/// macro. Neither [`cuda::Program`] nor [`cuda::Launch`] stores compiler
/// metadata or a runtime registry.
use cuda_core::{CudaContext, CudaStream};
use rccl_memory::MemoryError;

#[derive(Clone, Copy)]
pub struct Launch<'stream, 'program, M> {
    stream: &'stream CudaStream,
    module: &'program M,
}

impl<M> Launch<'_, '_, M> {
    pub fn stream(&self) -> &CudaStream {
        self.stream
    }

    #[doc(hidden)]
    pub fn module(&self) -> &M {
        self.module
    }
}

/// Runtime owner for one macro-generated CUB program.
pub struct Program<'context, M> {
    context: &'context CudaContext,
    module: M,
}

impl<'context, M> Program<'context, M> {
    #[doc(hidden)]
    pub fn __from_module(context: &'context CudaContext, module: M) -> Self {
        Self { context, module }
    }

    /// Bind this program to the stream used by a direct CUB call.
    pub fn on<'program, 'stream>(
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
