//! CUDA module ownership and stream binding for direct CUB operations.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use rccl_memory::MemoryError;

use crate::{histogram_kernels, ordering_kernels, set_kernels};

/// The fixed RCCL kernel set compiled for cargo-oxide's selected target.
///
/// This is a runtime resource owner, not an algorithm or architecture policy.
/// Its concrete type keeps module implementation details out of operation
/// signatures.
pub(crate) struct KernelModules {
    histogram: histogram_kernels::HistogramModule,
    ordering: ordering_kernels::OrderingModule,
    set: set_kernels::SetModule,
}

impl KernelModules {
    fn load(context: &Arc<CudaContext>) -> Result<Self, cuda_host::EmbeddedModuleError> {
        Ok(Self {
            histogram: histogram_kernels::load(context)?,
            ordering: ordering_kernels::load(context)?,
            set: set_kernels::load(context)?,
        })
    }

    pub(crate) fn histogram(&self) -> &histogram_kernels::HistogramModule {
        &self.histogram
    }

    pub(crate) fn ordering(&self) -> &ordering_kernels::OrderingModule {
        &self.ordering
    }

    pub(crate) fn set(&self) -> &set_kernels::SetModule {
        &self.set
    }
}

#[derive(Clone, Copy)]
pub struct Launch<'stream, 'program> {
    stream: &'stream CudaStream,
    modules: &'program KernelModules,
}

impl Launch<'_, '_> {
    pub fn stream(&self) -> &CudaStream {
        self.stream
    }

    pub(crate) fn module(&self) -> &KernelModules {
        self.modules
    }

    pub(crate) fn histogram_module(&self) -> &histogram_kernels::HistogramModule {
        self.modules.histogram()
    }

    pub(crate) fn ordering_module(&self) -> &ordering_kernels::OrderingModule {
        self.modules.ordering()
    }

    /// Apply a binary device function between every output item and one
    /// repeated value.
    ///
    /// The operation is an ordinary generic argument. It does not require a
    /// macro because cuda-oxide can monomorphize this method at the call site.
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
        if output.is_empty() {
            return Ok(());
        }
        let launch =
            cuda_core::LaunchConfig1D::new(output.len().div_ceil(8).div_ceil(256) as u32, 256, 0);
        let prepared = self
            .ordering_module()
            .prepare_transform_repeat_in_place_for(&repeated, &op, launch)?;
        self.ordering_module().transform_repeat_in_place(
            self.stream,
            &prepared,
            repeated,
            op,
            output,
        )
    }
}

/// Runtime owner for RCCL's fixed CUB kernel modules.
pub struct Program<'context> {
    context: &'context CudaContext,
    modules: KernelModules,
}

impl<'context> Program<'context> {
    /// Load the CUB kernels compiled for the cargo-oxide target.
    pub fn load(
        context: &'context Arc<CudaContext>,
    ) -> Result<Self, cuda_host::EmbeddedModuleError> {
        Ok(Self {
            context: context.as_ref(),
            modules: KernelModules::load(context)?,
        })
    }

    /// Bind this program to the stream used by a direct CUB call.
    pub fn on<'program, 'stream>(
        &'program self,
        stream: &'stream CudaStream,
    ) -> Result<Launch<'stream, 'program>, MemoryError> {
        if self.context != stream.context().as_ref() {
            return Err(MemoryError::ContextMismatch);
        }
        Ok(Launch {
            stream,
            modules: &self.modules,
        })
    }
}
