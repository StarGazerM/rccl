//! Synchronous Rust reduce functions over borrowed device buffers.

use cuda_core::DeviceBuffer;
use rccl_memory::{DeviceAllocator, DeviceScalar, TemporaryValue};

use crate::{DeviceAlgorithmError, cuda::Execution};

pub type Result<T, E = DeviceAlgorithmError> = core::result::Result<T, E>;

/// Reduce a device buffer with Thrust CUDA's default `plus` operation and zero init.
///
/// This is the supported `cub::DeviceReduce::Sum` slice: primitive `u32`, `i32`,
/// and `f32` inputs reduce on the device, then the synchronous Thrust facade
/// reads the one-element device result after stream completion.
pub fn reduce<T, A>(execution: &Execution<'_, '_, A>, input: &DeviceBuffer<T>) -> Result<T>
where
    A: DeviceAllocator,
    T: rccl_cub::ReduceSumValue + TemporaryValue + 'static,
{
    let mut output = DeviceScalar::new_in(execution.allocator(), execution.stream(), T::default())
        .map_err(DeviceAlgorithmError::Memory)?;
    let mut workspace = rccl_cub::device_reduce_sum_workspace_in::<_, T>(
        execution.allocator(),
        execution.launch(),
        input.len(),
    )?;
    let result = rccl_cub::device_reduce_sum_in(
        execution.launch(),
        input,
        output.as_buffer_mut(),
        &mut workspace,
    );
    execution.synchronize(result)?;
    output.get().map_err(DeviceAlgorithmError::Memory)
}

/// Method syntax for Rust-flavored device reductions.
pub trait DeviceReduceExt<T: rccl_cub::ReduceSumValue + TemporaryValue + 'static> {
    fn reduce<A>(&self, execution: &Execution<'_, '_, A>) -> Result<T>
    where
        A: DeviceAllocator;
}

impl<T> DeviceReduceExt<T> for DeviceBuffer<T>
where
    T: rccl_cub::ReduceSumValue + TemporaryValue + 'static,
{
    fn reduce<A>(&self, execution: &Execution<'_, '_, A>) -> Result<T>
    where
        A: DeviceAllocator,
    {
        reduce(execution, self)
    }
}
