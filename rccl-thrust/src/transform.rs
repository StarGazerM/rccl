use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_memory::DeviceAllocator;

use crate::{DeviceRange, TransformError, cuda::Execution, range::RepeatN};

/// Apply `op` between each output item and one repeated value.
///
/// This is the Rust equivalent of transforming with a finite Thrust constant
/// iterator. The range length is checked before the kernel is launched.
pub fn transform<T, F, A>(
    execution: &Execution<'_, '_, A>,
    output: &mut DeviceBuffer<T>,
    repeated: RepeatN<T>,
    op: F,
) -> Result<(), TransformError>
where
    T: DeviceCopy,
    F: Fn(T, T) -> T + Copy,
    A: DeviceAllocator,
{
    let input_len = DeviceRange::len(&repeated);
    let output_len = output.len();
    if input_len != output_len {
        return Err(TransformError::LengthMismatch {
            input: input_len,
            output: output_len,
        });
    }
    execution
        .transform_repeat_in_place(output, *repeated.value(), op)
        .map_err(TransformError::Launch)
}
