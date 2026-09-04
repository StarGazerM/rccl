use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_memory::{CudaCompletion, DeviceAllocator};

use crate::{
    DeviceAlgorithmError, DeviceRange, TransformError, backend,
    cuda::Execution,
    range::{LowerReadRange, LowerWriteRange, RepeatN},
};

/// In-flight zero-materialization transform retaining its input, affine
/// output, callable captures, execution program, and allocator.
pub type PendingRangeTransform<'borrow, 'stream, 'program, Input, Output, F, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (&'borrow Execution<'stream, 'program, A>, (Input, Output, F)),
>;

/// In-flight device `for_each` retaining the input expression and callable
/// captures until the stream completes.
pub type PendingForEach<'borrow, 'stream, 'program, Input, F, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (&'borrow Execution<'stream, 'program, A>, (Input, F)),
>;

/// Enqueue an allocation-free transform between concrete device-range
/// expressions.
///
/// All adaptors are lowered into one scalar kernel argument. The input,
/// affine output, callable, program, allocator, and stream borrow remain held
/// by the returned completion guard.
///
/// The affine output cannot be reused while the operation is pending:
///
/// ```compile_fail
/// use cuda_core::DeviceBuffer;
/// use rccl_memory::DeviceAllocator;
/// use rccl_thrust::{DeviceBufferExt, cuda::Execution, transform_into_async};
///
/// fn overlap<A: DeviceAllocator>(
///     execution: &Execution<'_, '_, A>,
///     input: &DeviceBuffer<i32>,
///     output: &mut DeviceBuffer<i32>,
/// ) {
///     let pending = transform_into_async(
///         execution,
///         input.view(),
///         output.view_mut(),
///         (|value| value) as fn(i32) -> i32,
///     );
///     let alias = output.view_mut();
///     pending.wait().unwrap();
///     drop(alias);
/// }
/// ```
pub fn transform_into_async<'borrow, 'stream, 'program, Input, Output, F, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    input: Input,
    output: Output,
    op: F,
) -> PendingRangeTransform<'borrow, 'stream, 'program, Input, Output, F, A>
where
    Input: LowerReadRange,
    Output: LowerWriteRange,
    F: Fn(Input::Item) -> Output::Item + Copy,
    A: DeviceAllocator,
{
    // SAFETY: every allocation and closure capture accessed by the launch is
    // owned by the retained resource tuple. All work is submitted through the
    // retained execution's stream.
    unsafe {
        execution.submit((input, output, op), |execution, resources| {
            backend::range_transform(execution, &resources.0, &resources.1, &resources.2)
        })
    }
}

/// Apply `op` to each input item and write it to an equally-sized affine
/// output range, synchronizing before returning.
pub fn transform_into<Input, Output, F, A>(
    execution: &Execution<'_, '_, A>,
    input: Input,
    output: Output,
    op: F,
) -> Result<(), DeviceAlgorithmError>
where
    Input: LowerReadRange,
    Output: LowerWriteRange,
    F: Fn(Input::Item) -> Output::Item + Copy,
    A: DeviceAllocator,
{
    transform_into_async(execution, input, output, op).wait()
}

/// Enqueue a Rust-style `for_each` over a concrete readable device range.
///
/// The callable executes only in generated CUDA code; it is never invoked on
/// the host. Dropping the returned guard synchronizes safely, while
/// [`CudaCompletion::wait`] also reports launch and execution failures.
///
/// A pending operation cannot let a locally borrowed input escape:
///
/// ```compile_fail
/// use cuda_core::DeviceBuffer;
/// use rccl_memory::DeviceAllocator;
/// use rccl_thrust::{
///     DeviceBufferExt, DeviceView, PendingForEach, cuda::Execution, for_each_async,
/// };
///
/// fn sink(_: i32) {}
///
/// fn escape<'a, 'stream, 'program, A: DeviceAllocator>(
///     execution: &'a Execution<'stream, 'program, A>,
/// ) -> PendingForEach<
///     'a,
///     'stream,
///     'program,
///     DeviceView<'a, i32>,
///     fn(i32),
///     A,
/// > {
///     let local = DeviceBuffer::<i32>::zeroed(execution.stream(), 4).unwrap();
///     for_each_async(execution, local.view(), sink as fn(i32))
/// }
/// ```
pub fn for_each_async<'borrow, 'stream, 'program, Input, F, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    input: Input,
    op: F,
) -> PendingForEach<'borrow, 'stream, 'program, Input, F, A>
where
    Input: LowerReadRange,
    F: Fn(Input::Item) + Copy,
    A: DeviceAllocator,
{
    // SAFETY: the retained tuple contains the complete input expression and
    // callable captures; the execution retains its program and allocator.
    unsafe {
        execution.submit((input, op), |execution, resources| {
            backend::range_for_each(execution, &resources.0, &resources.1)
        })
    }
}

/// Run a device `for_each` and synchronize before returning.
pub fn for_each<Input, F, A>(
    execution: &Execution<'_, '_, A>,
    input: Input,
    op: F,
) -> Result<(), DeviceAlgorithmError>
where
    Input: LowerReadRange,
    F: Fn(Input::Item) + Copy,
    A: DeviceAllocator,
{
    for_each_async(execution, input, op).wait()
}

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
    let result = execution
        .transform_repeat_in_place(output, *repeated.value(), op)
        .map_err(DeviceAlgorithmError::Launch);
    execution
        .synchronize(result)
        .map_err(TransformError::Execution)
}

/// Rust-named form of the implemented transform slice.
///
/// This is an eager device-side in-place map. Each element is updated as
/// `op(current, value)`, and the call synchronizes before returning.
pub fn map_with_value_in_place<T, F, A>(
    execution: &Execution<'_, '_, A>,
    output: &mut DeviceBuffer<T>,
    value: T,
    op: F,
) -> Result<(), TransformError>
where
    T: DeviceCopy,
    F: Fn(T, T) -> T + Copy,
    A: DeviceAllocator,
{
    let len = output.len();
    transform(execution, output, RepeatN::new(value, len), op)
}

/// Method syntax for Rust-flavored device-side in-place mapping.
pub trait DeviceTransformExt<T: DeviceCopy> {
    fn map_with_value_in_place<F, A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
        value: T,
        op: F,
    ) -> Result<(), TransformError>
    where
        F: Fn(T, T) -> T + Copy,
        A: DeviceAllocator;
}

impl<T: DeviceCopy> DeviceTransformExt<T> for DeviceBuffer<T> {
    fn map_with_value_in_place<F, A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
        value: T,
        op: F,
    ) -> Result<(), TransformError>
    where
        F: Fn(T, T) -> T + Copy,
        A: DeviceAllocator,
    {
        map_with_value_in_place(execution, self, value, op)
    }
}
