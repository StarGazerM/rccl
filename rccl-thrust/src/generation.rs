use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_memory::DeviceAllocator;

use crate::{
    DeviceAlgorithmError,
    copy::PendingRangeCopy,
    cuda::Execution,
    range::{DeviceBufferExt, DeviceViewMut, LowerWriteRange, RepeatN, repeat_n},
    transform::PendingRangeTransform,
};

/// In-flight fill of an affine device range.
pub type PendingRangeFill<'borrow, 'stream, 'program, Output, T, A> =
    PendingRangeCopy<'borrow, 'stream, 'program, RepeatN<T>, Output, A>;

/// In-flight full-buffer [`fill_async`] work.
pub type PendingFill<'borrow, 'stream, 'program, T, A> =
    PendingRangeFill<'borrow, 'stream, 'program, DeviceViewMut<'borrow, T>, T, A>;

/// In-flight index-generated output.
pub type PendingTabulate<'borrow, 'stream, 'program, Output, F, A> = PendingRangeTransform<
    'borrow,
    'stream,
    'program,
    rccl_cub::ArithmeticRange<usize>,
    Output,
    F,
    A,
>;

/// Enqueue a device fill and retain every borrowed resource until completion.
///
/// This is a CUDA completion guard rather than a Rust `Future`. Call
/// [`CudaCompletion::wait`] to observe launch/execution errors. Dropping the
/// guard safely synchronizes before releasing its borrows.
///
/// A mutable output cannot be submitted again while its first operation is
/// pending:
///
/// ```compile_fail
/// use cuda_core::DeviceBuffer;
/// use rccl_memory::DeviceAllocator;
/// use rccl_thrust::{cuda::Execution, fill_async};
///
/// fn reuse<A: DeviceAllocator>(
///     execution: &Execution<'_, '_, A>,
///     output: &mut DeviceBuffer<i32>,
/// ) {
///     let first = fill_async(execution, output, 1);
///     let second = fill_async(execution, output, 2);
///     first.wait().unwrap();
///     second.wait().unwrap();
/// }
/// ```
///
/// Locally borrowed state cannot escape inside a pending operation:
///
/// ```compile_fail
/// use cuda_core::DeviceBuffer;
/// use rccl_memory::DeviceAllocator;
/// use rccl_thrust::{PendingFill, cuda::Execution, fill_async};
///
/// fn escape<'a, 'stream, 'program, A: DeviceAllocator>(
///     execution: &'a Execution<'stream, 'program, A>,
/// ) -> PendingFill<'a, 'stream, 'program, i32, A> {
///     let mut local = DeviceBuffer::zeroed(execution.stream(), 4).unwrap();
///     fill_async(execution, &mut local, 7)
/// }
/// ```
pub fn fill_async<'borrow, 'stream, 'program, T, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    output: &'borrow mut DeviceBuffer<T>,
    value: T,
) -> PendingFill<'borrow, 'stream, 'program, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    fill_range_async(execution, output.view_mut(), value)
}

/// Enqueue a fill over an affine device-range expression.
///
/// The repeated value is lowered directly into the copy kernel descriptor; no
/// constant iterator or temporary buffer is materialized. Safe range
/// construction keeps the destination exclusively borrowed until completion.
pub fn fill_range_async<'borrow, 'stream, 'program, Output, T, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    output: Output,
    value: T,
) -> PendingRangeFill<'borrow, 'stream, 'program, Output, T, A>
where
    Output: LowerWriteRange<Item = T>,
    T: DeviceCopy,
    A: DeviceAllocator,
{
    let len = rccl_cub::DeviceRange::len(&output);
    // SAFETY: the completion retains the repeated value, affine output, and
    // execution until the selected generation kernel finishes.
    unsafe {
        execution.submit((repeat_n(value, len), output), |execution, resources| {
            crate::backend::range_fill(execution, &resources.1, *resources.0.value())
        })
    }
}

/// Fill an affine device range and synchronize before returning.
pub fn fill_range<Output, T, A>(
    execution: &Execution<'_, '_, A>,
    output: Output,
    value: T,
) -> Result<(), DeviceAlgorithmError>
where
    Output: LowerWriteRange<Item = T>,
    T: DeviceCopy,
    A: DeviceAllocator,
{
    fill_range_async(execution, output, value).wait()
}

/// Fill a device buffer with one value using the Thrust CUDA fill path.
pub fn fill<T, A>(
    execution: &Execution<'_, '_, A>,
    output: &mut DeviceBuffer<T>,
    value: T,
) -> Result<(), DeviceAlgorithmError>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    fill_async(execution, output, value).wait()
}

/// Enqueue index-generated values into an affine device range.
///
/// This is the pinned Thrust `tabulate` dependency expressed as an arithmetic
/// range transform. The `usize` extent is preserved, while the common launch
/// specializes to a 32-bit count when it fits.
pub fn tabulate_range_async<'borrow, 'stream, 'program, Output, F, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    output: Output,
    op: F,
) -> PendingTabulate<'borrow, 'stream, 'program, Output, F, A>
where
    Output: LowerWriteRange,
    F: Fn(usize) -> Output::Item + Copy,
    A: DeviceAllocator,
{
    let len = rccl_cub::DeviceRange::len(&output);
    let input = rccl_cub::arithmetic_range(0usize, 1usize, len);
    // SAFETY: the completion retains the arithmetic expression, affine output,
    // callable captures, and execution until the selected kernel finishes.
    unsafe {
        execution.submit((input, output, op), |execution, resources| {
            crate::backend::range_generate(execution, &resources.0, &resources.1, &resources.2)
        })
    }
}

/// Generate an affine device range from each logical index and synchronize.
pub fn tabulate_range<Output, F, A>(
    execution: &Execution<'_, '_, A>,
    output: Output,
    op: F,
) -> Result<(), DeviceAlgorithmError>
where
    Output: LowerWriteRange,
    F: Fn(usize) -> Output::Item + Copy,
    A: DeviceAllocator,
{
    tabulate_range_async(execution, output, op).wait()
}

/// Fill a complete device buffer with `op(index)`.
pub fn tabulate<T, F, A>(
    execution: &Execution<'_, '_, A>,
    output: &mut DeviceBuffer<T>,
    op: F,
) -> Result<(), DeviceAlgorithmError>
where
    T: DeviceCopy,
    F: Fn(usize) -> T + Copy,
    A: DeviceAllocator,
{
    tabulate_range(execution, output.view_mut(), op)
}

/// Values that can be generated by Thrust's arithmetic `sequence` contract.
pub trait SequenceValue: DeviceCopy + Copy {
    fn zero() -> Self;

    fn one() -> Self;

    fn sequence_value(init: Self, step: Self, index: usize) -> Self;
}

macro_rules! impl_sequence_value {
    ($($ty:ty),* $(,)?) => {
        $(
            impl SequenceValue for $ty {
                #[inline(always)]
                fn zero() -> Self {
                    0 as Self
                }

                #[inline(always)]
                fn one() -> Self {
                    1 as Self
                }

                #[inline(always)]
                fn sequence_value(init: Self, step: Self, index: usize) -> Self {
                    init + step * (index as Self)
                }
            }
        )*
    };
}

impl_sequence_value!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

/// Fill an affine output range with `0, 1, 2, ...`.
pub fn sequence_range<Output, T, A>(
    execution: &Execution<'_, '_, A>,
    output: Output,
) -> Result<(), DeviceAlgorithmError>
where
    Output: LowerWriteRange<Item = T>,
    T: SequenceValue,
    A: DeviceAllocator,
{
    sequence_range_by(execution, output, T::zero(), T::one())
}

/// Fill an affine output range with `init, init + 1, init + 2, ...`.
pub fn sequence_range_from<Output, T, A>(
    execution: &Execution<'_, '_, A>,
    output: Output,
    init: T,
) -> Result<(), DeviceAlgorithmError>
where
    Output: LowerWriteRange<Item = T>,
    T: SequenceValue,
    A: DeviceAllocator,
{
    sequence_range_by(execution, output, init, T::one())
}

/// Fill an affine output range with `init + step * index`.
pub fn sequence_range_by<Output, T, A>(
    execution: &Execution<'_, '_, A>,
    output: Output,
    init: T,
    step: T,
) -> Result<(), DeviceAlgorithmError>
where
    Output: LowerWriteRange<Item = T>,
    T: SequenceValue,
    A: DeviceAllocator,
{
    tabulate_range(execution, output, move |index| {
        T::sequence_value(init, step, index)
    })
}

/// Fill a device buffer with `0, 1, 2, ...`.
pub fn sequence<T, A>(
    execution: &Execution<'_, '_, A>,
    output: &mut DeviceBuffer<T>,
) -> Result<(), DeviceAlgorithmError>
where
    T: SequenceValue,
    A: DeviceAllocator,
{
    sequence_by(execution, output, T::zero(), T::one())
}

/// Fill a device buffer with `init, init + 1, init + 2, ...`.
pub fn sequence_from<T, A>(
    execution: &Execution<'_, '_, A>,
    output: &mut DeviceBuffer<T>,
    init: T,
) -> Result<(), DeviceAlgorithmError>
where
    T: SequenceValue,
    A: DeviceAllocator,
{
    sequence_by(execution, output, init, T::one())
}

/// Fill a device buffer with `init + step * index`.
pub fn sequence_by<T, A>(
    execution: &Execution<'_, '_, A>,
    output: &mut DeviceBuffer<T>,
    init: T,
    step: T,
) -> Result<(), DeviceAlgorithmError>
where
    T: SequenceValue,
    A: DeviceAllocator,
{
    sequence_range_by(execution, output.view_mut(), init, step)
}

/// Method syntax for Rust-flavored device generation algorithms.
pub trait DeviceGenerateExt<T: DeviceCopy> {
    fn fill_device<A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
        value: T,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator;

    fn fill_device_async<'borrow, 'stream, 'program, A>(
        &'borrow mut self,
        execution: &'borrow Execution<'stream, 'program, A>,
        value: T,
    ) -> PendingFill<'borrow, 'stream, 'program, T, A>
    where
        A: DeviceAllocator;

    fn tabulate<F, A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
        op: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        F: Fn(usize) -> T + Copy,
        A: DeviceAllocator;

    fn sequence<A>(&mut self, execution: &Execution<'_, '_, A>) -> Result<(), DeviceAlgorithmError>
    where
        T: SequenceValue,
        A: DeviceAllocator;

    fn sequence_from<A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
        init: T,
    ) -> Result<(), DeviceAlgorithmError>
    where
        T: SequenceValue,
        A: DeviceAllocator;

    fn sequence_by<A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
        init: T,
        step: T,
    ) -> Result<(), DeviceAlgorithmError>
    where
        T: SequenceValue,
        A: DeviceAllocator;
}

impl<T: DeviceCopy> DeviceGenerateExt<T> for DeviceBuffer<T> {
    fn fill_device<A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
        value: T,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
    {
        fill(execution, self, value)
    }

    fn fill_device_async<'borrow, 'stream, 'program, A>(
        &'borrow mut self,
        execution: &'borrow Execution<'stream, 'program, A>,
        value: T,
    ) -> PendingFill<'borrow, 'stream, 'program, T, A>
    where
        A: DeviceAllocator,
    {
        fill_async(execution, self, value)
    }

    fn tabulate<F, A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
        op: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        F: Fn(usize) -> T + Copy,
        A: DeviceAllocator,
    {
        tabulate(execution, self, op)
    }

    fn sequence<A>(&mut self, execution: &Execution<'_, '_, A>) -> Result<(), DeviceAlgorithmError>
    where
        T: SequenceValue,
        A: DeviceAllocator,
    {
        sequence(execution, self)
    }

    fn sequence_from<A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
        init: T,
    ) -> Result<(), DeviceAlgorithmError>
    where
        T: SequenceValue,
        A: DeviceAllocator,
    {
        sequence_from(execution, self, init)
    }

    fn sequence_by<A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
        init: T,
        step: T,
    ) -> Result<(), DeviceAlgorithmError>
    where
        T: SequenceValue,
        A: DeviceAllocator,
    {
        sequence_by(execution, self, init, step)
    }
}
