use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_memory::{CudaCompletion, DeviceAllocator};

use crate::{
    DeviceAlgorithmError,
    cuda::Execution,
    range::{LowerReadRange, LowerWriteRange, Select},
};

/// In-flight contiguous device-to-device copy.
///
/// The completion retains the shared source borrow, affine destination borrow,
/// execution program, allocator, and stream until CUDA has finished the copy.
pub type PendingCopy<'borrow, 'stream, 'program, T, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (
        &'borrow Execution<'stream, 'program, A>,
        (&'borrow DeviceBuffer<T>, &'borrow mut DeviceBuffer<T>),
    ),
>;

/// In-flight copy between zero-materialization device ranges.
pub type PendingRangeCopy<'borrow, 'stream, 'program, Input, Output, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (&'borrow Execution<'stream, 'program, A>, (Input, Output)),
>;

/// In-flight indexed gather between zero-materialization device ranges.
pub type PendingGather<'borrow, 'stream, 'program, Input, Indices, Output, A> =
    PendingRangeCopy<'borrow, 'stream, 'program, Select<Input, Indices>, Output, A>;

/// Enqueue a full contiguous device-to-device copy.
///
/// This common path uses CUDA's native D2D transfer rather than launching a
/// scalar iterator kernel. Source and destination lengths must match. Safe Rust
/// borrowing makes overlap impossible: the source is shared while the complete
/// destination allocation is borrowed exclusively. Use [`copy_range_async`]
/// for checked subranges and composed device ranges.
pub fn copy_async<'borrow, 'stream, 'program, T, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    input: &'borrow DeviceBuffer<T>,
    output: &'borrow mut DeviceBuffer<T>,
) -> PendingCopy<'borrow, 'stream, 'program, T, A>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    // SAFETY: the retained resources contain both complete allocations. Their
    // shared/exclusive Rust borrows prove the D2D regions do not overlap and
    // remain alive until completion.
    unsafe {
        execution.submit((input, output), |execution, resources| {
            let (input, output) = resources;
            if input.len() != output.len() {
                return Err(DeviceAlgorithmError::LengthMismatch {
                    role: "copy output",
                    expected: input.len(),
                    actual: output.len(),
                });
            }
            if input.is_empty() {
                return Ok(());
            }
            cuda_core::simt::memory::memcpy_dtod_async(
                output.cu_deviceptr(),
                input.cu_deviceptr(),
                input.num_bytes(),
                execution.stream().cu_stream(),
            )
            .map_err(DeviceAlgorithmError::Driver)
        })
    }
}

/// Copy one complete contiguous device buffer and synchronize before returning.
pub fn copy<T, A>(
    execution: &Execution<'_, '_, A>,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> Result<(), DeviceAlgorithmError>
where
    T: DeviceCopy,
    A: DeviceAllocator,
{
    copy_async(execution, input, output).wait()
}

/// Enqueue an allocation-free copy between equally sized device ranges.
///
/// Adaptors are lowered directly to scalar descriptors; no iterator or index
/// buffer is materialized. The destination expression is affine, so source and
/// destination storage cannot overlap through this safe API. The launch keeps
/// `usize` extents and uses the range backend's `u32` fast specialization when
/// the extent fits.
///
/// Rust does not need a separate `copy_n` cursor API: slice the validated
/// source and destination ranges to the requested count, then copy them.
pub fn copy_range_async<'borrow, 'stream, 'program, Input, Output, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    input: Input,
    output: Output,
) -> PendingRangeCopy<'borrow, 'stream, 'program, Input, Output, A>
where
    Input: LowerReadRange,
    Output: LowerWriteRange<Item = Input::Item>,
    A: DeviceAllocator,
{
    // SAFETY: both range expressions, including every allocation borrow, are
    // moved into the completion guard. LowerWriteRange is affine at safe
    // construction sites and all work is submitted on execution's stream.
    unsafe {
        execution.submit((input, output), |execution, resources| {
            crate::backend::range_transform(execution, &resources.0, &resources.1, &|value| value)
        })
    }
}

/// Copy between equally sized device ranges and synchronize before returning.
pub fn copy_range<Input, Output, A>(
    execution: &Execution<'_, '_, A>,
    input: Input,
    output: Output,
) -> Result<(), DeviceAlgorithmError>
where
    Input: LowerReadRange,
    Output: LowerWriteRange<Item = Input::Item>,
    A: DeviceAllocator,
{
    copy_range_async(execution, input, output).wait()
}

/// Enqueue an indexed gather into an equally sized affine output range.
///
/// Each unsigned device index selects one input item. `u32` is the workload
/// fast path; `u64` and `usize` preserve large relation indices. An index at or
/// beyond `input.len()` deterministically traps in the kernel and is reported
/// by [`CudaCompletion::wait`] as a CUDA driver error. No partial-output value
/// is promised after such a fault.
///
/// Input, indices, and output cannot alias mutably through this safe API. In
/// particular, an in-place permutation is rejected rather than reproducing
/// Thrust's undefined overlapping gather behavior:
///
/// ```compile_fail
/// use cuda_core::DeviceBuffer;
/// use rccl_memory::DeviceAllocator;
/// use rccl_thrust::{DeviceBufferExt, cuda::Execution, gather};
///
/// fn in_place<A: DeviceAllocator>(
///     execution: &Execution<'_, '_, A>,
///     values: &mut DeviceBuffer<u32>,
///     indices: &DeviceBuffer<u32>,
/// ) {
///     gather(execution, values.view(), indices.view(), values.view_mut()).unwrap();
/// }
/// ```
pub fn gather_async<'borrow, 'stream, 'program, Input, Indices, Output, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    input: Input,
    indices: Indices,
    output: Output,
) -> PendingGather<'borrow, 'stream, 'program, Input, Indices, Output, A>
where
    Select<Input, Indices>: LowerReadRange,
    Output: LowerWriteRange<Item = <Select<Input, Indices> as rccl_cub::ReadRange>::Item>,
    A: DeviceAllocator,
{
    copy_range_async(execution, Select::new(input, indices), output)
}

/// Gather indexed input items and synchronize before returning.
pub fn gather<Input, Indices, Output, A>(
    execution: &Execution<'_, '_, A>,
    input: Input,
    indices: Indices,
    output: Output,
) -> Result<(), DeviceAlgorithmError>
where
    Select<Input, Indices>: LowerReadRange,
    Output: LowerWriteRange<Item = <Select<Input, Indices> as rccl_cub::ReadRange>::Item>,
    A: DeviceAllocator,
{
    gather_async(execution, input, indices, output).wait()
}

/// Rust method syntax for readable device-range copies and gathers.
pub trait DeviceRangeCopyExt: LowerReadRange + Sized {
    fn copy_to<Output, A>(
        self,
        execution: &Execution<'_, '_, A>,
        output: Output,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Output: LowerWriteRange<Item = Self::Item>,
        A: DeviceAllocator,
    {
        copy_range(execution, self, output)
    }

    fn gather_to<Indices, Output, A>(
        self,
        execution: &Execution<'_, '_, A>,
        indices: Indices,
        output: Output,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Select<Self, Indices>: LowerReadRange,
        Output: LowerWriteRange<Item = <Select<Self, Indices> as rccl_cub::ReadRange>::Item>,
        A: DeviceAllocator,
    {
        gather(execution, self, indices, output)
    }
}

impl<Input: LowerReadRange> DeviceRangeCopyExt for Input {}
