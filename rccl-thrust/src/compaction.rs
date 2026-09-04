//! Stable select, adjacent unique, and two-way partition for workload ranges.

use cuda_core::DeviceBuffer;
use rccl_memory::{CudaCompletion, DeviceAllocator};

use crate::{
    DeviceAlgorithmError,
    cuda::Execution,
    range::{LowerReadRange, LowerWriteRange},
};

pub type PendingSelect<'borrow, 'stream, 'program, Input, Output, Predicate, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (
        &'borrow Execution<'stream, 'program, A>,
        (
            Input,
            Output,
            &'borrow mut DeviceBuffer<u64>,
            Predicate,
            rccl_cub::thrust_integration::CompactionWorkspace<'stream, &'borrow A>,
        ),
    ),
>;

pub type PendingFlaggedSelect<'borrow, 'stream, 'program, Input, Flags, Output, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (
        &'borrow Execution<'stream, 'program, A>,
        (
            Input,
            Flags,
            Output,
            &'borrow mut DeviceBuffer<u64>,
            rccl_cub::thrust_integration::CompactionWorkspace<'stream, &'borrow A>,
        ),
    ),
>;

pub type PendingUnique<'borrow, 'stream, 'program, Input, Output, Equivalent, A> =
    PendingSelect<'borrow, 'stream, 'program, Input, Output, Equivalent, A>;

pub type PendingInPlaceCompaction<'borrow, 'stream, 'program, Values, Scratch, Operation, A> =
    CudaCompletion<
        'borrow,
        (),
        DeviceAlgorithmError,
        (
            &'borrow Execution<'stream, 'program, A>,
            (
                Values,
                Scratch,
                &'borrow mut DeviceBuffer<u64>,
                Operation,
                rccl_cub::thrust_integration::CompactionWorkspace<'stream, &'borrow A>,
            ),
        ),
    >;

fn validate_count(count: &DeviceBuffer<u64>) -> Result<(), DeviceAlgorithmError> {
    if count.is_empty() {
        Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "selected-count output",
            required: 1,
            actual: 0,
        })
    } else {
        Ok(())
    }
}

fn validate_output<Input, Output>(
    input: &Input,
    output: &Output,
    role: &'static str,
) -> Result<usize, DeviceAlgorithmError>
where
    Input: LowerReadRange,
    Output: LowerWriteRange<Item = Input::Item>,
{
    let len = rccl_cub::DeviceRange::len(input);
    let capacity = rccl_cub::DeviceRange::len(output);
    if capacity < len {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role,
            required: len,
            actual: capacity,
        });
    }
    input
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    output
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    Ok(len)
}

/// Stably copy values satisfying `predicate` into the output prefix.
///
/// `output` must have capacity for the complete input because selection is
/// asynchronous and no host count is available before scatter. `selected_count`
/// must contain at least one `u64`; its first word is written on the operation
/// stream and cannot be observed until the returned completion finishes.
/// The predicate is evaluated exactly once per input item. Safe range borrows
/// prohibit input/output overlap.
pub fn select_if_async<'borrow, 'stream, 'program, Input, Output, Predicate, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    input: Input,
    output: Output,
    selected_count: &'borrow mut DeviceBuffer<u64>,
    predicate: Predicate,
) -> Result<
    PendingSelect<'borrow, 'stream, 'program, Input, Output, Predicate, A>,
    DeviceAlgorithmError,
>
where
    Input: LowerReadRange,
    Output: LowerWriteRange<Item = Input::Item>,
    Predicate: Fn(Input::Item) -> bool + Copy,
    A: DeviceAllocator,
{
    let len = validate_output(&input, &output, "select output")?;
    validate_count(selected_count)?;
    let workspace = rccl_cub::thrust_integration::compaction_workspace_in(
        execution.allocator(),
        execution.launch(),
        len,
    )?;
    Ok(unsafe {
        execution.submit(
            (input, output, selected_count, predicate, workspace),
            |execution, resources| {
                rccl_cub::thrust_integration::device_select_if(
                    execution.launch(),
                    resources.0.lower_read(),
                    resources.1.lower_write(),
                    resources.2,
                    resources.3,
                    len,
                    &mut resources.4,
                )
            },
        )
    })
}

pub fn select_if<Input, Output, Predicate, A>(
    execution: &Execution<'_, '_, A>,
    input: Input,
    output: Output,
    selected_count: &mut DeviceBuffer<u64>,
    predicate: Predicate,
) -> Result<(), DeviceAlgorithmError>
where
    Input: LowerReadRange,
    Output: LowerWriteRange<Item = Input::Item>,
    Predicate: Fn(Input::Item) -> bool + Copy,
    A: DeviceAllocator,
{
    select_if_async(execution, input, output, selected_count, predicate)?.wait()
}

/// Stably select values whose corresponding workload `u32` flag is nonzero.
pub fn select_flagged_async<'borrow, 'stream, 'program, Input, Flags, Output, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    input: Input,
    selected: Flags,
    output: Output,
    selected_count: &'borrow mut DeviceBuffer<u64>,
) -> Result<
    PendingFlaggedSelect<'borrow, 'stream, 'program, Input, Flags, Output, A>,
    DeviceAlgorithmError,
>
where
    Input: LowerReadRange,
    Flags: LowerReadRange<Item = u32>,
    Output: LowerWriteRange<Item = Input::Item>,
    A: DeviceAllocator,
{
    let len = validate_output(&input, &output, "flagged-select output")?;
    let flag_len = rccl_cub::DeviceRange::len(&selected);
    if flag_len != len {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "selection flags",
            expected: len,
            actual: flag_len,
        });
    }
    selected
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    validate_count(selected_count)?;
    let workspace = rccl_cub::thrust_integration::compaction_workspace_in(
        execution.allocator(),
        execution.launch(),
        len,
    )?;
    Ok(unsafe {
        execution.submit(
            (input, selected, output, selected_count, workspace),
            |execution, resources| {
                rccl_cub::thrust_integration::device_select_flagged(
                    execution.launch(),
                    resources.0.lower_read(),
                    resources.1.lower_read(),
                    resources.2.lower_write(),
                    resources.3,
                    len,
                    &mut resources.4,
                )
            },
        )
    })
}

pub fn select_flagged<Input, Flags, Output, A>(
    execution: &Execution<'_, '_, A>,
    input: Input,
    selected: Flags,
    output: Output,
    selected_count: &mut DeviceBuffer<u64>,
) -> Result<(), DeviceAlgorithmError>
where
    Input: LowerReadRange,
    Flags: LowerReadRange<Item = u32>,
    Output: LowerWriteRange<Item = Input::Item>,
    A: DeviceAllocator,
{
    select_flagged_async(execution, input, selected, output, selected_count)?.wait()
}

/// Stably copy the first item of each adjacent equivalence run.
///
/// Equivalence is intentionally explicit because SRDatalog deduplicates sorted
/// rows by selected key columns while associated provenance follows the kept
/// row. Non-adjacent equivalent items are not removed.
pub fn unique_copy_by_async<'borrow, 'stream, 'program, Input, Output, Equivalent, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    input: Input,
    output: Output,
    selected_count: &'borrow mut DeviceBuffer<u64>,
    equivalent: Equivalent,
) -> Result<
    PendingUnique<'borrow, 'stream, 'program, Input, Output, Equivalent, A>,
    DeviceAlgorithmError,
>
where
    Input: LowerReadRange,
    Output: LowerWriteRange<Item = Input::Item>,
    Equivalent: Fn(Input::Item, Input::Item) -> bool + Copy,
    A: DeviceAllocator,
{
    let len = validate_output(&input, &output, "unique-copy output")?;
    validate_count(selected_count)?;
    let workspace = rccl_cub::thrust_integration::compaction_workspace_in(
        execution.allocator(),
        execution.launch(),
        len,
    )?;
    Ok(unsafe {
        execution.submit(
            (input, output, selected_count, equivalent, workspace),
            |execution, resources| {
                rccl_cub::thrust_integration::device_unique_copy_by(
                    execution.launch(),
                    resources.0.lower_read(),
                    resources.1.lower_write(),
                    resources.2,
                    resources.3,
                    len,
                    &mut resources.4,
                )
            },
        )
    })
}

pub fn unique_copy_by<Input, Output, Equivalent, A>(
    execution: &Execution<'_, '_, A>,
    input: Input,
    output: Output,
    selected_count: &mut DeviceBuffer<u64>,
    equivalent: Equivalent,
) -> Result<(), DeviceAlgorithmError>
where
    Input: LowerReadRange,
    Output: LowerWriteRange<Item = Input::Item>,
    Equivalent: Fn(Input::Item, Input::Item) -> bool + Copy,
    A: DeviceAllocator,
{
    unique_copy_by_async(execution, input, output, selected_count, equivalent)?.wait()
}

/// Compact adjacent equivalence runs back into `values` using disjoint scratch.
///
/// `scratch` must have exactly the same length and cannot alias `values` through
/// safe range construction. Its complete range is initialized before the
/// selected prefix is scattered, so copying the full range back never reads an
/// uninitialized tail. Only the prefix reported in `selected_count[0]` is part
/// of the unique result.
pub fn unique_by_async<'borrow, 'stream, 'program, Values, Scratch, Equivalent, A, T>(
    execution: &'borrow Execution<'stream, 'program, A>,
    values: Values,
    scratch: Scratch,
    selected_count: &'borrow mut DeviceBuffer<u64>,
    equivalent: Equivalent,
) -> Result<
    PendingInPlaceCompaction<'borrow, 'stream, 'program, Values, Scratch, Equivalent, A>,
    DeviceAlgorithmError,
>
where
    Values: LowerReadRange<Item = T> + LowerWriteRange<Item = T>,
    Scratch: LowerReadRange<Item = T> + LowerWriteRange<Item = T>,
    Equivalent: Fn(T, T) -> bool + Copy,
    A: DeviceAllocator,
{
    let len = validate_output(&values, &scratch, "unique scratch")?;
    if rccl_cub::DeviceRange::len(&scratch) != len {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "unique scratch",
            expected: len,
            actual: rccl_cub::DeviceRange::len(&scratch),
        });
    }
    validate_count(selected_count)?;
    let workspace = rccl_cub::thrust_integration::compaction_workspace_in(
        execution.allocator(),
        execution.launch(),
        len,
    )?;
    Ok(unsafe {
        execution.submit(
            (values, scratch, selected_count, equivalent, workspace),
            |execution, resources| {
                crate::backend::range_transform(execution, &resources.0, &resources.1, &|value| {
                    value
                })?;
                rccl_cub::thrust_integration::device_unique_copy_by(
                    execution.launch(),
                    resources.0.lower_read(),
                    resources.1.lower_write(),
                    resources.2,
                    resources.3,
                    len,
                    &mut resources.4,
                )?;
                crate::backend::range_transform(execution, &resources.1, &resources.0, &|value| {
                    value
                })
            },
        )
    })
}

pub fn unique_by<Values, Scratch, Equivalent, A, T>(
    execution: &Execution<'_, '_, A>,
    values: Values,
    scratch: Scratch,
    selected_count: &mut DeviceBuffer<u64>,
    equivalent: Equivalent,
) -> Result<(), DeviceAlgorithmError>
where
    Values: LowerReadRange<Item = T> + LowerWriteRange<Item = T>,
    Scratch: LowerReadRange<Item = T> + LowerWriteRange<Item = T>,
    Equivalent: Fn(T, T) -> bool + Copy,
    A: DeviceAllocator,
{
    unique_by_async(execution, values, scratch, selected_count, equivalent)?.wait()
}

/// Stable two-way partition through caller-provided disjoint scratch.
///
/// Selected items occupy the prefix in input order; rejected items follow in
/// input order. `selected_count[0]` is the partition point after completion.
pub fn stable_partition_by_async<'borrow, 'stream, 'program, Values, Scratch, Predicate, A, T>(
    execution: &'borrow Execution<'stream, 'program, A>,
    values: Values,
    scratch: Scratch,
    selected_count: &'borrow mut DeviceBuffer<u64>,
    predicate: Predicate,
) -> Result<
    PendingInPlaceCompaction<'borrow, 'stream, 'program, Values, Scratch, Predicate, A>,
    DeviceAlgorithmError,
>
where
    Values: LowerReadRange<Item = T> + LowerWriteRange<Item = T>,
    Scratch: LowerReadRange<Item = T> + LowerWriteRange<Item = T>,
    Predicate: Fn(T) -> bool + Copy,
    A: DeviceAllocator,
{
    let len = validate_output(&values, &scratch, "stable-partition scratch")?;
    if rccl_cub::DeviceRange::len(&scratch) != len {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "stable-partition scratch",
            expected: len,
            actual: rccl_cub::DeviceRange::len(&scratch),
        });
    }
    validate_count(selected_count)?;
    let workspace = rccl_cub::thrust_integration::compaction_workspace_in(
        execution.allocator(),
        execution.launch(),
        len,
    )?;
    Ok(unsafe {
        execution.submit(
            (values, scratch, selected_count, predicate, workspace),
            |execution, resources| {
                rccl_cub::thrust_integration::device_stable_partition_by(
                    execution.launch(),
                    resources.0.lower_read(),
                    resources.1.lower_write(),
                    resources.2,
                    resources.3,
                    len,
                    &mut resources.4,
                )?;
                crate::backend::range_transform(execution, &resources.1, &resources.0, &|value| {
                    value
                })
            },
        )
    })
}

pub fn stable_partition_by<Values, Scratch, Predicate, A, T>(
    execution: &Execution<'_, '_, A>,
    values: Values,
    scratch: Scratch,
    selected_count: &mut DeviceBuffer<u64>,
    predicate: Predicate,
) -> Result<(), DeviceAlgorithmError>
where
    Values: LowerReadRange<Item = T> + LowerWriteRange<Item = T>,
    Scratch: LowerReadRange<Item = T> + LowerWriteRange<Item = T>,
    Predicate: Fn(T) -> bool + Copy,
    A: DeviceAllocator,
{
    stable_partition_by_async(execution, values, scratch, selected_count, predicate)?.wait()
}
