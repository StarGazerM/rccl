//! Deterministic reduction over sorted adjacent-key runs.

use cuda_core::DeviceBuffer;
use rccl_memory::{CudaCompletion, DeviceAllocator};

use crate::{
    DeviceAlgorithmError,
    cuda::Execution,
    range::{LowerReadRange, LowerWriteRange},
};

pub use rccl_cub::thrust_integration::{
    BooleanOr, BooleanProvenance, IndependentProbabilityOr, NaturalBagAdd,
};

pub type PendingReduceByKey<
    'borrow,
    'stream,
    'program,
    Keys,
    Values,
    OutputKeys,
    OutputValues,
    Equivalent,
    Operator,
    A,
> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (
        &'borrow Execution<'stream, 'program, A>,
        (
            Keys,
            Values,
            OutputKeys,
            OutputValues,
            &'borrow mut DeviceBuffer<u64>,
            Equivalent,
            Operator,
            rccl_cub::thrust_integration::CompactionWorkspace<'stream, &'borrow A>,
        ),
    ),
>;

fn validate_reduce_by_key<Keys, Values, OutputKeys, OutputValues>(
    keys: &Keys,
    values: &Values,
    output_keys: &OutputKeys,
    output_values: &OutputValues,
    group_count: &DeviceBuffer<u64>,
) -> Result<usize, DeviceAlgorithmError>
where
    Keys: LowerReadRange,
    Values: LowerReadRange,
    OutputKeys: LowerWriteRange<Item = Keys::Item>,
    OutputValues: LowerWriteRange<Item = Values::Item>,
{
    let len = rccl_cub::DeviceRange::len(keys);
    let values_len = rccl_cub::DeviceRange::len(values);
    if values_len != len {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "reduce-by-key values",
            expected: len,
            actual: values_len,
        });
    }
    let key_capacity = rccl_cub::DeviceRange::len(output_keys);
    if key_capacity < len {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "reduce-by-key key output",
            required: len,
            actual: key_capacity,
        });
    }
    let value_capacity = rccl_cub::DeviceRange::len(output_values);
    if value_capacity < len {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "reduce-by-key aggregate output",
            required: len,
            actual: value_capacity,
        });
    }
    if group_count.is_empty() {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "group-count output",
            required: 1,
            actual: 0,
        });
    }
    keys.validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    values
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    output_keys
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    output_values
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    Ok(len)
}

/// Reduce sorted adjacent-key runs into output prefixes.
///
/// `equivalent(previous, current)` explicitly defines run boundaries; equal
/// keys separated by a non-equivalent key form distinct groups. The first key
/// object in every run is copied unchanged, preserving any key-associated
/// provenance. Values are folded strictly left-to-right with the selected
/// fixed SRDatalog operator, so results are deterministic across launch shapes.
///
/// Both output ranges must have capacity for the complete input because the
/// group count is produced asynchronously. Only their prefixes of length
/// `group_count[0]` are results. `group_count` must contain at least one `u64`.
/// Safe range borrowing keeps all mutable outputs disjoint from inputs and from
/// one another. Extents remain `usize`; runs through `u32::MAX` items use the
/// common 32-bit position path and larger inputs select 64-bit positions.
///
/// Supported operators deliberately match real provenance workloads:
/// [`NaturalBagAdd`] wraps modulo 2^64, [`BooleanOr`] performs logical OR, and
/// [`IndependentProbabilityOr`] performs a clamped left fold for finite
/// probabilities in `[0, 1]`.
#[allow(clippy::too_many_arguments)]
pub fn reduce_by_key_async<
    'borrow,
    'stream,
    'program,
    Keys,
    Values,
    OutputKeys,
    OutputValues,
    Equivalent,
    Operator,
    A,
>(
    execution: &'borrow Execution<'stream, 'program, A>,
    keys: Keys,
    values: Values,
    output_keys: OutputKeys,
    output_values: OutputValues,
    group_count: &'borrow mut DeviceBuffer<u64>,
    equivalent: Equivalent,
    operator: Operator,
) -> Result<
    PendingReduceByKey<
        'borrow,
        'stream,
        'program,
        Keys,
        Values,
        OutputKeys,
        OutputValues,
        Equivalent,
        Operator,
        A,
    >,
    DeviceAlgorithmError,
>
where
    Keys: LowerReadRange,
    Values: LowerReadRange,
    OutputKeys: LowerWriteRange<Item = Keys::Item>,
    OutputValues: LowerWriteRange<Item = Values::Item>,
    Equivalent: Fn(Keys::Item, Keys::Item) -> bool + Copy,
    Operator: rccl_cub::thrust_integration::GroupReduceOperator<Values::Item>,
    A: DeviceAllocator,
{
    let len = validate_reduce_by_key(&keys, &values, &output_keys, &output_values, group_count)?;
    let workspace = rccl_cub::thrust_integration::grouped_reduction_workspace_in(
        execution.allocator(),
        execution.launch(),
        len,
    )?;
    Ok(unsafe {
        execution.submit(
            (
                keys,
                values,
                output_keys,
                output_values,
                group_count,
                equivalent,
                operator,
                workspace,
            ),
            |execution, resources| {
                rccl_cub::thrust_integration::device_reduce_by_key(
                    execution.launch(),
                    resources.0.lower_read(),
                    resources.1.lower_read(),
                    resources.2.lower_write(),
                    resources.3.lower_write(),
                    resources.4,
                    resources.5,
                    resources.6,
                    len,
                    &mut resources.7,
                )
            },
        )
    })
}

/// Synchronous form of [`reduce_by_key_async`].
#[allow(clippy::too_many_arguments)]
pub fn reduce_by_key<Keys, Values, OutputKeys, OutputValues, Equivalent, Operator, A>(
    execution: &Execution<'_, '_, A>,
    keys: Keys,
    values: Values,
    output_keys: OutputKeys,
    output_values: OutputValues,
    group_count: &mut DeviceBuffer<u64>,
    equivalent: Equivalent,
    operator: Operator,
) -> Result<(), DeviceAlgorithmError>
where
    Keys: LowerReadRange,
    Values: LowerReadRange,
    OutputKeys: LowerWriteRange<Item = Keys::Item>,
    OutputValues: LowerWriteRange<Item = Values::Item>,
    Equivalent: Fn(Keys::Item, Keys::Item) -> bool + Copy,
    Operator: rccl_cub::thrust_integration::GroupReduceOperator<Values::Item>,
    A: DeviceAllocator,
{
    reduce_by_key_async(
        execution,
        keys,
        values,
        output_keys,
        output_values,
        group_count,
        equivalent,
        operator,
    )?
    .wait()
}
