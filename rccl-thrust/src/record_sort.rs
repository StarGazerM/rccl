//! Stable lexicographic ordering for SRDatalog/HISA `u32` records.

use core::cmp::Ordering;

use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_memory::{CudaCompletion, DeviceAllocator};

use crate::{BooleanProvenance, DeviceAlgorithmError, cuda::Execution};

/// Padding-free row representation used by workload record sorting.
///
/// This is an alias for the sole `[u32; N]` field in HISA's `#[repr(C)]`
/// `Row<N>` layout, so no second record representation or padding policy is
/// introduced.
pub type RelationRow<const N: usize> = [u32; N];

mod payload {
    pub trait Sealed {}
    impl Sealed for u64 {}
    impl Sealed for f64 {}
    impl Sealed for crate::BooleanProvenance {}
}

/// Provenance payload supported by workload row sorting.
///
/// The sealed set is exactly SRDatalog's NaturalBag, independent-probability,
/// and Boolean provenance storage.
pub trait RecordPayload: DeviceCopy + payload::Sealed {}
impl RecordPayload for u64 {}
impl RecordPayload for f64 {}
impl RecordPayload for BooleanProvenance {}

pub type PendingRowSort<'borrow, 'stream, 'program, const N: usize, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (
        &'borrow Execution<'stream, 'program, A>,
        (
            &'borrow mut DeviceBuffer<RelationRow<N>>,
            rccl_cub::DeviceMergeSortKeysWorkspace<RelationRow<N>>,
        ),
    ),
>;

pub type PendingRowSortInto<'borrow, 'stream, 'program, const N: usize, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (
        &'borrow Execution<'stream, 'program, A>,
        (
            &'borrow DeviceBuffer<RelationRow<N>>,
            &'borrow mut DeviceBuffer<RelationRow<N>>,
            rccl_cub::DeviceMergeSortKeysWorkspace<RelationRow<N>>,
        ),
    ),
>;

pub type PendingRowPayloadSort<'borrow, 'stream, 'program, V, const N: usize, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (
        &'borrow Execution<'stream, 'program, A>,
        (
            &'borrow mut DeviceBuffer<RelationRow<N>>,
            &'borrow mut DeviceBuffer<V>,
            rccl_cub::DeviceMergeSortPairsWorkspace<RelationRow<N>, V>,
        ),
    ),
>;

pub type PendingRowPayloadSortInto<'borrow, 'stream, 'program, V, const N: usize, A> =
    CudaCompletion<
        'borrow,
        (),
        DeviceAlgorithmError,
        (
            &'borrow Execution<'stream, 'program, A>,
            (
                &'borrow DeviceBuffer<RelationRow<N>>,
                &'borrow DeviceBuffer<V>,
                &'borrow mut DeviceBuffer<RelationRow<N>>,
                &'borrow mut DeviceBuffer<V>,
                rccl_cub::DeviceMergeSortPairsWorkspace<RelationRow<N>, V>,
            ),
        ),
    >;

#[inline(always)]
pub(crate) fn ascending<const N: usize>(left: &RelationRow<N>, right: &RelationRow<N>) -> Ordering {
    macro_rules! compare_column {
        ($column:expr) => {
            if N > $column {
                if left[$column] < right[$column] {
                    return Ordering::Less;
                }
                if left[$column] > right[$column] {
                    return Ordering::Greater;
                }
            }
        };
    }

    // The supported arity is fixed at the call site. Spelling out the six
    // audited columns gives cuda-oxide the same compile-time unrolling as
    // pinned CUB's comparator and keeps register rows scalarized.
    compare_column!(0);
    compare_column!(1);
    compare_column!(2);
    compare_column!(3);
    compare_column!(4);
    compare_column!(5);
    Ordering::Equal
}

fn validate_arity<const N: usize>() -> Result<(), DeviceAlgorithmError> {
    if (2..=6).contains(&N) {
        Ok(())
    } else {
        Err(DeviceAlgorithmError::UnsupportedTarget(
            "record sort supports audited workload arities 2 through 6; use radix sort for scalar keys",
        ))
    }
}

fn validate_length(
    role: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), DeviceAlgorithmError> {
    if expected == actual {
        Ok(())
    } else {
        Err(DeviceAlgorithmError::LengthMismatch {
            role,
            expected,
            actual,
        })
    }
}

/// Stably sort padding-free `u32` rows in ascending lexicographic order.
///
/// This is deliberately the sole record comparator. Signed fields and floating
/// keys are absent because the audited SRDatalog/HISA row layout contains only
/// unsigned interned values. Equal rows preserve input order. Scalar-key users
/// should continue to use [`crate::DeviceSortExt`], which retains the existing
/// radix path instead of routing scalar keys through record comparison sort.
///
/// Scratch keys and merge partitions are allocated internally on the execution
/// stream and retained with `rows` by the returned [`CudaCompletion`]. Logical
/// extents remain `usize`; launch decomposition handles the device grid width.
pub fn sort_rows_lexicographic_async<'borrow, 'stream, 'program, const N: usize, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    rows: &'borrow mut DeviceBuffer<RelationRow<N>>,
) -> Result<PendingRowSort<'borrow, 'stream, 'program, N, A>, DeviceAlgorithmError>
where
    A: DeviceAllocator,
{
    validate_arity::<N>()?;
    let workspace = rccl_cub::device_merge_sort_keys_workspace(execution.launch(), rows.len())?;
    Ok(unsafe {
        execution.submit((rows, workspace), |execution, resources| {
            rccl_cub::device_merge_sort_keys(
                execution.launch(),
                &mut *resources.0,
                &mut resources.1,
                ascending::<N>,
            )
        })
    })
}

pub fn sort_rows_lexicographic<const N: usize, A>(
    execution: &Execution<'_, '_, A>,
    rows: &mut DeviceBuffer<RelationRow<N>>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
{
    sort_rows_lexicographic_async(execution, rows)?.wait()
}

/// Stable out-of-place row sort. Input and output lengths must match exactly.
/// Safe borrows prohibit aliasing the immutable input with the mutable output.
pub fn sort_rows_lexicographic_into_async<'borrow, 'stream, 'program, const N: usize, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    input: &'borrow DeviceBuffer<RelationRow<N>>,
    output: &'borrow mut DeviceBuffer<RelationRow<N>>,
) -> Result<PendingRowSortInto<'borrow, 'stream, 'program, N, A>, DeviceAlgorithmError>
where
    A: DeviceAllocator,
{
    validate_arity::<N>()?;
    validate_length("record-sort output", input.len(), output.len())?;
    let workspace = rccl_cub::device_merge_sort_keys_workspace(execution.launch(), input.len())?;
    Ok(unsafe {
        execution.submit((input, output, workspace), |execution, resources| {
            rccl_cub::device_merge_sort_keys_copy(
                execution.launch(),
                resources.0,
                &mut resources.2,
                &mut *resources.1,
                ascending::<N>,
            )
        })
    })
}

pub fn sort_rows_lexicographic_into<const N: usize, A>(
    execution: &Execution<'_, '_, A>,
    input: &DeviceBuffer<RelationRow<N>>,
    output: &mut DeviceBuffer<RelationRow<N>>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
{
    sort_rows_lexicographic_into_async(execution, input, output)?.wait()
}

/// Stably sort rows and provenance payloads in lockstep.
///
/// Ordering examines only the row. For duplicate rows, payloads retain their
/// original relative order, making first-provenance behavior deterministic.
/// No unstable record/payload variant is exposed. Key and payload lengths must
/// match; safe mutable borrows require disjoint storage.
pub fn sort_rows_with_payload_lexicographic_async<
    'borrow,
    'stream,
    'program,
    V,
    const N: usize,
    A,
>(
    execution: &'borrow Execution<'stream, 'program, A>,
    rows: &'borrow mut DeviceBuffer<RelationRow<N>>,
    payloads: &'borrow mut DeviceBuffer<V>,
) -> Result<PendingRowPayloadSort<'borrow, 'stream, 'program, V, N, A>, DeviceAlgorithmError>
where
    V: RecordPayload,
    A: DeviceAllocator,
{
    validate_arity::<N>()?;
    validate_length("record-sort payload", rows.len(), payloads.len())?;
    let workspace = rccl_cub::device_merge_sort_pairs_workspace(execution.launch(), rows.len())?;
    Ok(unsafe {
        execution.submit((rows, payloads, workspace), |execution, resources| {
            rccl_cub::device_merge_sort_pairs(
                execution.launch(),
                &mut *resources.0,
                &mut *resources.1,
                &mut resources.2,
                ascending::<N>,
            )
        })
    })
}

pub fn sort_rows_with_payload_lexicographic<V, const N: usize, A>(
    execution: &Execution<'_, '_, A>,
    rows: &mut DeviceBuffer<RelationRow<N>>,
    payloads: &mut DeviceBuffer<V>,
) -> Result<(), DeviceAlgorithmError>
where
    V: RecordPayload,
    A: DeviceAllocator,
{
    sort_rows_with_payload_lexicographic_async(execution, rows, payloads)?.wait()
}

/// Stable out-of-place row/payload sort.
///
/// Both input lengths and both output lengths must equal the row count. All
/// four buffers are retained until completion and are non-aliasing under safe
/// Rust borrows. Internal scratch has one row and one payload slot per input,
/// plus `usize` merge partitions; callers never observe partially initialized
/// scratch.
#[allow(clippy::too_many_arguments)]
pub fn sort_rows_with_payload_lexicographic_into_async<
    'borrow,
    'stream,
    'program,
    V,
    const N: usize,
    A,
>(
    execution: &'borrow Execution<'stream, 'program, A>,
    input_rows: &'borrow DeviceBuffer<RelationRow<N>>,
    input_payloads: &'borrow DeviceBuffer<V>,
    output_rows: &'borrow mut DeviceBuffer<RelationRow<N>>,
    output_payloads: &'borrow mut DeviceBuffer<V>,
) -> Result<PendingRowPayloadSortInto<'borrow, 'stream, 'program, V, N, A>, DeviceAlgorithmError>
where
    V: RecordPayload,
    A: DeviceAllocator,
{
    validate_arity::<N>()?;
    let len = input_rows.len();
    validate_length("record-sort input payload", len, input_payloads.len())?;
    validate_length("record-sort output rows", len, output_rows.len())?;
    validate_length("record-sort output payload", len, output_payloads.len())?;
    let workspace = rccl_cub::device_merge_sort_pairs_workspace(execution.launch(), len)?;
    Ok(unsafe {
        execution.submit(
            (
                input_rows,
                input_payloads,
                output_rows,
                output_payloads,
                workspace,
            ),
            |execution, resources| {
                rccl_cub::device_merge_sort_pairs_copy(
                    execution.launch(),
                    resources.0,
                    resources.1,
                    &mut resources.4,
                    &mut *resources.2,
                    &mut *resources.3,
                    ascending::<N>,
                )
            },
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn sort_rows_with_payload_lexicographic_into<V, const N: usize, A>(
    execution: &Execution<'_, '_, A>,
    input_rows: &DeviceBuffer<RelationRow<N>>,
    input_payloads: &DeviceBuffer<V>,
    output_rows: &mut DeviceBuffer<RelationRow<N>>,
    output_payloads: &mut DeviceBuffer<V>,
) -> Result<(), DeviceAlgorithmError>
where
    V: RecordPayload,
    A: DeviceAllocator,
{
    sort_rows_with_payload_lexicographic_into_async(
        execution,
        input_rows,
        input_payloads,
        output_rows,
        output_payloads,
    )?
    .wait()
}

#[cfg(test)]
mod tests {
    use super::{ascending, validate_arity};
    use core::cmp::Ordering;

    #[test]
    fn comparator_is_unsigned_lexicographic() {
        assert_eq!(ascending(&[0, u32::MAX], &[1, 0]), Ordering::Less);
        assert_eq!(ascending(&[4, 2, 7], &[4, 2, 7]), Ordering::Equal);
        assert_eq!(ascending(&[4, 3, 0], &[4, 2, u32::MAX]), Ordering::Greater);
    }

    #[test]
    fn workload_arity_is_closed() {
        assert!(validate_arity::<2>().is_ok());
        assert!(validate_arity::<6>().is_ok());
        assert!(validate_arity::<1>().is_err());
        assert!(validate_arity::<7>().is_err());
    }
}
