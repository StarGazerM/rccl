//! Vectorized binary search over sorted, zero-materialization device ranges.
//!
//! Haystacks must be sorted by the supplied comparator. Comparator equivalence
//! defines matches; indices are `u64` positions relative to the supplied view
//! (not its parent allocation) and lie in `0..=haystack.len()`. Every output
//! must have exactly as many items as the needle range. No temporary storage or
//! host readback is required. Natural-order forms require `Ord`; floating-point
//! keys require an explicit, consistent total-order comparator.

use core::cmp::Ordering;
use rccl_memory::{CudaCompletion, DeviceAllocator};

use crate::{
    DeviceAlgorithmError, DeviceRange,
    cuda::Execution,
    range::{LowerReadRange, LowerWriteRange},
};

/// Retains input/output ranges, comparator captures, and execution until completion.
pub type PendingSearch<'borrow, 'stream, 'program, H, N, O, F, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (&'borrow Execution<'stream, 'program, A>, (H, N, O, F)),
>;

/// Retains both disjoint output ranges together with the search resources.
pub type PendingEqualRange<'borrow, 'stream, 'program, H, N, B, E, F, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (&'borrow Execution<'stream, 'program, A>, (H, N, B, E, F)),
>;

fn validate_inputs<H, N>(haystack: &H, needles: &N) -> Result<(), DeviceAlgorithmError>
where
    H: LowerReadRange,
    N: LowerReadRange<Item = H::Item>,
{
    u64::try_from(haystack.len()).map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    haystack.validate_lowering().map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    needles.validate_lowering().map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    Ok(())
}

fn validate_output<O: LowerWriteRange<Item = u64>>(
    output: &O,
    len: usize,
    role: &'static str,
) -> Result<(), DeviceAlgorithmError> {
    if output.len() != len {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role,
            expected: len,
            actual: output.len(),
        });
    }
    output.validate_lowering().map_err(|_| DeviceAlgorithmError::SizeOverflow)
}

fn bound_by_async<'borrow, 'stream, 'program, H, N, O, F, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    haystack: H,
    needles: N,
    output: O,
    compare: F,
    upper: bool,
) -> PendingSearch<'borrow, 'stream, 'program, H, N, O, F, A>
where
    H: LowerReadRange,
    N: LowerReadRange<Item = H::Item>,
    O: LowerWriteRange<Item = u64>,
    F: Fn(&H::Item, &H::Item) -> Ordering + Copy,
    A: DeviceAllocator,
{
    // SAFETY: all range expressions and callable captures are retained, affine
    // outputs cannot overlap safe input views, and work uses execution's stream.
    unsafe {
        execution.submit((haystack, needles, output, compare), |execution, resources| {
            let (haystack, needles, output, compare) = resources;
            validate_inputs(haystack, needles)?;
            validate_output(output, needles.len(), "search output")?;
            execution.launch().range_bound(
                haystack.lower_read(), needles.lower_read(), output.lower_write(),
                *compare, haystack.len(), needles.len(), upper,
            ).map_err(DeviceAlgorithmError::Launch)
        })
    }
}

/// Enqueue the first position not ordered before each needle.
///
/// Validation and launch errors are reported by the returned completion's
/// `wait`. The comparator takes references to range values, not host buffers.
/// Borrowed device captures are retained until completion.
///
/// An unfinished operation keeps its output exclusively borrowed:
/// ```compile_fail
/// use cuda_core::DeviceBuffer;
/// use rccl_memory::DeviceAllocator;
/// use rccl_thrust::{cuda::Execution, DeviceBufferExt, lower_bound_by_async};
/// fn overlap<A: DeviceAllocator>(exec: &Execution<'_, '_, A>,
///     keys: &DeviceBuffer<u32>, out: &mut DeviceBuffer<u64>) {
///     let pending = lower_bound_by_async(exec, keys.view(), keys.view(),
///         out.view_mut(), |a, b| a.cmp(b));
///     let second = out.view_mut();
///     pending.wait().unwrap();
///     drop(second);
/// }
/// ```
///
/// Borrowed comparator state cannot escape with a pending operation:
/// ```compile_fail
/// use cuda_core::DeviceBuffer;
/// use rccl_memory::DeviceAllocator;
/// use rccl_thrust::{cuda::Execution, DeviceBufferExt, lower_bound_by_async};
/// fn capture<A: DeviceAllocator>(exec: &Execution<'_, '_, A>,
///     keys: &DeviceBuffer<u32>, out: &mut DeviceBuffer<u64>) {
///     let pending = {
///         let divisor = 10_u32;
///         lower_bound_by_async(exec, keys.view(), keys.view(), out.view_mut(),
///             |a, b| (a / divisor).cmp(&(b / divisor)))
///     };
///     pending.wait().unwrap();
/// }
/// ```
pub fn lower_bound_by_async<'borrow, 'stream, 'program, H, N, O, F, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    haystack: H,
    needles: N,
    output: O,
    compare: F,
) -> PendingSearch<'borrow, 'stream, 'program, H, N, O, F, A>
where
    H: LowerReadRange,
    N: LowerReadRange<Item = H::Item>,
    O: LowerWriteRange<Item = u64>,
    F: Fn(&H::Item, &H::Item) -> Ordering + Copy,
    A: DeviceAllocator,
{
    bound_by_async(execution, haystack, needles, output, compare, false)
}

/// Enqueue the first position ordered after each needle.
pub fn upper_bound_by_async<'borrow, 'stream, 'program, H, N, O, F, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    haystack: H,
    needles: N,
    output: O,
    compare: F,
) -> PendingSearch<'borrow, 'stream, 'program, H, N, O, F, A>
where
    H: LowerReadRange,
    N: LowerReadRange<Item = H::Item>,
    O: LowerWriteRange<Item = u64>,
    F: Fn(&H::Item, &H::Item) -> Ordering + Copy,
    A: DeviceAllocator,
{
    bound_by_async(execution, haystack, needles, output, compare, true)
}

/// Enqueue both insertion bounds into separate begin/end ranges.
///
/// Both output lengths and all range lowerings are validated before any work
/// is launched. Empty haystacks produce zero bounds; empty needles launch no
/// kernel. The completion retains both exclusive output borrows.
pub fn equal_range_by_async<'borrow, 'stream, 'program, H, N, B, E, F, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    haystack: H,
    needles: N,
    begins: B,
    ends: E,
    compare: F,
) -> PendingEqualRange<'borrow, 'stream, 'program, H, N, B, E, F, A>
where
    H: LowerReadRange,
    N: LowerReadRange<Item = H::Item>,
    B: LowerWriteRange<Item = u64>,
    E: LowerWriteRange<Item = u64>,
    F: Fn(&H::Item, &H::Item) -> Ordering + Copy,
    A: DeviceAllocator,
{
    // SAFETY: every input, output, comparator capture, and execution borrow is
    // moved into the completion; all GPU access occurs on its retained stream.
    unsafe {
        execution.submit((haystack, needles, begins, ends, compare), |execution, resources| {
            let (haystack, needles, begins, ends, compare) = resources;
            validate_inputs(haystack, needles)?;
            validate_output(begins, needles.len(), "equal-range begins")?;
            validate_output(ends, needles.len(), "equal-range ends")?;
            execution.launch().range_equal_range(
                haystack.lower_read(), needles.lower_read(), begins.lower_write(),
                ends.lower_write(), *compare, haystack.len(), needles.len(),
            ).map_err(DeviceAlgorithmError::Launch)
        })
    }
}

/// Synchronous comparator-based lower bound.
pub fn lower_bound_by<H, N, O, F, A>(
    execution: &Execution<'_, '_, A>, haystack: H, needles: N, output: O, compare: F,
) -> Result<(), DeviceAlgorithmError>
where
    H: LowerReadRange,
    N: LowerReadRange<Item = H::Item>,
    O: LowerWriteRange<Item = u64>,
    F: Fn(&H::Item, &H::Item) -> Ordering + Copy,
    A: DeviceAllocator,
{
    lower_bound_by_async(execution, haystack, needles, output, compare).wait()
}

/// Synchronous comparator-based upper bound.
pub fn upper_bound_by<H, N, O, F, A>(
    execution: &Execution<'_, '_, A>, haystack: H, needles: N, output: O, compare: F,
) -> Result<(), DeviceAlgorithmError>
where
    H: LowerReadRange,
    N: LowerReadRange<Item = H::Item>,
    O: LowerWriteRange<Item = u64>,
    F: Fn(&H::Item, &H::Item) -> Ordering + Copy,
    A: DeviceAllocator,
{
    upper_bound_by_async(execution, haystack, needles, output, compare).wait()
}

/// Synchronous comparator-based equal range.
pub fn equal_range_by<H, N, B, E, F, A>(
    execution: &Execution<'_, '_, A>, haystack: H, needles: N, begins: B, ends: E, compare: F,
) -> Result<(), DeviceAlgorithmError>
where
    H: LowerReadRange,
    N: LowerReadRange<Item = H::Item>,
    B: LowerWriteRange<Item = u64>,
    E: LowerWriteRange<Item = u64>,
    F: Fn(&H::Item, &H::Item) -> Ordering + Copy,
    A: DeviceAllocator,
{
    equal_range_by_async(execution, haystack, needles, begins, ends, compare).wait()
}

/// Enqueue natural-order lower bounds. Floating-point keys must use `_by`.
pub fn lower_bound_async<'borrow, 'stream, 'program, H, N, O, A>(
    execution: &'borrow Execution<'stream, 'program, A>, haystack: H, needles: N, output: O,
) -> PendingSearch<'borrow, 'stream, 'program, H, N, O,
    impl Fn(&H::Item, &H::Item) -> Ordering + Copy, A>
where
    H: LowerReadRange,
    H::Item: Ord,
    N: LowerReadRange<Item = H::Item>,
    O: LowerWriteRange<Item = u64>,
    A: DeviceAllocator,
{
    lower_bound_by_async(execution, haystack, needles, output, |a, b| a.cmp(b))
}

/// Enqueue natural-order upper bounds.
pub fn upper_bound_async<'borrow, 'stream, 'program, H, N, O, A>(
    execution: &'borrow Execution<'stream, 'program, A>, haystack: H, needles: N, output: O,
) -> PendingSearch<'borrow, 'stream, 'program, H, N, O,
    impl Fn(&H::Item, &H::Item) -> Ordering + Copy, A>
where
    H: LowerReadRange,
    H::Item: Ord,
    N: LowerReadRange<Item = H::Item>,
    O: LowerWriteRange<Item = u64>,
    A: DeviceAllocator,
{
    upper_bound_by_async(execution, haystack, needles, output, |a, b| a.cmp(b))
}

/// Enqueue natural-order equal ranges into separate begin/end outputs.
pub fn equal_range_async<'borrow, 'stream, 'program, H, N, B, E, A>(
    execution: &'borrow Execution<'stream, 'program, A>, haystack: H, needles: N, begins: B, ends: E,
) -> PendingEqualRange<'borrow, 'stream, 'program, H, N, B, E,
    impl Fn(&H::Item, &H::Item) -> Ordering + Copy, A>
where
    H: LowerReadRange,
    H::Item: Ord,
    N: LowerReadRange<Item = H::Item>,
    B: LowerWriteRange<Item = u64>,
    E: LowerWriteRange<Item = u64>,
    A: DeviceAllocator,
{
    equal_range_by_async(execution, haystack, needles, begins, ends, |a, b| a.cmp(b))
}

/// Write natural-order lower bounds and wait for completion.
pub fn lower_bound<H, N, O, A>(
    execution: &Execution<'_, '_, A>, haystack: H, needles: N, output: O,
) -> Result<(), DeviceAlgorithmError>
where
    H: LowerReadRange,
    H::Item: Ord,
    N: LowerReadRange<Item = H::Item>,
    O: LowerWriteRange<Item = u64>,
    A: DeviceAllocator,
{
    lower_bound_async(execution, haystack, needles, output).wait()
}

/// Write natural-order upper bounds and wait for completion.
pub fn upper_bound<H, N, O, A>(
    execution: &Execution<'_, '_, A>, haystack: H, needles: N, output: O,
) -> Result<(), DeviceAlgorithmError>
where
    H: LowerReadRange,
    H::Item: Ord,
    N: LowerReadRange<Item = H::Item>,
    O: LowerWriteRange<Item = u64>,
    A: DeviceAllocator,
{
    upper_bound_async(execution, haystack, needles, output).wait()
}

/// Write natural-order equal ranges and wait for completion.
pub fn equal_range<H, N, B, E, A>(
    execution: &Execution<'_, '_, A>, haystack: H, needles: N, begins: B, ends: E,
) -> Result<(), DeviceAlgorithmError>
where
    H: LowerReadRange,
    H::Item: Ord,
    N: LowerReadRange<Item = H::Item>,
    B: LowerWriteRange<Item = u64>,
    E: LowerWriteRange<Item = u64>,
    A: DeviceAllocator,
{
    equal_range_async(execution, haystack, needles, begins, ends).wait()
}
