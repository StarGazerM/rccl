//! Checked stable expansion and fused wrapping-u64 range sums.
//!
//! Endpoints are half-open. Reversed ranges, total overflow, and device output
//! capacity failures are device faults reported by completion, as with checked
//! permutation ranges. A fault may invalidate the CUDA context. Host shape and
//! workspace errors are returned without launching work. Expanded outputs are
//! probe-major; unused capacity is untouched. No host total is read by async APIs.
use crate::{
    DeviceAlgorithmError, DeviceRange,
    cuda::Execution,
    range::{LowerReadRange, LowerWriteRange},
};
use rccl_cub::{DeviceAtomicAddU64, DeviceWriteAt, ExpandWorkspace};
use rccl_memory::{CudaCompletion, DeviceAllocator};

pub type PendingExpand<'b, 's, 'p, B, E, P, R, A, W> = CudaCompletion<
    'b,
    (),
    DeviceAlgorithmError,
    (
        &'b Execution<'s, 'p, A>,
        (B, E, P, R, &'b mut ExpandWorkspace<'s, W>),
    ),
>;
pub type PendingRangeSum<'b, 's, 'p, V, B, E, O, A, W> = CudaCompletion<
    'b,
    (),
    DeviceAlgorithmError,
    (
        &'b Execution<'s, 'p, A>,
        (V, B, E, O, &'b mut ExpandWorkspace<'s, W>),
    ),
>;

pub fn expand_ranges_workspace<'b, 's, A: DeviceAllocator>(
    exec: &'b Execution<'s, '_, A>,
    ranges: usize,
) -> Result<ExpandWorkspace<'s, &'b A>, DeviceAlgorithmError> {
    rccl_cub::expand_ranges_workspace_in(exec.allocator(), exec.launch(), ranges)
}

fn validate<B: LowerReadRange<Item = u64>, E: LowerReadRange<Item = u64>>(
    begins: &B,
    ends: &E,
) -> Result<(), DeviceAlgorithmError> {
    if begins.len() != ends.len() {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "range endpoints",
            expected: begins.len(),
            actual: ends.len(),
        });
    }
    begins
        .validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    ends.validate_lowering()
        .map_err(|_| DeviceAlgorithmError::SizeOverflow)
}

/// Enqueue checked stable expansion while retaining every input/output borrow.
/// The workspace cannot be reused until the returned completion is released.
/// ```compile_fail
/// use cuda_core::DeviceBuffer;
/// use rccl_memory::DeviceAllocator;
/// use rccl_thrust::{cuda::Execution, DeviceBufferExt, expand_ranges_into_async};
/// fn overlap<'s, A: DeviceAllocator>(exec: &Execution<'s, '_, A>,
///     b: &DeviceBuffer<u64>, e: &DeviceBuffer<u64>, out: &mut DeviceBuffer<u64>,
///     rows: &mut DeviceBuffer<u64>, ws: &mut rccl_cub::ExpandWorkspace<'s, A>) {
///     let pending = expand_ranges_into_async(exec, b.view(), e.view(), out.view_mut(), rows.view_mut(), ws);
///     let second = out.view_mut();
///     pending.wait().unwrap();
///     drop(second);
/// }
/// ```
pub fn expand_ranges_into_async<'b, 's, 'p, B, E, P, R, A, W>(
    exec: &'b Execution<'s, 'p, A>,
    begins: B,
    ends: E,
    probes: P,
    rows: R,
    workspace: &'b mut ExpandWorkspace<'s, W>,
) -> PendingExpand<'b, 's, 'p, B, E, P, R, A, W>
where
    A: DeviceAllocator,
    W: DeviceAllocator,
    B: LowerReadRange<Item = u64>,
    E: LowerReadRange<Item = u64>,
    P: LowerWriteRange<Item = u64>,
    R: LowerWriteRange<Item = u64>,
{
    unsafe {
        exec.submit(
            (begins, ends, probes, rows, workspace),
            |exec, resources| {
                let (begins, ends, probes, rows, workspace) = resources;
                validate(begins, ends)?;
                if probes.len() != rows.len() {
                    return Err(DeviceAlgorithmError::LengthMismatch {
                        role: "expanded outputs",
                        expected: probes.len(),
                        actual: rows.len(),
                    });
                }
                probes
                    .validate_lowering()
                    .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
                rows.validate_lowering()
                    .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
                let ranges = rccl_cub::DeviceZip::new(begins.lower_read(), ends.lower_read());
                rccl_cub::device_prepare_ranges_in(
                    exec.launch(),
                    ranges,
                    |r| r,
                    |_, _| {},
                    begins.len(),
                    probes.len() as u64,
                    workspace,
                )?;
                rccl_cub::device_expand_ranges_in(
                    exec.launch(),
                    probes.lower_write(),
                    rows.lower_write(),
                    begins.len(),
                    workspace,
                )
            },
        )
    }
}

pub fn expand_ranges_into<'s, B, E, P, R, A, W>(
    exec: &Execution<'s, '_, A>,
    begins: B,
    ends: E,
    probes: P,
    rows: R,
    workspace: &mut ExpandWorkspace<'s, W>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    W: DeviceAllocator,
    B: LowerReadRange<Item = u64>,
    E: LowerReadRange<Item = u64>,
    P: LowerWriteRange<Item = u64>,
    R: LowerWriteRange<Item = u64>,
{
    expand_ranges_into_async(exec, begins, ends, probes, rows, workspace).wait()
}

/// Explicit synchronous count publication; expansion itself needs no readback.
pub fn count_ranges<'s, B, E, A, W>(
    exec: &Execution<'s, '_, A>,
    begins: B,
    ends: E,
    workspace: &mut ExpandWorkspace<'s, W>,
) -> Result<u64, DeviceAlgorithmError>
where
    A: DeviceAllocator,
    W: DeviceAllocator,
    B: LowerReadRange<Item = u64>,
    E: LowerReadRange<Item = u64>,
{
    unsafe {
        exec.submit((begins, ends, &mut *workspace), |exec, resources| {
            let (begins, ends, workspace) = resources;
            validate(begins, ends)?;
            let ranges = rccl_cub::DeviceZip::new(begins.lower_read(), ends.lower_read());
            rccl_cub::device_prepare_ranges_in(
                exec.launch(),
                ranges,
                |r| r,
                |_, _| {},
                begins.len(),
                u64::MAX,
                workspace,
            )
        })
    }
    .wait()?;
    // Preparation completed successfully, including the checked total write.
    unsafe { workspace.total_to_host() }
}

/// Sum a u64 value range over each interval without expanded row-id storage.
/// Inputs can be lazy projections; sums wrap modulo 2^64. Overlaps are counted
/// independently. Atomic-capable affine outputs include slices/reverse/stride.
pub fn sum_ranges_into_async<'b, 's, 'p, V, B, E, O, A, W>(
    exec: &'b Execution<'s, 'p, A>,
    values: V,
    begins: B,
    ends: E,
    output: O,
    workspace: &'b mut ExpandWorkspace<'s, W>,
) -> PendingRangeSum<'b, 's, 'p, V, B, E, O, A, W>
where
    A: DeviceAllocator,
    W: DeviceAllocator,
    V: LowerReadRange<Item = u64>,
    B: LowerReadRange<Item = u64>,
    E: LowerReadRange<Item = u64>,
    O: LowerWriteRange<Item = u64>,
    O::Device: DeviceAtomicAddU64,
{
    unsafe {
        exec.submit(
            (values, begins, ends, output, workspace),
            |exec, resources| {
                let (values, begins, ends, output, workspace) = resources;
                validate(begins, ends)?;
                if output.len() != begins.len() {
                    return Err(DeviceAlgorithmError::LengthMismatch {
                        role: "range sums",
                        expected: begins.len(),
                        actual: output.len(),
                    });
                }
                values
                    .validate_lowering()
                    .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
                output
                    .validate_lowering()
                    .map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
                let ranges = rccl_cub::DeviceZip::new(begins.lower_read(), ends.lower_read());
                let out = output.lower_write();
                let value_count = values.len() as u64;
                rccl_cub::device_prepare_ranges_in(
                    exec.launch(),
                    ranges,
                    move |r: (u64, u64)| {
                        if r.1 > value_count {
                            cuda_device::debug::trap();
                        }
                        r
                    },
                    move |i, _| {
                        out.write_at(i, 0);
                    },
                    begins.len(),
                    u64::MAX,
                    workspace,
                )?;
                let sources = rccl_cub::DeviceConstant::new(values.lower_read());
                rccl_cub::device_reduce_ranges_in(
                    exec.launch(),
                    sources,
                    move |i, value| {
                        out.add_at(i, value);
                    },
                    begins.len(),
                    workspace,
                )
            },
        )
    }
}

pub fn sum_ranges_into<'s, V, B, E, O, A, W>(
    exec: &Execution<'s, '_, A>,
    values: V,
    begins: B,
    ends: E,
    output: O,
    workspace: &mut ExpandWorkspace<'s, W>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    W: DeviceAllocator,
    V: LowerReadRange<Item = u64>,
    B: LowerReadRange<Item = u64>,
    E: LowerReadRange<Item = u64>,
    O: LowerWriteRange<Item = u64>,
    O::Device: DeviceAtomicAddU64,
{
    sum_ranges_into_async(exec, values, begins, ends, output, workspace).wait()
}
