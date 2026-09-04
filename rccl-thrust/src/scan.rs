//! Rust-style in-place prefix sums used by SRDatalog/HISA count buffers.

use cuda_core::DeviceBuffer;
use rccl_memory::{CudaCompletion, DeviceAllocator};

use crate::{DeviceAlgorithmError, cuda::Execution};

/// In-flight in-place device scan retaining its values and typed workspace.
pub type PendingScan<'borrow, 'stream, 'program, T, A> = CudaCompletion<
    'borrow,
    (),
    DeviceAlgorithmError,
    (
        &'borrow Execution<'stream, 'program, A>,
        (
            &'borrow mut DeviceBuffer<T>,
            rccl_cub::DeviceScanWorkspace<'stream, &'borrow A, T>,
        ),
    ),
>;

/// Enqueue an inclusive wrapping prefix sum over `u32` or `u64` counters.
pub fn inclusive_scan_async<'borrow, 'stream, 'program, T, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    values: &'borrow mut DeviceBuffer<T>,
) -> Result<PendingScan<'borrow, 'stream, 'program, T, A>, DeviceAlgorithmError>
where
    T: rccl_cub::DeviceScanValue,
    A: DeviceAllocator,
{
    let workspace = rccl_cub::device_scan_workspace_in::<_, T>(
        execution.allocator(),
        execution.launch(),
        values.len(),
    )?;
    // SAFETY: the completion retains the affine values buffer, typed temporary
    // storage, execution program, allocator, and stream used by every launch.
    Ok(unsafe {
        execution.submit((values, workspace), |execution, resources| {
            rccl_cub::device_inclusive_sum_in_place(
                execution.launch(),
                &mut *resources.0,
                &mut resources.1,
            )
        })
    })
}

/// Compute an inclusive wrapping prefix sum in place and synchronize.
pub fn inclusive_scan<T, A>(
    execution: &Execution<'_, '_, A>,
    values: &mut DeviceBuffer<T>,
) -> Result<(), DeviceAlgorithmError>
where
    T: rccl_cub::DeviceScanValue,
    A: DeviceAllocator,
{
    inclusive_scan_async(execution, values)?.wait()
}

/// Enqueue an exclusive wrapping prefix sum with zero identity.
///
/// This is the SRDatalog count/fanout contract. A sentinel input slot can be
/// appended when the caller needs the total as the last exclusive output.
pub fn exclusive_scan_async<'borrow, 'stream, 'program, T, A>(
    execution: &'borrow Execution<'stream, 'program, A>,
    values: &'borrow mut DeviceBuffer<T>,
) -> Result<PendingScan<'borrow, 'stream, 'program, T, A>, DeviceAlgorithmError>
where
    T: rccl_cub::DeviceScanValue,
    A: DeviceAllocator,
{
    let workspace = rccl_cub::device_scan_workspace_in::<_, T>(
        execution.allocator(),
        execution.launch(),
        values.len(),
    )?;
    // SAFETY: the completion retains the affine values buffer, typed temporary
    // storage, execution program, allocator, and stream used by every launch.
    Ok(unsafe {
        execution.submit((values, workspace), |execution, resources| {
            rccl_cub::device_exclusive_sum_in_place(
                execution.launch(),
                &mut *resources.0,
                &mut resources.1,
            )
        })
    })
}

/// Compute an exclusive wrapping prefix sum with zero identity in place.
pub fn exclusive_scan<T, A>(
    execution: &Execution<'_, '_, A>,
    values: &mut DeviceBuffer<T>,
) -> Result<(), DeviceAlgorithmError>
where
    T: rccl_cub::DeviceScanValue,
    A: DeviceAllocator,
{
    exclusive_scan_async(execution, values)?.wait()
}

/// Method syntax for workload counter buffers.
pub trait DeviceScanExt<T: rccl_cub::DeviceScanValue> {
    fn inclusive_scan<A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator;

    fn exclusive_scan<A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator;
}

impl<T: rccl_cub::DeviceScanValue> DeviceScanExt<T> for DeviceBuffer<T> {
    fn inclusive_scan<A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
    {
        inclusive_scan(execution, self)
    }

    fn exclusive_scan<A>(
        &mut self,
        execution: &Execution<'_, '_, A>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
    {
        exclusive_scan(execution, self)
    }
}
