//! Checked range preparation and load-balanced expansion/reduction.
use crate::{DeviceAlgorithmError, DeviceReadAt, DeviceWriteAt, cuda::Launch};
use core::mem;
use rccl_memory::ContiguousDeviceRange;
use rccl_memory::{DeviceAllocator, TemporaryWorkspace2};

/// Reusable row-prefix storage; no decoded values or expanded pairs are retained.
pub struct ExpandWorkspace<'stream, A: DeviceAllocator> {
    storage: TemporaryWorkspace2<'stream, A, u64, u64>,
    capacity: usize,
}
impl<A: DeviceAllocator> ExpandWorkspace<'_, A> {
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn allocation_bytes(&self) -> usize {
        self.storage.allocation_bytes()
    }
    /// Explicit synchronous publication of a previously prepared total.
    /// # Safety
    /// A successful preparation must precede this call on the workspace stream.
    #[doc(hidden)]
    pub unsafe fn total_to_host(&mut self) -> Result<u64, DeviceAlgorithmError> {
        self.storage
            .parts_mut()
            .1
            .first_to_host()
            .map_err(DeviceAlgorithmError::Memory)
    }
}

pub fn expand_ranges_workspace_in<'stream, A: DeviceAllocator>(
    allocator: A,
    launch: &Launch<'stream, '_>,
    ranges: usize,
) -> Result<ExpandWorkspace<'stream, A>, DeviceAlgorithmError> {
    let entries = ranges
        .checked_mul(3)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let tiles = ranges.div_ceil(256);
    let storage = TemporaryWorkspace2::new_in(
        allocator,
        launch.stream(),
        [entries, tiles + 1],
        [mem::align_of::<u64>(); 2],
    )
    .map_err(DeviceAlgorithmError::Memory)?;
    Ok(ExpandWorkspace {
        storage,
        capacity: ranges,
    })
}

/// Internal lowering boundary. Owners must retain all descriptors/captures and
/// workspace on this stream. Invalid endpoints, overflowing counts, or an
/// insufficient output capacity trap on device and surface at completion,
/// matching the existing checked permutation-range failure contract.
#[doc(hidden)]
pub fn device_prepare_ranges_in<A, R, B, F>(
    launch: &Launch<'_, '_>,
    ranges: R,
    bounds: B,
    record: F,
    len: usize,
    output_capacity: u64,
    workspace: &mut ExpandWorkspace<'_, A>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    R: DeviceReadAt,
    B: Fn(R::Item) -> (u64, u64) + Copy,
    F: Fn(usize, R::Item) + Copy,
{
    if len > workspace.capacity {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "range workspace",
            required: len,
            actual: workspace.capacity,
        });
    }
    let tiles = len.div_ceil(256);
    let (entries, totals) = workspace.storage.parts_mut();
    entries
        .ensure_usable_on(launch.stream())
        .map_err(DeviceAlgorithmError::Memory)?;
    let module = launch.module().expand();
    if len != 0 {
        let cfg = cuda_core::LaunchConfig1D::new(tiles.min(65535) as u32, 256, 64);
        let prepared = module
            .prepare_expand_bounds_for(&ranges, &bounds, &record, cfg)
            .map_err(DeviceAlgorithmError::Launch)?;
        module
            .expand_bounds(
                launch.stream(),
                &prepared,
                ranges,
                bounds,
                record,
                len,
                entries,
                totals,
            )
            .map_err(DeviceAlgorithmError::Launch)?;
    }
    let prepared = module
        .prepare_expand_offsets(cuda_core::LaunchConfig1D::new(1, 256, 64))
        .map_err(DeviceAlgorithmError::Launch)?;
    module
        .expand_offsets(launch.stream(), &prepared, totals, tiles, output_capacity)
        .map_err(DeviceAlgorithmError::Launch)?;
    Ok(())
}

/// Materialize stable (range-id, row-id) pairs from a prepared workspace.
#[doc(hidden)]
pub fn device_expand_ranges_in<A, P, R>(
    launch: &Launch<'_, '_>,
    probes: P,
    rows: R,
    len: usize,
    workspace: &mut ExpandWorkspace<'_, A>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    P: DeviceWriteAt<Item = u64>,
    R: DeviceWriteAt<Item = u64>,
{
    if len > workspace.capacity {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "range workspace",
            required: len,
            actual: workspace.capacity,
        });
    }
    let (entries, totals) = workspace.storage.parts_mut();
    entries
        .ensure_usable_on(launch.stream())
        .map_err(DeviceAlgorithmError::Memory)?;
    let module = launch.module().expand();
    let prepared = module
        .prepare_expand_pairs_for(&probes, &rows, cuda_core::LaunchConfig1D::new(1024, 256, 0))
        .map_err(DeviceAlgorithmError::Launch)?;
    module
        .expand_pairs(
            launch.stream(),
            &prepared,
            entries,
            totals,
            len,
            probes,
            rows,
        )
        .map_err(DeviceAlgorithmError::Launch)?;
    Ok(())
}

/// Fused wrapping-u64 reduction without materializing expanded pairs. A source
/// range is selected per interval; the sink receives interval-local partials.
#[doc(hidden)]
pub fn device_reduce_ranges_in<A, S, F>(
    launch: &Launch<'_, '_>,
    sources: S,
    sink: F,
    len: usize,
    workspace: &mut ExpandWorkspace<'_, A>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    S: DeviceReadAt,
    S::Item: DeviceReadAt<Item = u64>,
    F: Fn(usize, u64) + Copy,
{
    if len > workspace.capacity {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "range workspace",
            required: len,
            actual: workspace.capacity,
        });
    }
    let (entries, totals) = workspace.storage.parts_mut();
    entries
        .ensure_usable_on(launch.stream())
        .map_err(DeviceAlgorithmError::Memory)?;
    let module = launch.module().expand();
    let prepared = module
        .prepare_expand_reduce_for(
            &sources,
            &sink,
            cuda_core::LaunchConfig1D::new(1024, 256, 0),
        )
        .map_err(DeviceAlgorithmError::Launch)?;
    module
        .expand_reduce(
            launch.stream(),
            &prepared,
            entries,
            totals,
            len,
            sources,
            sink,
        )
        .map_err(DeviceAlgorithmError::Launch)?;
    Ok(())
}
