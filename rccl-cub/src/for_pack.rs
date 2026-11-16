//! Pool-owned metadata preparation for the interleaved FOR packer.
use crate::{DeviceAlgorithmError, DeviceReadAt, DeviceWriteAt, ForFrame, cuda::Launch};
use rccl_memory::{ContiguousDeviceRange, DeviceAllocator, TemporaryWorkspace2};

pub struct ForWorkspace<'s, A: DeviceAllocator> {
    storage: TemporaryWorkspace2<'s, A, u64, u64>,
    blocks: usize,
    prepared: bool,
}
impl<A: DeviceAllocator> ForWorkspace<'_, A> {
    pub fn capacity(&self) -> usize { self.blocks }
    pub fn allocation_bytes(&self) -> usize { self.storage.allocation_bytes() }
    /// Publish the used payload length after a completed pack operation.
    pub fn words_to_host(&mut self) -> Result<u64, DeviceAlgorithmError> {
        if !self.prepared { return Err(DeviceAlgorithmError::SizeOverflow); }
        self.storage.parts_mut().1.first_to_host().map_err(DeviceAlgorithmError::Memory)
    }
}
pub fn pack_for_workspace_in<'s, A: DeviceAllocator>(allocator: A, launch: &Launch<'s, '_>, blocks: usize) -> Result<ForWorkspace<'s, A>, DeviceAlgorithmError> {
    let entries = blocks.checked_mul(3).ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let storage = TemporaryWorkspace2::new_in(allocator, launch.stream(), [entries, 1], [8, 8]).map_err(DeviceAlgorithmError::Memory)?;
    Ok(ForWorkspace { storage, blocks, prepared: false })
}

/// Lowered asynchronous pack boundary. Output capacity is in u32 words and
/// checked on device before any writes. Owners retain all captures/allocations
/// until stream completion; use Thrust's completion-retaining safe facade.
#[doc(hidden)]
pub fn device_pack_for_into_in<R, P, F, A>(launch: &Launch<'_, '_>, input: R, payload: P, frames: F, len: usize, capacity: usize, frame_capacity: usize, block_rows: u32, lanes: u32, workspace: &mut ForWorkspace<'_, A>) -> Result<(), DeviceAlgorithmError>
where R: DeviceReadAt<Item = u64>, P: DeviceWriteAt<Item = u32>, F: DeviceWriteAt<Item = ForFrame>, A: DeviceAllocator {
    if !lanes.is_power_of_two() || lanes > 32 || block_rows == 0 || block_rows > lanes * 32 {
        return Err(DeviceAlgorithmError::SizeOverflow);
    }
    let blocks = len.div_ceil(block_rows as usize);
    if blocks > workspace.blocks { return Err(DeviceAlgorithmError::WorkspaceTooSmall { role: "FOR workspace", required: blocks, actual: workspace.blocks }); }
    if blocks > frame_capacity { return Err(DeviceAlgorithmError::WorkspaceTooSmall { role: "FOR frames", required: blocks, actual: frame_capacity }); }
    let (entries, total) = workspace.storage.parts_mut();
    entries.ensure_usable_on(launch.stream()).map_err(DeviceAlgorithmError::Memory)?;
    let module = launch.module().for_pack();
    if blocks != 0 {
        let cfg = cuda_core::LaunchConfig1D::new(blocks.min(65535) as u32, 256, 128);
        let prepared = module.prepare_for_analyze_for(&input, cfg).map_err(DeviceAlgorithmError::Launch)?;
        module.for_analyze(launch.stream(), &prepared, input, len, block_rows, lanes, &mut *entries).map_err(DeviceAlgorithmError::Launch)?;
    }
    let prepared = module.prepare_for_offsets(cuda_core::LaunchConfig1D::new(1, 256, 64)).map_err(DeviceAlgorithmError::Launch)?;
    module.for_offsets(launch.stream(), &prepared, &mut *entries, total, blocks, capacity as u64).map_err(DeviceAlgorithmError::Launch)?;
    if blocks != 0 {
        let cfg = cuda_core::LaunchConfig1D::new(blocks.min(65535) as u32, 256, 0);
        let prepared = module.prepare_for_pack_for(&input, &payload, &frames, cfg).map_err(DeviceAlgorithmError::Launch)?;
        module.for_pack(launch.stream(), &prepared, input, payload, frames, len, block_rows, lanes, entries).map_err(DeviceAlgorithmError::Launch)?;
    }
    workspace.prepared = true;
    Ok(())
}
