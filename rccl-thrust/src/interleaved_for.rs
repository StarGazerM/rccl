//! Borrow-retaining interleaved FOR range and packing operations.
use crate::{DeviceAlgorithmError, DeviceRange, RandomAccessRange, ReadRange, RangeError, cuda::Execution, range::{LowerReadRange, LowerWriteRange}};
use rccl_cub::{ForFrame, ForWorkspace};
use rccl_memory::{CudaCompletion, DeviceAllocator};

/// A lazily decoded column. Payload and frame ranges retain their own borrows.
#[derive(Clone, Copy)]
pub struct InterleavedForRange<P, F> {
    payload: P,
    frames: F,
    len: usize,
    block_rows: u32,
    lanes: u32,
}
impl<P: LowerReadRange<Item = u32>, F: LowerReadRange<Item = ForFrame>> InterleavedForRange<P, F> {
    /// # Safety
    /// Frames must be produced by the FOR packer (or its exact format), with
    /// matching geometry and row count, initialized payload offsets in bounds,
    /// widths 0..64 and bases whose decoded values do not overflow. Both ranges
    /// must remain immutable until all operations consuming this view complete.
    pub unsafe fn new(payload: P, frames: F, len: usize, block_rows: u32, lanes: u32) -> Result<Self, RangeError> {
        if !lanes.is_power_of_two() || lanes > 32 || block_rows > lanes * 32 { return Err(RangeError::BoundOverflow); }
        if block_rows == 0 { return Err(RangeError::ZeroStep); }
        let blocks = len.div_ceil(block_rows as usize);
        if frames.len() < blocks { return Err(RangeError::OutOfBounds { start: 0, end: blocks, len: frames.len() }); }
        payload.validate_lowering()?; frames.validate_lowering()?;
        Ok(Self { payload, frames, len, block_rows, lanes })
    }
}
impl<P, F> DeviceRange for InterleavedForRange<P, F> { fn len(&self) -> usize { self.len } }
impl<P, F> ReadRange for InterleavedForRange<P, F> { type Item = u64; }
impl<P, F> RandomAccessRange for InterleavedForRange<P, F> {}
impl<P: LowerReadRange<Item = u32>, F: LowerReadRange<Item = ForFrame>> LowerReadRange for InterleavedForRange<P, F> {
    type Device = rccl_cub::DeviceInterleavedForRange<P::Device, F::Device>;
    fn lower_read(&self) -> Self::Device {
        rccl_cub::DeviceInterleavedForRange { payload: self.payload.lower_read(), frames: self.frames.lower_read(), block_rows: self.block_rows, lanes: self.lanes }
    }
}

pub type PendingForPack<'b, 's, 'p, R, P, F, A, W> = CudaCompletion<'b, (), DeviceAlgorithmError, (&'b Execution<'s, 'p, A>, (R, P, F, &'b mut ForWorkspace<'s, W>))>;

pub fn pack_for_workspace<'b, 's, A: DeviceAllocator>(exec: &'b Execution<'s, '_, A>, blocks: usize) -> Result<ForWorkspace<'s, &'b A>, DeviceAlgorithmError> {
    rccl_cub::pack_for_workspace_in(exec.allocator(), exec.launch(), blocks)
}

/// Pack an arbitrary lowered u64 input. Payload capacity may exceed the used
/// length; unused output is untouched. Capacity failures are device faults
/// reported by completion, matching checked permutation/expansion APIs.
/// ```compile_fail
/// use cuda_core::DeviceBuffer;
/// use rccl_memory::DeviceAllocator;
/// use rccl_thrust::{cuda::Execution, DeviceBufferExt, ForFrame, ForWorkspace, pack_for_into_async};
/// fn overlap<'s, A: DeviceAllocator>(exec: &Execution<'s, '_, A>,
///     input: &DeviceBuffer<u64>, payload: &mut DeviceBuffer<u32>,
///     frames: &mut DeviceBuffer<ForFrame>, ws: &mut ForWorkspace<'s, A>) {
///     let pending = pack_for_into_async(exec, input.view(), payload.view_mut(), frames.view_mut(), 256, 8, ws);
///     let overlapping = payload.view_mut();
///     pending.wait().unwrap();
///     drop(overlapping);
/// }
/// ```
pub fn pack_for_into_async<'b, 's, 'p, R, P, F, A, W>(exec: &'b Execution<'s, 'p, A>, input: R, payload: P, frames: F, block_rows: u32, lanes: u32, workspace: &'b mut ForWorkspace<'s, W>) -> PendingForPack<'b, 's, 'p, R, P, F, A, W>
where A: DeviceAllocator, W: DeviceAllocator, R: LowerReadRange<Item = u64>, P: LowerWriteRange<Item = u32>, F: LowerWriteRange<Item = ForFrame> {
    unsafe { exec.submit((input, payload, frames, workspace), |exec, resources| {
        let (input, payload, frames, workspace) = resources;
        input.validate_lowering().map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
        payload.validate_lowering().map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
        frames.validate_lowering().map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
        rccl_cub::device_pack_for_into_in(exec.launch(), input.lower_read(), payload.lower_write(), frames.lower_write(), input.len(), payload.len(), frames.len(), block_rows, lanes, workspace)
    }) }
}

pub fn pack_for_into<'s, R, P, F, A, W>(exec: &Execution<'s, '_, A>, input: R, payload: P, frames: F, block_rows: u32, lanes: u32, workspace: &mut ForWorkspace<'s, W>) -> Result<(), DeviceAlgorithmError>
where A: DeviceAllocator, W: DeviceAllocator, R: LowerReadRange<Item = u64>, P: LowerWriteRange<Item = u32>, F: LowerWriteRange<Item = ForFrame> {
    pack_for_into_async(exec, input, payload, frames, block_rows, lanes, workspace).wait()
}
