//! Workload-facing inclusive and exclusive device prefix sums.

use core::{marker::PhantomData, mem};

use cuda_core::DeviceBuffer;
use rccl_memory::{
    DeviceAllocator, TemporaryRegion, TemporaryValue, TemporaryWorkspace2,
};

use crate::{DeviceAlgorithmError, block_scan::BlockScanValue, cuda::Launch};

const TILE_ITEMS: usize = crate::scan_kernels::SCAN_THREADS as usize;
const MAX_LEVELS: usize = 8;

type Result<T, E = DeviceAlgorithmError> = core::result::Result<T, E>;

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// Unsigned counter type supported by the SRDatalog/HISA scan path.
pub trait DeviceScanValue:
    BlockScanValue + TemporaryValue + sealed::Sealed + 'static
{
}

impl DeviceScanValue for u32 {}
impl DeviceScanValue for u64 {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceScanWorkspaceCounts {
    pub temporary_items: usize,
    pub levels: usize,
}

pub struct DeviceScanWorkspace<'stream, A: DeviceAllocator, T: DeviceScanValue> {
    storage: TemporaryWorkspace2<'stream, A, T, T>,
    counts: DeviceScanWorkspaceCounts,
}

impl<'stream, A, T> DeviceScanWorkspace<'stream, A, T>
where
    A: DeviceAllocator,
    T: DeviceScanValue,
{
    pub fn counts(&self) -> DeviceScanWorkspaceCounts {
        self.counts
    }

    pub fn allocation_bytes(&self) -> usize {
        self.storage.allocation_bytes()
    }

    fn parts_mut(
        &mut self,
    ) -> (
        &mut TemporaryRegion<'stream, T>,
        &mut TemporaryRegion<'stream, T>,
    ) {
        self.storage.parts_mut()
    }
}

#[derive(Clone, Copy)]
struct ScanLayout {
    offsets: [usize; MAX_LEVELS],
    lengths: [usize; MAX_LEVELS],
    levels: usize,
    total: usize,
}

fn scan_layout(len: usize) -> Result<ScanLayout> {
    let mut layout = ScanLayout {
        offsets: [0; MAX_LEVELS],
        lengths: [0; MAX_LEVELS],
        levels: 0,
        total: 0,
    };
    let mut count = len.div_ceil(TILE_ITEMS);
    if count <= 1 {
        return Ok(layout);
    }

    loop {
        if layout.levels == MAX_LEVELS {
            return Err(DeviceAlgorithmError::SizeOverflow);
        }
        layout.offsets[layout.levels] = layout.total;
        layout.lengths[layout.levels] = count;
        layout.total = layout
            .total
            .checked_add(count)
            .ok_or(DeviceAlgorithmError::SizeOverflow)?;
        layout.levels += 1;
        if count <= TILE_ITEMS {
            break;
        }
        count = count.div_ceil(TILE_ITEMS);
    }
    Ok(layout)
}

#[doc(hidden)]
pub fn device_scan_workspace_counts(len: usize) -> Result<DeviceScanWorkspaceCounts> {
    let layout = scan_layout(len)?;
    Ok(DeviceScanWorkspaceCounts {
        temporary_items: layout.total,
        levels: layout.levels,
    })
}

pub fn device_scan_workspace_in<'stream, A, T>(
    allocator: A,
    launch: &Launch<'stream, '_>,
    len: usize,
) -> Result<DeviceScanWorkspace<'stream, A, T>>
where
    A: DeviceAllocator,
    T: DeviceScanValue,
{
    let counts = device_scan_workspace_counts(len)?;
    let storage = TemporaryWorkspace2::new_in(
        allocator,
        launch.stream(),
        [counts.temporary_items, counts.temporary_items],
        [mem::align_of::<T>(), mem::align_of::<T>()],
    )
    .map_err(DeviceAlgorithmError::Memory)?;
    Ok(DeviceScanWorkspace { storage, counts })
}

pub fn device_scan_workspace<'stream, T>(
    launch: &Launch<'stream, '_>,
    len: usize,
) -> Result<
    DeviceScanWorkspace<'stream, rccl_memory::BorrowedCudaMemoryResource<'stream>, T>,
>
where
    T: DeviceScanValue,
{
    let allocator = rccl_memory::BorrowedCudaMemoryResource::new(launch.stream().context())
        .map_err(DeviceAlgorithmError::Memory)?;
    device_scan_workspace_in(allocator, launch, len)
}

#[derive(Clone, Copy)]
struct RawKernelSlice<T> {
    pointer: cuda_core::sys::CUdeviceptr,
    len: usize,
    marker: PhantomData<fn() -> T>,
}

impl<T> RawKernelSlice<T> {
    fn new(pointer: cuda_core::sys::CUdeviceptr, len: usize) -> Self {
        Self {
            pointer,
            len,
            marker: PhantomData,
        }
    }

    fn empty() -> Self {
        Self::new(0, 0)
    }
}

unsafe impl<T> cuda_host::KernelSliceArg for RawKernelSlice<T> {
    type Elem = T;

    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.pointer
    }

    fn len(&self) -> usize {
        self.len
    }
}

unsafe impl<T> cuda_host::KernelSliceArgMut for RawKernelSlice<T> {}

fn raw_subregion<T: DeviceScanValue>(
    region: &TemporaryRegion<'_, T>,
    offset: usize,
    len: usize,
) -> Result<RawKernelSlice<T>> {
    let end = offset
        .checked_add(len)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if end > cuda_host::KernelSliceArg::len(region) {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "scan temporary region",
            required: end,
            actual: cuda_host::KernelSliceArg::len(region),
        });
    }
    let byte_offset = offset
        .checked_mul(mem::size_of::<T>())
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let pointer = cuda_host::KernelSliceArg::cu_deviceptr(region)
        .checked_add(byte_offset as cuda_core::sys::CUdeviceptr)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    Ok(RawKernelSlice::new(pointer, len))
}

fn device_scan_sum_impl<A, T, const EXCLUSIVE: bool>(
    launch: &Launch<'_, '_>,
    pointer: cuda_core::sys::CUdeviceptr,
    len: usize,
    workspace: &mut DeviceScanWorkspace<'_, A, T>,
) -> Result<()>
where
    A: DeviceAllocator,
    T: DeviceScanValue,
{
    let layout = scan_layout(len)?;
    if workspace.counts.temporary_items < layout.total {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "scan temporary items",
            required: layout.total,
            actual: workspace.counts.temporary_items,
        });
    }

    let input = RawKernelSlice::<T>::new(pointer, len);
    let mut output = RawKernelSlice::<T>::new(pointer, len);
    let (partials, prefixes) = workspace.parts_mut();
    let mut first_sums = if layout.levels == 0 {
        RawKernelSlice::empty()
    } else {
        raw_subregion(partials, layout.offsets[0], layout.lengths[0])?
    };
    launch
        .scan_tiles::<T, _, _, _, EXCLUSIVE>(&input, &mut output, &mut first_sums, len)
        .map_err(DeviceAlgorithmError::Launch)?;

    for level in 0..layout.levels {
        let level_len = layout.lengths[level];
        let level_input = raw_subregion(partials, layout.offsets[level], level_len)?;
        let mut level_output = raw_subregion(prefixes, layout.offsets[level], level_len)?;
        let mut next_sums = if level + 1 < layout.levels {
            raw_subregion(
                partials,
                layout.offsets[level + 1],
                layout.lengths[level + 1],
            )?
        } else {
            RawKernelSlice::empty()
        };
        launch
            .scan_tiles::<T, _, _, _, true>(
                &level_input,
                &mut level_output,
                &mut next_sums,
                level_len,
            )
            .map_err(DeviceAlgorithmError::Launch)?;
    }

    if layout.levels > 1 {
        for lower in (0..layout.levels - 1).rev() {
            let mut values = raw_subregion(
                prefixes,
                layout.offsets[lower],
                layout.lengths[lower],
            )?;
            let offsets = raw_subregion(
                prefixes,
                layout.offsets[lower + 1],
                layout.lengths[lower + 1],
            )?;
            launch
                .scan_add_block_offsets::<T, _, _>(&mut values, &offsets, layout.lengths[lower])
                .map_err(DeviceAlgorithmError::Launch)?;
        }
    }

    if layout.levels != 0 {
        let offsets = raw_subregion(prefixes, layout.offsets[0], layout.lengths[0])?;
        launch
            .scan_add_block_offsets::<T, _, _>(&mut output, &offsets, len)
            .map_err(DeviceAlgorithmError::Launch)?;
    }
    Ok(())
}

pub(crate) fn device_exclusive_sum_u32_region<A, Values>(
    launch: &Launch<'_, '_>,
    values: &mut Values,
    workspace: &mut DeviceScanWorkspace<'_, A, u32>,
) -> Result<()>
where
    A: DeviceAllocator,
    Values: cuda_host::KernelSliceArgMut<Elem = u32>,
{
    let len = cuda_host::KernelSliceArg::len(values);
    let tile_count = len.div_ceil(crate::scan_kernels::LOOKBACK_SCAN_TILE_ITEMS);
    let state_items = tile_count
        .checked_add(crate::scan_kernels::LOOKBACK_SCAN_PADDING)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let state_bytes = state_items
        .checked_mul(mem::size_of::<u64>())
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let state = {
        let (storage, _) = workspace.parts_mut();
        let available_bytes = cuda_host::KernelSliceArg::len(storage)
            .checked_mul(mem::size_of::<u32>())
            .ok_or(DeviceAlgorithmError::SizeOverflow)?;
        let pointer = cuda_host::KernelSliceArg::cu_deviceptr(storage);
        if tile_count <= 1
            || available_bytes < state_bytes
            || !(pointer as usize).is_multiple_of(mem::align_of::<u64>())
        {
            None
        } else {
            Some(RawKernelSlice::<u64>::new(pointer, state_items))
        }
    };
    if let Some(mut state) = state {
        launch
            .scan_lookback_u32(values, &mut state, len)
            .map_err(DeviceAlgorithmError::Launch)
    } else {
        device_scan_sum_impl::<A, u32, true>(
            launch,
            cuda_host::KernelSliceArg::cu_deviceptr(values),
            len,
            workspace,
        )
    }
}

pub(crate) fn device_exclusive_sum_region<A, T, Values>(
    launch: &Launch<'_, '_>,
    values: &mut Values,
    workspace: &mut DeviceScanWorkspace<'_, A, T>,
) -> Result<()>
where
    A: DeviceAllocator,
    T: DeviceScanValue,
    Values: cuda_host::KernelSliceArgMut<Elem = T>,
{
    device_scan_sum_impl::<A, T, true>(
        launch,
        cuda_host::KernelSliceArg::cu_deviceptr(values),
        cuda_host::KernelSliceArg::len(values),
        workspace,
    )
}

/// Inclusive wrapping prefix sum in place.
pub fn device_inclusive_sum_in_place<A, T>(
    launch: &Launch<'_, '_>,
    values: &mut DeviceBuffer<T>,
    workspace: &mut DeviceScanWorkspace<'_, A, T>,
) -> Result<()>
where
    A: DeviceAllocator,
    T: DeviceScanValue,
{
    device_scan_sum_impl::<A, T, false>(launch, values.cu_deviceptr(), values.len(), workspace)
}

/// Exclusive wrapping prefix sum with zero identity in place.
pub fn device_exclusive_sum_in_place<A, T>(
    launch: &Launch<'_, '_>,
    values: &mut DeviceBuffer<T>,
    workspace: &mut DeviceScanWorkspace<'_, A, T>,
) -> Result<()>
where
    A: DeviceAllocator,
    T: DeviceScanValue,
{
    device_scan_sum_impl::<A, T, true>(launch, values.cu_deviceptr(), values.len(), workspace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_counts_cover_partial_and_recursive_tiles() {
        assert_eq!(
            device_scan_workspace_counts(0).unwrap(),
            DeviceScanWorkspaceCounts {
                temporary_items: 0,
                levels: 0,
            }
        );
        assert_eq!(device_scan_workspace_counts(256).unwrap().temporary_items, 0);
        assert_eq!(
            device_scan_workspace_counts(257).unwrap(),
            DeviceScanWorkspaceCounts {
                temporary_items: 2,
                levels: 1,
            }
        );
        assert_eq!(
            device_scan_workspace_counts(65_537).unwrap(),
            DeviceScanWorkspaceCounts {
                temporary_items: 259,
                levels: 2,
            }
        );
    }

    #[test]
    fn usize_extent_layout_never_narrows_to_u32() {
        let counts = device_scan_workspace_counts(usize::MAX).unwrap();
        assert!(counts.levels > 1);
        assert!(counts.temporary_items > u32::MAX as usize);
    }
}
