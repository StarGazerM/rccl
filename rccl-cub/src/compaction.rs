//! Private scan/scatter orchestration for workload compaction.

use cuda_core::DeviceBuffer;
use rccl_memory::{DeviceAllocator, TemporaryWorkspace2};

use crate::{
    DeviceAlgorithmError, DeviceScanValue, DeviceScanWorkspace, cuda::Launch,
    scan::{device_exclusive_sum_region, device_exclusive_sum_u32_region},
};

#[doc(hidden)]
pub trait CompactionIndex: DeviceScanValue + PartialEq {
    const GRID_STRIDE: bool;
    fn one() -> Self;
    fn to_usize(self) -> usize;
    fn to_u64(self) -> u64;
}

impl CompactionIndex for u32 {
    const GRID_STRIDE: bool = false;
    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl CompactionIndex for u64 {
    const GRID_STRIDE: bool = true;
    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn to_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn to_u64(self) -> u64 {
        self
    }
}

trait CompactionScanIndex: CompactionIndex {
    fn exclusive_sum<A, Values>(
        launch: &Launch<'_, '_>,
        values: &mut Values,
        workspace: &mut DeviceScanWorkspace<'_, A, Self>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        Values: cuda_host::KernelSliceArgMut<Elem = Self>;
}

impl CompactionScanIndex for u32 {
    fn exclusive_sum<A, Values>(
        launch: &Launch<'_, '_>,
        values: &mut Values,
        workspace: &mut DeviceScanWorkspace<'_, A, Self>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        Values: cuda_host::KernelSliceArgMut<Elem = Self>,
    {
        device_exclusive_sum_u32_region(launch, values, workspace)
    }
}

impl CompactionScanIndex for u64 {
    fn exclusive_sum<A, Values>(
        launch: &Launch<'_, '_>,
        values: &mut Values,
        workspace: &mut DeviceScanWorkspace<'_, A, Self>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        Values: cuda_host::KernelSliceArgMut<Elem = Self>,
    {
        device_exclusive_sum_region(launch, values, workspace)
    }
}

#[doc(hidden)]
pub enum CompactionWorkspace<'stream, A: DeviceAllocator> {
    U32 {
        marks: TemporaryWorkspace2<'stream, A, u32, u32>,
        scan: DeviceScanWorkspace<'stream, A, u32>,
    },
    U64 {
        marks: TemporaryWorkspace2<'stream, A, u64, u64>,
        scan: DeviceScanWorkspace<'stream, A, u64>,
    },
}

impl<A: DeviceAllocator> CompactionWorkspace<'_, A> {
    #[doc(hidden)]
    pub fn index_bits(&self) -> u32 {
        match self {
            Self::U32 { .. } => 32,
            Self::U64 { .. } => 64,
        }
    }

    #[doc(hidden)]
    pub fn allocation_bytes(&self) -> usize {
        match self {
            Self::U32 { marks, scan } => marks.allocation_bytes() + scan.allocation_bytes(),
            Self::U64 { marks, scan } => marks.allocation_bytes() + scan.allocation_bytes(),
        }
    }
}

#[doc(hidden)]
pub const fn compaction_index_bits(len: usize) -> u32 {
    if len <= u32::MAX as usize { 32 } else { 64 }
}
#[doc(hidden)]

pub fn compaction_workspace_in<'stream, A>(
    allocator: A,
    launch: &Launch<'stream, '_>,
    len: usize,
) -> Result<CompactionWorkspace<'stream, A>, DeviceAlgorithmError>
where
    A: DeviceAllocator + Copy,
{
    if compaction_index_bits(len) == 32 {
        let marks = TemporaryWorkspace2::new_in(
            allocator,
            launch.stream(),
            [len, len],
            [core::mem::align_of::<u32>(), core::mem::align_of::<u32>()],
        )
        .map_err(DeviceAlgorithmError::Memory)?;
        let scan = crate::device_scan_workspace_in::<_, u32>(allocator, launch, len)?;
        Ok(CompactionWorkspace::U32 { marks, scan })
    } else {
        let marks = TemporaryWorkspace2::new_in(
            allocator,
            launch.stream(),
            [len, len],
            [core::mem::align_of::<u64>(), core::mem::align_of::<u64>()],
        )
        .map_err(DeviceAlgorithmError::Memory)?;
        let scan = crate::device_scan_workspace_in::<_, u64>(allocator, launch, len)?;
        Ok(CompactionWorkspace::U64 { marks, scan })
    }
}

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

fn select_if_inner<A, Input, Output, Predicate, I>(
    launch: &Launch<'_, '_>,
    input: Input,
    output: Output,
    selected_count: &mut DeviceBuffer<u64>,
    predicate: Predicate,
    len: usize,
    marks: &mut TemporaryWorkspace2<'_, A, I, I>,
    scan: &mut DeviceScanWorkspace<'_, A, I>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    Input: crate::DeviceReadAt,
    Output: crate::DeviceWriteAt<Item = Input::Item>,
    Predicate: Fn(Input::Item) -> bool + Copy,
    I: CompactionScanIndex,
{
    let (flags, positions) = marks.parts_mut();
    launch
        .compaction_mark_predicate(input, flags, positions, predicate, len)
        .map_err(DeviceAlgorithmError::Launch)?;
    I::exclusive_sum(launch, positions, scan)?;
    launch
        .compaction_scatter_selected(input, output, flags, positions, selected_count, len)
        .map_err(DeviceAlgorithmError::Launch)?;
    Ok(())
}

#[doc(hidden)]
pub fn device_select_if<A, Input, Output, Predicate>(
    launch: &Launch<'_, '_>,
    input: Input,
    output: Output,
    selected_count: &mut DeviceBuffer<u64>,
    predicate: Predicate,
    len: usize,
    workspace: &mut CompactionWorkspace<'_, A>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    Input: crate::DeviceReadAt,
    Output: crate::DeviceWriteAt<Item = Input::Item>,
    Predicate: Fn(Input::Item) -> bool + Copy,
{
    validate_count(selected_count)?;
    match workspace {
        CompactionWorkspace::U32 { marks, scan } => select_if_inner::<_, _, _, _, u32>(
            launch, input, output, selected_count, predicate, len, marks, scan,
        ),
        CompactionWorkspace::U64 { marks, scan } => select_if_inner::<_, _, _, _, u64>(
            launch, input, output, selected_count, predicate, len, marks, scan,
        ),
    }
}

fn select_flagged_inner<A, Input, Flags, Output, I>(
    launch: &Launch<'_, '_>,
    input: Input,
    selected: Flags,
    output: Output,
    selected_count: &mut DeviceBuffer<u64>,
    len: usize,
    marks: &mut TemporaryWorkspace2<'_, A, I, I>,
    scan: &mut DeviceScanWorkspace<'_, A, I>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    Input: crate::DeviceReadAt,
    Flags: crate::DeviceReadAt<Item = u32>,
    Output: crate::DeviceWriteAt<Item = Input::Item>,
    I: CompactionScanIndex,
{
    let (flags, positions) = marks.parts_mut();
    launch
        .compaction_mark_flagged(selected, flags, positions, len)
        .map_err(DeviceAlgorithmError::Launch)?;
    I::exclusive_sum(launch, positions, scan)?;
    launch
        .compaction_scatter_selected(input, output, flags, positions, selected_count, len)
        .map_err(DeviceAlgorithmError::Launch)?;
    Ok(())
}

#[doc(hidden)]
pub fn device_select_flagged<A, Input, Flags, Output>(
    launch: &Launch<'_, '_>,
    input: Input,
    selected: Flags,
    output: Output,
    selected_count: &mut DeviceBuffer<u64>,
    len: usize,
    workspace: &mut CompactionWorkspace<'_, A>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    Input: crate::DeviceReadAt,
    Flags: crate::DeviceReadAt<Item = u32>,
    Output: crate::DeviceWriteAt<Item = Input::Item>,
{
    validate_count(selected_count)?;
    match workspace {
        CompactionWorkspace::U32 { marks, scan } => select_flagged_inner::<_, _, _, _, u32>(
            launch, input, selected, output, selected_count, len, marks, scan,
        ),
        CompactionWorkspace::U64 { marks, scan } => select_flagged_inner::<_, _, _, _, u64>(
            launch, input, selected, output, selected_count, len, marks, scan,
        ),
    }
}

fn unique_inner<A, Input, Output, Equivalent, I>(
    launch: &Launch<'_, '_>,
    input: Input,
    output: Output,
    selected_count: &mut DeviceBuffer<u64>,
    equivalent: Equivalent,
    len: usize,
    marks: &mut TemporaryWorkspace2<'_, A, I, I>,
    scan: &mut DeviceScanWorkspace<'_, A, I>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    Input: crate::DeviceReadAt,
    Output: crate::DeviceWriteAt<Item = Input::Item>,
    Equivalent: Fn(Input::Item, Input::Item) -> bool + Copy,
    I: CompactionScanIndex,
{
    let (flags, positions) = marks.parts_mut();
    launch
        .compaction_mark_adjacent_unique(input, flags, positions, equivalent, len)
        .map_err(DeviceAlgorithmError::Launch)?;
    I::exclusive_sum(launch, positions, scan)?;
    launch
        .compaction_scatter_selected(input, output, flags, positions, selected_count, len)
        .map_err(DeviceAlgorithmError::Launch)?;
    Ok(())
}

#[doc(hidden)]
pub fn device_unique_copy_by<A, Input, Output, Equivalent>(
    launch: &Launch<'_, '_>,
    input: Input,
    output: Output,
    selected_count: &mut DeviceBuffer<u64>,
    equivalent: Equivalent,
    len: usize,
    workspace: &mut CompactionWorkspace<'_, A>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    Input: crate::DeviceReadAt,
    Output: crate::DeviceWriteAt<Item = Input::Item>,
    Equivalent: Fn(Input::Item, Input::Item) -> bool + Copy,
{
    validate_count(selected_count)?;
    match workspace {
        CompactionWorkspace::U32 { marks, scan } => unique_inner::<_, _, _, _, u32>(
            launch, input, output, selected_count, equivalent, len, marks, scan,
        ),
        CompactionWorkspace::U64 { marks, scan } => unique_inner::<_, _, _, _, u64>(
            launch, input, output, selected_count, equivalent, len, marks, scan,
        ),
    }
}

fn partition_inner<A, Input, Output, Predicate, I>(
    launch: &Launch<'_, '_>,
    input: Input,
    output: Output,
    selected_count: &mut DeviceBuffer<u64>,
    predicate: Predicate,
    len: usize,
    marks: &mut TemporaryWorkspace2<'_, A, I, I>,
    scan: &mut DeviceScanWorkspace<'_, A, I>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    Input: crate::DeviceReadAt,
    Output: crate::DeviceWriteAt<Item = Input::Item>,
    Predicate: Fn(Input::Item) -> bool + Copy,
    I: CompactionScanIndex,
{
    let (flags, positions) = marks.parts_mut();
    launch
        .compaction_mark_predicate(input, flags, positions, predicate, len)
        .map_err(DeviceAlgorithmError::Launch)?;
    I::exclusive_sum(launch, positions, scan)?;
    launch
        .compaction_scatter_partition(input, output, flags, positions, selected_count, len)
        .map_err(DeviceAlgorithmError::Launch)?;
    Ok(())
}

#[doc(hidden)]
pub fn device_stable_partition_by<A, Input, Output, Predicate>(
    launch: &Launch<'_, '_>,
    input: Input,
    output: Output,
    selected_count: &mut DeviceBuffer<u64>,
    predicate: Predicate,
    len: usize,
    workspace: &mut CompactionWorkspace<'_, A>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    Input: crate::DeviceReadAt,
    Output: crate::DeviceWriteAt<Item = Input::Item>,
    Predicate: Fn(Input::Item) -> bool + Copy,
{
    validate_count(selected_count)?;
    match workspace {
        CompactionWorkspace::U32 { marks, scan } => partition_inner::<_, _, _, _, u32>(
            launch, input, output, selected_count, predicate, len, marks, scan,
        ),
        CompactionWorkspace::U64 { marks, scan } => partition_inner::<_, _, _, _, u64>(
            launch, input, output, selected_count, predicate, len, marks, scan,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::compaction_index_bits;

    #[test]
    fn common_and_large_extent_index_widths_do_not_narrow() {
        assert_eq!(compaction_index_bits(0), 32);
        assert_eq!(compaction_index_bits(u32::MAX as usize), 32);
        if usize::BITS > 32 {
            assert_eq!(compaction_index_bits(u32::MAX as usize + 1), 64);
        }
    }
}
