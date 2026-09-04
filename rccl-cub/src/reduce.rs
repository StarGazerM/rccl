//! CUB DeviceReduce support.

use cuda_core::DeviceBuffer;
use rccl_memory::{DeviceAllocator, TemporaryRegion, TemporaryValue, TemporaryWorkspace1};

use crate::{DeviceAlgorithmError, cuda::Launch, reduce_kernels};

pub use reduce_kernels::ReduceSumValue;

const ITEMS_PER_THREAD: usize = reduce_kernels::REDUCE_NOMINAL_4B_ITEMS;

type Result<T, E = DeviceAlgorithmError> = core::result::Result<T, E>;

struct RawKernelSlice<T> {
    ptr: cuda_core::sys::CUdeviceptr,
    len: usize,
    marker: core::marker::PhantomData<T>,
}

impl<T> RawKernelSlice<T> {
    fn new(ptr: cuda_core::sys::CUdeviceptr, len: usize) -> Self {
        Self {
            ptr,
            len,
            marker: core::marker::PhantomData,
        }
    }
}

unsafe impl<T> cuda_host::KernelSliceArg for RawKernelSlice<T> {
    type Elem = T;

    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

unsafe impl<T> cuda_host::KernelSliceArgMut for RawKernelSlice<T> {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceReduceSumWorkspaceCounts {
    pub partials: usize,
}

pub struct DeviceReduceSumWorkspace<'stream, A: DeviceAllocator, T: TemporaryValue> {
    storage: TemporaryWorkspace1<'stream, A, T>,
    counts: DeviceReduceSumWorkspaceCounts,
}

impl<'stream, A, T> DeviceReduceSumWorkspace<'stream, A, T>
where
    A: DeviceAllocator,
    T: TemporaryValue,
{
    pub fn counts(&self) -> DeviceReduceSumWorkspaceCounts {
        self.counts
    }

    pub fn allocation_bytes(&self) -> usize {
        self.storage.allocation_bytes()
    }

    pub(crate) fn partials_mut(&mut self) -> &mut TemporaryRegion<'stream, T> {
        let (partials,) = self.storage.parts_mut();
        partials
    }
}

#[doc(hidden)]
pub fn device_reduce_sum_workspace_counts<T>(
    launch: &Launch<'_, '_>,
    len: usize,
) -> Result<DeviceReduceSumWorkspaceCounts>
where
    T: ReduceSumValue + TemporaryValue,
{
    let geometry = reduce_geometry::<T>(launch, len)?;
    Ok(DeviceReduceSumWorkspaceCounts {
        partials: geometry.partial_capacity,
    })
}

pub fn device_reduce_sum_workspace_in<'stream, A, T>(
    allocator: A,
    launch: &Launch<'stream, '_>,
    len: usize,
) -> Result<DeviceReduceSumWorkspace<'stream, A, T>>
where
    A: DeviceAllocator,
    T: ReduceSumValue + TemporaryValue,
{
    let counts = device_reduce_sum_workspace_counts::<T>(launch, len)?;
    let storage = TemporaryWorkspace1::new_in(
        allocator,
        launch.stream(),
        [counts.partials],
        [core::mem::align_of::<T>()],
    )
    .map_err(DeviceAlgorithmError::Memory)?;
    Ok(DeviceReduceSumWorkspace { storage, counts })
}

pub fn device_reduce_sum_workspace<'stream, T>(
    launch: &Launch<'stream, '_>,
    len: usize,
) -> Result<DeviceReduceSumWorkspace<'stream, rccl_memory::BorrowedCudaMemoryResource<'stream>, T>>
where
    T: ReduceSumValue + TemporaryValue,
{
    let allocator = rccl_memory::BorrowedCudaMemoryResource::new(launch.stream().context())
        .map_err(DeviceAlgorithmError::Memory)?;
    device_reduce_sum_workspace_in(allocator, launch, len)
}

pub fn device_reduce_sum_in<A, T, Output>(
    launch: &Launch<'_, '_>,
    input: &DeviceBuffer<T>,
    output: &mut Output,
    workspace: &mut DeviceReduceSumWorkspace<'_, A, T>,
) -> Result<()>
where
    A: DeviceAllocator,
    T: ReduceSumValue + TemporaryValue + 'static,
    Output: cuda_host::KernelSliceArgMut<Elem = T>,
{
    let geometry = reduce_geometry::<T>(launch, input.len())?;
    if workspace.counts.partials < geometry.partial_capacity {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "reduce partials",
            required: geometry.partial_capacity,
            actual: workspace.counts.partials,
        });
    }
    if output.len() < 1 {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "reduce output",
            required: 1,
            actual: output.len(),
        });
    }
    if core::any::TypeId::of::<T>() == core::any::TypeId::of::<i32>() {
        let input_i32 = RawKernelSlice::<i32>::new(
            cuda_host::KernelSliceArg::cu_deviceptr(input),
            cuda_host::KernelSliceArg::len(input),
        );
        let mut output_i32 = RawKernelSlice::<i32>::new(
            cuda_host::KernelSliceArg::cu_deviceptr(output),
            cuda_host::KernelSliceArg::len(output),
        );
        if geometry.grid_size <= 1 {
            return launch
                .reduce_sum_single_tile_i32_as_u32(&input_i32, input.len(), &mut output_i32)
                .map_err(DeviceAlgorithmError::Launch);
        }

        let partials = workspace.partials_mut();
        let mut partials_i32 = RawKernelSlice::<i32>::new(
            cuda_host::KernelSliceArg::cu_deviceptr(partials),
            cuda_host::KernelSliceArg::len(partials),
        );
        launch
            .reduce_sum_blocks_i32_as_u32(
                &input_i32,
                input.len(),
                &mut partials_i32,
                geometry.grid_size,
            )
            .map_err(DeviceAlgorithmError::Launch)?;
        return launch
            .reduce_sum_single_tile_i32_as_u32(
                &partials_i32,
                geometry.grid_size as usize,
                &mut output_i32,
            )
            .map_err(DeviceAlgorithmError::Launch);
    }

    if geometry.grid_size <= 1 {
        launch
            .reduce_sum_single_tile::<T, _, _>(input, input.len(), output)
            .map_err(DeviceAlgorithmError::Launch)
    } else {
        let partials = workspace.partials_mut();
        launch
            .reduce_sum_blocks::<T, _, _>(input, input.len(), partials, geometry.grid_size)
            .map_err(DeviceAlgorithmError::Launch)?;
        launch
            .reduce_sum_single_tile::<T, _, _>(partials, geometry.grid_size as usize, output)
            .map_err(DeviceAlgorithmError::Launch)
    }
}

pub fn device_reduce_sum<T>(launch: &Launch<'_, '_>, input: &DeviceBuffer<T>) -> Result<T>
where
    T: ReduceSumValue + TemporaryValue + 'static,
{
    let allocator = rccl_memory::BorrowedCudaMemoryResource::new(launch.stream().context())
        .map_err(DeviceAlgorithmError::Memory)?;
    let mut output = rccl_memory::DeviceScalar::new_in(allocator, launch.stream(), T::default())
        .map_err(DeviceAlgorithmError::Memory)?;
    let mut workspace = device_reduce_sum_workspace::<T>(launch, input.len())?;
    device_reduce_sum_in(launch, input, output.as_buffer_mut(), &mut workspace)?;
    output.get().map_err(DeviceAlgorithmError::Memory)
}

#[derive(Clone, Copy)]
struct ReduceGeometry {
    grid_size: u32,
    partial_capacity: usize,
}

fn reduce_geometry<T>(launch: &Launch<'_, '_>, len: usize) -> Result<ReduceGeometry>
where
    T: ReduceSumValue + TemporaryValue,
{
    let tile_items = reduce_kernels::reduce_tile_items::<T>();
    if len <= tile_items {
        return Ok(ReduceGeometry {
            grid_size: 1,
            partial_capacity: 0,
        });
    }

    let shared_bytes = reduce_shared_bytes::<T>()?;
    let probe_launch =
        cuda_core::LaunchConfig1D::new(1, reduce_kernels::REDUCE_THREADS, shared_bytes);
    let witness = T::default();
    let prepared = launch
        .module()
        .reduce()
        .prepare_device_reduce_sum_blocks_for::<T, ITEMS_PER_THREAD>(&witness, probe_launch)
        .map_err(DeviceAlgorithmError::Launch)?;
    // Pinned CUB's SM89 policy for this 256-thread, 16-item, warp-reductions
    // kernel reports six resident blocks per SM. cuda-oxide has reported five
    // for some equivalent Rust monomorphizations, which shrinks the
    // subscription grid and temp storage below CUB. The value below is only
    // CUB's grid cap: it sizes/launches partial CTAs and does not override the
    // CUDA scheduler's true residency limit.
    let measured_blocks_per_sm = prepared
        .function()
        .max_active_blocks_per_multiprocessor(reduce_kernels::REDUCE_THREADS, shared_bytes)
        .map_err(DeviceAlgorithmError::Driver)?;
    let blocks_per_sm = measured_blocks_per_sm.max(reduce_kernels::REDUCE_CUB_SM89_BLOCKS_PER_SM);
    let sms = launch
        .stream()
        .context()
        .multiprocessor_count()
        .map_err(DeviceAlgorithmError::Driver)?;
    let resident = blocks_per_sm
        .checked_mul(sms)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let max_blocks = resident
        .checked_mul(reduce_kernels::REDUCE_SUBSCRIPTION_FACTOR)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if max_blocks == 0 {
        return Err(DeviceAlgorithmError::UnsupportedTarget(
            "DeviceReduce::Sum kernel has zero resident occupancy",
        ));
    }
    let total_tiles = len.div_ceil(tile_items);
    let grid_size = total_tiles.min(max_blocks as usize);
    let grid_size = u32::try_from(grid_size).map_err(|_| DeviceAlgorithmError::SizeOverflow)?;
    Ok(ReduceGeometry {
        grid_size,
        partial_capacity: max_blocks as usize,
    })
}

fn reduce_shared_bytes<T>() -> Result<u32> {
    let bytes = 8usize
        .checked_mul(core::mem::size_of::<T>())
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    u32::try_from(bytes).map_err(|_| DeviceAlgorithmError::SizeOverflow)
}
