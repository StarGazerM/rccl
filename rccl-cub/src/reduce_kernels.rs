//! Library-owned CUDA entries for CUB DeviceReduce.

use cuda_device::cuda_module;

pub(crate) const REDUCE_THREADS: u32 = 256;
pub(crate) const REDUCE_NOMINAL_4B_ITEMS: usize = 16;
pub(crate) const REDUCE_SUBSCRIPTION_FACTOR: u32 = 5;
pub(crate) const REDUCE_CUB_SM89_BLOCKS_PER_SM: u32 = 6;

pub trait ReduceSumValue:
    cuda_core::DeviceCopy + Copy + Default + core::ops::Add<Output = Self> + crate::warp::WarpValue
{
    fn reduce_full_tile<const ITEMS: usize>(
        input: &[Self],
        tile_offset: usize,
        thread_index: usize,
        aggregate: &mut Self,
    );

    fn reduce_partial_tile<const ITEMS: usize>(
        input: &[Self],
        tile_offset: usize,
        tile_end: usize,
        thread_index: usize,
        aggregate: &mut Self,
    );

    fn warp_reduce_sum_32(value: Self) -> Self;
    fn warp_reduce_sum_8(value: Self) -> Self;
}

#[inline(always)]
fn redux_sync_add_u32(value: u32) -> u32 {
    let result: u32;
    unsafe {
        cuda_device::ptx_asm!(
            "redux.sync.add.u32 %0, %1, 0xffffffff;",
            out("=r") result,
            in("r") value,
            options(register_only),
        );
    }
    result
}

impl ReduceSumValue for u32 {
    #[inline(always)]
    fn reduce_full_tile<const ITEMS: usize>(
        input: &[Self],
        tile_offset: usize,
        thread_index: usize,
        aggregate: &mut Self,
    ) {
        let vectors = unsafe {
            core::slice::from_raw_parts(
                input.as_ptr().cast::<cuda_device::vector::I32x4>(),
                input.len() / 4,
            )
        };
        let mut vector_offset = tile_offset / 4 + thread_index;
        let mut item = 0usize;
        while item < ITEMS / 4 {
            let lanes = cuda_device::read_only::load(&vectors[vector_offset]).to_array();
            *aggregate =
                *aggregate + lanes[0] as u32 + lanes[1] as u32 + lanes[2] as u32 + lanes[3] as u32;
            vector_offset += REDUCE_THREADS as usize;
            item += 1;
        }
    }

    #[inline(always)]
    fn reduce_partial_tile<const ITEMS: usize>(
        input: &[Self],
        tile_offset: usize,
        tile_end: usize,
        thread_index: usize,
        aggregate: &mut Self,
    ) {
        reduce_partial_tile_scalar::<Self, ITEMS>(
            input,
            tile_offset,
            tile_end,
            thread_index,
            aggregate,
        );
    }
    #[inline(always)]
    fn warp_reduce_sum_32(value: Self) -> Self {
        redux_sync_add_u32(value)
    }

    #[inline(always)]
    fn warp_reduce_sum_8(value: Self) -> Self {
        redux_sync_add_u32(value)
    }
}

impl ReduceSumValue for i32 {
    #[inline(always)]
    fn reduce_full_tile<const ITEMS: usize>(
        input: &[Self],
        tile_offset: usize,
        thread_index: usize,
        aggregate: &mut Self,
    ) {
        let vectors = unsafe {
            core::slice::from_raw_parts(
                input.as_ptr().cast::<cuda_device::vector::I32x4>(),
                input.len() / 4,
            )
        };
        let mut vector_offset = tile_offset / 4 + thread_index;
        let mut bits = *aggregate as u32;
        let mut item = 0usize;
        while item < ITEMS / 4 {
            let lanes = cuda_device::read_only::load(&vectors[vector_offset]).to_array();
            bits = bits + lanes[0] as u32 + lanes[1] as u32 + lanes[2] as u32 + lanes[3] as u32;
            vector_offset += REDUCE_THREADS as usize;
            item += 1;
        }
        *aggregate = bits as i32;
    }

    #[inline(always)]
    fn reduce_partial_tile<const ITEMS: usize>(
        input: &[Self],
        tile_offset: usize,
        tile_end: usize,
        thread_index: usize,
        aggregate: &mut Self,
    ) {
        reduce_partial_tile_scalar::<Self, ITEMS>(
            input,
            tile_offset,
            tile_end,
            thread_index,
            aggregate,
        );
    }
    #[inline(always)]
    fn warp_reduce_sum_32(value: Self) -> Self {
        redux_sync_add_u32(value as u32) as i32
    }

    #[inline(always)]
    fn warp_reduce_sum_8(value: Self) -> Self {
        redux_sync_add_u32(value as u32) as i32
    }
}

impl ReduceSumValue for f32 {
    #[inline(always)]
    fn reduce_full_tile<const ITEMS: usize>(
        input: &[Self],
        tile_offset: usize,
        thread_index: usize,
        aggregate: &mut Self,
    ) {
        let vectors = unsafe {
            core::slice::from_raw_parts(
                input.as_ptr().cast::<cuda_device::vector::I32x4>(),
                input.len() / 4,
            )
        };
        let mut vector_offset = tile_offset / 4 + thread_index;
        let mut item = 0usize;
        while item < ITEMS / 4 {
            let lanes = cuda_device::read_only::load(&vectors[vector_offset]).to_array();
            *aggregate = *aggregate
                + f32::from_bits(lanes[0] as u32)
                + f32::from_bits(lanes[1] as u32)
                + f32::from_bits(lanes[2] as u32)
                + f32::from_bits(lanes[3] as u32);
            vector_offset += REDUCE_THREADS as usize;
            item += 1;
        }
    }

    #[inline(always)]
    fn reduce_partial_tile<const ITEMS: usize>(
        input: &[Self],
        tile_offset: usize,
        tile_end: usize,
        thread_index: usize,
        aggregate: &mut Self,
    ) {
        reduce_partial_tile_scalar::<Self, ITEMS>(
            input,
            tile_offset,
            tile_end,
            thread_index,
            aggregate,
        );
    }
    #[inline(always)]
    fn warp_reduce_sum_32(value: Self) -> Self {
        crate::warp::warp_reduce::<Self, 32>(value, 32, |left, right| left + right)
    }

    #[inline(always)]
    fn warp_reduce_sum_8(value: Self) -> Self {
        crate::warp::warp_reduce::<Self, 32>(value, 8, |left, right| left + right)
    }
}

#[inline(always)]
fn reduce_partial_tile_scalar<T, const ITEMS: usize>(
    input: &[T],
    tile_offset: usize,
    tile_end: usize,
    thread_index: usize,
    aggregate: &mut T,
) where
    T: ReduceSumValue,
{
    let mut item = 0usize;
    while item < ITEMS {
        let index = tile_offset + item * REDUCE_THREADS as usize + thread_index;
        if index < tile_end {
            *aggregate = *aggregate + cuda_device::read_only::load(&input[index]);
        }
        item += 1;
    }
}
pub(crate) fn reduce_policy_items<T>() -> usize {
    let divisor = core::mem::size_of::<T>().max(4);
    let items = REDUCE_NOMINAL_4B_ITEMS * 4 / divisor;
    items.max(1)
}

pub(crate) fn reduce_tile_items<T>() -> usize {
    REDUCE_THREADS as usize * reduce_policy_items::<T>()
}

#[cuda_module]
#[doc(hidden)]
mod device {
    use cuda_device::vector::I32x4;
    use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, read_only, thread};

    #[inline(always)]
    fn reduce_sum_block<T, const ITEMS: usize>(
        input: &[T],
        num_items: usize,
        mut tile_offset: usize,
        tile_stride: usize,
        shared_memory: crate::__private::BlockSharedMemory,
    ) -> T
    where
        T: super::ReduceSumValue,
    {
        let thread_index = thread::threadIdx_x() as usize;
        let mut aggregate = T::default();
        let tile_items = super::REDUCE_THREADS as usize * ITEMS;
        while tile_offset + tile_items <= num_items {
            T::reduce_full_tile::<ITEMS>(input, tile_offset, thread_index, &mut aggregate);
            tile_offset += tile_stride;
        }
        if tile_offset < num_items {
            T::reduce_partial_tile::<ITEMS>(
                input,
                tile_offset,
                num_items,
                thread_index,
                &mut aggregate,
            );
        }

        let lane = thread::threadIdx_x() & 31;
        let warp_index = thread::threadIdx_x() >> 5;
        let warp_total = T::warp_reduce_sum_32(aggregate);
        let mut warps = shared_memory.region::<T>(0, 8);
        if lane == 0 {
            warps.write(warp_index as usize, warp_total);
        }
        thread::sync_threads();

        let warp_input = if warp_index == 0 && lane < 8 {
            warps.read(lane as usize)
        } else {
            T::default()
        };
        T::warp_reduce_sum_8(warp_input)
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1), dynamic_shared_range = (32, 64), dynamic_shared_alignment = 16)]
    pub fn device_reduce_sum_blocks<T, const ITEMS: usize>(
        input: &[T],
        num_items: usize,
        partials: DisjointSlice<T>,
    ) where
        T: super::ReduceSumValue,
    {
        let _ = launch_context;
        let shared_memory = crate::__private::BlockSharedMemory::new(8 * core::mem::size_of::<T>());
        let block_index = thread::blockIdx_x() as usize;
        let grid_size = thread::gridDim_x() as usize;
        let tile_items = super::REDUCE_THREADS as usize * ITEMS;
        let tile_offset = block_index * tile_items;
        let tile_stride = grid_size * tile_items;
        let block_total = reduce_sum_block::<T, ITEMS>(
            input,
            num_items.min(input.len()),
            tile_offset,
            tile_stride,
            shared_memory,
        );
        if thread::threadIdx_x() == 0 {
            let mut partials = crate::__private::DeviceRegion::new(partials);
            partials.write(block_index, block_total);
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1), dynamic_shared_range = (32, 64), dynamic_shared_alignment = 16)]
    pub fn device_reduce_sum_single_tile<T, const ITEMS: usize>(
        input: &[T],
        num_items: usize,
        output: DisjointSlice<T>,
    ) where
        T: super::ReduceSumValue,
    {
        let _ = launch_context;
        let shared_memory = crate::__private::BlockSharedMemory::new(8 * core::mem::size_of::<T>());
        let count = num_items.min(input.len());
        let block_total = reduce_sum_block::<T, ITEMS>(
            input,
            count,
            0,
            super::REDUCE_THREADS as usize * ITEMS,
            shared_memory,
        );
        if thread::threadIdx_x() == 0 {
            let mut output = crate::__private::DeviceRegion::new(output);
            output.write(0, block_total);
        }
    }

    #[inline(always)]
    fn reduce_sum_block_i32_as_u32<const ITEMS: usize>(
        input: &[i32],
        num_items: usize,
        mut tile_offset: usize,
        tile_stride: usize,
        shared_memory: crate::__private::BlockSharedMemory,
    ) -> i32 {
        let thread_index = thread::threadIdx_x() as usize;
        let mut aggregate = 0u32;
        let tile_items = super::REDUCE_THREADS as usize * ITEMS;
        let vectors =
            unsafe { core::slice::from_raw_parts(input.as_ptr().cast::<I32x4>(), input.len() / 4) };
        while tile_offset + tile_items <= num_items {
            let mut vector_offset = tile_offset / 4 + thread_index;
            let mut item = 0usize;
            while item < ITEMS / 4 {
                let lanes = read_only::load(&vectors[vector_offset]).to_array();
                aggregate = aggregate
                    + lanes[0] as u32
                    + lanes[1] as u32
                    + lanes[2] as u32
                    + lanes[3] as u32;
                vector_offset += super::REDUCE_THREADS as usize;
                item += 1;
            }
            tile_offset += tile_stride;
        }
        if tile_offset < num_items {
            let tile_end = num_items;
            let mut item = 0usize;
            while item < ITEMS {
                let index = tile_offset + item * super::REDUCE_THREADS as usize + thread_index;
                if index < tile_end {
                    aggregate += read_only::load(&input[index]) as u32;
                }
                item += 1;
            }
        }

        let lane = thread::threadIdx_x() & 31;
        let warp_index = thread::threadIdx_x() >> 5;
        let warp_total = super::redux_sync_add_u32(aggregate);
        let mut warps = shared_memory.region::<u32>(0, 8);
        if lane == 0 {
            warps.write(warp_index as usize, warp_total);
        }
        thread::sync_threads();

        let warp_input = if warp_index == 0 && lane < 8 {
            warps.read(lane as usize)
        } else {
            0u32
        };
        super::redux_sync_add_u32(warp_input) as i32
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1), dynamic_shared_range = (32, 64), dynamic_shared_alignment = 16)]
    pub fn device_reduce_sum_blocks_i32_as_u32<const ITEMS: usize>(
        input: &[i32],
        num_items: usize,
        partials: DisjointSlice<i32>,
    ) {
        let _ = launch_context;
        let shared_memory =
            crate::__private::BlockSharedMemory::new(8 * core::mem::size_of::<u32>());
        let block_index = thread::blockIdx_x() as usize;
        let grid_size = thread::gridDim_x() as usize;
        let tile_items = super::REDUCE_THREADS as usize * ITEMS;
        let tile_offset = block_index * tile_items;
        let tile_stride = grid_size * tile_items;
        let block_total = reduce_sum_block_i32_as_u32::<ITEMS>(
            input,
            num_items.min(input.len()),
            tile_offset,
            tile_stride,
            shared_memory,
        );
        if thread::threadIdx_x() == 0 {
            let mut partials = crate::__private::DeviceRegion::new(partials);
            partials.write(block_index, block_total);
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1), dynamic_shared_range = (32, 64), dynamic_shared_alignment = 16)]
    pub fn device_reduce_sum_single_tile_i32_as_u32<const ITEMS: usize>(
        input: &[i32],
        num_items: usize,
        output: DisjointSlice<i32>,
    ) {
        let _ = launch_context;
        let shared_memory =
            crate::__private::BlockSharedMemory::new(8 * core::mem::size_of::<u32>());
        let count = num_items.min(input.len());
        let block_total = reduce_sum_block_i32_as_u32::<ITEMS>(
            input,
            count,
            0,
            super::REDUCE_THREADS as usize * ITEMS,
            shared_memory,
        );
        if thread::threadIdx_x() == 0 {
            let mut output = crate::__private::DeviceRegion::new(output);
            output.write(0, block_total);
        }
    }
}

#[doc(hidden)]
pub type ReduceModule = device::LoadedModule;

#[doc(hidden)]
pub fn load(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> Result<ReduceModule, cuda_host::EmbeddedModuleError> {
    unsafe { device::load(context) }
}
