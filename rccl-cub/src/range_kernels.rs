//! Generated kernels for zero-materialization device-range algorithms.

use cuda_device::cuda_module;

pub(crate) const GENERATION_THREADS: u32 = 256;
pub(crate) const GENERATION_ITEMS_PER_THREAD: u32 = 8;
pub(crate) const GENERATION_TILE_ITEMS: u32 =
    GENERATION_THREADS * GENERATION_ITEMS_PER_THREAD;

pub(crate) const fn generation_blocks(len: u32) -> u32 {
    len.div_ceil(GENERATION_TILE_ITEMS)
}

#[cuda_module]
#[doc(hidden)]
mod device {
    use crate::range::DenseDeviceWriteAt;
    use cuda_device::{kernel, launch_bounds, launch_contract, thread};
    use super::{GENERATION_ITEMS_PER_THREAD, GENERATION_THREADS, GENERATION_TILE_ITEMS};

    #[derive(Clone, Copy)]
    #[repr(C, align(8))]
    struct AlignedPair<T> {
        first: T,
        second: T,
    }

    #[repr(C, align(16))]
    struct AlignedQuad<T> {
        first: T,
        second: T,
        third: T,
        fourth: T,
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn range_transform<Input, Output, F>(input: Input, output: Output, op: F, len: usize)
    where
        Input: crate::DeviceReadAt,
        Output: crate::DeviceWriteAt,
        F: Fn(Input::Item) -> Output::Item + Copy,
    {
        let _ = launch_context;
        let mut index = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;
        while index < len {
            // SAFETY: the loop bounds the logical index; the host lowering
            // retains all allocations and affine output authority.
            let value = unsafe { input.read_at(index) };
            unsafe { output.write_at(index, op(value)) };
            index += stride;
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn range_transform_u32<Input, Output, F>(input: Input, output: Output, op: F, len: u32)
    where
        Input: crate::DeviceReadAt,
        Output: crate::DeviceWriteAt,
        F: Fn(Input::Item) -> Output::Item + Copy,
    {
        let _ = launch_context;
        let index = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let stride = thread::gridDim_x() * thread::blockDim_x();
        if index < len {
            // SAFETY: the predicate bounds the logical index; the host lowering
            // retains all allocations and affine output authority.
            let value = unsafe { input.read_at(index as usize) };
            unsafe { output.write_at(index as usize, op(value)) };
        }
        let second = index + stride;
        if second < len {
            // SAFETY: as above. The <= u32 extent launch policy makes the
            // two-item index addition non-overflowing.
            let value = unsafe { input.read_at(second as usize) };
            unsafe { output.write_at(second as usize, op(value)) };
        }
    }

    /// Dense four-byte fill matching pinned DeviceTransform's SM89
    /// 256-thread, eight-item, two-element vector-store policy.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn range_fill_4b<Output>(output: Output, value: Output::Item, len: u32)
    where
        Output: DenseDeviceWriteAt,
    {
        let _ = launch_context;
        let tile_offset = thread::blockIdx_x() * GENERATION_TILE_ITEMS;
        let lane = thread::threadIdx_x();
        let remaining = len - tile_offset;
        if remaining >= GENERATION_TILE_ITEMS {
            let pair_base = unsafe {
                output
                    .as_mut_ptr()
                    .add(tile_offset as usize)
                    .cast::<AlignedPair<Output::Item>>()
            };
            let pair = AlignedPair {
                first: value,
                second: value,
            };
            let mut item = 0;
            while item < GENERATION_ITEMS_PER_THREAD / 2 {
                // SAFETY: the host selects this entry only for four-byte items
                // and an eight-byte-aligned complete allocation. A full tile
                // contains four disjoint pair stores per lane.
                unsafe {
                    pair_base
                        .add((lane + item * GENERATION_THREADS) as usize)
                        .write(pair)
                };
                item += 1;
            }
        } else {
            let mut item = 0;
            while item < GENERATION_ITEMS_PER_THREAD {
                let offset = item * GENERATION_THREADS + lane;
                if offset < remaining {
                    let index = tile_offset + offset;
                    // SAFETY: the tail predicate bounds the logical index.
                    unsafe { output.write_at(index as usize, value) };
                }
                item += 1;
            }
        }
    }

    /// Dense four-byte generation matching pinned DeviceTransform's SM89
    /// 256-thread, eight-item, four-element vector-store policy.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn range_generate_4b<Input, Output, F>(
        input: Input,
        output: Output,
        op: F,
        len: u32,
    ) where
        Input: crate::DeviceReadAt,
        Output: DenseDeviceWriteAt,
        F: Fn(Input::Item) -> Output::Item + Copy,
    {
        let _ = launch_context;
        let tile_offset = thread::blockIdx_x() * GENERATION_TILE_ITEMS;
        let lane = thread::threadIdx_x();
        let remaining = len - tile_offset;
        if remaining >= GENERATION_TILE_ITEMS {
            let quad_base = unsafe {
                output
                    .as_mut_ptr()
                    .add(tile_offset as usize)
                    .cast::<AlignedQuad<Output::Item>>()
            };
            let mut item = 0;
            while item < GENERATION_ITEMS_PER_THREAD / 4 {
                let first = tile_offset + lane * 4 + item * GENERATION_THREADS * 4;
                // SAFETY: the full-tile branch bounds all four logical
                // indices, and the input expression is retained by completion.
                let quad = unsafe {
                    AlignedQuad {
                        first: op(input.read_at(first as usize)),
                        second: op(input.read_at((first + 1) as usize)),
                        third: op(input.read_at((first + 2) as usize)),
                        fourth: op(input.read_at((first + 3) as usize)),
                    }
                };
                // SAFETY: the host proves a sixteen-byte-aligned complete
                // allocation. Two disjoint four-item stores cover each lane.
                unsafe {
                    quad_base
                        .add((lane + item * GENERATION_THREADS) as usize)
                        .write(quad)
                };
                item += 1;
            }
        } else {
            let mut item = 0;
            while item < GENERATION_ITEMS_PER_THREAD {
                let offset = item * GENERATION_THREADS + lane;
                if offset < remaining {
                    let index = tile_offset + offset;
                    // SAFETY: the tail predicate bounds the logical index.
                    let value = unsafe { input.read_at(index as usize) };
                    unsafe { output.write_at(index as usize, op(value)) };
                }
                item += 1;
            }
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn range_for_each<Input, F>(input: Input, op: F, len: usize)
    where
        Input: crate::DeviceReadAt,
        F: Fn(Input::Item) + Copy,
    {
        let _ = launch_context;
        let index = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;
        if index < len {
            // SAFETY: the predicate bounds the logical index and the completion
            // guard retains the input and callable captures.
            op(unsafe { input.read_at(index) });
        }
        let second = index + stride;
        if second < len {
            // SAFETY: as above. The host selects this entry only when its
            // two-item launch covers the full extent without index overflow.
            op(unsafe { input.read_at(second) });
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn range_for_each_grid<Input, F>(input: Input, op: F, len: usize)
    where
        Input: crate::DeviceReadAt,
        F: Fn(Input::Item) + Copy,
    {
        let _ = launch_context;
        let mut index = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;
        while index < len {
            // SAFETY: the loop bounds the logical index and the completion
            // guard retains the input and callable captures.
            op(unsafe { input.read_at(index) });
            index += stride;
        }
    }
}

#[doc(hidden)]
pub type RangeModule = device::LoadedModule;

#[doc(hidden)]
pub fn load(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> Result<RangeModule, cuda_host::EmbeddedModuleError> {
    // SAFETY: the loader is generated from the entries in this module.
    unsafe { device::load(context) }
}
