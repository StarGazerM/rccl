#![feature(generic_const_exprs)]
#![expect(
    incomplete_features,
    reason = "WarpMergeSort shared storage currently requires generic_const_exprs"
)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use std::error::Error;

const LOGICAL_WARP_THREADS: usize = 16;
const ITEMS_PER_THREAD: usize = 4;
const THREADS_PER_BLOCK: usize = 256;
const WARPS_PER_BLOCK: usize = THREADS_PER_BLOCK / LOGICAL_WARP_THREADS;
const ITEMS_PER_WARP: usize = LOGICAL_WARP_THREADS * ITEMS_PER_THREAD;

#[expect(
    dead_code,
    reason = "cuda-oxide consumes these kernels during device compilation"
)]
#[cuda_host::cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{
        DisjointSlice,
        atomic::{AtomicOrdering, DeviceAtomicU32},
        kernel, thread,
    };
    use rccl::cub::WarpMergeSort;

    #[kernel]
    pub unsafe fn fill(mut data: DisjointSlice<i32>, count: u32) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            let within_warp = index as usize % ITEMS_PER_WARP;
            *unsafe { data.get_unchecked_mut(index as usize) } =
                (ITEMS_PER_WARP - 1 - within_warp) as i32;
        }
    }

    #[kernel]
    pub unsafe fn example(mut data: DisjointSlice<i32>) {
        let thread_id = thread::threadIdx_x() as usize;
        let warp_id = thread_id / LOGICAL_WARP_THREADS;
        let linear_tid = thread_id % LOGICAL_WARP_THREADS;

        // Rust equivalent of:
        // __shared__ WarpMergeSortT::TempStorage storage[warps_per_block];
        let storage = unsafe {
            rccl::cub::warp_merge_sort_temp_storage!(
                TEMP_STORAGE,
                key = i32,
                items_per_thread = ITEMS_PER_THREAD,
                logical_warp_threads = LOGICAL_WARP_THREADS,
                warps = WARPS_PER_BLOCK,
                warp_index = warp_id,
            )
        };
        let warp_sort = WarpMergeSort::<i32, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS>::new(storage);

        let thread_offset = warp_id * ITEMS_PER_WARP + linear_tid * ITEMS_PER_THREAD;
        let mut keys = [*unsafe { data.get_unchecked_mut(thread_offset) }; ITEMS_PER_THREAD];
        let mut item = 0usize;
        while item < ITEMS_PER_THREAD {
            keys[item] = *unsafe { data.get_unchecked_mut(thread_offset + item) };
            item += 1;
        }

        unsafe { warp_sort.sort(&mut keys, i32::cmp) };

        let mut item = 0usize;
        while item < ITEMS_PER_THREAD {
            *unsafe { data.get_unchecked_mut(thread_offset + item) } = keys[item];
            item += 1;
        }
    }

    #[kernel]
    pub unsafe fn validate(data: &[i32], count: u32, mut errors: DisjointSlice<u32>) {
        let index = thread::index_1d().get() as u32;
        if index < count && !(index as usize).is_multiple_of(ITEMS_PER_WARP) {
            let previous = data[index as usize - 1];
            let current = data[index as usize];
            if previous > current {
                unsafe { DeviceAtomicU32::from_ptr(errors.as_mut_ptr()) }
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let count = THREADS_PER_BLOCK * ITEMS_PER_THREAD;
    let mut data = DeviceBuffer::<i32>::zeroed(&stream, count)?;
    let mut errors = DeviceBuffer::<u32>::zeroed(&stream, 1)?;

    unsafe {
        module.fill(
            &stream,
            LaunchConfig::for_num_elems(count as u32),
            &mut data,
            count as u32,
        )?;
        module.example(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (THREADS_PER_BLOCK as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &mut data,
        )?;
        module.validate(
            &stream,
            LaunchConfig::for_num_elems(count as u32),
            &data,
            count as u32,
            &mut errors,
        )?;
    }
    assert_eq!(errors.to_host_vec(&stream)?[0], 0);
    println!("CUB WarpMergeSort 16x4 header example: PASS");
    Ok(())
}
