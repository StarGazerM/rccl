#![feature(generic_const_exprs)]
#![expect(
    incomplete_features,
    reason = "WarpMergeSort benchmark kernels currently require generic_const_exprs"
)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use std::error::Error;

const ITEMS_PER_THREAD: usize = 4;
const LOGICAL_WARP_THREADS: usize = 32;
const THREADS_PER_BLOCK: usize = 256;
const WARPS_PER_BLOCK: usize = THREADS_PER_BLOCK / LOGICAL_WARP_THREADS;
const ITEMS_PER_WARP: usize = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS;
const ITEMS_PER_BLOCK: usize = THREADS_PER_BLOCK * ITEMS_PER_THREAD;

#[expect(
    dead_code,
    reason = "cuda-oxide consumes these kernels during device compilation"
)]
#[cuda_host::cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{
        DisjointSlice,
        atomic::{AtomicOrdering, DeviceAtomicU64},
        kernel, thread,
    };

    #[kernel]
    pub unsafe fn fill_descending_tiles(mut data: DisjointSlice<u32>, count: u64) {
        let index = thread::index_1d().get() as u64;
        if index < count {
            let within_warp = index as usize % ITEMS_PER_WARP;
            *unsafe { data.get_unchecked_mut(index as usize) } =
                (ITEMS_PER_WARP - 1 - within_warp) as u32;
        }
    }

    #[kernel]
    pub unsafe fn sort_tiles(mut data: DisjointSlice<u32>) {
        let thread_id = thread::threadIdx_x() as usize;
        let warp_id = thread_id / LOGICAL_WARP_THREADS;
        let linear_tid = thread_id % LOGICAL_WARP_THREADS;
        let block_offset = thread::blockIdx_x() as usize * ITEMS_PER_BLOCK;
        let thread_offset = block_offset + warp_id * ITEMS_PER_WARP + linear_tid * ITEMS_PER_THREAD;

        let storage = unsafe {
            rccl::cub::warp_merge_sort_temp_storage!(
                TEMP_STORAGE,
                key = u32,
                items_per_thread = ITEMS_PER_THREAD,
                logical_warp_threads = LOGICAL_WARP_THREADS,
                warps = WARPS_PER_BLOCK,
                warp_index = warp_id,
            )
        };
        let warp_sort = rccl::cub::WarpMergeSort::<u32, ITEMS_PER_THREAD>::new(storage);
        let mut keys = [*unsafe { data.get_unchecked_mut(thread_offset) }; ITEMS_PER_THREAD];
        let mut item = 1usize;
        while item < ITEMS_PER_THREAD {
            keys[item] = *unsafe { data.get_unchecked_mut(thread_offset + item) };
            item += 1;
        }
        unsafe { warp_sort.sort(&mut keys, u32::cmp) };
        let mut item = 0usize;
        while item < ITEMS_PER_THREAD {
            *unsafe { data.get_unchecked_mut(thread_offset + item) } = keys[item];
            item += 1;
        }
    }

    #[kernel]
    pub unsafe fn validate_tiles(data: &[u32], count: u64, mut errors: DisjointSlice<u64>) {
        let index = thread::index_1d().get() as u64;
        if index < count {
            let expected = index as usize % ITEMS_PER_WARP;
            if data[index as usize] != expected as u32 {
                unsafe { DeviceAtomicU64::from_ptr(errors.as_mut_ptr()) }
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }
}

fn argument(name: &str, default: usize) -> usize {
    let arguments = std::env::args().collect::<Vec<_>>();
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn Error>> {
    let requested = argument("--elements", 1 << 24);
    let iterations = argument("--iterations", 50);
    let count = requested / ITEMS_PER_BLOCK * ITEMS_PER_BLOCK;
    if count == 0 || iterations == 0 {
        return Err("elements must cover one block and iterations must be nonzero".into());
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let mut data = DeviceBuffer::<u32>::zeroed(&stream, count)?;
    let blocks = u32::try_from(count / ITEMS_PER_BLOCK)?;
    let sort_launch = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (THREADS_PER_BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let fill_launch = LaunchConfig::for_num_elems(u32::try_from(count)?);

    for _ in 0..5 {
        unsafe {
            module.fill_descending_tiles(&stream, fill_launch, &mut data, count as u64)?;
            module.sort_tiles(&stream, sort_launch, &mut data)?;
        }
    }
    stream.synchronize()?;

    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        unsafe {
            module.fill_descending_tiles(&stream, fill_launch, &mut data, count as u64)?;
        }
        let start = stream.record_event(Some(0))?;
        unsafe {
            module.sort_tiles(&stream, sort_launch, &mut data)?;
        }
        let end = stream.record_event(Some(0))?;
        end.synchronize()?;
        timings.push(start.elapsed_ms(&end)? as f64);
    }

    let mut errors = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    unsafe {
        module.validate_tiles(&stream, fill_launch, &data, count as u64, &mut errors)?;
    }
    let error_count = errors.to_host_vec(&stream)?[0];
    if error_count != 0 {
        return Err(format!("warp benchmark validation found {error_count} errors").into());
    }

    timings.sort_by(f64::total_cmp);
    let best = timings[0];
    let median = timings[timings.len() / 2];
    println!(
        "RCCL CUB WarpMergeSort u32 32x4: items={count}, best={best:.3} ms ({:.3} Gitem/s), median={median:.3} ms",
        count as f64 / best / 1.0e6,
    );
    Ok(())
}
