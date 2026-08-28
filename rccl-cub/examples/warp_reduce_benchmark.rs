//! Equivalent-work performance comparison for pinned CUB `WarpReduce`.
//!
//! The grid, block, input, output, warm-up, sample, and repeated-launch shape
//! are kept identical to `benchmarks/warp-reduce-reference.cu`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use std::error::Error;

const BLOCK_THREADS: usize = 256;
const WARP_THREADS: usize = 32;
const BLOCKS: usize = 32_768;
const REDUCTIONS_PER_VALUE: usize = 4;
const WARMUPS: usize = 20;
const SAMPLES: usize = 21;
const LAUNCHES: usize = 100;
const ITEMS: usize = BLOCK_THREADS * BLOCKS;
const WARPS: usize = ITEMS / WARP_THREADS;

#[rccl_cub::program]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn reduce_warps(mut values: DisjointSlice<i32>) {
        let index = thread::index_1d();
        if let Some(slot) = values.get_mut(index) {
            let value = *slot;
            let first = rccl_cub::warp_reduce!(::<i32, 32>(value, |left, right| left + right));
            let second =
                rccl_cub::warp_reduce!(::<i32, 32>(value ^ 0x55, |left, right| left + right));
            let third = rccl_cub::warp_reduce!(::<i32, 32>(value * 3, |left, right| left + right));
            let fourth = rccl_cub::warp_reduce!(::<i32, 32>(value - 7, |left, right| left + right));
            if warp::lane_id() == 0 {
                *slot = first + second + third + fourth;
            }
        }
    }
}

fn input_value(index: usize) -> i32 {
    ((index * 17 + 3) % 251) as i32 - 125
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = kernels::program!(&context)?;
    let launch = LaunchConfig1D::new(BLOCKS as u32, BLOCK_THREADS as u32, 0);
    let prepared = program.prepare_reduce_warps(launch)?;

    let host_input = (0..ITEMS).map(input_value).collect::<Vec<_>>();
    let mut values = DeviceBuffer::from_host(&stream, &host_input)?;

    for _ in 0..WARMUPS {
        program.reduce_warps(&stream, &prepared, &mut values)?;
    }
    stream.synchronize()?;

    let timing = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let mut timings = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = stream.record_event(timing)?;
        for _ in 0..LAUNCHES {
            program.reduce_warps(&stream, &prepared, &mut values)?;
        }
        let end = stream.record_event(timing)?;
        end.synchronize()?;
        timings.push(start.elapsed_ms(&end)? / LAUNCHES as f32);
    }

    values.copy_from_host(&stream, &host_input)?;
    program.reduce_warps(&stream, &prepared, &mut values)?;
    let host_output = values.to_host_vec(&stream)?;
    for warp in 0..WARPS {
        let values = (0..WARP_THREADS)
            .map(|lane| input_value(warp * WARP_THREADS + lane))
            .collect::<Vec<_>>();
        let expected = values.iter().copied().sum::<i32>()
            + values.iter().map(|value| value ^ 0x55).sum::<i32>()
            + values.iter().map(|value| value * 3).sum::<i32>()
            + values.iter().map(|value| value - 7).sum::<i32>();
        let result = host_output[warp * WARP_THREADS];
        assert_eq!(result, expected, "WarpReduce mismatch at warp {warp}");
    }

    timings.sort_by(f32::total_cmp);
    println!(
        "implementation=rccl primitive=warp_reduce type=i32 width=32 reductions_per_value={REDUCTIONS_PER_VALUE} items={ITEMS} warps={WARPS} best_ms={:.6} median_ms={:.6}",
        timings[0],
        timings[timings.len() / 2],
    );
    Ok(())
}
