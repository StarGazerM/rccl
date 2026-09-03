//! Rust port of the pinned CUB `WarpReduce` overview and focused Catch2 cases.
//!
//! Upstream sources at CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`:
//! - `cub/cub/warp/warp_reduce.cuh`, overview at lines 68-121;
//! - `cub/test/warp/catch2_test_warp_reduce.cu`.
//!
//! Run with `cargo oxide run warp_reduce`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};

const THREADS: usize = 128;
const ITEMS_PER_THREAD: usize = 4;

#[cuda_device::cuda_module]
mod kernels {
    use super::*;

    /// Pinned `warp_reduce.cuh` overview: four independent 32-lane sums in a
    /// 128-thread block. As in CUB, only logical lane zero is observed.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= 128, output.len() >= 128),
    )]
    pub fn overview(input: &[i32], mut output: DisjointSlice<i32>) {
        let index = thread::index_1d();
        let aggregate =
            rccl_cub::warp_reduce::<i32, 32>(input[index.get()], 32, |left, right| left + right);
        if let Some(slot) = output.get_mut(index) {
            *slot = aggregate;
        }
    }

    /// Pinned Catch2 partial-warp shape. Every lane receives the same runtime
    /// valid count, while the logical width remains compile-time macro input.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= 128, output.len() >= 128),
    )]
    pub fn partial_sum(input: &[i32], valid_items: u32, mut output: DisjointSlice<i32>) {
        let index = thread::index_1d();
        let aggregate =
            rccl_cub::warp_reduce::<i32, 32>(input[index.get()], valid_items, |left, right| {
                left + right
            });
        if let Some(slot) = output.get_mut(index) {
            *slot = aggregate;
        }
    }

    /// Pinned Catch2 logical-warp and custom-operation shape.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= 128, output.len() >= 128),
    )]
    pub fn logical_16_max(input: &[i32], mut output: DisjointSlice<i32>) {
        let index = thread::index_1d();
        let aggregate = rccl_cub::warp_reduce::<i32, 16>(input[index.get()], 16, |left, right| {
            if left > right { left } else { right }
        });
        if let Some(slot) = output.get_mut(index) {
            *slot = aggregate;
        }
    }

    /// Pinned Catch2 four-items-per-thread shape. A Rust array replaces the
    /// fixed-size C++ random-access range overload.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= 512, output.len() >= 128),
    )]
    pub fn multiple_items(input: &[i32], mut output: DisjointSlice<i32>) {
        let index = thread::index_1d();
        let first = index.get() * ITEMS_PER_THREAD;
        let items = [
            input[first],
            input[first + 1],
            input[first + 2],
            input[first + 3],
        ];
        let aggregate =
            rccl_cub::warp_reduce_items::<i32, 32, 4>(items, |left, right| left + right);
        if let Some(slot) = output.get_mut(index) {
            *slot = aggregate;
        }
    }

    /// The pinned scalar matrix includes 64-bit integers. This locks the
    /// two-register lane exchange used by the Rust backend.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= 128, output.len() >= 128),
    )]
    pub fn i64_sum(input: &[i64], mut output: DisjointSlice<i64>) {
        let index = thread::index_1d();
        let aggregate =
            rccl_cub::warp_reduce::<i64, 32>(input[index.get()], 32, |left, right| left + right);
        if let Some(slot) = output.get_mut(index) {
            *slot = aggregate;
        }
    }

    /// The pinned built-in matrix includes narrow integers. Wrapping addition
    /// makes the fixed-width integer behavior explicit in Rust.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= 128, output.len() >= 128),
    )]
    pub fn u8_sum(input: &[u8], mut output: DisjointSlice<u8>) {
        let index = thread::index_1d();
        let aggregate = rccl_cub::warp_reduce::<u8, 32>(input[index.get()], 32, |left, right| {
            left.wrapping_add(right)
        });
        if let Some(slot) = output.get_mut(index) {
            *slot = aggregate;
        }
    }
}

fn reference_groups<T: Copy>(
    input: &[T],
    logical_width: usize,
    valid_items: usize,
    op: impl Fn(T, T) -> T + Copy,
) -> Vec<T> {
    input
        .chunks_exact(logical_width)
        .map(|group| group[1..valid_items].iter().copied().fold(group[0], op))
        .collect()
}

fn lane_zero_values<T: Copy>(output: &[T], logical_width: usize) -> Vec<T> {
    output.iter().step_by(logical_width).copied().collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = unsafe { kernels::load(&context)? };
    let launch = LaunchConfig1D::new(1, THREADS as u32, 0);

    let overview_input = (0..THREADS as i32).collect::<Vec<_>>();
    let overview_device = DeviceBuffer::from_host(&stream, &overview_input)?;
    let mut overview_output = DeviceBuffer::<i32>::zeroed(&stream, THREADS)?;
    let prepared = program.prepare_overview(launch)?;
    program.overview(&stream, &prepared, &overview_device, &mut overview_output)?;
    let overview_result = overview_output.to_host_vec(&stream)?;
    let overview_lane_zero = lane_zero_values(&overview_result, 32);
    assert_eq!(overview_lane_zero, [496, 1520, 2544, 3568]);
    assert_eq!(
        overview_lane_zero,
        reference_groups(&overview_input, 32, 32, |left, right| left + right)
    );

    let valid_items = 13usize;
    let mut partial_output = DeviceBuffer::<i32>::zeroed(&stream, THREADS)?;
    let prepared = program.prepare_partial_sum(launch)?;
    program.partial_sum(
        &stream,
        &prepared,
        &overview_device,
        valid_items as u32,
        &mut partial_output,
    )?;
    assert_eq!(
        lane_zero_values(&partial_output.to_host_vec(&stream)?, 32),
        reference_groups(&overview_input, 32, valid_items, |left, right| left + right)
    );

    let mut max_output = DeviceBuffer::<i32>::zeroed(&stream, THREADS)?;
    let prepared = program.prepare_logical_16_max(launch)?;
    program.logical_16_max(&stream, &prepared, &overview_device, &mut max_output)?;
    assert_eq!(
        lane_zero_values(&max_output.to_host_vec(&stream)?, 16),
        reference_groups(&overview_input, 16, 16, i32::max)
    );

    let items_input = (0..(THREADS * ITEMS_PER_THREAD) as i32).collect::<Vec<_>>();
    let items_device = DeviceBuffer::from_host(&stream, &items_input)?;
    let mut items_output = DeviceBuffer::<i32>::zeroed(&stream, THREADS)?;
    let prepared = program.prepare_multiple_items(launch)?;
    program.multiple_items(&stream, &prepared, &items_device, &mut items_output)?;
    assert_eq!(
        lane_zero_values(&items_output.to_host_vec(&stream)?, 32),
        reference_groups(
            &items_input,
            32 * ITEMS_PER_THREAD,
            32 * ITEMS_PER_THREAD,
            |left, right| left + right
        )
    );

    let i64_input = (0..THREADS)
        .map(|index| (index as i64 - 64) * 1_000_003)
        .collect::<Vec<_>>();
    let i64_device = DeviceBuffer::from_host(&stream, &i64_input)?;
    let mut i64_output = DeviceBuffer::<i64>::zeroed(&stream, THREADS)?;
    let prepared = program.prepare_i64_sum(launch)?;
    program.i64_sum(&stream, &prepared, &i64_device, &mut i64_output)?;
    assert_eq!(
        lane_zero_values(&i64_output.to_host_vec(&stream)?, 32),
        reference_groups(&i64_input, 32, 32, |left, right| left + right)
    );

    let u8_input = (0..THREADS)
        .map(|index| (index * 17 + 251) as u8)
        .collect::<Vec<_>>();
    let u8_device = DeviceBuffer::from_host(&stream, &u8_input)?;
    let mut u8_output = DeviceBuffer::<u8>::zeroed(&stream, THREADS)?;
    let prepared = program.prepare_u8_sum(launch)?;
    program.u8_sum(&stream, &prepared, &u8_device, &mut u8_output)?;
    assert_eq!(
        lane_zero_values(&u8_output.to_host_vec(&stream)?, 32),
        reference_groups(&u8_input, 32, 32, u8::wrapping_add)
    );

    println!("pinned CCCL WarpReduce overview and focused Catch2 matrix: PASS");
    Ok(())
}
