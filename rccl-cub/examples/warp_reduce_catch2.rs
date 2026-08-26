//! Rust execution matrix corresponding to pinned CUB WarpReduce Catch2 cases.
//!
//! Upstream source at CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`:
//! `cub/test/warp/catch2_test_warp_reduce.cu`, lines 242-401.
//!
//! This covers every upstream builtin row expressible by cuda-oxide's safe
//! power-of-two warp tiles: `u8`, `u16`, `i32`, and `i64`; sum, max, and min;
//! logical widths 32, 16, and 1; full, partial, and four-items-per-thread
//! reductions. It also covers the upstream `f32` max/min rows for those widths.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig1D};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread};
use std::{error::Error, fmt::Debug};

const THREADS: usize = 128;
const ITEMS_PER_THREAD: usize = 4;

// Runtime selectors choose among already-compiled Catch2 rows. Literal widths
// and ordinary Rust closure expressions remain inputs to RCCL's proc macros.
macro_rules! full_integer_operation {
    ($operation:expr, $width:literal, $value:expr) => {
        match $operation {
            0 => rccl_cub::warp_reduce!($width, $value, |left, right| left.wrapping_add(right)),
            1 => rccl_cub::warp_reduce!($width, $value, |left, right| if left > right {
                left
            } else {
                right
            }),
            2 => rccl_cub::warp_reduce!($width, $value, |left, right| if left < right {
                left
            } else {
                right
            }),
            _ => return,
        }
    };
}

macro_rules! partial_integer_operation {
    ($operation:expr, $width:literal, $value:expr, $valid:expr) => {
        match $operation {
            0 => rccl_cub::warp_reduce!($width, $value, $valid, |left, right| left
                .wrapping_add(right)),
            1 => rccl_cub::warp_reduce!($width, $value, $valid, |left, right| if left > right {
                left
            } else {
                right
            }),
            2 => rccl_cub::warp_reduce!($width, $value, $valid, |left, right| if left < right {
                left
            } else {
                right
            }),
            _ => return,
        }
    };
}

macro_rules! items_integer_operation {
    ($operation:expr, $width:literal, $items:expr) => {
        match $operation {
            0 => {
                rccl_cub::warp_reduce_items!($width, $items, |left, right| left.wrapping_add(right))
            }
            1 => rccl_cub::warp_reduce_items!($width, $items, |left, right| if left > right {
                left
            } else {
                right
            }),
            2 => rccl_cub::warp_reduce_items!($width, $items, |left, right| if left < right {
                left
            } else {
                right
            }),
            _ => return,
        }
    };
}

macro_rules! integer_reduce_case {
    ($case:expr, $value:expr, $valid:expr, $items:expr) => {{
        let operation = $case % 3;
        match $case / 3 {
            0 => full_integer_operation!(operation, 32, $value),
            1 => full_integer_operation!(operation, 16, $value),
            2 => full_integer_operation!(operation, 1, $value),
            3 => partial_integer_operation!(operation, 32, $value, $valid),
            4 => partial_integer_operation!(operation, 16, $value, $valid),
            5 => partial_integer_operation!(operation, 1, $value, $valid),
            6 => items_integer_operation!(operation, 32, $items),
            7 => items_integer_operation!(operation, 16, $items),
            8 => items_integer_operation!(operation, 1, $items),
            _ => return,
        }
    }};
}

macro_rules! f32_min_max_operation {
    ($operation:expr, $width:literal, $value:expr) => {
        match $operation {
            0 => rccl_cub::warp_reduce!($width, $value, |left, right| if left > right {
                left
            } else {
                right
            }),
            1 => rccl_cub::warp_reduce!($width, $value, |left, right| if left < right {
                left
            } else {
                right
            }),
            _ => return,
        }
    };
}

macro_rules! integer_matrix_body {
    ($input:ident, $case:ident, $valid_items:ident, $output:ident, $index:ident) => {
        let first = $index.get() * ITEMS_PER_THREAD;
        let value = $input[$index.get()];
        let items = [
            $input[first],
            $input[first + 1],
            $input[first + 2],
            $input[first + 3],
        ];
        let aggregate = integer_reduce_case!($case, value, $valid_items, items);
        if let Some(slot) = $output.get_mut($index) {
            *slot = aggregate;
        }
    };
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= 512, output.len() >= 128),
    )]
    pub fn u8_matrix(input: &[u8], case: u32, valid_items: u32, mut output: DisjointSlice<u8>) {
        let index = thread::index_1d();
        integer_matrix_body!(input, case, valid_items, output, index);
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= 512, output.len() >= 128),
    )]
    pub fn u16_matrix(input: &[u16], case: u32, valid_items: u32, mut output: DisjointSlice<u16>) {
        let index = thread::index_1d();
        integer_matrix_body!(input, case, valid_items, output, index);
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= 512, output.len() >= 128),
    )]
    pub fn i32_matrix(input: &[i32], case: u32, valid_items: u32, mut output: DisjointSlice<i32>) {
        let index = thread::index_1d();
        integer_matrix_body!(input, case, valid_items, output, index);
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= 512, output.len() >= 128),
    )]
    pub fn i64_matrix(input: &[i64], case: u32, valid_items: u32, mut output: DisjointSlice<i64>) {
        let index = thread::index_1d();
        integer_matrix_body!(input, case, valid_items, output, index);
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= 128, output.len() >= 128),
    )]
    pub fn f32_min_max(input: &[f32], case: u32, mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let value = input[index.get()];
        let operation = case % 2;
        let aggregate = match case / 2 {
            0 => f32_min_max_operation!(operation, 32, value),
            1 => f32_min_max_operation!(operation, 16, value),
            2 => f32_min_max_operation!(operation, 1, value),
            _ => return,
        };
        if let Some(slot) = output.get_mut(index) {
            *slot = aggregate;
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Full,
    Partial,
    MultipleItems,
}

fn reference_groups<T: Copy>(
    input: &[T],
    logical_width: usize,
    valid_items: usize,
    op: fn(T, T) -> T,
) -> Vec<T> {
    input
        .chunks_exact(logical_width)
        .map(|group| group[1..valid_items].iter().copied().fold(group[0], op))
        .collect()
}

fn lane_zero_values<T: Copy>(output: &[T], logical_width: usize) -> Vec<T> {
    output.iter().step_by(logical_width).copied().collect()
}

fn run_integer_matrix<T>(
    stream: &CudaStream,
    input: Vec<T>,
    mut launch_case: impl FnMut(
        u32,
        u32,
        &DeviceBuffer<T>,
        &mut DeviceBuffer<T>,
    ) -> Result<(), Box<dyn Error>>,
    operations: [fn(T, T) -> T; 3],
) -> Result<usize, Box<dyn Error>>
where
    T: Copy + Debug + DeviceCopy + PartialEq,
{
    let device_input = DeviceBuffer::from_host(stream, &input)?;
    let mut device_output = DeviceBuffer::<T>::zeroed(stream, THREADS)?;
    let widths = [32usize, 16, 1];
    let partial_counts = [[1usize, 13], [1, 7], [1, 1]];
    let shapes = [Shape::Full, Shape::Partial, Shape::MultipleItems];
    let mut executed = 0;

    for (shape_index, shape) in shapes.into_iter().enumerate() {
        for (width_index, width) in widths.into_iter().enumerate() {
            for (operation_index, operation) in operations.into_iter().enumerate() {
                let case = (shape_index * 9 + width_index * 3 + operation_index) as u32;
                let valid_counts = match shape {
                    Shape::Partial => partial_counts[width_index].as_slice(),
                    Shape::Full | Shape::MultipleItems => &partial_counts[width_index][..1],
                };
                for &valid_items in valid_counts {
                    launch_case(case, valid_items as u32, &device_input, &mut device_output)?;
                    let observed = lane_zero_values(&device_output.to_host_vec(stream)?, width);
                    let expected = match shape {
                        Shape::Full => reference_groups(&input[..THREADS], width, width, operation),
                        Shape::Partial => {
                            reference_groups(&input[..THREADS], width, valid_items, operation)
                        }
                        Shape::MultipleItems => reference_groups(
                            &input,
                            width * ITEMS_PER_THREAD,
                            width * ITEMS_PER_THREAD,
                            operation,
                        ),
                    };
                    assert_eq!(observed, expected, "case {case}, valid_items {valid_items}");
                    executed += 1;
                }
            }
        }
    }
    Ok(executed)
}

fn run_f32_matrix(
    stream: &CudaStream,
    mut launch_case: impl FnMut(
        u32,
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ) -> Result<(), Box<dyn Error>>,
) -> Result<usize, Box<dyn Error>> {
    let input = (0..THREADS)
        .map(|index| ((index * 73 + 10) % 211) as f32 - 105.0)
        .collect::<Vec<_>>();
    let device_input = DeviceBuffer::from_host(stream, &input)?;
    let mut device_output = DeviceBuffer::<f32>::zeroed(stream, THREADS)?;
    let widths = [32usize, 16, 1];
    let operations: [fn(f32, f32) -> f32; 2] = [f32::max, f32::min];
    let mut executed = 0;

    for (width_index, width) in widths.into_iter().enumerate() {
        for (operation_index, operation) in operations.into_iter().enumerate() {
            let case = (width_index * 2 + operation_index) as u32;
            launch_case(case, &device_input, &mut device_output)?;
            assert_eq!(
                lane_zero_values(&device_output.to_host_vec(stream)?, width),
                reference_groups(&input, width, width, operation),
                "f32 case {case}",
            );
            executed += 1;
        }
    }
    Ok(executed)
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: this executable owns the one module bundle generated by its
    // single `#[cuda_module]` declaration.
    let module = unsafe { kernels::load(&context)? };
    let launch = LaunchConfig1D::new(1, THREADS as u32, 0);
    let mut executed = 0;

    let prepared = module.prepare_u8_matrix(launch)?;
    executed += run_integer_matrix(
        &stream,
        (0..THREADS * ITEMS_PER_THREAD)
            .map(|index| ((index * 37 + 10) % 251) as u8)
            .collect(),
        |case, valid, input, output| {
            module.u8_matrix(&stream, &prepared, input, case, valid, output)?;
            Ok(())
        },
        [u8::wrapping_add, u8::max, u8::min],
    )?;

    let prepared = module.prepare_u16_matrix(launch)?;
    executed += run_integer_matrix(
        &stream,
        (0..THREADS * ITEMS_PER_THREAD)
            .map(|index| ((index * 997 + 10) % 65_521) as u16)
            .collect(),
        |case, valid, input, output| {
            module.u16_matrix(&stream, &prepared, input, case, valid, output)?;
            Ok(())
        },
        [u16::wrapping_add, u16::max, u16::min],
    )?;

    let prepared = module.prepare_i32_matrix(launch)?;
    executed += run_integer_matrix(
        &stream,
        (0..THREADS * ITEMS_PER_THREAD)
            .map(|index| ((index * 193 + 10) % 20_003) as i32 - 10_001)
            .collect(),
        |case, valid, input, output| {
            module.i32_matrix(&stream, &prepared, input, case, valid, output)?;
            Ok(())
        },
        [i32::wrapping_add, i32::max, i32::min],
    )?;

    let prepared = module.prepare_i64_matrix(launch)?;
    executed += run_integer_matrix(
        &stream,
        (0..THREADS * ITEMS_PER_THREAD)
            .map(|index| ((index as i64 * 1_000_003 + 10) % 2_000_033) - 1_000_016)
            .collect(),
        |case, valid, input, output| {
            module.i64_matrix(&stream, &prepared, input, case, valid, output)?;
            Ok(())
        },
        [i64::wrapping_add, i64::max, i64::min],
    )?;

    let prepared = module.prepare_f32_min_max(launch)?;
    executed += run_f32_matrix(&stream, |case, input, output| {
        module.f32_min_max(&stream, &prepared, input, case, output)?;
        Ok(())
    })?;
    assert_eq!(executed, 150);
    println!("pinned CCCL WarpReduce Catch2 supported matrix: PASS ({executed} cases)");
    Ok(())
}
