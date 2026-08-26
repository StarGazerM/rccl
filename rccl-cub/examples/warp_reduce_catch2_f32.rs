//! Floating rows from pinned CUB `catch2_test_warp_reduce.cu`.
//!
//! Together with the 144 integer executions in `warp_reduce_catch2.rs`, this
//! is the complete 150-row supported matrix derived from the pinned test.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract, thread};
use std::error::Error;

const THREADS: usize = 128;

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

#[cuda_module]
mod kernels {
    use super::*;

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

fn reference_groups(
    input: &[f32],
    logical_width: usize,
    operation: fn(f32, f32) -> f32,
) -> Vec<f32> {
    input
        .chunks_exact(logical_width)
        .map(|group| group[1..].iter().copied().fold(group[0], operation))
        .collect()
}

fn lane_zero_values(output: &[f32], logical_width: usize) -> Vec<f32> {
    output.iter().step_by(logical_width).copied().collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: this executable owns the one cuda-oxide module bundle it binds.
    let module = unsafe { kernels::load(&context)? };
    let launch = LaunchConfig1D::new(1, THREADS as u32, 0);
    let prepared = module.prepare_f32_min_max(launch)?;
    let input = (0..THREADS)
        .map(|index| ((index * 73 + 10) % 211) as f32 - 105.0)
        .collect::<Vec<_>>();
    let device_input = DeviceBuffer::from_host(&stream, &input)?;
    let mut device_output = DeviceBuffer::<f32>::zeroed(&stream, THREADS)?;
    let widths = [32usize, 16, 1];
    let operations: [fn(f32, f32) -> f32; 2] = [f32::max, f32::min];
    let mut executed = 0;

    for (width_index, width) in widths.into_iter().enumerate() {
        for (operation_index, operation) in operations.into_iter().enumerate() {
            let case = (width_index * 2 + operation_index) as u32;
            module.f32_min_max(&stream, &prepared, &device_input, case, &mut device_output)?;
            assert_eq!(
                lane_zero_values(&device_output.to_host_vec(&stream)?, width),
                reference_groups(&input, width, operation),
                "f32 case {case}",
            );
            executed += 1;
        }
    }

    assert_eq!(executed, 6);
    println!("pinned CCCL WarpReduce Catch2 f32 matrix: PASS ({executed} cases)");
    Ok(())
}
