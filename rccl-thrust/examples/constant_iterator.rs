//! Rust port of pinned CCCL `thrust/examples/constant_iterator.cu`.

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{DeviceGenerateExt, DeviceReduceExt, DeviceTransformExt, repeat_n};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let thrust = rccl_thrust::cuda::Program::load(&context)?;
    let exec = thrust.par(&stream)?;

    let mut data = DeviceBuffer::from_host(&stream, &[3_i32, 7, 2, 5])?;
    let repeated = repeat_n(10_i32, data.len());
    rccl_thrust::transform(&exec, &mut data, repeated, |left, right| left + right)?;

    assert_eq!(data.to_host_vec(&stream)?, [13, 17, 12, 15]);

    let mut rust_data = DeviceBuffer::from_host(&stream, &[3_i32, 7, 2, 5])?;
    rust_data.map_with_value_in_place(&exec, 10_i32, |value, increment| value + increment)?;
    assert_eq!(rust_data.to_host_vec(&stream)?, [13, 17, 12, 15]);

    let mut filled = DeviceBuffer::<i32>::zeroed(&stream, 6)?;
    filled.fill_device(&exec, 137)?;
    assert_eq!(filled.to_host_vec(&stream)?, [137, 137, 137, 137, 137, 137]);

    let mut tabulated = DeviceBuffer::<i32>::zeroed(&stream, 6)?;
    tabulated.tabulate(&exec, |index| -(index as i32))?;
    assert_eq!(tabulated.to_host_vec(&stream)?, [0, -1, -2, -3, -4, -5]);

    let mut sequenced = DeviceBuffer::<i32>::zeroed(&stream, 6)?;
    sequenced.sequence_by(&exec, 1, 3)?;
    assert_eq!(sequenced.to_host_vec(&stream)?, [1, 4, 7, 10, 13, 16]);

    assert_eq!(filled.reduce(&exec)?, 822);
    assert_eq!(rccl_thrust::reduce(&exec, &filled)?, 822);

    let large = DeviceBuffer::from_host(&stream, &vec![1_i32; 5_000])?;
    assert_eq!(large.reduce(&exec)?, 5_000);

    let empty = DeviceBuffer::<i32>::zeroed(&stream, 0)?;
    assert_eq!(empty.reduce(&exec)?, 0);
    println!("pinned CCCL constant_iterator example: PASS");
    Ok(())
}
