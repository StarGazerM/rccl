//! Rust port of pinned CCCL `thrust/examples/constant_iterator.cu`.

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::repeat_n;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let thrust = rccl_thrust::thrust_program!(&context)?;
    let exec = thrust.par(&stream)?;

    let mut data = DeviceBuffer::from_host(&stream, &[3_i32, 7, 2, 5])?;
    let repeated = repeat_n(10_i32, data.len());
    rccl_thrust::transform!(&exec, &mut data, repeated, |left, right| left + right)?;

    assert_eq!(data.to_host_vec(&stream)?, [13, 17, 12, 15]);
    println!("pinned CCCL constant_iterator example: PASS");
    Ok(())
}
