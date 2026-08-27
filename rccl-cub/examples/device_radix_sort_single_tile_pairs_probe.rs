//! Source-mapped key/value probe for CUB's Policy800 single-tile radix route.
//!
//! Zero one-sweep workspace and duplicate keys make this a branch-and-
//! stability test, not merely another sortedness check.

use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

rccl_cub::cub_kernels!();

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: this executable owns the module emitted by its sole kernel DSL
    // invocation. The operation expansion and device algorithm are safe Rust.
    let module = unsafe { __rccl_cub_kernels::load(&context)? };
    let exec = rccl_cub::cuda::par(&stream, &module);

    let keys_input = (0..4_864u32)
        .map(|index| index.wrapping_mul(17) % 113)
        .collect::<Vec<_>>();
    let values_input = (0..keys_input.len() as u32).collect::<Vec<_>>();
    let mut expected = keys_input
        .iter()
        .copied()
        .zip(values_input.iter().copied())
        .collect::<Vec<_>>();
    expected.sort_by_key(|(key, _)| *key);

    let mut keys = DeviceBuffer::from_host(&stream, &keys_input)?;
    let mut values = DeviceBuffer::from_host(&stream, &values_input)?;
    let mut alternate_keys = DeviceBuffer::<u32>::zeroed(&stream, keys_input.len())?;
    let mut alternate_values = DeviceBuffer::<u32>::zeroed(&stream, keys_input.len())?;
    let mut bins = DeviceBuffer::<u64>::zeroed(&stream, 0)?;
    let mut lookback = DeviceBuffer::<i32>::zeroed(&stream, 0)?;
    let mut counters = DeviceBuffer::<i32>::zeroed(&stream, 0)?;
    let mut buffers = rccl_cub::DoubleBuffer::paired(
        (&mut keys, &mut values),
        (&mut alternate_keys, &mut alternate_values),
    )
    .expect("equal paired buffers form one structural selector");
    rccl_cub::device_radix_sort_pairs_buffer!(::<u32, u32, sm_89>(
        &exec,
        &mut buffers,
        0u32..32u32,
        &mut bins,
        &mut lookback,
        &mut counters,
    ))?;

    let output_keys = buffers.current().0.to_host_vec(&stream)?;
    let output_values = buffers.current().1.to_host_vec(&stream)?;
    assert_eq!(
        output_keys
            .into_iter()
            .zip(output_values)
            .collect::<Vec<_>>(),
        expected
    );
    assert!(!buffers.current_is_first());

    println!("CUB Policy800 single-tile pairs source probe: PASS");
    Ok(())
}
