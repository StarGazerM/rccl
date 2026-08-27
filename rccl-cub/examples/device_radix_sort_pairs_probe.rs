//! Pair cases from pinned CUB `catch2_test_device_radix_sort_pairs.cu` and
//! `examples/device/example_device_radix_sort.cu`.
//!
//! CUB checks 0/1/random sizes in both directions and verifies the stable
//! key-value permutation. Its shipped example uses `f32` keys with `i32`
//! values. The Rust port keeps those as direct borrowed ranges plus explicit,
//! caller-owned Policy800 workspace.

use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

rccl_cub::cub_kernels!();

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: this executable owns the artifact emitted by its sole
    // `cub_kernels!` invocation. Algorithm lowering itself is safe Rust.
    let module = unsafe { __rccl_cub_kernels::load(&context)? };
    let exec = rccl_cub::cuda::par(&stream, &module);

    let empty_workspace = rccl_cub::device_radix_sort_pairs_workspace!(::<u32, i32, sm_89>(0));
    let mut empty_keys = DeviceBuffer::from_host(&stream, &[])?;
    let mut empty_values = DeviceBuffer::from_host(&stream, &[])?;
    let mut empty_scratch_keys = DeviceBuffer::<u32>::zeroed(&stream, 0)?;
    let mut empty_scratch_values = DeviceBuffer::<i32>::zeroed(&stream, 0)?;
    let mut empty_bins = DeviceBuffer::<u64>::zeroed(&stream, empty_workspace.bins)?;
    let mut empty_lookback = DeviceBuffer::<i32>::zeroed(&stream, empty_workspace.lookback)?;
    let mut empty_counters = DeviceBuffer::<i32>::zeroed(&stream, empty_workspace.counters)?;
    rccl_cub::device_radix_sort_pairs!(::<u32, i32, sm_89>(
        &exec,
        &mut empty_keys,
        &mut empty_values,
        &mut empty_scratch_keys,
        &mut empty_scratch_values,
        &mut empty_bins,
        &mut empty_lookback,
        &mut empty_counters,
    ))?;
    assert!(empty_keys.to_host_vec(&stream)?.is_empty());
    assert!(empty_values.to_host_vec(&stream)?.is_empty());

    // Cross more than one 384 * 17 pair tile and leave a partial final tile.
    // A deliberately narrow key domain supplies many equal-key runs, so the
    // value sequence proves CUB's stable pair permutation rather than merely
    // checking that keys are ordered.
    let keys = (0..10_003u32)
        .map(|index| index.wrapping_mul(2_654_435_761).rotate_left(index & 31) & 0x3ff)
        .collect::<Vec<_>>();
    let values = (0..keys.len())
        .map(|index| index as i32)
        .collect::<Vec<_>>();
    let mut ascending_reference = keys
        .iter()
        .copied()
        .zip(values.iter().copied())
        .collect::<Vec<_>>();
    ascending_reference.sort_by_key(|&(key, _)| key);

    let workspace = rccl_cub::device_radix_sort_pairs_workspace!(::<u32, i32, sm_89>(keys.len()));
    let mut device_keys = DeviceBuffer::from_host(&stream, &keys)?;
    let mut device_values = DeviceBuffer::from_host(&stream, &values)?;
    let mut scratch_keys = DeviceBuffer::<u32>::zeroed(&stream, workspace.scratch_keys)?;
    let mut scratch_values = DeviceBuffer::<i32>::zeroed(&stream, workspace.scratch_values)?;
    let mut bins = DeviceBuffer::<u64>::zeroed(&stream, workspace.bins)?;
    let mut lookback = DeviceBuffer::<i32>::zeroed(&stream, workspace.lookback)?;
    let mut counters = DeviceBuffer::<i32>::zeroed(&stream, workspace.counters)?;
    rccl_cub::device_radix_sort_pairs!(::<u32, i32, sm_89>(
        &exec,
        &mut device_keys,
        &mut device_values,
        &mut scratch_keys,
        &mut scratch_values,
        &mut bins,
        &mut lookback,
        &mut counters,
    ))?;
    assert_eq!(
        device_keys.to_host_vec(&stream)?,
        ascending_reference
            .iter()
            .map(|&(key, _)| key)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        device_values.to_host_vec(&stream)?,
        ascending_reference
            .iter()
            .map(|&(_, value)| value)
            .collect::<Vec<_>>()
    );

    let mut descending_reference = keys
        .iter()
        .copied()
        .zip(values.iter().copied())
        .collect::<Vec<_>>();
    descending_reference.sort_by(|left, right| right.0.cmp(&left.0));
    let mut descending_keys = DeviceBuffer::from_host(&stream, &keys)?;
    let mut descending_values = DeviceBuffer::from_host(&stream, &values)?;
    rccl_cub::device_radix_sort_pairs_descending!(::<u32, i32, sm_89>(
        &exec,
        &mut descending_keys,
        &mut descending_values,
        &mut scratch_keys,
        &mut scratch_values,
        &mut bins,
        &mut lookback,
        &mut counters,
    ))?;
    assert_eq!(
        descending_keys.to_host_vec(&stream)?,
        descending_reference
            .iter()
            .map(|&(key, _)| key)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        descending_values.to_host_vec(&stream)?,
        descending_reference
            .iter()
            .map(|&(_, value)| value)
            .collect::<Vec<_>>()
    );

    // Pinned CUB's shipped pair example is f32/i32. This fixed case includes
    // both signed zeros and validates keys by bits so float equality cannot
    // mask a key twiddle or pair-permutation error.
    let float_input = [1.25_f32, -0.0, f32::INFINITY, -3.5, 0.0, f32::NEG_INFINITY];
    let float_input_values = [4_i32, 2, 5, 1, 3, 0];
    let float_workspace = rccl_cub::device_radix_sort_pairs_workspace!(::<f32, i32, sm_89>(
        float_input.len()
    ));
    let mut float_keys = DeviceBuffer::from_host(&stream, &float_input)?;
    let mut float_values = DeviceBuffer::from_host(&stream, &float_input_values)?;
    let mut float_scratch_keys =
        DeviceBuffer::<f32>::zeroed(&stream, float_workspace.scratch_keys)?;
    let mut float_scratch_values =
        DeviceBuffer::<i32>::zeroed(&stream, float_workspace.scratch_values)?;
    let mut float_bins = DeviceBuffer::<u64>::zeroed(&stream, float_workspace.bins)?;
    let mut float_lookback = DeviceBuffer::<i32>::zeroed(&stream, float_workspace.lookback)?;
    let mut float_counters = DeviceBuffer::<i32>::zeroed(&stream, float_workspace.counters)?;
    rccl_cub::device_radix_sort_pairs!(::<f32, i32, sm_89>(
        &exec,
        &mut float_keys,
        &mut float_values,
        &mut float_scratch_keys,
        &mut float_scratch_values,
        &mut float_bins,
        &mut float_lookback,
        &mut float_counters,
    ))?;
    assert_eq!(
        float_keys
            .to_host_vec(&stream)?
            .iter()
            .map(|key| key.to_bits())
            .collect::<Vec<_>>(),
        [
            f32::NEG_INFINITY.to_bits(),
            (-3.5_f32).to_bits(),
            (-0.0_f32).to_bits(),
            0.0_f32.to_bits(),
            1.25_f32.to_bits(),
            f32::INFINITY.to_bits(),
        ]
    );
    assert_eq!(float_values.to_host_vec(&stream)?, [0, 1, 2, 3, 4, 5]);

    println!("pinned CUB DeviceRadixSort pair cases: PASS");
    Ok(())
}
