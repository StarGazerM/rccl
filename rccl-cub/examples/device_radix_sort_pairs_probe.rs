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
    descending_reference.sort_by_key(|entry| std::cmp::Reverse(entry.0));
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

    // Pinned Catch2's `value_types` includes `uint64_t`; its larger payload
    // selects a distinct Policy800 `RegBoundScaling` result and static union.
    // Keep the same duplicate-heavy keys so an unstable or mismatched payload
    // permutation cannot pass as a correct key-only result.
    let wide_values = (0..keys.len())
        .map(|index| index as u64)
        .collect::<Vec<_>>();
    let mut wide_reference = keys
        .iter()
        .copied()
        .zip(wide_values.iter().copied())
        .collect::<Vec<_>>();
    wide_reference.sort_by_key(|&(key, _)| key);
    let wide_workspace =
        rccl_cub::device_radix_sort_pairs_workspace!(::<u32, u64, sm_89>(keys.len()));
    let mut wide_keys = DeviceBuffer::from_host(&stream, &keys)?;
    let mut wide_values = DeviceBuffer::from_host(&stream, &wide_values)?;
    let mut wide_scratch_keys = DeviceBuffer::<u32>::zeroed(&stream, wide_workspace.scratch_keys)?;
    let mut wide_scratch_values =
        DeviceBuffer::<u64>::zeroed(&stream, wide_workspace.scratch_values)?;
    let mut wide_bins = DeviceBuffer::<u64>::zeroed(&stream, wide_workspace.bins)?;
    let mut wide_lookback = DeviceBuffer::<i32>::zeroed(&stream, wide_workspace.lookback)?;
    let mut wide_counters = DeviceBuffer::<i32>::zeroed(&stream, wide_workspace.counters)?;
    rccl_cub::device_radix_sort_pairs!(::<u32, u64, sm_89>(
        &exec,
        &mut wide_keys,
        &mut wide_values,
        &mut wide_scratch_keys,
        &mut wide_scratch_values,
        &mut wide_bins,
        &mut wide_lookback,
        &mut wide_counters,
    ))?;
    assert_eq!(
        wide_keys.to_host_vec(&stream)?,
        wide_reference
            .iter()
            .map(|&(key, _)| key)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        wide_values.to_host_vec(&stream)?,
        wide_reference
            .iter()
            .map(|&(_, value)| value)
            .collect::<Vec<_>>()
    );

    // The public primitive pair subset also accepts the 64-bit key types from
    // pinned Catch2's key grid. This crosses the multi-tile route and prevents
    // the 32-bit pair union from silently becoming a fallback for that policy.
    let wide_keys_input = keys.iter().map(|&key| key as u64).collect::<Vec<_>>();
    let wide_key_values = (0..keys.len())
        .map(|index| index as u64)
        .collect::<Vec<_>>();
    let mut wide_key_reference = wide_keys_input
        .iter()
        .copied()
        .zip(wide_key_values.iter().copied())
        .collect::<Vec<_>>();
    wide_key_reference.sort_by_key(|&(key, _)| key);
    let wide_key_workspace =
        rccl_cub::device_radix_sort_pairs_workspace!(::<u64, u64, sm_89>(keys.len()));
    let mut wide_pair_keys = DeviceBuffer::from_host(&stream, &wide_keys_input)?;
    let mut wide_pair_values = DeviceBuffer::from_host(&stream, &wide_key_values)?;
    let mut wide_pair_scratch_keys =
        DeviceBuffer::<u64>::zeroed(&stream, wide_key_workspace.scratch_keys)?;
    let mut wide_pair_scratch_values =
        DeviceBuffer::<u64>::zeroed(&stream, wide_key_workspace.scratch_values)?;
    let mut wide_pair_bins = DeviceBuffer::<u64>::zeroed(&stream, wide_key_workspace.bins)?;
    let mut wide_pair_lookback = DeviceBuffer::<i32>::zeroed(&stream, wide_key_workspace.lookback)?;
    let mut wide_pair_counters = DeviceBuffer::<i32>::zeroed(&stream, wide_key_workspace.counters)?;
    rccl_cub::device_radix_sort_pairs!(::<u64, u64, sm_89>(
        &exec,
        &mut wide_pair_keys,
        &mut wide_pair_values,
        &mut wide_pair_scratch_keys,
        &mut wide_pair_scratch_values,
        &mut wide_pair_bins,
        &mut wide_pair_lookback,
        &mut wide_pair_counters,
    ))?;
    assert_eq!(
        wide_pair_keys.to_host_vec(&stream)?,
        wide_key_reference
            .iter()
            .map(|&(key, _)| key)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        wide_pair_values.to_host_vec(&stream)?,
        wide_key_reference
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

    // Pinned `example_device_radix_sort.cu` presents its public pair API with
    // two DoubleBuffers. Rust uses one DoubleBuffer over the key/value tuple,
    // so the selector cannot become desynchronized between the two ranges.
    let mut example_keys = DeviceBuffer::from_host(&stream, &float_input)?;
    let mut example_values = DeviceBuffer::from_host(&stream, &float_input_values)?;
    let mut example_alt_keys = DeviceBuffer::<f32>::zeroed(&stream, float_input.len())?;
    let mut example_alt_values = DeviceBuffer::<i32>::zeroed(&stream, float_input.len())?;
    let mut example_buffers = rccl_cub::DoubleBuffer::paired(
        (&mut example_keys, &mut example_values),
        (&mut example_alt_keys, &mut example_alt_values),
    )
    .expect("all four pair ranges must have the same extent");
    rccl_cub::device_radix_sort_pairs_buffer!(::<f32, i32, sm_89>(
        &exec,
        &mut example_buffers,
        0u32..32u32,
        &mut float_bins,
        &mut float_lookback,
        &mut float_counters,
    ))?;
    let (example_output_keys, example_output_values) = example_buffers.current();
    assert_eq!(
        example_output_keys
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
    assert_eq!(
        example_output_values.to_host_vec(&stream)?,
        [0, 1, 2, 3, 4, 5]
    );

    // Catch2's bit-window grid includes u32 0..24. Three passes force the
    // result into the alternate tuple. Duplicate 24-bit fields plus monotonic
    // payloads make both wrong selection and unstable scatter fail exactly.
    let pair_window = 0u32..24u32;
    let window_keys = (0..10_003u32)
        .map(|index| ((index & 0xff) << 24) | (index.wrapping_mul(97) & 0x3ff))
        .collect::<Vec<_>>();
    let window_values = (0..window_keys.len())
        .map(|index| index as i32)
        .collect::<Vec<_>>();
    let window_workspace = rccl_cub::device_radix_sort_pairs_workspace!(::<u32, i32, sm_89>(
        window_keys.len()
    ));
    let mut window_bins = DeviceBuffer::<u64>::zeroed(&stream, window_workspace.bins)?;
    let mut window_lookback = DeviceBuffer::<i32>::zeroed(&stream, window_workspace.lookback)?;
    let mut window_counters = DeviceBuffer::<i32>::zeroed(&stream, window_workspace.counters)?;
    for descending in [false, true] {
        let mut first_keys = DeviceBuffer::from_host(&stream, &window_keys)?;
        let mut first_values = DeviceBuffer::from_host(&stream, &window_values)?;
        let mut second_keys = DeviceBuffer::<u32>::zeroed(&stream, window_keys.len())?;
        let mut second_values = DeviceBuffer::<i32>::zeroed(&stream, window_values.len())?;
        let mut buffers = rccl_cub::DoubleBuffer::paired(
            (&mut first_keys, &mut first_values),
            (&mut second_keys, &mut second_values),
        )
        .expect("all four pair ranges must have the same extent");
        if descending {
            rccl_cub::device_radix_sort_pairs_buffer_descending!(::<u32, i32, sm_89>(
                &exec,
                &mut buffers,
                pair_window.clone(),
                &mut window_bins,
                &mut window_lookback,
                &mut window_counters,
            ))?;
        } else {
            rccl_cub::device_radix_sort_pairs_buffer!(::<u32, i32, sm_89>(
                &exec,
                &mut buffers,
                pair_window.clone(),
                &mut window_bins,
                &mut window_lookback,
                &mut window_counters,
            ))?;
        }
        let mut expected = window_keys
            .iter()
            .copied()
            .zip(window_values.iter().copied())
            .collect::<Vec<_>>();
        if descending {
            expected.sort_by_key(|entry| std::cmp::Reverse(entry.0 & 0x00ff_ffff));
        } else {
            expected.sort_by_key(|pair| pair.0 & 0x00ff_ffff);
        }
        let (output_keys, output_values) = buffers.current();
        assert_eq!(
            output_keys.to_host_vec(&stream)?,
            expected.iter().map(|pair| pair.0).collect::<Vec<_>>()
        );
        assert_eq!(
            output_values.to_host_vec(&stream)?,
            expected.iter().map(|pair| pair.1).collect::<Vec<_>>()
        );
    }

    println!("pinned CUB DeviceRadixSort pair cases: PASS");
    Ok(())
}
