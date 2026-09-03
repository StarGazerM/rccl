//! Pair cases from pinned CUB `catch2_test_device_radix_sort_pairs.cu` and
//! `examples/device/example_device_radix_sort.cu`.
//!
//! CUB checks 0/1/random sizes in both directions and verifies the stable
//! key-value permutation. Its shipped example uses `f32` keys with `i32`
//! values. The Rust port keeps those as direct borrowed ranges plus one
//! caller-owned opaque workspace.

use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let cub = rccl_cub::cuda::Program::load(&context)?;
    let launch = cub.on(&stream)?;

    let mut empty_workspace = rccl_cub::device_radix_sort_pairs_workspace::<u32, i32>(&launch, 0)?;
    let mut empty_keys = DeviceBuffer::from_host(&stream, &[])?;
    let mut empty_values = DeviceBuffer::from_host(&stream, &[])?;
    rccl_cub::device_radix_sort_pairs(
        &launch,
        &mut empty_keys,
        &mut empty_values,
        &mut empty_workspace,
    )?;
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

    let mut workspace =
        rccl_cub::device_radix_sort_pairs_workspace::<u32, i32>(&launch, keys.len())?;
    let mut device_keys = DeviceBuffer::from_host(&stream, &keys)?;
    let mut device_values = DeviceBuffer::from_host(&stream, &values)?;
    rccl_cub::device_radix_sort_pairs(
        &launch,
        &mut device_keys,
        &mut device_values,
        &mut workspace,
    )?;
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
    rccl_cub::device_radix_sort_pairs_descending(
        &launch,
        &mut descending_keys,
        &mut descending_values,
        &mut workspace,
    )?;
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
    let mut wide_workspace =
        rccl_cub::device_radix_sort_pairs_workspace::<u32, u64>(&launch, keys.len())?;
    let mut wide_keys = DeviceBuffer::from_host(&stream, &keys)?;
    let mut wide_values = DeviceBuffer::from_host(&stream, &wide_values)?;
    rccl_cub::device_radix_sort_pairs(
        &launch,
        &mut wide_keys,
        &mut wide_values,
        &mut wide_workspace,
    )?;
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
    let mut wide_key_workspace =
        rccl_cub::device_radix_sort_pairs_workspace::<u64, u64>(&launch, keys.len())?;
    let mut wide_pair_keys = DeviceBuffer::from_host(&stream, &wide_keys_input)?;
    let mut wide_pair_values = DeviceBuffer::from_host(&stream, &wide_key_values)?;
    rccl_cub::device_radix_sort_pairs(
        &launch,
        &mut wide_pair_keys,
        &mut wide_pair_values,
        &mut wide_key_workspace,
    )?;
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
    let mut float_workspace =
        rccl_cub::device_radix_sort_pairs_workspace::<f32, i32>(&launch, float_input.len())?;
    let mut float_keys = DeviceBuffer::from_host(&stream, &float_input)?;
    let mut float_values = DeviceBuffer::from_host(&stream, &float_input_values)?;
    rccl_cub::device_radix_sort_pairs(
        &launch,
        &mut float_keys,
        &mut float_values,
        &mut float_workspace,
    )?;
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
    let mut example_workspace =
        rccl_cub::device_radix_sort_pairs_buffer_workspace::<f32, i32>(&launch, float_input.len())?;
    rccl_cub::device_radix_sort_pairs_buffer(
        &launch,
        &mut example_buffers,
        0u32..32u32,
        &mut example_workspace,
    )?;
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
    let mut window_workspace =
        rccl_cub::device_radix_sort_pairs_buffer_workspace::<u32, i32>(&launch, window_keys.len())?;
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
            rccl_cub::device_radix_sort_pairs_buffer_descending(
                &launch,
                &mut buffers,
                pair_window.clone(),
                &mut window_workspace,
            )?;
        } else {
            rccl_cub::device_radix_sort_pairs_buffer(
                &launch,
                &mut buffers,
                pair_window.clone(),
                &mut window_workspace,
            )?;
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
