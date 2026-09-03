//! Keys-only cases from the pinned CUB `DeviceRadixSort` tests, using the
//! one typed, opaque Rust-owned workspace per operation.

use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

fn u32_field(value: u32, bits: core::ops::Range<u32>) -> u32 {
    let width = bits.end - bits.start;
    if width == u32::BITS {
        value
    } else if width == 0 {
        0
    } else {
        (value >> bits.start) & ((1u32 << width) - 1)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let cub = rccl_cub::cuda::Program::load(&context)?;
    let launch = cub.on(&stream)?;

    // Pinned basic testing starts with the zero-length and one-length cases.
    // The empty call must validate the Rust-owned workspace shape and return
    // without preparing or launching a CUDA kernel.
    let empty_input: [u32; 0] = [];
    let mut empty_keys = DeviceBuffer::from_host(&stream, &empty_input)?;
    let mut empty_workspace = rccl_cub::device_radix_sort_workspace::<u32>(&launch, 0)?;
    rccl_cub::device_radix_sort_keys(&launch, &mut empty_keys, &mut empty_workspace)?;
    assert!(empty_keys.to_host_vec(&stream)?.is_empty());

    // More than one 384 * 21 tile and a partial final tile exercise CUB's
    // dynamic block counter and decoupled lookback, not merely its one-block
    // special case.
    let mut input = (0..10_003u32)
        .map(|index| index.wrapping_mul(2_654_435_761).rotate_left(index & 31))
        .collect::<Vec<_>>();
    input[17] = 0;
    input[29] = u32::MAX;
    let mut expected = input.clone();
    expected.sort_unstable();
    let mut keys = DeviceBuffer::from_host(&stream, &input)?;
    let mut workspace = rccl_cub::device_radix_sort_workspace::<u32>(&launch, input.len())?;
    rccl_cub::device_radix_sort_keys(&launch, &mut keys, &mut workspace)?;
    assert_eq!(keys.to_host_vec(&stream)?, expected);

    // Pinned basic testing explicitly includes one item. This also checks that
    // the one-block path does not accidentally depend on a preceding block's
    // lookback publication.
    let one_input = [42_u32];
    let mut one_key = DeviceBuffer::from_host(&stream, &one_input)?;
    let mut one_workspace = rccl_cub::device_radix_sort_workspace::<u32>(&launch, one_input.len())?;
    rccl_cub::device_radix_sort_keys(&launch, &mut one_key, &mut one_workspace)?;
    assert_eq!(one_key.to_host_vec(&stream)?, one_input);

    // Policy800's RegBoundScaling changes the one-sweep tile from 384 * 21
    // for four-byte keys to 384 * 10 for eight-byte keys. Cross several of
    // those exact tiles so this case tests the macro-selected 64-bit policy,
    // rather than only the common bit-order conversion.
    let input64 = (0..10_003u64)
        .map(|index| {
            index
                .wrapping_mul(11_400_714_819_323_198_485)
                .rotate_left(index as u32 & 63)
        })
        .collect::<Vec<_>>();
    let mut expected64 = input64.clone();
    expected64.sort_unstable();
    let mut keys64 = DeviceBuffer::from_host(&stream, &input64)?;
    let mut workspace64 = rccl_cub::device_radix_sort_workspace::<u64>(&launch, input64.len())?;
    rccl_cub::device_radix_sort_keys(&launch, &mut keys64, &mut workspace64)?;
    assert_eq!(keys64.to_host_vec(&stream)?, expected64);

    let signed_input = [0_i32, -1, i32::MIN, 7, -9, i32::MAX, -1, 0];
    let mut signed_keys = DeviceBuffer::from_host(&stream, &signed_input)?;
    let mut signed_workspace =
        rccl_cub::device_radix_sort_workspace::<i32>(&launch, signed_input.len())?;
    rccl_cub::device_radix_sort_keys_descending(&launch, &mut signed_keys, &mut signed_workspace)?;
    assert_eq!(
        signed_keys.to_host_vec(&stream)?,
        [i32::MAX, 7, 0, 0, -1, -1, -9, i32::MIN]
    );

    let signed64_input = [0_i64, -1, i64::MIN, 7, -9, i64::MAX, -1, 0];
    let mut signed64_keys = DeviceBuffer::from_host(&stream, &signed64_input)?;
    let mut signed64_workspace =
        rccl_cub::device_radix_sort_workspace::<i64>(&launch, signed64_input.len())?;
    rccl_cub::device_radix_sort_keys_descending(
        &launch,
        &mut signed64_keys,
        &mut signed64_workspace,
    )?;
    assert_eq!(
        signed64_keys.to_host_vec(&stream)?,
        [i64::MAX, 7, 0, 0, -1, -1, -9, i64::MIN]
    );

    // CUB treats -0 and +0 as equal radix keys while preserving their bits.
    let floating_input = [3.0_f32, -0.0, f32::NEG_INFINITY, 0.0, -7.5, f32::INFINITY];
    let mut floating_keys = DeviceBuffer::from_host(&stream, &floating_input)?;
    let mut floating_workspace =
        rccl_cub::device_radix_sort_workspace::<f32>(&launch, floating_input.len())?;
    rccl_cub::device_radix_sort_keys(&launch, &mut floating_keys, &mut floating_workspace)?;
    let floating_output = floating_keys.to_host_vec(&stream)?;
    assert_eq!(
        floating_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            f32::NEG_INFINITY.to_bits(),
            (-7.5_f32).to_bits(),
            (-0.0_f32).to_bits(),
            0.0_f32.to_bits(),
            3.0_f32.to_bits(),
            f32::INFINITY.to_bits(),
        ]
    );

    // Pinned NaN handling compares the output as bits. Keep a fixed oracle so
    // the Rust implementation's own float twiddle cannot validate itself.
    let negative_quiet_nan32 = f32::from_bits(0xffc0_0001);
    let negative_signaling_nan32 = f32::from_bits(0xff80_0001);
    let positive_signaling_nan32 = f32::from_bits(0x7f80_0001);
    let positive_quiet_nan32 = f32::from_bits(0x7fc0_0001);
    let nan_input32 = [
        positive_quiet_nan32,
        1.0,
        negative_signaling_nan32,
        negative_quiet_nan32,
        -1.0,
        positive_signaling_nan32,
    ];
    let mut nan_workspace32 =
        rccl_cub::device_radix_sort_workspace::<f32>(&launch, nan_input32.len())?;
    for descending in [false, true] {
        let mut nan_keys32 = DeviceBuffer::from_host(&stream, &nan_input32)?;
        if descending {
            rccl_cub::device_radix_sort_keys_descending(
                &launch,
                &mut nan_keys32,
                &mut nan_workspace32,
            )?;
        } else {
            rccl_cub::device_radix_sort_keys(&launch, &mut nan_keys32, &mut nan_workspace32)?;
        }
        let output_bits = nan_keys32
            .to_host_vec(&stream)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let expected_bits = if descending {
            vec![
                positive_quiet_nan32.to_bits(),
                positive_signaling_nan32.to_bits(),
                1.0_f32.to_bits(),
                (-1.0_f32).to_bits(),
                negative_signaling_nan32.to_bits(),
                negative_quiet_nan32.to_bits(),
            ]
        } else {
            vec![
                negative_quiet_nan32.to_bits(),
                negative_signaling_nan32.to_bits(),
                (-1.0_f32).to_bits(),
                1.0_f32.to_bits(),
                positive_signaling_nan32.to_bits(),
                positive_quiet_nan32.to_bits(),
            ]
        };
        assert_eq!(output_bits, expected_bits);
    }

    let floating64_input = [3.0_f64, -0.0, f64::NEG_INFINITY, 0.0, -7.5, f64::INFINITY];
    let mut floating64_keys = DeviceBuffer::from_host(&stream, &floating64_input)?;
    let mut floating64_workspace =
        rccl_cub::device_radix_sort_workspace::<f64>(&launch, floating64_input.len())?;
    rccl_cub::device_radix_sort_keys(&launch, &mut floating64_keys, &mut floating64_workspace)?;
    let floating64_output = floating64_keys.to_host_vec(&stream)?;
    assert_eq!(
        floating64_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            f64::NEG_INFINITY.to_bits(),
            (-7.5_f64).to_bits(),
            (-0.0_f64).to_bits(),
            0.0_f64.to_bits(),
            3.0_f64.to_bits(),
            f64::INFINITY.to_bits(),
        ]
    );

    let negative_quiet_nan64 = f64::from_bits(0xfff8_0000_0000_0001);
    let negative_signaling_nan64 = f64::from_bits(0xfff0_0000_0000_0001);
    let positive_signaling_nan64 = f64::from_bits(0x7ff0_0000_0000_0001);
    let positive_quiet_nan64 = f64::from_bits(0x7ff8_0000_0000_0001);
    let nan_input64 = [
        positive_quiet_nan64,
        1.0,
        negative_signaling_nan64,
        negative_quiet_nan64,
        -1.0,
        positive_signaling_nan64,
    ];
    let mut nan_workspace64 =
        rccl_cub::device_radix_sort_workspace::<f64>(&launch, nan_input64.len())?;
    for descending in [false, true] {
        let mut nan_keys64 = DeviceBuffer::from_host(&stream, &nan_input64)?;
        if descending {
            rccl_cub::device_radix_sort_keys_descending(
                &launch,
                &mut nan_keys64,
                &mut nan_workspace64,
            )?;
        } else {
            rccl_cub::device_radix_sort_keys(&launch, &mut nan_keys64, &mut nan_workspace64)?;
        }
        let output_bits = nan_keys64
            .to_host_vec(&stream)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let expected_bits = if descending {
            vec![
                positive_quiet_nan64.to_bits(),
                positive_signaling_nan64.to_bits(),
                1.0_f64.to_bits(),
                (-1.0_f64).to_bits(),
                negative_signaling_nan64.to_bits(),
                negative_quiet_nan64.to_bits(),
            ]
        } else {
            vec![
                negative_quiet_nan64.to_bits(),
                negative_signaling_nan64.to_bits(),
                (-1.0_f64).to_bits(),
                1.0_f64.to_bits(),
                positive_signaling_nan64.to_bits(),
                positive_quiet_nan64.to_bits(),
            ]
        };
        assert_eq!(output_bits, expected_bits);
    }

    // Pinned entropy-reduction testing produces many duplicate, non-uniform
    // keys. Run both directions across many Policy800 tiles so the uniform
    // short-circuit cannot mask rank/scatter errors.
    let entropy_input = (0..((1 << 17) + 3) as u32)
        .map(|index| index.wrapping_mul(2_654_435_761) & 31)
        .collect::<Vec<_>>();
    let mut entropy_workspace =
        rccl_cub::device_radix_sort_workspace::<u32>(&launch, entropy_input.len())?;
    for descending in [false, true] {
        let mut entropy_keys = DeviceBuffer::from_host(&stream, &entropy_input)?;
        if descending {
            rccl_cub::device_radix_sort_keys_descending(
                &launch,
                &mut entropy_keys,
                &mut entropy_workspace,
            )?;
        } else {
            rccl_cub::device_radix_sort_keys(&launch, &mut entropy_keys, &mut entropy_workspace)?;
        }
        let mut expected_entropy = entropy_input.clone();
        if descending {
            expected_entropy.sort_unstable_by(|left, right| right.cmp(left));
        } else {
            expected_entropy.sort_unstable();
        }
        assert_eq!(entropy_keys.to_host_vec(&stream)?, expected_entropy);
    }

    // Pinned `catch2_test_device_radix_sort_keys.cu` has a dedicated
    // "uniform values" case in both directions. More than sixteen Policy800
    // tiles exercises CUB's counts-callback short circuit and lookback chain.
    let uniform_input = vec![4_u32; (1 << 17) + 3];
    let mut uniform_workspace =
        rccl_cub::device_radix_sort_workspace::<u32>(&launch, uniform_input.len())?;
    for descending in [false, true] {
        let mut uniform_keys = DeviceBuffer::from_host(&stream, &uniform_input)?;
        if descending {
            rccl_cub::device_radix_sort_keys_descending(
                &launch,
                &mut uniform_keys,
                &mut uniform_workspace,
            )?;
        } else {
            rccl_cub::device_radix_sort_keys(&launch, &mut uniform_keys, &mut uniform_workspace)?;
        }
        assert_eq!(uniform_keys.to_host_vec(&stream)?, uniform_input);
    }

    // Pinned CUB Catch2 enumerates begin/end bit windows and separately tests
    // its DoubleBuffer overload. Combine those contracts here: 0..24 is one
    // of the pinned u32 windows and takes three passes, so reading the original
    // allocation would fail even when its stale contents happen to be sorted.
    let window = 0u32..24u32;
    let window_input = (0..10_003u32)
        .map(|index| {
            let low = index.wrapping_mul(2_654_435_761) & 0x00ff_ffff;
            let high = (index & 0xff) << 24;
            high | low
        })
        .collect::<Vec<_>>();
    let mut window_workspace =
        rccl_cub::device_radix_sort_keys_buffer_workspace::<u32>(&launch, window_input.len())?;
    for descending in [false, true] {
        let mut first = DeviceBuffer::from_host(&stream, &window_input)?;
        let mut second = DeviceBuffer::<u32>::zeroed(&stream, window_input.len())?;
        let mut buffers = rccl_cub::DoubleBuffer::new(&mut first, &mut second)
            .expect("equal device ranges must construct");
        if descending {
            rccl_cub::device_radix_sort_keys_buffer_descending(
                &launch,
                &mut buffers,
                window.clone(),
                &mut window_workspace,
            )?;
        } else {
            rccl_cub::device_radix_sort_keys_buffer(
                &launch,
                &mut buffers,
                window.clone(),
                &mut window_workspace,
            )?;
        }
        let mut expected_window = window_input.clone();
        if descending {
            expected_window.sort_by(|left, right| {
                u32_field(*right, window.clone()).cmp(&u32_field(*left, window.clone()))
            });
        } else {
            expected_window.sort_by_key(|value| u32_field(*value, window.clone()));
        }
        assert_eq!(buffers.current().to_host_vec(&stream)?, expected_window);
    }

    // CUB's overwrite-okay DoubleBuffer path returns without changing its
    // selection for an empty bit window. Distinct buffer contents make a
    // spurious swap observable. The workspace macro selects zero agent state
    // for this single-tile extent.
    let no_op_input = [9u32, 1, 7, 3];
    let mut no_op_first = DeviceBuffer::from_host(&stream, &no_op_input)?;
    let mut no_op_second = DeviceBuffer::from_host(&stream, &[99u32, 99, 99, 99])?;
    let mut no_op_buffers = rccl_cub::DoubleBuffer::new(&mut no_op_first, &mut no_op_second)
        .expect("equal device ranges must construct");
    let mut no_op_workspace =
        rccl_cub::device_radix_sort_keys_buffer_workspace::<u32>(&launch, no_op_input.len())?;
    rccl_cub::device_radix_sort_keys_buffer(
        &launch,
        &mut no_op_buffers,
        12u32..12u32,
        &mut no_op_workspace,
    )?;
    assert_eq!(no_op_buffers.current().to_host_vec(&stream)?, no_op_input);

    let invalid_begin = 25u32;
    let invalid_end = 17u32;
    let invalid = rccl_cub::device_radix_sort_keys_buffer(
        &launch,
        &mut no_op_buffers,
        invalid_begin..invalid_end,
        &mut no_op_workspace,
    );
    assert!(matches!(
        invalid,
        Err(rccl_cub::DeviceAlgorithmError::InvalidBitRange {
            begin: 25,
            end: 17,
            key_bits: 32
        })
    ));
    assert_eq!(no_op_buffers.current().to_host_vec(&stream)?, no_op_input);

    // The same zero workspace must now succeed for a nonempty small problem;
    // otherwise the dispatcher is still masking a missing single-tile route.
    rccl_cub::device_radix_sort_keys_buffer(
        &launch,
        &mut no_op_buffers,
        0u32..8u32,
        &mut no_op_workspace,
    )?;
    let mut no_op_expected = no_op_input;
    no_op_expected.sort_by_key(|value| value & 0xff);
    assert_eq!(
        no_op_buffers.current().to_host_vec(&stream)?,
        no_op_expected
    );

    // One item beyond Policy800's exact 256 * 19 threshold selects the
    // one-sweep workspace and route.
    let boundary_input = (0..4_865u32).rev().collect::<Vec<_>>();
    let mut boundary_first = DeviceBuffer::from_host(&stream, &boundary_input)?;
    let mut boundary_second = DeviceBuffer::<u32>::zeroed(&stream, boundary_input.len())?;
    let mut boundary_buffers =
        rccl_cub::DoubleBuffer::new(&mut boundary_first, &mut boundary_second).unwrap();
    let mut boundary_workspace =
        rccl_cub::device_radix_sort_keys_buffer_workspace::<u32>(&launch, boundary_input.len())?;
    rccl_cub::device_radix_sort_keys_buffer(
        &launch,
        &mut boundary_buffers,
        0u32..8u32,
        &mut boundary_workspace,
    )?;
    let mut boundary_expected = boundary_input;
    boundary_expected.sort_by_key(|value| value & 0xff);
    assert_eq!(
        boundary_buffers.current().to_host_vec(&stream)?,
        boundary_expected
    );

    println!("pinned CUB DeviceRadixSort keys cases: PASS");
    Ok(())
}
