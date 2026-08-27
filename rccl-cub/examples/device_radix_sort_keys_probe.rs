//! Keys-only cases from the pinned CUB `DeviceRadixSort` tests, using the
//! Policy800 one-sweep workspace as distinct Rust-owned device buffers.

use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

rccl_cub::cub_kernels!();

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
    // SAFETY: this executable owns the artifact emitted by its sole
    // `cub_kernels!` invocation. Algorithm lowering itself is safe Rust.
    let module = unsafe { __rccl_cub_kernels::load(&context)? };
    let exec = rccl_cub::cuda::par(&stream, &module);

    // Pinned basic testing starts with the zero-length and one-length cases.
    // The empty call must validate the Rust-owned workspace shape and return
    // without preparing or launching a CUDA kernel.
    let empty_input: [u32; 0] = [];
    let mut empty_keys = DeviceBuffer::from_host(&stream, &empty_input)?;
    let empty_workspace = rccl_cub::device_radix_sort_workspace!(::<u32, sm_89>(0));
    let mut empty_scratch = DeviceBuffer::<u32>::zeroed(&stream, 0)?;
    let mut empty_bins = DeviceBuffer::<u64>::zeroed(&stream, empty_workspace.bins)?;
    let mut empty_lookback = DeviceBuffer::<i32>::zeroed(&stream, empty_workspace.lookback)?;
    let mut empty_counters = DeviceBuffer::<i32>::zeroed(&stream, empty_workspace.counters)?;
    rccl_cub::device_radix_sort_keys!(::<u32, sm_89>(
        &exec,
        &mut empty_keys,
        &mut empty_scratch,
        &mut empty_bins,
        &mut empty_lookback,
        &mut empty_counters,
    ))?;
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
    let workspace = rccl_cub::device_radix_sort_workspace!(::<u32, sm_89>(input.len()));
    let mut scratch = DeviceBuffer::<u32>::zeroed(&stream, workspace.scratch_keys)?;
    let mut bins = DeviceBuffer::<u64>::zeroed(&stream, workspace.bins)?;
    let mut lookback = DeviceBuffer::<i32>::zeroed(&stream, workspace.lookback)?;
    let mut counters = DeviceBuffer::<i32>::zeroed(&stream, workspace.counters)?;
    rccl_cub::device_radix_sort_keys!(::<u32, sm_89>(
        &exec,
        &mut keys,
        &mut scratch,
        &mut bins,
        &mut lookback,
        &mut counters,
    ))?;
    assert_eq!(keys.to_host_vec(&stream)?, expected);

    // Pinned basic testing explicitly includes one item. This also checks that
    // the one-block path does not accidentally depend on a preceding block's
    // lookback publication.
    let one_input = [42_u32];
    let mut one_key = DeviceBuffer::from_host(&stream, &one_input)?;
    let one_workspace = rccl_cub::device_radix_sort_workspace!(::<u32, sm_89>(one_input.len()));
    let mut one_scratch = DeviceBuffer::<u32>::zeroed(&stream, one_workspace.scratch_keys)?;
    let mut one_bins = DeviceBuffer::<u64>::zeroed(&stream, one_workspace.bins)?;
    let mut one_lookback = DeviceBuffer::<i32>::zeroed(&stream, one_workspace.lookback)?;
    let mut one_counters = DeviceBuffer::<i32>::zeroed(&stream, one_workspace.counters)?;
    rccl_cub::device_radix_sort_keys!(::<u32, sm_89>(
        &exec,
        &mut one_key,
        &mut one_scratch,
        &mut one_bins,
        &mut one_lookback,
        &mut one_counters,
    ))?;
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
    let workspace64 = rccl_cub::device_radix_sort_workspace!(::<u64, sm_89>(input64.len()));
    let mut scratch64 = DeviceBuffer::<u64>::zeroed(&stream, workspace64.scratch_keys)?;
    let mut bins64 = DeviceBuffer::<u64>::zeroed(&stream, workspace64.bins)?;
    let mut lookback64 = DeviceBuffer::<i32>::zeroed(&stream, workspace64.lookback)?;
    let mut counters64 = DeviceBuffer::<i32>::zeroed(&stream, workspace64.counters)?;
    rccl_cub::device_radix_sort_keys!(::<u64, sm_89>(
        &exec,
        &mut keys64,
        &mut scratch64,
        &mut bins64,
        &mut lookback64,
        &mut counters64,
    ))?;
    assert_eq!(keys64.to_host_vec(&stream)?, expected64);

    let signed_input = [0_i32, -1, i32::MIN, 7, -9, i32::MAX, -1, 0];
    let mut signed_keys = DeviceBuffer::from_host(&stream, &signed_input)?;
    let mut signed_scratch = DeviceBuffer::<i32>::zeroed(&stream, signed_input.len())?;
    let signed_workspace =
        rccl_cub::device_radix_sort_workspace!(::<i32, sm_89>(signed_input.len()));
    let mut signed_bins = DeviceBuffer::<u64>::zeroed(&stream, signed_workspace.bins)?;
    let mut signed_lookback = DeviceBuffer::<i32>::zeroed(&stream, signed_workspace.lookback)?;
    let mut signed_counters = DeviceBuffer::<i32>::zeroed(&stream, signed_workspace.counters)?;
    rccl_cub::device_radix_sort_keys_descending!(::<i32, sm_89>(
        &exec,
        &mut signed_keys,
        &mut signed_scratch,
        &mut signed_bins,
        &mut signed_lookback,
        &mut signed_counters,
    ))?;
    assert_eq!(
        signed_keys.to_host_vec(&stream)?,
        [i32::MAX, 7, 0, 0, -1, -1, -9, i32::MIN]
    );

    let signed64_input = [0_i64, -1, i64::MIN, 7, -9, i64::MAX, -1, 0];
    let mut signed64_keys = DeviceBuffer::from_host(&stream, &signed64_input)?;
    let mut signed64_scratch = DeviceBuffer::<i64>::zeroed(&stream, signed64_input.len())?;
    let signed64_workspace =
        rccl_cub::device_radix_sort_workspace!(::<i64, sm_89>(signed64_input.len()));
    let mut signed64_bins = DeviceBuffer::<u64>::zeroed(&stream, signed64_workspace.bins)?;
    let mut signed64_lookback = DeviceBuffer::<i32>::zeroed(&stream, signed64_workspace.lookback)?;
    let mut signed64_counters = DeviceBuffer::<i32>::zeroed(&stream, signed64_workspace.counters)?;
    rccl_cub::device_radix_sort_keys_descending!(::<i64, sm_89>(
        &exec,
        &mut signed64_keys,
        &mut signed64_scratch,
        &mut signed64_bins,
        &mut signed64_lookback,
        &mut signed64_counters,
    ))?;
    assert_eq!(
        signed64_keys.to_host_vec(&stream)?,
        [i64::MAX, 7, 0, 0, -1, -1, -9, i64::MIN]
    );

    // CUB treats -0 and +0 as equal radix keys while preserving their bits.
    let floating_input = [3.0_f32, -0.0, f32::NEG_INFINITY, 0.0, -7.5, f32::INFINITY];
    let mut floating_keys = DeviceBuffer::from_host(&stream, &floating_input)?;
    let mut floating_scratch = DeviceBuffer::<f32>::zeroed(&stream, floating_input.len())?;
    let floating_workspace =
        rccl_cub::device_radix_sort_workspace!(::<f32, sm_89>(floating_input.len()));
    let mut floating_bins = DeviceBuffer::<u64>::zeroed(&stream, floating_workspace.bins)?;
    let mut floating_lookback = DeviceBuffer::<i32>::zeroed(&stream, floating_workspace.lookback)?;
    let mut floating_counters = DeviceBuffer::<i32>::zeroed(&stream, floating_workspace.counters)?;
    rccl_cub::device_radix_sort_keys!(::<f32, sm_89>(
        &exec,
        &mut floating_keys,
        &mut floating_scratch,
        &mut floating_bins,
        &mut floating_lookback,
        &mut floating_counters,
    ))?;
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
    let nan_workspace32 = rccl_cub::device_radix_sort_workspace!(::<f32, sm_89>(nan_input32.len()));
    let mut nan_scratch32 = DeviceBuffer::<f32>::zeroed(&stream, nan_workspace32.scratch_keys)?;
    let mut nan_bins32 = DeviceBuffer::<u64>::zeroed(&stream, nan_workspace32.bins)?;
    let mut nan_lookback32 = DeviceBuffer::<i32>::zeroed(&stream, nan_workspace32.lookback)?;
    let mut nan_counters32 = DeviceBuffer::<i32>::zeroed(&stream, nan_workspace32.counters)?;
    for descending in [false, true] {
        let mut nan_keys32 = DeviceBuffer::from_host(&stream, &nan_input32)?;
        if descending {
            rccl_cub::device_radix_sort_keys_descending!(::<f32, sm_89>(
                &exec,
                &mut nan_keys32,
                &mut nan_scratch32,
                &mut nan_bins32,
                &mut nan_lookback32,
                &mut nan_counters32,
            ))?;
        } else {
            rccl_cub::device_radix_sort_keys!(::<f32, sm_89>(
                &exec,
                &mut nan_keys32,
                &mut nan_scratch32,
                &mut nan_bins32,
                &mut nan_lookback32,
                &mut nan_counters32,
            ))?;
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
    let mut floating64_scratch = DeviceBuffer::<f64>::zeroed(&stream, floating64_input.len())?;
    let floating64_workspace =
        rccl_cub::device_radix_sort_workspace!(::<f64, sm_89>(floating64_input.len()));
    let mut floating64_bins = DeviceBuffer::<u64>::zeroed(&stream, floating64_workspace.bins)?;
    let mut floating64_lookback =
        DeviceBuffer::<i32>::zeroed(&stream, floating64_workspace.lookback)?;
    let mut floating64_counters =
        DeviceBuffer::<i32>::zeroed(&stream, floating64_workspace.counters)?;
    rccl_cub::device_radix_sort_keys!(::<f64, sm_89>(
        &exec,
        &mut floating64_keys,
        &mut floating64_scratch,
        &mut floating64_bins,
        &mut floating64_lookback,
        &mut floating64_counters,
    ))?;
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
    let nan_workspace64 = rccl_cub::device_radix_sort_workspace!(::<f64, sm_89>(nan_input64.len()));
    let mut nan_scratch64 = DeviceBuffer::<f64>::zeroed(&stream, nan_workspace64.scratch_keys)?;
    let mut nan_bins64 = DeviceBuffer::<u64>::zeroed(&stream, nan_workspace64.bins)?;
    let mut nan_lookback64 = DeviceBuffer::<i32>::zeroed(&stream, nan_workspace64.lookback)?;
    let mut nan_counters64 = DeviceBuffer::<i32>::zeroed(&stream, nan_workspace64.counters)?;
    for descending in [false, true] {
        let mut nan_keys64 = DeviceBuffer::from_host(&stream, &nan_input64)?;
        if descending {
            rccl_cub::device_radix_sort_keys_descending!(::<f64, sm_89>(
                &exec,
                &mut nan_keys64,
                &mut nan_scratch64,
                &mut nan_bins64,
                &mut nan_lookback64,
                &mut nan_counters64,
            ))?;
        } else {
            rccl_cub::device_radix_sort_keys!(::<f64, sm_89>(
                &exec,
                &mut nan_keys64,
                &mut nan_scratch64,
                &mut nan_bins64,
                &mut nan_lookback64,
                &mut nan_counters64,
            ))?;
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
    let entropy_workspace =
        rccl_cub::device_radix_sort_workspace!(::<u32, sm_89>(entropy_input.len()));
    let mut entropy_scratch = DeviceBuffer::<u32>::zeroed(&stream, entropy_workspace.scratch_keys)?;
    let mut entropy_bins = DeviceBuffer::<u64>::zeroed(&stream, entropy_workspace.bins)?;
    let mut entropy_lookback = DeviceBuffer::<i32>::zeroed(&stream, entropy_workspace.lookback)?;
    let mut entropy_counters = DeviceBuffer::<i32>::zeroed(&stream, entropy_workspace.counters)?;
    for descending in [false, true] {
        let mut entropy_keys = DeviceBuffer::from_host(&stream, &entropy_input)?;
        if descending {
            rccl_cub::device_radix_sort_keys_descending!(::<u32, sm_89>(
                &exec,
                &mut entropy_keys,
                &mut entropy_scratch,
                &mut entropy_bins,
                &mut entropy_lookback,
                &mut entropy_counters,
            ))?;
        } else {
            rccl_cub::device_radix_sort_keys!(::<u32, sm_89>(
                &exec,
                &mut entropy_keys,
                &mut entropy_scratch,
                &mut entropy_bins,
                &mut entropy_lookback,
                &mut entropy_counters,
            ))?;
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
    let uniform_workspace =
        rccl_cub::device_radix_sort_workspace!(::<u32, sm_89>(uniform_input.len()));
    let mut uniform_scratch = DeviceBuffer::<u32>::zeroed(&stream, uniform_workspace.scratch_keys)?;
    let mut uniform_bins = DeviceBuffer::<u64>::zeroed(&stream, uniform_workspace.bins)?;
    let mut uniform_lookback = DeviceBuffer::<i32>::zeroed(&stream, uniform_workspace.lookback)?;
    let mut uniform_counters = DeviceBuffer::<i32>::zeroed(&stream, uniform_workspace.counters)?;
    for descending in [false, true] {
        let mut uniform_keys = DeviceBuffer::from_host(&stream, &uniform_input)?;
        if descending {
            rccl_cub::device_radix_sort_keys_descending!(::<u32, sm_89>(
                &exec,
                &mut uniform_keys,
                &mut uniform_scratch,
                &mut uniform_bins,
                &mut uniform_lookback,
                &mut uniform_counters,
            ))?;
        } else {
            rccl_cub::device_radix_sort_keys!(::<u32, sm_89>(
                &exec,
                &mut uniform_keys,
                &mut uniform_scratch,
                &mut uniform_bins,
                &mut uniform_lookback,
                &mut uniform_counters,
            ))?;
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
    let window_workspace =
        rccl_cub::device_radix_sort_workspace!(::<u32, sm_89>(window_input.len()));
    let mut window_bins = DeviceBuffer::<u64>::zeroed(&stream, window_workspace.bins)?;
    let mut window_lookback = DeviceBuffer::<i32>::zeroed(&stream, window_workspace.lookback)?;
    let mut window_counters = DeviceBuffer::<i32>::zeroed(&stream, window_workspace.counters)?;
    for descending in [false, true] {
        let mut first = DeviceBuffer::from_host(&stream, &window_input)?;
        let mut second = DeviceBuffer::<u32>::zeroed(&stream, window_input.len())?;
        let mut buffers = rccl_cub::DoubleBuffer::new(&mut first, &mut second)
            .expect("equal device ranges must construct");
        if descending {
            rccl_cub::device_radix_sort_keys_buffer_descending!(::<u32, sm_89>(
                &exec,
                &mut buffers,
                window.clone(),
                &mut window_bins,
                &mut window_lookback,
                &mut window_counters,
            ))?;
        } else {
            rccl_cub::device_radix_sort_keys_buffer!(::<u32, sm_89>(
                &exec,
                &mut buffers,
                window.clone(),
                &mut window_bins,
                &mut window_lookback,
                &mut window_counters,
            ))?;
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
    // spurious swap observable; zero-sized workspace proves no hidden launch
    // or workspace touch is used to make this case pass.
    let no_op_input = [9u32, 1, 7, 3];
    let mut no_op_first = DeviceBuffer::from_host(&stream, &no_op_input)?;
    let mut no_op_second = DeviceBuffer::from_host(&stream, &[99u32, 99, 99, 99])?;
    let mut no_op_buffers = rccl_cub::DoubleBuffer::new(&mut no_op_first, &mut no_op_second)
        .expect("equal device ranges must construct");
    let mut no_bins = DeviceBuffer::<u64>::zeroed(&stream, 0)?;
    let mut no_lookback = DeviceBuffer::<i32>::zeroed(&stream, 0)?;
    let mut no_counters = DeviceBuffer::<i32>::zeroed(&stream, 0)?;
    rccl_cub::device_radix_sort_keys_buffer!(::<u32, sm_89>(
        &exec,
        &mut no_op_buffers,
        12u32..12u32,
        &mut no_bins,
        &mut no_lookback,
        &mut no_counters,
    ))?;
    assert_eq!(no_op_buffers.current().to_host_vec(&stream)?, no_op_input);

    let invalid = rccl_cub::device_radix_sort_keys_buffer!(::<u32, sm_89>(
        &exec,
        &mut no_op_buffers,
        25u32..17u32,
        &mut no_bins,
        &mut no_lookback,
        &mut no_counters,
    ));
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
    rccl_cub::device_radix_sort_keys_buffer!(::<u32, sm_89>(
        &exec,
        &mut no_op_buffers,
        0u32..8u32,
        &mut no_bins,
        &mut no_lookback,
        &mut no_counters,
    ))?;
    let mut no_op_expected = no_op_input;
    no_op_expected.sort_by_key(|value| value & 0xff);
    assert_eq!(
        no_op_buffers.current().to_host_vec(&stream)?,
        no_op_expected
    );

    // One item beyond Policy800's exact 256 * 19 threshold must reject the
    // same workspace before any one-sweep launch.
    let boundary_input = (0..4_865u32).rev().collect::<Vec<_>>();
    let mut boundary_first = DeviceBuffer::from_host(&stream, &boundary_input)?;
    let mut boundary_second = DeviceBuffer::<u32>::zeroed(&stream, boundary_input.len())?;
    let mut boundary_buffers =
        rccl_cub::DoubleBuffer::new(&mut boundary_first, &mut boundary_second).unwrap();
    let undersized = rccl_cub::device_radix_sort_keys_buffer!(::<u32, sm_89>(
        &exec,
        &mut boundary_buffers,
        0u32..8u32,
        &mut no_bins,
        &mut no_lookback,
        &mut no_counters,
    ));
    assert!(matches!(
        undersized,
        Err(rccl_cub::DeviceAlgorithmError::WorkspaceTooSmall {
            role: "radix bins",
            required: 256,
            actual: 0
        })
    ));
    assert_eq!(
        boundary_buffers.current().to_host_vec(&stream)?,
        boundary_input
    );

    println!("pinned CUB DeviceRadixSort keys cases: PASS");
    Ok(())
}
