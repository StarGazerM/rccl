//! Calibration and check of `sort_columns_lexicographic`: the same K `u32`
//! key columns sorted into a stable lexicographic row permutation with each
//! forced strategy and with the model's choice. Every permutation is checked
//! against a host stable sort. Timing is GPU events around one call, median
//! of `--samples` after one warmup. The model's constants in
//! `src/lexicographic.rs` are fitted to these two forced timings.
//!
//! Usage: lexicographic_sort_strategies --items N --bits b0,b1,... [--samples S]

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{LexicographicStrategy, sort_columns_lexicographic};
use std::{cmp::Ordering, error::Error};

fn argument(name: &str) -> Option<String> {
    let arguments = std::env::args().collect::<Vec<_>>();
    arguments.windows(2).find(|pair| pair[0] == name).map(|pair| pair[1].clone())
}

fn column(rows: usize, bits: u32, seed: u64) -> Vec<u32> {
    let mask = if bits >= 32 { u32::MAX } else { (1_u32 << bits) - 1 };
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..rows)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as u32 & mask
        })
        .collect()
}

fn median(mut samples: Vec<f32>) -> f32 {
    samples.sort_by(f32::total_cmp);
    samples[samples.len() / 2]
}

/// Contract edges: empty input, a 0-bit (all-zero) column, duplicate-heavy
/// keys, and rejected arguments, for both strategies.
fn edges() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let thrust = rccl_thrust::cuda::Program::load(&context)?;
    let execution = thrust.par(&stream)?;
    let strategies = [LexicographicStrategy::RadixColumns, LexicographicStrategy::MergeTuples];
    // Empty input.
    let empty = DeviceBuffer::<u32>::zeroed(&stream, 0)?;
    let mut permutation = DeviceBuffer::<u32>::zeroed(&stream, 0)?;
    for strategy in strategies {
        sort_columns_lexicographic(&execution, &[&empty, &empty], &[3, 4], &mut permutation, Some(strategy))?;
    }
    // A 0-bit column (every value 0) before a duplicate-heavy 2-bit column:
    // the order is by the second column, ties by row.
    let rows = 5000;
    let zero = DeviceBuffer::from_host(&stream, &vec![0_u32; rows])?;
    let small: Vec<u32> = (0..rows as u32).map(|row| (row * 7 + row / 3) % 4).collect();
    let small_device = DeviceBuffer::from_host(&stream, &small)?;
    let mut expected: Vec<u32> = (0..rows as u32).collect();
    expected.sort_by_key(|&row| small[row as usize]);
    let mut permutation = DeviceBuffer::<u32>::zeroed(&stream, rows)?;
    for strategy in strategies {
        sort_columns_lexicographic(&execution, &[&zero, &small_device], &[0, 2], &mut permutation, Some(strategy))?;
        assert_eq!(permutation.to_host_vec(&stream)?, expected, "{strategy:?} 0-bit and duplicate keys");
    }
    // Rejected arguments.
    let seven = [&zero; 7];
    assert!(sort_columns_lexicographic(&execution, &seven, &[1; 7], &mut permutation, None).is_err(), "7 columns");
    assert!(sort_columns_lexicographic(&execution, &[&zero], &[33], &mut permutation, None).is_err(), "33 bits");
    assert!(sort_columns_lexicographic(&execution, &[&zero], &[1, 2], &mut permutation, None).is_err(), "bit count");
    let short = DeviceBuffer::<u32>::zeroed(&stream, rows - 1)?;
    assert!(sort_columns_lexicographic(&execution, &[&short], &[1], &mut permutation, None).is_err(), "column length");
    println!("lexicographic_sort_strategies edges=ok");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    if std::env::args().any(|argument| argument == "--edges") {
        return edges();
    }
    let rows: usize = argument("--items").ok_or("--items N")?.parse()?;
    let bits: Vec<u32> = argument("--bits").ok_or("--bits b0,b1,...")?.split(',').map(str::parse).collect::<Result<_, _>>()?;
    let samples: usize = argument("--samples").map(|s| s.parse()).transpose()?.unwrap_or(7);
    if rows == 0 || bits.iter().any(|&b| b == 0 || b > 32) {
        return Err("--items > 0, every bit width in 1..=32".into());
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // Setup, as an index build does: keep the stream-ordered allocator's
    // memory across synchronizations so scratch reuse is measured.
    unsafe {
        let mut pool = std::ptr::null_mut();
        if cuda_core::sys::cuDeviceGetDefaultMemPool(&mut pool, 0) != 0 {
            return Err("cuDeviceGetDefaultMemPool failed".into());
        }
        let mut threshold = u64::MAX;
        let attribute = cuda_core::sys::CUmemPool_attribute_enum_CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
        if cuda_core::sys::cuMemPoolSetAttribute(pool, attribute, &mut threshold as *mut u64 as *mut std::ffi::c_void) != 0 {
            return Err("cuMemPoolSetAttribute failed".into());
        }
    }
    let thrust = rccl_thrust::cuda::Program::load(&context)?;
    let execution = thrust.par(&stream)?;
    let timing = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);

    let host: Vec<Vec<u32>> = bits.iter().enumerate().map(|(j, &b)| column(rows, b, j as u64 + 1)).collect();
    let mut expected: Vec<u32> = (0..rows as u32).collect();
    expected.sort_by(|&a, &b| {
        host.iter().map(|c| c[a as usize].cmp(&c[b as usize])).find(|o| o.is_ne()).unwrap_or(Ordering::Equal)
    });
    let buffers = host.iter().map(|c| DeviceBuffer::from_host(&stream, c)).collect::<Result<Vec<_>, _>>()?;
    let columns: Vec<&DeviceBuffer<u32>> = buffers.iter().collect();
    let mut permutation = DeviceBuffer::<u32>::zeroed(&stream, rows)?;

    let mut measure = |strategy: Option<LexicographicStrategy>| -> Result<(LexicographicStrategy, f32), Box<dyn Error>> {
        let mut times = Vec::new();
        let mut used = None;
        for sample in 0..samples + 1 {
            let start = stream.record_event(timing)?;
            used = Some(sort_columns_lexicographic(&execution, &columns, &bits, &mut permutation, strategy)?);
            let end = stream.record_event(timing)?;
            end.synchronize()?;
            if sample > 0 {
                times.push(start.elapsed_ms(&end)?);
            }
        }
        assert_eq!(permutation.to_host_vec(&stream)?, expected, "{strategy:?} permutation");
        Ok((used.expect("one call"), median(times)))
    };
    let (_, radix) = measure(Some(LexicographicStrategy::RadixColumns))?;
    let (_, merge) = measure(Some(LexicographicStrategy::MergeTuples))?;
    let (chosen, auto) = measure(None)?;

    let bits_text = bits.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    println!(
        "items={rows} columns={} bits={bits_text} radix_columns_ms={radix:.4} merge_tuples_ms={merge:.4} auto={chosen:?} auto_ms={auto:.4}",
        bits.len()
    );
    Ok(())
}
