//! Large, equal-work benchmark for the currently exported CUB ordering slice.
//!
//! Timings include exactly what the Rust operation currently owns. In
//! particular, merge-sort scratch allocation is intentionally inside the
//! measurement; hiding it would mask an API/performance mismatch with CUB's
//! caller-owned temporary-storage contract.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use std::error::Error;
use std::time::Instant;

rccl_cub::cub_kernels!();

fn argument(name: &str, default: usize) -> usize {
    let arguments = std::env::args().collect::<Vec<_>>();
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse().ok())
        .unwrap_or(default)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

struct Timing {
    gpu_ms: f64,
    end_to_end_ms: f64,
}

fn timed(
    stream: &CudaStream,
    iterations: usize,
    mut operation: impl FnMut() -> Result<(), rccl_cub::DeviceAlgorithmError>,
) -> Result<Timing, Box<dyn Error>> {
    operation()?;
    operation()?;
    operation()?;
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut end_to_end_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let wall_start = Instant::now();
        let start = stream.record_event(Some(0))?;
        operation()?;
        let end = stream.record_event(Some(0))?;
        end.synchronize()?;
        end_to_end_samples.push(wall_start.elapsed().as_secs_f64() * 1_000.0);
        gpu_samples.push(start.elapsed_ms(&end)? as f64);
    }
    Ok(Timing {
        gpu_ms: median(gpu_samples),
        end_to_end_ms: median(end_to_end_samples),
    })
}

fn random_u32(count: usize) -> Vec<u32> {
    (0..count)
        .map(|index| {
            let mut value = index as u64 ^ 0x534d_4152_545f_534f_u64;
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
            value as u32
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let count = argument("--items", 1_000_003);
    let iterations = argument("--iterations", 7);
    if count == 0 || iterations == 0 {
        return Err("--items and --iterations must be non-zero".into());
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: this executable owns the sole artifact emitted above. RCCL's
    // generated algorithm and launch expressions contain no unsafe block.
    let module = unsafe { __rccl_cub_kernels::load(&context)? };
    let exec = rccl_cub::cuda::par(&stream, &module);

    let first_host = (0..count)
        .map(|index| (index * 2) as u32)
        .collect::<Vec<_>>();
    let second_host = (0..count)
        .map(|index| (index * 2 + 1) as u32)
        .collect::<Vec<_>>();
    let first = DeviceBuffer::from_host(&stream, &first_host)?;
    let second = DeviceBuffer::from_host(&stream, &second_host)?;
    let mut merged = DeviceBuffer::<u32>::zeroed(&stream, count * 2)?;
    let merge_timing = timed(&stream, iterations, || {
        rccl_cub::device_merge_keys!(::<u32>(
            &exec,
            &first,
            &second,
            &mut merged,
            |left: &u32, right: &u32| left.cmp(right),
        ))
    })?;
    let merged_host = merged.to_host_vec(&stream)?;
    assert!(
        merged_host
            .iter()
            .copied()
            .enumerate()
            .all(|(index, value)| value == index as u32)
    );

    let sort_input_host = random_u32(count);
    let mut sort_expected = sort_input_host.clone();
    sort_expected.sort_unstable();
    let sort_input = DeviceBuffer::from_host(&stream, &sort_input_host)?;
    let mut sorted = DeviceBuffer::<u32>::zeroed(&stream, count)?;
    let sort_timing = timed(&stream, iterations, || {
        rccl_cub::device_merge_sort_keys_copy!(::<u32>(
            &exec,
            &sort_input,
            &mut sorted,
            |left: &u32, right: &u32| left.cmp(right),
        ))
    })?;
    let sorted_host = sorted.to_host_vec(&stream)?;
    assert_eq!(sorted_host, sort_expected);
    assert_eq!(sort_input.to_host_vec(&stream)?, sort_input_host);

    let mut radix_inputs = (0..(iterations + 3))
        .map(|_| DeviceBuffer::from_host(&stream, &sort_input_host))
        .collect::<Result<Vec<_>, _>>()?;
    stream.synchronize()?;
    let mut radix_index = 0usize;
    let radix_timing = timed(&stream, iterations, || {
        let result = rccl_cub::device_radix_sort_keys!(::<u32>(
            &exec,
            &mut radix_inputs[radix_index],
        ));
        radix_index += 1;
        result
    })?;
    let radix_host = radix_inputs.last().unwrap().to_host_vec(&stream)?;
    assert_eq!(radix_host, sort_expected);

    println!(
        "RCCL_CUB_ORDERING items={count} \
         merge_gpu_median_ms={:.6} merge_end_to_end_median_ms={:.6} \
         sort_copy_gpu_median_ms={:.6} sort_copy_end_to_end_median_ms={:.6} \
         radix_gpu_median_ms={:.6} radix_end_to_end_median_ms={:.6}",
        merge_timing.gpu_ms,
        merge_timing.end_to_end_ms,
        sort_timing.gpu_ms,
        sort_timing.end_to_end_ms,
        radix_timing.gpu_ms,
        radix_timing.end_to_end_ms,
    );
    Ok(())
}
