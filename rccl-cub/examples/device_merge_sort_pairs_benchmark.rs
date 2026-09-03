//! Equal-work large-size benchmark for the pinned CUB
//! `DeviceMergeSort::SortPairsCopy` case.

use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;
use std::time::Instant;

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
    let cub = rccl_cub::cuda::Program::load(&context)?;
    let launch = cub.on(&stream)?;

    let input_host = random_u32(count);
    let mut expected = input_host.clone();
    expected.sort_unstable();
    let input_keys = DeviceBuffer::from_host(&stream, &input_host)?;
    let input_values = DeviceBuffer::from_host(&stream, &input_host)?;
    let mut output_keys = DeviceBuffer::<u32>::zeroed(&stream, count)?;
    let mut output_values = DeviceBuffer::<u32>::zeroed(&stream, count)?;
    let mut workspace = rccl_cub::device_merge_sort_pairs_workspace::<u32, u32>(&launch, count)?;

    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut end_to_end_samples = Vec::with_capacity(iterations);
    {
        let mut operation = || {
            rccl_cub::device_merge_sort_pairs_copy(
                &launch,
                &input_keys,
                &input_values,
                &mut workspace,
                &mut output_keys,
                &mut output_values,
                |left: &u32, right: &u32| left.cmp(right),
            )
        };
        operation()?;
        operation()?;
        operation()?;
        stream.synchronize()?;

        for _ in 0..iterations {
            let wall_start = Instant::now();
            let start = stream.record_event(Some(0))?;
            operation()?;
            let end = stream.record_event(Some(0))?;
            end.synchronize()?;
            end_to_end_samples.push(wall_start.elapsed().as_secs_f64() * 1_000.0);
            gpu_samples.push(start.elapsed_ms(&end)? as f64);
        }
    }

    assert_eq!(output_keys.to_host_vec(&stream)?, expected);
    assert_eq!(output_values.to_host_vec(&stream)?, expected);
    assert_eq!(input_keys.to_host_vec(&stream)?, input_host);
    assert_eq!(input_values.to_host_vec(&stream)?, input_host);
    println!(
        "RCCL_CUB_MERGE_SORT_PAIRS items={count} gpu_median_ms={:.6} end_to_end_median_ms={:.6}",
        median(gpu_samples),
        median(end_to_end_samples),
    );
    Ok(())
}
