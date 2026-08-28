//! Large benchmark for the Thrust ordering facade. Each operation lowers
//! through CUB and includes Thrust's synchronous completion contract.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
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

struct Timing {
    gpu_ms: f64,
    end_to_end_ms: f64,
}

fn timed(
    stream: &CudaStream,
    iterations: usize,
    mut operation: impl FnMut() -> Result<(), rccl_thrust::DeviceAlgorithmError>,
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
    let thrust = rccl_thrust::thrust_program!(&context)?;
    let exec = thrust.par(&stream)?;

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
        rccl_thrust::merge!(::<u32>(
            &exec,
            &first,
            &second,
            &mut merged,
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

    let first_values_host = (0..count)
        .map(|index| (index * 2) as u64)
        .collect::<Vec<_>>();
    let second_values_host = (0..count)
        .map(|index| (index * 2 + 1) as u64)
        .collect::<Vec<_>>();
    let first_values = DeviceBuffer::from_host(&stream, &first_values_host)?;
    let second_values = DeviceBuffer::from_host(&stream, &second_values_host)?;
    let mut merged_pair_keys = DeviceBuffer::<u32>::zeroed(&stream, count * 2)?;
    let mut merged_pair_values = DeviceBuffer::<u64>::zeroed(&stream, count * 2)?;
    let pair_merge_timing = timed(&stream, iterations, || {
        rccl_thrust::merge_by_key!(::<u32, u64>(
            &exec,
            &first,
            &first_values,
            &second,
            &second_values,
            &mut merged_pair_keys,
            &mut merged_pair_values,
        ))
    })?;
    assert!(
        merged_pair_values
            .to_host_vec(&stream)?
            .into_iter()
            .enumerate()
            .all(|(index, value)| value == index as u64)
    );

    let sort_input_host = random_u32(count);
    let mut sort_expected = sort_input_host.clone();
    sort_expected.sort_unstable();
    let mut sort_inputs = (0..(iterations + 3))
        .map(|_| DeviceBuffer::from_host(&stream, &sort_input_host))
        .collect::<Result<Vec<_>, _>>()?;
    stream.synchronize()?;
    let mut sort_index = 0usize;
    let sort_timing = timed(&stream, iterations, || {
        let result = rccl_thrust::sort!(::<u32>(
            &exec,
            &mut sort_inputs[sort_index],
        ));
        sort_index += 1;
        result
    })?;
    let sorted_host = sort_inputs.last().unwrap().to_host_vec(&stream)?;
    assert_eq!(sorted_host, sort_expected);

    let mut pair_sort_keys = (0..(iterations + 3))
        .map(|_| DeviceBuffer::from_host(&stream, &sort_input_host))
        .collect::<Result<Vec<_>, _>>()?;
    let mut pair_sort_values = (0..(iterations + 3))
        .map(|_| DeviceBuffer::from_host(&stream, &sort_input_host))
        .collect::<Result<Vec<_>, _>>()?;
    stream.synchronize()?;
    let mut pair_sort_index = 0usize;
    let pair_sort_timing = timed(&stream, iterations, || {
        let result = rccl_thrust::sort_by_key!(::<u32, u32>(
            &exec,
            &mut pair_sort_keys[pair_sort_index],
            &mut pair_sort_values[pair_sort_index],
        ));
        pair_sort_index += 1;
        result
    })?;
    assert_eq!(
        pair_sort_keys.last().unwrap().to_host_vec(&stream)?,
        sort_expected
    );
    assert_eq!(
        pair_sort_values.last().unwrap().to_host_vec(&stream)?,
        sort_expected
    );

    println!(
        "RCCL_THRUST_ORDERING items={count} \
         merge_gpu_median_ms={:.6} merge_end_to_end_median_ms={:.6} \
         pair_merge_gpu_median_ms={:.6} pair_merge_end_to_end_median_ms={:.6} \
         sort_gpu_median_ms={:.6} sort_end_to_end_median_ms={:.6} \
         sort_by_key_gpu_median_ms={:.6} sort_by_key_end_to_end_median_ms={:.6}",
        merge_timing.gpu_ms,
        merge_timing.end_to_end_ms,
        pair_merge_timing.gpu_ms,
        pair_merge_timing.end_to_end_ms,
        sort_timing.gpu_ms,
        sort_timing.end_to_end_ms,
        pair_sort_timing.gpu_ms,
        pair_sort_timing.end_to_end_ms,
    );
    Ok(())
}
