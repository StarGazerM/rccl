//! Large benchmark for the Thrust ordering facade. Each operation lowers
//! through CUB and includes Thrust's synchronous completion contract.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use rccl_thrust::prelude::*;
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

fn timed<State>(
    stream: &CudaStream,
    warmups: usize,
    iterations: usize,
    state: &mut State,
    mut prepare: impl FnMut(&mut State) -> Result<(), rccl_thrust::DeviceAlgorithmError>,
    mut operation: impl FnMut(&mut State) -> Result<(), rccl_thrust::DeviceAlgorithmError>,
) -> Result<Timing, Box<dyn Error>> {
    for _ in 0..warmups {
        prepare(state)?;
        operation(state)?;
    }
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut end_to_end_samples = Vec::with_capacity(iterations);
    // Reuse the event pair exactly as the pinned CCCL benchmarks do. Creating
    // and destroying events per sample perturbs a following synchronous
    // temporary allocation even though the kernel graph is unchanged.
    let start = stream.context().new_event(Some(0))?;
    let end = stream.context().new_event(Some(0))?;
    for _ in 0..iterations {
        // Pinned CCCL restores sort input and synchronizes before recording
        // the start event. Keeping preparation here prevents a result-only
        // benchmark from hiding allocation-pressure or reset-work mismatch.
        prepare(state)?;
        let wall_start = Instant::now();
        start.record(stream)?;
        operation(state)?;
        end.record(stream)?;
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
    // cuda-oxide's `new_stream` is explicitly non-blocking, whereas the
    // pinned reference uses cudaStreamCreate's blocking stream. With no other
    // streams active, the legacy default stream supplies the same blocking
    // ordering contract for this benchmark.
    let stream = context.default_stream();
    let thrust = rccl_thrust::thrust_program!(&context)?;
    let exec = thrust.par(&stream)?;

    // The pinned parity suite has separate merge and ordering executables.
    // Scope their allocations independently so one reference's live buffers
    // cannot perturb the other's temporary-allocation measurements.
    let (merge_timing, pair_merge_timing) = {
        let first_host = (0..count)
            .map(|index| (index * 2) as u32)
            .collect::<Vec<_>>();
        let second_host = (0..count)
            .map(|index| (index * 2 + 1) as u32)
            .collect::<Vec<_>>();
        let first = DeviceBuffer::from_host(&stream, &first_host)?;
        let second = DeviceBuffer::from_host(&stream, &second_host)?;
        let mut merged = DeviceBuffer::<u32>::zeroed(&stream, count * 2)?;
        let merge_timing = timed(
            &stream,
            0,
            iterations,
            &mut (),
            |_| Ok(()),
            |_| exec.merge_into(&first, &second, &mut merged),
        )?;
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
        let first_pairs = first.with_values(&first_values)?;
        let second_pairs = second.with_values(&second_values)?;
        let mut merged_pairs = merged_pair_keys.with_values_mut(&mut merged_pair_values)?;
        let pair_merge_timing = timed(
            &stream,
            0,
            iterations,
            &mut merged_pairs,
            |_| Ok(()),
            |output| exec.merge_into(first_pairs, second_pairs, output),
        )?;
        let (_, merged_pair_values) = merged_pairs.into_parts();
        assert!(
            merged_pair_values
                .to_host_vec(&stream)?
                .into_iter()
                .enumerate()
                .all(|(index, value)| value == index as u64)
        );
        (merge_timing, pair_merge_timing)
    };

    let (sort_timing, pair_sort_timing) = {
        let sort_input_host = random_u32(count);
        let mut sort_expected = sort_input_host.clone();
        sort_expected.sort_unstable();
        let sort_source = DeviceBuffer::from_host(&stream, &sort_input_host)?;
        let mut sort_input = DeviceBuffer::from_host(&stream, &sort_input_host)?;
        let sort_timing = timed(
            &stream,
            3,
            iterations,
            &mut sort_input,
            |target| {
                rccl_thrust::memory::copy_device_range::<u32, _, _>(&sort_source, target, &stream)
                    .map_err(rccl_thrust::DeviceAlgorithmError::Memory)?;
                stream
                    .synchronize()
                    .map_err(rccl_thrust::DeviceAlgorithmError::Driver)
            },
            |target| target.sort_unstable(&exec),
        )?;
        let sorted_host = sort_input.to_host_vec(&stream)?;
        assert_eq!(sorted_host, sort_expected);

        let pair_sort_source_keys = DeviceBuffer::from_host(&stream, &sort_input_host)?;
        let pair_sort_source_values = DeviceBuffer::from_host(&stream, &sort_input_host)?;
        let mut pair_sort_keys = DeviceBuffer::from_host(&stream, &sort_input_host)?;
        let mut pair_sort_values = DeviceBuffer::from_host(&stream, &sort_input_host)?;
        let mut pair_sort = pair_sort_keys.with_values_mut(&mut pair_sort_values)?;
        let pair_sort_timing = timed(
            &stream,
            3,
            iterations,
            &mut pair_sort,
            |pairs| {
                let (keys, values) = pairs.parts_mut();
                rccl_thrust::memory::copy_device_range::<u32, _, _>(
                    &pair_sort_source_keys,
                    keys,
                    &stream,
                )
                .map_err(rccl_thrust::DeviceAlgorithmError::Memory)?;
                rccl_thrust::memory::copy_device_range::<u32, _, _>(
                    &pair_sort_source_values,
                    values,
                    &stream,
                )
                .map_err(rccl_thrust::DeviceAlgorithmError::Memory)?;
                stream
                    .synchronize()
                    .map_err(rccl_thrust::DeviceAlgorithmError::Driver)
            },
            |pairs| pairs.sort_unstable(&exec),
        )?;
        let (pair_sort_keys, pair_sort_values) = pair_sort.into_parts();
        assert_eq!(pair_sort_keys.to_host_vec(&stream)?, sort_expected);
        assert_eq!(pair_sort_values.to_host_vec(&stream)?, sort_expected);
        (sort_timing, pair_sort_timing)
    };

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
