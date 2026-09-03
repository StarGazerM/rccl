//! Large, equal-work benchmark for the currently exported CUB ordering slice.
//!
//! All algorithm workspace is allocated before measurement, matching CUB's
//! caller-owned temporary-storage contract.

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

fn timed_batched(
    stream: &CudaStream,
    iterations: usize,
    batch: usize,
    mut operation: impl FnMut() -> Result<(), rccl_cub::DeviceAlgorithmError>,
) -> Result<Timing, Box<dyn Error>> {
    let mut timing = timed(stream, iterations, || {
        for _ in 0..batch {
            operation()?;
        }
        Ok(())
    })?;
    timing.gpu_ms /= batch as f64;
    timing.end_to_end_ms /= batch as f64;
    Ok(timing)
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

    let first_host = (0..count)
        .map(|index| (index * 2) as u32)
        .collect::<Vec<_>>();
    let second_host = (0..count)
        .map(|index| (index * 2 + 1) as u32)
        .collect::<Vec<_>>();
    let first = DeviceBuffer::from_host(&stream, &first_host)?;
    let second = DeviceBuffer::from_host(&stream, &second_host)?;
    let mut merged = DeviceBuffer::<u32>::zeroed(&stream, count * 2)?;
    let mut merge_workspace =
        rccl_cub::device_merge_keys_workspace::<u32>(&launch, first.len(), second.len())?;
    let merge_timing = timed(&stream, iterations, || {
        rccl_cub::device_merge_keys(
            &launch,
            &first,
            &second,
            &mut merge_workspace,
            &mut merged,
            |left: &u32, right: &u32| left.cmp(right),
        )
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
    let mut pair_workspace =
        rccl_cub::device_merge_pairs_workspace::<u32, u64>(&launch, first.len(), second.len())?;
    let pair_merge_timing = timed(&stream, iterations, || {
        rccl_cub::device_merge_pairs(
            &launch,
            &first,
            &first_values,
            &second,
            &second_values,
            &mut pair_workspace,
            &mut merged_pair_keys,
            &mut merged_pair_values,
            |left: &u32, right: &u32| left.cmp(right),
        )
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
    let mut window_expected = sort_input_host.clone();
    window_expected.sort_by_key(|key| key & 0x00ff_ffff);
    let sort_input = DeviceBuffer::from_host(&stream, &sort_input_host)?;
    let mut sorted = DeviceBuffer::<u32>::zeroed(&stream, count)?;
    let mut sort_workspace = rccl_cub::device_merge_sort_keys_workspace::<u32>(&launch, count)?;
    let sort_timing = timed(&stream, iterations, || {
        rccl_cub::device_merge_sort_keys_copy(
            &launch,
            &sort_input,
            &mut sort_workspace,
            &mut sorted,
            |left: &u32, right: &u32| left.cmp(right),
        )
    })?;
    let sorted_host = sorted.to_host_vec(&stream)?;
    assert_eq!(sorted_host, sort_expected);
    assert_eq!(sort_input.to_host_vec(&stream)?, sort_input_host);

    let mut radix_inputs = (0..(iterations + 3))
        .map(|_| DeviceBuffer::from_host(&stream, &sort_input_host))
        .collect::<Result<Vec<_>, _>>()?;
    let mut radix_workspace = rccl_cub::device_radix_sort_workspace::<u32>(&launch, count)?;
    stream.synchronize()?;
    let mut radix_index = 0usize;
    let radix_timing = timed(&stream, iterations, || {
        let result = rccl_cub::device_radix_sort_keys(
            &launch,
            &mut radix_inputs[radix_index],
            &mut radix_workspace,
        );
        radix_index += 1;
        result
    })?;
    let radix_host = radix_inputs.last().unwrap().to_host_vec(&stream)?;
    assert_eq!(radix_host, sort_expected);

    let mut radix_pair_keys = (0..(iterations + 3))
        .map(|_| DeviceBuffer::from_host(&stream, &sort_input_host))
        .collect::<Result<Vec<_>, _>>()?;
    let mut radix_pair_values = (0..(iterations + 3))
        .map(|_| DeviceBuffer::from_host(&stream, &sort_input_host))
        .collect::<Result<Vec<_>, _>>()?;
    let mut radix_pair_workspace =
        rccl_cub::device_radix_sort_pairs_workspace::<u32, u32>(&launch, count)?;
    stream.synchronize()?;
    let mut radix_pair_index = 0usize;
    let radix_pair_timing = timed(&stream, iterations, || {
        let result = rccl_cub::device_radix_sort_pairs(
            &launch,
            &mut radix_pair_keys[radix_pair_index],
            &mut radix_pair_values[radix_pair_index],
            &mut radix_pair_workspace,
        );
        radix_pair_index += 1;
        result
    })?;
    assert_eq!(
        radix_pair_keys.last().unwrap().to_host_vec(&stream)?,
        sort_expected
    );
    assert_eq!(
        radix_pair_values.last().unwrap().to_host_vec(&stream)?,
        sort_expected
    );

    let mut window_keys_0 = DeviceBuffer::from_host(&stream, &sort_input_host)?;
    let mut window_keys_1 = DeviceBuffer::<u32>::zeroed(&stream, count)?;
    let mut window_buffers = rccl_cub::DoubleBuffer::new(&mut window_keys_0, &mut window_keys_1)
        .expect("equal benchmark key buffers must construct");
    let mut radix_window_workspace =
        rccl_cub::device_radix_sort_keys_buffer_workspace::<u32>(&launch, count)?;
    stream.synchronize()?;
    let radix_window_timing = timed(&stream, iterations, || {
        rccl_cub::device_radix_sort_keys_buffer(
            &launch,
            &mut window_buffers,
            0u32..24u32,
            &mut radix_window_workspace,
        )
    })?;
    assert_eq!(
        window_buffers.current().to_host_vec(&stream)?,
        window_expected
    );

    let mut window_pair_keys_0 = DeviceBuffer::from_host(&stream, &sort_input_host)?;
    let mut window_pair_values_0 = DeviceBuffer::from_host(&stream, &sort_input_host)?;
    let mut window_pair_keys_1 = DeviceBuffer::<u32>::zeroed(&stream, count)?;
    let mut window_pair_values_1 = DeviceBuffer::<u32>::zeroed(&stream, count)?;
    let mut window_pair_buffers = rccl_cub::DoubleBuffer::paired(
        (&mut window_pair_keys_0, &mut window_pair_values_0),
        (&mut window_pair_keys_1, &mut window_pair_values_1),
    )
    .expect("equal benchmark pair buffers must construct");
    let mut radix_window_pair_workspace =
        rccl_cub::device_radix_sort_pairs_buffer_workspace::<u32, u32>(&launch, count)?;
    stream.synchronize()?;
    let radix_window_pair_timing = timed(&stream, iterations, || {
        rccl_cub::device_radix_sort_pairs_buffer(
            &launch,
            &mut window_pair_buffers,
            0u32..24u32,
            &mut radix_window_pair_workspace,
        )
    })?;
    let (window_pair_keys, window_pair_values) = window_pair_buffers.current();
    assert_eq!(window_pair_keys.to_host_vec(&stream)?, window_expected);
    assert_eq!(window_pair_values.to_host_vec(&stream)?, window_expected);

    // Policy800's exact u32 single-tile extent. Batching one-kernel calls per
    // event makes the sub-20-microsecond path measurable without changing the
    // launch work being compared with pinned CUB.
    const SINGLE_TILE_COUNT: usize = 256 * 19;
    const SINGLE_TILE_BATCH: usize = 512;
    let single_tile_input = random_u32(SINGLE_TILE_COUNT);
    let mut single_tile_expected = single_tile_input.clone();
    single_tile_expected.sort_unstable();
    let mut single_keys_0 = DeviceBuffer::from_host(&stream, &single_tile_input)?;
    let mut single_keys_1 = DeviceBuffer::<u32>::zeroed(&stream, SINGLE_TILE_COUNT)?;
    let mut single_keys = rccl_cub::DoubleBuffer::new(&mut single_keys_0, &mut single_keys_1)
        .expect("equal single-tile key buffers must construct");
    let mut single_keys_workspace =
        rccl_cub::device_radix_sort_keys_buffer_workspace::<u32>(&launch, SINGLE_TILE_COUNT)?;
    stream.synchronize()?;
    let radix_single_tile_timing = timed_batched(&stream, iterations, SINGLE_TILE_BATCH, || {
        rccl_cub::device_radix_sort_keys_buffer(
            &launch,
            &mut single_keys,
            0u32..32u32,
            &mut single_keys_workspace,
        )
    })?;
    assert_eq!(
        single_keys.current().to_host_vec(&stream)?,
        single_tile_expected
    );

    let mut single_pair_keys_0 = DeviceBuffer::from_host(&stream, &single_tile_input)?;
    let mut single_pair_keys_1 = DeviceBuffer::<u32>::zeroed(&stream, SINGLE_TILE_COUNT)?;
    let mut single_pair_values_0 = DeviceBuffer::from_host(&stream, &single_tile_input)?;
    let mut single_pair_values_1 = DeviceBuffer::<u32>::zeroed(&stream, SINGLE_TILE_COUNT)?;
    let mut single_pairs = rccl_cub::DoubleBuffer::paired(
        (&mut single_pair_keys_0, &mut single_pair_values_0),
        (&mut single_pair_keys_1, &mut single_pair_values_1),
    )
    .expect("equal single-tile pair buffers must construct");
    let mut single_pairs_workspace =
        rccl_cub::device_radix_sort_pairs_buffer_workspace::<u32, u32>(&launch, SINGLE_TILE_COUNT)?;
    stream.synchronize()?;
    let radix_single_tile_pair_timing =
        timed_batched(&stream, iterations, SINGLE_TILE_BATCH, || {
            rccl_cub::device_radix_sort_pairs_buffer(
                &launch,
                &mut single_pairs,
                0u32..32u32,
                &mut single_pairs_workspace,
            )
        })?;
    let (single_pair_keys, single_pair_values) = single_pairs.current();
    assert_eq!(single_pair_keys.to_host_vec(&stream)?, single_tile_expected);
    assert_eq!(
        single_pair_values.to_host_vec(&stream)?,
        single_tile_expected
    );

    println!(
        "RCCL_CUB_ORDERING items={count} \
         merge_gpu_median_ms={:.6} merge_end_to_end_median_ms={:.6} \
         pair_merge_gpu_median_ms={:.6} pair_merge_end_to_end_median_ms={:.6} \
         sort_copy_gpu_median_ms={:.6} sort_copy_end_to_end_median_ms={:.6} \
         radix_gpu_median_ms={:.6} radix_end_to_end_median_ms={:.6} \
         radix_pairs_gpu_median_ms={:.6} radix_pairs_end_to_end_median_ms={:.6} \
         radix_window_gpu_median_ms={:.6} radix_window_end_to_end_median_ms={:.6} \
         radix_window_pairs_gpu_median_ms={:.6} radix_window_pairs_end_to_end_median_ms={:.6} \
         radix_single_tile_gpu_median_ms={:.9} radix_single_tile_end_to_end_median_ms={:.9} \
         radix_single_tile_pairs_gpu_median_ms={:.9} radix_single_tile_pairs_end_to_end_median_ms={:.9}",
        merge_timing.gpu_ms,
        merge_timing.end_to_end_ms,
        pair_merge_timing.gpu_ms,
        pair_merge_timing.end_to_end_ms,
        sort_timing.gpu_ms,
        sort_timing.end_to_end_ms,
        radix_timing.gpu_ms,
        radix_timing.end_to_end_ms,
        radix_pair_timing.gpu_ms,
        radix_pair_timing.end_to_end_ms,
        radix_window_timing.gpu_ms,
        radix_window_timing.end_to_end_ms,
        radix_window_pair_timing.gpu_ms,
        radix_window_pair_timing.end_to_end_ms,
        radix_single_tile_timing.gpu_ms,
        radix_single_tile_timing.end_to_end_ms,
        radix_single_tile_pair_timing.gpu_ms,
        radix_single_tile_pair_timing.end_to_end_ms,
    );
    Ok(())
}
