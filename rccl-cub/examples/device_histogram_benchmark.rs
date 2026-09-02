//! Equal-work benchmark for pinned CUB DeviceHistogram entry points.
//!
//! The input distribution, bin layout, temporary-storage lifetime, warmups,
//! and event timing are shared with `benchmarks/histogram-reference.cu`.
//! `RCCL_HISTOGRAM_BATCHED=1` switches both programs to one event pair around
//! all iterations, reporting the per-call average used to isolate device
//! throughput from synchronized submission latency.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use rccl_cub::{DeviceHistogram, DeviceHistogramLevelsView, HistogramRegion};
use std::error::Error;

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

fn timed(
    stream: &CudaStream,
    iterations: usize,
    mut operation: impl FnMut() -> Result<(), rccl_cub::DeviceAlgorithmError>,
) -> Result<f64, Box<dyn Error>> {
    for _ in 0..3 {
        operation()?;
    }
    stream.synchronize()?;

    // Match the pinned CUB reference's event lifetime exactly. Reusing one
    // timing pair avoids measuring a different submission/frequency regime
    // caused by creating and destroying events between every launch pair.
    let start = stream.context().new_event(Some(0))?;
    let end = stream.context().new_event(Some(0))?;
    if std::env::var_os("RCCL_HISTOGRAM_BATCHED").is_some() {
        start.record(stream)?;
        for _ in 0..iterations {
            operation()?;
        }
        end.record(stream)?;
        end.synchronize()?;
        return Ok(start.elapsed_ms(&end)? as f64 / iterations as f64);
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        start.record(stream)?;
        operation()?;
        end.record(stream)?;
        end.synchronize()?;
        samples.push(start.elapsed_ms(&end)? as f64);
    }
    Ok(median(samples))
}

fn random_bins(count: usize, bins: usize) -> Vec<i32> {
    (0..count)
        .map(|index| {
            let mut value = index as u64 ^ 0x534d_4152_545f_534f_u64;
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
            (value % bins as u64) as i32
        })
        .collect()
}

fn expected_channel(samples: &[i32], bins: usize, channels: usize, channel: usize) -> Vec<i32> {
    let mut expected = vec![0; bins];
    for sample in samples.iter().skip(channel).step_by(channels) {
        expected[*sample as usize] += 1;
    }
    expected
}

fn main() -> Result<(), Box<dyn Error>> {
    let items = argument("--items", 16_777_216);
    let bins = argument("--bins", 128);
    let iterations = argument("--iterations", 15);
    if items == 0 || bins == 0 || bins >= i32::MAX as usize || iterations == 0 {
        return Err("--items, --bins, and --iterations must be valid non-zero CUB extents".into());
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let cub = rccl_cub::cub_program!(&context; histogram)?;
    let launch = cub.on(&stream)?;

    let single_host = random_bins(items, bins);
    let multi_host = random_bins(
        items
            .checked_mul(4)
            .ok_or("multi-channel benchmark extent overflowed")?,
        bins,
    );
    let single_expected = expected_channel(&single_host, bins, 1, 0);
    let multi_expected = [
        expected_channel(&multi_host, bins, 4, 0),
        expected_channel(&multi_host, bins, 4, 1),
        expected_channel(&multi_host, bins, 4, 2),
    ];
    let single_samples = DeviceBuffer::from_host(&stream, &single_host)?;
    let multi_samples = DeviceBuffer::from_host(&stream, &multi_host)?;
    let level_values = (0..=bins).map(|value| value as i32).collect::<Vec<_>>();
    let level_storage = DeviceBuffer::from_host(&stream, &level_values)?;

    let mut single_even = DeviceHistogram::<i32, 1>::new(&stream, [bins])?;
    let mut single_even_workspace = rccl_cub::device_histogram_even_workspace!(::<
        i32,
        i32,
        i32,
        1,
        1,
    >(
        &launch,
        &single_samples,
        HistogramRegion::All,
        &single_even,
    ))?;
    let single_even_ms = timed(&stream, iterations, || {
        rccl_cub::device_histogram_even!(::<i32, i32, i32, 1, 1>(
            &launch,
            &single_samples,
            HistogramRegion::All,
            &mut single_even,
            [0],
            [bins as i32],
            &mut single_even_workspace,
        ))
    })?;
    assert_eq!(single_even.to_host_channels(&stream)?[0], single_expected);

    // The pinned CCCL reference passes one device-level allocation to every
    // range-histogram channel. Preserve that pointer aliasing, and therefore
    // its read-only-cache behavior, through a lifetime-checked borrowed view.
    let single_levels =
        DeviceHistogramLevelsView::<i32, 1>::from_ranges(&level_storage, [0..level_values.len()])?;
    let mut single_range = DeviceHistogram::<i32, 1>::new(&stream, [bins])?;
    let mut single_range_workspace = rccl_cub::device_histogram_range_workspace!(::<
        i32,
        i32,
        i32,
        1,
        1,
    >(
        &launch,
        &single_samples,
        HistogramRegion::All,
        &single_range,
    ))?;
    let single_range_ms = timed(&stream, iterations, || {
        rccl_cub::device_histogram_range!(::<i32, i32, i32, 1, 1>(
            &launch,
            &single_samples,
            HistogramRegion::All,
            &mut single_range,
            &single_levels,
            &mut single_range_workspace,
        ))
    })?;
    assert_eq!(single_range.to_host_channels(&stream)?[0], single_expected);

    let mut multi_even = DeviceHistogram::<i32, 3>::new(&stream, [bins; 3])?;
    let mut multi_even_workspace = rccl_cub::device_histogram_even_workspace!(::<
        i32,
        i32,
        i32,
        4,
        3,
    >(
        &launch,
        &multi_samples,
        HistogramRegion::All,
        &multi_even,
    ))?;
    let multi_even_ms = timed(&stream, iterations, || {
        rccl_cub::device_histogram_even!(::<i32, i32, i32, 4, 3>(
            &launch,
            &multi_samples,
            HistogramRegion::All,
            &mut multi_even,
            [0; 3],
            [bins as i32; 3],
            &mut multi_even_workspace,
        ))
    })?;
    assert_eq!(multi_even.to_host_channels(&stream)?, multi_expected);

    let multi_levels = DeviceHistogramLevelsView::<i32, 3>::from_ranges(
        &level_storage,
        core::array::from_fn(|_| 0..level_values.len()),
    )?;
    let mut multi_range = DeviceHistogram::<i32, 3>::new(&stream, [bins; 3])?;
    let mut multi_range_workspace = rccl_cub::device_histogram_range_workspace!(::<
        i32,
        i32,
        i32,
        4,
        3,
    >(
        &launch,
        &multi_samples,
        HistogramRegion::All,
        &multi_range,
    ))?;
    let multi_range_ms = timed(&stream, iterations, || {
        rccl_cub::device_histogram_range!(::<i32, i32, i32, 4, 3>(
            &launch,
            &multi_samples,
            HistogramRegion::All,
            &mut multi_range,
            &multi_levels,
            &mut multi_range_workspace,
        ))
    })?;
    assert_eq!(multi_range.to_host_channels(&stream)?, multi_expected);

    println!(
        "RCCL_CUB_HISTOGRAM items={items} bins={bins} even_gpu_median_ms={single_even_ms:.9} range_gpu_median_ms={single_range_ms:.9} multi_even_gpu_median_ms={multi_even_ms:.9} multi_range_gpu_median_ms={multi_range_ms:.9}"
    );
    Ok(())
}
