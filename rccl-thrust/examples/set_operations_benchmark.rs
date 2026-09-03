//! Pinned CCCL set-operation benchmark axes: signed 8/16/32/64-bit keys,
//! 8/64-bit values, entropy 1.000/0.201, and input ratios 25/50/75.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use rccl_memory::{SingleStreamCachingMemoryResource, TemporaryValue};
use rccl_thrust::prelude::*;
use std::{error::Error, time::Instant};

fn argument(name: &str, default: &str) -> String {
    let arguments = std::env::args().collect::<Vec<_>>();
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .map_or_else(|| default.to_owned(), |pair| pair[1].clone())
}

fn usize_argument(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    Ok(argument(name, &default.to_string()).parse()?)
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

struct Timing {
    gpu_ms: f64,
    end_to_end_ms: f64,
}

type SplitBuffers<K> = (DeviceBuffer<K>, DeviceBuffer<K>, usize);

fn timed(
    stream: &CudaStream,
    iterations: usize,
    mut operation: impl FnMut() -> Result<usize, rccl_thrust::DeviceAlgorithmError>,
) -> Result<(Timing, usize), Box<dyn Error>> {
    let mut output_len = 0;
    for _ in 0..3 {
        output_len = operation()?;
    }
    let start = stream.context().new_event(Some(0))?;
    let end = stream.context().new_event(Some(0))?;
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut wall_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let wall_start = Instant::now();
        start.record(stream)?;
        output_len = operation()?;
        end.record(stream)?;
        end.synchronize()?;
        wall_samples.push(wall_start.elapsed().as_secs_f64() * 1_000.0);
        gpu_samples.push(start.elapsed_ms(&end)? as f64);
    }
    Ok((
        Timing {
            gpu_ms: median(gpu_samples),
            end_to_end_ms: median(wall_samples),
        },
        output_len,
    ))
}

trait BenchKey: TemporaryValue + Ord + 'static {
    fn from_random_bits(value: u64) -> Self;
}

macro_rules! bench_key {
    ($($ty:ty),+ $(,)?) => {$(
        impl BenchKey for $ty {
            fn from_random_bits(value: u64) -> Self {
                value as Self
            }
        }
    )+};
}

bench_key!(i8, i16, i32, i64);

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn generate<K: BenchKey>(count: usize, entropy: &str) -> Result<Vec<K>, Box<dyn Error>> {
    let rounds = match entropy {
        "1.000" => 1,
        // Five independent uniform words ANDed together give each bit
        // probability 1/32 and binary entropy 0.2006, CCCL's 0.201 row.
        "0.201" => 5,
        _ => return Err(format!("unsupported Entropy axis `{entropy}`").into()),
    };
    Ok((0..count)
        .map(|index| {
            let mut bits = u64::MAX;
            for round in 0..rounds {
                bits &= splitmix64(
                    42u64
                        .wrapping_add(index as u64)
                        .wrapping_add((round as u64) << 48),
                );
            }
            K::from_random_bits(bits)
        })
        .collect())
}

fn sorted_inputs<K: BenchKey>(
    stream: &CudaStream,
    count: usize,
    ratio: usize,
    entropy: &str,
) -> Result<SplitBuffers<K>, Box<dyn Error>> {
    let first_count = count
        .checked_mul(ratio)
        .ok_or("SizeRatio multiplication overflowed")?
        / 100;
    let mut host = generate::<K>(count, entropy)?;
    host[..first_count].sort();
    host[first_count..].sort();
    Ok((
        DeviceBuffer::from_host(stream, &host[..first_count])?,
        DeviceBuffer::from_host(stream, &host[first_count..])?,
        first_count,
    ))
}

fn print_timings(
    mode: &str,
    key: &str,
    value: &str,
    count: usize,
    entropy: &str,
    ratio: usize,
    timings: [&Timing; 4],
) {
    println!(
        "RCCL_THRUST_SET mode={mode} key={key} value={value} items={count} entropy={entropy} ratio={ratio} \
         union_gpu_median_ms={:.6} union_end_to_end_median_ms={:.6} \
         intersection_gpu_median_ms={:.6} intersection_end_to_end_median_ms={:.6} \
         difference_gpu_median_ms={:.6} difference_end_to_end_median_ms={:.6} \
         symmetric_gpu_median_ms={:.6} symmetric_end_to_end_median_ms={:.6}",
        timings[0].gpu_ms,
        timings[0].end_to_end_ms,
        timings[1].gpu_ms,
        timings[1].end_to_end_ms,
        timings[2].gpu_ms,
        timings[2].end_to_end_ms,
        timings[3].gpu_ms,
        timings[3].end_to_end_ms,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_keys<K, A>(
    execution: &rccl_thrust::cuda::Execution<'_, '_, A>,
    stream: &CudaStream,
    key_name: &str,
    count: usize,
    iterations: usize,
    entropy: &str,
    ratio: usize,
) -> Result<(), Box<dyn Error>>
where
    K: BenchKey,
    A: rccl_memory::DeviceAllocator,
{
    let (first, second, _) = sorted_inputs::<K>(stream, count, ratio, entropy)?;
    let mut output = DeviceBuffer::<K>::zeroed(stream, count)?;
    let (union, _) = timed(stream, iterations, || {
        first
            .view()
            .union(second.view())
            .write_into(output.view_mut(), execution)
    })?;
    let (intersection, _) = timed(stream, iterations, || {
        first
            .view()
            .intersection(second.view())
            .write_into(output.view_mut(), execution)
    })?;
    let (difference, _) = timed(stream, iterations, || {
        first
            .view()
            .difference(second.view())
            .write_into(output.view_mut(), execution)
    })?;
    let (symmetric, _) = timed(stream, iterations, || {
        first
            .view()
            .symmetric_difference(second.view())
            .write_into(output.view_mut(), execution)
    })?;
    print_timings(
        "keys",
        key_name,
        "none",
        count,
        entropy,
        ratio,
        [&union, &intersection, &difference, &symmetric],
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_pairs<K, V, A>(
    execution: &rccl_thrust::cuda::Execution<'_, '_, A>,
    stream: &CudaStream,
    key_name: &str,
    value_name: &str,
    count: usize,
    iterations: usize,
    entropy: &str,
    ratio: usize,
) -> Result<(), Box<dyn Error>>
where
    K: BenchKey,
    V: TemporaryValue,
    A: rccl_memory::DeviceAllocator,
{
    let (first_keys, second_keys, first_count) = sorted_inputs::<K>(stream, count, ratio, entropy)?;
    let second_count = count - first_count;
    let first_values = DeviceBuffer::<V>::zeroed(stream, first_count)?;
    let second_values = DeviceBuffer::<V>::zeroed(stream, second_count)?;
    let first = first_keys.view().with_values(first_values.view())?;
    let second = second_keys.view().with_values(second_values.view())?;
    let mut output_keys = DeviceBuffer::<K>::zeroed(stream, count)?;
    let mut output_values = DeviceBuffer::<V>::zeroed(stream, count)?;
    let mut output = output_keys
        .view_mut()
        .with_values(output_values.view_mut())?;
    let (union, _) = timed(stream, iterations, || {
        first.union(second).write_into(output.reborrow(), execution)
    })?;
    let (intersection, _) = timed(stream, iterations, || {
        first
            .intersection(second_keys.view())
            .write_into(output.reborrow(), execution)
    })?;
    let (difference, _) = timed(stream, iterations, || {
        first
            .difference(second)
            .write_into(output.reborrow(), execution)
    })?;
    let (symmetric, _) = timed(stream, iterations, || {
        first
            .symmetric_difference(second)
            .write_into(output.reborrow(), execution)
    })?;
    print_timings(
        "pairs",
        key_name,
        value_name,
        count,
        entropy,
        ratio,
        [&union, &intersection, &difference, &symmetric],
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let count = usize_argument("--items", 1 << 24)?;
    let iterations = usize_argument("--iterations", 7)?;
    let ratio = usize_argument("--ratio", 50)?;
    let entropy = argument("--entropy", "1.000");
    let mode = argument("--mode", "keys");
    let key = argument("--key", "i32");
    let value = argument("--value", "i64");
    if count < 2 || iterations == 0 || !matches!(ratio, 25 | 50 | 75) {
        return Err("items, iterations, or SizeRatio is outside the pinned benchmark axes".into());
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let thrust = rccl_thrust::cuda::Program::load(&context)?;
    let cache = SingleStreamCachingMemoryResource::new(&stream)?;
    let execution = thrust.par_in(&stream, &cache)?;

    match (mode.as_str(), key.as_str(), value.as_str()) {
        ("keys", "i8", _) => run_keys::<i8, _>(
            &execution, &stream, "i8", count, iterations, &entropy, ratio,
        ),
        ("keys", "i16", _) => run_keys::<i16, _>(
            &execution, &stream, "i16", count, iterations, &entropy, ratio,
        ),
        ("keys", "i32", _) => run_keys::<i32, _>(
            &execution, &stream, "i32", count, iterations, &entropy, ratio,
        ),
        ("keys", "i64", _) => run_keys::<i64, _>(
            &execution, &stream, "i64", count, iterations, &entropy, ratio,
        ),
        ("pairs", "i8", "i8") => run_pairs::<i8, i8, _>(
            &execution, &stream, "i8", "i8", count, iterations, &entropy, ratio,
        ),
        ("pairs", "i8", "i64") => run_pairs::<i8, i64, _>(
            &execution, &stream, "i8", "i64", count, iterations, &entropy, ratio,
        ),
        ("pairs", "i16", "i8") => run_pairs::<i16, i8, _>(
            &execution, &stream, "i16", "i8", count, iterations, &entropy, ratio,
        ),
        ("pairs", "i16", "i64") => run_pairs::<i16, i64, _>(
            &execution, &stream, "i16", "i64", count, iterations, &entropy, ratio,
        ),
        ("pairs", "i32", "i8") => run_pairs::<i32, i8, _>(
            &execution, &stream, "i32", "i8", count, iterations, &entropy, ratio,
        ),
        ("pairs", "i32", "i64") => run_pairs::<i32, i64, _>(
            &execution, &stream, "i32", "i64", count, iterations, &entropy, ratio,
        ),
        ("pairs", "i64", "i8") => run_pairs::<i64, i8, _>(
            &execution, &stream, "i64", "i8", count, iterations, &entropy, ratio,
        ),
        ("pairs", "i64", "i64") => run_pairs::<i64, i64, _>(
            &execution, &stream, "i64", "i64", count, iterations, &entropy, ratio,
        ),
        _ => Err(format!(
            "unsupported pinned set benchmark row mode={mode} key={key} value={value}"
        )
        .into()),
    }
}
