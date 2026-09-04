//! Equal-work benchmark for pinned CUB `DeviceReduce::Sum`.
//!
//! The input distribution, temporary-storage lifetime, warmups, and CUDA-event
//! timing are shared with `benchmarks/reduce-reference.cu`.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use rccl_cub::DeviceAlgorithmError;
use std::error::Error;

const WARMUPS: usize = 20;
const SAMPLES: usize = 21;
const LAUNCHES: usize = 50;

fn argument(name: &str, default: usize) -> usize {
    let arguments = std::env::args().collect::<Vec<_>>();
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse().ok())
        .unwrap_or(default)
}

fn timed(
    stream: &CudaStream,
    mut operation: impl FnMut() -> Result<(), DeviceAlgorithmError>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    for _ in 0..WARMUPS {
        operation()?;
    }
    stream.synchronize()?;

    let start = stream.context().new_event(Some(0))?;
    let end = stream.context().new_event(Some(0))?;
    let mut timings = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        start.record(stream)?;
        for _ in 0..LAUNCHES {
            operation()?;
        }
        end.record(stream)?;
        end.synchronize()?;
        timings.push(start.elapsed_ms(&end)? / LAUNCHES as f32);
    }
    timings.sort_by(f32::total_cmp);
    Ok(timings)
}

fn benchmark<T>(
    launch: &rccl_cub::cuda::Launch<'_, '_>,
    stream: &CudaStream,
    items: usize,
    label: &str,
) -> Result<(), Box<dyn Error>>
where
    T: rccl_cub::ReduceSumValue
        + rccl_memory::TemporaryValue
        + One
        + PartialEq
        + std::fmt::Debug
        + 'static,
{
    let input = DeviceBuffer::from_host(stream, &vec![T::one(); items])?;
    let mut output = rccl_memory::DeviceScalar::new_in(
        rccl_memory::BorrowedCudaMemoryResource::new(stream.context())?,
        stream,
        T::default(),
    )?;
    let mut workspace = rccl_cub::device_reduce_sum_workspace::<T>(launch, input.len())?;
    let allocation_bytes = workspace.allocation_bytes();
    let timings = timed(stream, || {
        rccl_cub::device_reduce_sum_in(launch, &input, output.as_buffer_mut(), &mut workspace)
    })?;
    let result = output.get()?;
    assert_eq!(result, expected::<T>(items));

    println!(
        "RCCL_REDUCE algorithm=device_reduce_sum type={label} items={items} temp_bytes={allocation_bytes} best_ms={:.9} median_ms={:.9}",
        timings[0],
        timings[timings.len() / 2],
    );
    Ok(())
}

trait One: Sized {
    fn one() -> Self;

    fn from_items(items: usize) -> Self;
}

impl One for u32 {
    fn one() -> Self {
        1
    }

    fn from_items(items: usize) -> Self {
        items.try_into().expect("benchmark size fits u32")
    }
}

impl One for i32 {
    fn one() -> Self {
        1
    }

    fn from_items(items: usize) -> Self {
        items.try_into().expect("benchmark size fits i32")
    }
}

impl One for f32 {
    fn one() -> Self {
        1.0
    }

    fn from_items(items: usize) -> Self {
        items as f32
    }
}

fn expected<T: One>(items: usize) -> T {
    T::from_items(items)
}

fn main() -> Result<(), Box<dyn Error>> {
    let items = argument("--items", 64 * 1024 * 1024);
    if items == 0 {
        return Err("items must be non-zero".into());
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let cub = rccl_cub::cuda::Program::load(&context)?;
    let launch = cub.on(&stream)?;

    benchmark::<u32>(&launch, &stream, items, "u32")?;
    benchmark::<i32>(&launch, &stream, items, "i32")?;
    benchmark::<f32>(&launch, &stream, items, "f32")?;
    Ok(())
}
