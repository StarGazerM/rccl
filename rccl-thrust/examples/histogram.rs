//! Host-facing histogram calls using the data from pinned
//! `thrust/examples/histogram.cu`.

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{
    DeviceAlgorithmError, DeviceBufferExt, DeviceHistogramLevels, HistogramRegion, arithmetic_range,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_thrust::cuda::Program::load(&context)?;
    let execution = program.par(&stream)?;
    let tracked_allocator = rccl_thrust::memory::TrackingMemoryResource::new(
        rccl_thrust::memory::CudaMemoryResource::new(context.clone()),
    );
    let tracked_execution = program.par_in(&stream, &tracked_allocator)?;

    // Pinned Thrust example fixture: two zeroes, five ones, three twos,
    // no threes, and one four.
    let samples = DeviceBuffer::from_host(&stream, &[2_i32, 1, 0, 0, 2, 2, 1, 1, 1, 1, 4])?;
    let dense = rccl_thrust::histogram_even::<i32, i32, i32, 1, 1>(
        samples.as_device_slice(),
        HistogramRegion::All,
        [5],
        [0],
        [5],
        &execution,
    )?;
    assert_eq!(dense.to_host_channels(&stream)?[0], [2, 5, 3, 0, 1]);
    let tracked_dense = rccl_thrust::histogram_even::<i32, i32, i32, 1, 1>(
        samples.as_device_slice(),
        HistogramRegion::All,
        [5],
        [0],
        [5],
        &tracked_execution,
    )?;
    assert_eq!(tracked_dense.to_host_channels(&stream)?[0], [2, 5, 3, 0, 1]);
    let tracked_stats = tracked_allocator.stats();
    assert!(
        tracked_stats.total_allocations >= 3,
        "histogram par_in must allocate output, privatized storage, and queue through the supplied allocator"
    );

    let levels = DeviceHistogramLevels::from_channels(&stream, [&[0_i32, 1, 2, 3, 4, 5]])?;
    let ranged = rccl_thrust::histogram_range::<i32, i32, i32, 1, 1>(
        samples.as_device_slice(),
        HistogramRegion::All,
        &levels,
        &execution,
    )?;
    assert_eq!(ranged.to_host_channels(&stream)?[0], [2, 5, 3, 0, 1]);

    // Pinned counting-iterator semantics remain an unmaterialized finite Rust
    // range all the way through the host facade.
    let counting = arithmetic_range(0_i32, 1_i32, 9);
    let arithmetic = rccl_thrust::histogram_even_arithmetic::<i32, i32, i32, 1, 1>(
        &counting,
        HistogramRegion::All,
        [3],
        [0],
        [9],
        &execution,
    )?;
    assert_eq!(arithmetic.to_host_channels(&stream)?[0], [3, 3, 3]);

    let arithmetic_levels = DeviceHistogramLevels::from_channels(&stream, [&[0_i32, 3, 6, 9]])?;
    let arithmetic_range = rccl_thrust::histogram_range_arithmetic::<i32, i32, i32, 1, 1>(
        &counting,
        HistogramRegion::All,
        &arithmetic_levels,
        &execution,
    )?;
    assert_eq!(arithmetic_range.to_host_channels(&stream)?[0], [3, 3, 3]);

    // The negative path checks the exact semantic error. Merely observing
    // `is_err()` would let an unrelated allocation or launch failure mask a
    // broken bounds check.
    let invalid = rccl_thrust::histogram_even::<i32, i32, i32, 1, 1>(
        samples.as_device_slice(),
        HistogramRegion::All,
        [5],
        [4],
        [4],
        &execution,
    );
    assert!(matches!(
        invalid,
        Err(DeviceAlgorithmError::InvalidHistogramShape(
            "even histogram bounds or bin arithmetic are invalid"
        ))
    ));

    println!("pinned Thrust histogram host facade: PASS");
    Ok(())
}
