//! Synchronous Rust histogram functions over borrowed device ranges.
//!
//! The const channel counts describe the data layout. Architecture and kernel
//! policy come only from cargo-oxide's target cfg and never appear in this API.

use crate::{
    ArithmeticRange, DeviceAlgorithmError, DeviceHistogram, DeviceView, HistogramRegion,
    cuda::Execution,
};

type Result<T> = core::result::Result<T, DeviceAlgorithmError>;

pub fn histogram_even<S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    samples: DeviceView<'_, S>,
    region: HistogramRegion,
    bin_counts: [usize; ACTIVE],
    lower_levels: [L; ACTIVE],
    upper_levels: [L; ACTIVE],
    execution: &Execution<'_, '_, impl rccl_memory::DeviceAllocator>,
) -> Result<DeviceHistogram<C, ACTIVE>>
where
    S: rccl_cub::HistogramScalar + rccl_cub::HistogramSample<L> + rccl_cub::HistogramFourSampleLoad,
    C: rccl_cub::HistogramScalar + rccl_cub::HistogramCounter,
    L: rccl_cub::HistogramScalar + PartialOrd,
{
    let samples = samples.__into_buffer();
    let mut output = DeviceHistogram::<C, ACTIVE>::new(execution.stream(), bin_counts)?;
    let mut workspace = rccl_cub::device_histogram_even_workspace::<S, C, L, CHANNELS, ACTIVE>(
        execution.launch(),
        samples,
        region,
        &output,
    )?;
    let result = rccl_cub::device_histogram_even::<S, C, L, CHANNELS, ACTIVE>(
        execution.launch(),
        samples,
        region,
        &mut output,
        lower_levels,
        upper_levels,
        &mut workspace,
    );
    execution.synchronize(result)?;
    Ok(output)
}

pub fn histogram_range<S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    samples: DeviceView<'_, S>,
    region: HistogramRegion,
    levels: &impl rccl_cub::HistogramLevels<L, ACTIVE>,
    execution: &Execution<'_, '_, impl rccl_memory::DeviceAllocator>,
) -> Result<DeviceHistogram<C, ACTIVE>>
where
    S: rccl_cub::HistogramScalar + rccl_cub::HistogramSample<L> + rccl_cub::HistogramFourSampleLoad,
    C: rccl_cub::HistogramScalar + rccl_cub::HistogramCounter,
    L: rccl_cub::HistogramScalar + PartialOrd,
{
    let samples = samples.__into_buffer();
    let bin_counts =
        core::array::from_fn(|channel| levels.histogram_level_parts().1[channel].saturating_sub(1));
    let mut output = DeviceHistogram::<C, ACTIVE>::new(execution.stream(), bin_counts)?;
    let mut workspace = rccl_cub::device_histogram_range_workspace::<S, C, L, CHANNELS, ACTIVE>(
        execution.launch(),
        samples,
        region,
        &output,
    )?;
    let result = rccl_cub::device_histogram_range::<S, C, L, CHANNELS, ACTIVE>(
        execution.launch(),
        samples,
        region,
        &mut output,
        levels,
        &mut workspace,
    );
    execution.synchronize(result)?;
    Ok(output)
}

pub fn histogram_even_arithmetic<S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    bin_counts: [usize; ACTIVE],
    lower_levels: [L; ACTIVE],
    upper_levels: [L; ACTIVE],
    execution: &Execution<'_, '_, impl rccl_memory::DeviceAllocator>,
) -> Result<DeviceHistogram<C, ACTIVE>>
where
    S: rccl_cub::HistogramScalar
        + rccl_cub::HistogramSample<L>
        + rccl_cub::HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: rccl_cub::HistogramScalar + rccl_cub::HistogramCounter,
    L: rccl_cub::HistogramScalar + PartialOrd,
{
    let mut output = DeviceHistogram::<C, ACTIVE>::new(execution.stream(), bin_counts)?;
    let mut workspace = rccl_cub::device_histogram_even_arithmetic_workspace::<
        S,
        C,
        L,
        CHANNELS,
        ACTIVE,
    >(execution.launch(), samples, region, &output)?;
    let result = rccl_cub::device_histogram_even_arithmetic::<S, C, L, CHANNELS, ACTIVE>(
        execution.launch(),
        samples,
        region,
        &mut output,
        lower_levels,
        upper_levels,
        &mut workspace,
    );
    execution.synchronize(result)?;
    Ok(output)
}

pub fn histogram_range_arithmetic<S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    levels: &impl rccl_cub::HistogramLevels<L, ACTIVE>,
    execution: &Execution<'_, '_, impl rccl_memory::DeviceAllocator>,
) -> Result<DeviceHistogram<C, ACTIVE>>
where
    S: rccl_cub::HistogramScalar
        + rccl_cub::HistogramSample<L>
        + rccl_cub::HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: rccl_cub::HistogramScalar + rccl_cub::HistogramCounter,
    L: rccl_cub::HistogramScalar + PartialOrd,
{
    let bin_counts =
        core::array::from_fn(|channel| levels.histogram_level_parts().1[channel].saturating_sub(1));
    let mut output = DeviceHistogram::<C, ACTIVE>::new(execution.stream(), bin_counts)?;
    let mut workspace = rccl_cub::device_histogram_range_arithmetic_workspace::<
        S,
        C,
        L,
        CHANNELS,
        ACTIVE,
    >(execution.launch(), samples, region, &output)?;
    let result = rccl_cub::device_histogram_range_arithmetic::<S, C, L, CHANNELS, ACTIVE>(
        execution.launch(),
        samples,
        region,
        &mut output,
        levels,
        &mut workspace,
    );
    execution.synchronize(result)?;
    Ok(output)
}
