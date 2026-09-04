//! Synchronous Rust histogram functions over borrowed device ranges.
//!
//! The const channel counts describe the data layout. Architecture and kernel
//! policy come only from cargo-oxide's target cfg and never appear in this API.

use crate::{
    ArithmeticRange, DeviceAlgorithmError, DeviceHistogram, DeviceView, HistogramRegion,
    cuda::Execution,
};

type Result<T> = core::result::Result<T, DeviceAlgorithmError>;
type ExecutionAllocator<'execution> = &'execution (dyn rccl_memory::DeviceAllocator + 'execution);
type ExecutionHistogram<'stream, 'execution, C, const ACTIVE: usize> = DeviceHistogram<
    C,
    ACTIVE,
    rccl_memory::DeviceBuffer<'stream, C, ExecutionAllocator<'execution>>,
>;

pub fn histogram_even<'execution, 'stream, S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    samples: DeviceView<'_, S>,
    region: HistogramRegion,
    bin_counts: [usize; ACTIVE],
    lower_levels: [L; ACTIVE],
    upper_levels: [L; ACTIVE],
    execution: &'execution Execution<'stream, '_, impl rccl_memory::DeviceAllocator>,
) -> Result<ExecutionHistogram<'stream, 'execution, C, ACTIVE>>
where
    S: rccl_cub::HistogramScalar + rccl_cub::HistogramSample<L> + rccl_cub::HistogramFourSampleLoad,
    C: rccl_cub::HistogramScalar + rccl_cub::HistogramCounter,
    L: rccl_cub::HistogramScalar + PartialOrd,
{
    let samples = samples.__into_buffer();
    let allocator: ExecutionAllocator<'execution> = execution.allocator();
    let mut output =
        DeviceHistogram::<C, ACTIVE, _>::new_in(allocator, execution.stream(), bin_counts)?;
    let mut workspace = rccl_cub::device_histogram_even_workspace_in::<S, C, L, _, CHANNELS, ACTIVE>(
        allocator,
        execution.launch(),
        samples,
        region,
        &output,
    )?;
    let result = rccl_cub::device_histogram_even_in::<S, C, L, _, CHANNELS, ACTIVE>(
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

pub fn histogram_range<'execution, 'stream, S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    samples: DeviceView<'_, S>,
    region: HistogramRegion,
    levels: &impl rccl_cub::HistogramLevels<L, ACTIVE>,
    execution: &'execution Execution<'stream, '_, impl rccl_memory::DeviceAllocator>,
) -> Result<ExecutionHistogram<'stream, 'execution, C, ACTIVE>>
where
    S: rccl_cub::HistogramScalar + rccl_cub::HistogramSample<L> + rccl_cub::HistogramFourSampleLoad,
    C: rccl_cub::HistogramScalar + rccl_cub::HistogramCounter,
    L: rccl_cub::HistogramScalar + PartialOrd,
{
    let samples = samples.__into_buffer();
    let bin_counts =
        core::array::from_fn(|channel| levels.histogram_level_parts().1[channel].saturating_sub(1));
    let allocator: ExecutionAllocator<'execution> = execution.allocator();
    let mut output =
        DeviceHistogram::<C, ACTIVE, _>::new_in(allocator, execution.stream(), bin_counts)?;
    let mut workspace = rccl_cub::device_histogram_range_workspace_in::<
        S,
        C,
        L,
        _,
        CHANNELS,
        ACTIVE,
    >(allocator, execution.launch(), samples, region, &output)?;
    let result = rccl_cub::device_histogram_range_in::<S, C, L, _, CHANNELS, ACTIVE>(
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

pub fn histogram_even_arithmetic<
    'execution,
    'stream,
    S,
    C,
    L,
    const CHANNELS: usize,
    const ACTIVE: usize,
>(
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    bin_counts: [usize; ACTIVE],
    lower_levels: [L; ACTIVE],
    upper_levels: [L; ACTIVE],
    execution: &'execution Execution<'stream, '_, impl rccl_memory::DeviceAllocator>,
) -> Result<ExecutionHistogram<'stream, 'execution, C, ACTIVE>>
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
    let allocator: ExecutionAllocator<'execution> = execution.allocator();
    let mut output =
        DeviceHistogram::<C, ACTIVE, _>::new_in(allocator, execution.stream(), bin_counts)?;
    let mut workspace = rccl_cub::device_histogram_even_arithmetic_workspace_in::<
        S,
        C,
        L,
        _,
        CHANNELS,
        ACTIVE,
    >(allocator, execution.launch(), samples, region, &output)?;
    let result = rccl_cub::device_histogram_even_arithmetic_in::<S, C, L, _, CHANNELS, ACTIVE>(
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

pub fn histogram_range_arithmetic<
    'execution,
    'stream,
    S,
    C,
    L,
    const CHANNELS: usize,
    const ACTIVE: usize,
>(
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    levels: &impl rccl_cub::HistogramLevels<L, ACTIVE>,
    execution: &'execution Execution<'stream, '_, impl rccl_memory::DeviceAllocator>,
) -> Result<ExecutionHistogram<'stream, 'execution, C, ACTIVE>>
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
    let allocator: ExecutionAllocator<'execution> = execution.allocator();
    let mut output =
        DeviceHistogram::<C, ACTIVE, _>::new_in(allocator, execution.stream(), bin_counts)?;
    let mut workspace = rccl_cub::device_histogram_range_arithmetic_workspace_in::<
        S,
        C,
        L,
        _,
        CHANNELS,
        ACTIVE,
    >(allocator, execution.launch(), samples, region, &output)?;
    let result = rccl_cub::device_histogram_range_arithmetic_in::<S, C, L, _, CHANNELS, ACTIVE>(
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
