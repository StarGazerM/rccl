//! Ordinary Rust entry points for the finite CUB histogram policy table.
//!
//! Channel counts are algorithm semantics and remain const generics. Target
//! architecture is selected by cargo-oxide cfgs inside `histogram_policy`.
//! The small closed matches below only bridge stable Rust's inability to use a
//! generic type's size in another const-generic argument.

use crate::{
    ArithmeticRange, DeviceAlgorithmError, DeviceHistogram, DeviceHistogramWorkspace, DeviceRange,
    HistogramRegion,
    cuda::Launch,
    histogram::{
        HistogramLevels, HistogramScalar,
        device_histogram_arithmetic_workspace as workspace_arithmetic,
        device_histogram_arithmetic_workspace_in as workspace_arithmetic_in,
        device_histogram_even as launch_even,
        device_histogram_even_arithmetic as launch_even_arithmetic,
        device_histogram_range as launch_range,
        device_histogram_range_arithmetic as launch_range_arithmetic,
        device_histogram_workspace as workspace_contiguous,
        device_histogram_workspace_in as workspace_contiguous_in, histogram_policy,
    },
    histogram_backend::{HistogramFourSampleLoad, HistogramSample},
};
type AllocHistogram<'stream, C, A, const ACTIVE: usize> =
    DeviceHistogram<C, ACTIVE, rccl_memory::DeviceBuffer<'stream, C, A>>;
type AllocHistogramWorkspace<'stream, C, A, const ACTIVE: usize> = DeviceHistogramWorkspace<
    C,
    ACTIVE,
    rccl_memory::DeviceBuffer<'stream, C, A>,
    rccl_memory::DeviceBuffer<'stream, i32, A>,
>;

macro_rules! histogram_active_1 {
    ($callback:ident, $even:literal, $execution:expr, $policy:expr, $shared:literal, $global:literal, ($($args:expr),* $(,)?)) => {
        match (
            <S as HistogramScalar>::SIZE,
            $policy.threads,
            $policy.pixels,
            $policy.rle,
        ) {
            (1, 928, 12, false) => $callback!($even, $execution, 928, 12, false, true, $shared, $global, ($($args),*)),
            (1, 448, 12, false) => $callback!($even, $execution, 448, 12, false, true, $shared, $global, ($($args),*)),
            (1, 768, 12, false) => $callback!($even, $execution, 768, 12, false, true, $shared, $global, ($($args),*)),
            (1, 384, 16, true) => $callback!($even, $execution, 384, 16, true, true, $shared, $global, ($($args),*)),
            (2, 960, 10, true) => $callback!($even, $execution, 960, 10, true, false, $shared, $global, ($($args),*)),
            (2 | 4, 384, 16, true) => $callback!($even, $execution, 384, 16, true, false, $shared, $global, ($($args),*)),
            (8, 384, 8, true) => $callback!($even, $execution, 384, 8, true, false, $shared, $global, ($($args),*)),
            _ => Err(DeviceAlgorithmError::SizeOverflow),
        }
    };
}

macro_rules! histogram_active_2 {
    ($callback:ident, $even:literal, $execution:expr, $policy:expr, $shared:literal, $global:literal, ($($args:expr),* $(,)?)) => {
        match (<S as HistogramScalar>::SIZE, $policy.threads, $policy.pixels, $policy.rle) {
            (1, 384, 8, true) => $callback!($even, $execution, 384, 8, true, true, $shared, $global, ($($args),*)),
            (2 | 4, 384, 8, true) => $callback!($even, $execution, 384, 8, true, false, $shared, $global, ($($args),*)),
            (8, 384, 4, true) => $callback!($even, $execution, 384, 4, true, false, $shared, $global, ($($args),*)),
            _ => Err(DeviceAlgorithmError::SizeOverflow),
        }
    };
}

macro_rules! histogram_active_3 {
    ($callback:ident, $even:literal, $execution:expr, $policy:expr, $shared:literal, $global:literal, ($($args:expr),* $(,)?)) => {
        match (<S as HistogramScalar>::SIZE, $policy.threads, $policy.pixels, $policy.rle) {
            (1, 384, 5, true) => $callback!($even, $execution, 384, 5, true, true, $shared, $global, ($($args),*)),
            (2 | 4, 384, 5, true) => $callback!($even, $execution, 384, 5, true, false, $shared, $global, ($($args),*)),
            (8, 384, 2, true) => $callback!($even, $execution, 384, 2, true, false, $shared, $global, ($($args),*)),
            _ => Err(DeviceAlgorithmError::SizeOverflow),
        }
    };
}

macro_rules! histogram_active_4 {
    ($callback:ident, $even:literal, $execution:expr, $policy:expr, $shared:literal, $global:literal, ($($args:expr),* $(,)?)) => {
        match (<S as HistogramScalar>::SIZE, $policy.threads, $policy.pixels, $policy.rle) {
            (1, 384, 4, true) => $callback!($even, $execution, 384, 4, true, true, $shared, $global, ($($args),*)),
            (2 | 4, 384, 4, true) => $callback!($even, $execution, 384, 4, true, false, $shared, $global, ($($args),*)),
            (8, 384, 2, true) => $callback!($even, $execution, 384, 2, true, false, $shared, $global, ($($args),*)),
            _ => Err(DeviceAlgorithmError::SizeOverflow),
        }
    };
}

macro_rules! histogram_dispatch {
    ($callback:ident, $even:literal, $execution:expr, ($($args:expr),* $(,)?)) => {{
        if CHANNELS == 0 || CHANNELS > 4 || ACTIVE == 0 || ACTIVE > CHANNELS {
            Err(DeviceAlgorithmError::InvalidHistogramShape(
                "histogram requires 1..=4 channels and 1..=channels active channels",
            ))
        } else {
            let __policy = histogram_policy(
                <S as HistogramScalar>::SIZE,
                <C as HistogramScalar>::SIZE,
                ACTIVE,
                CHANNELS,
                $even,
            );
            match (<C as HistogramScalar>::SIZE, ACTIVE) {
                (4, 1) => histogram_active_1!($callback, $even, $execution, __policy, 1036, 12, ($($args),*)),
                (4, 2) => histogram_active_2!($callback, $even, $execution, __policy, 2064, 16, ($($args),*)),
                (4, 3) => histogram_active_3!($callback, $even, $execution, __policy, 3092, 20, ($($args),*)),
                (4, 4) => histogram_active_4!($callback, $even, $execution, __policy, 4120, 24, ($($args),*)),
                (8, 1) => histogram_active_1!($callback, $even, $execution, __policy, 2064, 16, ($($args),*)),
                (8, 2) => histogram_active_2!($callback, $even, $execution, __policy, 4120, 24, ($($args),*)),
                (8, 3) => histogram_active_3!($callback, $even, $execution, __policy, 6176, 32, ($($args),*)),
                (8, 4) => histogram_active_4!($callback, $even, $execution, __policy, 8232, 40, ($($args),*)),
                _ => Err(DeviceAlgorithmError::SizeOverflow),
            }
        }
    }};
}

macro_rules! call_workspace_contiguous {
    ($even:literal, $execution:expr, $threads:literal, $pixels:literal, $rle:literal, $byte:literal, $shared:literal, $global:literal, ($samples:expr, $region:expr, $output:expr)) => {
        workspace_contiguous::<
            S,
            C,
            L,
            _,
            _,
            CHANNELS,
            ACTIVE,
            $even,
            $threads,
            $pixels,
            $rle,
            $byte,
            $shared,
            $global,
        >(
            $execution.stream(),
            $execution.histogram_module(),
            $samples,
            $region,
            $output,
        )
    };
}

macro_rules! call_workspace_arithmetic {
    ($even:literal, $execution:expr, $threads:literal, $pixels:literal, $rle:literal, $byte:literal, $shared:literal, $global:literal, ($samples:expr, $region:expr, $output:expr)) => {
        workspace_arithmetic::<
            S,
            C,
            L,
            _,
            CHANNELS,
            ACTIVE,
            $even,
            $threads,
            $pixels,
            $rle,
            $byte,
            $shared,
            $global,
        >(
            $execution.stream(),
            $execution.histogram_module(),
            $samples,
            $region,
            $output,
        )
    };
}

macro_rules! call_workspace_contiguous_in {
    ($even:literal, $execution:expr, $threads:literal, $pixels:literal, $rle:literal, $byte:literal, $shared:literal, $global:literal, ($allocator:expr, $samples:expr, $region:expr, $output:expr)) => {
        workspace_contiguous_in::<
            S,
            C,
            L,
            _,
            _,
            _,
            CHANNELS,
            ACTIVE,
            $even,
            $threads,
            $pixels,
            $rle,
            $byte,
            $shared,
            $global,
        >(
            $allocator,
            $execution.stream(),
            $execution.histogram_module(),
            $samples,
            $region,
            $output,
        )
    };
}

macro_rules! call_workspace_arithmetic_in {
    ($even:literal, $execution:expr, $threads:literal, $pixels:literal, $rle:literal, $byte:literal, $shared:literal, $global:literal, ($allocator:expr, $samples:expr, $region:expr, $output:expr)) => {
        workspace_arithmetic_in::<
            S,
            C,
            L,
            _,
            _,
            CHANNELS,
            ACTIVE,
            $even,
            $threads,
            $pixels,
            $rle,
            $byte,
            $shared,
            $global,
        >(
            $allocator,
            $execution.stream(),
            $execution.histogram_module(),
            $samples,
            $region,
            $output,
        )
    };
}

macro_rules! call_even {
    ($even:literal, $execution:expr, $threads:literal, $pixels:literal, $rle:literal, $byte:literal, $shared:literal, $global:literal, ($samples:expr, $region:expr, $output:expr, $lower:expr, $upper:expr, $workspace:expr)) => {
        launch_even::<
            S,
            C,
            L,
            _,
            _,
            _,
            _,
            CHANNELS,
            ACTIVE,
            $threads,
            $pixels,
            $rle,
            $byte,
            $shared,
            $global,
        >(
            $execution.stream(),
            $execution.histogram_module(),
            $samples,
            $region,
            $output,
            $lower,
            $upper,
            $workspace,
        )
    };
}

macro_rules! call_range {
    ($even:literal, $execution:expr, $threads:literal, $pixels:literal, $rle:literal, $byte:literal, $shared:literal, $global:literal, ($samples:expr, $region:expr, $output:expr, $levels:expr, $workspace:expr)) => {
        launch_range::<
            S,
            C,
            L,
            _,
            _,
            _,
            _,
            _,
            CHANNELS,
            ACTIVE,
            $threads,
            $pixels,
            $rle,
            $byte,
            $shared,
            $global,
        >(
            $execution.stream(),
            $execution.histogram_module(),
            $samples,
            $region,
            $output,
            $levels,
            $workspace,
        )
    };
}

macro_rules! call_even_arithmetic {
    ($even:literal, $execution:expr, $threads:literal, $pixels:literal, $rle:literal, $byte:literal, $shared:literal, $global:literal, ($samples:expr, $region:expr, $output:expr, $lower:expr, $upper:expr, $workspace:expr)) => {
        launch_even_arithmetic::<
            S,
            C,
            L,
            _,
            _,
            _,
            CHANNELS,
            ACTIVE,
            $threads,
            $pixels,
            $rle,
            $byte,
            $shared,
            $global,
        >(
            $execution.stream(),
            $execution.histogram_module(),
            $samples,
            $region,
            $output,
            $lower,
            $upper,
            $workspace,
        )
    };
}

macro_rules! call_range_arithmetic {
    ($even:literal, $execution:expr, $threads:literal, $pixels:literal, $rle:literal, $byte:literal, $shared:literal, $global:literal, ($samples:expr, $region:expr, $output:expr, $levels:expr, $workspace:expr)) => {
        launch_range_arithmetic::<
            S,
            C,
            L,
            _,
            _,
            _,
            _,
            CHANNELS,
            ACTIVE,
            $threads,
            $pixels,
            $rle,
            $byte,
            $shared,
            $global,
        >(
            $execution.stream(),
            $execution.histogram_module(),
            $samples,
            $region,
            $output,
            $levels,
            $workspace,
        )
    };
}

pub fn device_histogram_even_workspace<S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    execution: &Launch<'_, '_>,
    samples: &impl DeviceRange,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE>,
) -> Result<DeviceHistogramWorkspace<C, ACTIVE>, DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
{
    histogram_dispatch!(
        call_workspace_contiguous,
        true,
        execution,
        (samples, region, output)
    )
}

pub fn device_histogram_range_workspace<S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    execution: &Launch<'_, '_>,
    samples: &impl DeviceRange,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE>,
) -> Result<DeviceHistogramWorkspace<C, ACTIVE>, DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
{
    histogram_dispatch!(
        call_workspace_contiguous,
        false,
        execution,
        (samples, region, output)
    )
}

pub fn device_histogram_even_arithmetic_workspace<
    S,
    C,
    L,
    const CHANNELS: usize,
    const ACTIVE: usize,
>(
    execution: &Launch<'_, '_>,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE>,
) -> Result<DeviceHistogramWorkspace<C, ACTIVE>, DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
{
    histogram_dispatch!(
        call_workspace_arithmetic,
        true,
        execution,
        (samples, region, output)
    )
}

pub fn device_histogram_range_arithmetic_workspace<
    S,
    C,
    L,
    const CHANNELS: usize,
    const ACTIVE: usize,
>(
    execution: &Launch<'_, '_>,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE>,
) -> Result<DeviceHistogramWorkspace<C, ACTIVE>, DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
{
    histogram_dispatch!(
        call_workspace_arithmetic,
        false,
        execution,
        (samples, region, output)
    )
}

pub fn device_histogram_even_workspace_in<
    'stream,
    S,
    C,
    L,
    A,
    const CHANNELS: usize,
    const ACTIVE: usize,
>(
    allocator: A,
    execution: &Launch<'stream, '_>,
    samples: &impl DeviceRange,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE, rccl_memory::DeviceBuffer<'stream, C, A>>,
) -> Result<
    DeviceHistogramWorkspace<
        C,
        ACTIVE,
        rccl_memory::DeviceBuffer<'stream, C, A>,
        rccl_memory::DeviceBuffer<'stream, i32, A>,
    >,
    DeviceAlgorithmError,
>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    A: rccl_memory::DeviceAllocator + Clone,
{
    histogram_dispatch!(
        call_workspace_contiguous_in,
        true,
        execution,
        (allocator, samples, region, output)
    )
}

pub fn device_histogram_range_workspace_in<
    'stream,
    S,
    C,
    L,
    A,
    const CHANNELS: usize,
    const ACTIVE: usize,
>(
    allocator: A,
    execution: &Launch<'stream, '_>,
    samples: &impl DeviceRange,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE, rccl_memory::DeviceBuffer<'stream, C, A>>,
) -> Result<
    DeviceHistogramWorkspace<
        C,
        ACTIVE,
        rccl_memory::DeviceBuffer<'stream, C, A>,
        rccl_memory::DeviceBuffer<'stream, i32, A>,
    >,
    DeviceAlgorithmError,
>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    A: rccl_memory::DeviceAllocator + Clone,
{
    histogram_dispatch!(
        call_workspace_contiguous_in,
        false,
        execution,
        (allocator, samples, region, output)
    )
}

pub fn device_histogram_even_arithmetic_workspace_in<
    'stream,
    S,
    C,
    L,
    A,
    const CHANNELS: usize,
    const ACTIVE: usize,
>(
    allocator: A,
    execution: &Launch<'stream, '_>,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE, rccl_memory::DeviceBuffer<'stream, C, A>>,
) -> Result<
    DeviceHistogramWorkspace<
        C,
        ACTIVE,
        rccl_memory::DeviceBuffer<'stream, C, A>,
        rccl_memory::DeviceBuffer<'stream, i32, A>,
    >,
    DeviceAlgorithmError,
>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    A: rccl_memory::DeviceAllocator + Clone,
{
    histogram_dispatch!(
        call_workspace_arithmetic_in,
        true,
        execution,
        (allocator, samples, region, output)
    )
}

pub fn device_histogram_range_arithmetic_workspace_in<
    'stream,
    S,
    C,
    L,
    A,
    const CHANNELS: usize,
    const ACTIVE: usize,
>(
    allocator: A,
    execution: &Launch<'stream, '_>,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE, rccl_memory::DeviceBuffer<'stream, C, A>>,
) -> Result<
    DeviceHistogramWorkspace<
        C,
        ACTIVE,
        rccl_memory::DeviceBuffer<'stream, C, A>,
        rccl_memory::DeviceBuffer<'stream, i32, A>,
    >,
    DeviceAlgorithmError,
>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    A: rccl_memory::DeviceAllocator + Clone,
{
    histogram_dispatch!(
        call_workspace_arithmetic_in,
        false,
        execution,
        (allocator, samples, region, output)
    )
}

#[allow(clippy::too_many_arguments)]
pub fn device_histogram_even<S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    execution: &Launch<'_, '_>,
    samples: &(impl DeviceRange + cuda_host::KernelSliceArg<Elem = S>),
    region: HistogramRegion,
    output: &mut DeviceHistogram<C, ACTIVE>,
    lower_levels: [L; ACTIVE],
    upper_levels: [L; ACTIVE],
    workspace: &mut DeviceHistogramWorkspace<C, ACTIVE>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
{
    histogram_dispatch!(
        call_even,
        true,
        execution,
        (
            samples,
            region,
            output,
            lower_levels,
            upper_levels,
            workspace
        )
    )
}

pub fn device_histogram_range<S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    execution: &Launch<'_, '_>,
    samples: &(impl DeviceRange + cuda_host::KernelSliceArg<Elem = S>),
    region: HistogramRegion,
    output: &mut DeviceHistogram<C, ACTIVE>,
    levels: &impl HistogramLevels<L, ACTIVE>,
    workspace: &mut DeviceHistogramWorkspace<C, ACTIVE>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
{
    histogram_dispatch!(
        call_range,
        false,
        execution,
        (samples, region, output, levels, workspace)
    )
}

#[allow(clippy::too_many_arguments)]
pub fn device_histogram_even_arithmetic<S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    execution: &Launch<'_, '_>,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &mut DeviceHistogram<C, ACTIVE>,
    lower_levels: [L; ACTIVE],
    upper_levels: [L; ACTIVE],
    workspace: &mut DeviceHistogramWorkspace<C, ACTIVE>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
{
    histogram_dispatch!(
        call_even_arithmetic,
        true,
        execution,
        (
            samples,
            region,
            output,
            lower_levels,
            upper_levels,
            workspace
        )
    )
}

pub fn device_histogram_range_arithmetic<S, C, L, const CHANNELS: usize, const ACTIVE: usize>(
    execution: &Launch<'_, '_>,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &mut DeviceHistogram<C, ACTIVE>,
    levels: &impl HistogramLevels<L, ACTIVE>,
    workspace: &mut DeviceHistogramWorkspace<C, ACTIVE>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
{
    histogram_dispatch!(
        call_range_arithmetic,
        false,
        execution,
        (samples, region, output, levels, workspace)
    )
}

#[allow(clippy::too_many_arguments)]
pub fn device_histogram_even_in<'stream, S, C, L, A, const CHANNELS: usize, const ACTIVE: usize>(
    execution: &Launch<'stream, '_>,
    samples: &(impl DeviceRange + cuda_host::KernelSliceArg<Elem = S>),
    region: HistogramRegion,
    output: &mut AllocHistogram<'stream, C, A, ACTIVE>,
    lower_levels: [L; ACTIVE],
    upper_levels: [L; ACTIVE],
    workspace: &mut AllocHistogramWorkspace<'stream, C, A, ACTIVE>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    A: rccl_memory::DeviceAllocator,
{
    histogram_dispatch!(
        call_even,
        true,
        execution,
        (
            samples,
            region,
            output,
            lower_levels,
            upper_levels,
            workspace
        )
    )
}

pub fn device_histogram_range_in<'stream, S, C, L, A, const CHANNELS: usize, const ACTIVE: usize>(
    execution: &Launch<'stream, '_>,
    samples: &(impl DeviceRange + cuda_host::KernelSliceArg<Elem = S>),
    region: HistogramRegion,
    output: &mut AllocHistogram<'stream, C, A, ACTIVE>,
    levels: &impl HistogramLevels<L, ACTIVE>,
    workspace: &mut AllocHistogramWorkspace<'stream, C, A, ACTIVE>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    A: rccl_memory::DeviceAllocator,
{
    histogram_dispatch!(
        call_range,
        false,
        execution,
        (samples, region, output, levels, workspace)
    )
}

#[allow(clippy::too_many_arguments)]
pub fn device_histogram_even_arithmetic_in<
    'stream,
    S,
    C,
    L,
    A,
    const CHANNELS: usize,
    const ACTIVE: usize,
>(
    execution: &Launch<'stream, '_>,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &mut AllocHistogram<'stream, C, A, ACTIVE>,
    lower_levels: [L; ACTIVE],
    upper_levels: [L; ACTIVE],
    workspace: &mut AllocHistogramWorkspace<'stream, C, A, ACTIVE>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    A: rccl_memory::DeviceAllocator,
{
    histogram_dispatch!(
        call_even_arithmetic,
        true,
        execution,
        (
            samples,
            region,
            output,
            lower_levels,
            upper_levels,
            workspace
        )
    )
}

pub fn device_histogram_range_arithmetic_in<
    'stream,
    S,
    C,
    L,
    A,
    const CHANNELS: usize,
    const ACTIVE: usize,
>(
    execution: &Launch<'stream, '_>,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &mut AllocHistogram<'stream, C, A, ACTIVE>,
    levels: &impl HistogramLevels<L, ACTIVE>,
    workspace: &mut AllocHistogramWorkspace<'stream, C, A, ACTIVE>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    A: rccl_memory::DeviceAllocator,
{
    histogram_dispatch!(
        call_range_arithmetic,
        false,
        execution,
        (samples, region, output, levels, workspace)
    )
}
