//! Synchronous host-facing histogram calls.
//!
//! Pinned CCCL Thrust does not define a standalone `thrust::histogram`
//! algorithm; its `examples/histogram.cu` composes several independent
//! algorithms. RCCL nevertheless needs a safe host facade for the CUB
//! histogram primitive implemented below this crate. These shallow macros
//! provide that facade without inventing a runtime policy registry or exposing
//! CUB temporary storage.
//!
//! The sample, counter, level, and channel arguments remain macro-visible
//! because they select CUB policy and kernel families at compile time. All
//! ordinary operands remain Rust expressions, and the call synchronizes before
//! releasing the borrowed input or private workspace.

/// Compute an even-bin histogram from a borrowed contiguous device range.
///
/// The result owns one device allocation containing every active channel.
/// Channel lengths, lower levels, and upper levels are fixed-size Rust arrays;
/// no output-pointer array or temporary-storage byte count is exposed.
#[macro_export]
macro_rules! histogram_even {
    (
        ::<$sample:ty, $counter:ty, $level:ty, $channels:literal, $active:literal>(
            $samples:expr,
            $region:expr,
            $bin_counts:expr,
            $lower_levels:expr,
            $upper_levels:expr,
            $execution:expr $(,)?
        )
    ) => {{
        (|| -> ::core::result::Result<
                        $crate::DeviceHistogram<$counter, $active>,
                        $crate::DeviceAlgorithmError,
                    > {
                        let __rccl_execution = $execution;
                        let __rccl_samples_view: $crate::DeviceView<'_, $sample> = $samples;
                        let __rccl_samples = __rccl_samples_view.__into_buffer();
                        let __rccl_region = $region;
                        let __rccl_bin_counts: [usize; $active] = $bin_counts;
                        let mut __rccl_output = $crate::DeviceHistogram::<$counter, $active>::new(
                            __rccl_execution.stream(),
                            __rccl_bin_counts,
                        )?;
                        let mut __rccl_workspace =
                            $crate::__private::rccl_cub::device_histogram_even_workspace!(::<
                                $sample,
                                $counter,
                                $level,
                                $channels,
                                $active,
                            >(
                                __rccl_execution,
                                __rccl_samples,
                                __rccl_region,
                                &__rccl_output,
                            ))?;
                        let __rccl_result = $crate::__private::rccl_cub::device_histogram_even!(::<
                            $sample,
                            $counter,
                            $level,
                            $channels,
                            $active,
                        >(
                            __rccl_execution,
                            __rccl_samples,
                            __rccl_region,
                            &mut __rccl_output,
                            $lower_levels,
                            $upper_levels,
                            &mut __rccl_workspace,
                        ));
                        __rccl_execution.__synchronize(__rccl_result)?;
                        Ok(__rccl_output)
                    })()
    }};
}

/// Compute a boundary-defined histogram from a borrowed contiguous range.
#[macro_export]
macro_rules! histogram_range {
    (
        ::<$sample:ty, $counter:ty, $level:ty, $channels:literal, $active:literal>(
            $samples:expr,
            $region:expr,
            $levels:expr,
            $execution:expr $(,)?
        )
    ) => {{
        (|| -> ::core::result::Result<
                        $crate::DeviceHistogram<$counter, $active>,
                        $crate::DeviceAlgorithmError,
                    > {
                        let __rccl_execution = $execution;
                        let __rccl_samples_view: $crate::DeviceView<'_, $sample> = $samples;
                        let __rccl_samples = __rccl_samples_view.__into_buffer();
                        let __rccl_region = $region;
                        let __rccl_levels = $levels;
                        let __rccl_level_lengths = *__rccl_levels.lengths();
                        let __rccl_bin_counts: [usize; $active] = ::core::array::from_fn(|channel| {
                            __rccl_level_lengths[channel].saturating_sub(1)
                        });
                        let mut __rccl_output = $crate::DeviceHistogram::<$counter, $active>::new(
                            __rccl_execution.stream(),
                            __rccl_bin_counts,
                        )?;
                        let mut __rccl_workspace =
                            $crate::__private::rccl_cub::device_histogram_range_workspace!(::<
                                $sample,
                                $counter,
                                $level,
                                $channels,
                                $active,
                            >(
                                __rccl_execution,
                                __rccl_samples,
                                __rccl_region,
                                &__rccl_output,
                            ))?;
                        let __rccl_result = $crate::__private::rccl_cub::device_histogram_range!(::<
                            $sample,
                            $counter,
                            $level,
                            $channels,
                            $active,
                        >(
                            __rccl_execution,
                            __rccl_samples,
                            __rccl_region,
                            &mut __rccl_output,
                            __rccl_levels,
                            &mut __rccl_workspace,
                        ));
                        __rccl_execution.__synchronize(__rccl_result)?;
                        Ok(__rccl_output)
                    })()
    }};
}

/// Compute an even-bin histogram from an arithmetic random-access range.
#[macro_export]
macro_rules! histogram_even_arithmetic {
    (
        ::<$sample:ty, $counter:ty, $level:ty, $channels:literal, $active:literal>(
            $samples:expr,
            $region:expr,
            $bin_counts:expr,
            $lower_levels:expr,
            $upper_levels:expr,
            $execution:expr $(,)?
        )
    ) => {{
        (|| -> ::core::result::Result<
                    $crate::DeviceHistogram<$counter, $active>,
                    $crate::DeviceAlgorithmError,
                > {
                    let __rccl_execution = $execution;
                    let __rccl_samples: &$crate::ArithmeticRange<$sample> = $samples;
                    let __rccl_region = $region;
                    let __rccl_bin_counts: [usize; $active] = $bin_counts;
                    let mut __rccl_output = $crate::DeviceHistogram::<$counter, $active>::new(
                        __rccl_execution.stream(),
                        __rccl_bin_counts,
                    )?;
                    let mut __rccl_workspace =
                        $crate::__private::rccl_cub::device_histogram_even_arithmetic_workspace!(::<
                            $sample,
                            $counter,
                            $level,
                            $channels,
                            $active,
                        >(
                            __rccl_execution,
                            __rccl_samples,
                            __rccl_region,
                            &__rccl_output,
                        ))?;
                    let __rccl_result =
                        $crate::__private::rccl_cub::device_histogram_even_arithmetic!(::<
                            $sample,
                            $counter,
                            $level,
                            $channels,
                            $active,
                        >(
                            __rccl_execution,
                            __rccl_samples,
                            __rccl_region,
                            &mut __rccl_output,
                            $lower_levels,
                            $upper_levels,
                            &mut __rccl_workspace,
                        ));
                    __rccl_execution.__synchronize(__rccl_result)?;
                    Ok(__rccl_output)
                })()
    }};
}

/// Compute a boundary-defined histogram from an arithmetic range.
#[macro_export]
macro_rules! histogram_range_arithmetic {
    (
        ::<$sample:ty, $counter:ty, $level:ty, $channels:literal, $active:literal>(
            $samples:expr,
            $region:expr,
            $levels:expr,
            $execution:expr $(,)?
        )
    ) => {{
        (|| -> ::core::result::Result<
                $crate::DeviceHistogram<$counter, $active>,
                $crate::DeviceAlgorithmError,
            > {
                let __rccl_execution = $execution;
                let __rccl_samples: &$crate::ArithmeticRange<$sample> = $samples;
                let __rccl_region = $region;
                let __rccl_levels = $levels;
                let __rccl_level_lengths = *__rccl_levels.lengths();
                let __rccl_bin_counts: [usize; $active] = ::core::array::from_fn(|channel| {
                    __rccl_level_lengths[channel].saturating_sub(1)
                });
                let mut __rccl_output = $crate::DeviceHistogram::<$counter, $active>::new(
                    __rccl_execution.stream(),
                    __rccl_bin_counts,
                )?;
                let mut __rccl_workspace =
                    $crate::__private::rccl_cub::device_histogram_range_arithmetic_workspace!(::<
                        $sample,
                        $counter,
                        $level,
                        $channels,
                        $active,
                    >(
                        __rccl_execution,
                        __rccl_samples,
                        __rccl_region,
                        &__rccl_output,
                    ))?;
                let __rccl_result =
                    $crate::__private::rccl_cub::device_histogram_range_arithmetic!(::<
                        $sample,
                        $counter,
                        $level,
                        $channels,
                        $active,
                    >(
                        __rccl_execution,
                        __rccl_samples,
                        __rccl_region,
                        &mut __rccl_output,
                        __rccl_levels,
                        &mut __rccl_workspace,
                    ));
                __rccl_execution.__synchronize(__rccl_result)?;
                Ok(__rccl_output)
            })()
    }};
}
