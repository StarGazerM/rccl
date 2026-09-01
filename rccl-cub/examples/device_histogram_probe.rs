//! DeviceHistogram examples corresponding to the pinned CCCL env examples.

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cub::{
    DeviceAlgorithmError, DeviceHistogram, DeviceHistogramLevels, DeviceHistogramLevelsView,
    HistogramRegion,
};
use std::error::Error;

macro_rules! check_scalar_histograms {
    ($stream:expr, $launch:expr, $sample:ty, $level:ty) => {{
        let samples_host: [$sample; 9] = [
            0 as $sample,
            1 as $sample,
            2 as $sample,
            3 as $sample,
            0 as $sample,
            1 as $sample,
            2 as $sample,
            3 as $sample,
            4 as $sample,
        ];
        let samples = DeviceBuffer::from_host($stream, &samples_host)?;

        let mut even = DeviceHistogram::<i32, 1>::new($stream, [4])?;
        let mut even_workspace = rccl_cub::device_histogram_even_workspace!(::<
            $sample,
            i32,
            $level,
            1,
            1,
        >(
            $launch,
            &samples,
            HistogramRegion::All,
            &even,
        ))?;
        rccl_cub::device_histogram_even!(::<$sample, i32, $level, 1, 1>(
            $launch,
            &samples,
            HistogramRegion::All,
            &mut even,
            [0 as $level],
            [4 as $level],
            &mut even_workspace,
        ))?;
        assert_eq!(even.to_host_channels($stream)?[0], [2, 2, 2, 2]);

        let levels_host: [$level; 5] = [
            0 as $level,
            1 as $level,
            2 as $level,
            3 as $level,
            4 as $level,
        ];
        let levels = DeviceHistogramLevels::<$level, 1>::from_channels($stream, [&levels_host])?;
        let mut range = DeviceHistogram::<i32, 1>::new($stream, [4])?;
        let mut range_workspace = rccl_cub::device_histogram_range_workspace!(::<
            $sample,
            i32,
            $level,
            1,
            1,
        >(
            $launch,
            &samples,
            HistogramRegion::All,
            &range,
        ))?;
        rccl_cub::device_histogram_range!(::<$sample, i32, $level, 1, 1>(
            $launch,
            &samples,
            HistogramRegion::All,
            &mut range,
            &levels,
            &mut range_workspace,
        ))?;
        assert_eq!(range.to_host_channels($stream)?[0], [2, 2, 2, 2]);
    }};
}

macro_rules! check_counter_type {
    ($stream:expr, $launch:expr, $counter:ty) => {{
        let samples = DeviceBuffer::from_host($stream, &[0_i32, 1, 1, 2, 2, 2, 3])?;
        let mut histogram = DeviceHistogram::<$counter, 1>::new($stream, [4])?;
        let mut workspace = rccl_cub::device_histogram_even_workspace!(::<
            i32,
            $counter,
            i32,
            1,
            1,
        >(
            $launch,
            &samples,
            HistogramRegion::All,
            &histogram,
        ))?;
        rccl_cub::device_histogram_even!(::<i32, $counter, i32, 1, 1>(
            $launch,
            &samples,
            HistogramRegion::All,
            &mut histogram,
            [0],
            [4],
            &mut workspace,
        ))?;
        let expected: Vec<$counter> = vec![1, 2, 3, 1];
        assert_eq!(histogram.to_host_channels($stream)?[0], expected);
    }};
}

macro_rules! check_channel_config {
    (
        $stream:expr,
        $launch:expr,
        $channels:literal,
        $active:literal,
        $samples_host:expr,
        $expected:expr
    ) => {{
        let samples = DeviceBuffer::from_host($stream, &$samples_host)?;
        let expected: [Vec<i32>; $active] = $expected;

        let mut even = DeviceHistogram::<i32, $active>::new($stream, [4; $active])?;
        let mut even_workspace = rccl_cub::device_histogram_even_workspace!(::<
            i32,
            i32,
            i32,
            $channels,
            $active,
        >(
            $launch,
            &samples,
            HistogramRegion::All,
            &even,
        ))?;
        rccl_cub::device_histogram_even!(::<i32, i32, i32, $channels, $active>(
            $launch,
            &samples,
            HistogramRegion::All,
            &mut even,
            [0; $active],
            [4; $active],
            &mut even_workspace,
        ))?;
        assert_eq!(even.to_host_channels($stream)?, expected);

        let channel_levels = [0_i32, 1, 2, 3, 4];
        let levels = DeviceHistogramLevels::<i32, $active>::from_channels(
            $stream,
            [&channel_levels; $active],
        )?;
        let mut range = DeviceHistogram::<i32, $active>::new($stream, [4; $active])?;
        let mut range_workspace = rccl_cub::device_histogram_range_workspace!(::<
            i32,
            i32,
            i32,
            $channels,
            $active,
        >(
            $launch,
            &samples,
            HistogramRegion::All,
            &range,
        ))?;
        rccl_cub::device_histogram_range!(::<i32, i32, i32, $channels, $active>(
            $launch,
            &samples,
            HistogramRegion::All,
            &mut range,
            &levels,
            &mut range_workspace,
        ))?;
        assert_eq!(range.to_host_channels($stream)?, expected);
    }};
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let cub = rccl_cub::cub_program!(&context; histogram)?;
    let launch = cub.on(&stream)?;

    // Pinned CUB `types` matrix. Integral samples use `int` levels in the
    // basic-use fixture; floating samples keep their own level type.
    check_scalar_histograms!(&stream, &launch, i8, i32);
    check_scalar_histograms!(&stream, &launch, u8, i32);
    check_scalar_histograms!(&stream, &launch, i16, i32);
    check_scalar_histograms!(&stream, &launch, u16, i32);
    check_scalar_histograms!(&stream, &launch, i32, i32);
    check_scalar_histograms!(&stream, &launch, u32, i32);
    check_scalar_histograms!(&stream, &launch, i64, i32);
    check_scalar_histograms!(&stream, &launch, u64, i32);
    check_scalar_histograms!(&stream, &launch, f32, f32);
    check_scalar_histograms!(&stream, &launch, f64, f64);

    // CUDA atomic counter families accepted by pinned DeviceHistogram.
    check_counter_type!(&stream, &launch, i32);
    check_counter_type!(&stream, &launch, u32);
    check_counter_type!(&stream, &launch, i64);
    check_counter_type!(&stream, &launch, u64);

    // Pinned counting-iterator fixture, represented as a finite arithmetic
    // random-access range rather than a pair of C++ cursors.
    let counting = rccl_cub::arithmetic_range(0_i32, 1_i32, (100 + 13) * 4 * 30);
    let counting_region = HistogramRegion::rows(100, 30, (100 + 13) * 4);
    let mut counting_histogram = DeviceHistogram::<i32, 3>::new(&stream, [10, 2, 1])?;
    let mut counting_workspace = rccl_cub::device_histogram_even_arithmetic_workspace!(::<i32, i32, i32, 4, 3>(
        &launch,
        &counting,
        counting_region,
        &counting_histogram,
    ))?;
    rccl_cub::device_histogram_even_arithmetic!(::<i32, i32, i32, 4, 3>(
        &launch,
        &counting,
        counting_region,
        &mut counting_histogram,
        [0, -10, i32::MIN],
        [counting.len() as i32, 10, i32::MAX],
        &mut counting_workspace,
    ))?;
    assert_eq!(
        counting_histogram.to_host_channels(&stream)?,
        [vec![300; 10], vec![0, 3], vec![3_000]]
    );

    let arithmetic_range_samples = rccl_cub::arithmetic_range(0_i32, 1_i32, 9);
    let arithmetic_range_levels =
        DeviceHistogramLevels::<i32, 1>::from_channels(&stream, [&[0_i32, 3, 6, 9]])?;
    let mut arithmetic_range_histogram = DeviceHistogram::<i32, 1>::new(&stream, [3])?;
    let mut arithmetic_range_workspace = rccl_cub::device_histogram_range_arithmetic_workspace!(::<i32, i32, i32, 1, 1>(
        &launch,
        &arithmetic_range_samples,
        HistogramRegion::All,
        &arithmetic_range_histogram,
    ))?;
    rccl_cub::device_histogram_range_arithmetic!(::<i32, i32, i32, 1, 1>(
        &launch,
        &arithmetic_range_samples,
        HistogramRegion::All,
        &mut arithmetic_range_histogram,
        &arithmetic_range_levels,
        &mut arithmetic_range_workspace,
    ))?;
    assert_eq!(
        arithmetic_range_histogram.to_host_channels(&stream)?[0],
        [3, 3, 3]
    );

    // Pinned channel configurations beyond the 1/1 and 4/3 examples below.
    check_channel_config!(
        &stream,
        &launch,
        3,
        3,
        [0_i32, 1, 2, 3, 0, 1],
        [vec![1, 0, 0, 1], vec![1, 1, 0, 0], vec![0, 1, 1, 0]]
    );
    check_channel_config!(
        &stream,
        &launch,
        4,
        4,
        [0_i32, 1, 2, 3, 3, 2, 1, 0],
        [
            vec![1, 0, 0, 1],
            vec![0, 1, 1, 0],
            vec![0, 1, 1, 0],
            vec![1, 0, 0, 1],
        ]
    );

    // Pinned HistogramEven example: [0, 1), ..., [4, 5).
    let even_samples = DeviceBuffer::from_host(&stream, &[0_i32, 2, 1, 0, 3, 4, 2, 1])?;
    let mut even = DeviceHistogram::<i32, 1>::new(&stream, [5])?;
    let mut even_workspace = rccl_cub::device_histogram_even_workspace!(::<
        i32,
        i32,
        i32,
        1,
        1,
    >(
        &launch,
        &even_samples,
        HistogramRegion::All,
        &even,
    ))?;
    rccl_cub::device_histogram_even!(::<i32, i32, i32, 1, 1>(
        &launch,
        &even_samples,
        HistogramRegion::All,
        &mut even,
        [0],
        [5],
        &mut even_workspace,
    ))?;
    assert_eq!(even.to_host_channels(&stream)?[0], [2, 2, 2, 1, 1]);

    // Pinned HistogramRange example.
    let range_samples =
        DeviceBuffer::from_host(&stream, &[2.2_f32, 6.1, 7.5, 2.9, 3.5, 0.3, 2.9, 2.1])?;
    let levels =
        DeviceHistogramLevels::<f32, 1>::from_channels(&stream, [&[0.0, 2.0, 4.0, 6.0, 8.0]])?;
    let mut range = DeviceHistogram::<i32, 1>::new(&stream, [4])?;
    let mut range_workspace = rccl_cub::device_histogram_range_workspace!(::<
        f32,
        i32,
        f32,
        1,
        1,
    >(
        &launch,
        &range_samples,
        HistogramRegion::All,
        &range,
    ))?;
    rccl_cub::device_histogram_range!(::<f32, i32, f32, 1, 1>(
        &launch,
        &range_samples,
        HistogramRegion::All,
        &mut range,
        &levels,
        &mut range_workspace,
    ))?;
    assert_eq!(range.to_host_channels(&stream)?[0], [1, 5, 0, 2]);

    // Pinned RGBA -> RGB MultiHistogramEven example.
    let rgba = DeviceBuffer::from_host(&stream, &[0_u8, 2, 1, 255, 3, 4, 2, 128])?;
    let mut rgb = DeviceHistogram::<i32, 3>::new(&stream, [4, 4, 4])?;
    let mut rgb_workspace = rccl_cub::device_histogram_even_workspace!(::<
        u8,
        i32,
        u8,
        4,
        3,
    >(
        &launch,
        &rgba,
        HistogramRegion::All,
        &rgb,
    ))?;
    rccl_cub::device_histogram_even!(::<u8, i32, u8, 4, 3>(
        &launch,
        &rgba,
        HistogramRegion::All,
        &mut rgb,
        [0, 0, 0],
        [4, 4, 4],
        &mut rgb_workspace,
    ))?;
    assert_eq!(
        rgb.to_host_channels(&stream)?,
        [vec![1, 0, 0, 1], vec![0, 0, 1, 0], vec![0, 1, 1, 0]]
    );

    // Pinned RGBA -> RGB MultiHistogramRange example, including unequal
    // channel level counts in one affine level allocation.
    let rgb_levels = DeviceHistogramLevels::<u8, 3>::from_channels(
        &stream,
        [&[0, 2, 4], &[0, 3, 5], &[0, 1, 2, 3]],
    )?;
    let mut rgb_range = DeviceHistogram::<i32, 3>::new(&stream, [2, 2, 3])?;
    let mut rgb_range_workspace = rccl_cub::device_histogram_range_workspace!(::<
        u8,
        i32,
        u8,
        4,
        3,
    >(
        &launch,
        &rgba,
        HistogramRegion::All,
        &rgb_range,
    ))?;
    rccl_cub::device_histogram_range!(::<u8, i32, u8, 4, 3>(
        &launch,
        &rgba,
        HistogramRegion::All,
        &mut rgb_range,
        &rgb_levels,
        &mut rgb_range_workspace,
    ))?;
    assert_eq!(
        rgb_range.to_host_channels(&stream)?,
        [vec![1, 1], vec![1, 1], vec![0, 1, 1]]
    );

    // Pinned 2D ROI example: one padding sample after each active row.
    let pitched = DeviceBuffer::from_host(&stream, &[0_i32, 1, 2, -1, 1, 2, 0, -1])?;
    let mut rows = DeviceHistogram::<i32, 1>::new(&stream, [3])?;
    let mut rows_workspace = rccl_cub::device_histogram_even_workspace!(::<
        i32,
        i32,
        i32,
        1,
        1,
    >(
        &launch,
        &pitched,
        HistogramRegion::rows(3, 2, 4),
        &rows,
    ))?;
    rccl_cub::device_histogram_even!(::<i32, i32, i32, 1, 1>(
        &launch,
        &pitched,
        HistogramRegion::rows(3, 2, 4),
        &mut rows,
        [0],
        [3],
        &mut rows_workspace,
    ))?;
    assert_eq!(rows.to_host_channels(&stream)?[0], [2, 2, 2]);

    // Pinned 2D RGBA examples: two active pixels and one padding pixel per
    // row. Both multi-channel binning modes must ignore padding.
    let rgba_rows = DeviceBuffer::from_host(
        &stream,
        &[
            0_u8, 2, 1, 255, 3, 4, 2, 128, 0, 0, 0, 0, 1, 1, 3, 200, 2, 3, 0, 100, 0, 0, 0, 0,
        ],
    )?;
    let rgba_region = HistogramRegion::rows(2, 2, 12);
    let mut rgb_rows = DeviceHistogram::<i32, 3>::new(&stream, [4, 4, 4])?;
    let mut rgb_rows_workspace = rccl_cub::device_histogram_even_workspace!(::<
        u8,
        i32,
        u8,
        4,
        3,
    >(
        &launch,
        &rgba_rows,
        rgba_region,
        &rgb_rows,
    ))?;
    rccl_cub::device_histogram_even!(::<u8, i32, u8, 4, 3>(
        &launch,
        &rgba_rows,
        rgba_region,
        &mut rgb_rows,
        [0, 0, 0],
        [4, 4, 4],
        &mut rgb_rows_workspace,
    ))?;
    assert_eq!(
        rgb_rows.to_host_channels(&stream)?,
        [vec![1, 1, 1, 1], vec![0, 1, 1, 1], vec![1, 1, 1, 1]]
    );

    let mut rgb_range_rows = DeviceHistogram::<i32, 3>::new(&stream, [2, 2, 3])?;
    let mut rgb_range_rows_workspace = rccl_cub::device_histogram_range_workspace!(::<
        u8,
        i32,
        u8,
        4,
        3,
    >(
        &launch,
        &rgba_rows,
        rgba_region,
        &rgb_range_rows,
    ))?;
    rccl_cub::device_histogram_range!(::<u8, i32, u8, 4, 3>(
        &launch,
        &rgba_rows,
        rgba_region,
        &mut rgb_range_rows,
        &rgb_levels,
        &mut rgb_range_rows_workspace,
    ))?;
    assert_eq!(
        rgb_range_rows.to_host_channels(&stream)?,
        [vec![2, 2], vec![2, 2], vec![1, 1, 1]]
    );

    // Pinned NVIDIA/cub#489 integer-bin rounding regression.
    let regression_samples =
        DeviceBuffer::from_host(&stream, &[2_i32, 6, 7, 2, 3, 0, 2, 2, 6, 999])?;
    let mut regression = DeviceHistogram::<i32, 1>::new(&stream, [7])?;
    let mut regression_workspace = rccl_cub::device_histogram_even_workspace!(::<
        i32,
        i32,
        i32,
        1,
        1,
    >(
        &launch,
        &regression_samples,
        HistogramRegion::All,
        &regression,
    ))?;
    rccl_cub::device_histogram_even!(::<i32, i32, i32, 1, 1>(
        &launch,
        &regression_samples,
        HistogramRegion::All,
        &mut regression,
        [0],
        [12],
        &mut regression_workspace,
    ))?;
    assert_eq!(
        regression.to_host_channels(&stream)?[0],
        [1, 5, 0, 2, 1, 0, 0]
    );

    // Pinned range-level/sample aliasing test, expressed as a lifetime-bound
    // device view rather than unrelated C++ pointers.
    let alias_samples = DeviceBuffer::from_host(
        &stream,
        &[
            0_i32, 2, 4, 6, 8, 10, 12, 1, 3, 3, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9, 9, 11, 11, 11, 11,
            11, 11,
        ],
    )?;
    let alias_levels =
        DeviceHistogramLevelsView::from_ranges(&alias_samples, core::array::from_fn(|_| 0..7))?;
    let mut alias_histogram = DeviceHistogram::<i32, 1>::new(&stream, [6])?;
    let mut alias_workspace = rccl_cub::device_histogram_range_workspace!(::<
        i32,
        i32,
        i32,
        1,
        1,
    >(
        &launch,
        &alias_samples,
        HistogramRegion::All,
        &alias_histogram,
    ))?;
    rccl_cub::device_histogram_range!(::<i32, i32, i32, 1, 1>(
        &launch,
        &alias_samples,
        HistogramRegion::All,
        &mut alias_histogram,
        &alias_levels,
        &mut alias_workspace,
    ))?;
    assert_eq!(
        alias_histogram.to_host_channels(&stream)?[0],
        [2, 3, 4, 5, 6, 7]
    );

    // More than 256 bins selects CUB's global privatization branch.
    let wide_samples_host: Vec<i32> = (0..1024).collect();
    let wide_samples = DeviceBuffer::from_host(&stream, &wide_samples_host)?;
    let mut wide = DeviceHistogram::<i32, 1>::new(&stream, [2048])?;
    let mut wide_workspace = rccl_cub::device_histogram_even_workspace!(::<
        i32,
        i32,
        i32,
        1,
        1,
    >(
        &launch,
        &wide_samples,
        HistogramRegion::All,
        &wide,
    ))?;
    rccl_cub::device_histogram_even!(::<i32, i32, i32, 1, 1>(
        &launch,
        &wide_samples,
        HistogramRegion::All,
        &mut wide,
        [0],
        [2048],
        &mut wide_workspace,
    ))?;
    let wide_host = wide.to_host_channels(&stream)?;
    assert!(wide_host[0][..1024].iter().all(|&count| count == 1));
    assert!(wide_host[0][1024..].iter().all(|&count| count == 0));

    // Pinned odd-size and entropy cases. The non-tile-aligned extent catches
    // validity mistakes; the two distributions exercise RLE compression
    // without permitting either case to reuse the other's expected result.
    let low_entropy_host = vec![0_u8; 10_003];
    let high_entropy_host = (0..10_003)
        .map(|index| (index & 255) as u8)
        .collect::<Vec<_>>();
    for (host, low_entropy) in [
        (low_entropy_host.as_slice(), true),
        (high_entropy_host.as_slice(), false),
    ] {
        let samples = DeviceBuffer::from_host(&stream, host)?;
        let mut histogram = DeviceHistogram::<i32, 1>::new(&stream, [256])?;
        let mut workspace = rccl_cub::device_histogram_even_workspace!(::<
            u8,
            i32,
            i32,
            1,
            1,
        >(
            &launch,
            &samples,
            HistogramRegion::All,
            &histogram,
        ))?;
        rccl_cub::device_histogram_even!(::<u8, i32, i32, 1, 1>(
            &launch,
            &samples,
            HistogramRegion::All,
            &mut histogram,
            [0],
            [256],
            &mut workspace,
        ))?;
        let counts = histogram.to_host_channels(&stream)?;
        if low_entropy {
            assert_eq!(counts[0][0], 10_003);
            assert!(counts[0][1..].iter().all(|&count| count == 0));
        } else {
            assert_eq!(counts[0].iter().sum::<i32>(), 10_003);
            assert!(counts[0].iter().all(|&count| count == 39 || count == 40));
        }
    }

    let empty = DeviceBuffer::<i32>::zeroed(&stream, 0)?;
    let mut empty_histogram = DeviceHistogram::<i32, 3>::new(&stream, [4, 4, 4])?;
    let mut empty_workspace = rccl_cub::device_histogram_even_workspace!(::<
        i32,
        i32,
        i32,
        4,
        3,
    >(
        &launch,
        &empty,
        HistogramRegion::rows(1, 0, 4),
        &empty_histogram,
    ))?;
    rccl_cub::device_histogram_even!(::<i32, i32, i32, 4, 3>(
        &launch,
        &empty,
        HistogramRegion::rows(1, 0, 4),
        &mut empty_histogram,
        [0, 0, 0],
        [4, 4, 4],
        &mut empty_workspace,
    ))?;
    assert!(
        empty_histogram
            .to_host_channels(&stream)?
            .iter()
            .flatten()
            .all(|&count| count == 0)
    );

    // Negative paths are deliberate fault injections: they prove the launch
    // wrappers surface invalid TMP/runtime combinations instead of masking
    // them to preserve the positive fixture results.
    let invalid_bounds = rccl_cub::device_histogram_even!(::<i32, i32, i32, 1, 1>(
        &launch,
        &even_samples,
        HistogramRegion::All,
        &mut even,
        [5],
        [0],
        &mut even_workspace,
    ));
    assert!(matches!(
        invalid_bounds,
        Err(DeviceAlgorithmError::InvalidHistogramShape(_))
    ));

    let too_few_levels = DeviceHistogramLevels::<f32, 1>::from_channels(&stream, [&[0.0, 8.0]])?;
    let invalid_levels = rccl_cub::device_histogram_range!(::<f32, i32, f32, 1, 1>(
        &launch,
        &range_samples,
        HistogramRegion::All,
        &mut range,
        &too_few_levels,
        &mut range_workspace,
    ));
    assert!(matches!(
        invalid_levels,
        Err(DeviceAlgorithmError::LengthMismatch { .. })
    ));

    let mismatched_region = rccl_cub::device_histogram_even!(::<i32, i32, i32, 1, 1>(
        &launch,
        &even_samples,
        HistogramRegion::rows(3, 2, 5),
        &mut even,
        [0],
        [5],
        &mut even_workspace,
    ));
    assert!(matches!(
        mismatched_region,
        Err(DeviceAlgorithmError::InvalidHistogramShape(_))
    ));

    println!("pinned DeviceHistogram examples, regressions, and fault paths: PASS");
    Ok(())
}
