use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cub::{DeviceHistogram, DeviceHistogramLevels, HistogramRegion};
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
        let mut even_workspace = rccl_cub::device_histogram_even_workspace::<
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
        )?;
        rccl_cub::device_histogram_even::<$sample, i32, $level, 1, 1>(
            $launch,
            &samples,
            HistogramRegion::All,
            &mut even,
            [0 as $level],
            [4 as $level],
            &mut even_workspace,
        )?;
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
        let mut range_workspace = rccl_cub::device_histogram_range_workspace::<
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
        )?;
        rccl_cub::device_histogram_range::<$sample, i32, $level, 1, 1>(
            $launch,
            &samples,
            HistogramRegion::All,
            &mut range,
            &levels,
            &mut range_workspace,
        )?;
        assert_eq!(range.to_host_channels($stream)?[0], [2, 2, 2, 2]);
    }};
}

macro_rules! histogram_scalar_probe {
    ($name:literal; $(($sample:ty, $level:ty)),+ $(,)?) => {
        fn main() -> Result<(), Box<dyn Error>> {
            let context = CudaContext::new(0)?;
            let stream = context.default_stream();
            let cub = rccl_cub::cuda::Program::load(&context)?;
            let launch = cub.on(&stream)?;

            $(check_scalar_histograms!(&stream, &launch, $sample, $level);)+

            println!("{name}: PASS", name = $name);
            Ok(())
        }
    };
}
