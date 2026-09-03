use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cub::{DeviceHistogram, HistogramRegion};
use std::error::Error;

macro_rules! check_counter_type {
    ($stream:expr, $launch:expr, $counter:ty) => {{
        let samples = DeviceBuffer::from_host($stream, &[0_i32, 1, 1, 2, 2, 2, 3])?;
        let mut histogram = DeviceHistogram::<$counter, 1>::new($stream, [4])?;
        let mut workspace = rccl_cub::device_histogram_even_workspace::<
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
        )?;
        rccl_cub::device_histogram_even::<i32, $counter, i32, 1, 1>(
            $launch,
            &samples,
            HistogramRegion::All,
            &mut histogram,
            [0],
            [4],
            &mut workspace,
        )?;
        let expected: Vec<$counter> = vec![1, 2, 3, 1];
        assert_eq!(histogram.to_host_channels($stream)?[0], expected);
    }};
}

macro_rules! histogram_counter_probe {
    ($name:literal; $($counter:ty),+ $(,)?) => {
        fn main() -> Result<(), Box<dyn Error>> {
            let context = CudaContext::new(0)?;
            let stream = context.default_stream();
            let cub = rccl_cub::cuda::Program::load(&context)?;
            let launch = cub.on(&stream)?;

            $(check_counter_type!(&stream, &launch, $counter);)+

            println!("{name}: PASS", name = $name);
            Ok(())
        }
    };
}
