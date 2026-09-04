use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cub::{DeviceAlgorithmError, DeviceSliceRead, DeviceSliceWrite};
use std::error::Error;

const ITEMS: usize = (1 << 20) + 13;
const WARMUPS: usize = 10;
const SAMPLES: usize = 21;
const LAUNCHES: usize = 10;

fn read<T>(buffer: &DeviceBuffer<T>) -> DeviceSliceRead<T> {
    DeviceSliceRead::new(buffer.cu_deviceptr() as usize as *const T)
}

fn write<T>(buffer: &mut DeviceBuffer<T>) -> DeviceSliceWrite<T> {
    DeviceSliceWrite::new(buffer.cu_deviceptr() as usize as *mut T)
}

fn measure(
    stream: &cuda_core::CudaStream,
    mut operation: impl FnMut() -> Result<(), DeviceAlgorithmError>,
) -> Result<(f32, f32), Box<dyn Error>> {
    for _ in 0..WARMUPS {
        for _ in 0..LAUNCHES {
            operation()?;
        }
    }
    stream.synchronize()?;
    let timing = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = stream.record_event(timing)?;
        for _ in 0..LAUNCHES {
            operation()?;
        }
        let end = stream.record_event(timing)?;
        end.synchronize()?;
        samples.push(start.elapsed_ms(&end)? / LAUNCHES as f32);
    }
    samples.sort_by(f32::total_cmp);
    Ok((samples[0], samples[SAMPLES / 2]))
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_cub::cuda::Program::load(&context)?;
    let launch = program.on(&stream)?;
    let allocator = rccl_memory::BorrowedCudaMemoryResource::new(&context)?;

    let keys_host = (0..ITEMS)
        .map(|index| (index / 3) as u32)
        .collect::<Vec<_>>();
    let values_host = (0..ITEMS)
        .map(|index| (index % 17 + 1) as u64)
        .collect::<Vec<_>>();
    let keys = DeviceBuffer::from_host(&stream, &keys_host)?;
    let values = DeviceBuffer::from_host(&stream, &values_host)?;
    let mut output_keys = DeviceBuffer::from_host(&stream, &vec![0_u32; ITEMS])?;
    let mut output_values = DeviceBuffer::from_host(&stream, &vec![0_u64; ITEMS])?;
    let mut count = DeviceBuffer::from_host(&stream, &[0_u64])?;
    let mut workspace =
        rccl_cub::thrust_integration::grouped_reduction_workspace_in(&allocator, &launch, ITEMS)?;
    let bytes = workspace.allocation_bytes();

    let timing = measure(&stream, || {
        rccl_cub::thrust_integration::device_reduce_by_key(
            &launch,
            read(&keys),
            read(&values),
            write(&mut output_keys),
            write(&mut output_values),
            &mut count,
            |left, right| left == right,
            rccl_cub::thrust_integration::NaturalBagAdd,
            ITEMS,
            &mut workspace,
        )
    })?;
    println!(
        "implementation=rccl algorithm=reduce_by_key type=u32-u64 items={ITEMS} threads=256 launches={LAUNCHES} best_ms={:.6} median_ms={:.6} temp_bytes={bytes}",
        timing.0, timing.1
    );

    stream.synchronize()?;
    let groups = count.to_host_vec(&stream)?[0] as usize;
    assert_eq!(groups, ITEMS.div_ceil(3));
    let actual_keys = output_keys.to_host_vec(&stream)?;
    let actual_values = output_values.to_host_vec(&stream)?;
    for group in 0..groups {
        let begin = group * 3;
        let end = (begin + 3).min(ITEMS);
        assert_eq!(actual_keys[group], group as u32);
        let expected = values_host[begin..end]
            .iter()
            .copied()
            .fold(0_u64, u64::wrapping_add);
        assert_eq!(actual_values[group], expected);
    }
    Ok(())
}
