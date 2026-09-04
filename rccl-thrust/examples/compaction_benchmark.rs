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

fn report(algorithm: &str, timing: (f32, f32), bytes: usize) {
    println!(
        "implementation=rccl algorithm={algorithm} type=u32 items={ITEMS} threads=256 launches={LAUNCHES} best_ms={:.6} median_ms={:.6} temp_bytes={bytes}",
        timing.0, timing.1
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_cub::cuda::Program::load(&context)?;
    let launch = program.on(&stream)?;
    let allocator = rccl_memory::BorrowedCudaMemoryResource::new(&context)?;
    let input_host = (0..ITEMS).map(|index| index as u32).collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&stream, &input_host)?;
    let flags_host = (0..ITEMS)
        .map(|index| u32::from(index % 5 <= 1))
        .collect::<Vec<_>>();
    let flags = DeviceBuffer::from_host(&stream, &flags_host)?;
    let unique_host = (0..ITEMS)
        .map(|index| (index / 3) as u32)
        .collect::<Vec<_>>();
    let unique_input = DeviceBuffer::from_host(&stream, &unique_host)?;
    let mut output = DeviceBuffer::from_host(&stream, &vec![0_u32; ITEMS])?;
    let mut count = DeviceBuffer::from_host(&stream, &[0_u64])?;
    let mut workspace =
        rccl_cub::thrust_integration::compaction_workspace_in(&allocator, &launch, ITEMS)?;
    let bytes = workspace.allocation_bytes();

    let timing = measure(&stream, || {
        rccl_cub::thrust_integration::device_select_if(
            &launch,
            read(&input),
            write(&mut output),
            &mut count,
            |value| value % 5 <= 1,
            ITEMS,
            &mut workspace,
        )
    })?;
    report("select_if", timing, bytes);

    let timing = measure(&stream, || {
        rccl_cub::thrust_integration::device_select_flagged(
            &launch,
            read(&input),
            read(&flags),
            write(&mut output),
            &mut count,
            ITEMS,
            &mut workspace,
        )
    })?;
    report("select_flagged", timing, bytes);

    let timing = measure(&stream, || {
        rccl_cub::thrust_integration::device_unique_copy_by(
            &launch,
            read(&unique_input),
            write(&mut output),
            &mut count,
            |left, right| left == right,
            ITEMS,
            &mut workspace,
        )
    })?;
    report("unique_copy", timing, bytes);

    let timing = measure(&stream, || {
        rccl_cub::thrust_integration::device_stable_partition_by(
            &launch,
            read(&input),
            write(&mut output),
            &mut count,
            |value| value % 5 <= 1,
            ITEMS,
            &mut workspace,
        )
    })?;
    report("stable_partition", timing, bytes);

    stream.synchronize()?;
    let selected = count.to_host_vec(&stream)?[0] as usize;
    let expected = input_host
        .iter()
        .copied()
        .filter(|value| value % 5 <= 1)
        .chain(input_host.iter().copied().filter(|value| value % 5 > 1))
        .collect::<Vec<_>>();
    assert_eq!(
        selected,
        flags_host.iter().filter(|&&flag| flag != 0).count()
    );
    assert_eq!(output.to_host_vec(&stream)?, expected);
    Ok(())
}
