use cuda_core::{CudaContext, DeviceBuffer};
use std::{cmp::Ordering, error::Error};

const ITEMS: usize = (1 << 20) + 13;
const WARMUPS: usize = 10;
const SAMPLES: usize = 21;
const LAUNCHES: usize = 10;
const TEMP_BYTES: usize = 25_174_344;

type Row4 = [u32; 4];

#[inline(always)]
fn ascending(left: &Row4, right: &Row4) -> Ordering {
    macro_rules! compare_field {
        ($field:expr) => {
            if left[$field] < right[$field] {
                return Ordering::Less;
            }
            if left[$field] > right[$field] {
                return Ordering::Greater;
            }
        };
    }

    // Match the pinned CUB comparator's fully unrolled four-field body.
    compare_field!(0);
    compare_field!(1);
    compare_field!(2);
    compare_field!(3);
    Ordering::Equal
}

fn make_row(index: usize) -> Row4 {
    let mut row = [0_u32; 4];
    for (field, value) in row.iter_mut().enumerate() {
        *value = ((index * (field * 6 + 5) + field * 3) % (11 + field * 2)) as u32;
    }
    row
}

fn measure(
    stream: &cuda_core::CudaStream,
    mut operation: impl FnMut() -> Result<(), rccl_thrust::DeviceAlgorithmError>,
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

    let rows_host = (0..ITEMS).map(make_row).collect::<Vec<_>>();
    let payloads_host = (0..ITEMS as u64).collect::<Vec<_>>();
    let source_rows = DeviceBuffer::from_host(&stream, &rows_host)?;
    let source_payloads = DeviceBuffer::from_host(&stream, &payloads_host)?;
    let mut rows = DeviceBuffer::from_host(&stream, &rows_host)?;
    let mut payloads = DeviceBuffer::from_host(&stream, &payloads_host)?;
    let mut workspace =
        rccl_cub::device_merge_sort_pairs_workspace::<Row4, u64>(&launch, ITEMS)?;
    let partitions = rccl_cub::thrust_integration::device_merge_sort_partition_count::<Row4>(ITEMS)?;
    let actual_temp_bytes = ITEMS * (core::mem::size_of::<Row4>() + core::mem::size_of::<u64>())
        + partitions * core::mem::size_of::<usize>();
    assert_eq!(actual_temp_bytes, TEMP_BYTES);

    let timing = measure(&stream, || {
        rccl_thrust::memory::copy_device_range::<Row4, _, _>(&source_rows, &mut rows, &stream)
            .map_err(rccl_thrust::DeviceAlgorithmError::Memory)?;
        rccl_thrust::memory::copy_device_range::<u64, _, _>(
            &source_payloads,
            &mut payloads,
            &stream,
        )
        .map_err(rccl_thrust::DeviceAlgorithmError::Memory)?;
        rccl_cub::device_merge_sort_pairs(
            &launch,
            &mut rows,
            &mut payloads,
            &mut workspace,
            ascending,
        )
    })?;
    println!(
        "implementation=rccl algorithm=stable_record_sort type=row4-u64 items={ITEMS} threads=256 launches={LAUNCHES} best_ms={:.6} median_ms={:.6} temp_bytes={TEMP_BYTES}",
        timing.0, timing.1
    );

    let mut expected = rows_host
        .iter()
        .copied()
        .zip(payloads_host.iter().copied())
        .collect::<Vec<_>>();
    expected.sort_by_key(|pair| pair.0);
    let actual = rows
        .to_host_vec(&stream)?
        .into_iter()
        .zip(payloads.to_host_vec(&stream)?)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    Ok(())
}
