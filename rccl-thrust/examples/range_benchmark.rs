use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{arithmetic_range, for_each_async, indexed_output, transform_into_async};
use std::error::Error;

const ITEMS: usize = 8 * 1024 * 1024;
const WARMUPS: usize = 20;
const SAMPLES: usize = 21;
const LAUNCHES: usize = 100;
const MASK: u32 = 0x5a5a_1234;

fn median_timings(mut timings: Vec<f32>) -> (f32, f32) {
    timings.sort_by(f32::total_cmp);
    (timings[0], timings[timings.len() / 2])
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_thrust::cuda::Program::load(&context)?;
    let execution = program.par(&stream)?;
    let output = DeviceBuffer::<u32>::zeroed(&stream, ITEMS)?;
    let output_ptr = output.cu_deviceptr() as usize as *mut u32;
    for _ in 0..WARMUPS {
        transform_into_async(
            &execution,
            arithmetic_range(0_u32, 1_u32, ITEMS),
            indexed_output(ITEMS, move |index, value| unsafe {
                output_ptr.add(index).write(value)
            }),
            |value| value.wrapping_mul(17).wrapping_add(3) ^ MASK,
        )
        .wait()?;
    }

    let timing = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let mut transform_timings = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = stream.record_event(timing)?;
        let mut pending = Vec::with_capacity(LAUNCHES);
        for _ in 0..LAUNCHES {
            pending.push(transform_into_async(
                &execution,
                arithmetic_range(0_u32, 1_u32, ITEMS),
                indexed_output(ITEMS, move |index, value| unsafe {
                    output_ptr.add(index).write(value)
                }),
                |value| value.wrapping_mul(17).wrapping_add(3) ^ MASK,
            ));
        }
        let end = stream.record_event(timing)?;
        end.synchronize()?;
        transform_timings.push(start.elapsed_ms(&end)? / LAUNCHES as f32);
        for completion in pending {
            completion.wait()?;
        }
    }
    let expected_last = ((ITEMS as u32 - 1).wrapping_mul(17).wrapping_add(3)) ^ MASK;
    assert_eq!(output.to_host_vec(&stream)?[ITEMS - 1], expected_last);

    for _ in 0..WARMUPS {
        for_each_async(
            &execution,
            arithmetic_range(0_u32, 1_u32, ITEMS),
            move |index| unsafe {
                output_ptr
                    .add(index as usize)
                    .write(index.wrapping_mul(17).wrapping_add(3) ^ MASK)
            },
        )
        .wait()?;
    }

    let mut for_each_timings = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = stream.record_event(timing)?;
        let mut pending = Vec::with_capacity(LAUNCHES);
        for _ in 0..LAUNCHES {
            pending.push(for_each_async(
                &execution,
                arithmetic_range(0_u32, 1_u32, ITEMS),
                move |index| unsafe {
                    output_ptr
                        .add(index as usize)
                        .write(index.wrapping_mul(17).wrapping_add(3) ^ MASK)
                },
            ));
        }
        let end = stream.record_event(timing)?;
        end.synchronize()?;
        for_each_timings.push(start.elapsed_ms(&end)? / LAUNCHES as f32);
        for completion in pending {
            completion.wait()?;
        }
    }
    assert_eq!(output.to_host_vec(&stream)?[ITEMS - 1], expected_last);

    let (transform_best, transform_median) = median_timings(transform_timings);
    let (for_each_best, for_each_median) = median_timings(for_each_timings);
    println!(
        "implementation=rccl algorithm=transform adaptor=arithmetic type=u32 items={ITEMS} best_ms={transform_best:.6} median_ms={transform_median:.6}"
    );
    println!(
        "implementation=rccl algorithm=for_each adaptor=arithmetic type=u32 items={ITEMS} best_ms={for_each_best:.6} median_ms={for_each_median:.6}"
    );
    Ok(())
}
