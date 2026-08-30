//! Equivalent-work benchmark for the pinned Thrust constant-iterator example.

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::repeat_n;
use std::error::Error;

const ITEMS: usize = 8 * 1024 * 1024;
const WARMUPS: usize = 20;
const SAMPLES: usize = 21;
const LAUNCHES: usize = 100;
const ADD: u32 = 10;

fn input_value(index: usize) -> u32 {
    ((index * 17 + 3) % 251) as u32
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let thrust = rccl_thrust::thrust_program!(&context)?;
    let exec = thrust.par(&stream)?;

    let host_input = (0..ITEMS).map(input_value).collect::<Vec<_>>();
    let mut data = DeviceBuffer::from_host(&stream, &host_input)?;
    let repeated = repeat_n(ADD, ITEMS);

    for _ in 0..WARMUPS {
        rccl_thrust::transform!(&exec, &mut data, repeated, |left, right| left + right)?;
    }
    stream.synchronize()?;

    let timing = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let mut timings = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = stream.record_event(timing)?;
        for _ in 0..LAUNCHES {
            rccl_thrust::transform!(&exec, &mut data, repeated, |left, right| left + right)?;
        }
        let end = stream.record_event(timing)?;
        end.synchronize()?;
        timings.push(start.elapsed_ms(&end)? / LAUNCHES as f32);
    }

    let increment = ADD * (WARMUPS + SAMPLES * LAUNCHES) as u32;
    for (index, result) in data.to_host_vec(&stream)?.into_iter().enumerate() {
        assert_eq!(
            result,
            input_value(index) + increment,
            "transform mismatch at {index}"
        );
    }

    timings.sort_by(f32::total_cmp);
    println!(
        "implementation=rccl algorithm=transform adaptor=constant_iterator type=u32 items={ITEMS} best_ms={:.6} median_ms={:.6}",
        timings[0],
        timings[timings.len() / 2],
    );
    Ok(())
}
