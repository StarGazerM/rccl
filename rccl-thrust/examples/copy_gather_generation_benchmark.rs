use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{
    DeviceBufferExt, copy_async, fill_range_async, gather_async, tabulate, tabulate_range_async,
};
use std::error::Error;

const ITEMS: usize = 16 * 1024 * 1024;
const WARMUPS: usize = 10;
const SAMPLES: usize = 31;
const MASK: u32 = 0x5a5a_1234;

fn summarize(mut values: Vec<f32>) -> (f32, f32) {
    values.sort_by(f32::total_cmp);
    (values[0], values[values.len() / 2])
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_thrust::cuda::Program::load(&context)?;
    let execution = program.par(&stream)?;
    let timing = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);

    let mut input = DeviceBuffer::<u32>::zeroed(&stream, ITEMS)?;
    let mut indices = DeviceBuffer::<u32>::zeroed(&stream, ITEMS)?;
    let mut output = DeviceBuffer::<u32>::zeroed(&stream, ITEMS)?;
    tabulate(&execution, &mut input, |index| index as u32 ^ MASK)?;
    tabulate(&execution, &mut indices, |index| {
        (index as u32).wrapping_mul(65_537) & (ITEMS as u32 - 1)
    })?;

    for _ in 0..WARMUPS {
        copy_async(&execution, &input, &mut output).wait()?;
    }
    let mut copy_values = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = stream.record_event(timing)?;
        let pending = copy_async(&execution, &input, &mut output);
        let end = stream.record_event(timing)?;
        end.synchronize()?;
        copy_values.push(start.elapsed_ms(&end)?);
        pending.wait()?;
    }
    let copied = output.to_host_vec(&stream)?;
    assert_eq!(copied[0], MASK);
    assert_eq!(copied[ITEMS - 1], ITEMS as u32 - 1 ^ MASK);

    for _ in 0..WARMUPS {
        gather_async(&execution, input.view(), indices.view(), output.view_mut()).wait()?;
    }
    let mut gather_values = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = stream.record_event(timing)?;
        let pending = gather_async(&execution, input.view(), indices.view(), output.view_mut());
        let end = stream.record_event(timing)?;
        end.synchronize()?;
        gather_values.push(start.elapsed_ms(&end)?);
        pending.wait()?;
    }
    let gathered = output.to_host_vec(&stream)?;
    let last_index = ((ITEMS - 1) as u32).wrapping_mul(65_537) & (ITEMS as u32 - 1);
    assert_eq!(gathered[0], MASK);
    assert_eq!(gathered[ITEMS - 1], last_index ^ MASK);

    for _ in 0..WARMUPS {
        fill_range_async(&execution, output.view_mut(), 0xdead_beef_u32).wait()?;
    }
    let mut fill_values = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = stream.record_event(timing)?;
        let pending = fill_range_async(&execution, output.view_mut(), 0xdead_beef_u32);
        let end = stream.record_event(timing)?;
        end.synchronize()?;
        fill_values.push(start.elapsed_ms(&end)?);
        pending.wait()?;
    }
    assert!(
        output
            .to_host_vec(&stream)?
            .into_iter()
            .all(|value| value == 0xdead_beef)
    );

    for _ in 0..WARMUPS {
        tabulate_range_async(&execution, output.view_mut(), |index| {
            7_u32.wrapping_add(3_u32.wrapping_mul(index as u32))
        })
        .wait()?;
    }
    let mut sequence_values = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = stream.record_event(timing)?;
        let pending = tabulate_range_async(&execution, output.view_mut(), |index| {
            7_u32.wrapping_add(3_u32.wrapping_mul(index as u32))
        });
        let end = stream.record_event(timing)?;
        end.synchronize()?;
        sequence_values.push(start.elapsed_ms(&end)?);
        pending.wait()?;
    }
    let sequenced = output.to_host_vec(&stream)?;
    assert_eq!(sequenced[0], 7);
    assert_eq!(
        sequenced[ITEMS - 1],
        7_u32.wrapping_add(3_u32.wrapping_mul((ITEMS - 1) as u32))
    );

    for (algorithm, values) in [
        ("copy", copy_values),
        ("gather", gather_values),
        ("fill", fill_values),
        ("sequence", sequence_values),
    ] {
        let (best, median) = summarize(values);
        println!(
            "implementation=rccl algorithm={algorithm} type=u32 items={ITEMS} best_ms={best:.6} median_ms={median:.6}"
        );
    }
    Ok(())
}
