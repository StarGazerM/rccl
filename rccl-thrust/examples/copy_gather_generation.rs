use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{
    DeviceBufferExt, DeviceRangeExt, copy, copy_async, copy_range, fill_range, gather,
    sequence_range_by, tabulate_range_async,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_thrust::cuda::Program::load(&context)?;
    let execution = program.par(&stream)?;

    if std::env::args().any(|argument| argument == "--index-oob") {
        let input = DeviceBuffer::from_host(&stream, &[10_u32, 20, 30])?;
        let indices = DeviceBuffer::from_host(&stream, &[3_u32])?;
        let mut output = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        if let Err(error) = gather(&execution, input.view(), indices.view(), output.view_mut()) {
            eprintln!("gather index OOB surfaced: {error}");
            std::process::exit(0);
        }
        panic!("out-of-bounds gather unexpectedly completed");
    }

    let input = DeviceBuffer::from_host(&stream, &[11_u32, 22, 33, 44])?;
    let mut contiguous = DeviceBuffer::<u32>::zeroed(&stream, input.len())?;
    copy_async(&execution, &input, &mut contiguous).wait()?;
    assert_eq!(contiguous.to_host_vec(&stream)?, [11, 22, 33, 44]);

    let mut wrong_length = DeviceBuffer::<u32>::zeroed(&stream, input.len() - 1)?;
    assert!(copy(&execution, &input, &mut wrong_length).is_err());

    let mut sliced = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    copy_range(&execution, input.view().slice(1..3)?, sliced.view_mut())?;
    assert_eq!(sliced.to_host_vec(&stream)?, [22, 33]);

    let indices32 = DeviceBuffer::from_host(&stream, &[3_u32, 0, 3, 1])?;
    let mut gathered32 = DeviceBuffer::<u32>::zeroed(&stream, indices32.len())?;
    gather(
        &execution,
        input.view(),
        indices32.view(),
        gathered32.view_mut(),
    )?;
    assert_eq!(gathered32.to_host_vec(&stream)?, [44, 11, 44, 22]);

    let mut wrong_gather_length = DeviceBuffer::<u32>::zeroed(&stream, indices32.len() - 1)?;
    assert!(
        gather(
            &execution,
            input.view(),
            indices32.view(),
            wrong_gather_length.view_mut(),
        )
        .is_err()
    );

    let indices64 = DeviceBuffer::from_host(&stream, &[2_u64, 1, 0])?;
    let mut gathered64 = DeviceBuffer::<u32>::zeroed(&stream, indices64.len())?;
    gather(
        &execution,
        input.view(),
        indices64.view(),
        gathered64.view_mut(),
    )?;
    assert_eq!(gathered64.to_host_vec(&stream)?, [33, 22, 11]);

    let mut sparse_fill = DeviceBuffer::from_host(&stream, &[99_i32; 8])?;
    let sparse = sparse_fill.view_mut().slice(1..7)?.step_by(2)?;
    fill_range(&execution, sparse, 7_i32)?;
    assert_eq!(
        sparse_fill.to_host_vec(&stream)?,
        [99, 7, 99, 7, 99, 7, 99, 99]
    );

    let mut dense_fill = DeviceBuffer::<u32>::zeroed(&stream, 2_051)?;
    fill_range(&execution, dense_fill.view_mut(), 0xA5A5_5A5A)?;
    let dense_fill_host = dense_fill.to_host_vec(&stream)?;
    assert!(dense_fill_host.iter().all(|&value| value == 0xA5A5_5A5A));

    let mut reverse_sequence = DeviceBuffer::<u64>::zeroed(&stream, 5)?;
    sequence_range_by(&execution, reverse_sequence.view_mut().rev(), 10_u64, 2_u64)?;
    assert_eq!(reverse_sequence.to_host_vec(&stream)?, [18, 16, 14, 12, 10]);

    let mut generated = DeviceBuffer::<u32>::zeroed(&stream, 2_305)?;
    tabulate_range_async(&execution, generated.view_mut(), |index| {
        (index as u32).wrapping_mul(17).wrapping_add(5)
    })
    .wait()?;
    let generated_host = generated.to_host_vec(&stream)?;
    for index in [0, 256, 2_047, 2_048, 2_304] {
        assert_eq!(
            generated_host[index],
            (index as u32).wrapping_mul(17).wrapping_add(5)
        );
    }

    let empty_input = DeviceBuffer::<u32>::zeroed(&stream, 0)?;
    let mut empty_output = DeviceBuffer::<u32>::zeroed(&stream, 0)?;
    copy(&execution, &empty_input, &mut empty_output)?;
    fill_range(&execution, empty_output.view_mut(), 9_u32)?;
    gather(
        &execution,
        empty_input.view(),
        DeviceBuffer::<u32>::zeroed(&stream, 0)?.view(),
        empty_output.view_mut(),
    )?;

    println!("copy/gather/generation correctness and boundaries: PASS");
    Ok(())
}
