use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{
    DeviceBufferExt, DeviceRangeExt, arithmetic_range, select_flagged, select_flagged_async,
    select_if, select_if_async, stable_partition_by, unique_by, unique_copy_by,
};
use std::error::Error;

fn count(
    stream: &cuda_core::CudaStream,
    value: &DeviceBuffer<u64>,
) -> Result<usize, Box<dyn Error>> {
    Ok(value.to_host_vec(stream)?[0] as usize)
}

fn selection_boundaries(
    execution: &rccl_thrust::cuda::Execution<'_, '_, rccl_memory::BorrowedCudaMemoryResource<'_>>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    for len in [0_usize, 1, 31, 32, 33, 255, 256, 257, 513, 65_537] {
        let mut output = DeviceBuffer::from_host(stream, &vec![0_u32; len])?;
        let mut selected_count = DeviceBuffer::from_host(stream, &[u64::MAX])?;
        select_if(
            execution,
            arithmetic_range(0_u32, 1, len),
            output.view_mut(),
            &mut selected_count,
            |value| value % 7 == 0 || value == 255,
        )?;
        let expected = (0..len)
            .map(|value| value as u32)
            .filter(|&value| value % 7 == 0 || value == 255)
            .collect::<Vec<_>>();
        let actual = output.to_host_vec(stream)?;
        let selected = count(stream, &selected_count)?;
        assert_eq!(selected, expected.len());
        assert_eq!(&actual[..selected], expected);
    }
    Ok(())
}

fn flagged_selection(
    execution: &rccl_thrust::cuda::Execution<'_, '_, rccl_memory::BorrowedCudaMemoryResource<'_>>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let flags_host = [0_u32, 2, 0, 1, u32::MAX, 0, 4];
    let flags = DeviceBuffer::from_host(stream, &flags_host)?;
    let mut output = DeviceBuffer::from_host(stream, &[0_u32; 7])?;
    let mut selected_count = DeviceBuffer::from_host(stream, &[u64::MAX])?;
    select_flagged_async(
        execution,
        arithmetic_range(10_u32, 3, flags_host.len()),
        flags.view(),
        output.view_mut(),
        &mut selected_count,
    )?
    .wait()?;
    let selected = count(stream, &selected_count)?;
    assert_eq!(selected, 4);
    assert_eq!(&output.to_host_vec(stream)?[..selected], [13, 19, 22, 28]);
    Ok(())
}

fn adjacent_unique(
    execution: &rccl_thrust::cuda::Execution<'_, '_, rccl_memory::BorrowedCudaMemoryResource<'_>>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let keys_host = [1_u32, 1, 2, 2, 2, 1, 3, 3];
    let provenance_host = [10_u64, 11, 20, 21, 22, 12, 30, 31];
    let keys = DeviceBuffer::from_host(stream, &keys_host)?;
    let provenance = DeviceBuffer::from_host(stream, &provenance_host)?;
    let mut out_keys = DeviceBuffer::from_host(stream, &[0_u32; 8])?;
    let mut out_provenance = DeviceBuffer::from_host(stream, &[0_u64; 8])?;
    let mut selected_count = DeviceBuffer::from_host(stream, &[u64::MAX])?;
    unique_copy_by(
        execution,
        keys.view().with_values(provenance.view())?,
        out_keys.view_mut().with_values(out_provenance.view_mut())?,
        &mut selected_count,
        |left, right| left.0 == right.0,
    )?;
    let selected = count(stream, &selected_count)?;
    assert_eq!(selected, 4);
    assert_eq!(&out_keys.to_host_vec(stream)?[..selected], [1, 2, 1, 3]);
    assert_eq!(
        &out_provenance.to_host_vec(stream)?[..selected],
        [10, 20, 12, 30]
    );

    let mut in_place_keys = DeviceBuffer::from_host(stream, &keys_host)?;
    let mut in_place_provenance = DeviceBuffer::from_host(stream, &provenance_host)?;
    let mut scratch_keys = DeviceBuffer::from_host(stream, &[0_u32; 8])?;
    let mut scratch_provenance = DeviceBuffer::from_host(stream, &[0_u64; 8])?;
    unique_by(
        execution,
        in_place_keys
            .view_mut()
            .with_values(in_place_provenance.view_mut())?,
        scratch_keys
            .view_mut()
            .with_values(scratch_provenance.view_mut())?,
        &mut selected_count,
        |left, right| left.0 == right.0,
    )?;
    let selected = count(stream, &selected_count)?;
    assert_eq!(selected, 4);
    assert_eq!(
        &in_place_keys.to_host_vec(stream)?[..selected],
        [1, 2, 1, 3]
    );
    assert_eq!(
        &in_place_provenance.to_host_vec(stream)?[..selected],
        [10, 20, 12, 30]
    );
    Ok(())
}

fn unique_boundaries(
    execution: &rccl_thrust::cuda::Execution<'_, '_, rccl_memory::BorrowedCudaMemoryResource<'_>>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    for len in [0_usize, 1, 31, 32, 33, 255, 256, 257, 513] {
        let input_host = (0..len).map(|index| (index / 3) as u32).collect::<Vec<_>>();
        let expected = input_host
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, value)| {
                (index == 0 || input_host[index - 1] != value).then_some(value)
            })
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(stream, &input_host)?;
        let mut output = DeviceBuffer::from_host(stream, &vec![0_u32; len])?;
        let mut selected_count = DeviceBuffer::from_host(stream, &[u64::MAX])?;
        unique_copy_by(
            execution,
            input.view(),
            output.view_mut(),
            &mut selected_count,
            |left, right| left == right,
        )?;
        let selected = count(stream, &selected_count)?;
        assert_eq!(selected, expected.len());
        assert_eq!(&output.to_host_vec(stream)?[..selected], expected);
    }
    Ok(())
}

fn partition_boundaries(
    execution: &rccl_thrust::cuda::Execution<'_, '_, rccl_memory::BorrowedCudaMemoryResource<'_>>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    for len in [0_usize, 1, 255, 256, 257, 513] {
        for mode in 0_u32..3 {
            let input = (0..len).map(|index| index as u32).collect::<Vec<_>>();
            let predicate = move |value: u32| match mode {
                0 => false,
                1 => true,
                _ => value % 5 <= 1,
            };
            let mut expected = input
                .iter()
                .copied()
                .filter(|&value| predicate(value))
                .collect::<Vec<_>>();
            let selected = expected.len();
            expected.extend(input.iter().copied().filter(|&value| !predicate(value)));
            let mut values = DeviceBuffer::from_host(stream, &input)?;
            let mut scratch = DeviceBuffer::from_host(stream, &vec![0_u32; len])?;
            let mut selected_count = DeviceBuffer::from_host(stream, &[u64::MAX])?;
            stable_partition_by(
                execution,
                values.view_mut(),
                scratch.view_mut(),
                &mut selected_count,
                predicate,
            )?;
            assert_eq!(count(stream, &selected_count)?, selected);
            assert_eq!(values.to_host_vec(stream)?, expected);
        }
    }
    Ok(())
}

fn stable_partition(
    execution: &rccl_thrust::cuda::Execution<'_, '_, rccl_memory::BorrowedCudaMemoryResource<'_>>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let mut keys = DeviceBuffer::from_host(stream, &[5_u32, 2, 7, 4, 1, 8])?;
    let mut provenance = DeviceBuffer::from_host(stream, &[50_u64, 20, 70, 40, 10, 80])?;
    let mut scratch_keys = DeviceBuffer::from_host(stream, &[0_u32; 6])?;
    let mut scratch_provenance = DeviceBuffer::from_host(stream, &[0_u64; 6])?;
    let mut selected_count = DeviceBuffer::from_host(stream, &[u64::MAX])?;
    stable_partition_by(
        execution,
        keys.view_mut().with_values(provenance.view_mut())?,
        scratch_keys
            .view_mut()
            .with_values(scratch_provenance.view_mut())?,
        &mut selected_count,
        |item| item.0 % 2 == 1,
    )?;
    assert_eq!(count(stream, &selected_count)?, 3);
    assert_eq!(keys.to_host_vec(stream)?, [5, 7, 1, 2, 4, 8]);
    assert_eq!(provenance.to_host_vec(stream)?, [50, 70, 10, 20, 40, 80]);
    Ok(())
}

fn contract_errors(
    execution: &rccl_thrust::cuda::Execution<'_, '_, rccl_memory::BorrowedCudaMemoryResource<'_>>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let input = DeviceBuffer::from_host(stream, &[1_u32, 2, 3])?;
    let mut short = DeviceBuffer::from_host(stream, &[0_u32; 2])?;
    let mut selected_count = DeviceBuffer::from_host(stream, &[0_u64])?;
    let error = select_if(
        execution,
        input.view(),
        short.view_mut(),
        &mut selected_count,
        |_| true,
    )
    .unwrap_err();
    assert!(error.to_string().contains("select output"));

    let mut output = DeviceBuffer::from_host(stream, &[0_u32; 3])?;
    let mut empty_count = DeviceBuffer::from_host(stream, &[])?;
    let error = select_if(
        execution,
        input.view(),
        output.view_mut(),
        &mut empty_count,
        |_| true,
    )
    .unwrap_err();
    assert!(error.to_string().contains("selected-count output"));

    let short_flags = DeviceBuffer::from_host(stream, &[1_u32, 0])?;
    let error = select_flagged(
        execution,
        input.view(),
        short_flags.view(),
        output.view_mut(),
        &mut selected_count,
    )
    .unwrap_err();
    assert!(error.to_string().contains("selection flags"));
    Ok(())
}

fn injected_oob(
    execution: &rccl_thrust::cuda::Execution<'_, '_, rccl_memory::BorrowedCudaMemoryResource<'_>>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let input = DeviceBuffer::from_host(stream, &[3_u32, 4, 5])?;
    let indices = DeviceBuffer::from_host(stream, &[0_usize, 3])?;
    let mut output = DeviceBuffer::from_host(stream, &[0_u32; 2])?;
    let mut selected_count = DeviceBuffer::from_host(stream, &[0_u64])?;
    let pending = select_if_async(
        execution,
        input.view().gather(indices.view()),
        output.view_mut(),
        &mut selected_count,
        |_| true,
    )?;
    match pending.wait() {
        Err(error) => {
            eprintln!("compaction range OOB surfaced: {error}");
            Ok(())
        }
        Ok(()) => panic!("compaction range OOB did not fault"),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_thrust::cuda::Program::load(&context)?;
    let execution = program.par(&stream)?;
    if std::env::args().any(|arg| arg == "--range-oob") {
        return injected_oob(&execution, &stream);
    }
    selection_boundaries(&execution, &stream)?;
    flagged_selection(&execution, &stream)?;
    adjacent_unique(&execution, &stream)?;
    unique_boundaries(&execution, &stream)?;
    partition_boundaries(&execution, &stream)?;
    stable_partition(&execution, &stream)?;
    contract_errors(&execution, &stream)?;
    println!("select/unique/stable-partition differential matrix: PASS");
    Ok(())
}
