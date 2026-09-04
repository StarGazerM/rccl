use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{
    BooleanProvenance, RelationRow, sort_rows_lexicographic, sort_rows_lexicographic_into,
    sort_rows_with_payload_lexicographic, sort_rows_with_payload_lexicographic_async,
    sort_rows_with_payload_lexicographic_into,
};
use std::error::Error;

type Exec<'a> = rccl_thrust::cuda::Execution<'a, 'a, rccl_memory::BorrowedCudaMemoryResource<'a>>;

fn rows<const N: usize>(len: usize) -> Vec<RelationRow<N>> {
    (0..len)
        .map(|index| {
            let mut row = [0_u32; N];
            let mut column = 0;
            while column < N {
                row[column] = ((index * (column * 6 + 5) + column * 3) % (11 + column * 2)) as u32;
                column += 1;
            }
            row
        })
        .collect()
}

fn stable_payload_boundaries(
    execution: &Exec<'_>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    for len in [0_usize, 1, 31, 32, 33, 255, 256, 257, 513, 65_537] {
        let input = rows::<3>(len);
        let payload = (0..len as u64).collect::<Vec<_>>();
        let mut expected = input
            .iter()
            .copied()
            .zip(payload.iter().copied())
            .collect::<Vec<_>>();
        expected.sort_by_key(|pair| pair.0);

        let mut device_rows = DeviceBuffer::from_host(stream, &input)?;
        let mut device_payload = DeviceBuffer::from_host(stream, &payload)?;
        sort_rows_with_payload_lexicographic_async(
            execution,
            &mut device_rows,
            &mut device_payload,
        )?
        .wait()?;
        let actual = device_rows
            .to_host_vec(stream)?
            .into_iter()
            .zip(device_payload.to_host_vec(stream)?)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "stable row payload length {len}");
    }
    Ok(())
}

fn supported_arities(
    execution: &Exec<'_>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    macro_rules! check {
        ($arity:literal) => {{
            let input = rows::<$arity>(10_003);
            let mut expected = input.clone();
            expected.sort();
            let mut device = DeviceBuffer::from_host(stream, &input)?;
            sort_rows_lexicographic(execution, &mut device)?;
            assert_eq!(device.to_host_vec(stream)?, expected);
        }};
    }
    check!(2);
    check!(4);
    check!(5);
    check!(6);
    Ok(())
}

fn out_of_place(
    execution: &Exec<'_>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let input = rows::<5>(513);
    let payload = (0..input.len())
        .map(|index| index as f64 / 513.0)
        .collect::<Vec<_>>();
    let mut expected = input
        .iter()
        .copied()
        .zip(payload.iter().copied())
        .collect::<Vec<_>>();
    expected.sort_by_key(|pair| pair.0);

    let input_rows = DeviceBuffer::from_host(stream, &input)?;
    let input_payload = DeviceBuffer::from_host(stream, &payload)?;
    let mut output_rows = DeviceBuffer::from_host(stream, &vec![[u32::MAX; 5]; input.len()])?;
    let mut output_payload = DeviceBuffer::from_host(stream, &vec![-1.0_f64; input.len()])?;
    sort_rows_with_payload_lexicographic_into(
        execution,
        &input_rows,
        &input_payload,
        &mut output_rows,
        &mut output_payload,
    )?;
    let actual = output_rows
        .to_host_vec(stream)?
        .into_iter()
        .zip(output_payload.to_host_vec(stream)?)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert_eq!(input_rows.to_host_vec(stream)?, input);
    assert_eq!(input_payload.to_host_vec(stream)?, payload);

    let key_input = rows::<6>(257);
    let mut key_expected = key_input.clone();
    key_expected.sort();
    let key_input_device = DeviceBuffer::from_host(stream, &key_input)?;
    let mut key_output_device = DeviceBuffer::from_host(stream, &vec![[0_u32; 6]; 257])?;
    sort_rows_lexicographic_into(execution, &key_input_device, &mut key_output_device)?;
    assert_eq!(key_output_device.to_host_vec(stream)?, key_expected);
    assert_eq!(key_input_device.to_host_vec(stream)?, key_input);
    Ok(())
}

fn duplicate_payloads(
    execution: &Exec<'_>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let input = [[2_u32, 1], [1, 7], [2, 1], [1, 7], [1, 6], [2, 1]];
    let payload = [90_u64, 10, 91, 11, 12, 92];
    let mut device_rows = DeviceBuffer::from_host(stream, &input)?;
    let mut device_payload = DeviceBuffer::from_host(stream, &payload)?;
    sort_rows_with_payload_lexicographic(execution, &mut device_rows, &mut device_payload)?;
    assert_eq!(
        device_rows.to_host_vec(stream)?,
        [[1, 6], [1, 7], [1, 7], [2, 1], [2, 1], [2, 1]]
    );
    assert_eq!(
        device_payload.to_host_vec(stream)?,
        [12, 10, 11, 90, 91, 92]
    );

    let boolean_payload = [
        BooleanProvenance::FALSE,
        BooleanProvenance::TRUE,
        BooleanProvenance::TRUE,
        BooleanProvenance::FALSE,
        BooleanProvenance::TRUE,
        BooleanProvenance::FALSE,
    ];
    let mut bool_rows = DeviceBuffer::from_host(stream, &input)?;
    let mut bool_payload = DeviceBuffer::from_host(stream, &boolean_payload)?;
    sort_rows_with_payload_lexicographic(execution, &mut bool_rows, &mut bool_payload)?;
    assert_eq!(
        bool_payload
            .to_host_vec(stream)?
            .into_iter()
            .map(BooleanProvenance::get)
            .collect::<Vec<_>>(),
        [true, true, false, false, true, false]
    );
    Ok(())
}

fn contract_errors(
    execution: &Exec<'_>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let mut rows = DeviceBuffer::from_host(stream, &[[2_u32, 1], [1, 0], [3, 4]])?;
    let mut short_payload = DeviceBuffer::from_host(stream, &[1_u64, 2])?;
    let error =
        sort_rows_with_payload_lexicographic(execution, &mut rows, &mut short_payload).unwrap_err();
    assert!(error.to_string().contains("record-sort payload"));

    let input = DeviceBuffer::from_host(stream, &[[2_u32, 1], [1, 0], [3, 4]])?;
    let mut short_output = DeviceBuffer::from_host(stream, &[[0_u32, 0]; 2])?;
    let error = sort_rows_lexicographic_into(execution, &input, &mut short_output).unwrap_err();
    assert!(error.to_string().contains("record-sort output"));

    let payload = DeviceBuffer::from_host(stream, &[1_u64, 2, 3])?;
    let mut output = DeviceBuffer::from_host(stream, &[[0_u32, 0]; 3])?;
    let mut short_output_payload = DeviceBuffer::from_host(stream, &[0_u64; 2])?;
    let error = sort_rows_with_payload_lexicographic_into(
        execution,
        &input,
        &payload,
        &mut output,
        &mut short_output_payload,
    )
    .unwrap_err();
    assert!(error.to_string().contains("output payload"));

    let mut scalar_rows = DeviceBuffer::from_host(stream, &[[3_u32], [1], [2]])?;
    let error = sort_rows_lexicographic(execution, &mut scalar_rows).unwrap_err();
    assert!(error.to_string().contains("use radix sort for scalar keys"));

    let mut wide_rows = DeviceBuffer::from_host(stream, &[[0_u32; 7]; 1])?;
    let error = sort_rows_lexicographic(execution, &mut wide_rows).unwrap_err();
    assert!(error.to_string().contains("arities 2 through 6"));
    Ok(())
}

#[inline(always)]
fn faulting_compare(left: &RelationRow<3>, right: &RelationRow<3>) -> std::cmp::Ordering {
    if left[0] == 3 || right[0] == 3 {
        cuda_device::debug::trap();
    }
    left.cmp(right)
}

fn injected_fault(
    context: &std::sync::Arc<CudaContext>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let program = rccl_cub::cuda::Program::load(context)?;
    let launch = program.on(stream)?;
    let mut data = rows::<3>(257);
    data[0][0] = 3;
    let mut device_rows = DeviceBuffer::from_host(stream, &data)?;
    let mut workspace =
        rccl_cub::device_merge_sort_keys_workspace::<RelationRow<3>>(&launch, data.len())?;
    rccl_cub::device_merge_sort_keys(&launch, &mut device_rows, &mut workspace, faulting_compare)?;
    match stream.synchronize() {
        Err(error) => {
            eprintln!("record sort comparator fault surfaced: {error}");
            Ok(())
        }
        Ok(()) => panic!("record sort comparator fault did not surface"),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_thrust::cuda::Program::load(&context)?;
    let execution = program.par(&stream)?;
    if std::env::args().any(|argument| argument == "--fault") {
        return injected_fault(&context, &stream);
    }
    stable_payload_boundaries(&execution, &stream)?;
    supported_arities(&execution, &stream)?;
    out_of_place(&execution, &stream)?;
    duplicate_payloads(&execution, &stream)?;
    contract_errors(&execution, &stream)?;
    println!("stable lexicographic record sort CPU differential matrix: PASS");
    Ok(())
}
