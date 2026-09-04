use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{
    BooleanOr, BooleanProvenance, DeviceBufferExt, DeviceRangeExt, IndependentProbabilityOr,
    NaturalBagAdd, arithmetic_range, reduce_by_key, reduce_by_key_async,
};
use std::error::Error;

type Exec<'a> = rccl_thrust::cuda::Execution<'a, 'a, rccl_memory::BorrowedCudaMemoryResource<'a>>;

fn count(
    stream: &cuda_core::CudaStream,
    value: &DeviceBuffer<u64>,
) -> Result<usize, Box<dyn Error>> {
    Ok(value.to_host_vec(stream)?[0] as usize)
}

fn cpu_u64(keys: &[u32], values: &[u64]) -> (Vec<u32>, Vec<u64>) {
    let mut out_keys = Vec::new();
    let mut out_values = Vec::new();
    for (&key, &value) in keys.iter().zip(values) {
        if out_keys.last().copied() == Some(key) {
            let last = out_values.last_mut().unwrap();
            *last = u64::wrapping_add(*last, value);
        } else {
            out_keys.push(key);
            out_values.push(value);
        }
    }
    (out_keys, out_values)
}

fn u64_boundaries(
    execution: &Exec<'_>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    for len in [0_usize, 1, 31, 32, 33, 255, 256, 257, 513, 65_537] {
        for pattern in 0..3 {
            let keys = (0..len)
                .map(|i| match pattern {
                    0 => 7,
                    1 => i as u32,
                    _ => ((i / 3) % 11) as u32,
                })
                .collect::<Vec<_>>();
            let values = (0..len)
                .map(|i| {
                    if i % 17 == 0 {
                        u64::MAX
                    } else {
                        (i % 13 + 1) as u64
                    }
                })
                .collect::<Vec<_>>();
            let expected = cpu_u64(&keys, &values);
            let keys_device = DeviceBuffer::from_host(stream, &keys)?;
            let values_device = DeviceBuffer::from_host(stream, &values)?;
            let mut out_keys = DeviceBuffer::from_host(stream, &vec![u32::MAX; len])?;
            let mut out_values = DeviceBuffer::from_host(stream, &vec![u64::MAX; len])?;
            let mut groups = DeviceBuffer::from_host(stream, &[u64::MAX])?;
            reduce_by_key(
                execution,
                keys_device.view(),
                values_device.view(),
                out_keys.view_mut(),
                out_values.view_mut(),
                &mut groups,
                |left, right| left == right,
                NaturalBagAdd,
            )?;
            let actual_groups = count(stream, &groups)?;
            assert_eq!(
                actual_groups,
                expected.0.len(),
                "len={len} pattern={pattern}"
            );
            assert_eq!(&out_keys.to_host_vec(stream)?[..actual_groups], expected.0);
            assert_eq!(
                &out_values.to_host_vec(stream)?[..actual_groups],
                expected.1
            );
        }
    }
    Ok(())
}

fn generated_ranges(
    execution: &Exec<'_>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let len = 513;
    let mut out_keys = DeviceBuffer::from_host(stream, &vec![0_u32; len])?;
    let mut out_values = DeviceBuffer::from_host(stream, &vec![0_u64; len])?;
    let mut groups = DeviceBuffer::from_host(stream, &[0_u64])?;
    reduce_by_key(
        execution,
        arithmetic_range(0_u32, 1, len),
        arithmetic_range(10_u64, 2, len),
        out_keys.view_mut(),
        out_values.view_mut(),
        &mut groups,
        |left, right| left / 5 == right / 5,
        NaturalBagAdd,
    )?;
    let group_count = count(stream, &groups)?;
    assert_eq!(group_count, len.div_ceil(5));
    let keys = out_keys.to_host_vec(stream)?;
    let values = out_values.to_host_vec(stream)?;
    for group in 0..group_count {
        let start = group * 5;
        let end = (start + 5).min(len);
        assert_eq!(keys[group], start as u32);
        let expected = (start..end)
            .map(|i| 10_u64 + 2 * i as u64)
            .fold(0_u64, u64::wrapping_add);
        assert_eq!(values[group], expected);
    }
    Ok(())
}

fn provenance_and_adjacent(
    execution: &Exec<'_>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let columns = DeviceBuffer::from_host(stream, &[1_u32, 1, 2, 1, 1])?;
    let key_provenance = DeviceBuffer::from_host(stream, &[90_u32, 91, 92, 93, 94])?;
    let values = DeviceBuffer::from_host(stream, &[2_u64, 3, 5, 7, 11])?;
    let mut out_columns = DeviceBuffer::from_host(stream, &[0_u32; 5])?;
    let mut out_key_provenance = DeviceBuffer::from_host(stream, &[0_u32; 5])?;
    let mut out_values = DeviceBuffer::from_host(stream, &[0_u64; 5])?;
    let mut groups = DeviceBuffer::from_host(stream, &[0_u64])?;
    reduce_by_key_async(
        execution,
        columns.view().with_values(key_provenance.view())?,
        values.view(),
        out_columns
            .view_mut()
            .with_values(out_key_provenance.view_mut())?,
        out_values.view_mut(),
        &mut groups,
        |left, right| left.0 == right.0,
        NaturalBagAdd,
    )?
    .wait()?;
    let group_count = count(stream, &groups)?;
    assert_eq!(group_count, 3);
    assert_eq!(&out_columns.to_host_vec(stream)?[..group_count], [1, 2, 1]);
    assert_eq!(
        &out_key_provenance.to_host_vec(stream)?[..group_count],
        [90, 92, 93]
    );
    assert_eq!(&out_values.to_host_vec(stream)?[..group_count], [5, 5, 18]);
    Ok(())
}

fn workload_operators(
    execution: &Exec<'_>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let keys = DeviceBuffer::from_host(stream, &[0_u32, 0, 0, 1, 1, 2])?;

    let bool_values = DeviceBuffer::from_host(
        stream,
        &[
            BooleanProvenance::FALSE,
            BooleanProvenance::TRUE,
            BooleanProvenance::FALSE,
            BooleanProvenance::FALSE,
            BooleanProvenance::FALSE,
            BooleanProvenance::TRUE,
        ],
    )?;
    let mut bool_out = DeviceBuffer::from_host(stream, &[BooleanProvenance::FALSE; 6])?;
    let mut bool_keys = DeviceBuffer::from_host(stream, &[0_u32; 6])?;
    let mut groups = DeviceBuffer::from_host(stream, &[0_u64])?;
    reduce_by_key(
        execution,
        keys.view(),
        bool_values.view(),
        bool_keys.view_mut(),
        bool_out.view_mut(),
        &mut groups,
        |a, b| a == b,
        BooleanOr,
    )?;
    assert_eq!(
        bool_out.to_host_vec(stream)?[..3]
            .iter()
            .map(|value| value.get())
            .collect::<Vec<_>>(),
        [true, false, true]
    );

    let probabilities = DeviceBuffer::from_host(stream, &[0.1_f64, 0.2, 0.3, 0.4, 0.5, 1.0])?;
    let mut probability_out = DeviceBuffer::from_host(stream, &[0.0_f64; 6])?;
    reduce_by_key(
        execution,
        keys.view(),
        probabilities.view(),
        bool_keys.view_mut(),
        probability_out.view_mut(),
        &mut groups,
        |a, b| a == b,
        IndependentProbabilityOr,
    )?;
    let actual = probability_out.to_host_vec(stream)?;
    let first =
        IndependentProbabilityOrCpu::combine(IndependentProbabilityOrCpu::combine(0.1, 0.2), 0.3);
    assert_eq!(actual[0].to_bits(), first.to_bits());
    assert_eq!(actual[1].to_bits(), (0.4_f64 + 0.5 - 0.4 * 0.5).to_bits());
    assert_eq!(actual[2], 1.0);

    let overflow_values = DeviceBuffer::from_host(stream, &[u64::MAX, 2, 3, u64::MAX, 2, 0])?;
    let mut overflow_out = DeviceBuffer::from_host(stream, &[0_u64; 6])?;
    reduce_by_key(
        execution,
        keys.view(),
        overflow_values.view(),
        bool_keys.view_mut(),
        overflow_out.view_mut(),
        &mut groups,
        |a, b| a == b,
        NaturalBagAdd,
    )?;
    assert_eq!(&overflow_out.to_host_vec(stream)?[..3], [4, 1, 0]);
    Ok(())
}

struct IndependentProbabilityOrCpu;
impl IndependentProbabilityOrCpu {
    fn combine(left: f64, right: f64) -> f64 {
        (left + right - left * right).clamp(0.0, 1.0)
    }
}

fn contract_errors(
    execution: &Exec<'_>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let keys = DeviceBuffer::from_host(stream, &[1_u32, 1, 2])?;
    let short_values = DeviceBuffer::from_host(stream, &[1_u64, 2])?;
    let values = DeviceBuffer::from_host(stream, &[1_u64, 2, 3])?;
    let mut out_keys = DeviceBuffer::from_host(stream, &[0_u32; 3])?;
    let mut out_values = DeviceBuffer::from_host(stream, &[0_u64; 3])?;
    let mut groups = DeviceBuffer::from_host(stream, &[0_u64])?;
    let error = reduce_by_key(
        execution,
        keys.view(),
        short_values.view(),
        out_keys.view_mut(),
        out_values.view_mut(),
        &mut groups,
        |a, b| a == b,
        NaturalBagAdd,
    )
    .unwrap_err();
    assert!(error.to_string().contains("values length"));

    let mut short_keys = DeviceBuffer::from_host(stream, &[0_u32; 2])?;
    let error = reduce_by_key(
        execution,
        keys.view(),
        values.view(),
        short_keys.view_mut(),
        out_values.view_mut(),
        &mut groups,
        |a, b| a == b,
        NaturalBagAdd,
    )
    .unwrap_err();
    assert!(error.to_string().contains("key output"));

    let mut short_aggregates = DeviceBuffer::from_host(stream, &[0_u64; 2])?;
    let error = reduce_by_key(
        execution,
        keys.view(),
        values.view(),
        out_keys.view_mut(),
        short_aggregates.view_mut(),
        &mut groups,
        |a, b| a == b,
        NaturalBagAdd,
    )
    .unwrap_err();
    assert!(error.to_string().contains("aggregate output"));

    let mut empty_count = DeviceBuffer::from_host(stream, &[])?;
    let error = reduce_by_key(
        execution,
        keys.view(),
        values.view(),
        out_keys.view_mut(),
        out_values.view_mut(),
        &mut empty_count,
        |a, b| a == b,
        NaturalBagAdd,
    )
    .unwrap_err();
    assert!(error.to_string().contains("group-count output"));
    Ok(())
}

fn injected_oob(
    execution: &Exec<'_>,
    stream: &cuda_core::CudaStream,
) -> Result<(), Box<dyn Error>> {
    let keys = DeviceBuffer::from_host(stream, &[1_u32, 1])?;
    let values = DeviceBuffer::from_host(stream, &[4_u64, 5])?;
    let indices = DeviceBuffer::from_host(stream, &[0_usize, 2])?;
    let mut out_keys = DeviceBuffer::from_host(stream, &[0_u32; 2])?;
    let mut out_values = DeviceBuffer::from_host(stream, &[0_u64; 2])?;
    let mut groups = DeviceBuffer::from_host(stream, &[0_u64])?;
    let pending = reduce_by_key_async(
        execution,
        keys.view(),
        values.view().gather(indices.view()),
        out_keys.view_mut(),
        out_values.view_mut(),
        &mut groups,
        |a, b| a == b,
        NaturalBagAdd,
    )?;
    match pending.wait() {
        Err(error) => {
            eprintln!("grouped reduction range OOB surfaced: {error}");
            Ok(())
        }
        Ok(()) => panic!("grouped reduction range OOB did not fault"),
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
    u64_boundaries(&execution, &stream)?;
    generated_ranges(&execution, &stream)?;
    provenance_and_adjacent(&execution, &stream)?;
    workload_operators(&execution, &stream)?;
    contract_errors(&execution, &stream)?;
    println!("adjacent reduce-by-key CPU differential matrix: PASS");
    Ok(())
}
