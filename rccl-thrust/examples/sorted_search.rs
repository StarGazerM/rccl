use core::cmp::Ordering;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use rccl_thrust::{
    DeviceAlgorithmError, DeviceBufferExt, DeviceRangeExt, arithmetic_range, discard,
    equal_range, equal_range_async, equal_range_by, equal_range_by_async,
    lower_bound, lower_bound_async, lower_bound_by, lower_bound_by_async,
    upper_bound, upper_bound_async, upper_bound_by, upper_bound_by_async,
};
use std::error::Error;

type Exec<'a> = rccl_thrust::cuda::Execution<'a, 'a, rccl_memory::BorrowedCudaMemoryResource<'a>>;

fn oracle<T, F>(haystack: &[T], needles: &[T], compare: F) -> (Vec<u64>, Vec<u64>)
where
    F: Fn(&T, &T) -> Ordering,
{
    let begins = needles.iter().map(|needle| {
        haystack.partition_point(|value| compare(value, needle) == Ordering::Less) as u64
    }).collect();
    let ends = needles.iter().map(|needle| {
        haystack.partition_point(|value| compare(value, needle) != Ordering::Greater) as u64
    }).collect();
    (begins, ends)
}

fn natural<T: DeviceCopy + Ord>(
    exec: &Exec<'_>, stream: &CudaStream, haystack: &[T], needles: &[T],
) -> Result<(), Box<dyn Error>> {
    let expected = oracle(haystack, needles, |a, b| a.cmp(b));
    let hay = DeviceBuffer::from_host(stream, haystack)?;
    let queries = DeviceBuffer::from_host(stream, needles)?;
    let mut begins = DeviceBuffer::from_host(stream, &vec![u64::MAX; needles.len()])?;
    let mut ends = DeviceBuffer::from_host(stream, &vec![u64::MAX; needles.len()])?;
    lower_bound(exec, hay.view(), queries.view(), begins.view_mut())?;
    assert_eq!(begins.to_host_vec(stream)?, expected.0);
    upper_bound(exec, hay.view(), queries.view(), ends.view_mut())?;
    assert_eq!(ends.to_host_vec(stream)?, expected.1);
    equal_range(exec, hay.view(), queries.view(), begins.view_mut(), ends.view_mut())?;
    assert_eq!(begins.to_host_vec(stream)?, expected.0);
    assert_eq!(ends.to_host_vec(stream)?, expected.1);
    lower_bound_async(exec, hay.view(), queries.view(), begins.view_mut()).wait()?;
    assert_eq!(begins.to_host_vec(stream)?, expected.0);
    upper_bound_async(exec, hay.view(), queries.view(), ends.view_mut()).wait()?;
    assert_eq!(ends.to_host_vec(stream)?, expected.1);
    equal_range_async(exec, hay.view(), queries.view(), begins.view_mut(), ends.view_mut()).wait()?;
    assert_eq!(begins.to_host_vec(stream)?, expected.0);
    assert_eq!(ends.to_host_vec(stream)?, expected.1);
    Ok(())
}

fn ordered<T: DeviceCopy, F: Fn(&T, &T) -> Ordering + Copy>(
    exec: &Exec<'_>, stream: &CudaStream, haystack: &[T], needles: &[T], compare: F,
) -> Result<(), Box<dyn Error>> {
    let expected = oracle(haystack, needles, compare);
    let hay = DeviceBuffer::from_host(stream, haystack)?;
    let queries = DeviceBuffer::from_host(stream, needles)?;
    let mut begins = DeviceBuffer::from_host(stream, &vec![u64::MAX; needles.len()])?;
    let mut ends = DeviceBuffer::from_host(stream, &vec![u64::MAX; needles.len()])?;
    lower_bound_by(exec, hay.view(), queries.view(), begins.view_mut(), compare)?;
    assert_eq!(begins.to_host_vec(stream)?, expected.0);
    upper_bound_by(exec, hay.view(), queries.view(), ends.view_mut(), compare)?;
    assert_eq!(ends.to_host_vec(stream)?, expected.1);
    equal_range_by(exec, hay.view(), queries.view(), begins.view_mut(), ends.view_mut(), compare)?;
    assert_eq!(begins.to_host_vec(stream)?, expected.0);
    assert_eq!(ends.to_host_vec(stream)?, expected.1);
    lower_bound_by_async(exec, hay.view(), queries.view(), begins.view_mut(), compare).wait()?;
    assert_eq!(begins.to_host_vec(stream)?, expected.0);
    upper_bound_by_async(exec, hay.view(), queries.view(), ends.view_mut(), compare).wait()?;
    assert_eq!(ends.to_host_vec(stream)?, expected.1);
    equal_range_by_async(exec, hay.view(), queries.view(), begins.view_mut(), ends.view_mut(), compare).wait()?;
    assert_eq!(begins.to_host_vec(stream)?, expected.0);
    assert_eq!(ends.to_host_vec(stream)?, expected.1);
    Ok(())
}

fn boundary_matrix(exec: &Exec<'_>, stream: &CudaStream) -> Result<(), Box<dyn Error>> {
    for len in [0_usize, 1, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 65_537] {
        for pattern in 0..3 {
            let hay = (0..len).map(|i| match pattern {
                0 => 12_u32,
                1 => 2 * i as u32 + 2,
                _ => 2 * (i / 7) as u32 + 2,
            }).collect::<Vec<_>>();
            // All hits, all misses (including both boundaries), and no needles.
            natural(exec, stream, &hay, &hay)?;
            let misses = (0..len + 3).map(|i| 2 * i as u32 + 1).collect::<Vec<_>>();
            natural(exec, stream, &hay, &misses)?;
            natural(exec, stream, &hay, &[])?;
        }
    }
    for descending in [false, true] {
        // Captured bucket width makes equivalence different from value equality.
        let width = 10_u32;
        let mut hay = vec![10_u32, 12, 19, 20, 23, 40, 41];
        if descending { hay.reverse(); }
        ordered(exec, stream, &hay, &[0, 11, 18, 25, 35, 44, 50], move |a, b| {
            let order = (a / width).cmp(&(b / width));
            if descending { order.reverse() } else { order }
        })?;
    }
    Ok(())
}

fn wide_and_row_keys(exec: &Exec<'_>, stream: &CudaStream) -> Result<(), Box<dyn Error>> {
    // Most-significant column first preserves lexicographic order. Values cross
    // both 32- and 64-bit boundaries, so truncating a packed key cannot pass.
    let pack = |a: u32, b: u32, c: u32, d: u32| {
        ((a as u128) << 96) | ((b as u128) << 64) | ((c as u128) << 32) | d as u128
    };
    let hay = [pack(0, 0, 0, 9), pack(0, 1, 0, 0), pack(1, 0, 0, 0),
        pack(1, 0, 0, 0), pack(u32::MAX, 0, 0, 0), u128::MAX];
    let needles = [0, hay[0], pack(0, 0, 1, 0), hay[1], hay[2],
        pack(1, 0, 0, 1), u128::MAX];
    natural(exec, stream, &hay, &needles)?;
    let rows = [[0_u32, 1, 9], [0, 2, 0], [1, 0, 0], [1, 0, 0], [u32::MAX, 0, 0]];
    let row_needles = [[0_u32, 0, 0], [0, 1, 9], [0, 1, 10], [1, 0, 0], [u32::MAX; 3]];
    natural(exec, stream, &rows, &row_needles)?;
    ordered(exec, stream, &rows, &row_needles, |a, b| a.cmp(b))?;
    // Float order is explicit, including signed zero, infinities, and a NaN.
    ordered(exec, stream,
        &[f32::NEG_INFINITY, -0.0, 0.0, 1.0, 1.0, f32::INFINITY, f32::NAN],
        &[f32::NEG_INFINITY, -1.0, -0.0, 0.0, 1.0, 2.0, f32::INFINITY, f32::NAN],
        |a, b| a.total_cmp(b))?;
    Ok(())
}

fn composed_ranges(exec: &Exec<'_>, stream: &CudaStream) -> Result<(), Box<dyn Error>> {
    let hay = DeviceBuffer::from_host(stream, &[0_u32, 2, 2, 4, 8, 99])?;
    let queries = DeviceBuffer::from_host(stream, &[99_u32, 1, 2, 3, 8, 9, 99])?;
    let mut begins = DeviceBuffer::from_host(stream, &[u64::MAX; 7])?;
    let mut ends = DeviceBuffer::from_host(stream, &[u64::MAX; 7])?;
    equal_range_async(exec, hay.view().slice(1..5)?, queries.view().slice(1..6)?,
        begins.view_mut().slice(1..6)?, ends.view_mut().slice(1..6)?).wait()?;
    assert_eq!(begins.to_host_vec(stream)?, [u64::MAX, 0, 0, 2, 3, 4, u64::MAX]);
    assert_eq!(ends.to_host_vec(stream)?, [u64::MAX, 0, 2, 2, 4, 4, u64::MAX]);

    // Lowered generated/mapped inputs and noncontiguous writable outputs.
    let mut strided = DeviceBuffer::from_host(stream, &[u64::MAX; 10])?;
    lower_bound(exec, arithmetic_range(0_u32, 1, 8).map(|v| v * 2),
        arithmetic_range(1_u32, 2, 5), strided.view_mut().step_by(2)?)?;
    assert_eq!(strided.to_host_vec(stream)?,
        [1, u64::MAX, 2, u64::MAX, 3, u64::MAX, 4, u64::MAX, 5, u64::MAX]);

    // A row-valued zip is searched without materializing a record buffer.
    let first = DeviceBuffer::from_host(stream, &[1_u32, 1, 2, 2])?;
    let second = DeviceBuffer::from_host(stream, &[2_u32, 2, 0, 3])?;
    let nfirst = DeviceBuffer::from_host(stream, &[1_u32, 2, 3])?;
    let nsecond = DeviceBuffer::from_host(stream, &[2_u32, 1, 0])?;
    let mut row_begins = DeviceBuffer::from_host(stream, &[u64::MAX; 3])?;
    let mut row_ends = DeviceBuffer::from_host(stream, &[u64::MAX; 3])?;
    equal_range(exec, first.view().zip_eq(second.view())?,
        nfirst.view().zip_eq(nsecond.view())?, row_begins.view_mut(), row_ends.view_mut())?;
    assert_eq!(row_begins.to_host_vec(stream)?, [0, 3, 4]);
    assert_eq!(row_ends.to_host_vec(stream)?, [2, 3, 4]);

    // Dropping completion synchronizes and releases the affine output borrow.
    drop(lower_bound_async(exec, hay.view(), queries.view(), begins.view_mut()));
    assert_eq!(begins.to_host_vec(stream)?, oracle(&[0_u32, 2, 2, 4, 8, 99],
        &[99, 1, 2, 3, 8, 9, 99], |a, b| a.cmp(b)).0);
    Ok(())
}

fn contract_errors(exec: &Exec<'_>, stream: &CudaStream) -> Result<(), Box<dyn Error>> {
    let hay = DeviceBuffer::from_host(stream, &[1_u32, 2])?;
    let queries = DeviceBuffer::from_host(stream, &[2_u32])?;
    let mut valid = DeviceBuffer::from_host(stream, &[77_u64])?;
    for len in [0_usize, 2] {
        let mut wrong = DeviceBuffer::from_host(stream, &vec![88_u64; len])?;
        let error = lower_bound(exec, hay.view(), queries.view(), wrong.view_mut()).unwrap_err();
        assert!(matches!(error, DeviceAlgorithmError::LengthMismatch {
            expected: 1, actual, ..
        } if actual == len));
        let error = upper_bound_async(exec, hay.view(), queries.view(), wrong.view_mut()).wait().unwrap_err();
        assert!(matches!(error, DeviceAlgorithmError::LengthMismatch { .. }));
        let error = equal_range_async(exec, hay.view(), queries.view(), valid.view_mut(), wrong.view_mut())
            .wait().unwrap_err();
        assert!(matches!(error, DeviceAlgorithmError::LengthMismatch { role: "equal-range ends", .. }));
        assert_eq!(valid.to_host_vec(stream)?, [77]);
        let error = equal_range(exec, hay.view(), queries.view(), wrong.view_mut(), valid.view_mut()).unwrap_err();
        assert!(matches!(error, DeviceAlgorithmError::LengthMismatch { role: "equal-range begins", .. }));
        assert_eq!(valid.to_host_vec(stream)?, [77]);
        assert_eq!(wrong.to_host_vec(stream)?, vec![88; len]);
    }
    // Empty needles still enforce exact output length and validate lowering.
    let error = lower_bound(exec, hay.view(), queries.view().slice(0..0)?, valid.view_mut()).unwrap_err();
    assert!(matches!(error, DeviceAlgorithmError::LengthMismatch { expected: 0, actual: 1, .. }));
    let too_long = u32::MAX as usize + 2;
    let error = lower_bound_async(exec, arithmetic_range(0_u32, 1, too_long),
        queries.view(), valid.view_mut()).wait().unwrap_err();
    assert!(matches!(error, DeviceAlgorithmError::SizeOverflow));
    let error = upper_bound(exec, hay.view(), arithmetic_range(0_u32, 1, too_long), discard::<u64>(too_long)).unwrap_err();
    assert!(matches!(error, DeviceAlgorithmError::SizeOverflow));
    // A failed completion releases resources and leaves subsequent work usable.
    upper_bound(exec, hay.view(), queries.view(), valid.view_mut())?;
    assert_eq!(valid.to_host_vec(stream)?, [2]);
    Ok(())
}

fn injected_oob(exec: &Exec<'_>, stream: &CudaStream) -> Result<(), Box<dyn Error>> {
    let hay = DeviceBuffer::from_host(stream, &[1_u32, 2])?;
    let indices = DeviceBuffer::from_host(stream, &[2_usize])?;
    let mut output = DeviceBuffer::from_host(stream, &[u64::MAX])?;
    let pending = lower_bound_async(exec, hay.view(), hay.view().gather(indices.view()), output.view_mut());
    match pending.wait() {
        Err(error) => {
            eprintln!("sorted search range OOB surfaced: {error}");
            Ok(())
        }
        Ok(()) => panic!("sorted search range OOB did not fault"),
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
    boundary_matrix(&execution, &stream)?;
    wide_and_row_keys(&execution, &stream)?;
    composed_ranges(&execution, &stream)?;
    contract_errors(&execution, &stream)?;
    println!("sorted search CPU differential matrix: PASS");
    Ok(())
}
