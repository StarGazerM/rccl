use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use rccl_thrust::{
    DeviceBufferExt, DeviceRangeExt, count_ranges, expand_ranges_into, expand_ranges_into_async,
    expand_ranges_workspace, sum_ranges_into, sum_ranges_into_async,
};
use std::error::Error;
type Exec<'a> = rccl_thrust::cuda::Execution<'a, 'a, rccl_memory::BorrowedCudaMemoryResource<'a>>;

fn check(
    exec: &Exec<'_>,
    stream: &CudaStream,
    ranges: &[(u64, u64)],
) -> Result<(), Box<dyn Error>> {
    let begins: Vec<_> = ranges.iter().map(|r| r.0).collect();
    let ends: Vec<_> = ranges.iter().map(|r| r.1).collect();
    let size = ends.iter().copied().max().unwrap_or(0) as usize;
    let values: Vec<u64> = (0..size)
        .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
        .collect();
    let mut pairs = Vec::new();
    let mut sums = Vec::new();
    for (i, &(begin, end)) in ranges.iter().enumerate() {
        for row in begin..end {
            pairs.push((i as u64, row));
        }
        sums.push(
            values[begin as usize..end as usize]
                .iter()
                .fold(0u64, |s, &v| s.wrapping_add(v)),
        );
    }
    let b = DeviceBuffer::from_host(stream, &begins)?;
    let e = DeviceBuffer::from_host(stream, &ends)?;
    let v = DeviceBuffer::from_host(stream, &values)?;
    let mut p = DeviceBuffer::from_host(stream, &vec![u64::MAX; pairs.len() + 3])?;
    let mut r = DeviceBuffer::from_host(stream, &vec![u64::MAX; pairs.len() + 3])?;
    let mut out = DeviceBuffer::from_host(stream, &vec![19u64; ranges.len()])?;
    let mut workspace = expand_ranges_workspace(exec, ranges.len())?;
    assert_eq!(
        count_ranges(exec, b.view(), e.view(), &mut workspace)?,
        pairs.len() as u64
    );
    expand_ranges_into(
        exec,
        b.view(),
        e.view(),
        p.view_mut(),
        r.view_mut(),
        &mut workspace,
    )?;
    let actual: Vec<_> = p
        .to_host_vec(stream)?
        .into_iter()
        .zip(r.to_host_vec(stream)?)
        .collect();
    assert_eq!(&actual[..pairs.len()], &pairs);
    assert_eq!(&actual[pairs.len()..], &[(u64::MAX, u64::MAX); 3]);
    drop(expand_ranges_into_async(
        exec,
        b.view(),
        e.view(),
        p.view_mut(),
        r.view_mut(),
        &mut workspace,
    ));
    sum_ranges_into(
        exec,
        v.view(),
        b.view(),
        e.view(),
        out.view_mut(),
        &mut workspace,
    )?;
    assert_eq!(out.to_host_vec(stream)?, sums);
    let mut affine = DeviceBuffer::from_host(stream, &vec![71u64; ranges.len() * 2])?;
    sum_ranges_into_async(
        exec,
        v.view().map(|x: u64| x.wrapping_mul(3)),
        b.view(),
        e.view(),
        affine.view_mut().step_by(2)?,
        &mut workspace,
    )
    .wait()?;
    let expected: Vec<_> = sums.iter().flat_map(|s| [s.wrapping_mul(3), 71]).collect();
    assert_eq!(affine.to_host_vec(stream)?, expected);
    Ok(())
}

fn fault(exec: &Exec<'_>, stream: &CudaStream, kind: &str) -> Result<(), Box<dyn Error>> {
    let (begins, ends, capacity) = match kind {
        "reverse" => (vec![5u64], vec![4u64], 1),
        "overflow" => (vec![0, 0], vec![u64::MAX, 1], 1),
        "overflow-tiles" => {
            let mut ends = vec![0; 257];
            ends[0] = u64::MAX;
            ends[256] = 1;
            (vec![0; 257], ends, 1)
        }
        "capacity" => (vec![0], vec![10], 9),
        "sum-oob" => (vec![0], vec![2], 1),
        _ => return Err("unknown fault".into()),
    };
    let b = DeviceBuffer::from_host(stream, &begins)?;
    let e = DeviceBuffer::from_host(stream, &ends)?;
    let mut p = DeviceBuffer::<u64>::zeroed(stream, capacity)?;
    let mut r = DeviceBuffer::<u64>::zeroed(stream, capacity)?;
    let mut workspace = expand_ranges_workspace(exec, begins.len())?;
    let result = if kind == "sum-oob" {
        let v = DeviceBuffer::from_host(stream, &[1u64])?;
        sum_ranges_into_async(
            exec,
            v.view(),
            b.view(),
            e.view(),
            p.view_mut(),
            &mut workspace,
        )
        .wait()
    } else {
        expand_ranges_into_async(
            exec,
            b.view(),
            e.view(),
            p.view_mut(),
            r.view_mut(),
            &mut workspace,
        )
        .wait()
    };
    assert!(result.is_err(), "{kind} was not rejected");
    println!("checked expansion fault {kind}: PASS");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_thrust::cuda::Program::load(&context)?;
    let exec = program.par(&stream)?;
    if let Some(kind) = std::env::args().nth(1) {
        return fault(&exec, &stream, &kind);
    }
    for n in [0, 1, 31, 32, 33, 255, 256, 257, 65537] {
        let ranges: Vec<_> = (0..n)
            .map(|i| {
                let b = i as u64 % 97;
                (b, b + (i as u64 % 8))
            })
            .collect();
        check(&exec, &stream, &ranges)?;
    }
    check(&exec, &stream, &[(0, 0), (0, 0), (17, 17)])?;
    check(&exec, &stream, &[(0, 33), (0, 33), (1, 32), (5, 5)])?;
    let mut skew = vec![(0, 1_000_000)];
    skew.extend((0..10000).map(|i| (i, i + 1)));
    check(&exec, &stream, &skew)?;
    let b = DeviceBuffer::from_host(&stream, &[0u64])?;
    let e = DeviceBuffer::from_host(&stream, &[0u64, 0])?;
    let mut p = DeviceBuffer::from_host(&stream, &[77u64])?;
    let mut r = DeviceBuffer::from_host(&stream, &[88u64])?;
    let mut workspace = expand_ranges_workspace(&exec, 1)?;
    assert!(matches!(
        expand_ranges_into(
            &exec,
            b.view(),
            e.view(),
            p.view_mut(),
            r.view_mut(),
            &mut workspace
        ),
        Err(rccl_thrust::DeviceAlgorithmError::LengthMismatch { .. })
    ));
    let mut too_small = expand_ranges_workspace(&exec, 0)?;
    assert!(matches!(
        expand_ranges_into(
            &exec,
            b.view(),
            b.view(),
            p.view_mut(),
            r.view_mut(),
            &mut too_small
        ),
        Err(rccl_thrust::DeviceAlgorithmError::WorkspaceTooSmall { .. })
    ));
    assert_eq!(p.to_host_vec(&stream)?, [77]);
    assert_eq!(r.to_host_vec(&stream)?, [88]);
    println!("checked expansion and fused sums CPU oracle: PASS");
    Ok(())
}
