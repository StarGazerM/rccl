use std::{collections::HashSet, error::Error};

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cuco::{ExactRowSetConfig, ExactSetError, Row};

fn generated(count: usize) -> Vec<Row<4>> {
    (0..count)
        .map(|index| {
            let identity = (index / 3) as u64;
            let mixed = identity.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            Row::new([
                identity as u32,
                (identity >> 32) as u32,
                mixed as u32,
                (mixed >> 32) as u32,
            ])
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let program = rccl_cuco::cuco_program!(&context)?;
    let launch = program.on(&stream)?;
    let sentinel = Row::new([u32::MAX; 4]);

    let empty_rows = DeviceBuffer::<Row<4>>::from_host(&stream, &[])?;
    let mut empty_output = DeviceBuffer::<u32>::zeroed(&stream, 0)?;
    let mut set = ExactRowSetConfig::for_insert_count(20_000, 0.7, sentinel)
        .with_seed(0x5eed)
        .build(&launch)?;
    assert_eq!(set.insert(&empty_rows, &launch)?.inserted, 0);
    set.contains(&empty_rows, &mut empty_output, &launch)?;

    // Row<1> exercises the minimum HISA arity through the general full-row
    // publication path (the packed Row<2> benchmark path is separate).
    let unary_rows = DeviceBuffer::from_host(
        &stream,
        &[Row::new([1]), Row::new([2]), Row::new([1])],
    )?;
    let mut unary = ExactRowSetConfig::with_capacity(8, Row::new([u32::MAX])).build(&launch)?;
    let unary_report = unary.insert(&unary_rows, &launch)?;
    assert_eq!(unary_report.inserted, 2);
    assert_eq!(unary_report.duplicates, 1);
    let unary_queries = DeviceBuffer::from_host(
        &stream,
        &[Row::new([1]), Row::new([2]), Row::new([3])],
    )?;
    let mut unary_output = DeviceBuffer::<u32>::zeroed(&stream, 3)?;
    unary.contains(&unary_queries, &mut unary_output, &launch)?;
    assert_eq!(unary_output.to_host_vec(&stream)?, [1, 1, 0]);

    let input_host = generated(30_000);
    let oracle: HashSet<Row<4>> = input_host.iter().copied().collect();
    assert_eq!(oracle.len(), 10_000);
    let input = DeviceBuffer::from_host(&stream, &input_host)?;
    let report = set.insert(&input, &launch)?;
    assert_eq!(report.inserted, oracle.len());
    assert_eq!(report.duplicates, input_host.len() - oracle.len());
    assert_eq!(report.capacity_exhausted, 0);

    let mut present = DeviceBuffer::<u32>::zeroed(&stream, input.len())?;
    set.contains_async(&input, &mut present, &launch)?.wait()?;
    assert!(present.to_host_vec(&stream)?.into_iter().all(|value| value == 1));

    let absent_host: Vec<Row<4>> = (20_000..30_000)
        .map(|value| Row::new([value, value ^ 0xdead_beef, 17, 29]))
        .collect();
    assert!(absent_host.iter().all(|row| !oracle.contains(row)));
    let absent = DeviceBuffer::from_host(&stream, &absent_host)?;
    let mut absent_output = DeviceBuffer::<u32>::zeroed(&stream, absent.len())?;
    set.contains(&absent, &mut absent_output, &launch)?;
    assert!(absent_output.to_host_vec(&stream)?.into_iter().all(|value| value == 0));

    // A deliberately identical hash for every row proves that the hash is only
    // a probe/filter value: full row equality remains authoritative.
    let collision_host = [
        Row::new([1, 2, 3, 4]),
        Row::new([5, 6, 7, 8]),
        Row::new([9, 10, 11, 12]),
    ];
    let collision_rows = DeviceBuffer::from_host(&stream, &collision_host)?;
    let mut collision_set = ExactRowSetConfig::with_capacity(32, sentinel).build(&launch)?;
    let collision_report = collision_set.insert_forced_hash(&collision_rows, 0xabad_1dea, &launch)?;
    assert_eq!(collision_report.inserted, 3);
    assert_eq!(collision_report.duplicates, 0);
    let collision_queries = DeviceBuffer::from_host(
        &stream,
        &[
            collision_host[0], collision_host[1], collision_host[2],
            Row::new([1, 2, 3, 5]),
        ],
    )?;
    let mut collision_output = DeviceBuffer::<u32>::zeroed(&stream, 4)?;
    collision_set.contains_forced_hash(
        &collision_queries, &mut collision_output, 0xabad_1dea, &launch,
    )?;
    assert_eq!(collision_output.to_host_vec(&stream)?, [1, 1, 1, 0]);

    // The valid extent is exactly eight slots. Overflow is explicit and misses
    // remain false, allowing a caller to route the two rows to an exact fallback.
    let boundary_host: Vec<Row<4>> = (0..10)
        .map(|value| Row::new([value, value + 1, value + 2, value + 3]))
        .collect();
    let boundary_rows = DeviceBuffer::from_host(&stream, &boundary_host)?;
    let mut boundary = ExactRowSetConfig::with_capacity(8, sentinel).build(&launch)?;
    assert_eq!(boundary.capacity(), 8);
    let boundary_report = boundary.insert(&boundary_rows, &launch)?;
    assert_eq!(boundary_report.inserted, 8);
    assert_eq!(boundary_report.capacity_exhausted, 2);
    let mut boundary_output = DeviceBuffer::<u32>::zeroed(&stream, 10)?;
    boundary.contains(&boundary_rows, &mut boundary_output, &launch)?;
    let boundary_presence = boundary_output.to_host_vec(&stream)?;
    assert_eq!(boundary_presence.iter().filter(|value| **value == 1).count(), 8);
    assert_eq!(boundary_presence.iter().filter(|value| **value == 0).count(), 2);

    let sentinel_rows = DeviceBuffer::from_host(&stream, &[sentinel])?;
    match boundary.insert(&sentinel_rows, &launch) {
        Err(ExactSetError::ReservedRows { count: 1 }) => {}
        other => panic!("expected one rejected sentinel row, got {other:?}"),
    }

    boundary.clear(&launch)?;
    let singleton = DeviceBuffer::from_host(&stream, &[Row::new([7, 8, 9, 10])])?;
    boundary.insert_prevalidated_async(&singleton, &launch)?.wait()?;
    let mut singleton_output = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    boundary.contains(&singleton, &mut singleton_output, &launch)?;
    assert_eq!(singleton_output.to_host_vec(&stream)?, [1]);

    println!(
        "Rust exact HISA row set: PASS unique={} duplicates={} absent={} boundary_inserted={} boundary_overflow={} collision_distinct=3 collision_false_positive=0",
        report.inserted,
        report.duplicates,
        absent_host.len(),
        boundary_report.inserted,
        boundary_report.capacity_exhausted,
    );
    Ok(())
}
