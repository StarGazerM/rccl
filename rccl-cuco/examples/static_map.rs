use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cuco::{DoubleHashing, MapError, Sentinels, StaticMapConfig, XxHash32, entries};

fn same_initial_group(capacity: usize) -> (u32, u32) {
    let groups = capacity / 4;
    let hash = XxHash32::default();
    for first in 0_u32..1024 {
        for second in first + 1..1024 {
            if hash.hash(first) as usize % groups == hash.hash(second) as usize % groups {
                return (first, second);
            }
        }
    }
    unreachable!("small u32 keys contain a collision")
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_cuco::cuco_program!(&context)?;
    let launch = program.on(&stream)?;

    let sentinels = Sentinels::new(u32::MAX, u32::MAX).with_erased_key(u32::MAX - 1)?;
    let mut map = StaticMapConfig::for_insert_count(6, 0.5, sentinels).build(&launch)?;

    let keys = DeviceBuffer::from_host(&stream, &[7_u32, 11, 19, 23, 29, 31])?;
    let values = DeviceBuffer::from_host(&stream, &[70_u32, 110, 190, 230, 290, 310])?;
    assert_eq!(map.insert(entries(&keys, &values)?, &launch)?, 6);

    let queries = DeviceBuffer::from_host(&stream, &[31_u32, 8, 7])?;
    let mut found = DeviceBuffer::<u32>::zeroed(&stream, queries.len())?;
    let mut present = DeviceBuffer::<u32>::zeroed(&stream, queries.len())?;
    map.get(&queries, &mut found, &mut present, &launch)?;
    assert_eq!(found.to_host_vec(&stream)?, [310, u32::MAX, 70]);
    assert_eq!(present.to_host_vec(&stream)?, [1, 0, 1]);

    let replacement_key = DeviceBuffer::from_host(&stream, &[7_u32])?;
    let replacement_value = DeviceBuffer::from_host(&stream, &[700_u32])?;
    assert_eq!(
        map.insert_or_assign(entries(&replacement_key, &replacement_value)?, &launch,)?,
        0
    );
    let mut replaced = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    map.find(&replacement_key, &mut replaced, &launch)?;
    assert_eq!(replaced.to_host_vec(&stream)?, [700]);

    let reserved_key = DeviceBuffer::from_host(&stream, &[u32::MAX])?;
    let ordinary_value = DeviceBuffer::from_host(&stream, &[1_u32])?;
    assert!(matches!(
        map.insert(entries(&reserved_key, &ordinary_value)?, &launch),
        Err(MapError::ReservedSentinels { count: 1 })
    ));

    let mut retrieved_keys = DeviceBuffer::<u32>::zeroed(&stream, map.capacity())?;
    let mut retrieved_values = DeviceBuffer::<u32>::zeroed(&stream, map.capacity())?;
    let retrieved = map.retrieve_all(&mut retrieved_keys, &mut retrieved_values, &launch)?;
    assert_eq!(retrieved, 6);
    let mut retrieved_pairs = retrieved_keys
        .to_host_vec(&stream)?
        .into_iter()
        .zip(retrieved_values.to_host_vec(&stream)?)
        .take(retrieved)
        .collect::<Vec<_>>();
    retrieved_pairs.sort_unstable();
    assert_eq!(
        retrieved_pairs,
        [
            (7, 700),
            (11, 110),
            (19, 190),
            (23, 230),
            (29, 290),
            (31, 310)
        ]
    );

    let erased = DeviceBuffer::from_host(&stream, &[11_u32, 23])?;
    assert_eq!(map.remove(&erased, &launch)?, 2);
    assert_eq!(map.len(&launch)?, 4);

    // Regression for NVIDIA/cuCollections#606: a tombstone before an existing
    // colliding key must not be reclaimed until the duplicate search finishes.
    map.clear(&launch)?;
    let (first, second) = same_initial_group(map.capacity());
    let first_key = DeviceBuffer::from_host(&stream, &[first])?;
    let first_value = DeviceBuffer::from_host(&stream, &[10_u32])?;
    let second_key = DeviceBuffer::from_host(&stream, &[second])?;
    let second_value = DeviceBuffer::from_host(&stream, &[20_u32])?;
    assert_eq!(map.insert(entries(&first_key, &first_value)?, &launch)?, 1);
    assert_eq!(
        map.insert(entries(&second_key, &second_value)?, &launch)?,
        1
    );
    assert_eq!(map.remove(&first_key, &launch)?, 1);
    let duplicate_value = DeviceBuffer::from_host(&stream, &[200_u32])?;
    assert_eq!(
        map.insert(entries(&second_key, &duplicate_value)?, &launch)?,
        0
    );
    assert_eq!(map.len(&launch)?, 1);
    map.rehash(20, &launch)?;
    assert_eq!(map.capacity(), 20);
    let mut collision_value = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    map.find(&second_key, &mut collision_value, &launch)?;
    assert_eq!(collision_value.to_host_vec(&stream)?, [20]);

    // A full table terminates rather than inheriting upstream's undefined
    // over-capacity behavior.
    map.clear(&launch)?;
    let full_keys: Vec<u32> = (100..100 + map.capacity() as u32).collect();
    let full_values: Vec<u32> = full_keys.iter().map(|key| key + 1_000).collect();
    let full_keys = DeviceBuffer::from_host(&stream, &full_keys)?;
    let full_values = DeviceBuffer::from_host(&stream, &full_values)?;
    assert_eq!(
        map.insert(entries(&full_keys, &full_values)?, &launch)?,
        map.capacity()
    );
    let overflow_key = DeviceBuffer::from_host(&stream, &[50_000_u32])?;
    let overflow_value = DeviceBuffer::from_host(&stream, &[60_000_u32])?;
    assert_eq!(
        map.insert(entries(&overflow_key, &overflow_value)?, &launch)?,
        0
    );

    // Double hashing uses prime group-count rounding and the same Rust API.
    let mut double = StaticMapConfig::with_capacity(9, sentinels)
        .with_probing(DoubleHashing::default())
        .build(&launch)?;
    assert_eq!(double.capacity(), 12);
    assert_eq!(
        double.insert(entries(&keys, &values)?, &launch)?,
        keys.len()
    );

    let empty = StaticMapConfig::with_capacity(0, sentinels).build(&launch)?;
    assert_eq!(empty.capacity(), 8);

    println!("Rust-native cuCollections static-map correctness matrix: PASS");
    Ok(())
}
