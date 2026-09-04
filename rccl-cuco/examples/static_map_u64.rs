use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cuco::{MapError, Sentinels, StaticMapConfig, XxHash32, entries};

fn compress_u32(start: u32, len: u32) -> u64 {
    ((start as u64) << 32) | len as u64
}

fn same_initial_group(capacity: usize) -> (u32, u32) {
    let groups = capacity / 4;
    let hash = XxHash32::default();
    for first in 0_u32..4096 {
        for second in first + 1..4096 {
            if hash.hash(first) as usize % groups == hash.hash(second) as usize % groups {
                return (first, second);
            }
        }
    }
    unreachable!("small u32 keys contain a collision")
}

fn sorted_pairs(keys: Vec<u32>, values: Vec<u64>, take: usize) -> Vec<(u32, u64)> {
    let mut pairs = keys.into_iter().zip(values).take(take).collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_cuco::cuco_program!(&context)?;
    let launch = program.on(&stream)?;

    // vflog::CREATE_V_MAP uses u32 keys and u64 compressed ranges with the
    // empty value sentinel UINT32_MAX widened to uint64_t.
    let empty_key = u32::MAX;
    let erased_key = u32::MAX - 1;
    let empty_value = u32::MAX as u64;
    let sentinels = Sentinels::new(empty_key, empty_value).with_erased_key(erased_key)?;
    let mut map = StaticMapConfig::for_insert_count(10, 0.5, sentinels).build(&launch)?;

    let keys = [7_u32, 11, 19, 23, 29, 31, 37, 41];
    let values = [
        compress_u32(0, 1),
        compress_u32(1, 2),
        compress_u32(0x7fff_fffe, 0xffff_fffe),
        compress_u32(0x8000_0000, 3),
        compress_u32(0xffff_fffe, 0xffff_fffe),
        u64::MAX - 5,
        compress_u32(12_345, 67_890),
        (1_u64 << 63) | 17,
    ];
    let key_buf = DeviceBuffer::from_host(&stream, &keys)?;
    let value_buf = DeviceBuffer::from_host(&stream, &values)?;
    assert_eq!(
        map.insert(entries(&key_buf, &value_buf)?, &launch)?,
        keys.len()
    );

    let queries = DeviceBuffer::from_host(&stream, &[31_u32, 8, 7, 23, 41])?;
    let mut found = DeviceBuffer::<u64>::zeroed(&stream, queries.len())?;
    let mut present = DeviceBuffer::<u32>::zeroed(&stream, queries.len())?;
    map.get(&queries, &mut found, &mut present, &launch)?;
    assert_eq!(
        found.to_host_vec(&stream)?,
        [values[5], empty_value, values[0], values[3], values[7]]
    );
    assert_eq!(present.to_host_vec(&stream)?, [1, 0, 1, 1, 1]);

    let duplicate_key = DeviceBuffer::from_host(&stream, &[7_u32])?;
    let rejected_value = DeviceBuffer::from_host(&stream, &[compress_u32(99, 99)])?;
    assert_eq!(
        map.insert(entries(&duplicate_key, &rejected_value)?, &launch)?,
        0
    );
    let mut duplicate_found = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    map.find(&duplicate_key, &mut duplicate_found, &launch)?;
    assert_eq!(duplicate_found.to_host_vec(&stream)?, [values[0]]);

    let assigned_value =
        DeviceBuffer::from_host(&stream, &[compress_u32(0xffff_fffd, 0xffff_fffc)])?;
    assert_eq!(
        map.insert_or_assign(entries(&duplicate_key, &assigned_value)?, &launch)?,
        0
    );
    map.find(&duplicate_key, &mut duplicate_found, &launch)?;
    assert_eq!(
        duplicate_found.to_host_vec(&stream)?,
        [compress_u32(0xffff_fffd, 0xffff_fffc)]
    );

    let absent = DeviceBuffer::from_host(&stream, &[123_456_u32])?;
    let mut absent_value = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    map.find(&absent, &mut absent_value, &launch)?;
    assert_eq!(absent_value.to_host_vec(&stream)?, [empty_value]);

    let reserved_value = DeviceBuffer::from_host(&stream, &[empty_value])?;
    assert!(matches!(
        map.insert(entries(&duplicate_key, &reserved_value)?, &launch),
        Err(MapError::ReservedSentinels { count: 1 })
    ));

    let mut retrieved_keys = DeviceBuffer::<u32>::zeroed(&stream, map.capacity())?;
    let mut retrieved_values = DeviceBuffer::<u64>::zeroed(&stream, map.capacity())?;
    let retrieved = map.retrieve_all(&mut retrieved_keys, &mut retrieved_values, &launch)?;
    assert_eq!(retrieved, keys.len());
    let mut expected = keys.into_iter().zip(values).collect::<Vec<_>>();
    expected[0].1 = compress_u32(0xffff_fffd, 0xffff_fffc);
    expected.sort_unstable();
    assert_eq!(
        sorted_pairs(
            retrieved_keys.to_host_vec(&stream)?,
            retrieved_values.to_host_vec(&stream)?,
            retrieved,
        ),
        expected
    );

    let erased = DeviceBuffer::from_host(&stream, &[11_u32, 23, 123_456])?;
    assert_eq!(map.remove(&erased, &launch)?, 2);
    assert_eq!(map.len(&launch)?, keys.len() - 2);

    let reinsertion_keys = DeviceBuffer::from_host(&stream, &[11_u32, 23])?;
    let reinsertion_values = DeviceBuffer::from_host(
        &stream,
        &[compress_u32(0x8000_0001, 4), compress_u32(0, 0xffff_fffe)],
    )?;
    assert_eq!(
        map.insert(entries(&reinsertion_keys, &reinsertion_values)?, &launch)?,
        2
    );
    assert_eq!(map.len(&launch)?, keys.len());
    map.rehash(40, &launch)?;
    assert!(map.capacity() >= 40);
    let mut rehashed = DeviceBuffer::<u64>::zeroed(&stream, reinsertion_keys.len())?;
    map.find(&reinsertion_keys, &mut rehashed, &launch)?;
    assert_eq!(
        rehashed.to_host_vec(&stream)?,
        [compress_u32(0x8000_0001, 4), compress_u32(0, 0xffff_fffe)]
    );

    // Regression for reclaiming a tombstone before finishing duplicate search,
    // now on the two-word u64-value protocol.
    map.clear(&launch)?;
    let (first, second) = same_initial_group(map.capacity());
    let first_key = DeviceBuffer::from_host(&stream, &[first])?;
    let second_key = DeviceBuffer::from_host(&stream, &[second])?;
    let first_value = DeviceBuffer::from_host(&stream, &[compress_u32(42, 1)])?;
    let second_value = DeviceBuffer::from_host(&stream, &[compress_u32(43, 2)])?;
    assert_eq!(map.insert(entries(&first_key, &first_value)?, &launch)?, 1);
    assert_eq!(
        map.insert(entries(&second_key, &second_value)?, &launch)?,
        1
    );
    assert_eq!(map.remove(&first_key, &launch)?, 1);
    let duplicate_second = DeviceBuffer::from_host(&stream, &[compress_u32(44, 3)])?;
    assert_eq!(
        map.insert(entries(&second_key, &duplicate_second)?, &launch)?,
        0
    );
    let mut second_found = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    map.find(&second_key, &mut second_found, &launch)?;
    assert_eq!(second_found.to_host_vec(&stream)?, [compress_u32(43, 2)]);

    let high_count = 230_usize;
    let mut high = StaticMapConfig::for_insert_count(high_count, 0.9, sentinels).build(&launch)?;
    let high_capacity = high.capacity();
    let high_keys = (0..high_count as u32)
        .map(|k| k + 1_000)
        .collect::<Vec<_>>();
    let high_values = high_keys
        .iter()
        .enumerate()
        .map(|(idx, key)| compress_u32(0x8000_0000 | idx as u32, key.wrapping_mul(17)))
        .collect::<Vec<_>>();
    let high_key_buf = DeviceBuffer::from_host(&stream, &high_keys)?;
    let high_value_buf = DeviceBuffer::from_host(&stream, &high_values)?;
    assert_eq!(
        high.insert(entries(&high_key_buf, &high_value_buf)?, &launch)?,
        high_count
    );
    assert_eq!(high.len(&launch)?, high_count);
    assert!((high_count as f64 / high_capacity as f64) >= 0.85);
    let mut high_found = DeviceBuffer::<u64>::zeroed(&stream, high_keys.len())?;
    high.find(&high_key_buf, &mut high_found, &launch)?;
    assert_eq!(high_found.to_host_vec(&stream)?, high_values);

    println!("Rust-native cuCollections static-map u32->u64 compressed-range matrix: PASS");
    Ok(())
}
