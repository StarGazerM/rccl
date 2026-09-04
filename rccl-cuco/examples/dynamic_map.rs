use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cuco::{DynamicMapConfig, Sentinels, entries};

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_cuco::cuco_program!(&context)?;
    let launch = program.on(&stream)?;

    let sentinels = Sentinels::new(u32::MAX, u32::MAX).with_erased_key(u32::MAX - 1)?;
    let mut map = DynamicMapConfig::new(sentinels)
        .initial_capacity(4)
        .load_factor(0.5)
        .growth_factor(2.0)
        .build(&launch)?;

    let first_keys = DeviceBuffer::from_host(&stream, &[1_u32, 2])?;
    let first_values = DeviceBuffer::from_host(&stream, &[10_u32, 20])?;
    assert_eq!(
        map.insert(entries(&first_keys, &first_values)?, &launch)?,
        2
    );
    assert_eq!(map.len(), 2);
    assert_eq!(map.segment_count(), 1);

    let second_keys = DeviceBuffer::from_host(&stream, &[3_u32, 4, 5])?;
    let second_values = DeviceBuffer::from_host(&stream, &[30_u32, 40, 50])?;
    assert_eq!(
        map.insert(entries(&second_keys, &second_values)?, &launch)?,
        3
    );
    assert_eq!(map.len(), 5);
    assert_eq!(map.segment_count(), 2);

    let queries = DeviceBuffer::from_host(&stream, &[5_u32, 1, 9, 3])?;
    let mut values = DeviceBuffer::<u32>::zeroed(&stream, queries.len())?;
    let mut present = DeviceBuffer::<u32>::zeroed(&stream, queries.len())?;
    map.get(&queries, &mut values, &mut present, &launch)?;
    assert_eq!(values.to_host_vec(&stream)?, [50, 10, u32::MAX, 30]);
    assert_eq!(present.to_host_vec(&stream)?, [1, 1, 0, 1]);

    let replacement_key = DeviceBuffer::from_host(&stream, &[1_u32])?;
    let replacement_value = DeviceBuffer::from_host(&stream, &[100_u32])?;
    assert_eq!(
        map.insert(entries(&replacement_key, &replacement_value)?, &launch)?,
        0
    );
    map.get(&replacement_key, &mut values, &mut present, &launch)?;
    assert_eq!(values.to_host_vec(&stream)?[0], 100);
    assert_eq!(present.to_host_vec(&stream)?[0], 1);

    let remove_keys = DeviceBuffer::from_host(&stream, &[1_u32, 5])?;
    assert_eq!(map.remove(&remove_keys, &launch)?, 2);
    assert_eq!(map.len(), 3);
    map.contains_key(&queries, &mut present, &launch)?;
    assert_eq!(present.to_host_vec(&stream)?, [0, 0, 0, 1]);

    map.reserve(16, &launch)?;
    assert!(map.segment_count() >= 3);
    let old_capacity = map.capacity();
    map.rehash(32, &launch)?;
    assert_eq!(map.segment_count(), 1);
    assert!(map.capacity() >= 32);
    assert!(map.capacity() <= old_capacity + 32);
    map.contains_key(&queries, &mut present, &launch)?;
    assert_eq!(present.to_host_vec(&stream)?, [0, 0, 0, 1]);

    map.clear(&launch)?;
    assert_eq!(map.len(), 0);
    assert!(!map.segments().is_empty());
    map.contains_key(&queries, &mut present, &launch)?;
    assert_eq!(present.to_host_vec(&stream)?, [0, 0, 0, 0]);

    println!("Rust-native dynamic-map segment orchestration probe: PASS");
    Ok(())
}
