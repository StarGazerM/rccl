use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cuco::{BloomFilterConfig, memory::RecordedCudaEvent};

fn generated(count: usize, salt: u64) -> Vec<u64> {
    (0..count)
        .map(|index| {
            let mut value = index as u64 ^ salt;
            value ^= value >> 30;
            value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value ^= value >> 27;
            value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^ (value >> 31)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let program = rccl_cuco::cuco_program!(&context)?;
    let launch = program.on(&stream)?;

    let mut filter = BloomFilterConfig::for_capacity(100_000, 0.01)
        .with_seed(0x5eed)
        .build::<u64, _>(&launch)?;

    let empty = DeviceBuffer::<u64>::from_host(&stream, &[])?;
    let mut empty_output = DeviceBuffer::<u8>::zeroed(&stream, 0)?;
    filter.add(&empty, &launch)?;
    filter.contains(&empty, &mut empty_output, &launch)?;

    let singleton = DeviceBuffer::from_host(&stream, &[7u64])?;
    filter.add(&singleton, &launch)?;
    let mut singleton_output = DeviceBuffer::<u8>::zeroed(&stream, 1)?;
    filter.contains(&singleton, &mut singleton_output, &launch)?;
    assert_eq!(singleton_output.to_host_vec(&stream)?, [1]);

    filter.clear(&launch)?;
    let inserted_host = generated(100_000, 0x1234_5678);
    let inserted = DeviceBuffer::from_host(&stream, &inserted_host)?;
    filter.add(&inserted, &launch)?;
    let mut present = DeviceBuffer::<u8>::zeroed(&stream, inserted.len())?;
    filter.contains(&inserted, &mut present, &launch)?;
    assert!(
        present
            .to_host_vec(&stream)?
            .into_iter()
            .all(|value| value == 1)
    );

    let absent_host = generated(100_000, 0xfeed_face_dead_beef);
    let absent = DeviceBuffer::from_host(&stream, &absent_host)?;
    let mut absent_output = DeviceBuffer::<u8>::zeroed(&stream, absent.len())?;
    filter.contains(&absent, &mut absent_output, &launch)?;
    let false_positives = absent_output
        .to_host_vec(&stream)?
        .into_iter()
        .filter(|value| *value != 0)
        .count();
    let measured_fpr = false_positives as f64 / absent.len() as f64;
    assert!(
        measured_fpr <= 0.01,
        "measured FPR {measured_fpr} exceeds configured 0.01 bound"
    );

    filter.clear(&launch)?;
    let duplicate_host = vec![0xabad_1deau64; 65_536];
    let duplicates = DeviceBuffer::from_host(&stream, &duplicate_host)?;
    filter.add(&duplicates, &launch)?;
    let mut duplicate_output = DeviceBuffer::<u8>::zeroed(&stream, duplicates.len())?;
    filter.contains(&duplicates, &mut duplicate_output, &launch)?;
    assert!(
        duplicate_output
            .to_host_vec(&stream)?
            .into_iter()
            .all(|value| value == 1)
    );

    filter.clear(&launch)?;
    let skewed_host: Vec<u64> = (0..65_536).map(|index| (index % 17) as u64).collect();
    let skewed = DeviceBuffer::from_host(&stream, &skewed_host)?;
    filter.add(&skewed, &launch)?;
    let mut skewed_output = DeviceBuffer::<u8>::zeroed(&stream, skewed.len())?;
    filter.contains(&skewed, &mut skewed_output, &launch)?;
    assert!(
        skewed_output
            .to_host_vec(&stream)?
            .into_iter()
            .all(|v| v == 1)
    );

    // A one-block filter is the maximum-occupancy boundary: publication remains
    // race-free and cannot create false negatives even when every key contends.
    let mut saturated = BloomFilterConfig::with_num_blocks(1).build::<u64, _>(&launch)?;
    let saturated_host = generated(32_768, 99);
    let saturated_keys = DeviceBuffer::from_host(&stream, &saturated_host)?;
    saturated.add(&saturated_keys, &launch)?;
    let mut saturated_output = DeviceBuffer::<u8>::zeroed(&stream, saturated_keys.len())?;
    saturated.contains(&saturated_keys, &mut saturated_output, &launch)?;
    assert!(
        saturated_output
            .to_host_vec(&stream)?
            .into_iter()
            .all(|value| value == 1)
    );

    for seed in [0, 1, u64::MAX, 0x0123_4567_89ab_cdef] {
        let mut seeded = BloomFilterConfig::with_num_blocks(257)
            .with_seed(seed)
            .build::<u64, _>(&launch)?;
        seeded.add(&singleton, &launch)?;
        seeded.contains(&singleton, &mut singleton_output, &launch)?;
        assert_eq!(singleton_output.to_host_vec(&stream)?, [1]);
    }

    // Deterministic negative control: omitting one of the eight publications
    // must make the ordinary no-false-negative assertion fail.
    let mut faulty = BloomFilterConfig::with_num_blocks(1024)
        .with_seed(17)
        .build::<u64, _>(&launch)?;
    faulty.add_omitting_word_for_test(&singleton, 3, &launch)?;
    faulty.contains(&singleton, &mut singleton_output, &launch)?;
    assert_eq!(singleton_output.to_host_vec(&stream)?, [0]);

    // Independent stream-bound owners submit concurrently. Events prove both
    // producer completion points without sharing or detaching borrowed storage.
    let other_stream = context.new_stream()?;
    let other_launch = program.on(&other_stream)?;
    let mut left = BloomFilterConfig::with_num_blocks(4096).build::<u64, _>(&launch)?;
    let mut right = BloomFilterConfig::with_num_blocks(4096).build::<u64, _>(&other_launch)?;
    {
        let left_pending = left.add_async(&inserted, &launch)?;
        let right_keys = DeviceBuffer::from_host(&other_stream, &inserted_host)?;
        let right_pending = right.add_async(&right_keys, &other_launch)?;
        left_pending.wait()?;
        right_pending.wait()?;
    }
    let left_done = RecordedCudaEvent::record(&stream)?;
    let right_done = RecordedCudaEvent::record(&other_stream)?;
    left_done.synchronize()?;
    right_done.synchronize()?;

    println!(
        "Rust HISA Bloom correctness matrix: PASS (blocks={}, measured_fpr={measured_fpr:.6})",
        filter.num_blocks()
    );
    Ok(())
}
