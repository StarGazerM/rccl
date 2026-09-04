//! Equal-work add and contains benchmark for the pinned HISA cuCollections policy.

use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cuco::BloomFilterConfig;

fn argument(name: &str, default: usize) -> usize {
    let args = std::env::args().collect::<Vec<_>>();
    args.windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse().ok())
        .unwrap_or(default)
}

fn key(index: usize) -> u64 {
    let mut value = index as u64 + 1;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let items = argument("--items", 8_000_000);
    let blocks = argument("--blocks", 10_000_000);
    let iterations = argument("--iterations", 10);
    if items == 0 || items > u32::MAX as usize || blocks == 0 || iterations == 0 {
        return Err("invalid --items, --blocks, or --iterations".into());
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_cuco::cuco_program!(&context)?;
    let launch = program.on(&stream)?;
    let host_keys = (0..items).map(key).collect::<Vec<_>>();
    let keys = DeviceBuffer::from_host(&stream, &host_keys)?;
    let mut output = DeviceBuffer::<u8>::zeroed(&stream, items)?;
    let mut filter = BloomFilterConfig::with_num_blocks(blocks).build::<u64, _>(&launch)?;

    for _ in 0..3 {
        filter.clear(&launch)?;
        filter.add(&keys, &launch)?;
        filter.contains(&keys, &mut output, &launch)?;
    }

    let start = stream.context().new_event(Some(0))?;
    let end = stream.context().new_event(Some(0))?;
    let mut add_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        filter.clear(&launch)?;
        start.record(&stream)?;
        let pending = filter.add_async(&keys, &launch)?;
        end.record(&stream)?;
        end.synchronize()?;
        add_times.push(start.elapsed_ms(&end)? as f64);
        pending.wait()?;
    }

    let mut contains_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        start.record(&stream)?;
        let pending = filter.contains_async(&keys, &mut output, &launch)?;
        end.record(&stream)?;
        end.synchronize()?;
        contains_times.push(start.elapsed_ms(&end)? as f64);
        pending.wait()?;
    }
    assert!(
        output
            .to_host_vec(&stream)?
            .into_iter()
            .all(|value| value == 1)
    );
    let mixed_host = (0..items)
        .map(|index| {
            if index % 2 == 0 {
                host_keys[index]
            } else {
                key(index + items * 3 + 17)
            }
        })
        .collect::<Vec<_>>();
    let mixed = DeviceBuffer::from_host(&stream, &mixed_host)?;
    filter.contains(&mixed, &mut output, &launch)?;
    let checksum = output
        .to_host_vec(&stream)?
        .into_iter()
        .fold(0xcbf2_9ce4_8422_2325u64, |state, byte| {
            (state ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        });

    println!(
        "implementation=rccl-cuco algorithm=bloom type=u64 items={items} blocks={blocks} checksum={checksum:016x} add_median_ms={:.9} contains_median_ms={:.9}",
        median(add_times),
        median(contains_times),
    );
    Ok(())
}
