//! Equal-work insertion benchmark for pinned cuCollections `static_map<u32, u64>`.

use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cuco::{LinearProbing, Sentinels, StaticMapConfig, entries};

fn argument(name: &str, default: usize) -> usize {
    let arguments = std::env::args().collect::<Vec<_>>();
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse().ok())
        .unwrap_or(default)
}

fn float_argument(name: &str, default: f64) -> f64 {
    let arguments = std::env::args().collect::<Vec<_>>();
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse().ok())
        .unwrap_or(default)
}

fn key(index: u32) -> u32 {
    let mut value = index.wrapping_add(1);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn compressed_value(index: u32, key: u32) -> u64 {
    ((0x8000_0000_u64 | index as u64) << 32) | key.wrapping_mul(17) as u64
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let items = argument("--items", 8_000_000);
    let iterations = argument("--iterations", 10);
    let occupancy = float_argument("--occupancy", 0.5);
    if items == 0
        || items >= u32::MAX as usize
        || iterations == 0
        || !occupancy.is_finite()
        || occupancy <= 0.0
        || occupancy > 1.0
    {
        return Err("invalid --items, --iterations, or --occupancy".into());
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_cuco::cuco_program!(&context)?;
    let launch = program.on(&stream)?;
    let host_keys = (0..items)
        .map(|index| key(index as u32))
        .collect::<Vec<_>>();
    let host_values = host_keys
        .iter()
        .enumerate()
        .map(|(index, value)| compressed_value(index as u32, *value))
        .collect::<Vec<_>>();
    let keys = DeviceBuffer::from_host(&stream, &host_keys)?;
    let values = DeviceBuffer::from_host(&stream, &host_values)?;
    let sentinels = Sentinels::new(u32::MAX, u32::MAX as u64).with_erased_key(u32::MAX - 1)?;
    let requested_capacity = (items as f64 / occupancy) as usize;
    let mut map = StaticMapConfig::with_capacity(requested_capacity, sentinels)
        .with_probing(LinearProbing::default())
        .build(&launch)?;

    for _ in 0..3 {
        map.clear(&launch)?;
        map.insert_prevalidated(entries(&keys, &values)?, &launch)?;
    }

    let start = stream.context().new_event(Some(0))?;
    let end = stream.context().new_event(Some(0))?;
    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        map.clear(&launch)?;
        start.record(&stream)?;
        let pending = map.insert_prevalidated_async(entries(&keys, &values)?, &launch)?;
        end.record(&stream)?;
        end.synchronize()?;
        timings.push(start.elapsed_ms(&end)? as f64);
        pending.wait()?;
    }

    assert_eq!(map.len(&launch)?, items);
    println!(
        "implementation=rccl-cuco algorithm=static_map_insert type=u32_u64 items={items} occupancy={occupancy:.2} capacity={} median_ms={:.9}",
        map.capacity(),
        median(timings),
    );
    Ok(())
}
