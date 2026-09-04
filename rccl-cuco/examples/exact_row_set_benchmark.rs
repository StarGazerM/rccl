//! Equal-work benchmark for the pinned cuCollections `static_set<Row<2>>` oracle.

use std::error::Error;

use cuda_core::{CudaContext, DeviceBuffer};
use rccl_cuco::{ExactRowSetConfig, Row};

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

fn string_argument<'a>(arguments: &'a [String], name: &str, default: &'a str) -> &'a str {
    arguments
        .windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].as_str())
        .unwrap_or(default)
}

fn row(index: u32) -> Row<2> {
    let mut value = index.wrapping_add(1);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    Row::new([value, value ^ 0x9e37_79b9])
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    let items = argument("--items", 8_000_000);
    let iterations = argument("--iterations", 10);
    let occupancy = float_argument("--occupancy", 0.5);
    let operation = string_argument(&arguments, "--operation", "insert");
    if items == 0
        || items >= u32::MAX as usize
        || iterations == 0
        || !occupancy.is_finite()
        || occupancy <= 0.0
        || occupancy > 1.0
        || !matches!(operation, "insert" | "contains")
    {
        return Err("invalid benchmark arguments".into());
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_cuco::cuco_program!(&context)?;
    let launch = program.on(&stream)?;
    let host_rows = (0..items)
        .map(|index| row(index as u32))
        .collect::<Vec<_>>();
    let rows = DeviceBuffer::from_host(&stream, &host_rows)?;
    let requested_capacity = (items as f64 / occupancy) as usize;
    let mut set = ExactRowSetConfig::with_capacity(requested_capacity, Row::new([u32::MAX; 2]))
        .build(&launch)?;
    let mut output = DeviceBuffer::<u32>::zeroed(&stream, items)?;

    if operation == "contains" {
        set.insert_prevalidated(&rows, &launch)?;
    }
    for _ in 0..3 {
        if operation == "insert" {
            set.clear(&launch)?;
            set.insert_prevalidated(&rows, &launch)?;
        } else {
            set.contains(&rows, &mut output, &launch)?;
        }
    }

    let start = stream.context().new_event(Some(0))?;
    let end = stream.context().new_event(Some(0))?;
    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        if operation == "insert" {
            set.clear(&launch)?;
        }
        start.record(&stream)?;
        let pending = if operation == "insert" {
            set.insert_prevalidated_async(&rows, &launch)?
        } else {
            set.contains_async(&rows, &mut output, &launch)?
        };
        end.record(&stream)?;
        end.synchronize()?;
        timings.push(start.elapsed_ms(&end)? as f64);
        pending.wait()?;
    }
    if operation == "contains" {
        assert!(
            output
                .to_host_vec(&stream)?
                .into_iter()
                .all(|value| value == 1)
        );
    }
    println!(
        "implementation=rccl-cuco algorithm=exact_row_set_{operation} type=row2_u32 items={items} occupancy={occupancy:.2} capacity={} median_ms={:.9}",
        set.capacity(),
        median(timings),
    );
    Ok(())
}
