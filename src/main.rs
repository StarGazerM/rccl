use cuda_core::{CudaContext, DeviceBuffer};
use rccl::{GpuSorter, SortOrder};
use std::{error::Error, time::Duration};

const DEFAULT_BYTES: usize = 4 * 1024 * 1024 * 1024;

fn parse_usize(value: &str) -> Result<usize, Box<dyn Error>> {
    let (digits, multiplier) = if let Some(value) = value.strip_suffix("GiB") {
        (value, 1024usize.pow(3))
    } else if let Some(value) = value.strip_suffix("MiB") {
        (value, 1024usize.pow(2))
    } else if let Some(value) = value.strip_suffix('G') {
        (value, 1_000_000_000)
    } else if let Some(value) = value.strip_suffix('M') {
        (value, 1_000_000)
    } else {
        (value, 1)
    };
    Ok(digits
        .parse::<usize>()?
        .checked_mul(multiplier)
        .ok_or("size overflow")?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut bytes = match std::env::var("RCCL_ELEMENTS") {
        Ok(value) => parse_usize(&value)?
            .checked_mul(size_of::<u32>())
            .ok_or("size overflow")?,
        Err(_) => std::env::var("RCCL_BYTES")
            .ok()
            .map(|value| parse_usize(&value))
            .transpose()?
            .unwrap_or(DEFAULT_BYTES),
    };
    let mut iterations = std::env::var("RCCL_ITERATIONS")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(3usize);
    let mut order = SortOrder::Ascending;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bytes" => bytes = parse_usize(&args.next().ok_or("--bytes needs a value")?)?,
            "--elements" => {
                let elements = parse_usize(&args.next().ok_or("--elements needs a value")?)?;
                bytes = elements
                    .checked_mul(size_of::<u32>())
                    .ok_or("size overflow")?;
            }
            "--iterations" => {
                iterations = args.next().ok_or("--iterations needs a value")?.parse()?;
            }
            "--descending" => order = SortOrder::Descending,
            "--help" | "-h" => {
                println!(
                    "rccl GPU-resident radix benchmark\n\n\
                     Usage: rccl [--bytes 4GiB | --elements N] [--iterations N] [--descending]\n\
                     Environment: RCCL_BYTES, RCCL_ELEMENTS, RCCL_ITERATIONS\n\
                     The key array is generated, sorted, and validated on the GPU."
                );
                return Ok(());
            }
            unknown => return Err(format!("unknown argument `{unknown}`").into()),
        }
    }
    if bytes % size_of::<u32>() != 0 {
        return Err("--bytes must be a multiple of four for u32 keys".into());
    }
    if iterations == 0 {
        return Err("--iterations must be non-zero".into());
    }
    let elements = bytes / size_of::<u32>();

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let sorter = GpuSorter::new(&context)?;
    let config = sorter.policy_config();
    let mut keys = DeviceBuffer::<u32>::zeroed(&stream, elements)?;
    let mut workspace = sorter.allocate_u32_workspace(&stream, elements)?;

    println!("GPU: {}", context.device_name()?);
    println!(
        "policy: {} (sm_{}, {} threads x {} items, {} radix bits)",
        sorter.policy_name(),
        sorter.sm(),
        config.block_threads,
        config.items_per_thread,
        config.radix_bits,
    );
    println!(
        "problem: {} u32 keys ({:.3} GiB), device temporary storage {:.3} GiB",
        elements,
        bytes as f64 / 1024f64.powi(3),
        workspace.temporary_bytes() as f64 / 1024f64.powi(3),
    );

    // Warm-up compiles/loads the selected specialization and heats caches. The
    // refill is also a device kernel and is deliberately outside sort timing.
    sorter.fill_random_u32(&stream, &mut keys, 0x7263_636c)?;
    sorter.sort_u32(&stream, &mut keys, &mut workspace, order)?;
    stream.synchronize()?;

    let mut best = Duration::MAX;
    let mut total_ms = 0.0f64;
    for iteration in 0..iterations {
        sorter.fill_random_u32(&stream, &mut keys, 0x7263_636c_u64 + iteration as u64 + 1)?;
        let start = stream.record_event(Some(0))?;
        sorter.sort_u32(&stream, &mut keys, &mut workspace, order)?;
        let end = stream.record_event(Some(0))?;
        let elapsed_ms = start.elapsed_ms(&end)? as f64;
        let elapsed = Duration::from_secs_f64(elapsed_ms / 1000.0);
        best = best.min(elapsed);
        total_ms += elapsed_ms;
        println!("iteration {iteration}: {elapsed_ms:.3} ms");
    }

    let order_errors = sorter.validate_u32(&stream, &keys, &mut workspace, order)?;
    if order_errors != 0 {
        return Err(format!("device validation found {order_errors} adjacent inversions").into());
    }

    let best_s = best.as_secs_f64();
    let billion_keys_s = elements as f64 / best_s / 1.0e9;
    let effective_gib_s = bytes as f64 / best_s / 1024f64.powi(3);
    println!("validation: PASS (device order check; no key readback)");
    println!(
        "best: {:.3} ms, {:.3} Gkey/s, {:.3} GiB/s input rate; mean {:.3} ms",
        best_s * 1000.0,
        billion_keys_s,
        effective_gib_s,
        total_ms / iterations as f64,
    );
    Ok(())
}
