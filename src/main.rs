use cuda_core::{CudaContext, DeviceBuffer};
use std::{error::Error, time::Duration};

#[path = "../examples/gpu_support.rs"]
mod gpu_support;

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
    let mut descending = false;
    let mut pattern = gpu_support::U32Pattern::Random;
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
            "--descending" => descending = true,
            "--pattern" => {
                pattern = match args.next().ok_or("--pattern needs a value")?.as_str() {
                    "random" => gpu_support::U32Pattern::Random,
                    "uniform" => gpu_support::U32Pattern::Uniform,
                    "ascending" => gpu_support::U32Pattern::Ascending,
                    "descending" => gpu_support::U32Pattern::Descending,
                    "low-entropy" => gpu_support::U32Pattern::LowEntropy,
                    "alternating" => gpu_support::U32Pattern::Alternating,
                    value => return Err(format!("unknown pattern `{value}`").into()),
                };
            }
            "--help" | "-h" => {
                println!(
                    "rccl public-facade GPU sort benchmark\n\n\
                     Usage: rccl [--bytes 4GiB | --elements N] [--iterations N] [--descending]\n\
                     \x20      [--pattern random|uniform|ascending|descending|low-entropy|alternating]"
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
    let support = gpu_support::GpuSupport::new(&context)?;
    let exec = rccl::cuda::par(&stream)?;
    let mut keys = DeviceBuffer::<u32>::zeroed(&stream, elements)?;

    println!("GPU: {}", context.device_name()?);
    println!(
        "public operation: {} on {} u32 keys ({:.3} GiB)",
        if descending {
            "sort_by(descending closure)"
        } else {
            "sort"
        },
        elements,
        bytes as f64 / 1024f64.powi(3),
    );

    support.fill_pattern_u32(&stream, &mut keys, 0x7263_636c, pattern)?;
    if descending {
        rccl::sort_by(&exec, &mut keys, |left: &u32, right: &u32| right.cmp(left))?;
    } else {
        rccl::sort(&exec, &mut keys)?;
    }

    let mut best = Duration::MAX;
    let mut total_ms = 0.0;
    for iteration in 0..iterations {
        support.fill_pattern_u32(
            &stream,
            &mut keys,
            0x7263_636c_u64 + iteration as u64 + 1,
            pattern,
        )?;
        let start = stream.record_event(Some(0))?;
        if descending {
            rccl::sort_by(&exec, &mut keys, |left: &u32, right: &u32| right.cmp(left))?;
        } else {
            rccl::sort(&exec, &mut keys)?;
        }
        let end = stream.record_event(Some(0))?;
        let elapsed_ms = start.elapsed_ms(&end)? as f64;
        best = best.min(Duration::from_secs_f64(elapsed_ms / 1000.0));
        total_ms += elapsed_ms;
        println!("iteration {iteration}: {elapsed_ms:.3} ms");
    }

    let errors = if descending {
        support.validate_order(&stream, &keys, |left: &u32, right: &u32| right.cmp(left))?
    } else {
        support.validate_order(&stream, &keys, u32::cmp)?
    };
    if errors != 0 {
        return Err(format!("GPU validation found {errors} adjacent inversions").into());
    }

    let best_s = best.as_secs_f64();
    println!("validation: PASS (no key readback)");
    println!(
        "best: {:.3} ms, {:.3} Gkey/s, {:.3} GiB/s; mean {:.3} ms",
        best_s * 1000.0,
        elements as f64 / best_s / 1.0e9,
        bytes as f64 / best_s / 1024f64.powi(3),
        total_ms / iterations as f64,
    );
    Ok(())
}
