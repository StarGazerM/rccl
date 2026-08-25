use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use std::error::Error;

#[cuda_host::cuda_module]
mod fixtures {
    use cuda_device::{
        atomic::{AtomicOrdering, DeviceAtomicU64},
        kernel, thread,
    };

    #[kernel]
    pub unsafe fn fill_sorted(pointer: *mut u32, len: u64, parity: u32) {
        let index = thread::index_1d().get() as u64;
        if index < len {
            unsafe { pointer.add(index as usize).write(index as u32 * 2 + parity) };
        }
    }

    #[kernel]
    pub unsafe fn fill_sorted_values(pointer: *mut u64, len: u64, parity: u64) {
        let index = thread::index_1d().get() as u64;
        if index < len {
            unsafe { pointer.add(index as usize).write(index * 2 + parity) };
        }
    }

    #[kernel]
    pub unsafe fn validate_iota(pointer: *const u32, len: u64, errors: *mut u64) {
        let index = thread::index_1d().get() as u64;
        if index < len && unsafe { pointer.add(index as usize).read() } != index as u32 {
            unsafe { DeviceAtomicU64::from_ptr(errors) }.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    pub unsafe fn validate_iota_u64(pointer: *const u64, len: u64, errors: *mut u64) {
        let index = thread::index_1d().get() as u64;
        if index < len && unsafe { pointer.add(index as usize).read() } != index {
            unsafe { DeviceAtomicU64::from_ptr(errors) }.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }
}

fn argument(name: &str, default: usize) -> usize {
    let args = std::env::args().collect::<Vec<_>>();
    args.windows(2)
        .find(|pair| pair[0] == name)
        .and_then(|pair| pair[1].parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn Error>> {
    let count = argument("--elements-per-input", 1 << 24);
    let iterations = argument("--iterations", 20);
    let total = count
        .checked_mul(2)
        .ok_or("combined benchmark length overflow")?;
    if total > u32::MAX as usize {
        return Err("the iota validator requires a combined length <= u32::MAX".into());
    }

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let exec = rccl::cuda::par(&stream)?;
    let fixtures = fixtures::load(&context)?;
    let first = DeviceBuffer::<u32>::zeroed(&stream, count)?;
    let second = DeviceBuffer::<u32>::zeroed(&stream, count)?;
    let mut output = DeviceBuffer::<u32>::zeroed(&stream, total)?;
    let launch = LaunchConfig::for_num_elems(u32::try_from(count)?);
    unsafe {
        fixtures.fill_sorted(
            &stream,
            launch,
            first.cu_deviceptr() as *mut u32,
            count as u64,
            0,
        )?;
        fixtures.fill_sorted(
            &stream,
            launch,
            second.cu_deviceptr() as *mut u32,
            count as u64,
            1,
        )?;
    }

    for _ in 0..3 {
        rccl::merge(&exec, &first, &second, &mut output)?;
    }

    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = stream.record_event(Some(0))?;
        rccl::merge(&exec, &first, &second, &mut output)?;
        let end = stream.record_event(Some(0))?;
        end.synchronize()?;
        timings.push(start.elapsed_ms(&end)? as f64);
    }

    let errors = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    unsafe {
        fixtures.validate_iota(
            &stream,
            LaunchConfig::for_num_elems(u32::try_from(total)?),
            output.cu_deviceptr() as *const u32,
            total as u64,
            errors.cu_deviceptr() as *mut u64,
        )?;
    }
    let error_count = errors.to_host_vec(&stream)?[0];
    if error_count != 0 {
        return Err(format!("merge validation found {error_count} incorrect elements").into());
    }

    timings.sort_by(f64::total_cmp);
    let best = timings[0];
    let median = timings[timings.len() / 2];
    println!(
        "RCCL merge u32 (synchronous public facade): inputs={count}+{count}, best={best:.3} ms ({:.3} Gitem/s), median={median:.3} ms",
        total as f64 / best / 1.0e6,
    );

    let first_values = DeviceBuffer::<u64>::zeroed(&stream, count)?;
    let second_values = DeviceBuffer::<u64>::zeroed(&stream, count)?;
    let mut output_values = DeviceBuffer::<u64>::zeroed(&stream, total)?;
    unsafe {
        fixtures.fill_sorted_values(
            &stream,
            launch,
            first_values.cu_deviceptr() as *mut u64,
            count as u64,
            0,
        )?;
        fixtures.fill_sorted_values(
            &stream,
            launch,
            second_values.cu_deviceptr() as *mut u64,
            count as u64,
            1,
        )?;
    }
    for _ in 0..3 {
        rccl::merge_by_key(
            &exec,
            &first,
            &first_values,
            &second,
            &second_values,
            &mut output,
            &mut output_values,
        )?;
    }

    let mut pair_timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = stream.record_event(Some(0))?;
        rccl::merge_by_key(
            &exec,
            &first,
            &first_values,
            &second,
            &second_values,
            &mut output,
            &mut output_values,
        )?;
        let end = stream.record_event(Some(0))?;
        end.synchronize()?;
        pair_timings.push(start.elapsed_ms(&end)? as f64);
    }
    let pair_errors = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    unsafe {
        fixtures.validate_iota_u64(
            &stream,
            LaunchConfig::for_num_elems(u32::try_from(total)?),
            output_values.cu_deviceptr() as *const u64,
            total as u64,
            pair_errors.cu_deviceptr() as *mut u64,
        )?;
    }
    let pair_error_count = pair_errors.to_host_vec(&stream)?[0];
    if pair_error_count != 0 {
        return Err(format!("pair merge validation found {pair_error_count} errors").into());
    }
    pair_timings.sort_by(f64::total_cmp);
    let pair_best = pair_timings[0];
    let pair_median = pair_timings[pair_timings.len() / 2];
    println!(
        "RCCL merge_by_key u32/u64 (synchronous public facade): inputs={count}+{count}, best={pair_best:.3} ms ({:.3} Gpair/s), median={pair_median:.3} ms",
        total as f64 / pair_best / 1.0e6,
    );
    Ok(())
}
