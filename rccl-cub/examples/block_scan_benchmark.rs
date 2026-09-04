//! Equal-work performance comparison for pinned CUB `BlockScan`.

use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use std::error::Error;

const BLOCK_THREADS: usize = 256;
const BLOCKS: usize = 32_768;
const FULL_ITEMS: usize = BLOCK_THREADS * BLOCKS;
const PARTIAL_ITEMS: usize = FULL_ITEMS - 63;
const WARMUPS: usize = 20;
const SAMPLES: usize = 21;
const LAUNCHES: usize = 100;

#[cuda_device::cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared_range = (32, 64), dynamic_shared_alignment = 8)]
    pub fn scan_blocks<T>(values: DisjointSlice<T>)
    where
        T: rccl_cub::__private::BlockScanValue,
    {
        let thread_index = thread::threadIdx_x() as usize;
        let tile_base = thread::blockIdx_x() as usize * BLOCK_THREADS;
        let index = tile_base + thread_index;
        let len = values.len();
        let valid = len.saturating_sub(tile_base).min(BLOCK_THREADS) as u32;
        let mut values = rccl_cub::__private::DeviceRegion::new(values);
        let value = if index < len {
            values.read(index)
        } else {
            T::zero()
        };
        let shared = rccl_cub::__private::BlockSharedMemory::new(8 * core::mem::size_of::<T>());
        let scanned = rccl_cub::__private::block_scan_sum_256(value, valid, &shared);
        if thread_index < valid as usize {
            values.write(index, scanned.exclusive);
        }
    }
}

fn reset<T: DeviceCopy>(
    stream: &cuda_core::CudaStream,
    baseline: &DeviceBuffer<T>,
    values: &mut DeviceBuffer<T>,
) -> Result<(), cuda_core::DriverError> {
    unsafe {
        cuda_core::memory::memcpy_dtod_async(
            values.cu_deviceptr(),
            baseline.cu_deviceptr(),
            baseline.num_bytes(),
            stream.cu_stream(),
        )?;
    }
    Ok(())
}

fn run<T>(
    program: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    items: usize,
    type_name: &str,
    one: T,
    expected: impl Fn(usize) -> T,
) -> Result<(), Box<dyn Error>>
where
    T: rccl_cub::__private::BlockScanValue + DeviceCopy + PartialEq + core::fmt::Debug + 'static,
{
    let host = vec![one; items];
    let baseline = DeviceBuffer::from_host(stream, &host)?;
    let mut values = DeviceBuffer::from_host(stream, &host)?;
    let grid = items.div_ceil(BLOCK_THREADS) as u32;
    let shared = (8 * core::mem::size_of::<T>()) as u32;
    let prepared = program.prepare_scan_blocks::<T>(LaunchConfig1D::new(
        grid,
        BLOCK_THREADS as u32,
        shared,
    ))?;

    for _ in 0..WARMUPS {
        reset(stream, &baseline, &mut values)?;
        for _ in 0..LAUNCHES {
            program.scan_blocks(stream, &prepared, &mut values)?;
        }
    }
    stream.synchronize()?;

    let timing = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        reset(stream, &baseline, &mut values)?;
        let start = stream.record_event(timing)?;
        for _ in 0..LAUNCHES {
            program.scan_blocks(stream, &prepared, &mut values)?;
        }
        let end = stream.record_event(timing)?;
        end.synchronize()?;
        samples.push(start.elapsed_ms(&end)? / LAUNCHES as f32);
    }

    reset(stream, &baseline, &mut values)?;
    program.scan_blocks(stream, &prepared, &mut values)?;
    let result = values.to_host_vec(stream)?;
    for (index, value) in result.into_iter().enumerate() {
        assert_eq!(value, expected(index % BLOCK_THREADS));
    }

    samples.sort_by(f32::total_cmp);
    println!(
        "implementation=rccl algorithm=block_exclusive_sum type={type_name} threads={BLOCK_THREADS} items={items} blocks={grid} launches={LAUNCHES} best_ms={:.6} median_ms={:.6}",
        samples[0],
        samples[samples.len() / 2],
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = unsafe { kernels::load(&context)? };

    run(&program, &stream, FULL_ITEMS, "u32", 1_u32, |index| {
        index as u32
    })?;
    run(&program, &stream, PARTIAL_ITEMS, "u32", 1_u32, |index| {
        index as u32
    })?;
    run(&program, &stream, FULL_ITEMS, "u64", 1_u64, |index| {
        index as u64
    })?;
    run(&program, &stream, PARTIAL_ITEMS, "u64", 1_u64, |index| {
        index as u64
    })?;
    Ok(())
}
