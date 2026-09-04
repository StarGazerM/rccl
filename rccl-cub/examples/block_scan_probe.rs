use cuda_core::{CudaContext, DeviceBuffer, DeviceCopy, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use std::error::Error;

#[cuda_device::cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared_range = (32, 64), dynamic_shared_alignment = 8)]
    pub fn block_scan<T>(
        input: &[T],
        inclusive: DisjointSlice<T>,
        exclusive: DisjointSlice<T>,
        aggregate: DisjointSlice<T>,
        valid: u32,
    ) where
        T: rccl_cub::__private::BlockScanValue,
    {
        let lane = thread::threadIdx_x() as usize;
        let value = input.get(lane).copied().unwrap_or_else(T::zero);
        let shared =
            rccl_cub::__private::BlockSharedMemory::new(8 * core::mem::size_of::<T>());
        let result = rccl_cub::__private::block_scan_sum_256(value, valid, &shared);
        let mut inclusive = rccl_cub::__private::DeviceRegion::new(inclusive);
        let mut exclusive = rccl_cub::__private::DeviceRegion::new(exclusive);
        let mut aggregate = rccl_cub::__private::DeviceRegion::new(aggregate);
        if lane < valid as usize {
            inclusive.write(lane, result.inclusive);
            exclusive.write(lane, result.exclusive);
        }
        if lane == 0 {
            aggregate.write(0, result.aggregate);
        }
    }
}

fn run_case<T>(
    program: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    input: &[T],
) -> Result<(), Box<dyn Error>>
where
    T: rccl_cub::__private::BlockScanValue
        + DeviceCopy
        + PartialEq
        + core::fmt::Debug
        + 'static,
{
    let input_device = DeviceBuffer::from_host(stream, input)?;
    let zeros = vec![T::zero(); input.len()];
    let mut inclusive = DeviceBuffer::from_host(stream, &zeros)?;
    let mut exclusive = DeviceBuffer::from_host(stream, &zeros)?;
    let mut aggregate = DeviceBuffer::from_host(stream, &[T::zero()])?;
    let config = LaunchConfig1D::new(
        1,
        256,
        (8 * core::mem::size_of::<T>()) as u32,
    );
    let prepared = program.prepare_block_scan::<T>(config)?;
    program.block_scan(
        stream,
        &prepared,
        &input_device,
        &mut inclusive,
        &mut exclusive,
        &mut aggregate,
        input.len() as u32,
    )?;

    let inclusive = inclusive.to_host_vec(stream)?;
    let exclusive = exclusive.to_host_vec(stream)?;
    let aggregate = aggregate.to_host_vec(stream)?[0];
    let mut prefix = T::zero();
    for index in 0..input.len() {
        assert_eq!(exclusive[index], prefix);
        prefix = T::add(prefix, input[index]);
        assert_eq!(inclusive[index], prefix);
    }
    assert_eq!(aggregate, prefix);
    Ok(())
}

fn inject_invalid_valid_count(
    context: &std::sync::Arc<CudaContext>,
    stream: &cuda_core::CudaStream,
    program: &kernels::LoadedModule,
) -> Result<(), Box<dyn Error>> {
    let input = DeviceBuffer::from_host(stream, &[1_u32; 256])?;
    let mut inclusive = DeviceBuffer::from_host(stream, &[0_u32; 256])?;
    let mut exclusive = DeviceBuffer::from_host(stream, &[0_u32; 256])?;
    let mut aggregate = DeviceBuffer::from_host(stream, &[0_u32])?;
    let prepared = program.prepare_block_scan::<u32>(LaunchConfig1D::new(1, 256, 32))?;
    program.block_scan(
        stream,
        &prepared,
        &input,
        &mut inclusive,
        &mut exclusive,
        &mut aggregate,
        257,
    )?;
    match stream.synchronize() {
        Err(error) => {
            eprintln!("invalid block-scan valid count surfaced: {error}");
            Ok(())
        }
        Ok(()) => {
            drop((context, inclusive, exclusive, aggregate));
            panic!("invalid block-scan valid count did not trap")
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = unsafe { kernels::load(&context)? };
    if std::env::args().any(|arg| arg == "--invalid-valid") {
        return inject_invalid_valid_count(&context, &stream, &program);
    }

    for len in [0_usize, 1, 31, 32, 33, 255, 256] {
        let input_u32 = (0..len)
            .map(|index| (index as u32).wrapping_mul(31).wrapping_add(7))
            .collect::<Vec<_>>();
        run_case(&program, &stream, &input_u32)?;
        let input_u64 = (0..len)
            .map(|index| (index as u64).wrapping_mul(1_000_000_007))
            .collect::<Vec<_>>();
        run_case(&program, &stream, &input_u64)?;
    }
    run_case(&program, &stream, &[u32::MAX, 2, u32::MAX, 5])?;
    println!("private block scan inclusive/exclusive/aggregate matrix: PASS");
    Ok(())
}
