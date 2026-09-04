use cuda_core::{CudaContext, DeviceBuffer};
use rccl_thrust::{exclusive_scan, inclusive_scan, inclusive_scan_async};
use std::error::Error;

fn inclusive_u32(input: &[u32]) -> Vec<u32> {
    let mut aggregate = 0_u32;
    input
        .iter()
        .map(|&value| {
            aggregate = aggregate.wrapping_add(value);
            aggregate
        })
        .collect()
}

fn exclusive_u32(input: &[u32]) -> Vec<u32> {
    let mut aggregate = 0_u32;
    input
        .iter()
        .map(|&value| {
            let prefix = aggregate;
            aggregate = aggregate.wrapping_add(value);
            prefix
        })
        .collect()
}

fn exclusive_u64(input: &[u64]) -> Vec<u64> {
    let mut aggregate = 0_u64;
    input
        .iter()
        .map(|&value| {
            let prefix = aggregate;
            aggregate = aggregate.wrapping_add(value);
            prefix
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let program = rccl_thrust::cuda::Program::load(&context)?;
    let execution = program.par(&stream)?;

    for len in [0_usize, 1, 31, 32, 33, 255, 256, 257, 513, 65_537] {
        let input: Vec<u32> = (0..len)
            .map(|index| (index as u32).wrapping_mul(17).wrapping_add(1))
            .collect();
        let mut inclusive = DeviceBuffer::from_host(&stream, &input)?;
        if len == 257 {
            inclusive_scan_async(&execution, &mut inclusive)?.wait()?;
        } else {
            inclusive_scan(&execution, &mut inclusive)?;
        }
        assert_eq!(inclusive.to_host_vec(&stream)?, inclusive_u32(&input));

        let mut exclusive = DeviceBuffer::from_host(&stream, &input)?;
        exclusive_scan(&execution, &mut exclusive)?;
        assert_eq!(exclusive.to_host_vec(&stream)?, exclusive_u32(&input));
    }

    let wide_input: Vec<u64> = (0..70_001)
        .map(|index| (index as u64).wrapping_mul(1_000_000_007))
        .collect();
    let mut wide = DeviceBuffer::from_host(&stream, &wide_input)?;
    exclusive_scan(&execution, &mut wide)?;
    assert_eq!(wide.to_host_vec(&stream)?, exclusive_u64(&wide_input));

    let mut sentinel = DeviceBuffer::from_host(&stream, &[4_u32, 9, 2, 0])?;
    exclusive_scan(&execution, &mut sentinel)?;
    assert_eq!(sentinel.to_host_vec(&stream)?, [0, 4, 13, 15]);

    let wrapping_input = [u32::MAX, 2, u32::MAX, 5];
    let mut wrapping = DeviceBuffer::from_host(&stream, &wrapping_input)?;
    inclusive_scan(&execution, &mut wrapping)?;
    assert_eq!(
        wrapping.to_host_vec(&stream)?,
        inclusive_u32(&wrapping_input)
    );

    println!("device inclusive/exclusive scan boundaries: PASS");
    Ok(())
}
