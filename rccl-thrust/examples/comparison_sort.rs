//! Comparison-sort rows from pinned Thrust `sort.cu`, `stable_sort.cu`, and
//! `sort_by_key.cu`, isolated so comparator lowering is tested independently
//! from the natural-order radix route.

use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

fn shuffled_unique(count: usize) -> Vec<u32> {
    let mut values = (0..count as u32).collect::<Vec<_>>();
    let mut state = 0x534d_4152_545f_534f_u64;
    for index in (1..count).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.swap(index, state as usize % (index + 1));
    }
    values
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let thrust = rccl_thrust::thrust_program!(&context)?;
    let exec = thrust.par(&stream)?;

    let input = shuffled_unique(10_003);

    // Pinned Thrust comparator sort: descending order is selected by the Rust
    // closure. Its CUB schedule and temporary storage stay implementation
    // details of the macro and execution policy.
    let mut descending = DeviceBuffer::from_host(&stream, &input)?;
    rccl_thrust::sort_by!(::<u32>(
        &exec,
        &mut descending,
        |left: &u32, right: &u32| right.cmp(left),
    ))?;
    assert_eq!(
        descending.to_host_vec(&stream)?,
        (0..input.len() as u32).rev().collect::<Vec<_>>()
    );

    // Pinned Thrust comparison `sort_by_key`: values must follow their keys.
    let mut pair_keys = DeviceBuffer::from_host(&stream, &input)?;
    let mut pair_values = DeviceBuffer::from_host(&stream, &input)?;
    rccl_thrust::sort_by_key_with!(::<u32, u32>(
        &exec,
        &mut pair_keys,
        &mut pair_values,
        |left: &u32, right: &u32| right.cmp(left),
    ))?;
    let descending_expected = (0..input.len() as u32).rev().collect::<Vec<_>>();
    assert_eq!(pair_keys.to_host_vec(&stream)?, descending_expected);
    assert_eq!(pair_values.to_host_vec(&stream)?, descending_expected);

    // Pinned stable comparator route with many equivalent keys. The values
    // are original positions, making stability observable rather than merely
    // checking sortedness.
    let stable_keys_host = input.iter().map(|key| key % 128).collect::<Vec<_>>();
    let stable_values_host = (0..input.len() as u32).collect::<Vec<_>>();
    let mut expected = stable_keys_host
        .iter()
        .copied()
        .zip(stable_values_host.iter().copied())
        .collect::<Vec<_>>();
    expected.sort_by_key(|pair| pair.0);
    let mut stable_keys = DeviceBuffer::from_host(&stream, &stable_keys_host)?;
    let mut stable_values = DeviceBuffer::from_host(&stream, &stable_values_host)?;
    rccl_thrust::stable_sort_by_key_with!(::<u32, u32>(
        &exec,
        &mut stable_keys,
        &mut stable_values,
        |left: &u32, right: &u32| left.cmp(right),
    ))?;
    let actual = stable_keys
        .to_host_vec(&stream)?
        .into_iter()
        .zip(stable_values.to_host_vec(&stream)?)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);

    println!("pinned Thrust comparison sort routes: PASS");
    Ok(())
}
