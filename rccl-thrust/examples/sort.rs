//! Rust port of the exact simple cases in pinned CCCL
//! `thrust/testing/sort.cu` and `thrust/testing/sort_by_key.cu`.

use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let thrust = rccl_thrust::thrust_program!(&context)?;
    let exec = thrust.par(&stream)?;

    // Exact integer rows printed by pinned `thrust/examples/sort.cu`. The
    // source's default_random_engine seed 123456 produces this input.
    let upstream_input = [
        79_i32, 78, 62, 78, 94, 40, 86, 57, 40, 16, 28, 54, 77, 87, 93, 98,
    ];
    let mut upstream_natural = DeviceBuffer::from_host(&stream, &upstream_input)?;
    rccl_thrust::sort!(::<i32>(
        &exec,
        &mut upstream_natural,
    ))?;
    assert_eq!(
        upstream_natural.to_host_vec(&stream)?,
        [
            16, 28, 40, 40, 54, 57, 62, 77, 78, 78, 79, 86, 87, 93, 94, 98
        ]
    );

    let mut upstream_descending = DeviceBuffer::from_host(&stream, &upstream_input)?;
    rccl_thrust::sort_by!(::<i32>(
        &exec,
        &mut upstream_descending,
        |left: &i32, right: &i32| right.cmp(left),
    ))?;
    assert_eq!(
        upstream_descending.to_host_vec(&stream)?,
        [
            98, 94, 93, 87, 86, 79, 78, 78, 77, 62, 57, 54, 40, 40, 28, 16
        ]
    );

    let mut upstream_custom = DeviceBuffer::from_host(&stream, &upstream_input)?;
    rccl_thrust::sort_by!(::<i32>(
        &exec,
        &mut upstream_custom,
        |left: &i32, right: &i32| match ((left & 1) == 0, (right & 1) == 0) {
            (true, false) => core::cmp::Ordering::Less,
            (false, true) => core::cmp::Ordering::Greater,
            _ => left.cmp(right),
        },
    ))?;
    assert_eq!(
        upstream_custom.to_host_vec(&stream)?,
        [
            16, 28, 40, 40, 54, 62, 78, 78, 86, 94, 98, 57, 77, 79, 87, 93
        ]
    );

    let mut upstream_pair_keys = DeviceBuffer::from_host(&stream, &upstream_input)?;
    let mut upstream_pair_values = DeviceBuffer::from_host(
        &stream,
        &[0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )?;
    rccl_thrust::sort_by_key!(::<i32, i32>(
        &exec,
        &mut upstream_pair_keys,
        &mut upstream_pair_values,
    ))?;
    assert_eq!(
        upstream_pair_keys.to_host_vec(&stream)?,
        [
            16, 28, 40, 40, 54, 57, 62, 77, 78, 78, 79, 86, 87, 93, 94, 98
        ]
    );
    assert_eq!(
        upstream_pair_values.to_host_vec(&stream)?,
        [9, 10, 5, 8, 11, 7, 2, 12, 1, 3, 0, 6, 13, 14, 4, 15]
    );

    let mut upstream_pair_keys_descending = DeviceBuffer::from_host(&stream, &upstream_input)?;
    let mut upstream_pair_values_descending = DeviceBuffer::from_host(
        &stream,
        &[0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )?;
    rccl_thrust::sort_by_key_with!(::<i32, i32>(
        &exec,
        &mut upstream_pair_keys_descending,
        &mut upstream_pair_values_descending,
        |left: &i32, right: &i32| right.cmp(left),
    ))?;
    assert_eq!(
        upstream_pair_keys_descending.to_host_vec(&stream)?,
        [
            98, 94, 93, 87, 86, 79, 78, 78, 77, 62, 57, 54, 40, 40, 28, 16
        ]
    );
    assert_eq!(
        upstream_pair_values_descending.to_host_vec(&stream)?,
        [15, 4, 14, 13, 6, 0, 1, 3, 12, 2, 7, 11, 5, 8, 10, 9]
    );

    let mut keys = DeviceBuffer::from_host(&stream, &[1_i32, 3, 6, 5, 2, 0, 4])?;
    rccl_thrust::sort!(::<i32>(&exec, &mut keys))?;
    assert_eq!(keys.to_host_vec(&stream)?, [0, 1, 2, 3, 4, 5, 6]);

    let mut descending = DeviceBuffer::from_host(&stream, &[1_i32, 3, 6, 5, 2, 0, 4])?;
    rccl_thrust::sort_by!(::<i32>(
        &exec,
        &mut descending,
        |left: &i32, right: &i32| right.cmp(left),
    ))?;
    assert_eq!(descending.to_host_vec(&stream)?, [6, 5, 4, 3, 2, 1, 0]);

    let mut custom = DeviceBuffer::from_host(&stream, &[5_i32, 2, 9, 4, 7, 0, 3, 6])?;
    rccl_thrust::sort_by!(::<i32>(
        &exec,
        &mut custom,
        |left: &i32, right: &i32| {
            (left & 1).cmp(&(right & 1)).then_with(|| left.cmp(right))
        },
    ))?;
    assert_eq!(custom.to_host_vec(&stream)?, [0, 2, 4, 6, 3, 5, 7, 9]);

    // Exact `TestStableSortSimple` input and equivalence relation from pinned
    // `thrust/testing/stable_sort.cu`.
    let stable_input = [25_i32, 14, 35, 16, 26, 34, 36, 24, 15];
    let stable_expected = [14_i32, 16, 15, 25, 26, 24, 35, 34, 36];
    let mut stable = DeviceBuffer::from_host(&stream, &stable_input)?;
    rccl_thrust::stable_sort_by!(::<i32>(
        &exec,
        &mut stable,
        |left: &i32, right: &i32| (left / 10).cmp(&(right / 10)),
    ))?;
    assert_eq!(stable.to_host_vec(&stream)?, stable_expected);

    let mut natural_stable = DeviceBuffer::from_host(&stream, &[3_i32, 1, 2, 1])?;
    rccl_thrust::stable_sort!(::<i32>(&exec, &mut natural_stable))?;
    assert_eq!(natural_stable.to_host_vec(&stream)?, [1, 1, 2, 3]);

    let mut pair_keys = DeviceBuffer::from_host(&stream, &[1_i32, 3, 6, 5, 2, 0, 4])?;
    let mut pair_values = DeviceBuffer::from_host(&stream, &[0_i32, 1, 2, 3, 4, 5, 6])?;
    rccl_thrust::stable_sort_by_key!(::<i32, i32>(
        &exec,
        &mut pair_keys,
        &mut pair_values,
    ))?;
    assert_eq!(pair_keys.to_host_vec(&stream)?, [0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(pair_values.to_host_vec(&stream)?, [5, 0, 4, 1, 6, 3, 2]);

    let mut descending_keys = DeviceBuffer::from_host(&stream, &[1_i32, 3, 6, 5, 2, 0, 4])?;
    let mut descending_values = DeviceBuffer::from_host(&stream, &[0_i32, 1, 2, 3, 4, 5, 6])?;
    rccl_thrust::sort_by_key_with!(::<i32, i32>(
        &exec,
        &mut descending_keys,
        &mut descending_values,
        |left: &i32, right: &i32| right.cmp(left),
    ))?;
    assert_eq!(descending_keys.to_host_vec(&stream)?, [6, 5, 4, 3, 2, 1, 0]);
    assert_eq!(
        descending_values.to_host_vec(&stream)?,
        [2, 3, 6, 1, 4, 0, 5]
    );

    let mut unstable_keys = DeviceBuffer::from_host(&stream, &[1_i32, 3, 6, 5, 2, 0, 4])?;
    let mut unstable_values = DeviceBuffer::from_host(&stream, &[0_i32, 1, 2, 3, 4, 5, 6])?;
    rccl_thrust::sort_by_key!(::<i32, i32>(
        &exec,
        &mut unstable_keys,
        &mut unstable_values,
    ))?;
    assert_eq!(unstable_keys.to_host_vec(&stream)?, [0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(unstable_values.to_host_vec(&stream)?, [5, 0, 4, 1, 6, 3, 2]);

    let mut stable_keys = DeviceBuffer::from_host(&stream, &stable_input)?;
    let mut stable_values = DeviceBuffer::from_host(&stream, &[0_i32, 1, 2, 3, 4, 5, 6, 7, 8])?;
    rccl_thrust::stable_sort_by_key_with!(::<i32, i32>(
        &exec,
        &mut stable_keys,
        &mut stable_values,
        |left: &i32, right: &i32| (left / 10).cmp(&(right / 10)),
    ))?;
    assert_eq!(stable_keys.to_host_vec(&stream)?, stable_expected);
    assert_eq!(
        stable_values.to_host_vec(&stream)?,
        [1, 3, 8, 0, 4, 7, 2, 5, 6]
    );

    // Pinned CUB is Thrust CUDA's natural arithmetic sort dependency. Keep
    // the signed-zero check bitwise so Rust `f32` equality cannot mask a
    // Thrust-to-CUB lowering error or unstable pair movement.
    let mut float_keys = DeviceBuffer::from_host(
        &stream,
        &[1.25_f32, -0.0, f32::INFINITY, -3.5, 0.0, f32::NEG_INFINITY],
    )?;
    let mut float_values = DeviceBuffer::from_host(&stream, &[4_i32, 2, 5, 1, 3, 0])?;
    rccl_thrust::stable_sort_by_key!(::<f32, i32>(
        &exec,
        &mut float_keys,
        &mut float_values,
    ))?;
    assert_eq!(
        float_keys
            .to_host_vec(&stream)?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [
            f32::NEG_INFINITY.to_bits(),
            (-3.5_f32).to_bits(),
            (-0.0_f32).to_bits(),
            0.0_f32.to_bits(),
            1.25_f32.to_bits(),
            f32::INFINITY.to_bits(),
        ]
    );
    assert_eq!(float_values.to_host_vec(&stream)?, [0, 1, 2, 3, 4, 5]);

    println!("pinned CCCL Thrust sort simple cases: PASS");
    Ok(())
}
