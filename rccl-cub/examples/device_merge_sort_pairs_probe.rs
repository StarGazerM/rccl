//! Key/value ports of the `DeviceMergeSort` pair cases in
//! `cub/test/catch2_test_device_merge_sort.cu`.

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
    let cub = rccl_cub::cub_program!(&context)?;
    let launch = cub.on(&stream)?;

    // `DeviceMergeSort::SortPairsCopy works`: the value is the shuffled rank,
    // so both output ranges must become the ascending counting sequence.
    let ranks = shuffled_unique(10_003);
    let input_keys = DeviceBuffer::from_host(&stream, &ranks)?;
    let input_values = DeviceBuffer::from_host(&stream, &ranks)?;
    let mut output_keys = DeviceBuffer::<u32>::zeroed(&stream, ranks.len())?;
    let mut output_values = DeviceBuffer::<u32>::zeroed(&stream, ranks.len())?;
    let mut workspace =
        rccl_cub::device_merge_sort_pairs_workspace!(::<u32, u32>(&launch, ranks.len()))?;
    rccl_cub::device_merge_sort_pairs_copy!(::<u32, u32>(
        &launch,
        &input_keys,
        &input_values,
        &mut workspace,
        &mut output_keys,
        &mut output_values,
        |left: &u32, right: &u32| left.cmp(right),
    ))?;
    let ascending = (0..ranks.len() as u32).collect::<Vec<_>>();
    assert_eq!(output_keys.to_host_vec(&stream)?, ascending);
    assert_eq!(output_values.to_host_vec(&stream)?, ascending);
    assert_eq!(input_keys.to_host_vec(&stream)?, ranks);
    assert_eq!(input_values.to_host_vec(&stream)?, ranks);

    // `DeviceMergeSort::SortPairs works`, with the same pair association.
    let mut in_place_keys = DeviceBuffer::from_host(&stream, &ranks)?;
    let mut in_place_values = DeviceBuffer::from_host(&stream, &ranks)?;
    let mut in_place_workspace =
        rccl_cub::device_merge_sort_pairs_workspace!(::<u32, u32>(&launch, ranks.len()))?;
    rccl_cub::device_merge_sort_pairs!(::<u32, u32>(
        &launch,
        &mut in_place_keys,
        &mut in_place_values,
        &mut in_place_workspace,
        |left: &u32, right: &u32| left.cmp(right),
    ))?;
    assert_eq!(in_place_keys.to_host_vec(&stream)?, ascending);
    assert_eq!(in_place_values.to_host_vec(&stream)?, ascending);

    // `DeviceMergeSort::StableSortPairs works and performs a stable sort`.
    // Repeated keys make value order observable across multiple block tiles.
    let stable_keys_host = ranks.iter().map(|rank| rank % 128).collect::<Vec<_>>();
    let stable_values_host = (0..ranks.len() as u32).collect::<Vec<_>>();
    let mut expected = stable_keys_host
        .iter()
        .copied()
        .zip(stable_values_host.iter().copied())
        .collect::<Vec<_>>();
    expected.sort_by_key(|pair| pair.0);
    let mut stable_keys = DeviceBuffer::from_host(&stream, &stable_keys_host)?;
    let mut stable_values = DeviceBuffer::from_host(&stream, &stable_values_host)?;
    let mut stable_workspace =
        rccl_cub::device_merge_sort_pairs_workspace!(::<u32, u32>(&launch, ranks.len()))?;
    rccl_cub::device_stable_merge_sort_pairs!(::<u32, u32>(
        &launch,
        &mut stable_keys,
        &mut stable_values,
        &mut stable_workspace,
        |left: &u32, right: &u32| left.cmp(right),
    ))?;
    let actual = stable_keys
        .to_host_vec(&stream)?
        .into_iter()
        .zip(stable_values.to_host_vec(&stream)?)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);

    println!("pinned CUB DeviceMergeSort pair cases: PASS");
    Ok(())
}
