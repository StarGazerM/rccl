//! Keys-only ports of the four `DeviceMergeSort` cases in
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

    // `DeviceMergeSort::SortKeysCopy works`, including more than one full
    // 256 * 17 SM60+ tile and a partial final tile.
    let unique = shuffled_unique(10_003);
    let input = DeviceBuffer::from_host(&stream, &unique)?;
    let mut output = DeviceBuffer::<u32>::zeroed(&stream, unique.len())?;
    let mut workspace =
        rccl_cub::device_merge_sort_keys_workspace!(::<u32>(&launch, unique.len()))?;
    rccl_cub::device_merge_sort_keys_copy!(::<u32>(
        &launch,
        &input,
        &mut workspace,
        &mut output,
        |left: &u32, right: &u32| left.cmp(right),
    ))?;
    assert_eq!(
        output.to_host_vec(&stream)?,
        (0..10_003_u32).collect::<Vec<_>>()
    );
    assert_eq!(input.to_host_vec(&stream)?, unique);

    // `DeviceMergeSort::SortKeys works` with the same shuffled ranks.
    let mut in_place = DeviceBuffer::from_host(&stream, &unique)?;
    let mut in_place_workspace =
        rccl_cub::device_merge_sort_keys_workspace!(::<u32>(&launch, unique.len()))?;
    rccl_cub::device_merge_sort_keys!(::<u32>(
        &launch,
        &mut in_place,
        &mut in_place_workspace,
        |left: &u32, right: &u32| left.cmp(right),
    ))?;
    assert_eq!(
        in_place.to_host_vec(&stream)?,
        (0..10_003_u32).collect::<Vec<_>>()
    );

    // Policy600 scales the nominal 17 four-byte items to 8 u64 items. That
    // crosses CUB BlockExchange's power-of-two padding boundary, so this case
    // intentionally spans two full 256 * 8 tiles and one partial tile. The
    // u32 cases above cannot expose a missing padded shared-memory index.
    let unique_u64 = shuffled_unique(5_003)
        .into_iter()
        .map(u64::from)
        .collect::<Vec<_>>();
    let input_u64 = DeviceBuffer::from_host(&stream, &unique_u64)?;
    let mut output_u64 = DeviceBuffer::<u64>::zeroed(&stream, unique_u64.len())?;
    let mut workspace_u64 =
        rccl_cub::device_merge_sort_keys_workspace!(::<u64>(&launch, unique_u64.len()))?;
    rccl_cub::device_merge_sort_keys_copy!(::<u64>(
        &launch,
        &input_u64,
        &mut workspace_u64,
        &mut output_u64,
        |left: &u64, right: &u64| left.cmp(right),
    ))?;
    assert_eq!(
        output_u64.to_host_vec(&stream)?,
        (0..5_003_u64).collect::<Vec<_>>()
    );
    assert_eq!(input_u64.to_host_vec(&stream)?, unique_u64);

    // `StableSortKeysCopy works ... when there are a lot sort-keys that
    // compare equal`. The low decimal digit is the original-order tag.
    let stable_input_host = vec![25_i32, 14, 35, 16, 26, 34, 36, 24, 15];
    let stable_input = DeviceBuffer::from_host(&stream, &stable_input_host)?;
    let mut stable_output = DeviceBuffer::<i32>::zeroed(&stream, stable_input_host.len())?;
    let mut stable_workspace = rccl_cub::device_merge_sort_keys_workspace!(::<i32>(
        &launch,
        stable_input_host.len(),
    ))?;
    rccl_cub::device_stable_merge_sort_keys_copy!(::<i32>(
        &launch,
        &stable_input,
        &mut stable_workspace,
        &mut stable_output,
        |left: &i32, right: &i32| (left / 10).cmp(&(right / 10)),
    ))?;
    assert_eq!(
        stable_output.to_host_vec(&stream)?,
        [14, 16, 15, 25, 26, 24, 35, 34, 36]
    );
    assert_eq!(stable_input.to_host_vec(&stream)?, stable_input_host);

    // `DeviceMergeSort::StableSortKeys works`, in place.
    let mut stable_in_place = DeviceBuffer::from_host(&stream, &stable_input_host)?;
    let mut stable_in_place_workspace = rccl_cub::device_merge_sort_keys_workspace!(::<i32>(
        &launch,
        stable_input_host.len(),
    ))?;
    rccl_cub::device_stable_merge_sort_keys!(::<i32>(
        &launch,
        &mut stable_in_place,
        &mut stable_in_place_workspace,
        |left: &i32, right: &i32| (left / 10).cmp(&(right / 10)),
    ))?;
    assert_eq!(
        stable_in_place.to_host_vec(&stream)?,
        [14, 16, 15, 25, 26, 24, 35, 34, 36]
    );

    println!("pinned CUB DeviceMergeSort keys cases: PASS");
    Ok(())
}
