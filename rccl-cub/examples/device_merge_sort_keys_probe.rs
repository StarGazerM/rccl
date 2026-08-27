//! Keys-only ports of the four `DeviceMergeSort` cases in
//! `cub/test/catch2_test_device_merge_sort.cu`.

use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

rccl_cub::cub_kernels!();

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
    // SAFETY: this executable owns the sole artifact emitted by the one
    // `cub_kernels!` invocation. Operation lowering itself is safe Rust.
    let module = unsafe { __rccl_cub_kernels::load(&context)? };
    let exec = rccl_cub::cuda::par(&stream, &module);

    // `DeviceMergeSort::SortKeysCopy works`, including more than one full
    // 256 * 17 SM60+ tile and a partial final tile.
    let unique = shuffled_unique(10_003);
    let input = DeviceBuffer::from_host(&stream, &unique)?;
    let mut output = DeviceBuffer::<u32>::zeroed(&stream, unique.len())?;
    let mut scratch = DeviceBuffer::<u32>::zeroed(&stream, unique.len())?;
    let partition_count = rccl_cub::device_merge_sort_partition_count!(::<u32, sm_89>(
        unique.len(),
    ))?;
    let mut partitions = DeviceBuffer::<usize>::zeroed(&stream, partition_count)?;
    rccl_cub::device_merge_sort_keys_copy!(::<u32, sm_89>(
        &exec,
        &input,
        &mut scratch,
        &mut partitions,
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
    let mut in_place_scratch = DeviceBuffer::<u32>::zeroed(&stream, unique.len())?;
    let mut in_place_partitions = DeviceBuffer::<usize>::zeroed(&stream, partition_count)?;
    rccl_cub::device_merge_sort_keys!(::<u32, sm_89>(
        &exec,
        &mut in_place,
        &mut in_place_scratch,
        &mut in_place_partitions,
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
    let mut scratch_u64 = DeviceBuffer::<u64>::zeroed(&stream, unique_u64.len())?;
    let partition_count_u64 = rccl_cub::device_merge_sort_partition_count!(::<u64, sm_89>(
        unique_u64.len(),
    ))?;
    let mut partitions_u64 = DeviceBuffer::<usize>::zeroed(&stream, partition_count_u64)?;
    rccl_cub::device_merge_sort_keys_copy!(::<u64, sm_89>(
        &exec,
        &input_u64,
        &mut scratch_u64,
        &mut partitions_u64,
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
    let mut stable_scratch = DeviceBuffer::<i32>::zeroed(&stream, stable_input_host.len())?;
    let stable_partition_count = rccl_cub::device_merge_sort_partition_count!(::<i32, sm_89>(
        stable_input_host.len(),
    ))?;
    let mut stable_partitions = DeviceBuffer::<usize>::zeroed(&stream, stable_partition_count)?;
    rccl_cub::device_stable_merge_sort_keys_copy!(::<i32, sm_89>(
        &exec,
        &stable_input,
        &mut stable_scratch,
        &mut stable_partitions,
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
    let mut stable_in_place_scratch =
        DeviceBuffer::<i32>::zeroed(&stream, stable_input_host.len())?;
    let mut stable_in_place_partitions =
        DeviceBuffer::<usize>::zeroed(&stream, stable_partition_count)?;
    rccl_cub::device_stable_merge_sort_keys!(::<i32, sm_89>(
        &exec,
        &mut stable_in_place,
        &mut stable_in_place_scratch,
        &mut stable_in_place_partitions,
        |left: &i32, right: &i32| (left / 10).cmp(&(right / 10)),
    ))?;
    assert_eq!(
        stable_in_place.to_host_vec(&stream)?,
        [14, 16, 15, 25, 26, 24, 35, 34, 36]
    );

    println!("pinned CUB DeviceMergeSort keys cases: PASS");
    Ok(())
}
