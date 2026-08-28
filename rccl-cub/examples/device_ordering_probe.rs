//! Executable ports of the implemented DeviceMerge, DeviceMergeSort, and
//! DeviceRadixSort cases in the pinned CCCL test suite.

use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let cub = rccl_cub::cub_program!(&context)?;
    let launch = cub.on(&stream)?;

    // `cub/test/catch2_test_device_merge_api.cu`, merge-keys example.
    let first = DeviceBuffer::from_host(&stream, &[0_i32, 2, 5])?;
    let second = DeviceBuffer::from_host(&stream, &[0_i32, 3, 3, 4])?;
    let mut merge_workspace = rccl_cub::device_merge_keys_workspace!(::<i32>(
        &launch,
        first.len(),
        second.len(),
    ))?;
    let mut merged = DeviceBuffer::<i32>::zeroed(&stream, first.len() + second.len())?;
    rccl_cub::device_merge_keys!(::<i32>(
        &launch,
        &first,
        &second,
        &mut merge_workspace,
        &mut merged,
        |left: &i32, right: &i32| left.cmp(right),
    ))?;
    assert_eq!(merged.to_host_vec(&stream)?, [0, 0, 2, 3, 3, 4, 5]);

    // `cub/test/catch2_test_device_merge_api.cu`, merge-pairs example. Rust
    // `u8` is the byte-sized support type corresponding to the C++ `char`
    // payload; Rust `char` is a four-byte Unicode scalar and is not the match.
    let first_values = DeviceBuffer::from_host(&stream, b"abc")?;
    let second_values = DeviceBuffer::from_host(&stream, b"ABCD")?;
    let mut pair_workspace = rccl_cub::device_merge_pairs_workspace!(::<i32, u8>(
        &launch,
        first.len(),
        second.len(),
    ))?;
    let mut merged_keys = DeviceBuffer::<i32>::zeroed(&stream, 7)?;
    let mut merged_values = DeviceBuffer::<u8>::zeroed(&stream, 7)?;
    rccl_cub::device_merge_pairs!(::<i32, u8>(
        &launch,
        &first,
        &first_values,
        &second,
        &second_values,
        &mut pair_workspace,
        &mut merged_keys,
        &mut merged_values,
        |left: &i32, right: &i32| left.cmp(right),
    ))?;
    assert_eq!(merged_keys.to_host_vec(&stream)?, [0, 0, 2, 3, 3, 4, 5]);
    assert_eq!(merged_values.to_host_vec(&stream)?, *b"aAbBCDc");

    // `catch2_test_device_merge.cu`, `DeviceMerge::MergePairs key types`,
    // with its default 3,623/6,346 input sizes and the `int16_t` key / `int`
    // value instantiation. Selecting the pinned SM52 policy gives eight
    // items per thread, which exercises BlockExchange's padded storage path.
    let first_i16 = (0..3_623)
        .map(|index| (index * 2) as i16)
        .collect::<Vec<_>>();
    let second_i16 = (0..6_346)
        .map(|index| (index * 2 + 1) as i16)
        .collect::<Vec<_>>();
    let first_i16_values = (0..3_623).collect::<Vec<_>>();
    let second_i16_values = (0..6_346).map(|index| 10_000 + index).collect::<Vec<_>>();
    let first_i16_device = DeviceBuffer::from_host(&stream, &first_i16)?;
    let second_i16_device = DeviceBuffer::from_host(&stream, &second_i16)?;
    let first_i16_values_device = DeviceBuffer::from_host(&stream, &first_i16_values)?;
    let second_i16_values_device = DeviceBuffer::from_host(&stream, &second_i16_values)?;
    let mut i16_workspace = rccl_cub::device_merge_pairs_workspace!(::<i16, i32>(
        &launch,
        first_i16.len(),
        second_i16.len(),
    ))?;
    let mut merged_i16 = DeviceBuffer::<i16>::zeroed(&stream, first_i16.len() + second_i16.len())?;
    let mut merged_i16_values =
        DeviceBuffer::<i32>::zeroed(&stream, first_i16.len() + second_i16.len())?;
    rccl_cub::device_merge_pairs!(::<i16, i32>(
        &launch,
        &first_i16_device,
        &first_i16_values_device,
        &second_i16_device,
        &second_i16_values_device,
        &mut i16_workspace,
        &mut merged_i16,
        &mut merged_i16_values,
        |left: &i16, right: &i16| left.cmp(right),
    ))?;
    let mut expected_i16 = Vec::with_capacity(first_i16.len() + second_i16.len());
    let mut expected_i16_values = Vec::with_capacity(expected_i16.capacity());
    let (mut first_index, mut second_index) = (0, 0);
    while first_index < first_i16.len() || second_index < second_i16.len() {
        if second_index == second_i16.len()
            || (first_index < first_i16.len() && first_i16[first_index] <= second_i16[second_index])
        {
            expected_i16.push(first_i16[first_index]);
            expected_i16_values.push(first_i16_values[first_index]);
            first_index += 1;
        } else {
            expected_i16.push(second_i16[second_index]);
            expected_i16_values.push(second_i16_values[second_index]);
            second_index += 1;
        }
    }
    assert_eq!(merged_i16.to_host_vec(&stream)?, expected_i16);
    assert_eq!(merged_i16_values.to_host_vec(&stream)?, expected_i16_values);

    let mut merge_sort_keys_7 =
        rccl_cub::device_merge_sort_keys_workspace!(::<i32>(&launch, 7usize))?;
    let mut merge_sort_keys_9 =
        rccl_cub::device_merge_sort_keys_workspace!(::<i32>(&launch, 9usize))?;
    let mut merge_sort_keys_5 =
        rccl_cub::device_merge_sort_keys_workspace!(::<i32>(&launch, 5usize))?;
    let mut merge_sort_pairs_7 =
        rccl_cub::device_merge_sort_pairs_workspace!(::<i32, u32>(&launch, 7usize))?;
    let mut merge_sort_pairs_5 =
        rccl_cub::device_merge_sort_pairs_workspace!(::<i32, u32>(&launch, 5usize))?;

    let mut unstable_keys = DeviceBuffer::from_host(&stream, &[5_i32, 1, 3, 1, 4, 2, 0])?;
    rccl_cub::device_merge_sort_keys!(::<i32>(
        &launch,
        &mut unstable_keys,
        &mut merge_sort_keys_7,
        |left: &i32, right: &i32| left.cmp(right),
    ))?;
    assert_eq!(unstable_keys.to_host_vec(&stream)?, [0, 1, 1, 2, 3, 4, 5]);

    let mut stable_keys_only =
        DeviceBuffer::from_host(&stream, &[25_i32, 14, 35, 16, 26, 34, 36, 24, 15])?;
    rccl_cub::device_stable_merge_sort_keys!(::<i32>(
        &launch,
        &mut stable_keys_only,
        &mut merge_sort_keys_9,
        |left: &i32, right: &i32| (left / 10).cmp(&(right / 10)),
    ))?;
    assert_eq!(
        stable_keys_only.to_host_vec(&stream)?,
        [14, 16, 15, 25, 26, 24, 35, 34, 36]
    );

    let copy_input = DeviceBuffer::from_host(&stream, &[4_i32, 1, 3, 0, 2])?;
    let mut copy_output = DeviceBuffer::<i32>::zeroed(&stream, 5)?;
    rccl_cub::device_merge_sort_keys_copy!(::<i32>(
        &launch,
        &copy_input,
        &mut merge_sort_keys_5,
        &mut copy_output,
        |left: &i32, right: &i32| left.cmp(right),
    ))?;
    assert_eq!(copy_input.to_host_vec(&stream)?, [4, 1, 3, 0, 2]);
    assert_eq!(copy_output.to_host_vec(&stream)?, [0, 1, 2, 3, 4]);

    let stable_copy_input =
        DeviceBuffer::from_host(&stream, &[25_i32, 14, 35, 16, 26, 34, 36, 24, 15])?;
    let mut stable_copy_output = DeviceBuffer::<i32>::zeroed(&stream, 9)?;
    rccl_cub::device_stable_merge_sort_keys_copy!(::<i32>(
        &launch,
        &stable_copy_input,
        &mut merge_sort_keys_9,
        &mut stable_copy_output,
        |left: &i32, right: &i32| (left / 10).cmp(&(right / 10)),
    ))?;
    assert_eq!(
        stable_copy_input.to_host_vec(&stream)?,
        [25, 14, 35, 16, 26, 34, 36, 24, 15]
    );
    assert_eq!(
        stable_copy_output.to_host_vec(&stream)?,
        [14, 16, 15, 25, 26, 24, 35, 34, 36]
    );

    let mut keys = DeviceBuffer::from_host(&stream, &[5_i32, 1, 3, 1, 4, 2, 0])?;
    let mut values = DeviceBuffer::from_host(&stream, &[0_u32, 1, 2, 3, 4, 5, 6])?;
    rccl_cub::device_stable_merge_sort_pairs!(::<i32, u32>(
        &launch,
        &mut keys,
        &mut values,
        &mut merge_sort_pairs_7,
        |left: &i32, right: &i32| left.cmp(right),
    ))?;
    assert_eq!(keys.to_host_vec(&stream)?, [0, 1, 1, 2, 3, 4, 5]);
    assert_eq!(values.to_host_vec(&stream)?, [6, 1, 3, 5, 2, 4, 0]);

    let mut nonstable_pair_keys = DeviceBuffer::from_host(&stream, &[4_i32, 1, 3, 0, 2])?;
    let mut nonstable_pair_values = DeviceBuffer::from_host(&stream, &[40_u32, 10, 30, 0, 20])?;
    rccl_cub::device_merge_sort_pairs!(::<i32, u32>(
        &launch,
        &mut nonstable_pair_keys,
        &mut nonstable_pair_values,
        &mut merge_sort_pairs_5,
        |left: &i32, right: &i32| left.cmp(right),
    ))?;
    assert_eq!(nonstable_pair_keys.to_host_vec(&stream)?, [0, 1, 2, 3, 4]);
    assert_eq!(
        nonstable_pair_values.to_host_vec(&stream)?,
        [0, 10, 20, 30, 40]
    );

    let pair_copy_input_keys = DeviceBuffer::from_host(&stream, &[4_i32, 1, 3, 0, 2])?;
    let pair_copy_input_values = DeviceBuffer::from_host(&stream, &[40_u32, 10, 30, 0, 20])?;
    let mut pair_copy_output_keys = DeviceBuffer::<i32>::zeroed(&stream, 5)?;
    let mut pair_copy_output_values = DeviceBuffer::<u32>::zeroed(&stream, 5)?;
    rccl_cub::device_merge_sort_pairs_copy!(::<i32, u32>(
        &launch,
        &pair_copy_input_keys,
        &pair_copy_input_values,
        &mut merge_sort_pairs_5,
        &mut pair_copy_output_keys,
        &mut pair_copy_output_values,
        |left: &i32, right: &i32| left.cmp(right),
    ))?;
    assert_eq!(pair_copy_input_keys.to_host_vec(&stream)?, [4, 1, 3, 0, 2]);
    assert_eq!(
        pair_copy_input_values.to_host_vec(&stream)?,
        [40, 10, 30, 0, 20]
    );
    assert_eq!(pair_copy_output_keys.to_host_vec(&stream)?, [0, 1, 2, 3, 4]);
    assert_eq!(
        pair_copy_output_values.to_host_vec(&stream)?,
        [0, 10, 20, 30, 40]
    );

    // Pinned DeviceRadixSort keys semantics. The caller owns one typed,
    // reusable workspace; its CUB agent buffers remain private.
    let radix_input = [u32::MAX, 0, 0x8000_0000, 7, 3, 7, 1];
    let mut radix_keys = DeviceBuffer::from_host(&stream, &radix_input)?;
    let mut radix_workspace =
        rccl_cub::device_radix_sort_workspace!(::<u32>(&launch, radix_input.len()))?;
    rccl_cub::device_radix_sort_keys!(::<u32>(
        &launch,
        &mut radix_keys,
        &mut radix_workspace,
    ))?;
    assert_eq!(
        radix_keys.to_host_vec(&stream)?,
        [0, 1, 3, 7, 7, 0x8000_0000, u32::MAX]
    );

    let signed_input = [0_i32, -1, i32::MIN, 7, -9, i32::MAX];
    let mut signed_radix_keys = DeviceBuffer::from_host(&stream, &signed_input)?;
    let mut signed_workspace =
        rccl_cub::device_radix_sort_workspace!(::<i32>(&launch, signed_input.len()))?;
    rccl_cub::device_radix_sort_keys!(::<i32>(
        &launch,
        &mut signed_radix_keys,
        &mut signed_workspace,
    ))?;
    assert_eq!(
        signed_radix_keys.to_host_vec(&stream)?,
        [i32::MIN, -9, -1, 0, 7, i32::MAX]
    );

    rccl_cub::device_radix_sort_keys_descending!(::<i32>(
        &launch,
        &mut signed_radix_keys,
        &mut signed_workspace,
    ))?;
    assert_eq!(
        signed_radix_keys.to_host_vec(&stream)?,
        [i32::MAX, 7, 0, -1, -9, i32::MIN]
    );

    // Radix pairs stay absent here: the proc-macro test requires them to fail
    // closed until the exact CUB pair histogram/scan/scatter graph exists.
    println!("generated CUB DeviceMerge/DeviceMergeSort/DeviceRadixSort keys probe: PASS");
    Ok(())
}
