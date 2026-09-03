use core::cmp::Ordering;

use cuda_core::{CudaStream, DeviceCopy, LaunchConfig1D};
use cuda_host::{KernelSliceArg, KernelSliceArgMut};

use crate::{DeviceAlgorithmError, DeviceMergeWorkspace, workspace::WorkspaceBuffer};

const MERGE_THREADS: usize = 512;

#[doc(hidden)]
pub const fn merge_items(item_size: usize) -> usize {
    if item_size == 0 {
        return 0;
    }
    let items = 15 * 4 / item_size;
    if items == 0 {
        1
    } else if items > 15 {
        15
    } else {
        items
    }
}

const fn align_16(bytes: usize) -> usize {
    bytes.div_ceil(16) * 16
}

const fn load_to_shared_buffer_bytes(items: usize, item_size: usize) -> usize {
    let bytes = items * item_size;
    if item_size >= 16 || bytes == 0 {
        bytes
    } else {
        align_16(bytes) + 16
    }
}

const fn merge_representation_bytes(tile_items: usize, item_size: usize, load: bool) -> usize {
    if load {
        load_to_shared_buffer_bytes(tile_items + 1, item_size) + if item_size < 16 { 32 } else { 0 }
    } else {
        (tile_items + 1) * item_size
    }
}

const fn merge_block_store_bytes(exchange_items: usize, item_size: usize) -> usize {
    align_16(align_16(exchange_items * item_size) + 4)
}

#[doc(hidden)]
pub const fn merge_load_to_shared_keys(key_size: usize, value_size: usize) -> bool {
    #[cfg(cuda_arch_ge_100)]
    {
        return true;
    }
    #[cfg(all(cuda_arch_ge_90, not(cuda_arch_ge_100)))]
    {
        return if value_size == 0 {
            key_size != 8
        } else {
            match key_size {
                8 => value_size != 8,
                16 => value_size == 1 || value_size == 8,
                _ => true,
            }
        };
    }
    #[cfg(not(cuda_arch_ge_90))]
    {
        if value_size == 0 {
            key_size < 4
        } else {
            key_size == 1 || (key_size == 2 && value_size < 4) || (key_size == 4 && value_size == 1)
        }
    }
}

#[doc(hidden)]
pub const fn merge_shared_bytes(
    items: usize,
    key_size: usize,
    value_size: usize,
    load_keys: bool,
    load_values: bool,
) -> usize {
    let tile_items = MERGE_THREADS * items;
    let exchange_items = if items > 4 && items.is_power_of_two() {
        tile_items + tile_items / 32
    } else {
        tile_items
    };
    let mut union_bytes = {
        let representation = merge_representation_bytes(tile_items, key_size, load_keys);
        let store = merge_block_store_bytes(exchange_items, key_size);
        if representation > store {
            representation
        } else {
            store
        }
    };
    if value_size != 0 {
        let value_representation = merge_representation_bytes(tile_items, value_size, load_values);
        let value_store = merge_block_store_bytes(exchange_items, value_size);
        if value_representation > union_bytes {
            union_bytes = value_representation;
        }
        if value_store > union_bytes {
            union_bytes = value_store;
        }
    }
    union_bytes = align_16(union_bytes);
    if load_keys || load_values {
        align_16(union_bytes + 8)
    } else {
        union_bytes
    }
}

/// Launch boundary for the two DeviceMerge kernels.
///
/// The concrete CUDA module implements this internal interface. Validation and
/// algorithm control flow remain in ordinary Rust below.
#[doc(hidden)]
pub trait MergeKeysKernel<K, F> {
    #[allow(clippy::too_many_arguments)]
    fn launch_merge_keys<
        First,
        Second,
        Partitions,
        Output,
        const ITEMS: usize,
        const K_SIZE: usize,
        const LOAD: bool,
        const SHARED: usize,
    >(
        &self,
        stream: &CudaStream,
        first: &First,
        second: &Second,
        partitions: &mut Partitions,
        output: &mut Output,
        order: F,
        num_tiles: usize,
    ) -> Result<(), DeviceAlgorithmError>
    where
        First: KernelSliceArg<Elem = K>,
        Second: KernelSliceArg<Elem = K>,
        Partitions: KernelSliceArgMut<Elem = usize>,
        Output: KernelSliceArgMut<Elem = K>;
}

#[doc(hidden)]
pub trait MergePairsKernel<K, V, F> {
    #[allow(clippy::too_many_arguments)]
    fn launch_merge_pairs<
        FirstKeys,
        FirstValues,
        SecondKeys,
        SecondValues,
        Partitions,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
        const K_SIZE: usize,
        const V_SIZE: usize,
        const LOAD_KEYS: bool,
        const LOAD_VALUES: bool,
        const SHARED: usize,
    >(
        &self,
        stream: &CudaStream,
        first_keys: &FirstKeys,
        first_values: &FirstValues,
        second_keys: &SecondKeys,
        second_values: &SecondValues,
        partitions: &mut Partitions,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
        order: F,
        num_tiles: usize,
    ) -> Result<(), DeviceAlgorithmError>
    where
        FirstKeys: KernelSliceArg<Elem = K>,
        FirstValues: KernelSliceArg<Elem = V>,
        SecondKeys: KernelSliceArg<Elem = K>,
        SecondValues: KernelSliceArg<Elem = V>,
        Partitions: KernelSliceArgMut<Elem = usize>,
        OutputKeys: KernelSliceArgMut<Elem = K>,
        OutputValues: KernelSliceArgMut<Elem = V>;
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_merge_keys<
    K,
    M,
    F,
    P,
    First,
    Second,
    Output,
    const ITEMS: usize,
    const K_SIZE: usize,
    const LOAD: bool,
    const SHARED: usize,
>(
    module: &M,
    stream: &CudaStream,
    first: &First,
    second: &Second,
    workspace: &mut DeviceMergeWorkspace<P>,
    output: &mut Output,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    P: WorkspaceBuffer<usize>,
    P::Range: KernelSliceArgMut<Elem = usize>,
    First: KernelSliceArg<Elem = K>,
    Second: KernelSliceArg<Elem = K>,
    Output: KernelSliceArgMut<Elem = K>,
    M: MergeKeysKernel<K, F>,
{
    let output_len = first
        .len()
        .checked_add(second.len())
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if output.len() != output_len {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "output",
            expected: output_len,
            actual: output.len(),
        });
    }
    let partitions = workspace.__partitions_mut();
    if output_len == 0 {
        return Ok(());
    }
    let num_tiles = output_len.div_ceil(ITEMS * MERGE_THREADS);
    let required_partitions = num_tiles + 1;
    if partitions.len() != required_partitions {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "merge partitions",
            expected: required_partitions,
            actual: partitions.len(),
        });
    }
    module.launch_merge_keys::<_, _, _, _, ITEMS, K_SIZE, LOAD, SHARED>(
        stream, first, second, partitions, output, order, num_tiles,
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_merge_pairs<
    K,
    V,
    M,
    F,
    P,
    FirstKeys,
    FirstValues,
    SecondKeys,
    SecondValues,
    OutputKeys,
    OutputValues,
    const ITEMS: usize,
    const K_SIZE: usize,
    const V_SIZE: usize,
    const LOAD_KEYS: bool,
    const LOAD_VALUES: bool,
    const SHARED: usize,
>(
    module: &M,
    stream: &CudaStream,
    first_keys: &FirstKeys,
    first_values: &FirstValues,
    second_keys: &SecondKeys,
    second_values: &SecondValues,
    workspace: &mut DeviceMergeWorkspace<P>,
    output_keys: &mut OutputKeys,
    output_values: &mut OutputValues,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: DeviceCopy,
    V: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    P: WorkspaceBuffer<usize>,
    P::Range: KernelSliceArgMut<Elem = usize>,
    FirstKeys: KernelSliceArg<Elem = K>,
    FirstValues: KernelSliceArg<Elem = V>,
    SecondKeys: KernelSliceArg<Elem = K>,
    SecondValues: KernelSliceArg<Elem = V>,
    OutputKeys: KernelSliceArgMut<Elem = K>,
    OutputValues: KernelSliceArgMut<Elem = V>,
    M: MergePairsKernel<K, V, F>,
{
    let first_len = first_keys.len();
    let second_len = second_keys.len();
    for (role, expected, actual) in [
        ("first values", first_len, first_values.len()),
        ("second values", second_len, second_values.len()),
    ] {
        if actual != expected {
            return Err(DeviceAlgorithmError::LengthMismatch {
                role,
                expected,
                actual,
            });
        }
    }
    let output_len = first_len
        .checked_add(second_len)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    for (role, actual) in [
        ("output keys", output_keys.len()),
        ("output values", output_values.len()),
    ] {
        if actual != output_len {
            return Err(DeviceAlgorithmError::LengthMismatch {
                role,
                expected: output_len,
                actual,
            });
        }
    }
    let partitions = workspace.__partitions_mut();
    if output_len == 0 {
        return Ok(());
    }
    let num_tiles = output_len.div_ceil(ITEMS * MERGE_THREADS);
    let required_partitions = num_tiles + 1;
    if partitions.len() != required_partitions {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "merge partitions",
            expected: required_partitions,
            actual: partitions.len(),
        });
    }
    module.launch_merge_pairs::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            ITEMS,
            K_SIZE,
            V_SIZE,
            LOAD_KEYS,
            LOAD_VALUES,
            SHARED,
        >(
            stream,
            first_keys,
            first_values,
            second_keys,
            second_values,
            partitions,
            output_keys,
            output_values,
            order,
            num_tiles,
        )
}

#[doc(hidden)]
pub fn merge_partition_launch(num_tiles: usize) -> LaunchConfig1D {
    LaunchConfig1D::new(
        (num_tiles + 1).div_ceil(256).min(u32::MAX as usize) as u32,
        256,
        0,
    )
}

#[doc(hidden)]
pub fn merge_agent_launch(num_tiles: usize, shared: usize) -> LaunchConfig1D {
    LaunchConfig1D::new(
        num_tiles.min(u32::MAX as usize) as u32,
        MERGE_THREADS as u32,
        shared as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concrete_rust_layout_replaces_proc_macro_type_parsing() {
        type KeyAlias = u32;
        #[repr(C)]
        struct UserKey {
            words: [u32; 3],
        }

        assert_eq!(merge_items(core::mem::size_of::<KeyAlias>()), 15);
        assert_eq!(merge_items(core::mem::size_of::<UserKey>()), 5);
    }

    #[test]
    #[cfg(not(cuda_arch_ge_90))]
    fn sm80_policy_matches_the_removed_proc_macro() {
        assert!(!merge_load_to_shared_keys(4, 0));
        assert_eq!(merge_shared_bytes(15, 4, 0, false, false), 30_736);

        assert!(merge_load_to_shared_keys(1, 0));
        assert_eq!(merge_shared_bytes(15, 1, 0, true, false), 7_760);

        assert!(!merge_load_to_shared_keys(4, 8));
        assert_eq!(merge_shared_bytes(5, 4, 8, false, false), 20_496);

        assert!(merge_load_to_shared_keys(4, 1));
        assert_eq!(merge_shared_bytes(12, 4, 1, true, true), 24_656);
    }
}
