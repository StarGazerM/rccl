use core::cmp::Ordering;

use cuda_core::{CudaStream, DeviceCopy, LaunchConfig1D};
use cuda_host::{KernelSliceArg, KernelSliceArgMut};

use crate::{
    DeviceAlgorithmError, DeviceMergeSortKeysWorkspace, DeviceMergeSortPairsWorkspace,
    workspace::WorkspaceBuffer,
};

const MERGE_SORT_THREADS: usize = 256;

#[doc(hidden)]
pub const fn merge_sort_items(key_size: usize) -> usize {
    if key_size == 0 {
        return 0;
    }
    let items = 17 * 4 / key_size;
    if items == 0 {
        1
    } else if items > 17 {
        17
    } else {
        items
    }
}

const fn align_16(bytes: usize) -> usize {
    bytes.div_ceil(16) * 16
}

#[doc(hidden)]
pub const fn merge_sort_shared_bytes(items: usize, key_size: usize, value_size: usize) -> usize {
    let tile_items = MERGE_SORT_THREADS * items;
    let exchange_items = if items > 4 && items.is_power_of_two() {
        tile_items + tile_items / 32
    } else {
        tile_items
    };
    let key_merge = (tile_items + 1) * key_size;
    let key_store = align_16(align_16(exchange_items * key_size) + 4);
    let mut bytes = if key_merge > key_store {
        key_merge
    } else {
        key_store
    };
    if value_size != 0 {
        let value_merge = (tile_items + 1) * value_size;
        let value_store = align_16(align_16(exchange_items * value_size) + 4);
        if value_merge > bytes {
            bytes = value_merge;
        }
        if value_store > bytes {
            bytes = value_store;
        }
    }
    align_16(bytes)
}

/// Generated-module boundary for DeviceMergeSort's key kernels.
///
/// The implementation only prepares and launches generated kernels. Shape
/// checks, ping-pong selection, and pass control stay in ordinary Rust.
#[doc(hidden)]
pub trait MergeSortKeysKernel<K, F> {
    fn launch_merge_sort_block_keys<Input, Output, const ITEMS: usize, const SHARED: usize>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        input: &Input,
        order: F,
        output: &mut Output,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Input: KernelSliceArg<Elem = K>,
        Output: KernelSliceArgMut<Elem = K>;

    fn launch_merge_sort_block_keys_in_place<Keys, const ITEMS: usize, const SHARED: usize>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        order: F,
        keys: &mut Keys,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Keys: KernelSliceArgMut<Elem = K>;

    fn launch_merge_sort_partition<Input, Partitions, const ITEMS: usize>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        input: &Input,
        target_merged_tiles: usize,
        order: F,
        partitions: &mut Partitions,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Input: KernelSliceArg<Elem = K>,
        Partitions: KernelSliceArgMut<Elem = usize>;

    #[allow(clippy::too_many_arguments)]
    fn launch_merge_sort_merge_keys<
        Input,
        Partitions,
        Output,
        const ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        input: &Input,
        partitions: &Partitions,
        target_merged_tiles: usize,
        order: F,
        output: &mut Output,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Input: KernelSliceArg<Elem = K>,
        Partitions: KernelSliceArg<Elem = usize>,
        Output: KernelSliceArgMut<Elem = K>;
}

fn validate_len(
    role: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), DeviceAlgorithmError> {
    if actual == expected {
        Ok(())
    } else {
        Err(DeviceAlgorithmError::LengthMismatch {
            role,
            expected,
            actual,
        })
    }
}

fn merge_sort_shape(
    len: usize,
    items: usize,
    partition_len: usize,
) -> Result<(usize, usize), DeviceAlgorithmError> {
    if len == 0 {
        validate_len("merge-sort partitions", 0, partition_len)?;
        return Ok((0, 0));
    }
    let num_tiles = len.div_ceil(MERGE_SORT_THREADS * items);
    validate_len("merge-sort partitions", num_tiles + 1, partition_len)?;
    let num_passes = usize::BITS as usize - (num_tiles - 1).leading_zeros() as usize;
    Ok((num_tiles, num_passes))
}

#[allow(clippy::too_many_arguments)]
fn merge_sort_keys_passes<
    K,
    M,
    F,
    Primary,
    Scratch,
    Partitions,
    const ITEMS: usize,
    const SHARED: usize,
>(
    module: &M,
    stream: &CudaStream,
    primary: &mut Primary,
    scratch: &mut Scratch,
    partitions: &mut Partitions,
    order: F,
    num_tiles: usize,
    num_passes: usize,
    mut ping: bool,
) -> Result<(), DeviceAlgorithmError>
where
    K: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    Primary: KernelSliceArgMut<Elem = K>,
    Scratch: KernelSliceArgMut<Elem = K>,
    Partitions: KernelSliceArgMut<Elem = usize>,
    M: MergeSortKeysKernel<K, F>,
{
    let agent_config = LaunchConfig1D::new(
        num_tiles.min(u32::MAX as usize) as u32,
        MERGE_SORT_THREADS as u32,
        SHARED as u32,
    );
    let partition_config = LaunchConfig1D::new(
        (num_tiles + 1).div_ceil(256).min(u32::MAX as usize) as u32,
        256,
        0,
    );
    for pass in 0..num_passes {
        let target_merged_tiles = 2usize << pass;
        if ping {
            module.launch_merge_sort_partition::<_, _, ITEMS>(
                stream,
                partition_config,
                &*primary,
                target_merged_tiles,
                order,
                partitions,
            )?;
            module.launch_merge_sort_merge_keys::<_, _, _, ITEMS, SHARED>(
                stream,
                agent_config,
                &*primary,
                &*partitions,
                target_merged_tiles,
                order,
                scratch,
            )?;
        } else {
            module.launch_merge_sort_partition::<_, _, ITEMS>(
                stream,
                partition_config,
                &*scratch,
                target_merged_tiles,
                order,
                partitions,
            )?;
            module.launch_merge_sort_merge_keys::<_, _, _, ITEMS, SHARED>(
                stream,
                agent_config,
                &*scratch,
                &*partitions,
                target_merged_tiles,
                order,
                primary,
            )?;
        }
        ping = !ping;
    }
    Ok(())
}

#[doc(hidden)]
pub(crate) fn device_merge_sort_keys<K, M, F, S, P, Keys, const ITEMS: usize, const SHARED: usize>(
    module: &M,
    stream: &CudaStream,
    keys: &mut Keys,
    workspace: &mut DeviceMergeSortKeysWorkspace<K, S, P>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    P: WorkspaceBuffer<usize>,
    S::Range: KernelSliceArgMut<Elem = K>,
    P::Range: KernelSliceArgMut<Elem = usize>,
    Keys: KernelSliceArgMut<Elem = K>,
    M: MergeSortKeysKernel<K, F>,
{
    let len = keys.len();
    let (scratch, partitions) = workspace.__parts_mut();
    validate_len("merge-sort scratch keys", len, scratch.len())?;
    let (num_tiles, num_passes) = merge_sort_shape(len, ITEMS, partitions.len())?;
    if len == 0 {
        return Ok(());
    }
    let ping = (num_passes & 1) == 0;
    let config = LaunchConfig1D::new(
        num_tiles.min(u32::MAX as usize) as u32,
        MERGE_SORT_THREADS as u32,
        SHARED as u32,
    );
    if ping {
        module.launch_merge_sort_block_keys_in_place::<_, ITEMS, SHARED>(
            stream, config, order, keys,
        )?;
    } else {
        module.launch_merge_sort_block_keys::<_, _, ITEMS, SHARED>(
            stream, config, &*keys, order, scratch,
        )?;
    }
    merge_sort_keys_passes::<K, _, _, _, _, _, ITEMS, SHARED>(
        module, stream, keys, scratch, partitions, order, num_tiles, num_passes, ping,
    )
}

#[doc(hidden)]
pub(crate) fn device_merge_sort_keys_copy<
    K,
    M,
    F,
    S,
    P,
    Input,
    Output,
    const ITEMS: usize,
    const SHARED: usize,
>(
    module: &M,
    stream: &CudaStream,
    input: &Input,
    workspace: &mut DeviceMergeSortKeysWorkspace<K, S, P>,
    output: &mut Output,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    P: WorkspaceBuffer<usize>,
    S::Range: KernelSliceArgMut<Elem = K>,
    P::Range: KernelSliceArgMut<Elem = usize>,
    Input: KernelSliceArg<Elem = K>,
    Output: KernelSliceArgMut<Elem = K>,
    M: MergeSortKeysKernel<K, F>,
{
    let len = input.len();
    validate_len("output keys", len, output.len())?;
    let (scratch, partitions) = workspace.__parts_mut();
    validate_len("merge-sort scratch keys", len, scratch.len())?;
    let (num_tiles, num_passes) = merge_sort_shape(len, ITEMS, partitions.len())?;
    if len == 0 {
        return Ok(());
    }
    let ping = (num_passes & 1) == 0;
    let config = LaunchConfig1D::new(
        num_tiles.min(u32::MAX as usize) as u32,
        MERGE_SORT_THREADS as u32,
        SHARED as u32,
    );
    if ping {
        module.launch_merge_sort_block_keys::<_, _, ITEMS, SHARED>(
            stream, config, input, order, output,
        )?;
    } else {
        module.launch_merge_sort_block_keys::<_, _, ITEMS, SHARED>(
            stream, config, input, order, scratch,
        )?;
    }
    merge_sort_keys_passes::<K, _, _, _, _, _, ITEMS, SHARED>(
        module, stream, output, scratch, partitions, order, num_tiles, num_passes, ping,
    )
}

/// Generated-module boundary for DeviceMergeSort's key/value kernels.
#[doc(hidden)]
pub trait MergeSortPairsKernel<K, V, F> {
    #[allow(clippy::too_many_arguments)]
    fn launch_merge_sort_block_pairs<
        InputKeys,
        InputValues,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        input_keys: &InputKeys,
        input_values: &InputValues,
        order: F,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
    ) -> Result<(), DeviceAlgorithmError>
    where
        InputKeys: KernelSliceArg<Elem = K>,
        InputValues: KernelSliceArg<Elem = V>,
        OutputKeys: KernelSliceArgMut<Elem = K>,
        OutputValues: KernelSliceArgMut<Elem = V>;

    fn launch_merge_sort_block_pairs_in_place<
        Keys,
        Values,
        const ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        order: F,
        keys: &mut Keys,
        values: &mut Values,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Keys: KernelSliceArgMut<Elem = K>,
        Values: KernelSliceArgMut<Elem = V>;

    #[allow(clippy::too_many_arguments)]
    fn launch_merge_sort_merge_pairs<
        InputKeys,
        InputValues,
        Partitions,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        input_keys: &InputKeys,
        input_values: &InputValues,
        partitions: &Partitions,
        target_merged_tiles: usize,
        order: F,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
    ) -> Result<(), DeviceAlgorithmError>
    where
        InputKeys: KernelSliceArg<Elem = K>,
        InputValues: KernelSliceArg<Elem = V>,
        Partitions: KernelSliceArg<Elem = usize>,
        OutputKeys: KernelSliceArgMut<Elem = K>,
        OutputValues: KernelSliceArgMut<Elem = V>;
}

#[allow(clippy::too_many_arguments)]
fn merge_sort_pairs_passes<
    K,
    V,
    M,
    F,
    PrimaryKeys,
    PrimaryValues,
    ScratchKeys,
    ScratchValues,
    Partitions,
    const ITEMS: usize,
    const SHARED: usize,
>(
    module: &M,
    stream: &CudaStream,
    primary_keys: &mut PrimaryKeys,
    primary_values: &mut PrimaryValues,
    scratch_keys: &mut ScratchKeys,
    scratch_values: &mut ScratchValues,
    partitions: &mut Partitions,
    order: F,
    num_tiles: usize,
    num_passes: usize,
    mut ping: bool,
) -> Result<(), DeviceAlgorithmError>
where
    K: DeviceCopy,
    V: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    PrimaryKeys: KernelSliceArgMut<Elem = K>,
    PrimaryValues: KernelSliceArgMut<Elem = V>,
    ScratchKeys: KernelSliceArgMut<Elem = K>,
    ScratchValues: KernelSliceArgMut<Elem = V>,
    Partitions: KernelSliceArgMut<Elem = usize>,
    M: MergeSortKeysKernel<K, F> + MergeSortPairsKernel<K, V, F>,
{
    let agent_config = LaunchConfig1D::new(
        num_tiles.min(u32::MAX as usize) as u32,
        MERGE_SORT_THREADS as u32,
        SHARED as u32,
    );
    let partition_config = LaunchConfig1D::new(
        (num_tiles + 1).div_ceil(256).min(u32::MAX as usize) as u32,
        256,
        0,
    );
    for pass in 0..num_passes {
        let target_merged_tiles = 2usize << pass;
        if ping {
            module.launch_merge_sort_partition::<_, _, ITEMS>(
                stream,
                partition_config,
                &*primary_keys,
                target_merged_tiles,
                order,
                partitions,
            )?;
            module.launch_merge_sort_merge_pairs::<_, _, _, _, _, ITEMS, SHARED>(
                stream,
                agent_config,
                &*primary_keys,
                &*primary_values,
                &*partitions,
                target_merged_tiles,
                order,
                scratch_keys,
                scratch_values,
            )?;
        } else {
            module.launch_merge_sort_partition::<_, _, ITEMS>(
                stream,
                partition_config,
                &*scratch_keys,
                target_merged_tiles,
                order,
                partitions,
            )?;
            module.launch_merge_sort_merge_pairs::<_, _, _, _, _, ITEMS, SHARED>(
                stream,
                agent_config,
                &*scratch_keys,
                &*scratch_values,
                &*partitions,
                target_merged_tiles,
                order,
                primary_keys,
                primary_values,
            )?;
        }
        ping = !ping;
    }
    Ok(())
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_merge_sort_pairs<
    K,
    V,
    M,
    F,
    S,
    U,
    P,
    Keys,
    Values,
    const ITEMS: usize,
    const SHARED: usize,
>(
    module: &M,
    stream: &CudaStream,
    keys: &mut Keys,
    values: &mut Values,
    workspace: &mut DeviceMergeSortPairsWorkspace<K, V, S, U, P>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: DeviceCopy,
    V: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    P: WorkspaceBuffer<usize>,
    S::Range: KernelSliceArgMut<Elem = K>,
    U::Range: KernelSliceArgMut<Elem = V>,
    P::Range: KernelSliceArgMut<Elem = usize>,
    Keys: KernelSliceArgMut<Elem = K>,
    Values: KernelSliceArgMut<Elem = V>,
    M: MergeSortKeysKernel<K, F> + MergeSortPairsKernel<K, V, F>,
{
    let len = keys.len();
    validate_len("values", len, values.len())?;
    let (scratch_keys, scratch_values, partitions) = workspace.__parts_mut();
    validate_len("merge-sort scratch keys", len, scratch_keys.len())?;
    validate_len("merge-sort scratch values", len, scratch_values.len())?;
    let (num_tiles, num_passes) = merge_sort_shape(len, ITEMS, partitions.len())?;
    if len == 0 {
        return Ok(());
    }
    let ping = (num_passes & 1) == 0;
    let config = LaunchConfig1D::new(
        num_tiles.min(u32::MAX as usize) as u32,
        MERGE_SORT_THREADS as u32,
        SHARED as u32,
    );
    if ping {
        module.launch_merge_sort_block_pairs_in_place::<_, _, ITEMS, SHARED>(
            stream, config, order, keys, values,
        )?;
    } else {
        module.launch_merge_sort_block_pairs::<_, _, _, _, ITEMS, SHARED>(
            stream,
            config,
            &*keys,
            &*values,
            order,
            scratch_keys,
            scratch_values,
        )?;
    }
    merge_sort_pairs_passes::<K, V, _, _, _, _, _, _, _, ITEMS, SHARED>(
        module,
        stream,
        keys,
        values,
        scratch_keys,
        scratch_values,
        partitions,
        order,
        num_tiles,
        num_passes,
        ping,
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_merge_sort_pairs_copy<
    K,
    V,
    M,
    F,
    S,
    U,
    P,
    InputKeys,
    InputValues,
    OutputKeys,
    OutputValues,
    const ITEMS: usize,
    const SHARED: usize,
>(
    module: &M,
    stream: &CudaStream,
    input_keys: &InputKeys,
    input_values: &InputValues,
    workspace: &mut DeviceMergeSortPairsWorkspace<K, V, S, U, P>,
    output_keys: &mut OutputKeys,
    output_values: &mut OutputValues,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: DeviceCopy,
    V: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    P: WorkspaceBuffer<usize>,
    S::Range: KernelSliceArgMut<Elem = K>,
    U::Range: KernelSliceArgMut<Elem = V>,
    P::Range: KernelSliceArgMut<Elem = usize>,
    InputKeys: KernelSliceArg<Elem = K>,
    InputValues: KernelSliceArg<Elem = V>,
    OutputKeys: KernelSliceArgMut<Elem = K>,
    OutputValues: KernelSliceArgMut<Elem = V>,
    M: MergeSortKeysKernel<K, F> + MergeSortPairsKernel<K, V, F>,
{
    let len = input_keys.len();
    validate_len("input values", len, input_values.len())?;
    validate_len("output keys", len, output_keys.len())?;
    validate_len("output values", len, output_values.len())?;
    let (scratch_keys, scratch_values, partitions) = workspace.__parts_mut();
    validate_len("merge-sort scratch keys", len, scratch_keys.len())?;
    validate_len("merge-sort scratch values", len, scratch_values.len())?;
    let (num_tiles, num_passes) = merge_sort_shape(len, ITEMS, partitions.len())?;
    if len == 0 {
        return Ok(());
    }
    let ping = (num_passes & 1) == 0;
    let config = LaunchConfig1D::new(
        num_tiles.min(u32::MAX as usize) as u32,
        MERGE_SORT_THREADS as u32,
        SHARED as u32,
    );
    if ping {
        module.launch_merge_sort_block_pairs::<_, _, _, _, ITEMS, SHARED>(
            stream,
            config,
            input_keys,
            input_values,
            order,
            output_keys,
            output_values,
        )?;
    } else {
        module.launch_merge_sort_block_pairs::<_, _, _, _, ITEMS, SHARED>(
            stream,
            config,
            input_keys,
            input_values,
            order,
            scratch_keys,
            scratch_values,
        )?;
    }
    merge_sort_pairs_passes::<K, V, _, _, _, _, _, _, _, ITEMS, SHARED>(
        module,
        stream,
        output_keys,
        output_values,
        scratch_keys,
        scratch_values,
        partitions,
        order,
        num_tiles,
        num_passes,
        ping,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_layout_drives_merge_sort_policy() {
        type Alias = u32;
        #[repr(C)]
        struct UserKey([u32; 3]);

        assert_eq!(merge_sort_items(core::mem::size_of::<Alias>()), 17);
        assert_eq!(merge_sort_items(core::mem::size_of::<UserKey>()), 5);
        assert_eq!(merge_sort_shared_bytes(17, 4, 0), 17_424);
    }
}
