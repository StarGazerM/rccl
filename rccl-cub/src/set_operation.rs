use core::cmp::Ordering;

use cuda_core::{CudaStream, DeviceCopy};
use cuda_host::{KernelSliceArg, KernelSliceArgMut};

use crate::{
    ArithmeticRange, DeviceAlgorithmError, DeviceSetOperationWorkspace,
    ordering_backend::{SerialSetOperation, SetIndex},
    workspace::WorkspaceBuffer,
};

const SET_THREADS: usize = 512;

#[doc(hidden)]
pub const fn set_items(key_size: usize) -> usize {
    if key_size == 0 {
        return 0;
    }
    let items = (19usize * 4).div_ceil(key_size);
    if items == 0 {
        1
    } else if items > 19 {
        19
    } else {
        items
    }
}

#[doc(hidden)]
pub const fn set_shared_bytes(items: usize, key_size: usize, value_size: usize) -> usize {
    let tile_items = SET_THREADS * items - 1;
    let shared_items = tile_items + SET_THREADS;
    let union_bytes = shared_items
        * if key_size > value_size {
            key_size
        } else {
            value_size
        };
    (SET_THREADS * 4 + union_bytes).div_ceil(16) * 16
}

/// Generated-module boundary for the set-operation kernel graph.
///
/// Separate implementations for `i32` and `i64` select the generated kernel
/// spelling. Runtime validation and index-width selection remain outside this
/// trait in ordinary Rust and the Thrust backend respectively.
#[doc(hidden)]
pub trait SetKernel<I, K, F, Op>
where
    I: SetIndex,
    Op: SerialSetOperation,
{
    #[allow(clippy::too_many_arguments)]
    fn launch_set_keys<
        First,
        Second,
        O,
        D,
        P,
        W,
        Output,
        const ITEMS: usize,
        const TILE_ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &CudaStream,
        first: &First,
        second: &Second,
        output_count: &mut O,
        tile_state: &mut D,
        partitions: &mut P,
        virtual_shared: &mut W,
        output: &mut Output,
        order: F,
        num_tiles: usize,
    ) -> Result<(), DeviceAlgorithmError>
    where
        First: KernelSliceArg<Elem = K>,
        Second: KernelSliceArg<Elem = K>,
        O: KernelSliceArgMut<Elem = usize>,
        D: KernelSliceArgMut<Elem = I::Transaction>,
        P: KernelSliceArgMut<Elem = I::Transaction>,
        W: KernelSliceArgMut<Elem = u8>,
        Output: KernelSliceArgMut<Elem = K>;

    #[allow(clippy::too_many_arguments)]
    fn launch_set_discard<
        First,
        Second,
        O,
        D,
        P,
        W,
        const ITEMS: usize,
        const TILE_ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &CudaStream,
        first: &First,
        second: &Second,
        output_count: &mut O,
        tile_state: &mut D,
        partitions: &mut P,
        virtual_shared: &mut W,
        order: F,
        num_tiles: usize,
    ) -> Result<(), DeviceAlgorithmError>
    where
        First: KernelSliceArg<Elem = K>,
        Second: KernelSliceArg<Elem = K>,
        O: KernelSliceArgMut<Elem = usize>,
        D: KernelSliceArgMut<Elem = I::Transaction>,
        P: KernelSliceArgMut<Elem = I::Transaction>,
        W: KernelSliceArgMut<Elem = u8>;

    #[allow(clippy::too_many_arguments)]
    fn launch_set_pairs<
        V,
        FirstKeys,
        FirstValues,
        SecondKeys,
        SecondValues,
        O,
        D,
        P,
        W,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
        const TILE_ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &CudaStream,
        first_keys: &FirstKeys,
        first_values: &FirstValues,
        second_keys: &SecondKeys,
        second_values: &SecondValues,
        output_count: &mut O,
        tile_state: &mut D,
        partitions: &mut P,
        virtual_shared: &mut W,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
        order: F,
        num_tiles: usize,
    ) -> Result<(), DeviceAlgorithmError>
    where
        V: DeviceCopy,
        FirstKeys: KernelSliceArg<Elem = K>,
        FirstValues: KernelSliceArg<Elem = V>,
        SecondKeys: KernelSliceArg<Elem = K>,
        SecondValues: KernelSliceArg<Elem = V>,
        O: KernelSliceArgMut<Elem = usize>,
        D: KernelSliceArgMut<Elem = I::Transaction>,
        P: KernelSliceArgMut<Elem = I::Transaction>,
        W: KernelSliceArgMut<Elem = u8>,
        OutputKeys: KernelSliceArgMut<Elem = K>,
        OutputValues: KernelSliceArgMut<Elem = V>;
}

#[doc(hidden)]
pub trait ArithmeticSetKernel<I, K, F, Op>
where
    I: SetIndex,
    Op: SerialSetOperation,
{
    #[allow(clippy::too_many_arguments)]
    fn launch_set_arithmetic<
        O,
        D,
        P,
        W,
        Output,
        const ITEMS: usize,
        const TILE_ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &CudaStream,
        first_start: K,
        first_step: K,
        first_len: usize,
        second_start: K,
        second_step: K,
        second_len: usize,
        output_count: &mut O,
        tile_state: &mut D,
        partitions: &mut P,
        virtual_shared: &mut W,
        output: &mut Output,
        order: F,
        num_tiles: usize,
    ) -> Result<(), DeviceAlgorithmError>
    where
        O: KernelSliceArgMut<Elem = usize>,
        D: KernelSliceArgMut<Elem = I::Transaction>,
        P: KernelSliceArgMut<Elem = I::Transaction>,
        W: KernelSliceArgMut<Elem = u8>,
        Output: KernelSliceArgMut<Elem = K>;
}

fn validate_len(
    role: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), DeviceAlgorithmError> {
    if expected == actual {
        Ok(())
    } else {
        Err(DeviceAlgorithmError::LengthMismatch {
            role,
            expected,
            actual,
        })
    }
}

fn validate_workspace<I, O, D, P, W>(
    output_count: &O,
    tile_state: &D,
    partitions: &P,
    virtual_shared: &W,
    num_tiles: usize,
    shared: usize,
) -> Result<(), DeviceAlgorithmError>
where
    I: SetIndex,
    O: KernelSliceArg<Elem = usize>,
    D: KernelSliceArg<Elem = I::Transaction>,
    P: KernelSliceArg<Elem = I::Transaction>,
    W: KernelSliceArg<Elem = u8>,
{
    validate_len("set output count", 1, output_count.len())?;
    validate_len("set tile descriptors", num_tiles + 32, tile_state.len())?;
    validate_len("set balanced partitions", num_tiles + 1, partitions.len())?;
    if !virtual_shared.is_empty() {
        let required = shared
            .checked_mul(num_tiles)
            .ok_or(DeviceAlgorithmError::SizeOverflow)?;
        validate_len("set virtual shared memory", required, virtual_shared.len())?;
    }
    Ok(())
}

#[allow(clippy::type_complexity)]
fn set_parts<I, O, D, P, W>(
    workspace: &mut DeviceSetOperationWorkspace<O, D, P, W>,
) -> (&mut O::Range, &mut D::Range, &mut P::Range, &mut W::Range)
where
    I: SetIndex,
    I::Transaction: DeviceCopy,
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<I::Transaction>,
    P: WorkspaceBuffer<I::Transaction>,
    W: WorkspaceBuffer<u8>,
{
    workspace.__parts_mut::<I>()
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_set_keys<
    I,
    K,
    M,
    F,
    Op,
    O,
    D,
    P,
    W,
    First,
    Second,
    Output,
    const ITEMS: usize,
    const TILE_ITEMS: usize,
    const SHARED: usize,
>(
    module: &M,
    stream: &CudaStream,
    first: &First,
    second: &Second,
    workspace: &mut DeviceSetOperationWorkspace<O, D, P, W>,
    output: &mut Output,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    I: SetIndex,
    I::Transaction: DeviceCopy,
    K: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: SerialSetOperation,
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<I::Transaction>,
    P: WorkspaceBuffer<I::Transaction>,
    W: WorkspaceBuffer<u8>,
    O::Range: KernelSliceArgMut<Elem = usize>,
    D::Range: KernelSliceArgMut<Elem = I::Transaction>,
    P::Range: KernelSliceArgMut<Elem = I::Transaction>,
    W::Range: KernelSliceArgMut<Elem = u8>,
    First: KernelSliceArg<Elem = K>,
    Second: KernelSliceArg<Elem = K>,
    Output: KernelSliceArgMut<Elem = K>,
    M: SetKernel<I, K, F, Op>,
{
    let total = first
        .len()
        .checked_add(second.len())
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if total == 0 {
        return Ok(());
    }
    let num_tiles = total.div_ceil(TILE_ITEMS);
    let (output_count, tile_state, partitions, virtual_shared) =
        set_parts::<I, _, _, _, _>(workspace);
    validate_workspace::<I, _, _, _, _>(
        output_count,
        tile_state,
        partitions,
        virtual_shared,
        num_tiles,
        SHARED,
    )?;
    module.launch_set_keys::<_, _, _, _, _, _, _, ITEMS, TILE_ITEMS, SHARED>(
        stream,
        first,
        second,
        output_count,
        tile_state,
        partitions,
        virtual_shared,
        output,
        order,
        num_tiles,
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_set_discard<
    I,
    K,
    M,
    F,
    Op,
    O,
    D,
    P,
    W,
    First,
    Second,
    const ITEMS: usize,
    const TILE_ITEMS: usize,
    const SHARED: usize,
>(
    module: &M,
    stream: &CudaStream,
    first: &First,
    second: &Second,
    workspace: &mut DeviceSetOperationWorkspace<O, D, P, W>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    I: SetIndex,
    I::Transaction: DeviceCopy,
    K: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: SerialSetOperation,
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<I::Transaction>,
    P: WorkspaceBuffer<I::Transaction>,
    W: WorkspaceBuffer<u8>,
    O::Range: KernelSliceArgMut<Elem = usize>,
    D::Range: KernelSliceArgMut<Elem = I::Transaction>,
    P::Range: KernelSliceArgMut<Elem = I::Transaction>,
    W::Range: KernelSliceArgMut<Elem = u8>,
    First: KernelSliceArg<Elem = K>,
    Second: KernelSliceArg<Elem = K>,
    M: SetKernel<I, K, F, Op>,
{
    let total = first
        .len()
        .checked_add(second.len())
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if total == 0 {
        return Ok(());
    }
    let num_tiles = total.div_ceil(TILE_ITEMS);
    let (output_count, tile_state, partitions, virtual_shared) =
        set_parts::<I, _, _, _, _>(workspace);
    validate_workspace::<I, _, _, _, _>(
        output_count,
        tile_state,
        partitions,
        virtual_shared,
        num_tiles,
        SHARED,
    )?;
    module.launch_set_discard::<_, _, _, _, _, _, ITEMS, TILE_ITEMS, SHARED>(
        stream,
        first,
        second,
        output_count,
        tile_state,
        partitions,
        virtual_shared,
        order,
        num_tiles,
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_set_arithmetic<
    I,
    K,
    M,
    F,
    Op,
    O,
    D,
    P,
    W,
    Output,
    const ITEMS: usize,
    const TILE_ITEMS: usize,
    const SHARED: usize,
>(
    module: &M,
    stream: &CudaStream,
    first: ArithmeticRange<K>,
    second: ArithmeticRange<K>,
    workspace: &mut DeviceSetOperationWorkspace<O, D, P, W>,
    output: &mut Output,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    I: SetIndex,
    I::Transaction: DeviceCopy,
    K: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: SerialSetOperation,
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<I::Transaction>,
    P: WorkspaceBuffer<I::Transaction>,
    W: WorkspaceBuffer<u8>,
    O::Range: KernelSliceArgMut<Elem = usize>,
    D::Range: KernelSliceArgMut<Elem = I::Transaction>,
    P::Range: KernelSliceArgMut<Elem = I::Transaction>,
    W::Range: KernelSliceArgMut<Elem = u8>,
    Output: KernelSliceArgMut<Elem = K>,
    M: ArithmeticSetKernel<I, K, F, Op>,
{
    let total = first
        .len()
        .checked_add(second.len())
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if total == 0 {
        return Ok(());
    }
    let num_tiles = total.div_ceil(TILE_ITEMS);
    let (output_count, tile_state, partitions, virtual_shared) =
        set_parts::<I, _, _, _, _>(workspace);
    validate_workspace::<I, _, _, _, _>(
        output_count,
        tile_state,
        partitions,
        virtual_shared,
        num_tiles,
        SHARED,
    )?;
    module.launch_set_arithmetic::<_, _, _, _, _, ITEMS, TILE_ITEMS, SHARED>(
        stream,
        *first.start(),
        *first.step(),
        first.len(),
        *second.start(),
        *second.step(),
        second.len(),
        output_count,
        tile_state,
        partitions,
        virtual_shared,
        output,
        order,
        num_tiles,
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_set_pairs<
    I,
    K,
    V,
    M,
    F,
    Op,
    O,
    D,
    P,
    W,
    FirstKeys,
    FirstValues,
    SecondKeys,
    SecondValues,
    OutputKeys,
    OutputValues,
    const ITEMS: usize,
    const TILE_ITEMS: usize,
    const SHARED: usize,
>(
    module: &M,
    stream: &CudaStream,
    first_keys: &FirstKeys,
    first_values: &FirstValues,
    second_keys: &SecondKeys,
    second_values: &SecondValues,
    workspace: &mut DeviceSetOperationWorkspace<O, D, P, W>,
    output_keys: &mut OutputKeys,
    output_values: &mut OutputValues,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    I: SetIndex,
    I::Transaction: DeviceCopy,
    K: DeviceCopy,
    V: DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: SerialSetOperation,
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<I::Transaction>,
    P: WorkspaceBuffer<I::Transaction>,
    W: WorkspaceBuffer<u8>,
    O::Range: KernelSliceArgMut<Elem = usize>,
    D::Range: KernelSliceArgMut<Elem = I::Transaction>,
    P::Range: KernelSliceArgMut<Elem = I::Transaction>,
    W::Range: KernelSliceArgMut<Elem = u8>,
    FirstKeys: KernelSliceArg<Elem = K>,
    FirstValues: KernelSliceArg<Elem = V>,
    SecondKeys: KernelSliceArg<Elem = K>,
    SecondValues: KernelSliceArg<Elem = V>,
    OutputKeys: KernelSliceArgMut<Elem = K>,
    OutputValues: KernelSliceArgMut<Elem = V>,
    M: SetKernel<I, K, F, Op>,
{
    let first_len = first_keys.len();
    let second_len = second_keys.len();
    let total = first_len
        .checked_add(second_len)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if total == 0 {
        return Ok(());
    }
    validate_len("first set values", first_len, first_values.len())?;
    if Op::READS_SECOND_VALUES {
        validate_len("second set values", second_len, second_values.len())?;
    }
    validate_len("set output values", output_keys.len(), output_values.len())?;
    let num_tiles = total.div_ceil(TILE_ITEMS);
    let (output_count, tile_state, partitions, virtual_shared) =
        set_parts::<I, _, _, _, _>(workspace);
    validate_workspace::<I, _, _, _, _>(
        output_count,
        tile_state,
        partitions,
        virtual_shared,
        num_tiles,
        SHARED,
    )?;
    module.launch_set_pairs::<V, _, _, _, _, _, _, _, _, _, _, ITEMS, TILE_ITEMS, SHARED>(
        stream,
        first_keys,
        first_values,
        second_keys,
        second_values,
        output_count,
        tile_state,
        partitions,
        virtual_shared,
        output_keys,
        output_values,
        order,
        num_tiles,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_policy_uses_concrete_rust_layout() {
        type Alias = u32;
        #[repr(C)]
        struct UserKey([u32; 3]);

        assert_eq!(set_items(core::mem::size_of::<Alias>()), 19);
        assert_eq!(set_items(core::mem::size_of::<UserKey>()), 7);
        assert_eq!(set_shared_bytes(19, 4, 4), 43_008);
    }
}
