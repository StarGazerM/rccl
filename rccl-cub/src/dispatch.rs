//! Ordinary Rust dispatch from numeric layouts to const-generic CUDA kernels.
//!
//! Rust cannot yet use `size_of::<T>()` directly as a stable const-generic
//! argument. The high-level API therefore performs one closed layout match and
//! calls a const-specialized helper. `T` remains the user's data type; kernel
//! policy and the compiled CUDA module never become API type parameters.

use core::{cmp::Ordering, mem::size_of, ops::DerefMut};

use cuda_core::{CudaStream, DeviceBuffer};
use rccl_memory::{TemporaryRegion, TemporaryValue};

use crate::{
    ArithmeticRange, DeviceAlgorithmError, DeviceMergeSortKeysWorkspace,
    DeviceMergeSortPairsWorkspace, DeviceMergeWorkspace, DeviceRadixSortBufferWorkspace,
    DeviceRadixSortKeysWorkspace, DeviceRadixSortPairsWorkspace, DeviceSetOperationWorkspace,
    DoubleBuffer, RadixKey, RadixValue,
    cuda::{KernelModules, Launch},
    merge::{
        device_merge_keys as launch_merge_keys, device_merge_pairs as launch_merge_pairs,
        merge_items, merge_load_to_shared_keys, merge_shared_bytes,
    },
    merge_sort::{
        device_merge_sort_keys as launch_merge_sort_keys,
        device_merge_sort_keys_copy as launch_merge_sort_keys_copy,
        device_merge_sort_pairs as launch_merge_sort_pairs,
        device_merge_sort_pairs_copy as launch_merge_sort_pairs_copy, merge_sort_items,
        merge_sort_shared_bytes,
    },
    radix::{
        device_radix_sort_keys as launch_radix_sort_keys,
        device_radix_sort_keys_buffer as launch_radix_sort_keys_buffer,
        device_radix_sort_pairs as launch_radix_sort_pairs,
        device_radix_sort_pairs_buffer as launch_radix_sort_pairs_buffer, radix_items,
        radix_shared_bytes, radix_single_items,
    },
    set_operation::{
        ArithmeticSetKernel, SetKernel, device_set_arithmetic, device_set_discard, device_set_keys,
        device_set_pairs, set_items, set_shared_bytes,
    },
    workspace::WorkspaceBuffer,
};

fn unsupported_layout() -> DeviceAlgorithmError {
    // `TemporaryValue` is sealed to the numeric primitives, whose layouts are
    // all covered below. Keep a defensive error instead of an unreachable
    // assertion so this remains sound if that sealed set grows.
    DeviceAlgorithmError::SizeOverflow
}

fn merge_sort_keys_for<K, F, S, P, Keys, const ITEMS: usize, const SHARED: usize>(
    modules: &KernelModules,
    stream: &CudaStream,
    keys: &mut Keys,
    workspace: &mut DeviceMergeSortKeysWorkspace<K, S, P>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    P: WorkspaceBuffer<usize>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    Keys: cuda_host::KernelSliceArgMut<Elem = K>,
{
    launch_merge_sort_keys::<K, _, _, _, _, _, ITEMS, SHARED>(
        modules.ordering(),
        stream,
        keys,
        workspace,
        order,
    )
}

fn merge_sort_keys_dispatch<K, F, S, P, Keys>(
    modules: &KernelModules,
    stream: &CudaStream,
    keys: &mut Keys,
    workspace: &mut DeviceMergeSortKeysWorkspace<K, S, P>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    P: WorkspaceBuffer<usize>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    Keys: cuda_host::KernelSliceArgMut<Elem = K>,
{
    match size_of::<K>() {
        1 => merge_sort_keys_for::<
            K,
            F,
            S,
            P,
            Keys,
            { merge_sort_items(1) },
            { merge_sort_shared_bytes(merge_sort_items(1), 1, 0) },
        >(modules, stream, keys, workspace, order),
        2 => merge_sort_keys_for::<
            K,
            F,
            S,
            P,
            Keys,
            { merge_sort_items(2) },
            { merge_sort_shared_bytes(merge_sort_items(2), 2, 0) },
        >(modules, stream, keys, workspace, order),
        4 => merge_sort_keys_for::<
            K,
            F,
            S,
            P,
            Keys,
            { merge_sort_items(4) },
            { merge_sort_shared_bytes(merge_sort_items(4), 4, 0) },
        >(modules, stream, keys, workspace, order),
        8 => merge_sort_keys_for::<
            K,
            F,
            S,
            P,
            Keys,
            { merge_sort_items(8) },
            { merge_sort_shared_bytes(merge_sort_items(8), 8, 0) },
        >(modules, stream, keys, workspace, order),
        16 => merge_sort_keys_for::<
            K,
            F,
            S,
            P,
            Keys,
            { merge_sort_items(16) },
            { merge_sort_shared_bytes(merge_sort_items(16), 16, 0) },
        >(modules, stream, keys, workspace, order),
        _ => Err(unsupported_layout()),
    }
}

/// Stable merge-sort using the policy compiled for cargo-oxide's target.
pub fn device_merge_sort_keys<K, F, S, P, Keys>(
    execution: &Launch<'_, '_>,
    keys: &mut Keys,
    workspace: &mut DeviceMergeSortKeysWorkspace<K, S, P>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    P: WorkspaceBuffer<usize>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    Keys: cuda_host::KernelSliceArgMut<Elem = K>,
{
    merge_sort_keys_dispatch(
        execution.module(),
        execution.stream(),
        keys,
        workspace,
        order,
    )
}

fn merge_sort_keys_copy_for<K, F, S, P, Input, Output, const ITEMS: usize, const SHARED: usize>(
    execution: &Launch<'_, '_>,
    input: &Input,
    workspace: &mut DeviceMergeSortKeysWorkspace<K, S, P>,
    output: &mut Output,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    P: WorkspaceBuffer<usize>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    Input: cuda_host::KernelSliceArg<Elem = K>,
    Output: cuda_host::KernelSliceArgMut<Elem = K>,
{
    launch_merge_sort_keys_copy::<K, _, _, _, _, _, _, ITEMS, SHARED>(
        execution.ordering_module(),
        execution.stream(),
        input,
        workspace,
        output,
        order,
    )
}

/// Stable out-of-place merge-sort using the compiled target policy.
pub fn device_merge_sort_keys_copy<K, F, S, P, Input, Output>(
    execution: &Launch<'_, '_>,
    input: &Input,
    workspace: &mut DeviceMergeSortKeysWorkspace<K, S, P>,
    output: &mut Output,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    P: WorkspaceBuffer<usize>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    Input: cuda_host::KernelSliceArg<Elem = K>,
    Output: cuda_host::KernelSliceArgMut<Elem = K>,
{
    match size_of::<K>() {
        1 => merge_sort_keys_copy_for::<
            K,
            F,
            S,
            P,
            Input,
            Output,
            { merge_sort_items(1) },
            { merge_sort_shared_bytes(merge_sort_items(1), 1, 0) },
        >(execution, input, workspace, output, order),
        2 => merge_sort_keys_copy_for::<
            K,
            F,
            S,
            P,
            Input,
            Output,
            { merge_sort_items(2) },
            { merge_sort_shared_bytes(merge_sort_items(2), 2, 0) },
        >(execution, input, workspace, output, order),
        4 => merge_sort_keys_copy_for::<
            K,
            F,
            S,
            P,
            Input,
            Output,
            { merge_sort_items(4) },
            { merge_sort_shared_bytes(merge_sort_items(4), 4, 0) },
        >(execution, input, workspace, output, order),
        8 => merge_sort_keys_copy_for::<
            K,
            F,
            S,
            P,
            Input,
            Output,
            { merge_sort_items(8) },
            { merge_sort_shared_bytes(merge_sort_items(8), 8, 0) },
        >(execution, input, workspace, output, order),
        16 => merge_sort_keys_copy_for::<
            K,
            F,
            S,
            P,
            Input,
            Output,
            { merge_sort_items(16) },
            { merge_sort_shared_bytes(merge_sort_items(16), 16, 0) },
        >(execution, input, workspace, output, order),
        _ => Err(unsupported_layout()),
    }
}

impl Launch<'_, '_> {
    #[doc(hidden)]
    pub fn merge_sort<K, F>(
        &self,
        keys: &mut DeviceBuffer<K>,
        scratch: &mut TemporaryRegion<'_, K>,
        partitions: &mut TemporaryRegion<'_, usize>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        K: TemporaryValue,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        let mut workspace = DeviceMergeSortKeysWorkspace::__from_parts(scratch, partitions);
        merge_sort_keys_dispatch(self.module(), self.stream(), keys, &mut workspace, order)
    }
}

impl Launch<'_, '_> {
    #[doc(hidden)]
    pub fn set_workspace_counts<K, V>(
        &self,
        first_len: usize,
        second_len: usize,
    ) -> Result<crate::SetOperationWorkspaceCounts, DeviceAlgorithmError> {
        crate::workspace::device_set_workspace_counts::<K, V>(self.stream(), first_len, second_len)
    }
}

#[allow(clippy::too_many_arguments)]
fn set_keys_for<
    I,
    K,
    F,
    Op,
    O,
    D,
    P,
    W,
    const ITEMS: usize,
    const TILE_ITEMS: usize,
    const SHARED: usize,
>(
    modules: &KernelModules,
    stream: &CudaStream,
    first: &DeviceBuffer<K>,
    second: &DeviceBuffer<K>,
    workspace: &mut DeviceSetOperationWorkspace<O, D, P, W>,
    output: Option<&mut DeviceBuffer<K>>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    I: crate::__private::SetIndex,
    I::Transaction: cuda_core::DeviceCopy,
    K: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: crate::__private::SerialSetOperation,
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<I::Transaction>,
    P: WorkspaceBuffer<I::Transaction>,
    W: WorkspaceBuffer<u8>,
    O::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    D::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
    W::Range: cuda_host::KernelSliceArgMut<Elem = u8>,
    crate::__private::SetModule: SetKernel<I, K, F, Op>,
{
    if let Some(output) = output {
        device_set_keys::<I, K, _, _, Op, _, _, _, _, _, _, _, ITEMS, TILE_ITEMS, SHARED>(
            modules.set(),
            stream,
            first,
            second,
            workspace,
            output,
            order,
        )
    } else {
        device_set_discard::<I, K, _, _, Op, _, _, _, _, _, _, ITEMS, TILE_ITEMS, SHARED>(
            modules.set(),
            stream,
            first,
            second,
            workspace,
            order,
        )
    }
}

impl Launch<'_, '_> {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_keys<I, K, F, Op, O, D, P, W>(
        &self,
        first: &DeviceBuffer<K>,
        second: &DeviceBuffer<K>,
        workspace: &mut DeviceSetOperationWorkspace<O, D, P, W>,
        output: Option<&mut DeviceBuffer<K>>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        I: crate::__private::SetIndex,
        I::Transaction: cuda_core::DeviceCopy,
        K: cuda_core::DeviceCopy,
        F: Fn(&K, &K) -> Ordering + Copy,
        Op: crate::__private::SerialSetOperation,
        O: WorkspaceBuffer<usize>,
        D: WorkspaceBuffer<I::Transaction>,
        P: WorkspaceBuffer<I::Transaction>,
        W: WorkspaceBuffer<u8>,
        O::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
        D::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        P::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        W::Range: cuda_host::KernelSliceArgMut<Elem = u8>,
        crate::__private::SetModule: SetKernel<I, K, F, Op>,
    {
        match size_of::<K>() {
            1 => set_keys_for::<
                I,
                K,
                F,
                Op,
                _,
                _,
                _,
                _,
                { set_items(1) },
                { 512 * set_items(1) - 1 },
                { set_shared_bytes(set_items(1), 1, 1) },
            >(
                self.module(),
                self.stream(),
                first,
                second,
                workspace,
                output,
                order,
            ),
            2 => set_keys_for::<
                I,
                K,
                F,
                Op,
                _,
                _,
                _,
                _,
                { set_items(2) },
                { 512 * set_items(2) - 1 },
                { set_shared_bytes(set_items(2), 2, 2) },
            >(
                self.module(),
                self.stream(),
                first,
                second,
                workspace,
                output,
                order,
            ),
            4 => set_keys_for::<
                I,
                K,
                F,
                Op,
                _,
                _,
                _,
                _,
                { set_items(4) },
                { 512 * set_items(4) - 1 },
                { set_shared_bytes(set_items(4), 4, 4) },
            >(
                self.module(),
                self.stream(),
                first,
                second,
                workspace,
                output,
                order,
            ),
            8 => set_keys_for::<
                I,
                K,
                F,
                Op,
                _,
                _,
                _,
                _,
                { set_items(8) },
                { 512 * set_items(8) - 1 },
                { set_shared_bytes(set_items(8), 8, 8) },
            >(
                self.module(),
                self.stream(),
                first,
                second,
                workspace,
                output,
                order,
            ),
            16 => set_keys_for::<
                I,
                K,
                F,
                Op,
                _,
                _,
                _,
                _,
                { set_items(16) },
                { 512 * set_items(16) - 1 },
                { set_shared_bytes(set_items(16), 16, 16) },
            >(
                self.module(),
                self.stream(),
                first,
                second,
                workspace,
                output,
                order,
            ),
            _ => Err(unsupported_layout()),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn set_arithmetic_for<
    I,
    K,
    F,
    Op,
    O,
    D,
    P,
    W,
    const ITEMS: usize,
    const TILE_ITEMS: usize,
    const SHARED: usize,
>(
    modules: &KernelModules,
    stream: &CudaStream,
    first: ArithmeticRange<K>,
    second: ArithmeticRange<K>,
    workspace: &mut DeviceSetOperationWorkspace<O, D, P, W>,
    output: &mut DeviceBuffer<K>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    I: crate::__private::SetIndex,
    I::Transaction: cuda_core::DeviceCopy,
    K: TemporaryValue + TryFrom<usize> + core::ops::Add<Output = K> + core::ops::Mul<Output = K>,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: crate::__private::SerialSetOperation,
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<I::Transaction>,
    P: WorkspaceBuffer<I::Transaction>,
    W: WorkspaceBuffer<u8>,
    O::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    D::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
    W::Range: cuda_host::KernelSliceArgMut<Elem = u8>,
    crate::__private::SetModule: ArithmeticSetKernel<I, K, F, Op>,
{
    device_set_arithmetic::<I, K, _, _, Op, _, _, _, _, _, ITEMS, TILE_ITEMS, SHARED>(
        modules.set(),
        stream,
        first,
        second,
        workspace,
        output,
        order,
    )
}

impl Launch<'_, '_> {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_arithmetic<I, K, F, Op, O, D, P, W>(
        &self,
        first: ArithmeticRange<K>,
        second: ArithmeticRange<K>,
        workspace: &mut DeviceSetOperationWorkspace<O, D, P, W>,
        output: &mut DeviceBuffer<K>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        I: crate::__private::SetIndex,
        I::Transaction: cuda_core::DeviceCopy,
        K: TemporaryValue
            + TryFrom<usize>
            + core::ops::Add<Output = K>
            + core::ops::Mul<Output = K>,
        F: Fn(&K, &K) -> Ordering + Copy,
        Op: crate::__private::SerialSetOperation,
        O: WorkspaceBuffer<usize>,
        D: WorkspaceBuffer<I::Transaction>,
        P: WorkspaceBuffer<I::Transaction>,
        W: WorkspaceBuffer<u8>,
        O::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
        D::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        P::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        W::Range: cuda_host::KernelSliceArgMut<Elem = u8>,
        crate::__private::SetModule: ArithmeticSetKernel<I, K, F, Op>,
    {
        match size_of::<K>() {
            1 => set_arithmetic_for::<
                I,
                K,
                F,
                Op,
                _,
                _,
                _,
                _,
                { set_items(1) },
                { 512 * set_items(1) - 1 },
                { set_shared_bytes(set_items(1), 1, 1) },
            >(
                self.module(),
                self.stream(),
                first,
                second,
                workspace,
                output,
                order,
            ),
            2 => set_arithmetic_for::<
                I,
                K,
                F,
                Op,
                _,
                _,
                _,
                _,
                { set_items(2) },
                { 512 * set_items(2) - 1 },
                { set_shared_bytes(set_items(2), 2, 2) },
            >(
                self.module(),
                self.stream(),
                first,
                second,
                workspace,
                output,
                order,
            ),
            4 => set_arithmetic_for::<
                I,
                K,
                F,
                Op,
                _,
                _,
                _,
                _,
                { set_items(4) },
                { 512 * set_items(4) - 1 },
                { set_shared_bytes(set_items(4), 4, 4) },
            >(
                self.module(),
                self.stream(),
                first,
                second,
                workspace,
                output,
                order,
            ),
            8 => set_arithmetic_for::<
                I,
                K,
                F,
                Op,
                _,
                _,
                _,
                _,
                { set_items(8) },
                { 512 * set_items(8) - 1 },
                { set_shared_bytes(set_items(8), 8, 8) },
            >(
                self.module(),
                self.stream(),
                first,
                second,
                workspace,
                output,
                order,
            ),
            16 => set_arithmetic_for::<
                I,
                K,
                F,
                Op,
                _,
                _,
                _,
                _,
                { set_items(16) },
                { 512 * set_items(16) - 1 },
                { set_shared_bytes(set_items(16), 16, 16) },
            >(
                self.module(),
                self.stream(),
                first,
                second,
                workspace,
                output,
                order,
            ),
            _ => Err(unsupported_layout()),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn set_pairs_for<
    I,
    K,
    V,
    F,
    Op,
    O,
    D,
    P,
    W,
    const ITEMS: usize,
    const TILE_ITEMS: usize,
    const SHARED: usize,
>(
    modules: &KernelModules,
    stream: &CudaStream,
    first_keys: &DeviceBuffer<K>,
    first_values: &DeviceBuffer<V>,
    second_keys: &DeviceBuffer<K>,
    second_values: &DeviceBuffer<V>,
    workspace: &mut DeviceSetOperationWorkspace<O, D, P, W>,
    output_keys: &mut DeviceBuffer<K>,
    output_values: &mut DeviceBuffer<V>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    I: crate::__private::SetIndex,
    I::Transaction: cuda_core::DeviceCopy,
    K: TemporaryValue,
    V: TemporaryValue,
    F: Fn(&K, &K) -> Ordering + Copy,
    Op: crate::__private::SerialSetOperation,
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<I::Transaction>,
    P: WorkspaceBuffer<I::Transaction>,
    W: WorkspaceBuffer<u8>,
    O::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    D::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
    W::Range: cuda_host::KernelSliceArgMut<Elem = u8>,
    crate::__private::SetModule: SetKernel<I, K, F, Op>,
{
    device_set_pairs::<I, K, V, _, _, Op, _, _, _, _, _, _, _, _, _, _, ITEMS, TILE_ITEMS, SHARED>(
        modules.set(),
        stream,
        first_keys,
        first_values,
        second_keys,
        second_values,
        workspace,
        output_keys,
        output_values,
        order,
    )
}

macro_rules! set_pair_value {
    ($self:expr, $stream:expr, $first_keys:expr, $first_values:expr, $second_keys:expr, $second_values:expr, $workspace:expr, $output_keys:expr, $output_values:expr, $order:expr, $key_size:literal, $value_size:literal) => {
        set_pairs_for::<
            I,
            K,
            V,
            F,
            Op,
            _,
            _,
            _,
            _,
            { set_items($key_size) },
            { 512 * set_items($key_size) - 1 },
            { set_shared_bytes(set_items($key_size), $key_size, $value_size) },
        >(
            $self,
            $stream,
            $first_keys,
            $first_values,
            $second_keys,
            $second_values,
            $workspace,
            $output_keys,
            $output_values,
            $order,
        )
    };
}

macro_rules! set_pair_key {
    ($self:expr, $stream:expr, $first_keys:expr, $first_values:expr, $second_keys:expr, $second_values:expr, $workspace:expr, $output_keys:expr, $output_values:expr, $order:expr, $key_size:literal) => {
        match size_of::<V>() {
            1 => set_pair_value!(
                $self,
                $stream,
                $first_keys,
                $first_values,
                $second_keys,
                $second_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                1
            ),
            2 => set_pair_value!(
                $self,
                $stream,
                $first_keys,
                $first_values,
                $second_keys,
                $second_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                2
            ),
            4 => set_pair_value!(
                $self,
                $stream,
                $first_keys,
                $first_values,
                $second_keys,
                $second_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                4
            ),
            8 => set_pair_value!(
                $self,
                $stream,
                $first_keys,
                $first_values,
                $second_keys,
                $second_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                8
            ),
            16 => set_pair_value!(
                $self,
                $stream,
                $first_keys,
                $first_values,
                $second_keys,
                $second_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                16
            ),
            _ => Err(unsupported_layout()),
        }
    };
}

impl Launch<'_, '_> {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_pairs<I, K, V, F, Op, O, D, P, W>(
        &self,
        first_keys: &DeviceBuffer<K>,
        first_values: &DeviceBuffer<V>,
        second_keys: &DeviceBuffer<K>,
        second_values: &DeviceBuffer<V>,
        workspace: &mut DeviceSetOperationWorkspace<O, D, P, W>,
        output_keys: &mut DeviceBuffer<K>,
        output_values: &mut DeviceBuffer<V>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        I: crate::__private::SetIndex,
        I::Transaction: cuda_core::DeviceCopy,
        K: TemporaryValue,
        V: TemporaryValue,
        F: Fn(&K, &K) -> Ordering + Copy,
        Op: crate::__private::SerialSetOperation,
        O: WorkspaceBuffer<usize>,
        D: WorkspaceBuffer<I::Transaction>,
        P: WorkspaceBuffer<I::Transaction>,
        W: WorkspaceBuffer<u8>,
        O::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
        D::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        P::Range: cuda_host::KernelSliceArgMut<Elem = I::Transaction>,
        W::Range: cuda_host::KernelSliceArgMut<Elem = u8>,
        crate::__private::SetModule: SetKernel<I, K, F, Op>,
    {
        match size_of::<K>() {
            1 => set_pair_key!(
                self.module(),
                self.stream(),
                first_keys,
                first_values,
                second_keys,
                second_values,
                workspace,
                output_keys,
                output_values,
                order,
                1
            ),
            2 => set_pair_key!(
                self.module(),
                self.stream(),
                first_keys,
                first_values,
                second_keys,
                second_values,
                workspace,
                output_keys,
                output_values,
                order,
                2
            ),
            4 => set_pair_key!(
                self.module(),
                self.stream(),
                first_keys,
                first_values,
                second_keys,
                second_values,
                workspace,
                output_keys,
                output_values,
                order,
                4
            ),
            8 => set_pair_key!(
                self.module(),
                self.stream(),
                first_keys,
                first_values,
                second_keys,
                second_values,
                workspace,
                output_keys,
                output_values,
                order,
                8
            ),
            16 => set_pair_key!(
                self.module(),
                self.stream(),
                first_keys,
                first_values,
                second_keys,
                second_values,
                workspace,
                output_keys,
                output_values,
                order,
                16
            ),
            _ => Err(unsupported_layout()),
        }
    }
}

fn merge_sort_pairs_for<K, V, F, S, U, P, Keys, Values, const ITEMS: usize, const SHARED: usize>(
    modules: &KernelModules,
    stream: &CudaStream,
    keys: &mut Keys,
    values: &mut Values,
    workspace: &mut DeviceMergeSortPairsWorkspace<K, V, S, U, P>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    V: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    P: WorkspaceBuffer<usize>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K>,
    U::Range: cuda_host::KernelSliceArgMut<Elem = V>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    Keys: cuda_host::KernelSliceArgMut<Elem = K>,
    Values: cuda_host::KernelSliceArgMut<Elem = V>,
{
    launch_merge_sort_pairs::<K, V, _, _, _, _, _, _, _, ITEMS, SHARED>(
        modules.ordering(),
        stream,
        keys,
        values,
        workspace,
        order,
    )
}

macro_rules! merge_sort_pair_value {
    ($self:expr, $stream:expr, $keys:expr, $values:expr, $workspace:expr, $order:expr, $key_size:literal, $value_size:literal) => {
        merge_sort_pairs_for::<
            K,
            V,
            F,
            _,
            _,
            _,
            _,
            _,
            { merge_sort_items($key_size) },
            { merge_sort_shared_bytes(merge_sort_items($key_size), $key_size, $value_size) },
        >($self, $stream, $keys, $values, $workspace, $order)
    };
}

macro_rules! merge_sort_pair_key {
    ($self:expr, $stream:expr, $keys:expr, $values:expr, $workspace:expr, $order:expr, $key_size:literal) => {
        match size_of::<V>() {
            1 => merge_sort_pair_value!(
                $self, $stream, $keys, $values, $workspace, $order, $key_size, 1
            ),
            2 => merge_sort_pair_value!(
                $self, $stream, $keys, $values, $workspace, $order, $key_size, 2
            ),
            4 => merge_sort_pair_value!(
                $self, $stream, $keys, $values, $workspace, $order, $key_size, 4
            ),
            8 => merge_sort_pair_value!(
                $self, $stream, $keys, $values, $workspace, $order, $key_size, 8
            ),
            16 => merge_sort_pair_value!(
                $self, $stream, $keys, $values, $workspace, $order, $key_size, 16
            ),
            _ => Err(unsupported_layout()),
        }
    };
}

#[allow(clippy::too_many_arguments)]
fn merge_sort_pairs_dispatch<K, V, F, S, U, P, Keys, Values>(
    modules: &KernelModules,
    stream: &CudaStream,
    keys: &mut Keys,
    values: &mut Values,
    workspace: &mut DeviceMergeSortPairsWorkspace<K, V, S, U, P>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    V: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    P: WorkspaceBuffer<usize>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K>,
    U::Range: cuda_host::KernelSliceArgMut<Elem = V>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    Keys: cuda_host::KernelSliceArgMut<Elem = K>,
    Values: cuda_host::KernelSliceArgMut<Elem = V>,
{
    match size_of::<K>() {
        1 => merge_sort_pair_key!(modules, stream, keys, values, workspace, order, 1),
        2 => merge_sort_pair_key!(modules, stream, keys, values, workspace, order, 2),
        4 => merge_sort_pair_key!(modules, stream, keys, values, workspace, order, 4),
        8 => merge_sort_pair_key!(modules, stream, keys, values, workspace, order, 8),
        16 => merge_sort_pair_key!(modules, stream, keys, values, workspace, order, 16),
        _ => Err(unsupported_layout()),
    }
}

/// Stable in-place merge-sort of key/value ranges.
#[allow(clippy::too_many_arguments)]
pub fn device_merge_sort_pairs<K, V, F, S, U, P, Keys, Values>(
    execution: &Launch<'_, '_>,
    keys: &mut Keys,
    values: &mut Values,
    workspace: &mut DeviceMergeSortPairsWorkspace<K, V, S, U, P>,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    V: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    P: WorkspaceBuffer<usize>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K>,
    U::Range: cuda_host::KernelSliceArgMut<Elem = V>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    Keys: cuda_host::KernelSliceArgMut<Elem = K>,
    Values: cuda_host::KernelSliceArgMut<Elem = V>,
{
    merge_sort_pairs_dispatch(
        execution.module(),
        execution.stream(),
        keys,
        values,
        workspace,
        order,
    )
}

#[allow(clippy::too_many_arguments)]
fn merge_sort_pairs_copy_for<
    K,
    V,
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
    execution: &Launch<'_, '_>,
    input_keys: &InputKeys,
    input_values: &InputValues,
    workspace: &mut DeviceMergeSortPairsWorkspace<K, V, S, U, P>,
    output_keys: &mut OutputKeys,
    output_values: &mut OutputValues,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    V: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    P: WorkspaceBuffer<usize>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K>,
    U::Range: cuda_host::KernelSliceArgMut<Elem = V>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    InputKeys: cuda_host::KernelSliceArg<Elem = K>,
    InputValues: cuda_host::KernelSliceArg<Elem = V>,
    OutputKeys: cuda_host::KernelSliceArgMut<Elem = K>,
    OutputValues: cuda_host::KernelSliceArgMut<Elem = V>,
{
    launch_merge_sort_pairs_copy::<K, V, _, _, _, _, _, _, _, _, _, ITEMS, SHARED>(
        execution.ordering_module(),
        execution.stream(),
        input_keys,
        input_values,
        workspace,
        output_keys,
        output_values,
        order,
    )
}

macro_rules! merge_sort_pair_copy_value {
    ($execution:expr, $input_keys:expr, $input_values:expr, $workspace:expr, $output_keys:expr, $output_values:expr, $order:expr, $key_size:literal, $value_size:literal) => {
        merge_sort_pairs_copy_for::<
            K,
            V,
            F,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            { merge_sort_items($key_size) },
            { merge_sort_shared_bytes(merge_sort_items($key_size), $key_size, $value_size) },
        >(
            $execution,
            $input_keys,
            $input_values,
            $workspace,
            $output_keys,
            $output_values,
            $order,
        )
    };
}

macro_rules! merge_sort_pair_copy_key {
    ($execution:expr, $input_keys:expr, $input_values:expr, $workspace:expr, $output_keys:expr, $output_values:expr, $order:expr, $key_size:literal) => {
        match size_of::<V>() {
            1 => merge_sort_pair_copy_value!(
                $execution,
                $input_keys,
                $input_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                1
            ),
            2 => merge_sort_pair_copy_value!(
                $execution,
                $input_keys,
                $input_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                2
            ),
            4 => merge_sort_pair_copy_value!(
                $execution,
                $input_keys,
                $input_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                4
            ),
            8 => merge_sort_pair_copy_value!(
                $execution,
                $input_keys,
                $input_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                8
            ),
            16 => merge_sort_pair_copy_value!(
                $execution,
                $input_keys,
                $input_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                16
            ),
            _ => Err(unsupported_layout()),
        }
    };
}

/// Stable out-of-place merge-sort of key/value ranges.
#[allow(clippy::too_many_arguments)]
pub fn device_merge_sort_pairs_copy<
    K,
    V,
    F,
    S,
    U,
    P,
    InputKeys,
    InputValues,
    OutputKeys,
    OutputValues,
>(
    execution: &Launch<'_, '_>,
    input_keys: &InputKeys,
    input_values: &InputValues,
    workspace: &mut DeviceMergeSortPairsWorkspace<K, V, S, U, P>,
    output_keys: &mut OutputKeys,
    output_values: &mut OutputValues,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    V: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    P: WorkspaceBuffer<usize>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K>,
    U::Range: cuda_host::KernelSliceArgMut<Elem = V>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    InputKeys: cuda_host::KernelSliceArg<Elem = K>,
    InputValues: cuda_host::KernelSliceArg<Elem = V>,
    OutputKeys: cuda_host::KernelSliceArgMut<Elem = K>,
    OutputValues: cuda_host::KernelSliceArgMut<Elem = V>,
{
    match size_of::<K>() {
        1 => merge_sort_pair_copy_key!(
            execution,
            input_keys,
            input_values,
            workspace,
            output_keys,
            output_values,
            order,
            1
        ),
        2 => merge_sort_pair_copy_key!(
            execution,
            input_keys,
            input_values,
            workspace,
            output_keys,
            output_values,
            order,
            2
        ),
        4 => merge_sort_pair_copy_key!(
            execution,
            input_keys,
            input_values,
            workspace,
            output_keys,
            output_values,
            order,
            4
        ),
        8 => merge_sort_pair_copy_key!(
            execution,
            input_keys,
            input_values,
            workspace,
            output_keys,
            output_values,
            order,
            8
        ),
        16 => merge_sort_pair_copy_key!(
            execution,
            input_keys,
            input_values,
            workspace,
            output_keys,
            output_values,
            order,
            16
        ),
        _ => Err(unsupported_layout()),
    }
}

impl Launch<'_, '_> {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn merge_sort_pairs<K, V, F>(
        &self,
        keys: &mut DeviceBuffer<K>,
        values: &mut DeviceBuffer<V>,
        scratch_keys: &mut TemporaryRegion<'_, K>,
        scratch_values: &mut TemporaryRegion<'_, V>,
        partitions: &mut TemporaryRegion<'_, usize>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        K: TemporaryValue,
        V: TemporaryValue,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        let mut workspace =
            DeviceMergeSortPairsWorkspace::__from_parts(scratch_keys, scratch_values, partitions);
        merge_sort_pairs_dispatch(
            self.module(),
            self.stream(),
            keys,
            values,
            &mut workspace,
            order,
        )
    }
}

fn merge_keys_for<
    K,
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
    modules: &KernelModules,
    stream: &CudaStream,
    first: &First,
    second: &Second,
    workspace: &mut DeviceMergeWorkspace<P>,
    output: &mut Output,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    P: WorkspaceBuffer<usize>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    First: cuda_host::KernelSliceArg<Elem = K>,
    Second: cuda_host::KernelSliceArg<Elem = K>,
    Output: cuda_host::KernelSliceArgMut<Elem = K>,
{
    launch_merge_keys::<K, _, _, _, _, _, _, ITEMS, K_SIZE, LOAD, SHARED>(
        modules.ordering(),
        stream,
        first,
        second,
        workspace,
        output,
        order,
    )
}

#[allow(clippy::too_many_arguments)]
fn merge_keys_dispatch<K, F, P, First, Second, Output>(
    modules: &KernelModules,
    stream: &CudaStream,
    first: &First,
    second: &Second,
    workspace: &mut DeviceMergeWorkspace<P>,
    output: &mut Output,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    P: WorkspaceBuffer<usize>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    First: cuda_host::KernelSliceArg<Elem = K>,
    Second: cuda_host::KernelSliceArg<Elem = K>,
    Output: cuda_host::KernelSliceArgMut<Elem = K>,
{
    match size_of::<K>() {
        1 => merge_keys_for::<
            K,
            F,
            P,
            First,
            Second,
            Output,
            { merge_items(1) },
            1,
            { merge_load_to_shared_keys(1, 0) },
            { merge_shared_bytes(merge_items(1), 1, 0, merge_load_to_shared_keys(1, 0), false) },
        >(modules, stream, first, second, workspace, output, order),
        2 => merge_keys_for::<
            K,
            F,
            P,
            First,
            Second,
            Output,
            { merge_items(2) },
            2,
            { merge_load_to_shared_keys(2, 0) },
            { merge_shared_bytes(merge_items(2), 2, 0, merge_load_to_shared_keys(2, 0), false) },
        >(modules, stream, first, second, workspace, output, order),
        4 => merge_keys_for::<
            K,
            F,
            P,
            First,
            Second,
            Output,
            { merge_items(4) },
            4,
            { merge_load_to_shared_keys(4, 0) },
            { merge_shared_bytes(merge_items(4), 4, 0, merge_load_to_shared_keys(4, 0), false) },
        >(modules, stream, first, second, workspace, output, order),
        8 => merge_keys_for::<
            K,
            F,
            P,
            First,
            Second,
            Output,
            { merge_items(8) },
            8,
            { merge_load_to_shared_keys(8, 0) },
            { merge_shared_bytes(merge_items(8), 8, 0, merge_load_to_shared_keys(8, 0), false) },
        >(modules, stream, first, second, workspace, output, order),
        16 => merge_keys_for::<
            K,
            F,
            P,
            First,
            Second,
            Output,
            { merge_items(16) },
            16,
            { merge_load_to_shared_keys(16, 0) },
            {
                merge_shared_bytes(
                    merge_items(16),
                    16,
                    0,
                    merge_load_to_shared_keys(16, 0),
                    false,
                )
            },
        >(modules, stream, first, second, workspace, output, order),
        _ => Err(unsupported_layout()),
    }
}

/// Merge two sorted key ranges.
#[allow(clippy::too_many_arguments)]
pub fn device_merge_keys<K, F, P, First, Second, Output>(
    execution: &Launch<'_, '_>,
    first: &First,
    second: &Second,
    workspace: &mut DeviceMergeWorkspace<P>,
    output: &mut Output,
    order: F,
) -> Result<(), DeviceAlgorithmError>
where
    K: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    P: WorkspaceBuffer<usize>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    First: cuda_host::KernelSliceArg<Elem = K>,
    Second: cuda_host::KernelSliceArg<Elem = K>,
    Output: cuda_host::KernelSliceArgMut<Elem = K>,
{
    merge_keys_dispatch(
        execution.module(),
        execution.stream(),
        first,
        second,
        workspace,
        output,
        order,
    )
}

impl Launch<'_, '_> {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn merge<K, F>(
        &self,
        first: &DeviceBuffer<K>,
        second: &DeviceBuffer<K>,
        partitions: &mut TemporaryRegion<'_, usize>,
        output: &mut DeviceBuffer<K>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        K: TemporaryValue,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        let mut workspace = DeviceMergeWorkspace::__from_partitions(partitions);
        merge_keys_dispatch(
            self.module(),
            self.stream(),
            first,
            second,
            &mut workspace,
            output,
            order,
        )
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "private helper preserves the paired CUB input/output ranges"
)]
fn merge_pairs_for<
    K,
    V,
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
    const LOAD: bool,
    const SHARED: usize,
>(
    modules: &KernelModules,
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
    K: cuda_core::DeviceCopy,
    V: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    P: WorkspaceBuffer<usize>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    FirstKeys: cuda_host::KernelSliceArg<Elem = K>,
    FirstValues: cuda_host::KernelSliceArg<Elem = V>,
    SecondKeys: cuda_host::KernelSliceArg<Elem = K>,
    SecondValues: cuda_host::KernelSliceArg<Elem = V>,
    OutputKeys: cuda_host::KernelSliceArgMut<Elem = K>,
    OutputValues: cuda_host::KernelSliceArgMut<Elem = V>,
{
    launch_merge_pairs::<K, V, _, _, _, _, _, _, _, _, _, ITEMS, K_SIZE, V_SIZE, LOAD, LOAD, SHARED>(
        modules.ordering(),
        stream,
        first_keys,
        first_values,
        second_keys,
        second_values,
        workspace,
        output_keys,
        output_values,
        order,
    )
}

macro_rules! merge_pair_value {
    ($self:expr, $stream:expr, $first_keys:expr, $first_values:expr, $second_keys:expr, $second_values:expr, $workspace:expr, $output_keys:expr, $output_values:expr, $order:expr, $key_size:literal, $value_size:literal) => {
        merge_pairs_for::<
            K,
            V,
            F,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            { merge_items($key_size + $value_size) },
            $key_size,
            $value_size,
            { merge_load_to_shared_keys($key_size, $value_size) },
            {
                merge_shared_bytes(
                    merge_items($key_size + $value_size),
                    $key_size,
                    $value_size,
                    merge_load_to_shared_keys($key_size, $value_size),
                    merge_load_to_shared_keys($key_size, $value_size),
                )
            },
        >(
            $self,
            $stream,
            $first_keys,
            $first_values,
            $second_keys,
            $second_values,
            $workspace,
            $output_keys,
            $output_values,
            $order,
        )
    };
}

macro_rules! merge_pair_key {
    ($self:expr, $stream:expr, $first_keys:expr, $first_values:expr, $second_keys:expr, $second_values:expr, $workspace:expr, $output_keys:expr, $output_values:expr, $order:expr, $key_size:literal) => {
        match size_of::<V>() {
            1 => merge_pair_value!(
                $self,
                $stream,
                $first_keys,
                $first_values,
                $second_keys,
                $second_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                1
            ),
            2 => merge_pair_value!(
                $self,
                $stream,
                $first_keys,
                $first_values,
                $second_keys,
                $second_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                2
            ),
            4 => merge_pair_value!(
                $self,
                $stream,
                $first_keys,
                $first_values,
                $second_keys,
                $second_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                4
            ),
            8 => merge_pair_value!(
                $self,
                $stream,
                $first_keys,
                $first_values,
                $second_keys,
                $second_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                8
            ),
            16 => merge_pair_value!(
                $self,
                $stream,
                $first_keys,
                $first_values,
                $second_keys,
                $second_values,
                $workspace,
                $output_keys,
                $output_values,
                $order,
                $key_size,
                16
            ),
            _ => Err(unsupported_layout()),
        }
    };
}

#[allow(clippy::too_many_arguments)]
fn merge_pairs_dispatch<
    K,
    V,
    F,
    P,
    FirstKeys,
    FirstValues,
    SecondKeys,
    SecondValues,
    OutputKeys,
    OutputValues,
>(
    modules: &KernelModules,
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
    K: cuda_core::DeviceCopy,
    V: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    P: WorkspaceBuffer<usize>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    FirstKeys: cuda_host::KernelSliceArg<Elem = K>,
    FirstValues: cuda_host::KernelSliceArg<Elem = V>,
    SecondKeys: cuda_host::KernelSliceArg<Elem = K>,
    SecondValues: cuda_host::KernelSliceArg<Elem = V>,
    OutputKeys: cuda_host::KernelSliceArgMut<Elem = K>,
    OutputValues: cuda_host::KernelSliceArgMut<Elem = V>,
{
    match size_of::<K>() {
        1 => merge_pair_key!(
            modules,
            stream,
            first_keys,
            first_values,
            second_keys,
            second_values,
            workspace,
            output_keys,
            output_values,
            order,
            1
        ),
        2 => merge_pair_key!(
            modules,
            stream,
            first_keys,
            first_values,
            second_keys,
            second_values,
            workspace,
            output_keys,
            output_values,
            order,
            2
        ),
        4 => merge_pair_key!(
            modules,
            stream,
            first_keys,
            first_values,
            second_keys,
            second_values,
            workspace,
            output_keys,
            output_values,
            order,
            4
        ),
        8 => merge_pair_key!(
            modules,
            stream,
            first_keys,
            first_values,
            second_keys,
            second_values,
            workspace,
            output_keys,
            output_values,
            order,
            8
        ),
        16 => merge_pair_key!(
            modules,
            stream,
            first_keys,
            first_values,
            second_keys,
            second_values,
            workspace,
            output_keys,
            output_values,
            order,
            16
        ),
        _ => Err(unsupported_layout()),
    }
}

/// Merge two sorted key/value ranges.
#[allow(clippy::too_many_arguments)]
pub fn device_merge_pairs<
    K,
    V,
    F,
    P,
    FirstKeys,
    FirstValues,
    SecondKeys,
    SecondValues,
    OutputKeys,
    OutputValues,
>(
    execution: &Launch<'_, '_>,
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
    K: cuda_core::DeviceCopy,
    V: cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> Ordering + Copy,
    P: WorkspaceBuffer<usize>,
    P::Range: cuda_host::KernelSliceArgMut<Elem = usize>,
    FirstKeys: cuda_host::KernelSliceArg<Elem = K>,
    FirstValues: cuda_host::KernelSliceArg<Elem = V>,
    SecondKeys: cuda_host::KernelSliceArg<Elem = K>,
    SecondValues: cuda_host::KernelSliceArg<Elem = V>,
    OutputKeys: cuda_host::KernelSliceArgMut<Elem = K>,
    OutputValues: cuda_host::KernelSliceArgMut<Elem = V>,
{
    merge_pairs_dispatch(
        execution.module(),
        execution.stream(),
        first_keys,
        first_values,
        second_keys,
        second_values,
        workspace,
        output_keys,
        output_values,
        order,
    )
}

impl Launch<'_, '_> {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn merge_pairs<K, V, F>(
        &self,
        first_keys: &DeviceBuffer<K>,
        first_values: &DeviceBuffer<V>,
        second_keys: &DeviceBuffer<K>,
        second_values: &DeviceBuffer<V>,
        partitions: &mut TemporaryRegion<'_, usize>,
        output_keys: &mut DeviceBuffer<K>,
        output_values: &mut DeviceBuffer<V>,
        order: F,
    ) -> Result<(), DeviceAlgorithmError>
    where
        K: TemporaryValue,
        V: TemporaryValue,
        F: Fn(&K, &K) -> Ordering + Copy,
    {
        let mut workspace = DeviceMergeWorkspace::__from_partitions(partitions);
        merge_pairs_dispatch(
            self.module(),
            self.stream(),
            first_keys,
            first_values,
            second_keys,
            second_values,
            &mut workspace,
            output_keys,
            output_values,
            order,
        )
    }
}

fn radix_keys_for<
    K,
    S,
    B,
    L,
    C,
    Keys,
    const SINGLE_ITEMS: usize,
    const ITEMS: usize,
    const UNION: usize,
    const SHARED: usize,
    const PASSES: usize,
    const DESC: bool,
>(
    modules: &KernelModules,
    stream: &CudaStream,
    keys: &mut Keys,
    workspace: &mut DeviceRadixSortKeysWorkspace<K, S, B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    Keys: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRangeMut<K>,
    S: WorkspaceBuffer<K>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRange<K>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    launch_radix_sort_keys::<K, _, _, _, _, _, _, SINGLE_ITEMS, ITEMS, UNION, SHARED, PASSES, DESC>(
        modules.ordering(),
        stream,
        keys,
        workspace,
    )
}

fn radix_keys_dispatch<K, S, B, L, C, Keys, const DESC: bool>(
    modules: &KernelModules,
    stream: &CudaStream,
    keys: &mut Keys,
    workspace: &mut DeviceRadixSortKeysWorkspace<K, S, B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    Keys: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRangeMut<K>,
    S: WorkspaceBuffer<K>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRange<K>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    crate::radix::ensure_radix_target()?;
    match size_of::<K>() {
        4 => radix_keys_for::<
            K,
            S,
            B,
            L,
            C,
            Keys,
            { radix_single_items(4) },
            { radix_items(21, 4) },
            { 384 * radix_items(21, 4) * 4 },
            { radix_shared_bytes(radix_items(21, 4), 4) },
            4,
            DESC,
        >(modules, stream, keys, workspace),
        8 => radix_keys_for::<
            K,
            S,
            B,
            L,
            C,
            Keys,
            { radix_single_items(8) },
            { radix_items(21, 8) },
            { 384 * radix_items(21, 8) * 8 },
            { radix_shared_bytes(radix_items(21, 8), 8) },
            8,
            DESC,
        >(modules, stream, keys, workspace),
        _ => Err(unsupported_layout()),
    }
}

/// Sort keys in ascending radix order.
pub fn device_radix_sort_keys<K, S, B, L, C, Keys>(
    execution: &Launch<'_, '_>,
    keys: &mut Keys,
    workspace: &mut DeviceRadixSortKeysWorkspace<K, S, B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    Keys: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRangeMut<K>,
    S: WorkspaceBuffer<K>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRange<K>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    radix_keys_dispatch::<K, S, B, L, C, Keys, false>(
        execution.module(),
        execution.stream(),
        keys,
        workspace,
    )
}

/// Sort keys in descending radix order.
pub fn device_radix_sort_keys_descending<K, S, B, L, C, Keys>(
    execution: &Launch<'_, '_>,
    keys: &mut Keys,
    workspace: &mut DeviceRadixSortKeysWorkspace<K, S, B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    Keys: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRangeMut<K>,
    S: WorkspaceBuffer<K>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRange<K>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    radix_keys_dispatch::<K, S, B, L, C, Keys, true>(
        execution.module(),
        execution.stream(),
        keys,
        workspace,
    )
}

fn radix_keys_buffer_for<
    K,
    First,
    Second,
    B,
    L,
    C,
    const SINGLE_ITEMS: usize,
    const ITEMS: usize,
    const UNION: usize,
    const SHARED: usize,
    const PASSES: usize,
    const DESC: bool,
>(
    execution: &Launch<'_, '_>,
    buffers: &mut DoubleBuffer<First, Second>,
    bit_range: core::ops::Range<u32>,
    workspace: &mut DeviceRadixSortBufferWorkspace<B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    First: DerefMut,
    Second: DerefMut,
    First::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    Second::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    launch_radix_sort_keys_buffer::<
        K,
        _,
        _,
        _,
        _,
        _,
        _,
        SINGLE_ITEMS,
        ITEMS,
        UNION,
        SHARED,
        PASSES,
        DESC,
    >(
        execution.ordering_module(),
        execution.stream(),
        buffers,
        bit_range,
        workspace,
    )
}

fn radix_keys_buffer_dispatch<K, First, Second, B, L, C, const DESC: bool>(
    execution: &Launch<'_, '_>,
    buffers: &mut DoubleBuffer<First, Second>,
    bit_range: core::ops::Range<u32>,
    workspace: &mut DeviceRadixSortBufferWorkspace<B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    First: DerefMut,
    Second: DerefMut,
    First::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    Second::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    crate::radix::ensure_radix_target()?;
    match size_of::<K>() {
        4 => radix_keys_buffer_for::<
            K,
            First,
            Second,
            B,
            L,
            C,
            { radix_single_items(4) },
            { radix_items(21, 4) },
            { 384 * radix_items(21, 4) * 4 },
            { radix_shared_bytes(radix_items(21, 4), 4) },
            4,
            DESC,
        >(execution, buffers, bit_range, workspace),
        8 => radix_keys_buffer_for::<
            K,
            First,
            Second,
            B,
            L,
            C,
            { radix_single_items(8) },
            { radix_items(21, 8) },
            { 384 * radix_items(21, 8) * 8 },
            { radix_shared_bytes(radix_items(21, 8), 8) },
            8,
            DESC,
        >(execution, buffers, bit_range, workspace),
        _ => Err(unsupported_layout()),
    }
}

/// Sort a selected key bit range using caller-owned double buffers.
pub fn device_radix_sort_keys_buffer<K, First, Second, B, L, C>(
    execution: &Launch<'_, '_>,
    buffers: &mut DoubleBuffer<First, Second>,
    bit_range: core::ops::Range<u32>,
    workspace: &mut DeviceRadixSortBufferWorkspace<B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    First: DerefMut,
    Second: DerefMut,
    First::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    Second::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    radix_keys_buffer_dispatch::<K, First, Second, B, L, C, false>(
        execution, buffers, bit_range, workspace,
    )
}

/// Sort a selected key bit range in descending order using double buffers.
pub fn device_radix_sort_keys_buffer_descending<K, First, Second, B, L, C>(
    execution: &Launch<'_, '_>,
    buffers: &mut DoubleBuffer<First, Second>,
    bit_range: core::ops::Range<u32>,
    workspace: &mut DeviceRadixSortBufferWorkspace<B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    First: DerefMut,
    Second: DerefMut,
    First::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    Second::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    radix_keys_buffer_dispatch::<K, First, Second, B, L, C, true>(
        execution, buffers, bit_range, workspace,
    )
}

impl Launch<'_, '_> {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn radix_sort<K>(
        &self,
        keys: &mut DeviceBuffer<K>,
        scratch: &mut TemporaryRegion<'_, K>,
        bins: &mut TemporaryRegion<'_, u64>,
        lookback: &mut TemporaryRegion<'_, i32>,
        counters: &mut TemporaryRegion<'_, i32>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        K: RadixKey + TemporaryValue,
    {
        let mut workspace = DeviceRadixSortKeysWorkspace::<K, _, _, _, _>::__from_parts(
            scratch, bins, lookback, counters,
        );
        radix_keys_dispatch::<K, _, _, _, _, _, false>(
            self.module(),
            self.stream(),
            keys,
            &mut workspace,
        )
    }
}

fn radix_pairs_for<
    K,
    V,
    S,
    U,
    B,
    L,
    C,
    Keys,
    Values,
    const SINGLE_ITEMS: usize,
    const ITEMS: usize,
    const UNION: usize,
    const SHARED: usize,
    const PASSES: usize,
    const DESC: bool,
>(
    modules: &KernelModules,
    stream: &CudaStream,
    keys: &mut Keys,
    values: &mut Values,
    workspace: &mut DeviceRadixSortPairsWorkspace<K, V, S, U, B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    V: RadixValue,
    Keys: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRangeMut<K>,
    Values: cuda_host::KernelSliceArgMut<Elem = V> + rccl_memory::ContiguousDeviceRangeMut<V>,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRange<K>,
    U::Range: cuda_host::KernelSliceArgMut<Elem = V> + rccl_memory::ContiguousDeviceRange<V>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    launch_radix_sort_pairs::<
        K,
        V,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        SINGLE_ITEMS,
        ITEMS,
        UNION,
        SHARED,
        PASSES,
        DESC,
    >(modules.ordering(), stream, keys, values, workspace)
}

macro_rules! radix_pair {
    ($self:expr, $stream:expr, $keys:expr, $values:expr, $workspace:expr, $key_size:literal, $value_size:literal, $nominal:literal, $descending:ident) => {
        radix_pairs_for::<
            K,
            V,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            {
                radix_single_items(if $key_size > $value_size {
                    $key_size
                } else {
                    $value_size
                })
            },
            {
                radix_items(
                    $nominal,
                    if $key_size > $value_size {
                        $key_size
                    } else {
                        $value_size
                    },
                )
            },
            {
                384 * radix_items(
                    $nominal,
                    if $key_size > $value_size {
                        $key_size
                    } else {
                        $value_size
                    },
                ) * (if $key_size > $value_size {
                    $key_size
                } else {
                    $value_size
                })
            },
            {
                radix_shared_bytes(
                    radix_items(
                        $nominal,
                        if $key_size > $value_size {
                            $key_size
                        } else {
                            $value_size
                        },
                    ),
                    if $key_size > $value_size {
                        $key_size
                    } else {
                        $value_size
                    },
                )
            },
            $key_size,
            $descending,
        >($self, $stream, $keys, $values, $workspace)
    };
}

macro_rules! radix_pair_key {
    ($self:expr, $stream:expr, $keys:expr, $values:expr, $workspace:expr, $key_size:literal, $nominal:literal, $descending:ident) => {
        match size_of::<V>() {
            1 => radix_pair!(
                $self,
                $stream,
                $keys,
                $values,
                $workspace,
                $key_size,
                1,
                $nominal,
                $descending
            ),
            2 => radix_pair!(
                $self,
                $stream,
                $keys,
                $values,
                $workspace,
                $key_size,
                2,
                $nominal,
                $descending
            ),
            4 => radix_pair!(
                $self,
                $stream,
                $keys,
                $values,
                $workspace,
                $key_size,
                4,
                $nominal,
                $descending
            ),
            8 => radix_pair!(
                $self,
                $stream,
                $keys,
                $values,
                $workspace,
                $key_size,
                8,
                $nominal,
                $descending
            ),
            16 => radix_pair!(
                $self,
                $stream,
                $keys,
                $values,
                $workspace,
                $key_size,
                16,
                $nominal,
                $descending
            ),
            _ => Err(unsupported_layout()),
        }
    };
}

fn radix_pairs_dispatch<K, V, S, U, B, L, C, Keys, Values, const DESC: bool>(
    modules: &KernelModules,
    stream: &CudaStream,
    keys: &mut Keys,
    values: &mut Values,
    workspace: &mut DeviceRadixSortPairsWorkspace<K, V, S, U, B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    V: RadixValue,
    Keys: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRangeMut<K>,
    Values: cuda_host::KernelSliceArgMut<Elem = V> + rccl_memory::ContiguousDeviceRangeMut<V>,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRange<K>,
    U::Range: cuda_host::KernelSliceArgMut<Elem = V> + rccl_memory::ContiguousDeviceRange<V>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    crate::radix::ensure_radix_target()?;
    match size_of::<K>() {
        4 => radix_pair_key!(modules, stream, keys, values, workspace, 4, 17, DESC),
        8 => radix_pair_key!(modules, stream, keys, values, workspace, 8, 21, DESC),
        _ => Err(unsupported_layout()),
    }
}

/// Sort key/value ranges in ascending radix order.
pub fn device_radix_sort_pairs<K, V, S, U, B, L, C, Keys, Values>(
    execution: &Launch<'_, '_>,
    keys: &mut Keys,
    values: &mut Values,
    workspace: &mut DeviceRadixSortPairsWorkspace<K, V, S, U, B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    V: RadixValue,
    Keys: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRangeMut<K>,
    Values: cuda_host::KernelSliceArgMut<Elem = V> + rccl_memory::ContiguousDeviceRangeMut<V>,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRange<K>,
    U::Range: cuda_host::KernelSliceArgMut<Elem = V> + rccl_memory::ContiguousDeviceRange<V>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    radix_pairs_dispatch::<K, V, S, U, B, L, C, Keys, Values, false>(
        execution.module(),
        execution.stream(),
        keys,
        values,
        workspace,
    )
}

/// Sort key/value ranges in descending radix order.
pub fn device_radix_sort_pairs_descending<K, V, S, U, B, L, C, Keys, Values>(
    execution: &Launch<'_, '_>,
    keys: &mut Keys,
    values: &mut Values,
    workspace: &mut DeviceRadixSortPairsWorkspace<K, V, S, U, B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    V: RadixValue,
    Keys: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRangeMut<K>,
    Values: cuda_host::KernelSliceArgMut<Elem = V> + rccl_memory::ContiguousDeviceRangeMut<V>,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    S::Range: cuda_host::KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRange<K>,
    U::Range: cuda_host::KernelSliceArgMut<Elem = V> + rccl_memory::ContiguousDeviceRange<V>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    radix_pairs_dispatch::<K, V, S, U, B, L, C, Keys, Values, true>(
        execution.module(),
        execution.stream(),
        keys,
        values,
        workspace,
    )
}

#[allow(clippy::too_many_arguments)]
fn radix_pairs_buffer_for<
    K,
    V,
    FirstKeys,
    FirstValues,
    SecondKeys,
    SecondValues,
    B,
    L,
    C,
    const SINGLE_ITEMS: usize,
    const ITEMS: usize,
    const UNION: usize,
    const SHARED: usize,
    const PASSES: usize,
    const DESC: bool,
>(
    execution: &Launch<'_, '_>,
    buffers: &mut DoubleBuffer<(FirstKeys, FirstValues), (SecondKeys, SecondValues)>,
    bit_range: core::ops::Range<u32>,
    workspace: &mut DeviceRadixSortBufferWorkspace<B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    V: RadixValue,
    FirstKeys: DerefMut,
    FirstValues: DerefMut,
    SecondKeys: DerefMut,
    SecondValues: DerefMut,
    FirstKeys::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    FirstValues::Target: cuda_host::KernelSliceArgMut<Elem = V> + Sized,
    SecondKeys::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    SecondValues::Target: cuda_host::KernelSliceArgMut<Elem = V> + Sized,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    launch_radix_sort_pairs_buffer::<
        K,
        V,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        SINGLE_ITEMS,
        ITEMS,
        UNION,
        SHARED,
        PASSES,
        DESC,
    >(
        execution.ordering_module(),
        execution.stream(),
        buffers,
        bit_range,
        workspace,
    )
}

macro_rules! radix_pair_buffer_value {
    ($execution:expr, $buffers:expr, $bit_range:expr, $workspace:expr, $key_size:literal, $value_size:literal, $nominal:literal, $descending:ident) => {
        radix_pairs_buffer_for::<
            K,
            V,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            {
                radix_single_items(if $key_size > $value_size {
                    $key_size
                } else {
                    $value_size
                })
            },
            {
                radix_items(
                    $nominal,
                    if $key_size > $value_size {
                        $key_size
                    } else {
                        $value_size
                    },
                )
            },
            {
                384 * radix_items(
                    $nominal,
                    if $key_size > $value_size {
                        $key_size
                    } else {
                        $value_size
                    },
                ) * (if $key_size > $value_size {
                    $key_size
                } else {
                    $value_size
                })
            },
            {
                radix_shared_bytes(
                    radix_items(
                        $nominal,
                        if $key_size > $value_size {
                            $key_size
                        } else {
                            $value_size
                        },
                    ),
                    if $key_size > $value_size {
                        $key_size
                    } else {
                        $value_size
                    },
                )
            },
            $key_size,
            $descending,
        >($execution, $buffers, $bit_range, $workspace)
    };
}

macro_rules! radix_pair_buffer_key {
    ($execution:expr, $buffers:expr, $bit_range:expr, $workspace:expr, $key_size:literal, $nominal:literal, $descending:ident) => {
        match size_of::<V>() {
            1 => radix_pair_buffer_value!(
                $execution,
                $buffers,
                $bit_range,
                $workspace,
                $key_size,
                1,
                $nominal,
                $descending
            ),
            2 => radix_pair_buffer_value!(
                $execution,
                $buffers,
                $bit_range,
                $workspace,
                $key_size,
                2,
                $nominal,
                $descending
            ),
            4 => radix_pair_buffer_value!(
                $execution,
                $buffers,
                $bit_range,
                $workspace,
                $key_size,
                4,
                $nominal,
                $descending
            ),
            8 => radix_pair_buffer_value!(
                $execution,
                $buffers,
                $bit_range,
                $workspace,
                $key_size,
                8,
                $nominal,
                $descending
            ),
            16 => radix_pair_buffer_value!(
                $execution,
                $buffers,
                $bit_range,
                $workspace,
                $key_size,
                16,
                $nominal,
                $descending
            ),
            _ => Err(unsupported_layout()),
        }
    };
}

#[allow(clippy::too_many_arguments)]
fn radix_pairs_buffer_dispatch<
    K,
    V,
    FirstKeys,
    FirstValues,
    SecondKeys,
    SecondValues,
    B,
    L,
    C,
    const DESC: bool,
>(
    execution: &Launch<'_, '_>,
    buffers: &mut DoubleBuffer<(FirstKeys, FirstValues), (SecondKeys, SecondValues)>,
    bit_range: core::ops::Range<u32>,
    workspace: &mut DeviceRadixSortBufferWorkspace<B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    V: RadixValue,
    FirstKeys: DerefMut,
    FirstValues: DerefMut,
    SecondKeys: DerefMut,
    SecondValues: DerefMut,
    FirstKeys::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    FirstValues::Target: cuda_host::KernelSliceArgMut<Elem = V> + Sized,
    SecondKeys::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    SecondValues::Target: cuda_host::KernelSliceArgMut<Elem = V> + Sized,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    crate::radix::ensure_radix_target()?;
    match size_of::<K>() {
        4 => radix_pair_buffer_key!(execution, buffers, bit_range, workspace, 4, 17, DESC),
        8 => radix_pair_buffer_key!(execution, buffers, bit_range, workspace, 8, 21, DESC),
        _ => Err(unsupported_layout()),
    }
}

/// Sort a selected key bit range and carry values using double buffers.
#[allow(clippy::too_many_arguments)]
pub fn device_radix_sort_pairs_buffer<
    K,
    V,
    FirstKeys,
    FirstValues,
    SecondKeys,
    SecondValues,
    B,
    L,
    C,
>(
    execution: &Launch<'_, '_>,
    buffers: &mut DoubleBuffer<(FirstKeys, FirstValues), (SecondKeys, SecondValues)>,
    bit_range: core::ops::Range<u32>,
    workspace: &mut DeviceRadixSortBufferWorkspace<B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    V: RadixValue,
    FirstKeys: DerefMut,
    FirstValues: DerefMut,
    SecondKeys: DerefMut,
    SecondValues: DerefMut,
    FirstKeys::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    FirstValues::Target: cuda_host::KernelSliceArgMut<Elem = V> + Sized,
    SecondKeys::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    SecondValues::Target: cuda_host::KernelSliceArgMut<Elem = V> + Sized,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    radix_pairs_buffer_dispatch::<
        K,
        V,
        FirstKeys,
        FirstValues,
        SecondKeys,
        SecondValues,
        B,
        L,
        C,
        false,
    >(execution, buffers, bit_range, workspace)
}

/// Descending selected-bit pair sort using caller-owned double buffers.
#[allow(clippy::too_many_arguments)]
pub fn device_radix_sort_pairs_buffer_descending<
    K,
    V,
    FirstKeys,
    FirstValues,
    SecondKeys,
    SecondValues,
    B,
    L,
    C,
>(
    execution: &Launch<'_, '_>,
    buffers: &mut DoubleBuffer<(FirstKeys, FirstValues), (SecondKeys, SecondValues)>,
    bit_range: core::ops::Range<u32>,
    workspace: &mut DeviceRadixSortBufferWorkspace<B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    V: RadixValue,
    FirstKeys: DerefMut,
    FirstValues: DerefMut,
    SecondKeys: DerefMut,
    SecondValues: DerefMut,
    FirstKeys::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    FirstValues::Target: cuda_host::KernelSliceArgMut<Elem = V> + Sized,
    SecondKeys::Target: cuda_host::KernelSliceArgMut<Elem = K> + Sized,
    SecondValues::Target: cuda_host::KernelSliceArgMut<Elem = V> + Sized,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: cuda_host::KernelSliceArgMut<Elem = u64>,
    L::Range: cuda_host::KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: cuda_host::KernelSliceArgMut<Elem = i32>,
{
    radix_pairs_buffer_dispatch::<
        K,
        V,
        FirstKeys,
        FirstValues,
        SecondKeys,
        SecondValues,
        B,
        L,
        C,
        true,
    >(execution, buffers, bit_range, workspace)
}

impl Launch<'_, '_> {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn radix_sort_pairs<K, V>(
        &self,
        keys: &mut DeviceBuffer<K>,
        values: &mut DeviceBuffer<V>,
        scratch_keys: &mut TemporaryRegion<'_, K>,
        scratch_values: &mut TemporaryRegion<'_, V>,
        bins: &mut TemporaryRegion<'_, u64>,
        lookback: &mut TemporaryRegion<'_, i32>,
        counters: &mut TemporaryRegion<'_, i32>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        K: RadixKey + TemporaryValue,
        V: RadixValue + TemporaryValue,
    {
        let mut workspace = DeviceRadixSortPairsWorkspace::<K, V, _, _, _, _, _>::__from_parts(
            scratch_keys,
            scratch_values,
            bins,
            lookback,
            counters,
        );
        radix_pairs_dispatch::<K, V, _, _, _, _, _, _, _, false>(
            self.module(),
            self.stream(),
            keys,
            values,
            &mut workspace,
        )
    }
}
