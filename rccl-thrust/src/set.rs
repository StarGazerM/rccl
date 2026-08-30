use core::cmp::Ordering;

use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_cub::__private::{
    SerialSetOperation, SetDifference, SetIntersection, SetSymmetricDifference, SetUnion,
};
use rccl_memory::{DeviceAllocator, TemporaryValue};

use crate::{
    DeviceAlgorithmError,
    backend::{ArithmeticSetBackend, SetBackend, SetPairsBackend},
    cuda::Execution,
    range::{ArithmeticRange, Discard, KeyValueRange, KeyValueRangeMut},
};

#[inline]
fn natural_order<K: PartialOrd>(left: &K, right: &K) -> Ordering {
    if left < right {
        Ordering::Less
    } else if right < left {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn set_keys_by<K, A, M, F, Op>(
    execution: &Execution<'_, '_, A, M>,
    first: &DeviceBuffer<K>,
    second: &DeviceBuffer<K>,
    mut output: Option<&mut DeviceBuffer<K>>,
    order: F,
) -> Result<usize, DeviceAlgorithmError>
where
    K: DeviceCopy + 'static,
    A: DeviceAllocator,
    M: SetBackend<K>,
    F: Fn(&K, &K) -> Ordering + Copy + 'static,
    Op: SerialSetOperation,
{
    let total = first
        .len()
        .checked_add(second.len())
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if total == 0 {
        return Ok(0);
    }
    let counts = execution
        .module()
        .set_workspace_counts(execution, first.len(), second.len())?;
    if counts.wide_indices {
        let mut temporary =
            crate::memory::TemporaryWorkspace5::<_, usize, u128, u8, u128, u8>::new_in_with_extra_tail(
                execution.allocator(),
                execution.stream(),
                [
                    1,
                    counts.tile_descriptors,
                    1,
                    counts.partitions,
                    counts.virtual_shared_bytes,
                ],
                [256; 5],
                256,
            )
            .map_err(DeviceAlgorithmError::Memory)?;
        let (output_count, descriptors, _tile_alias_tail, partitions, virtual_shared) =
            temporary.parts_mut();
        let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
            &mut *output_count,
            &mut *descriptors,
            &mut *partitions,
            &mut *virtual_shared,
        );
        let result = if let Some(output) = output.as_deref_mut() {
            execution.module().set_i64::<_, _, Op>(
                execution,
                first,
                second,
                &mut workspace,
                output,
                order,
            )
        } else {
            execution.module().set_discard_i64::<_, _, Op>(
                execution,
                first,
                second,
                &mut workspace,
                order,
            )
        };
        drop(workspace);
        result?;
        output_count
            .first_to_host()
            .map_err(DeviceAlgorithmError::Memory)
    } else {
        let mut temporary =
            crate::memory::TemporaryWorkspace5::<_, usize, u64, u8, u64, u8>::new_in_with_extra_tail(
                execution.allocator(),
                execution.stream(),
                [
                    1,
                    counts.tile_descriptors,
                    1,
                    counts.partitions,
                    counts.virtual_shared_bytes,
                ],
                [256; 5],
                256,
            )
            .map_err(DeviceAlgorithmError::Memory)?;
        let (output_count, descriptors, _tile_alias_tail, partitions, virtual_shared) =
            temporary.parts_mut();
        let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
            &mut *output_count,
            &mut *descriptors,
            &mut *partitions,
            &mut *virtual_shared,
        );
        let result = if let Some(output) = output.as_deref_mut() {
            execution.module().set_i32::<_, _, Op>(
                execution,
                first,
                second,
                &mut workspace,
                output,
                order,
            )
        } else {
            execution.module().set_discard_i32::<_, _, Op>(
                execution,
                first,
                second,
                &mut workspace,
                order,
            )
        };
        drop(workspace);
        result?;
        output_count
            .first_to_host()
            .map_err(DeviceAlgorithmError::Memory)
    }
}

fn set_arithmetic_by<K, A, M, F, Op>(
    execution: &Execution<'_, '_, A, M>,
    first: ArithmeticRange<K>,
    second: ArithmeticRange<K>,
    output: &mut DeviceBuffer<K>,
    order: F,
) -> Result<usize, DeviceAlgorithmError>
where
    K: TemporaryValue
        + TryFrom<usize>
        + core::ops::Add<Output = K>
        + core::ops::Mul<Output = K>
        + 'static,
    A: DeviceAllocator,
    M: SetBackend<K> + ArithmeticSetBackend<K>,
    F: Fn(&K, &K) -> Ordering + Copy + 'static,
    Op: SerialSetOperation,
{
    let first_len = rccl_cub::DeviceRange::len(&first);
    let second_len = rccl_cub::DeviceRange::len(&second);
    let total = first_len
        .checked_add(second_len)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if total == 0 {
        return Ok(0);
    }
    let counts = execution
        .module()
        .set_workspace_counts(execution, first_len, second_len)?;
    if counts.wide_indices {
        let mut temporary =
            crate::memory::TemporaryWorkspace5::<_, usize, u128, u8, u128, u8>::new_in_with_extra_tail(
                execution.allocator(),
                execution.stream(),
                [
                    1,
                    counts.tile_descriptors,
                    1,
                    counts.partitions,
                    counts.virtual_shared_bytes,
                ],
                [256; 5],
                256,
            )
            .map_err(DeviceAlgorithmError::Memory)?;
        let (output_count, descriptors, _tile_alias_tail, partitions, virtual_shared) =
            temporary.parts_mut();
        let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
            &mut *output_count,
            &mut *descriptors,
            &mut *partitions,
            &mut *virtual_shared,
        );
        let result = execution.module().set_arithmetic_i64::<_, _, Op>(
            execution,
            first,
            second,
            &mut workspace,
            output,
            order,
        );
        drop(workspace);
        result?;
        output_count
            .first_to_host()
            .map_err(DeviceAlgorithmError::Memory)
    } else {
        let mut temporary =
            crate::memory::TemporaryWorkspace5::<_, usize, u64, u8, u64, u8>::new_in_with_extra_tail(
                execution.allocator(),
                execution.stream(),
                [
                    1,
                    counts.tile_descriptors,
                    1,
                    counts.partitions,
                    counts.virtual_shared_bytes,
                ],
                [256; 5],
                256,
            )
            .map_err(DeviceAlgorithmError::Memory)?;
        let (output_count, descriptors, _tile_alias_tail, partitions, virtual_shared) =
            temporary.parts_mut();
        let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
            &mut *output_count,
            &mut *descriptors,
            &mut *partitions,
            &mut *virtual_shared,
        );
        let result = execution.module().set_arithmetic_i32::<_, _, Op>(
            execution,
            first,
            second,
            &mut workspace,
            output,
            order,
        );
        drop(workspace);
        result?;
        output_count
            .first_to_host()
            .map_err(DeviceAlgorithmError::Memory)
    }
}

#[allow(clippy::too_many_arguments)]
fn set_pairs_by<K, V, A, M, F, Op>(
    execution: &Execution<'_, '_, A, M>,
    first_keys: &DeviceBuffer<K>,
    first_values: &DeviceBuffer<V>,
    second_keys: &DeviceBuffer<K>,
    second_values: &DeviceBuffer<V>,
    output_keys: &mut DeviceBuffer<K>,
    output_values: &mut DeviceBuffer<V>,
    order: F,
) -> Result<usize, DeviceAlgorithmError>
where
    K: TemporaryValue + 'static,
    V: TemporaryValue,
    A: DeviceAllocator,
    M: SetPairsBackend<K, V>,
    F: Fn(&K, &K) -> Ordering + Copy + 'static,
    Op: SerialSetOperation,
{
    if first_values.len() != first_keys.len() {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "first set values",
            expected: first_keys.len(),
            actual: first_values.len(),
        });
    }
    if Op::READS_SECOND_VALUES && second_values.len() != second_keys.len() {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "second set values",
            expected: second_keys.len(),
            actual: second_values.len(),
        });
    }
    if output_values.len() != output_keys.len() {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "set output values",
            expected: output_keys.len(),
            actual: output_values.len(),
        });
    }
    let total = first_keys
        .len()
        .checked_add(second_keys.len())
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if total == 0 {
        return Ok(0);
    }
    let counts = execution.module().set_pairs_workspace_counts(
        execution,
        first_keys.len(),
        second_keys.len(),
    )?;
    if counts.wide_indices {
        let mut temporary =
            crate::memory::TemporaryWorkspace5::<_, usize, u128, u8, u128, u8>::new_in_with_extra_tail(
                execution.allocator(),
                execution.stream(),
                [
                    1,
                    counts.tile_descriptors,
                    1,
                    counts.partitions,
                    counts.virtual_shared_bytes,
                ],
                [256; 5],
                256,
            )
            .map_err(DeviceAlgorithmError::Memory)?;
        let (output_count, descriptors, _tile_alias_tail, partitions, virtual_shared) =
            temporary.parts_mut();
        let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
            &mut *output_count,
            &mut *descriptors,
            &mut *partitions,
            &mut *virtual_shared,
        );
        let result = execution.module().set_pairs_i64::<_, _, Op>(
            execution,
            first_keys,
            first_values,
            second_keys,
            second_values,
            &mut workspace,
            output_keys,
            output_values,
            order,
        );
        drop(workspace);
        result?;
        output_count
            .first_to_host()
            .map_err(DeviceAlgorithmError::Memory)
    } else {
        let mut temporary =
            crate::memory::TemporaryWorkspace5::<_, usize, u64, u8, u64, u8>::new_in_with_extra_tail(
                execution.allocator(),
                execution.stream(),
                [
                    1,
                    counts.tile_descriptors,
                    1,
                    counts.partitions,
                    counts.virtual_shared_bytes,
                ],
                [256; 5],
                256,
            )
            .map_err(DeviceAlgorithmError::Memory)?;
        let (output_count, descriptors, _tile_alias_tail, partitions, virtual_shared) =
            temporary.parts_mut();
        let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
            &mut *output_count,
            &mut *descriptors,
            &mut *partitions,
            &mut *virtual_shared,
        );
        let result = execution.module().set_pairs_i32::<_, _, Op>(
            execution,
            first_keys,
            first_values,
            second_keys,
            second_values,
            &mut workspace,
            output_keys,
            output_values,
            order,
        );
        drop(workspace);
        result?;
        output_count
            .first_to_host()
            .map_err(DeviceAlgorithmError::Memory)
    }
}

macro_rules! key_set_trait {
    ($trait_name:ident, $method:ident, $method_by:ident, $operation:ty) => {
        pub trait $trait_name<First, Second, Output> {
            type Key;

            fn $method(
                &self,
                first: First,
                second: Second,
                output: Output,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                Self::Key: PartialOrd;

            fn $method_by<F>(
                &self,
                first: First,
                second: Second,
                output: Output,
                order: F,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                F: Fn(&Self::Key, &Self::Key) -> Ordering + Copy + 'static;
        }

        impl<'a, 'b, 'c, K, A, M>
            $trait_name<&'a DeviceBuffer<K>, &'b DeviceBuffer<K>, &'c mut DeviceBuffer<K>>
            for Execution<'_, '_, A, M>
        where
            K: DeviceCopy + 'static,
            A: DeviceAllocator,
            M: SetBackend<K>,
        {
            type Key = K;

            fn $method(
                &self,
                first: &'a DeviceBuffer<K>,
                second: &'b DeviceBuffer<K>,
                output: &'c mut DeviceBuffer<K>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                K: PartialOrd,
            {
                set_keys_by::<K, A, M, _, $operation>(
                    self,
                    first,
                    second,
                    Some(output),
                    natural_order,
                )
            }

            fn $method_by<F>(
                &self,
                first: &'a DeviceBuffer<K>,
                second: &'b DeviceBuffer<K>,
                output: &'c mut DeviceBuffer<K>,
                order: F,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                F: Fn(&K, &K) -> Ordering + Copy + 'static,
            {
                set_keys_by::<K, A, M, F, $operation>(self, first, second, Some(output), order)
            }
        }

        impl<'a, 'b, K, A, M> $trait_name<&'a DeviceBuffer<K>, &'b DeviceBuffer<K>, Discard<K>>
            for Execution<'_, '_, A, M>
        where
            K: DeviceCopy + 'static,
            A: DeviceAllocator,
            M: SetBackend<K>,
        {
            type Key = K;

            fn $method(
                &self,
                first: &'a DeviceBuffer<K>,
                second: &'b DeviceBuffer<K>,
                output: Discard<K>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                K: PartialOrd,
            {
                self.$method_by(first, second, output, natural_order)
            }

            fn $method_by<F>(
                &self,
                first: &'a DeviceBuffer<K>,
                second: &'b DeviceBuffer<K>,
                _output: Discard<K>,
                order: F,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                F: Fn(&K, &K) -> Ordering + Copy + 'static,
            {
                set_keys_by::<K, A, M, F, $operation>(self, first, second, None, order)
            }
        }

        impl<'output, K, A, M>
            $trait_name<ArithmeticRange<K>, ArithmeticRange<K>, &'output mut DeviceBuffer<K>>
            for Execution<'_, '_, A, M>
        where
            K: TemporaryValue
                + TryFrom<usize>
                + core::ops::Add<Output = K>
                + core::ops::Mul<Output = K>
                + 'static,
            A: DeviceAllocator,
            M: SetBackend<K> + ArithmeticSetBackend<K>,
        {
            type Key = K;

            fn $method(
                &self,
                first: ArithmeticRange<K>,
                second: ArithmeticRange<K>,
                output: &'output mut DeviceBuffer<K>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                K: PartialOrd,
            {
                self.$method_by(first, second, output, natural_order)
            }

            fn $method_by<F>(
                &self,
                first: ArithmeticRange<K>,
                second: ArithmeticRange<K>,
                output: &'output mut DeviceBuffer<K>,
                order: F,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                F: Fn(&K, &K) -> Ordering + Copy + 'static,
            {
                set_arithmetic_by::<K, A, M, F, $operation>(self, first, second, output, order)
            }
        }
    };
}

key_set_trait!(DeviceSetUnionExt, union_into, union_by_into, SetUnion);
key_set_trait!(
    DeviceSetIntersectionExt,
    intersection_into,
    intersection_by_into,
    SetIntersection
);
key_set_trait!(
    DeviceSetDifferenceExt,
    difference_into,
    difference_by_into,
    SetDifference
);
key_set_trait!(
    DeviceSetSymmetricDifferenceExt,
    symmetric_difference_into,
    symmetric_difference_by_into,
    SetSymmetricDifference
);

macro_rules! pair_set_impl {
    ($trait_name:ident, $method:ident, $method_by:ident, $operation:ty) => {
        impl<'a, 'b, 'view, 'borrow, K, V, A, M>
            $trait_name<
                KeyValueRange<'a, K, V>,
                KeyValueRange<'b, K, V>,
                &'borrow mut KeyValueRangeMut<'view, K, V>,
            > for Execution<'_, '_, A, M>
        where
            K: TemporaryValue + 'static,
            V: TemporaryValue,
            A: DeviceAllocator,
            M: SetPairsBackend<K, V>,
        {
            type Key = K;

            fn $method(
                &self,
                first: KeyValueRange<'a, K, V>,
                second: KeyValueRange<'b, K, V>,
                output: &'borrow mut KeyValueRangeMut<'view, K, V>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                K: PartialOrd,
            {
                self.$method_by(first, second, output, natural_order)
            }

            fn $method_by<F>(
                &self,
                first: KeyValueRange<'a, K, V>,
                second: KeyValueRange<'b, K, V>,
                output: &'borrow mut KeyValueRangeMut<'view, K, V>,
                order: F,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                F: Fn(&K, &K) -> Ordering + Copy + 'static,
            {
                let (first_keys, first_values) = first.parts();
                let (second_keys, second_values) = second.parts();
                let (output_keys, output_values) = output.parts_mut();
                set_pairs_by::<K, V, A, M, F, $operation>(
                    self,
                    first_keys,
                    first_values,
                    second_keys,
                    second_values,
                    output_keys,
                    output_values,
                    order,
                )
            }
        }
    };
}

pair_set_impl!(DeviceSetUnionExt, union_into, union_by_into, SetUnion);
pair_set_impl!(
    DeviceSetDifferenceExt,
    difference_into,
    difference_by_into,
    SetDifference
);
pair_set_impl!(
    DeviceSetSymmetricDifferenceExt,
    symmetric_difference_into,
    symmetric_difference_by_into,
    SetSymmetricDifference
);

impl<'a, 'b, 'view, 'borrow, K, V, A, M>
    DeviceSetIntersectionExt<
        KeyValueRange<'a, K, V>,
        &'b DeviceBuffer<K>,
        &'borrow mut KeyValueRangeMut<'view, K, V>,
    > for Execution<'_, '_, A, M>
where
    K: TemporaryValue + 'static,
    V: TemporaryValue,
    A: DeviceAllocator,
    M: SetPairsBackend<K, V>,
{
    type Key = K;

    fn intersection_into(
        &self,
        first: KeyValueRange<'a, K, V>,
        second: &'b DeviceBuffer<K>,
        output: &'borrow mut KeyValueRangeMut<'view, K, V>,
    ) -> Result<usize, DeviceAlgorithmError>
    where
        K: PartialOrd,
    {
        self.intersection_by_into(first, second, output, natural_order)
    }

    fn intersection_by_into<F>(
        &self,
        first: KeyValueRange<'a, K, V>,
        second: &'b DeviceBuffer<K>,
        output: &'borrow mut KeyValueRangeMut<'view, K, V>,
        order: F,
    ) -> Result<usize, DeviceAlgorithmError>
    where
        F: Fn(&K, &K) -> Ordering + Copy + 'static,
    {
        let (first_keys, first_values) = first.parts();
        let (output_keys, output_values) = output.parts_mut();
        set_pairs_by::<K, V, A, M, F, SetIntersection>(
            self,
            first_keys,
            first_values,
            second,
            first_values,
            output_keys,
            output_values,
            order,
        )
    }
}
