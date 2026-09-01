use core::cmp::Ordering;

use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_cub::__private::{
    SerialSetOperation, SetDifference as CubSetDifference, SetIntersection as CubSetIntersection,
    SetSymmetricDifference as CubSetSymmetricDifference, SetUnion as CubSetUnion,
};
use rccl_memory::{DeviceAllocator, TemporaryValue};

use crate::{
    DeviceAlgorithmError,
    backend::{ArithmeticSetBackend, SetBackend, SetPairsBackend},
    cuda::Execution,
    range::{ArithmeticRange, DeviceView, DeviceViewMut, KeyValueRange, KeyValueRangeMut},
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
    output: Option<&mut DeviceBuffer<K>>,
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
        let result = {
            let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
                &mut *output_count,
                &mut *descriptors,
                &mut *partitions,
                &mut *virtual_shared,
            );
            if let Some(output) = output {
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
            }
        };
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
        let result = {
            let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
                &mut *output_count,
                &mut *descriptors,
                &mut *partitions,
                &mut *virtual_shared,
            );
            if let Some(output) = output {
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
            }
        };
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
        let result = {
            let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
                &mut *output_count,
                &mut *descriptors,
                &mut *partitions,
                &mut *virtual_shared,
            );
            execution.module().set_arithmetic_i64::<_, _, Op>(
                execution,
                first,
                second,
                &mut workspace,
                output,
                order,
            )
        };
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
        let result = {
            let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
                &mut *output_count,
                &mut *descriptors,
                &mut *partitions,
                &mut *virtual_shared,
            );
            execution.module().set_arithmetic_i32::<_, _, Op>(
                execution,
                first,
                second,
                &mut workspace,
                output,
                order,
            )
        };
        result?;
        output_count
            .first_to_host()
            .map_err(DeviceAlgorithmError::Memory)
    }
}

fn set_pairs_by<K, V, A, M, F, Op>(
    execution: &Execution<'_, '_, A, M>,
    first: KeyValueRange<'_, K, V>,
    second: (&DeviceBuffer<K>, &DeviceBuffer<V>),
    mut output: KeyValueRangeMut<'_, K, V>,
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
    let (first_keys, first_values) = first.parts();
    let (second_keys, second_values) = second;
    let (output_keys, output_values) = output.parts_mut();
    if Op::READS_SECOND_VALUES && second_values.len() != second_keys.len() {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "second set values",
            expected: second_keys.len(),
            actual: second_values.len(),
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
        let result = {
            let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
                &mut *output_count,
                &mut *descriptors,
                &mut *partitions,
                &mut *virtual_shared,
            );
            execution.module().set_pairs_i64::<_, _, Op>(
                execution,
                first_keys,
                first_values,
                second_keys,
                second_values,
                &mut workspace,
                output_keys,
                output_values,
                order,
            )
        };
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
        let result = {
            let mut workspace = rccl_cub::DeviceSetOperationWorkspace::__from_parts(
                &mut *output_count,
                &mut *descriptors,
                &mut *partitions,
                &mut *virtual_shared,
            );
            execution.module().set_pairs_i32::<_, _, Op>(
                execution,
                first_keys,
                first_values,
                second_keys,
                second_values,
                &mut workspace,
                output_keys,
                output_values,
                order,
            )
        };
        result?;
        output_count
            .first_to_host()
            .map_err(DeviceAlgorithmError::Memory)
    }
}

macro_rules! define_key_set_expression {
    ($expression:ident, $expression_by:ident, $method:ident, $method_by:ident, $operation:ty) => {
        /// A lazy set operation over two sorted device sequences.
        pub struct $expression<First, Second> {
            first: First,
            second: Second,
        }

        /// A lazy set operation carrying a Rust ordering closure.
        pub struct $expression_by<First, Second, F> {
            first: First,
            second: Second,
            order: F,
        }

        impl<'a, K: DeviceCopy> DeviceView<'a, K> {
            pub fn $method<'b>(
                self,
                second: DeviceView<'b, K>,
            ) -> $expression<Self, DeviceView<'b, K>> {
                $expression {
                    first: self,
                    second,
                }
            }

            pub fn $method_by<'b, F>(
                self,
                second: DeviceView<'b, K>,
                order: F,
            ) -> $expression_by<Self, DeviceView<'b, K>, F>
            where
                F: Fn(&K, &K) -> Ordering + Copy + 'static,
            {
                $expression_by {
                    first: self,
                    second,
                    order,
                }
            }
        }

        impl<'a, 'b, K> $expression<DeviceView<'a, K>, DeviceView<'b, K>>
        where
            K: DeviceCopy + PartialOrd + 'static,
        {
            pub fn write_into<A, M>(
                self,
                output: DeviceViewMut<'_, K>,
                execution: &Execution<'_, '_, A, M>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                A: DeviceAllocator,
                M: SetBackend<K>,
            {
                set_keys_by::<K, A, M, _, $operation>(
                    execution,
                    self.first.into_buffer(),
                    self.second.into_buffer(),
                    Some(output.into_buffer()),
                    natural_order,
                )
            }

            /// Execute the operation without materializing its keys.
            pub fn count<A, M>(
                self,
                execution: &Execution<'_, '_, A, M>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                A: DeviceAllocator,
                M: SetBackend<K>,
            {
                set_keys_by::<K, A, M, _, $operation>(
                    execution,
                    self.first.into_buffer(),
                    self.second.into_buffer(),
                    None,
                    natural_order,
                )
            }
        }

        impl<'a, 'b, K, F> $expression_by<DeviceView<'a, K>, DeviceView<'b, K>, F>
        where
            K: DeviceCopy + 'static,
            F: Fn(&K, &K) -> Ordering + Copy + 'static,
        {
            pub fn write_into<A, M>(
                self,
                output: DeviceViewMut<'_, K>,
                execution: &Execution<'_, '_, A, M>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                A: DeviceAllocator,
                M: SetBackend<K>,
            {
                set_keys_by::<K, A, M, F, $operation>(
                    execution,
                    self.first.into_buffer(),
                    self.second.into_buffer(),
                    Some(output.into_buffer()),
                    self.order,
                )
            }

            pub fn count<A, M>(
                self,
                execution: &Execution<'_, '_, A, M>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                A: DeviceAllocator,
                M: SetBackend<K>,
            {
                set_keys_by::<K, A, M, F, $operation>(
                    execution,
                    self.first.into_buffer(),
                    self.second.into_buffer(),
                    None,
                    self.order,
                )
            }
        }

        impl<K> $expression<ArithmeticRange<K>, ArithmeticRange<K>>
        where
            K: TemporaryValue
                + PartialOrd
                + TryFrom<usize>
                + core::ops::Add<Output = K>
                + core::ops::Mul<Output = K>
                + 'static,
        {
            pub fn write_into<A, M>(
                self,
                output: DeviceViewMut<'_, K>,
                execution: &Execution<'_, '_, A, M>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                A: DeviceAllocator,
                M: SetBackend<K> + ArithmeticSetBackend<K>,
            {
                set_arithmetic_by::<K, A, M, _, $operation>(
                    execution,
                    self.first,
                    self.second,
                    output.into_buffer(),
                    natural_order,
                )
            }
        }

        impl<K, F> $expression_by<ArithmeticRange<K>, ArithmeticRange<K>, F>
        where
            K: TemporaryValue
                + TryFrom<usize>
                + core::ops::Add<Output = K>
                + core::ops::Mul<Output = K>
                + 'static,
            F: Fn(&K, &K) -> Ordering + Copy + 'static,
        {
            pub fn write_into<A, M>(
                self,
                output: DeviceViewMut<'_, K>,
                execution: &Execution<'_, '_, A, M>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                A: DeviceAllocator,
                M: SetBackend<K> + ArithmeticSetBackend<K>,
            {
                set_arithmetic_by::<K, A, M, F, $operation>(
                    execution,
                    self.first,
                    self.second,
                    output.into_buffer(),
                    self.order,
                )
            }
        }
    };
}

define_key_set_expression!(Union, UnionBy, union, union_by, CubSetUnion);
define_key_set_expression!(
    Intersection,
    IntersectionBy,
    intersection,
    intersection_by,
    CubSetIntersection
);
define_key_set_expression!(
    Difference,
    DifferenceBy,
    difference,
    difference_by,
    CubSetDifference
);
define_key_set_expression!(
    SymmetricDifference,
    SymmetricDifferenceBy,
    symmetric_difference,
    symmetric_difference_by,
    CubSetSymmetricDifference
);

/// Fluent sorted-set operations for an unmaterialized arithmetic range.
///
/// The range itself is a CUB input abstraction. These Thrust methods live in
/// an extension trait so the lower-level crate retains ownership of the type.
pub trait ArithmeticSetRangeExt<K>: Sized {
    fn union(self, second: ArithmeticRange<K>) -> Union<Self, ArithmeticRange<K>>;

    fn union_by<F>(
        self,
        second: ArithmeticRange<K>,
        order: F,
    ) -> UnionBy<Self, ArithmeticRange<K>, F>
    where
        F: Fn(&K, &K) -> Ordering + Copy + 'static;

    fn intersection(self, second: ArithmeticRange<K>) -> Intersection<Self, ArithmeticRange<K>>;

    fn intersection_by<F>(
        self,
        second: ArithmeticRange<K>,
        order: F,
    ) -> IntersectionBy<Self, ArithmeticRange<K>, F>
    where
        F: Fn(&K, &K) -> Ordering + Copy + 'static;

    fn difference(self, second: ArithmeticRange<K>) -> Difference<Self, ArithmeticRange<K>>;

    fn difference_by<F>(
        self,
        second: ArithmeticRange<K>,
        order: F,
    ) -> DifferenceBy<Self, ArithmeticRange<K>, F>
    where
        F: Fn(&K, &K) -> Ordering + Copy + 'static;

    fn symmetric_difference(
        self,
        second: ArithmeticRange<K>,
    ) -> SymmetricDifference<Self, ArithmeticRange<K>>;

    fn symmetric_difference_by<F>(
        self,
        second: ArithmeticRange<K>,
        order: F,
    ) -> SymmetricDifferenceBy<Self, ArithmeticRange<K>, F>
    where
        F: Fn(&K, &K) -> Ordering + Copy + 'static;
}

impl<K> ArithmeticSetRangeExt<K> for ArithmeticRange<K> {
    fn union(self, second: ArithmeticRange<K>) -> Union<Self, ArithmeticRange<K>> {
        Union {
            first: self,
            second,
        }
    }

    fn union_by<F>(
        self,
        second: ArithmeticRange<K>,
        order: F,
    ) -> UnionBy<Self, ArithmeticRange<K>, F>
    where
        F: Fn(&K, &K) -> Ordering + Copy + 'static,
    {
        UnionBy {
            first: self,
            second,
            order,
        }
    }

    fn intersection(self, second: ArithmeticRange<K>) -> Intersection<Self, ArithmeticRange<K>> {
        Intersection {
            first: self,
            second,
        }
    }

    fn intersection_by<F>(
        self,
        second: ArithmeticRange<K>,
        order: F,
    ) -> IntersectionBy<Self, ArithmeticRange<K>, F>
    where
        F: Fn(&K, &K) -> Ordering + Copy + 'static,
    {
        IntersectionBy {
            first: self,
            second,
            order,
        }
    }

    fn difference(self, second: ArithmeticRange<K>) -> Difference<Self, ArithmeticRange<K>> {
        Difference {
            first: self,
            second,
        }
    }

    fn difference_by<F>(
        self,
        second: ArithmeticRange<K>,
        order: F,
    ) -> DifferenceBy<Self, ArithmeticRange<K>, F>
    where
        F: Fn(&K, &K) -> Ordering + Copy + 'static,
    {
        DifferenceBy {
            first: self,
            second,
            order,
        }
    }

    fn symmetric_difference(
        self,
        second: ArithmeticRange<K>,
    ) -> SymmetricDifference<Self, ArithmeticRange<K>> {
        SymmetricDifference {
            first: self,
            second,
        }
    }

    fn symmetric_difference_by<F>(
        self,
        second: ArithmeticRange<K>,
        order: F,
    ) -> SymmetricDifferenceBy<Self, ArithmeticRange<K>, F>
    where
        F: Fn(&K, &K) -> Ordering + Copy + 'static,
    {
        SymmetricDifferenceBy {
            first: self,
            second,
            order,
        }
    }
}

macro_rules! define_pair_set_expression {
    ($expression:ident, $expression_by:ident, $method:ident, $method_by:ident, $operation:ty) => {
        impl<'a, K: DeviceCopy, V: DeviceCopy> KeyValueRange<'a, K, V> {
            pub fn $method<'b>(
                self,
                second: KeyValueRange<'b, K, V>,
            ) -> $expression<Self, KeyValueRange<'b, K, V>> {
                $expression {
                    first: self,
                    second,
                }
            }

            pub fn $method_by<'b, F>(
                self,
                second: KeyValueRange<'b, K, V>,
                order: F,
            ) -> $expression_by<Self, KeyValueRange<'b, K, V>, F>
            where
                F: Fn(&K, &K) -> Ordering + Copy + 'static,
            {
                $expression_by {
                    first: self,
                    second,
                    order,
                }
            }
        }

        impl<'a, 'b, K, V> $expression<KeyValueRange<'a, K, V>, KeyValueRange<'b, K, V>>
        where
            K: TemporaryValue + PartialOrd + 'static,
            V: TemporaryValue,
        {
            pub fn write_into<A, M>(
                self,
                output: KeyValueRangeMut<'_, K, V>,
                execution: &Execution<'_, '_, A, M>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                A: DeviceAllocator,
                M: SetPairsBackend<K, V>,
            {
                let (second_keys, second_values) = self.second.parts();
                set_pairs_by::<K, V, A, M, _, $operation>(
                    execution,
                    self.first,
                    (second_keys, second_values),
                    output,
                    natural_order,
                )
            }
        }

        impl<'a, 'b, K, V, F> $expression_by<KeyValueRange<'a, K, V>, KeyValueRange<'b, K, V>, F>
        where
            K: TemporaryValue + 'static,
            V: TemporaryValue,
            F: Fn(&K, &K) -> Ordering + Copy + 'static,
        {
            pub fn write_into<A, M>(
                self,
                output: KeyValueRangeMut<'_, K, V>,
                execution: &Execution<'_, '_, A, M>,
            ) -> Result<usize, DeviceAlgorithmError>
            where
                A: DeviceAllocator,
                M: SetPairsBackend<K, V>,
            {
                let (second_keys, second_values) = self.second.parts();
                set_pairs_by::<K, V, A, M, F, $operation>(
                    execution,
                    self.first,
                    (second_keys, second_values),
                    output,
                    self.order,
                )
            }
        }
    };
}

define_pair_set_expression!(Union, UnionBy, union, union_by, CubSetUnion);
define_pair_set_expression!(
    Difference,
    DifferenceBy,
    difference,
    difference_by,
    CubSetDifference
);
define_pair_set_expression!(
    SymmetricDifference,
    SymmetricDifferenceBy,
    symmetric_difference,
    symmetric_difference_by,
    CubSetSymmetricDifference
);

impl<'a, K: DeviceCopy, V: DeviceCopy> KeyValueRange<'a, K, V> {
    /// Intersect a key/value input with a key-only input. Matching values are
    /// selected from the first sequence, as in pinned Thrust.
    pub fn intersection<'b>(
        self,
        second: DeviceView<'b, K>,
    ) -> Intersection<Self, DeviceView<'b, K>> {
        Intersection {
            first: self,
            second,
        }
    }

    pub fn intersection_by<'b, F>(
        self,
        second: DeviceView<'b, K>,
        order: F,
    ) -> IntersectionBy<Self, DeviceView<'b, K>, F>
    where
        F: Fn(&K, &K) -> Ordering + Copy + 'static,
    {
        IntersectionBy {
            first: self,
            second,
            order,
        }
    }
}

impl<'a, 'b, K, V> Intersection<KeyValueRange<'a, K, V>, DeviceView<'b, K>>
where
    K: TemporaryValue + PartialOrd + 'static,
    V: TemporaryValue,
{
    pub fn write_into<A, M>(
        self,
        output: KeyValueRangeMut<'_, K, V>,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<usize, DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: SetPairsBackend<K, V>,
    {
        execute_pair_intersection(execution, self.first, self.second, output, natural_order)
    }
}

impl<'a, 'b, K, V, F> IntersectionBy<KeyValueRange<'a, K, V>, DeviceView<'b, K>, F>
where
    K: TemporaryValue + 'static,
    V: TemporaryValue,
    F: Fn(&K, &K) -> Ordering + Copy + 'static,
{
    pub fn write_into<A, M>(
        self,
        output: KeyValueRangeMut<'_, K, V>,
        execution: &Execution<'_, '_, A, M>,
    ) -> Result<usize, DeviceAlgorithmError>
    where
        A: DeviceAllocator,
        M: SetPairsBackend<K, V>,
    {
        execute_pair_intersection(execution, self.first, self.second, output, self.order)
    }
}

fn execute_pair_intersection<K, V, A, M, F>(
    execution: &Execution<'_, '_, A, M>,
    first: KeyValueRange<'_, K, V>,
    second: DeviceView<'_, K>,
    output: KeyValueRangeMut<'_, K, V>,
    order: F,
) -> Result<usize, DeviceAlgorithmError>
where
    K: TemporaryValue + 'static,
    V: TemporaryValue,
    A: DeviceAllocator,
    M: SetPairsBackend<K, V>,
    F: Fn(&K, &K) -> Ordering + Copy + 'static,
{
    let (_, first_values) = first.parts();
    set_pairs_by::<K, V, A, M, F, CubSetIntersection>(
        execution,
        first,
        (second.into_buffer(), first_values),
        output,
        order,
    )
}

#[cfg(test)]
mod tests {
    use cuda_core::DeviceBuffer;

    use crate::DeviceBufferExt;

    fn compose_without_launching<'a, 'b>(
        first: &'a DeviceBuffer<i32>,
        second: &'b DeviceBuffer<i32>,
    ) -> super::Union<crate::DeviceView<'a, i32>, crate::DeviceView<'b, i32>> {
        first.view().union(second.view())
    }

    #[test]
    fn set_composition_preserves_both_input_lifetimes() {
        let _ = compose_without_launching;
    }
}
