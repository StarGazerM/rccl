//! Private segmented composition for workload reduce-by-key.

use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_memory::DeviceAllocator;

use crate::{
    DeviceAlgorithmError,
    compaction::{CompactionIndex, CompactionWorkspace, compaction_workspace_in},
    cuda::Launch,
    scan::device_exclusive_sum_region,
};

mod sealed {
    pub trait Sealed {}
}

/// Fixed workload reduction operator accepted by adjacent-key grouping.
///
/// This trait is sealed: RCCL intentionally supports only the provenance
/// operations exercised by SRDatalog rather than exposing a general CUB facade.
#[doc(hidden)]
pub trait GroupReduceOperator<T>: sealed::Sealed + Copy {
    fn combine(self, left: T, right: T) -> T;
}

/// Natural-bag provenance addition modulo 2^64.
#[derive(Clone, Copy, Debug, Default)]
pub struct NaturalBagAdd;

impl sealed::Sealed for NaturalBagAdd {}
impl GroupReduceOperator<u64> for NaturalBagAdd {
    #[inline(always)]
    fn combine(self, left: u64, right: u64) -> u64 {
        left.wrapping_add(right)
    }
}

/// Byte-compatible Boolean provenance value with normalized 0/1 storage.
///
/// Rust `bool` is intentionally not `DeviceCopy`; this transparent wrapper
/// preserves SRDatalog's one-byte representation without permitting invalid
/// device-produced bit patterns to become Rust booleans.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, DeviceCopy)]
pub struct BooleanProvenance(u8);

impl BooleanProvenance {
    pub const FALSE: Self = Self(0);
    pub const TRUE: Self = Self(1);

    pub const fn new(value: bool) -> Self {
        if value { Self::TRUE } else { Self::FALSE }
    }

    pub const fn get(self) -> bool {
        self.0 != 0
    }
}

/// Boolean-semiring provenance addition.
#[derive(Clone, Copy, Debug, Default)]
pub struct BooleanOr;

impl sealed::Sealed for BooleanOr {}
impl GroupReduceOperator<BooleanProvenance> for BooleanOr {
    #[inline(always)]
    fn combine(
        self,
        left: BooleanProvenance,
        right: BooleanProvenance,
    ) -> BooleanProvenance {
        BooleanProvenance::new(left.get() || right.get())
    }
}

/// Independent-probability semiring addition for finite inputs in [0, 1].
#[derive(Clone, Copy, Debug, Default)]
pub struct IndependentProbabilityOr;

impl sealed::Sealed for IndependentProbabilityOr {}
impl GroupReduceOperator<f64> for IndependentProbabilityOr {
    #[inline(always)]
    fn combine(self, left: f64, right: f64) -> f64 {
        let combined = left + right - left * right;
        if combined < 0.0 {
            0.0
        } else if combined > 1.0 {
            1.0
        } else {
            combined
        }
    }
}

#[doc(hidden)]
pub fn grouped_reduction_workspace_in<'stream, A>(
    allocator: A,
    launch: &Launch<'stream, '_>,
    len: usize,
) -> Result<CompactionWorkspace<'stream, A>, DeviceAlgorithmError>
where
    A: DeviceAllocator + Copy,
{
    compaction_workspace_in(allocator, launch, len)
}

fn reduce_by_key_inner<A, Keys, Values, OutputKeys, OutputValues, Equivalent, Operator, I>(
    launch: &Launch<'_, '_>,
    keys: Keys,
    values: Values,
    output_keys: OutputKeys,
    output_values: OutputValues,
    group_count: &mut DeviceBuffer<u64>,
    equivalent: Equivalent,
    operator: Operator,
    len: usize,
    marks: &mut rccl_memory::TemporaryWorkspace2<'_, A, I, I>,
    scan: &mut crate::DeviceScanWorkspace<'_, A, I>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    Keys: crate::DeviceReadAt,
    Values: crate::DeviceReadAt,
    OutputKeys: crate::DeviceWriteAt<Item = Keys::Item>,
    OutputValues: crate::DeviceWriteAt<Item = Values::Item>,
    Equivalent: Fn(Keys::Item, Keys::Item) -> bool + Copy,
    Operator: GroupReduceOperator<Values::Item>,
    I: CompactionIndex,
{
    let (heads, positions) = marks.parts_mut();
    launch
        .compaction_mark_adjacent_unique(keys, heads, positions, equivalent, len)
        .map_err(DeviceAlgorithmError::Launch)?;
    device_exclusive_sum_region(launch, positions, scan)?;
    launch
        .compaction_reduce_adjacent_groups(
            keys,
            values,
            output_keys,
            output_values,
            heads,
            positions,
            group_count,
            operator,
            len,
        )
        .map_err(DeviceAlgorithmError::Launch)
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn device_reduce_by_key<A, Keys, Values, OutputKeys, OutputValues, Equivalent, Operator>(
    launch: &Launch<'_, '_>,
    keys: Keys,
    values: Values,
    output_keys: OutputKeys,
    output_values: OutputValues,
    group_count: &mut DeviceBuffer<u64>,
    equivalent: Equivalent,
    operator: Operator,
    len: usize,
    workspace: &mut CompactionWorkspace<'_, A>,
) -> Result<(), DeviceAlgorithmError>
where
    A: DeviceAllocator,
    Keys: crate::DeviceReadAt,
    Values: crate::DeviceReadAt,
    OutputKeys: crate::DeviceWriteAt<Item = Keys::Item>,
    OutputValues: crate::DeviceWriteAt<Item = Values::Item>,
    Equivalent: Fn(Keys::Item, Keys::Item) -> bool + Copy,
    Operator: GroupReduceOperator<Values::Item>,
{
    if group_count.is_empty() {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "group-count output",
            required: 1,
            actual: 0,
        });
    }
    match workspace {
        CompactionWorkspace::U32 { marks, scan } => reduce_by_key_inner::<
            _, _, _, _, _, _, _, u32,
        >(
            launch,
            keys,
            values,
            output_keys,
            output_values,
            group_count,
            equivalent,
            operator,
            len,
            marks,
            scan,
        ),
        CompactionWorkspace::U64 { marks, scan } => reduce_by_key_inner::<
            _, _, _, _, _, _, _, u64,
        >(
            launch,
            keys,
            values,
            output_keys,
            output_values,
            group_count,
            equivalent,
            operator,
            len,
            marks,
            scan,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BooleanOr, BooleanProvenance, GroupReduceOperator, IndependentProbabilityOr,
        NaturalBagAdd,
    };

    #[test]
    fn workload_operator_contracts_are_explicit() {
        assert_eq!(
            NaturalBagAdd.combine(u64::MAX, 2),
            1,
            "natural-bag addition wraps modulo 2^64"
        );
        assert_eq!(
            BooleanOr
                .combine(BooleanProvenance::FALSE, BooleanProvenance::TRUE)
                .get(),
            true
        );
        let first = IndependentProbabilityOr.combine(0.1, 0.2);
        let second = IndependentProbabilityOr.combine(first, 0.3);
        assert_eq!(second.to_bits(), 4_602_606_761_578_608_985);
        assert_eq!(IndependentProbabilityOr.combine(1.0, 0.5), 1.0);
    }

    #[test]
    fn grouped_positions_use_common_u32_then_u64() {
        assert_eq!(crate::compaction::compaction_index_bits(u32::MAX as usize), 32);
        assert_eq!(
            crate::compaction::compaction_index_bits(u32::MAX as usize + 1),
            64
        );
    }
}
