//! Narrow internal bridge used by the `rccl-thrust` facade.
//!
//! These are workspace queries and set-operation implementation markers, not
//! user-selectable policies. Device modules use the separate `__private`
//! namespace required by cuda-oxide's split compilation.

#[doc(hidden)]
pub use crate::compaction::{
    CompactionWorkspace, compaction_workspace_in, device_select_flagged, device_select_if,
    device_stable_partition_by, device_unique_copy_by,
};
#[doc(hidden)]
pub use crate::grouped_reduction::{
    BooleanOr, BooleanProvenance, GroupReduceOperator, IndependentProbabilityOr, NaturalBagAdd,
    device_reduce_by_key, grouped_reduction_workspace_in,
};
pub use crate::ordering_backend::{
    SerialSetOperation, SetDifference, SetIntersection, SetSymmetricDifference, SetUnion,
};
pub use crate::workspace::{
    device_merge_pairs_partition_count, device_merge_partition_count,
    device_merge_sort_partition_count, device_radix_sort_pairs_workspace_counts,
    device_radix_sort_workspace_counts,
};

pub fn transform_repeat_in_place<T, F>(
    launch: &crate::cuda::Launch<'_, '_>,
    output: &mut cuda_core::DeviceBuffer<T>,
    repeated: T,
    op: F,
) -> Result<(), cuda_core::LaunchContractError>
where
    T: cuda_core::DeviceCopy,
    F: Fn(T, T) -> T + Copy,
{
    launch.transform_repeat_in_place(output, repeated, op)
}
