//! Thrust-level Rust support.
//!
//! Runtime values, range adaptors, and algorithm calls use ordinary Rust
//! traits and methods. Architecture selection belongs to cargo-oxide's target
//! configuration; it is not represented in the public API.

mod backend;
mod compaction;
mod copy;
mod error;
mod generation;
mod grouped_reduction;
mod histogram;
mod merge;
mod ordering;
mod record_sort;
mod reduce;
mod scan;
mod search;
mod set;
mod transform;

#[cfg(test)]
mod api_tests;

pub mod cuda;
pub mod range;

pub use compaction::{
    PendingFlaggedSelect, PendingInPlaceCompaction, PendingSelect, PendingUnique, select_flagged,
    select_flagged_async, select_if, select_if_async, stable_partition_by,
    stable_partition_by_async, unique_by, unique_by_async, unique_copy_by, unique_copy_by_async,
};
pub use copy::{
    DeviceRangeCopyExt, PendingCopy, PendingGather, PendingRangeCopy, copy, copy_async, copy_range,
    copy_range_async, gather, gather_async,
};
pub use cuda_core::DeviceCopy;
pub use error::TransformError;
pub use generation::{
    DeviceGenerateExt, PendingFill, PendingRangeFill, PendingTabulate, SequenceValue, fill,
    fill_async, fill_range, fill_range_async, sequence, sequence_by, sequence_from, sequence_range,
    sequence_range_by, sequence_range_from, tabulate, tabulate_range, tabulate_range_async,
};
pub use grouped_reduction::{
    BooleanOr, BooleanProvenance, IndependentProbabilityOr, NaturalBagAdd, PendingReduceByKey,
    reduce_by_key, reduce_by_key_async,
};
pub use histogram::{
    histogram_even, histogram_even_arithmetic, histogram_range, histogram_range_arithmetic,
};
pub use merge::{Merge, MergeBy};
pub use ordering::{DeviceKeyValueSortExt, DeviceSortExt};
pub use rccl_cub::{
    ArithmeticRange, ContiguousRange, DeviceAlgorithmError, DeviceHistogram, DeviceHistogramLevels,
    DeviceHistogramLevelsView, DeviceRange, DoubleBuffer, HistogramRegion, KeyValue,
    RandomAccessRange, ReadRange, WriteRange,
};
pub use rccl_memory as memory;
pub use record_sort::{
    PendingRowPayloadSort, PendingRowPayloadSortInto, PendingRowSort, PendingRowSortInto,
    RecordPayload, RelationRow, sort_rows_lexicographic, sort_rows_lexicographic_async,
    sort_rows_lexicographic_into, sort_rows_lexicographic_into_async,
    sort_rows_with_payload_lexicographic, sort_rows_with_payload_lexicographic_async,
    sort_rows_with_payload_lexicographic_into, sort_rows_with_payload_lexicographic_into_async,
};
pub use reduce::{DeviceReduceExt, reduce};
pub use scan::{
    DeviceScanExt, PendingScan, exclusive_scan, exclusive_scan_async, inclusive_scan,
    inclusive_scan_async,
};
pub use search::{
    PendingEqualRange, PendingSearch, equal_range, equal_range_async, equal_range_by,
    equal_range_by_async, lower_bound, lower_bound_async, lower_bound_by, lower_bound_by_async,
    upper_bound, upper_bound_async, upper_bound_by, upper_bound_by_async,
};
pub use set::{
    ArithmeticSetRangeExt, Difference, DifferenceBy, Intersection, IntersectionBy,
    SymmetricDifference, SymmetricDifferenceBy, Union, UnionBy,
};
pub use transform::{
    DeviceTransformExt, PendingForEach, PendingRangeTransform, for_each, for_each_async,
    map_with_value_in_place, transform, transform_into, transform_into_async,
};

/// Extension traits used by ordinary Rust method syntax.
pub mod prelude {
    pub use crate::{
        ArithmeticSetRangeExt, DeviceBufferExt, DeviceGenerateExt, DeviceKeyValueSortExt,
        DeviceRangeCopyExt, DeviceRangeExt, DeviceReduceExt, DeviceScanExt, DeviceSortExt,
        DeviceTransformExt,
    };
}

pub use range::{
    DeviceBufferExt, DeviceRangeExt, DeviceSlice, DeviceSliceMut, DeviceView, DeviceViewMut,
    KeyValueRange, KeyValueRangeMut, RangeError, arithmetic_range, discard, indexed_output,
    repeat_n,
};
