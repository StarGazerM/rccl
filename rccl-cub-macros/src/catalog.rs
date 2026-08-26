//! Typed, private compiler data for the supported CUB surface.
//!
//! This is intentionally not a DSL. Users never author this representation;
//! shallow call-site macros parse ordinary Rust syntax into operation-specific
//! `syn` ASTs and consult this closed semantic catalog while lowering.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Target {
    Sm80,
    Sm89,
    Sm90,
    Sm100,
}

pub(crate) const TARGETS: &[Target] = &[Target::Sm80, Target::Sm89, Target::Sm90, Target::Sm100];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    Planned,
    NativeRust,
    NativeRustPartial,
    BackendBlocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    Thread,
    Warp,
    Block,
    Device,
    Iterator,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Capability {
    Read,
    Write,
    RandomAccess,
    Contiguous,
    Collective,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Callable {
    None,
    Unary,
    Binary,
    Predicate,
    Ordering,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Storage {
    None,
    Shared,
    RuntimeLayout,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Contract {
    pub(crate) capabilities: &'static [Capability],
    pub(crate) callable: Callable,
    pub(crate) storage: Storage,
}

const THREAD_READ: Contract = Contract {
    capabilities: &[Capability::Read, Capability::RandomAccess],
    callable: Callable::None,
    storage: Storage::None,
};
const THREAD_WRITE: Contract = Contract {
    capabilities: &[Capability::Write, Capability::RandomAccess],
    callable: Callable::None,
    storage: Storage::None,
};
const THREAD_FOLD: Contract = Contract {
    capabilities: &[Capability::Read],
    callable: Callable::Binary,
    storage: Storage::None,
};
const THREAD_PREDICATE: Contract = Contract {
    capabilities: &[Capability::Read],
    callable: Callable::Predicate,
    storage: Storage::None,
};
const THREAD_ORDER: Contract = Contract {
    capabilities: &[Capability::Read, Capability::Write],
    callable: Callable::Ordering,
    storage: Storage::None,
};
const THREAD_COMPUTE: Contract = Contract {
    capabilities: &[],
    callable: Callable::Unary,
    storage: Storage::None,
};
const COLLECTIVE_MOVE: Contract = Contract {
    capabilities: &[
        Capability::Read,
        Capability::Write,
        Capability::RandomAccess,
        Capability::Collective,
    ],
    callable: Callable::None,
    storage: Storage::Shared,
};
const COLLECTIVE_REDUCE: Contract = Contract {
    capabilities: &[Capability::Read, Capability::Write, Capability::Collective],
    callable: Callable::Binary,
    storage: Storage::Shared,
};
const COLLECTIVE_SCAN: Contract = Contract {
    capabilities: &[Capability::Read, Capability::Write, Capability::Collective],
    callable: Callable::Binary,
    storage: Storage::Shared,
};
const COLLECTIVE_ORDER: Contract = Contract {
    capabilities: &[
        Capability::Read,
        Capability::Write,
        Capability::RandomAccess,
        Capability::Collective,
    ],
    callable: Callable::Ordering,
    storage: Storage::Shared,
};
const COLLECTIVE_TRANSFORM: Contract = Contract {
    capabilities: &[Capability::Read, Capability::Write, Capability::Collective],
    callable: Callable::Unary,
    storage: Storage::Shared,
};
const DEVICE_TRANSFORM: Contract = Contract {
    capabilities: &[
        Capability::Read,
        Capability::Write,
        Capability::RandomAccess,
    ],
    callable: Callable::Unary,
    storage: Storage::None,
};
const DEVICE_COPY: Contract = Contract {
    capabilities: &[Capability::Read, Capability::Write, Capability::Contiguous],
    callable: Callable::None,
    storage: Storage::None,
};
const DEVICE_QUERY: Contract = Contract {
    capabilities: &[Capability::Read, Capability::RandomAccess],
    callable: Callable::Predicate,
    storage: Storage::RuntimeLayout,
};
const DEVICE_REDUCE: Contract = Contract {
    capabilities: &[Capability::Read, Capability::RandomAccess],
    callable: Callable::Binary,
    storage: Storage::RuntimeLayout,
};
const DEVICE_SCAN: Contract = Contract {
    capabilities: &[
        Capability::Read,
        Capability::Write,
        Capability::RandomAccess,
    ],
    callable: Callable::Binary,
    storage: Storage::RuntimeLayout,
};
const DEVICE_COMPACTION: Contract = Contract {
    capabilities: &[
        Capability::Read,
        Capability::Write,
        Capability::RandomAccess,
    ],
    callable: Callable::Predicate,
    storage: Storage::RuntimeLayout,
};
const DEVICE_HISTOGRAM: Contract = Contract {
    capabilities: &[
        Capability::Read,
        Capability::Write,
        Capability::RandomAccess,
    ],
    callable: Callable::Unary,
    storage: Storage::RuntimeLayout,
};
const DEVICE_ORDERED: Contract = Contract {
    capabilities: &[
        Capability::Read,
        Capability::Write,
        Capability::RandomAccess,
    ],
    callable: Callable::Ordering,
    storage: Storage::RuntimeLayout,
};
const DEVICE_TOPK: Contract = Contract {
    capabilities: &[
        Capability::Read,
        Capability::Write,
        Capability::RandomAccess,
    ],
    callable: Callable::Ordering,
    storage: Storage::RuntimeLayout,
};
const ITERATOR_INPUT: Contract = Contract {
    capabilities: &[Capability::Read, Capability::RandomAccess],
    callable: Callable::None,
    storage: Storage::None,
};
const ITERATOR_OUTPUT: Contract = Contract {
    capabilities: &[Capability::Write, Capability::RandomAccess],
    callable: Callable::None,
    storage: Storage::None,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Primitive {
    pub(crate) name: &'static str,
    pub(crate) status: Status,
    pub(crate) scope: Scope,
    pub(crate) requires: &'static [&'static str],
    pub(crate) schedule: Option<&'static str>,
    pub(crate) kernels: &'static [&'static str],
    pub(crate) contract: Contract,
}

macro_rules! primitive {
    ($name:literal, $status:ident, $scope:ident, [$($requires:literal),*], $schedule:expr, [$($kernels:literal),*], $contract:ident) => {
        Primitive { name: $name, status: Status::$status, scope: Scope::$scope, requires: &[$($requires),*], schedule: $schedule, kernels: &[$($kernels),*], contract: $contract }
    };
}

pub(crate) const PRIMITIVES: &[Primitive] = &[
    primitive!("thread_load", Planned, Thread, [], None, [], THREAD_READ),
    primitive!("thread_store", Planned, Thread, [], None, [], THREAD_WRITE),
    primitive!(
        "thread_reduce",
        NativeRustPartial,
        Thread,
        [],
        None,
        [],
        THREAD_FOLD
    ),
    primitive!("thread_scan", Planned, Thread, [], None, [], THREAD_FOLD),
    primitive!(
        "thread_search",
        Planned,
        Thread,
        [],
        None,
        [],
        THREAD_PREDICATE
    ),
    primitive!("thread_sort", Planned, Thread, [], None, [], THREAD_ORDER),
    primitive!("thread_simd", Planned, Thread, [], None, [], THREAD_COMPUTE),
    primitive!(
        "thread_operators",
        NativeRust,
        Thread,
        [],
        None,
        [],
        THREAD_COMPUTE
    ),
    primitive!(
        "warp_load",
        Planned,
        Warp,
        ["thread_load"],
        None,
        [],
        COLLECTIVE_MOVE
    ),
    primitive!(
        "warp_store",
        Planned,
        Warp,
        ["thread_store"],
        None,
        [],
        COLLECTIVE_MOVE
    ),
    primitive!(
        "warp_exchange",
        Planned,
        Warp,
        ["warp_load", "warp_store"],
        None,
        [],
        COLLECTIVE_MOVE
    ),
    primitive!(
        "warp_reduce",
        NativeRustPartial,
        Warp,
        ["thread_reduce"],
        None,
        [],
        COLLECTIVE_REDUCE
    ),
    primitive!(
        "warp_reduce_batched",
        Planned,
        Warp,
        ["warp_reduce"],
        None,
        [],
        COLLECTIVE_REDUCE
    ),
    primitive!(
        "warp_scan",
        Planned,
        Warp,
        ["thread_scan"],
        None,
        [],
        COLLECTIVE_SCAN
    ),
    primitive!(
        "warp_bitonic_sort",
        Planned,
        Warp,
        ["thread_sort", "warp_exchange"],
        None,
        [],
        COLLECTIVE_ORDER
    ),
    primitive!(
        "warp_merge_sort",
        BackendBlocked,
        Warp,
        ["thread_sort", "warp_exchange"],
        None,
        [],
        COLLECTIVE_ORDER
    ),
    primitive!("warp_utils", Planned, Warp, [], None, [], COLLECTIVE_MOVE),
    primitive!(
        "block_load",
        Planned,
        Block,
        ["warp_load"],
        None,
        [],
        COLLECTIVE_MOVE
    ),
    primitive!(
        "block_load_to_shared",
        Planned,
        Block,
        ["block_load"],
        None,
        [],
        COLLECTIVE_MOVE
    ),
    primitive!(
        "block_store",
        Planned,
        Block,
        ["warp_store"],
        None,
        [],
        COLLECTIVE_MOVE
    ),
    primitive!(
        "block_exchange",
        Planned,
        Block,
        ["warp_exchange"],
        None,
        [],
        COLLECTIVE_MOVE
    ),
    primitive!(
        "block_shuffle",
        Planned,
        Block,
        ["block_exchange"],
        None,
        [],
        COLLECTIVE_MOVE
    ),
    primitive!(
        "block_reduce",
        Planned,
        Block,
        ["warp_reduce"],
        None,
        [],
        COLLECTIVE_REDUCE
    ),
    primitive!(
        "block_scan",
        Planned,
        Block,
        ["warp_scan"],
        None,
        [],
        COLLECTIVE_SCAN
    ),
    primitive!(
        "block_adjacent_difference",
        Planned,
        Block,
        ["block_load", "block_store"],
        None,
        [],
        COLLECTIVE_SCAN
    ),
    primitive!(
        "block_discontinuity",
        Planned,
        Block,
        ["block_adjacent_difference"],
        None,
        [],
        COLLECTIVE_SCAN
    ),
    primitive!(
        "block_merge_sort",
        Planned,
        Block,
        ["warp_merge_sort", "block_exchange"],
        None,
        [],
        COLLECTIVE_ORDER
    ),
    primitive!(
        "block_radix_rank",
        Planned,
        Block,
        ["block_scan"],
        None,
        [],
        COLLECTIVE_SCAN
    ),
    primitive!(
        "block_radix_sort",
        Planned,
        Block,
        ["block_radix_rank", "block_exchange"],
        None,
        [],
        COLLECTIVE_ORDER
    ),
    primitive!(
        "radix_rank_sort_operations",
        Planned,
        Block,
        ["block_radix_rank"],
        None,
        [],
        COLLECTIVE_ORDER
    ),
    primitive!(
        "block_histogram",
        Planned,
        Block,
        ["block_reduce"],
        None,
        [],
        COLLECTIVE_TRANSFORM
    ),
    primitive!(
        "block_run_length_decode",
        Planned,
        Block,
        ["block_scan"],
        None,
        [],
        COLLECTIVE_SCAN
    ),
    primitive!(
        "block_topk",
        Planned,
        Block,
        ["block_merge_sort"],
        None,
        [],
        COLLECTIVE_ORDER
    ),
    primitive!(
        "block_raking_layout",
        Planned,
        Block,
        [],
        None,
        [],
        COLLECTIVE_SCAN
    ),
    primitive!(
        "device_for",
        Planned,
        Device,
        ["block_load"],
        None,
        ["for_each"],
        DEVICE_TRANSFORM
    ),
    primitive!(
        "device_transform",
        Planned,
        Device,
        ["device_for"],
        None,
        ["transform"],
        DEVICE_TRANSFORM
    ),
    primitive!(
        "device_copy",
        Planned,
        Device,
        ["block_load", "block_store"],
        None,
        ["copy"],
        DEVICE_COPY
    ),
    primitive!(
        "device_memcpy",
        Planned,
        Device,
        ["device_copy"],
        None,
        ["memcpy"],
        DEVICE_COPY
    ),
    primitive!(
        "device_find",
        Planned,
        Device,
        ["block_reduce"],
        None,
        ["find"],
        DEVICE_QUERY
    ),
    primitive!(
        "device_reduce",
        Planned,
        Device,
        ["block_reduce"],
        None,
        ["reduce"],
        DEVICE_REDUCE
    ),
    primitive!(
        "device_scan",
        Planned,
        Device,
        ["block_scan"],
        None,
        ["scan"],
        DEVICE_SCAN
    ),
    primitive!(
        "device_segmented_reduce",
        Planned,
        Device,
        ["device_reduce"],
        None,
        ["segmented_reduce"],
        DEVICE_REDUCE
    ),
    primitive!(
        "device_segmented_scan",
        Planned,
        Device,
        ["device_scan"],
        None,
        ["segmented_scan"],
        DEVICE_SCAN
    ),
    primitive!(
        "device_adjacent_difference",
        Planned,
        Device,
        ["block_adjacent_difference"],
        None,
        ["adjacent_difference"],
        DEVICE_SCAN
    ),
    primitive!(
        "device_select",
        Planned,
        Device,
        ["device_scan"],
        None,
        ["select"],
        DEVICE_COMPACTION
    ),
    primitive!(
        "device_partition",
        Planned,
        Device,
        ["device_select"],
        None,
        ["partition"],
        DEVICE_COMPACTION
    ),
    primitive!(
        "device_run_length_encode",
        Planned,
        Device,
        ["block_discontinuity", "device_scan"],
        None,
        ["run_length_encode"],
        DEVICE_COMPACTION
    ),
    primitive!(
        "device_histogram",
        Planned,
        Device,
        ["block_histogram", "device_reduce"],
        None,
        ["histogram"],
        DEVICE_HISTOGRAM
    ),
    primitive!(
        "device_merge",
        BackendBlocked,
        Device,
        ["block_load", "block_store"],
        Some("merge"),
        ["partition", "merge_keys", "merge_pairs"],
        DEVICE_ORDERED
    ),
    primitive!(
        "device_merge_sort",
        BackendBlocked,
        Device,
        ["device_merge"],
        Some("comparison_sort"),
        ["merge_pass"],
        DEVICE_ORDERED
    ),
    primitive!(
        "device_radix_sort",
        BackendBlocked,
        Device,
        ["block_radix_sort", "device_scan"],
        Some("radix_sort"),
        ["histogram", "prefix", "scatter"],
        DEVICE_ORDERED
    ),
    primitive!(
        "device_segmented_radix_sort",
        Planned,
        Device,
        ["device_radix_sort"],
        Some("radix_sort"),
        ["segmented_histogram", "segmented_scatter"],
        DEVICE_ORDERED
    ),
    primitive!(
        "device_segmented_sort",
        Planned,
        Device,
        ["device_merge_sort"],
        Some("comparison_sort"),
        ["segmented_merge_pass"],
        DEVICE_ORDERED
    ),
    primitive!(
        "device_topk",
        Planned,
        Device,
        ["block_topk", "device_select"],
        None,
        ["topk"],
        DEVICE_TOPK
    ),
    primitive!(
        "device_batched_topk",
        Planned,
        Device,
        ["device_topk"],
        None,
        ["batched_topk"],
        DEVICE_TOPK
    ),
    primitive!(
        "arg_index_input_iterator",
        Planned,
        Iterator,
        [],
        None,
        [],
        ITERATOR_INPUT
    ),
    primitive!(
        "cache_modified_input_iterator",
        Planned,
        Iterator,
        [],
        None,
        [],
        ITERATOR_INPUT
    ),
    primitive!(
        "cache_modified_output_iterator",
        Planned,
        Iterator,
        [],
        None,
        [],
        ITERATOR_OUTPUT
    ),
    primitive!(
        "tex_obj_input_iterator",
        Planned,
        Iterator,
        [],
        None,
        [],
        ITERATOR_INPUT
    ),
];

/// One public CUB method reached by the pinned executable examples or API
/// tests. This is an evidence inventory, not a claim that every declaration
/// in all 59 headers has an example.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExercisedOperation {
    pub(crate) cccl: &'static str,
    pub(crate) primitive: &'static str,
    pub(crate) status: Status,
}

macro_rules! exercised_operation {
    ($cccl:literal, $primitive:literal) => {
        ExercisedOperation {
            cccl: $cccl,
            primitive: $primitive,
            status: Status::Planned,
        }
    };
}

pub(crate) const EXERCISED_OPERATIONS: &[ExercisedOperation] = &[
    exercised_operation!(
        "DeviceAdjacentDifference::SubtractLeft",
        "device_adjacent_difference"
    ),
    exercised_operation!(
        "DeviceAdjacentDifference::SubtractLeftCopy",
        "device_adjacent_difference"
    ),
    exercised_operation!(
        "DeviceAdjacentDifference::SubtractRight",
        "device_adjacent_difference"
    ),
    exercised_operation!(
        "DeviceAdjacentDifference::SubtractRightCopy",
        "device_adjacent_difference"
    ),
    exercised_operation!("DeviceBatchedTopK::MaxKeys", "device_batched_topk"),
    exercised_operation!("DeviceBatchedTopK::MaxPairs", "device_batched_topk"),
    exercised_operation!("DeviceBatchedTopK::MinKeys", "device_batched_topk"),
    exercised_operation!("DeviceBatchedTopK::MinPairs", "device_batched_topk"),
    exercised_operation!("DeviceCopy::Batched", "device_copy"),
    exercised_operation!("DeviceCopy::Copy", "device_copy"),
    exercised_operation!("DeviceFind::FindIf", "device_find"),
    exercised_operation!("DeviceFind::LowerBound", "device_find"),
    exercised_operation!("DeviceFind::LowerBoundSortedValues", "device_find"),
    exercised_operation!("DeviceFind::UpperBound", "device_find"),
    exercised_operation!("DeviceFind::UpperBoundSortedValues", "device_find"),
    exercised_operation!("DeviceFor::Bulk", "device_for"),
    exercised_operation!("DeviceFor::ForEach", "device_for"),
    exercised_operation!("DeviceFor::ForEachCopy", "device_for"),
    exercised_operation!("DeviceFor::ForEachCopyN", "device_for"),
    exercised_operation!("DeviceFor::ForEachInExtents", "device_for"),
    exercised_operation!("DeviceFor::ForEachInLayout", "device_for"),
    exercised_operation!("DeviceFor::ForEachN", "device_for"),
    exercised_operation!("DeviceHistogram::HistogramEven", "device_histogram"),
    exercised_operation!("DeviceHistogram::HistogramRange", "device_histogram"),
    exercised_operation!("DeviceHistogram::MultiHistogramEven", "device_histogram"),
    exercised_operation!("DeviceHistogram::MultiHistogramRange", "device_histogram"),
    exercised_operation!("DeviceMemcpy::Batched", "device_memcpy"),
    exercised_operation!("DeviceMerge::MergeKeys", "device_merge"),
    exercised_operation!("DeviceMerge::MergePairs", "device_merge"),
    exercised_operation!("DeviceMergeSort::SortKeys", "device_merge_sort"),
    exercised_operation!("DeviceMergeSort::SortKeysCopy", "device_merge_sort"),
    exercised_operation!("DeviceMergeSort::SortPairs", "device_merge_sort"),
    exercised_operation!("DeviceMergeSort::SortPairsCopy", "device_merge_sort"),
    exercised_operation!("DeviceMergeSort::StableSortKeys", "device_merge_sort"),
    exercised_operation!("DeviceMergeSort::StableSortKeysCopy", "device_merge_sort"),
    exercised_operation!("DeviceMergeSort::StableSortPairs", "device_merge_sort"),
    exercised_operation!("DevicePartition::Flagged", "device_partition"),
    exercised_operation!("DevicePartition::If", "device_partition"),
    exercised_operation!("DeviceRadixSort::SortKeys", "device_radix_sort"),
    exercised_operation!("DeviceRadixSort::SortKeysDescending", "device_radix_sort"),
    exercised_operation!("DeviceRadixSort::SortPairs", "device_radix_sort"),
    exercised_operation!("DeviceRadixSort::SortPairsDescending", "device_radix_sort"),
    exercised_operation!("DeviceReduce::ArgMax", "device_reduce"),
    exercised_operation!("DeviceReduce::ArgMin", "device_reduce"),
    exercised_operation!("DeviceReduce::ArgMinMax", "device_reduce"),
    exercised_operation!("DeviceReduce::Max", "device_reduce"),
    exercised_operation!("DeviceReduce::Min", "device_reduce"),
    exercised_operation!("DeviceReduce::Reduce", "device_reduce"),
    exercised_operation!("DeviceReduce::ReduceByKey", "device_reduce"),
    exercised_operation!("DeviceReduce::Sum", "device_reduce"),
    exercised_operation!("DeviceReduce::TransformReduce", "device_reduce"),
    exercised_operation!("DeviceRunLengthEncode::Encode", "device_run_length_encode"),
    exercised_operation!(
        "DeviceRunLengthEncode::NonTrivialRuns",
        "device_run_length_encode"
    ),
    exercised_operation!("DeviceScan::ExclusiveScan", "device_scan"),
    exercised_operation!("DeviceScan::ExclusiveScanByKey", "device_scan"),
    exercised_operation!("DeviceScan::ExclusiveSum", "device_scan"),
    exercised_operation!("DeviceScan::ExclusiveSumByKey", "device_scan"),
    exercised_operation!("DeviceScan::InclusiveScan", "device_scan"),
    exercised_operation!("DeviceScan::InclusiveScanByKey", "device_scan"),
    exercised_operation!("DeviceScan::InclusiveScanInit", "device_scan"),
    exercised_operation!("DeviceScan::InclusiveSum", "device_scan"),
    exercised_operation!("DeviceScan::InclusiveSumByKey", "device_scan"),
    exercised_operation!(
        "DeviceSegmentedRadixSort::SortKeys",
        "device_segmented_radix_sort"
    ),
    exercised_operation!(
        "DeviceSegmentedRadixSort::SortKeysDescending",
        "device_segmented_radix_sort"
    ),
    exercised_operation!(
        "DeviceSegmentedRadixSort::SortPairs",
        "device_segmented_radix_sort"
    ),
    exercised_operation!(
        "DeviceSegmentedRadixSort::SortPairsDescending",
        "device_segmented_radix_sort"
    ),
    exercised_operation!("DeviceSegmentedReduce::ArgMax", "device_segmented_reduce"),
    exercised_operation!("DeviceSegmentedReduce::ArgMin", "device_segmented_reduce"),
    exercised_operation!("DeviceSegmentedReduce::Max", "device_segmented_reduce"),
    exercised_operation!("DeviceSegmentedReduce::Min", "device_segmented_reduce"),
    exercised_operation!("DeviceSegmentedReduce::Reduce", "device_segmented_reduce"),
    exercised_operation!("DeviceSegmentedReduce::Sum", "device_segmented_reduce"),
    exercised_operation!(
        "DeviceSegmentedScan::ExclusiveSegmentedScan",
        "device_segmented_scan"
    ),
    exercised_operation!(
        "DeviceSegmentedScan::ExclusiveSegmentedSum",
        "device_segmented_scan"
    ),
    exercised_operation!(
        "DeviceSegmentedScan::InclusiveSegmentedScan",
        "device_segmented_scan"
    ),
    exercised_operation!(
        "DeviceSegmentedScan::InclusiveSegmentedScanInit",
        "device_segmented_scan"
    ),
    exercised_operation!(
        "DeviceSegmentedScan::InclusiveSegmentedSum",
        "device_segmented_scan"
    ),
    exercised_operation!("DeviceSegmentedSort::SortKeys", "device_segmented_sort"),
    exercised_operation!(
        "DeviceSegmentedSort::SortKeysDescending",
        "device_segmented_sort"
    ),
    exercised_operation!("DeviceSegmentedSort::SortPairs", "device_segmented_sort"),
    exercised_operation!(
        "DeviceSegmentedSort::SortPairsDescending",
        "device_segmented_sort"
    ),
    exercised_operation!(
        "DeviceSegmentedSort::StableSortKeys",
        "device_segmented_sort"
    ),
    exercised_operation!(
        "DeviceSegmentedSort::StableSortKeysDescending",
        "device_segmented_sort"
    ),
    exercised_operation!(
        "DeviceSegmentedSort::StableSortPairs",
        "device_segmented_sort"
    ),
    exercised_operation!(
        "DeviceSegmentedSort::StableSortPairsDescending",
        "device_segmented_sort"
    ),
    exercised_operation!("DeviceSelect::Flagged", "device_select"),
    exercised_operation!("DeviceSelect::FlaggedIf", "device_select"),
    exercised_operation!("DeviceSelect::If", "device_select"),
    exercised_operation!("DeviceSelect::Unique", "device_select"),
    exercised_operation!("DeviceSelect::UniqueByKey", "device_select"),
    exercised_operation!("DeviceTopK::MaxKeys", "device_topk"),
    exercised_operation!("DeviceTopK::MaxPairs", "device_topk"),
    exercised_operation!("DeviceTopK::MinKeys", "device_topk"),
    exercised_operation!("DeviceTopK::MinPairs", "device_topk"),
    exercised_operation!("DeviceTransform::Fill", "device_transform"),
    exercised_operation!("DeviceTransform::Generate", "device_transform"),
    exercised_operation!("DeviceTransform::Transform", "device_transform"),
    exercised_operation!("DeviceTransform::TransformIf", "device_transform"),
    exercised_operation!(
        "DeviceTransform::TransformStableArgumentAddresses",
        "device_transform"
    ),
    exercised_operation!("WarpReduceBatched::Reduce", "warp_reduce_batched"),
    exercised_operation!("WarpReduceBatched::ReduceToBlocked", "warp_reduce_batched"),
    exercised_operation!("WarpReduceBatched::ReduceToStriped", "warp_reduce_batched"),
    exercised_operation!("WarpReduceBatched::Sum", "warp_reduce_batched"),
    exercised_operation!("WarpReduceBatched::SumToBlocked", "warp_reduce_batched"),
    exercised_operation!("WarpReduceBatched::SumToStriped", "warp_reduce_batched"),
];

#[derive(Clone, Copy, Debug)]
pub(crate) struct ScheduleRow {
    pub(crate) target: Target,
    pub(crate) block_threads: u16,
    pub(crate) items_per_thread: u8,
    pub(crate) min_blocks_per_sm: Option<u8>,
    pub(crate) fallback_threads: Option<u16>,
    pub(crate) radix_bits: Option<u8>,
    pub(crate) unroll: u8,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Schedule {
    pub(crate) name: &'static str,
    pub(crate) rows: &'static [ScheduleRow],
}

const RADIX_SORT_ROWS: &[ScheduleRow] = &[
    ScheduleRow {
        target: Target::Sm80,
        block_threads: 256,
        items_per_thread: 12,
        min_blocks_per_sm: Some(2),
        fallback_threads: None,
        radix_bits: Some(8),
        unroll: 4,
    },
    ScheduleRow {
        target: Target::Sm89,
        block_threads: 256,
        items_per_thread: 16,
        min_blocks_per_sm: Some(2),
        fallback_threads: None,
        radix_bits: Some(8),
        unroll: 4,
    },
    ScheduleRow {
        target: Target::Sm90,
        block_threads: 384,
        items_per_thread: 16,
        min_blocks_per_sm: Some(2),
        fallback_threads: None,
        radix_bits: Some(8),
        unroll: 4,
    },
    ScheduleRow {
        target: Target::Sm100,
        block_threads: 512,
        items_per_thread: 16,
        min_blocks_per_sm: Some(1),
        fallback_threads: None,
        radix_bits: Some(8),
        unroll: 4,
    },
];
const MERGE_ROWS: &[ScheduleRow] = &[
    ScheduleRow {
        target: Target::Sm80,
        block_threads: 256,
        items_per_thread: 15,
        min_blocks_per_sm: None,
        fallback_threads: Some(128),
        radix_bits: None,
        unroll: 15,
    },
    ScheduleRow {
        target: Target::Sm89,
        block_threads: 256,
        items_per_thread: 15,
        min_blocks_per_sm: None,
        fallback_threads: Some(128),
        radix_bits: None,
        unroll: 15,
    },
    ScheduleRow {
        target: Target::Sm90,
        block_threads: 256,
        items_per_thread: 15,
        min_blocks_per_sm: None,
        fallback_threads: Some(128),
        radix_bits: None,
        unroll: 15,
    },
    ScheduleRow {
        target: Target::Sm100,
        block_threads: 256,
        items_per_thread: 15,
        min_blocks_per_sm: None,
        fallback_threads: Some(128),
        radix_bits: None,
        unroll: 15,
    },
];
const COMPARISON_SORT_ROWS: &[ScheduleRow] = &[
    ScheduleRow {
        target: Target::Sm80,
        block_threads: 256,
        items_per_thread: 1,
        min_blocks_per_sm: Some(1),
        fallback_threads: None,
        radix_bits: None,
        unroll: 1,
    },
    ScheduleRow {
        target: Target::Sm89,
        block_threads: 256,
        items_per_thread: 1,
        min_blocks_per_sm: Some(1),
        fallback_threads: None,
        radix_bits: None,
        unroll: 1,
    },
    ScheduleRow {
        target: Target::Sm90,
        block_threads: 256,
        items_per_thread: 1,
        min_blocks_per_sm: Some(1),
        fallback_threads: None,
        radix_bits: None,
        unroll: 1,
    },
    ScheduleRow {
        target: Target::Sm100,
        block_threads: 256,
        items_per_thread: 1,
        min_blocks_per_sm: Some(1),
        fallback_threads: None,
        radix_bits: None,
        unroll: 1,
    },
];

pub(crate) const SCHEDULES: &[Schedule] = &[
    Schedule {
        name: "radix_sort",
        rows: RADIX_SORT_ROWS,
    },
    Schedule {
        name: "merge",
        rows: MERGE_ROWS,
    },
    Schedule {
        name: "comparison_sort",
        rows: COMPARISON_SORT_ROWS,
    },
];

pub(crate) fn validate() -> Result<(), String> {
    unique("primitive", PRIMITIVES.iter().map(|item| item.name))?;
    unique(
        "exercised CUB operation",
        EXERCISED_OPERATIONS.iter().map(|operation| operation.cccl),
    )?;
    let schedule_names = unique("schedule", SCHEDULES.iter().map(|item| item.name))?;

    for schedule in SCHEDULES {
        if schedule.rows.len() != TARGETS.len() {
            return Err(format!(
                "schedule `{}` does not cover every target",
                schedule.name
            ));
        }
        let row_targets = unique(
            "schedule target",
            schedule.rows.iter().map(|row| match row.target {
                Target::Sm80 => "sm80",
                Target::Sm89 => "sm89",
                Target::Sm90 => "sm90",
                Target::Sm100 => "sm100",
            }),
        )?;
        if row_targets.len() != TARGETS.len() {
            return Err(format!("schedule `{}` repeats a target", schedule.name));
        }
        for row in schedule.rows {
            if row.block_threads == 0 || row.items_per_thread == 0 || row.unroll == 0 {
                return Err(format!(
                    "schedule `{}` contains a zero tuning value",
                    schedule.name
                ));
            }
            let _optional_tuning = (row.min_blocks_per_sm, row.fallback_threads, row.radix_bits);
        }
    }

    let names = PRIMITIVES
        .iter()
        .map(|item| item.name)
        .collect::<BTreeSet<_>>();
    for operation in EXERCISED_OPERATIONS {
        if !names.contains(operation.primitive) {
            return Err(format!(
                "exercised operation `{}` maps to unknown primitive `{}`",
                operation.cccl, operation.primitive
            ));
        }
        let _status = operation.status;
    }
    for primitive in PRIMITIVES {
        for dependency in primitive.requires {
            if !names.contains(dependency) {
                return Err(format!(
                    "primitive `{}` depends on unknown `{dependency}`",
                    primitive.name
                ));
            }
        }
        if primitive
            .schedule
            .is_some_and(|name| !schedule_names.contains(name))
        {
            return Err(format!(
                "primitive `{}` names an unknown schedule",
                primitive.name
            ));
        }
        unique("kernel", primitive.kernels.iter().copied())?;
        unique(
            "operation capability",
            primitive
                .contract
                .capabilities
                .iter()
                .map(|capability| match capability {
                    Capability::Read => "read",
                    Capability::Write => "write",
                    Capability::RandomAccess => "random_access",
                    Capability::Contiguous => "contiguous",
                    Capability::Collective => "collective",
                }),
        )?;
        match primitive.contract.storage {
            Storage::Shared if !matches!(primitive.scope, Scope::Warp | Scope::Block) => {
                return Err(format!(
                    "primitive `{}` has illegal shared storage",
                    primitive.name
                ));
            }
            Storage::RuntimeLayout if primitive.scope != Scope::Device => {
                return Err(format!(
                    "primitive `{}` has illegal runtime storage",
                    primitive.name
                ));
            }
            _ => {}
        }
        let _closed_semantics = (primitive.status, primitive.contract.callable);
    }
    validate_acyclic()
}

fn unique<'a>(
    kind: &str,
    names: impl IntoIterator<Item = &'a str>,
) -> Result<BTreeSet<&'a str>, String> {
    let mut unique = BTreeSet::new();
    for name in names {
        if !unique.insert(name) {
            return Err(format!("duplicate {kind} `{name}`"));
        }
    }
    Ok(unique)
}

fn validate_acyclic() -> Result<(), String> {
    fn visit<'a>(
        name: &'a str,
        by_name: &BTreeMap<&'a str, &'a Primitive>,
        active: &mut BTreeSet<&'a str>,
        complete: &mut BTreeSet<&'a str>,
    ) -> Result<(), String> {
        if complete.contains(name) {
            return Ok(());
        }
        if !active.insert(name) {
            return Err(format!("CUB primitive dependency cycle reaches `{name}`"));
        }
        for dependency in by_name[name].requires {
            visit(dependency, by_name, active, complete)?;
        }
        active.remove(name);
        complete.insert(name);
        Ok(())
    }

    let by_name = PRIMITIVES
        .iter()
        .map(|primitive| (primitive.name, primitive))
        .collect::<BTreeMap<_, _>>();
    let mut active = BTreeSet::new();
    let mut complete = BTreeSet::new();
    for primitive in PRIMITIVES {
        visit(primitive.name, &by_name, &mut active, &mut complete)?;
    }
    Ok(())
}
