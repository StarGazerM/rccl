//! CUB-level Rust support.
//!
//! These are ordinary runtime values and Rust capability traits. Closed CUB
//! policy selection, kernel construction, and launch lowering belong to
//! `rccl-cub-macros`; support vocabulary does not.

mod double_buffer;
mod error;
mod histogram;
mod histogram_backend;
mod macros;
mod ordering_backend;
mod range;
mod workspace;

#[doc(hidden)]
pub mod __private;
pub mod cuda;

pub use double_buffer::DoubleBuffer;
pub use error::DeviceAlgorithmError;
pub use histogram::{
    DeviceHistogram, DeviceHistogramLevels, DeviceHistogramLevelsView, DeviceHistogramWorkspace,
    HistogramRegion,
};
pub use range::{
    ArithmeticRange, ContiguousRange, DeviceRange, KeyValue, RandomAccessRange, ReadRange,
    WriteRange, arithmetic_range,
};
pub use workspace::{
    DeviceMergeSortKeysWorkspace, DeviceMergeSortPairsWorkspace, DeviceMergeWorkspace,
    DeviceRadixSortBufferWorkspace, DeviceRadixSortKeysWorkspace, DeviceRadixSortPairsWorkspace,
    DeviceSetOperationWorkspace, RadixSortPairsWorkspaceCounts, RadixSortWorkspaceCounts,
    SetOperationWorkspaceCounts, WorkspaceBuffer,
};

pub use rccl_cub_macros::{
    device_histogram_even, device_histogram_even_arithmetic,
    device_histogram_even_arithmetic_workspace, device_histogram_even_workspace,
    device_histogram_range, device_histogram_range_arithmetic,
    device_histogram_range_arithmetic_workspace, device_histogram_range_workspace,
    device_merge_keys, device_merge_keys_workspace, device_merge_pairs,
    device_merge_pairs_workspace, device_merge_sort_keys, device_merge_sort_keys_copy,
    device_merge_sort_keys_workspace, device_merge_sort_pairs, device_merge_sort_pairs_copy,
    device_merge_sort_pairs_workspace, device_radix_sort_keys, device_radix_sort_keys_buffer,
    device_radix_sort_keys_buffer_descending, device_radix_sort_keys_buffer_workspace,
    device_radix_sort_keys_descending, device_radix_sort_pairs, device_radix_sort_pairs_buffer,
    device_radix_sort_pairs_buffer_descending, device_radix_sort_pairs_buffer_workspace,
    device_radix_sort_pairs_descending, device_radix_sort_pairs_workspace,
    device_radix_sort_workspace, device_stable_merge_sort_keys, device_stable_merge_sort_keys_copy,
    device_stable_merge_sort_pairs, program, warp_reduce, warp_reduce_items,
};
