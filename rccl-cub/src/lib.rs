//! CUB-level Rust support.
//!
//! These are ordinary runtime values and Rust capability traits. Closed CUB
//! architecture policy is selected by cargo-oxide's target cfgs, while
//! algorithm control remains ordinary internal Rust.

mod dispatch;
mod double_buffer;
mod error;
mod histogram;
mod histogram_backend;
mod histogram_dispatch;
mod histogram_kernels;
mod merge;
mod merge_sort;
mod ordering_backend;
mod ordering_kernels;
mod radix;
mod range;
mod set_kernels;
mod set_operation;
mod warp;
mod workspace;

#[cfg(test)]
mod api_tests;

#[doc(hidden)]
pub mod __private;
pub mod cuda;
#[doc(hidden)]
pub mod thrust_integration;

pub use dispatch::{
    device_merge_keys, device_merge_pairs, device_merge_sort_keys, device_merge_sort_keys_copy,
    device_merge_sort_pairs, device_merge_sort_pairs_copy, device_radix_sort_keys,
    device_radix_sort_keys_buffer, device_radix_sort_keys_buffer_descending,
    device_radix_sort_keys_descending, device_radix_sort_pairs, device_radix_sort_pairs_buffer,
    device_radix_sort_pairs_buffer_descending, device_radix_sort_pairs_descending,
};
pub use double_buffer::DoubleBuffer;
pub use error::DeviceAlgorithmError;
pub use histogram::{
    DeviceHistogram, DeviceHistogramLevels, DeviceHistogramLevelsView, DeviceHistogramWorkspace,
    HistogramLevels, HistogramRegion, HistogramScalar,
};
pub use histogram_backend::{HistogramFourSampleLoad, HistogramSample};
pub use histogram_dispatch::{
    device_histogram_even, device_histogram_even_arithmetic,
    device_histogram_even_arithmetic_workspace, device_histogram_even_workspace,
    device_histogram_range, device_histogram_range_arithmetic,
    device_histogram_range_arithmetic_workspace, device_histogram_range_workspace,
};
pub use radix::{RadixKey, RadixValue};
pub use range::{
    ArithmeticRange, ContiguousRange, DeviceRange, KeyValue, RandomAccessRange, ReadRange,
    WriteRange, arithmetic_range,
};
pub use rccl_memory::device::HistogramCounter;
pub use warp::{warp_reduce, warp_reduce_items};
pub use workspace::{
    DeviceMergeSortKeysWorkspace, DeviceMergeSortPairsWorkspace, DeviceMergeWorkspace,
    DeviceRadixSortBufferWorkspace, DeviceRadixSortKeysWorkspace, DeviceRadixSortPairsWorkspace,
    DeviceSetOperationWorkspace, RadixSortPairsWorkspaceCounts, RadixSortWorkspaceCounts,
    SetOperationWorkspaceCounts, WorkspaceBuffer, device_merge_pairs_workspace,
    device_merge_sort_keys_workspace, device_merge_sort_pairs_workspace,
    device_merge_workspace as device_merge_keys_workspace, device_radix_sort_keys_buffer_workspace,
    device_radix_sort_keys_workspace as device_radix_sort_workspace,
    device_radix_sort_pairs_buffer_workspace, device_radix_sort_pairs_workspace,
};

pub use {
    device_merge_sort_keys as device_stable_merge_sort_keys,
    device_merge_sort_keys_copy as device_stable_merge_sort_keys_copy,
    device_merge_sort_pairs as device_stable_merge_sort_pairs,
};
