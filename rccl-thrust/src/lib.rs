#![feature(specialization)]
#![expect(
    incomplete_features,
    reason = "CUDA algorithm dispatch currently requires generic_const_exprs"
)]

//! Thrust-shaped host algorithm facade backed by `rccl-cub`.

mod algorithm;
pub mod cuda;
mod merge;

pub use algorithm::{
    SortError, sort, sort_by, sort_by_key, sort_by_key_with, stable_sort, stable_sort_by,
    stable_sort_by_key, stable_sort_by_key_with,
};
pub use cuda_core::DeviceCopy;
pub use merge::{merge, merge_by, merge_by_key, merge_by_key_with};
pub use rccl_cub::dispatch::merge::MergeError;
pub use rccl_cub::{BFloat16, DeviceBool, F16};
pub use rccl_memory as memory;
