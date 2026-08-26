//! RCCL compatibility umbrella.
//!
//! Host algorithms live in `rccl-thrust`; device-wide and cooperative
//! primitives live in `rccl-cub`. This crate keeps the implementation graph
//! acyclic and contains no algorithm implementation.

pub use rccl_thrust::{
    BFloat16, DeviceBool, DeviceCopy, F16, MergeError, SortError, cuda, memory, merge, merge_by,
    merge_by_key, merge_by_key_with, sort, sort_by, sort_by_key, sort_by_key_with, stable_sort,
    stable_sort_by, stable_sort_by_key, stable_sort_by_key_with,
};

/// CUB-level device-wide and cooperative primitives.
pub mod cub {
    pub use rccl_cub::*;
}
