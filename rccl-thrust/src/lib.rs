//! Thrust-level Rust support.
//!
//! Runtime values and range adaptors are ordinary Rust. Algorithm macros are
//! exported separately, one operation at a time, after their CUB/cuda-oxide
//! lowering exists.

pub use cuda_core::DeviceCopy;
pub use rccl_cub::{
    ContiguousRange, DeviceRange, DoubleBuffer, KeyValue, RandomAccessRange, ReadRange,
    TempStorage, TempStorageError, TempStorageLayout, WriteRange,
};
pub use rccl_memory as memory;

pub mod cuda;
pub mod range;
