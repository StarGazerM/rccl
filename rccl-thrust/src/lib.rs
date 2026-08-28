//! Thrust-level Rust support.
//!
//! Runtime values and range adaptors are ordinary Rust. Algorithm macros are
//! exported separately, one operation at a time, after their CUB/cuda-oxide
//! lowering exists.

mod error;
mod macros;

#[doc(hidden)]
pub mod __private;
pub mod cuda;
pub mod range;

pub use cuda_core::DeviceCopy;
pub use error::TransformError;
pub use rccl_cub::{
    ContiguousRange, DeviceAlgorithmError, DeviceRange, DoubleBuffer, KeyValue, RandomAccessRange,
    ReadRange, WriteRange,
};
pub use rccl_memory as memory;
