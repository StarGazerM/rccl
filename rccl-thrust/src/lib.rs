//! Thrust-level Rust support.
//!
//! Runtime values, range adaptors, and algorithm calls use ordinary Rust
//! traits and methods. The program macro is the compile-time boundary that
//! materializes the target-specific CUB/cuda-oxide backend implementations.

mod backend;
mod error;
mod macros;
mod merge;
mod ordering;
mod set;

#[doc(hidden)]
pub mod __private;
pub mod cuda;
pub mod range;

pub use cuda_core::DeviceCopy;
pub use error::TransformError;
pub use merge::{Merge, MergeBy};
pub use ordering::{DeviceKeyValueSortExt, DeviceSortExt};
pub use rccl_cub::{
    ContiguousRange, DeviceAlgorithmError, DeviceRange, DoubleBuffer, KeyValue, RandomAccessRange,
    ReadRange, WriteRange,
};
pub use rccl_memory as memory;
pub use set::{
    Difference, DifferenceBy, Intersection, IntersectionBy, SymmetricDifference,
    SymmetricDifferenceBy, Union, UnionBy,
};

/// Extension traits used by ordinary Rust method syntax.
pub mod prelude {
    pub use crate::{DeviceBufferExt, DeviceKeyValueSortExt, DeviceRangeExt, DeviceSortExt};
}

pub use range::{
    DeviceBufferExt, DeviceRangeExt, DeviceView, DeviceViewMut, KeyValueRange, KeyValueRangeMut,
    RangeError, arithmetic_range, discard, indexed_output, repeat_n,
};
