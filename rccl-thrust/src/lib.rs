//! Thrust-level Rust support.
//!
//! Runtime values, range adaptors, and algorithm calls use ordinary Rust
//! traits and methods. Architecture selection belongs to cargo-oxide's target
//! configuration; it is not represented in the public API.

mod backend;
mod error;
mod histogram;
mod merge;
mod ordering;
mod set;
mod transform;

#[cfg(test)]
mod api_tests;

pub mod cuda;
pub mod range;

pub use cuda_core::DeviceCopy;
pub use error::TransformError;
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
pub use set::{
    ArithmeticSetRangeExt, Difference, DifferenceBy, Intersection, IntersectionBy,
    SymmetricDifference, SymmetricDifferenceBy, Union, UnionBy,
};
pub use transform::transform;

/// Extension traits used by ordinary Rust method syntax.
pub mod prelude {
    pub use crate::{
        ArithmeticSetRangeExt, DeviceBufferExt, DeviceKeyValueSortExt, DeviceRangeExt,
        DeviceSortExt,
    };
}

pub use range::{
    DeviceBufferExt, DeviceRangeExt, DeviceView, DeviceViewMut, KeyValueRange, KeyValueRangeMut,
    RangeError, arithmetic_range, discard, indexed_output, repeat_n,
};
