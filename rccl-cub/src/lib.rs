#![feature(generic_const_exprs)]
#![expect(
    incomplete_features,
    reason = "cuda-oxide kernels currently require generic_const_exprs"
)]

//! CUB-level device-wide algorithms and cooperative primitives.

mod comparison;
mod context;
mod device;
mod merge;
mod radix;
mod schedule;
mod warp_merge_sort;

#[doc(hidden)]
pub use context::{KernelContext, KernelContextError};
pub use device::{BFloat16, DeviceBool, F16};
pub use warp_merge_sort::{
    WarpMergeSort, WarpMergeSortConfig, WarpMergeSortTempStorage,
    warp_merge_sort_temp_storage_align, warp_merge_sort_temp_storage_bytes,
};

/// CUB device-wide dispatch surface consumed by host algorithm layers.
#[doc(hidden)]
pub mod dispatch {
    pub use crate::comparison::{ComparisonSortError, DeviceMergeSort};
    pub use crate::radix::sort_abi::{SortError as RadixSortError, backend::DeviceRadixSort};

    pub mod merge {
        pub use crate::merge::{MergeError, merge, merge_by, merge_by_key, merge_by_key_with};
    }

    pub mod schedule {
        pub use crate::schedule::*;
    }
}
