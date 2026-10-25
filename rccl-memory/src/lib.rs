//! RMM-style, stream-ordered device memory management for Rust CUDA code.
//!
//! The crate separates three concerns:
//!
//! - [`DeviceAllocator`] is the stream-ordered allocation policy boundary.
//! - resource implementations select synchronous, CUDA-default-pool, explicit
//!   pool, or instrumented allocation behavior.
//! - [`DeviceBuffer`] provides typed RAII ownership over an allocator-backed
//!   allocation.
//!
//! `std::alloc::Allocator` is intentionally not implemented. It describes
//! host-addressable memory and has no CUDA stream parameter, while device
//! allocation and deallocation are ordered operations in a stream DAG.

use cuda_core::{
    CudaContext, CudaStream, DeviceCopy, DriverError, IntoResult,
    simt::memory::{
        free_async, free_host, free_sync, malloc_async, malloc_host, malloc_sync,
        memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async, memset_d8_async,
    },
    sys,
};
use std::{
    alloc::Layout,
    cell::RefCell,
    error::Error,
    ffi::c_void,
    fmt,
    marker::PhantomData,
    mem::MaybeUninit,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

mod adaptors;
mod buffer;
mod completion;
pub mod device;
mod fixed;
mod pool;
mod range;
mod resource;
mod temporary;
#[cfg(test)]
mod tests;

pub use adaptors::{
    AlignedMemoryResource, CallbackMemoryResource, FailureCallbackMemoryResource,
    LimitingMemoryResource, LoggingMemoryResource, PrefetchMemoryResource, StatisticsCounter,
    StatisticsMemoryResource, StatisticsSnapshot, ThreadSafeMemoryResource,
};
pub use buffer::{DeviceBuffer, DeviceScalar, DeviceUninitBuffer};
pub use completion::{CudaCompletion, RecordedCudaEvent, cross_stream_dependency};
pub use fixed::{BinningMemoryResource, FixedSizeMemoryResource};
pub use pool::{PoolMemoryResource, SuballocatorStats};
pub use range::{ContiguousDeviceRange, ContiguousDeviceRangeMut, copy_device_range};
pub use resource::*;
pub(crate) use resource::{validate_layout, validate_stream};
#[cfg(test)]
pub(crate) use temporary::temporary_layout;
pub use temporary::{
    TemporaryRegion, TemporaryValue, TemporaryWorkspace1, TemporaryWorkspace2, TemporaryWorkspace3,
    TemporaryWorkspace4, TemporaryWorkspace5,
};
