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
pub use rccl_thrust_macros::{thrust_kernels, transform};

/// Errors from the implemented in-place repeated-value transform slice.
#[derive(Debug)]
pub enum TransformError {
    LengthMismatch { input: usize, output: usize },
    Launch(cuda_core::LaunchContractError),
}

impl core::fmt::Display for TransformError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { input, output } => {
                write!(
                    formatter,
                    "transform input length {input} differs from output length {output}"
                )
            }
            Self::Launch(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TransformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LengthMismatch { .. } => None,
            Self::Launch(error) => Some(error),
        }
    }
}

#[doc(hidden)]
pub mod __private {
    pub use cuda_core;
}

pub mod cuda;
pub mod range;
