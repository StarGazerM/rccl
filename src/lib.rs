//! Thin RCCL umbrella.
//!
//! Host algorithms live in `rccl-thrust`; device-wide and cooperative
//! primitives live in `rccl-cub`. This crate keeps the implementation graph
//! acyclic and contains no algorithm implementation.

pub use rccl_thrust::{DeviceCopy, cuda, memory};

/// CUB-level device-wide and cooperative primitives.
pub mod cub {
    pub use rccl_cub::*;
}
