//! Thin RCCL umbrella.
//!
//! Host algorithms live in `rccl-thrust`; device-wide and cooperative
//! primitives live in `rccl-cub`; cuCollections-style containers live in
//! `rccl-cuco`. This crate keeps the implementation graph acyclic and contains
//! no algorithm implementation.

pub use rccl_thrust::{DeviceCopy, cuda, memory};
pub mod cuco {
    pub use rccl_cuco::*;
}

/// CUB-level device-wide and cooperative primitives.
pub mod cub {
    pub use rccl_cub::*;
}
