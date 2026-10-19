# rccl-cub

Rust-native device-wide and cooperative CUDA primitives. This crate provides
typed launch contexts, range/workspace contracts, scans, reductions, ordering,
histograms, and fixed cuda-oxide kernel modules for the implemented subset.

Coverage is intentionally partial; consult the repository capability ledgers.
GPU examples require the pinned cuda-oxide toolchain and CUDA installation.

The fixed u32 lookback scan uses 8192 bytes of 16-byte-aligned dynamic shared
memory, supplied by its launcher. It does not export a static shared-memory
global, allowing its program module to coexist with caller-monomorphized radix
kernels without a duplicate `__shared_mem_0` in merged PTX. The device scan
example covers both sides of the 1920-item lookback tile boundary.

Documentation: <https://docs.rs/rccl-cub>
