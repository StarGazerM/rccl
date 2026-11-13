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

Radix key/pair workspaces (including double-buffer controls), merge partitions,
and merge-sort workspaces have `_in(allocator, launch, ...)` constructors.
They allocate through the supplied `rccl_memory::DeviceAllocator` and initialize
storage with stream-ordered memset. Pass a pool resource to avoid direct CUDA
allocations; retain the returned workspace for repeated sorts within its capacity.
The original constructors remain available for callers using direct buffers.

Documentation: <https://docs.rs/rccl-cub>
