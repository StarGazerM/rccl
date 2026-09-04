# rccl-thrust

A Rust-native, Thrust-shaped facade over the implemented RCCL device
algorithms. Public APIs use typed device ranges, ordinary Rust closures,
borrowed execution contexts, and explicit stream/allocator lifetimes.

Only audited capability slices are implemented; this crate is not a complete
Thrust compatibility layer and does not expose C++ iterator facades or a JIT.

Documentation: <https://docs.rs/rccl-thrust>
