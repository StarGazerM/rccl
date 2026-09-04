# rccl-cuco

Rust-native GPU concurrent containers modeled on the reachable cuCollections
subset. The crate includes stream-bound static and dynamic map surfaces,
exact row sets, Bloom filters, and checked device-side probing paths.

Static-map range composition and some workload-specific integrations remain
unfinished; this is not a claim of full cuCollections parity. CUDA examples
require the pinned cuda-oxide toolchain and a CUDA installation.

Documentation: <https://docs.rs/rccl-cuco>
