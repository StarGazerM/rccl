# rccl-cuco

Rust-native GPU concurrent containers modeled on the reachable cuCollections
subset. The crate includes stream-bound static and dynamic map surfaces,
exact row sets, Bloom filters, and checked device-side probing paths.

Static maps accept contiguous buffers plus zero-materialization arithmetic,
transform, entry, and permutation ranges. Checked permutation access and
stencil-gated `contains_if` report host-visible errors without trapping the
CUDA context. Workload-specific integrations remain unfinished; this is not a
claim of full cuCollections parity. CUDA examples require the pinned
cuda-oxide toolchain and a CUDA installation.

Documentation: <https://docs.rs/rccl-cuco>
