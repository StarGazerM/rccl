# RCCL: pure-Rust CUDA primitives

This repository uses NVIDIA Research's `cuda-oxide` compiler backend. Kernel
bodies are ordinary Rust compiled through MIR to PTX; the sort implementation
does not call Thrust, CUB, CUDA C++, NVRTC, or a host-side sorting library.

The first executable slice is an out-of-place, stable, eight-pass LSD radix sort
for `u32` keys. Random construction, per-pass histograms, hierarchical scans,
stable scatter, and order validation all operate on CUDA allocations. Only GPU
event timings and one four-byte validation counter reach the CPU.

```bash
source scripts/oxide-env.sh
cargo oxide doctor
RCCL_ELEMENTS=1048576 RCCL_ITERATIONS=1 cargo oxide run
RCCL_BYTES=4GiB RCCL_ITERATIONS=3 cargo oxide run
```

`cargo-oxide` 0.2.1 does not forward trailing application arguments. After an
oxide build, the ordinary Cargo form can also be used when CLI flags are more
convenient: `cargo run --release -- --elements 1048576 --iterations 1`.

## TMP/policy model

`rccl-macros::radix_policy_chain!` emits zero-sized SM policy types. The radix
kernels are generic over those types and use cuda-oxide's policy-aware
`#[launch_bounds]` and `#[unroll]` attributes. Rust monomorphization therefore
produces a different PTX entry for each SM family with no policy argument or
selection branch in device code. The host queries compute capability once and
dispatches the best compatible compiled policy.

## Thrust-compatible target surface

The intended public family is:

- `sort` and `stable_sort`
- `sort_by_key` and `stable_sort_by_key`
- ascending and descending built-in ordering
- primitive integer and floating-point radix transforms
- custom key types and strict-weak-order comparators through stable merge sort
- automatic radix-versus-merge selection matching Thrust's boundary
- reusable temporary storage, streams, and per-SM generated policies

The current implemented data path is the `u32` key-only radix vertical slice.
The remaining type transforms, key/value scatter, and merge fallback are kept
as explicit next implementation stages; they are not claimed complete yet.
