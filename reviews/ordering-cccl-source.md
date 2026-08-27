# CUB/Thrust ordering upstream correspondence and masking review

Date: 2026-08-27

Pinned CCCL revision: `571f2fc3bc53cd710e306ad43d58995c1fe4219f`.

Sources read for the currently exported intersection:

- `cub/test/catch2_test_device_merge_api.cu`;
- `cub/test/catch2_test_device_merge.cu`;
- `cub/test/catch2_test_device_merge_sort_iterators.cu`;
- `cub/test/catch2_test_device_radix_sort_keys.cu`;
- `cub/test/catch2_test_device_radix_sort_pairs.cu`;
- `cub/examples/device/example_device_radix_sort.cu`;
- `cub/cub/device/device_radix_sort.cuh` and `cub/cub/util_type.cuh`;
- `thrust/examples/sort.cu`;
- `thrust/testing/sort.cu`, `stable_sort.cu`, `sort_by_key.cu`, and
  `stable_sort_by_key.cu`;
- `thrust/testing/merge.cu` and `merge_key_value.cu`.

The direct public Rust calls are the `device_merge_*!`,
`device_merge_sort_*!`, and `device_radix_sort_*!` CUB macros. Thrust
`sort*!` and `merge*!` macros lower through those exact CUB operation macros;
Thrust owns no ordering kernel.

## TMP boundary

C++ deduces `KeyT`, `ValueT`, iterator value types, and policy inputs before
instantiating dispatch and agents. A stable Rust proc macro cannot query
rustc's later inference. Every fact that changes this lowering is therefore
macro input:

- key and value types select layout-dependent merge policy;
- primitive radix category selects unsigned, sign-bit, or floating bit
  transformation;
- radix width selects the stable digit-pass tree;
- a custom decomposer and its component types will be required explicitly
  before custom radix keys can be exported; and
- fixed extents, bit windows, and offset category must likewise be explicit
  when their forms are implemented.

Primitive aliases and custom records currently fail closed because their
layout/decomposition is opaque to the macro. No `TypeId`, runtime trait
object, type registry, inferred fallback, or comparison-sort substitution is
used to hide that missing TMP input.

## Result masking audit

- `rccl-cub/examples/device_ordering_probe.rs` uses cuda-oxide GPU launches
  and checks complete outputs. It contains the exact `MergeKeys` and
  `MergePairs` example input/output from pinned CCCL.
- Merge-pair payloads independently expose first-range precedence for equal
  keys. Checking sorted keys alone would miss this error.
- Merge-sort key/value cases check complete permutations and stable payload
  movement, not only sortedness.
- Radix integer cases include signed minima/maxima, duplicates, ascending and
  descending forms, and payload tags.
- Floating radix checks `to_bits()` for every output. Rust floating equality
  would consider `-0.0 == +0.0` and mask the required stable signed-zero
  behavior, so it is not used as the oracle.
- Copy forms verify both the full output and that the full input remains
  unchanged.
- All calls propagate `Result`; no launch error is discarded. None of the
  semantic fixtures is empty, so the zero-length fast path cannot make them
  vacuously pass.
- The executable is built with `cargo oxide build` immediately before
  execution. Both fault iterations reported recompilation of the macro and
  runtime crates, ruling out a stale pre-fault executable.
- `benchmarks/ordering-reference.cu` and the Rust benchmarks use complete
  device-resident validation. The strict 1.001 performance ratio is still
  failing, so result parity cannot be misreported as CCCL performance parity.
  After repairing the equal-input oracle described below, the run measured
  4.241231x for direct merge, 5.535783x for merge sort, 30.605403x for radix,
  2.918400x for Thrust merge, and 9.363180x for Thrust sort. The gate exited
  nonzero with `implementation gap is not masked`.

## Injected faults

Both faults were temporary and removed after observing the intended failure.

| Fault | Intended detector | Observed failure |
|---|---|---|
| Give the second merge input precedence when keys compare equal | exact pinned `MergePairs` payload assertion | GPU executable exited 101: output bytes `[65, 97, 98, 66, 67, 68, 99]`, expected `[97, 65, 98, 66, 67, 68, 99]` (`Aa...` instead of `aA...`) |
| Disable floating negative-zero normalization in both digit encoders | descending bitwise floating assertion | GPU executable exited 101: zero bit patterns were `[0, 2147483648]`, expected stable `[2147483648, 0]` |

The first performance review also found that the Rust benchmark generated one
stateful xorshift stream while the C++ reference generated each value from its
index. Identical counts were therefore not identical input work. Both sides
now use the same per-index formula, and both Rust programs compare every output
against an independently CPU-sorted vector. The C++ reference separately
copies and compares the complete CUB radix, CUB merge-sort, and Thrust outputs;
sortedness alone is no longer accepted.

The restored implementation must pass the focused GPU executable again before
the aggregate gate is evaluated.

## Remaining upstream product

This review is partial. Exact CUB block/agent scheduling and performance,
caller-provided temporary storage, resource-owned allocation, double buffers,
bit windows, half/bfloat, decomposers/custom records, NaN matrices, 32/64-bit
offset selection, general iterator/adaptor products, segmented sorting, and
all architecture rows remain. Thrust permutation/zip/fancy-iterator ordering
and device-callable sequential forms also remain. None is represented by a
passing stub or CPU fallback.
