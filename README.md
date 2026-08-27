# RCCL: Rust-native CCCL algorithms

RCCL is being rebuilt as a Rust compiler expansion of the Thrust/CUB subset
in NVIDIA CCCL. It is not a C++ API clone and does not contain a runtime
template engine.

The architectural pipeline is:

```text
Rust call-site macro -> operation-specific syn 3 AST -> typed compiler catalog
rccl-thrust-macros  -> CUB semantic lowering
rccl-cub-macros     -> cuda-oxide Rust

rccl-memory -> rccl-cub -> rccl-thrust -> rccl
                 runtime values only
```

Proc-macro expansion performs RCCL's compile-time semantic work and emits only
Rust. All authored schedules, specialization rules, kernel families, and
Thrust-to-CUB choices live in the two macro crates. Ordinary support types are
handwritten Rust in the runtime crates; they are not generated merely because
their CCCL counterparts are templates. cuda-oxide exclusively owns everything
after the expanded Rust source.

## Current rewrite state

The compiler catalogs and coverage ledger now cover:

- all 59 public CUB thread/warp/block/device/iterator headers;
- all 34 top-level Thrust algorithm headers;
- all 19 Thrust iterator headers;
- the required support values, random engines, and memory-resource families
  in the ordinary-runtime coverage ledger rather than the macro catalogs.

The catalogs are typed Rust compiler data and are validated for duplicates,
unknown dependencies, target-row drift, selection errors, and CUB dependency
cycles. Separate operation-specific parsers built with `syn` 3.0.4 and
`syn_derive` retain ordinary Rust expressions at the shallow user macro
boundary, retain TMP-relevant types as explicit macro generics, and reject
missing types or incorrect operand counts. A call such as
`sort!(::<u32>(&exec, &mut keys))` therefore gives the proc macro the same key
fact that drives CCCL's template selection; the deleted untyped form cannot
silently choose merge sort.

The old handwritten registry/JIT/erased-comparator sort and merge layer has
been removed. The current DeviceMerge, DeviceMergeSort, and primitive
integer/`f32`/`f64` DeviceRadixSort operations are safe proc-macro-generated
cuda-oxide Rust. The Policy800 radix implementation selects CUB's exact
single-tile extent before the multi-block one-sweep graph and keeps its stable
blocked-rank-to-striped-store topology.
Thrust sort/merge forms are exported only through those CUB operations. Their
upstream-derived result tests pass, while the equal-work ordering performance
gate intentionally fails until the CUB schedules and throughput match CCCL.
There is no panic stub, CPU fallback, RCCL-authored compiler/loader, symbol
lookup, or runtime "compiler unavailable" error.

`WarpReduce` is the first implemented CUB vertical slice. Its macro keeps the
logical-warp width in the compile-time layer and keeps the reduction operator
as an ordinary Rust closure. The corresponding pinned CCCL overview and the
150-case supported Catch2 matrix run on the GPU through cuda-oxide; see
[rccl-cub/CCCL_WARP_REDUCE_PORT.md](rccl-cub/CCCL_WARP_REDUCE_PORT.md).

The first executable Thrust intersection was binary in-place `transform!` with
a finite constant input range. It runs the exact pinned `constant_iterator.cu`
values and result and has an equal-work performance reference; see
[rccl-thrust/CCCL_CONSTANT_ITERATOR_TRANSFORM_PORT.md](rccl-thrust/CCCL_CONSTANT_ITERATOR_TRANSFORM_PORT.md).

The handwritten runtime surface contains CUB range/storage values, Thrust
range-expression adaptors, and `cuda::Execution`, which borrows a stream,
allocator, and the cuda-oxide module value generated at the program's item
boundary. It contains no algorithm policy, schedule, type map, module family,
or cache.

## Generated-Rust safety rule

Before returning tokens, both proc macros parse their own output back through
`syn::File`. Expansion fails if generated algorithm code contains:

- `Arc`;
- raw pointer types, casts, raw-address expressions, pointer extraction,
  `NonNull`, `UnsafeCell`, `ptr`, or `transmute` plumbing;
- unsafe blocks, signatures, traits, impls, and bare function types;
- FFI blocks, unions, or `static mut`;
- opaque nested macro calls or unaudited attributes. Only the enumerated
  Thrust-to-CUB operation edges are allowed, and each receiving CUB proc macro
  runs its own structural audit on the final Rust.

Generated code uses ownership, borrows, slices, checked disjoint slices,
`MaybeUninit`, closures, and typed cuda-oxide arguments. Any CUDA operation
that cannot be represented safely must first be isolated and reviewed as a
small backend primitive; there is no broad generator waiver. The hand-written
`rccl-memory` CUDA FFI/resource layer remains a separate unsafe/ownership audit
surface and does not weaken this generator rule.

## Rust API direction

Native Rust vocabulary replaces support-library spelling:

| CCCL | RCCL/Rust |
|---|---|
| `span` | kernel slices and checked disjoint slices |
| `array` | `[T; N]` |
| `pair`, `tuple` | Rust tuples or a named `repr(C)` record when names/layout matter |
| TMP/type traits | proc-macro validation, generics, bounds, associated types, const generics |
| `optional`, `expected` | `Option`, `Result` |
| `device_vector` | `rccl_memory::DeviceBuffer` |
| function object/comparator | an ordinary copyable Rust closure |
| iterator facade/categories | read/write/random-access capability traits and zero-copy adaptors |
| execution policy | borrowed stream and allocator runtime values only |

There are no compatibility `Less`, `Greater`, `Comparator`, `DeviceSpan`,
iterator-pair, schedule, policy, or legacy module aliases.

The precise compiler contract is in
[RCCL_COMPILER_DSL.md](RCCL_COMPILER_DSL.md). The exact pinned CUB example and
API-test input is in
[CCCL_CUB_EXAMPLE_TEST_CONTRACT.md](CCCL_CUB_EXAMPLE_TEST_CONTRACT.md). The
complete product inventory, dependency order, examples/tests stage, and
adversarial error-masking review are in
[CCCL_THRUST_CUB_COVERAGE_PLAN.md](CCCL_THRUST_CUB_COVERAGE_PLAN.md).
The current ordering source correspondence, explicit TMP boundary, and seeded
fault evidence are recorded in
[reviews/ordering-cccl-source.md](reviews/ordering-cccl-source.md).

## Verification during the compiler rewrite

```bash
scripts/test-gpu.sh
```

The current gate validates the pinned example/API-test inventory, the two
compiler catalogs, generated-token safety, the runtime/compiler boundary, the
absence of RCCL-owned compiler/runtime-TMP code, formatting, warning-free host
compilation, tests, docs, the three cuda-oxide `WarpReduce` correctness
executables, the exact Thrust constant-iterator executable, the CUB/Thrust
ordering examples, and equal-work performance comparisons against pinned CCCL.
The gate currently stops at the explicit ordering performance failure; that is
the intended evidence that semantic success is not being promoted to full
parity. The latest passing-slice ratios are recorded in
[reviews/current-implemented-performance.md](reviews/current-implemented-performance.md).
