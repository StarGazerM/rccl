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
cycles. Separate operation-specific parsers built with `syn` 3 and
`syn_derive` retain ordinary Rust expressions at the shallow user macro
boundary and reject incorrect operand counts.

The old handwritten sort, merge, radix, schedule, module-set, and runtime
loader implementation has been removed. Those algorithms are intentionally
not exported while their safe proc-macro-generated cuda-oxide path is being
written. RCCL does not make the source appear runnable with a panic, CPU
fallback, RCCL-authored compiler/loader, symbol lookup, or runtime "compiler
unavailable" error.

`WarpReduce` is the first implemented vertical slice. Its macro keeps the
logical-warp width in the compile-time layer and keeps the reduction operator
as an ordinary Rust closure. The corresponding pinned CCCL overview and
focused Catch2 cases run on the GPU through cuda-oxide; see
[rccl-cub/CCCL_WARP_REDUCE_PORT.md](rccl-cub/CCCL_WARP_REDUCE_PORT.md).

The handwritten runtime surface contains CUB range/storage values, Thrust
range-expression adaptors, and `cuda::Execution`, which borrows a stream and
allocator. It contains no algorithm policy, schedule, type map, module
family, or cache.

## Generated-Rust safety rule

Before returning tokens, both proc macros parse their own output back through
`syn::File`. Expansion fails if generated algorithm code contains:

- `Arc`;
- raw pointer types, casts, raw-address expressions, pointer extraction,
  `NonNull`, `UnsafeCell`, `ptr`, or `transmute` plumbing;
- unsafe blocks, signatures, traits, impls, and bare function types;
- FFI blocks, unions, or `static mut`;
- opaque nested macro calls or unaudited attributes.

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

## Verification during the compiler rewrite

```bash
scripts/test-gpu.sh
```

The current gate validates the pinned example/API-test inventory, the two
compiler catalogs, generated-token safety, the runtime/compiler boundary, the
absence of RCCL-owned compiler/runtime-TMP code, formatting, warning-free host
compilation, tests, docs, and the cuda-oxide `WarpReduce` executable. Other
algorithms remain absent until they have the same safe generated path and
upstream-corresponding evidence.
