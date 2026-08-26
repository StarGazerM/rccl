# RCCL: Rust-native CCCL algorithms

RCCL is being rebuilt as a Rust compiler expansion of the Thrust/CUB subset
in NVIDIA CCCL. It is not a C++ API clone and does not contain a runtime
template engine.

The architectural pipeline is:

```text
Rust call-site macro -> operation-specific syn 2 AST -> typed compiler catalog
rccl-thrust-macros  -> CUB semantic lowering
rccl-cub-macros     -> cuda-oxide Rust

rccl-memory -> rccl-cub -> rccl-thrust -> rccl
                 runtime values only
```

The proc-macro expansion is RCCL's AOT boundary. All authored target
schedules, specialization rules, kernel families, support mappings, and
Thrust-to-CUB choices live in the two macro crates. The runtime crates do not
store or rediscover that information.

## Current rewrite state

The two complete catalogs now cover:

- all 59 public CUB thread/warp/block/device/iterator headers;
- all 34 top-level Thrust algorithm headers;
- all 19 Thrust iterator headers;
- the required support values, random engines, and memory-resource families.

The catalogs are typed Rust compiler data and are validated for duplicates,
unknown dependencies, target-row drift, selection errors, and CUB dependency
cycles. Separate operation-specific `syn` 2 parsers retain ordinary Rust
expressions at the shallow user macro boundary and reject missing or duplicate
semantic fields.

The old handwritten sort, merge, radix, schedule, module-set, and runtime
loader implementation has been removed. Those algorithms are intentionally
not exported while their safe proc-macro-generated cuda-oxide path is being
written. RCCL does not make the source appear runnable with a panic, CPU
fallback, PTX merge, JIT, symbol lookup, or runtime "compiler unavailable"
error.

The currently generated runtime surface contains safe CUB range/storage
values, safe Thrust range-expression adaptors, and `cuda::Execution`, which
borrows a stream and allocator. It contains no algorithm policy, schedule,
type map, module family, or cache.

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
[RCCL_COMPILER_DSL.md](RCCL_COMPILER_DSL.md). The complete product inventory,
dependency order, examples/tests stage, and adversarial error-masking review
are in [CCCL_THRUST_CUB_COVERAGE_PLAN.md](CCCL_THRUST_CUB_COVERAGE_PLAN.md).

## Verification during the compiler rewrite

```bash
scripts/test-gpu.sh
```

The current gate validates the two compiler catalogs, generated-token safety,
the runtime-only crate boundary, the absence of JIT/runtime-TMP code, formatting,
warning-free host compilation, tests, and docs. It deliberately does not run a
GPU program before an algorithm has a safe generated launch path.
