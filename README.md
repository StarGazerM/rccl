# RCCL: Rust-native CCCL algorithms

RCCL is a Rust implementation of the supported Thrust/CUB subset in NVIDIA
CCCL. It is not a C++ API clone and it has no runtime template engine, JIT
layer, type registry, or algorithm dispatcher.

The implementation boundary is:

```text
Thrust call
  -> rccl-thrust macro_rules! syntax/lifetime wrapper
  -> rccl-cub proc macro only when CUB TMP policy must be selected
  -> concrete cuda-oxide Rust
  -> rustc/cuda-oxide monomorphization

rccl-memory -> rccl-cub -> rccl-thrust -> rccl
                 runtime values only
```

Ordinary Rust support is ordinary Rust. Slices, tuples, arrays, ranges,
allocators, workspaces, closures, and execution borrows are not generated.
A declarative macro is used when a public operation only needs fixed
syntax-directed lowering. A proc macro is retained only when expansion must
choose a target/type-dependent CUB schedule, a concrete device instruction, or
a static kernel family.

## Current implemented slices

The current executable vertical slices include:

- CUB `WarpReduce` for the supported primitive/width matrix;
- CUB DeviceMerge, DeviceMergeSort, and primitive
  integer/`f32`/`f64` DeviceRadixSort paths;
- Thrust binary in-place transform with a finite constant input;
- Thrust sort/stable-sort/by-key and merge/by-key forms lowered through CUB.

The exact coverage ledger remains explicit about missing and partial CCCL
surfaces. Passing tests for these slices is not described as full CUB or
Thrust parity. The ordering result tests pass; the known cuda-oxide
register/throughput gap remains a measured performance issue rather than a
relaxed test.

User kernels are declared with `#[rccl_cub::program]` and bound with
`kernels::program!(&context)`. Device-wide users construct
`cub_program!(&context)` or `thrust_program!(&context)`, then borrow the
program and stream through `on`/`par`. Public calls never name a module or
architecture.

## Macro boundary

The public calls stay shallow Rust syntax:

```rust,ignore
let thrust = rccl_thrust::thrust_program!(&context)?;
let exec = thrust.par(&stream)?;

rccl_thrust::sort_by!(::<i64>(
    &exec,
    &mut keys,
    |left: &i64, right: &i64| right.cmp(left),
))?;
```

Thrust macros are declarative because they only preserve operands, allocate
the required private workspace, and select the corresponding CUB operation.
The inner CUB macro receives the primitive type token unchanged and performs
the real compile-time schedule selection. Architecture is read once inside
the CUB policy boundary; it is neither a public operand nor runtime state.

The remaining proc macros are intentionally limited to:

- `#[rccl_cub::program]`: an attribute transformation that only an attribute
  proc macro can perform;
- `warp_reduce!` and `warp_reduce_items!`: scalar representation, tile
  width, and concrete instruction/tree selection;
- `__device_kernels!`: construction of the fixed static cuda-oxide kernel
  family;
- CUB merge/radix operations and workspace formulas whose literal policy
  depends on target architecture and macro-visible type layout.

Fixed program construction and transform launch wrappers are `macro_rules!`,
not quote-only proc macros.

## Rust API direction

| CCCL | RCCL/Rust |
|---|---|
| `span` | kernel slices and checked disjoint slices |
| `array` | `[T; N]` |
| `pair`, `tuple` | Rust tuples or a named `repr(C)` record when names/layout matter |
| TMP/type traits | generics, trait bounds, associated types, const generics; proc expansion only for closed CUB policy selection |
| `optional`, `expected` | `Option`, `Result` |
| `device_vector` | `rccl_memory::DeviceBuffer` |
| comparator/function object | an ordinary copyable Rust closure |
| iterator facade/categories | read/write/random-access capabilities and zero-copy range expressions |
| CUB `d_temp_storage` | a caller-owned typed opaque workspace |
| execution policy | typed program plus a borrowed stream; `par_in` for an explicit allocator |

There are no compatibility `Less`, `Greater`, `Comparator`, pointer-pair
span, runtime schedule, policy object, or legacy module aliases.

The two exact `unsafe` expressions in the algorithm-facing runtime sources
are cuda-oxide artifact-provenance binds inside the declarative
`cub_program!` and `thrust_program!` wrappers. Algorithm bodies, range
logic, temporary-storage splitting, and generated launch arguments use Rust
ownership, borrows, slices, checked disjoint regions, and typed cuda-oxide
interfaces.

The detailed macro/TMP contract is in
[RCCL_COMPILER_DSL.md](RCCL_COMPILER_DSL.md). The pinned CUB example/API-test
inventory is in
[CCCL_CUB_EXAMPLE_TEST_CONTRACT.md](CCCL_CUB_EXAMPLE_TEST_CONTRACT.md). The
complete dependency-ordered plan, including the related-example/unit-test
stage and adversarial error-masking iteration, is in
[CCCL_THRUST_CUB_COVERAGE_PLAN.md](CCCL_THRUST_CUB_COVERAGE_PLAN.md).

## Verification

```bash
scripts/test-gpu.sh
```

The gate checks the crate boundary, rejects quote-only public proc macros,
keeps architecture selection inside the CUB policy crate, compiles every
registered public example, runs host/GPU correctness tests, and runs the
upstream-corresponding result and performance comparisons.
