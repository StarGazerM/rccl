# RCCL: Rust-native CCCL algorithms

RCCL is a Rust implementation of the supported Thrust/CUB subset in NVIDIA
CCCL. It is not a C++ API clone and it has no runtime template engine, JIT
layer, type registry, or algorithm dispatcher.

The implementation boundary is:

```text
Thrust method on a typed device range or execution value
  -> ordinary rccl-thrust trait/generic code
  -> backend trait implementation emitted with the program's CUB policy
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
- Thrust sort/stable-sort/by-key, merge/by-key, and four set-operation families
  lowered through their pinned CUB/CUDA substrate.

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

The public calls are ordinary, type-inferred Rust method syntax:

```rust,ignore
use rccl_thrust::prelude::*;

let thrust = rccl_thrust::thrust_program!(&context)?;
let exec = thrust.par(&stream)?;

keys.sort_unstable_by(&exec, |left, right| right.cmp(left))?;

keys.view_mut().with_values(values.view_mut())?
    .sort_by(&exec, |left, right| left.cmp(right))?;

first
    .view()
    .merge(second.view())
    .write_into(output.view_mut(), &exec)?;

let initialized = first
    .view()
    .union_by(second.view(), |left, right| left.cmp(right))
    .write_into(output.view_mut(), &exec)?;
let intersection_size = first
    .view()
    .intersection(second.view())
    .count(&exec)?;
```

`view()` and `view_mut()` carry the input and output borrows through the
composed expression. The terminal method allocates the same private workspace
and calls a statically selected backend trait implementation. `thrust_program!` is the item
generation boundary: declarative scaffolding implements those traits for its
cuda-oxide module, and the nested CUB proc macros emit the exact policy and
launch graphs. Architecture/type policy therefore remains compile time
without making every algorithm call a macro. Architecture is neither a public
operand nor runtime state.

A program that uses only set algorithms may write
`thrust_program!(&context; set_operations)`. This compile-time capability
selection emits only set backend trait implementations; it does not select a
set algorithm or policy, change the method API, or introduce runtime dispatch.
Capabilities compose, so the pinned example uses
`thrust_program!(&context; merge, set_operations)`. The default form remains
the full implemented backend catalog.

Comparator closures are copied into device kernel arguments. A comparator
that captures values therefore uses Rust's `move` closure syntax; the public
set traits require `'static + Copy`, making an accidental host-reference
capture a compile-time error.

The set policy retains CCCL's SM52 and SM60+ branches. Current cuda-oxide
supports basic Rust CUDA modules only from SM70, so SM52 is policy/layout-tested
but cannot yet be device-built; SM70 and native SM89 builds are validated.

Fancy-iterator behavior is expressed as lazy finite range values and method
adaptors, not a C++ iterator category/facade hierarchy:

```rust,ignore
use rccl_thrust::{arithmetic_range, repeat_n};

let odds = arithmetic_range(1_u32, 2, count);
let selected = values.slice(4..end)?.rev().gather(indices);
let paired = odds.zip_eq(repeat_n(10_u32, count))?;
```

`zip` follows Rust iterator shortest-range semantics; `zip_eq`, `slice`, and
`step_by` expose checked variants where a GPU algorithm requires an exact
shape. A descriptor becomes operational only when that algorithm implements
its generated lowering; constructing it never executes a device closure on
the host.

The remaining proc macros are intentionally limited to:

- `#[rccl_cub::program]`: an attribute transformation that only an attribute
  proc macro can perform;
- `warp_reduce!` and `warp_reduce_items!`: scalar representation, tile
  width, and concrete instruction/tree selection;
- `__device_kernels!`: construction of the fixed static cuda-oxide kernel
  family;
- CUB merge/radix operations and workspace formulas whose literal policy
  depends on target architecture and macro-visible type layout.

Fixed program construction and the currently implemented constant-input
transform launch wrapper are `macro_rules!`, not quote-only proc macros.

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
