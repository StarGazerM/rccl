# RCCL: Rust-native CCCL algorithms

RCCL is a Rust implementation of the supported Thrust/CUB subset in NVIDIA
CCCL. It is not a C++ API clone and it has no runtime template engine, JIT
layer, type registry, or algorithm dispatcher.

The implementation boundary is:

```text
Thrust method on a typed device range or execution value
  -> ordinary rccl-thrust trait/generic code
  -> ordinary rccl-cub validation and dispatch
  -> fixed cuda-oxide kernel modules
  -> rustc/cuda-oxide monomorphization

rccl-memory -> rccl-cub -> rccl-thrust -> rccl
                 runtime values only
```

Ordinary Rust support is ordinary Rust. Slices, tuples, arrays, ranges,
allocators, workspaces, closures, and execution borrows are not generated.
Local declarative macros only reduce repetitive implementation source or
unroll fixed device loops. RCCL owns no proc-macro crate. Target-dependent CUB
policy is ordinary Rust selected by compiler cfg; type behavior uses generics
and sealed capabilities.

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

User kernels use cuda-oxide's `#[cuda_device::cuda_module]` directly.
Device-wide users load `rccl_cub::cuda::Program` or
`rccl_thrust::cuda::Program`, then borrow the program and stream through
`on`/`par`. Public calls never name a module, architecture, or policy type.

## Macro boundary

The public calls are ordinary, type-inferred Rust method syntax:

```rust,ignore
use rccl_thrust::prelude::*;

let thrust = rccl_thrust::cuda::Program::load(&context)?;
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
and calls ordinary CUB functions against the fixed program modules. CUB owns
policy, validation, and launch control. Architecture policy reads only
compiler-emitted cfgs. Architecture is neither a public operand nor runtime
state.

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

All 51 former RCCL proc-macro entries and the `rccl-cub-macros` crate are
removed. Warp reduction, device-wide operations, histogram/transform calls,
workspace construction, and program loading are ordinary functions. Remaining
private `macro_rules!` definitions only spell finite literal dispatch arms,
primitive implementations, or compile-time device-loop unrolling; none are
exported or expanded by users.

## Rust API direction

| CCCL | RCCL/Rust |
|---|---|
| `span` | kernel slices and checked disjoint slices |
| `array` | `[T; N]` |
| `pair`, `tuple` | Rust tuples or a named `repr(C)` record when names/layout matter |
| TMP/type traits | generics, sealed capability traits, associated types, and const generics |
| `optional`, `expected` | `Option`, `Result` |
| `device_vector` | `rccl_memory::DeviceBuffer` |
| comparator/function object | an ordinary copyable Rust closure |
| iterator facade/categories | read/write/random-access capabilities and zero-copy range expressions |
| CUB `d_temp_storage` | a caller-owned typed opaque workspace |
| execution policy | typed program plus a borrowed stream; `par_in` for an explicit allocator |

There are no compatibility `Less`, `Greater`, `Comparator`, pointer-pair
span, runtime schedule, policy object, or legacy module aliases.

The three internal `unsafe` expressions in the algorithm-facing runtime
sources bind the fixed cuda-oxide histogram, ordering, and set artifacts to
their adjacent checked-in module definitions. Public program construction is
safe. Algorithm bodies, range
logic, temporary-storage splitting, and generated launch arguments use Rust
ownership, borrows, slices, checked disjoint regions, and typed cuda-oxide
interfaces.

The detailed compile-time boundary is in
[RCCL_COMPILER_DSL.md](RCCL_COMPILER_DSL.md). The pinned CUB example/API-test
inventory is in
[CCCL_CUB_EXAMPLE_TEST_CONTRACT.md](CCCL_CUB_EXAMPLE_TEST_CONTRACT.md). The
complete dependency-ordered plan, including the related-example/unit-test
stage and adversarial error-masking iteration, is in
[CCCL_THRUST_CUB_COVERAGE_PLAN.md](CCCL_THRUST_CUB_COVERAGE_PLAN.md).

## Reproducible CUDA toolchain

The CUDA compiler contract is intentionally single-source and checked:

| Input | Required identity |
|---|---|
| Rust | `nightly-2026-08-28` from `rust-toolchain.toml` |
| cuda-oxide | `97f8b2b7882f0c15ad9ce9b53abed5553920caa8` |
| compiler patch | `patches/cuda-oxide-read-only-i32x4.patch` |
| patch SHA-256 | `9d92f1e838614af92514b062fc3f0540bf22dee6dcbe1596cfd045d6341f163f` |
| default CUDA/NVVM/libdevice | NVIDIA HPC SDK 25.5, CUDA 12.9, as recorded in `.cargo/config.toml` |

`scripts/setup-cuda-oxide.sh` is the only compiler setup flow. The canonical
patch contains the architecture cfg, read-only vector, safe bounds-proved
`ThreadTile`, and ordering/warp-vote semantic and code-shape fixture changes.
Setup verifies its checksum before applying it to the pinned cuda-oxide commit.
The root manifest's three vendored crate overrides and every cuda-oxide Git
dependency remain tied to that same commit; `Cargo.lock` records no other
cuda-oxide source revision. CUDA path defaults remain overridable through the
environment.

Run the identity check and its deterministic rejection fixtures from any
working directory:

```bash
scripts/check-oxide-contract.sh
scripts/test-oxide-contract-negative.sh
```

`scripts/check-clean-clone.sh` clones only a selected committed tree (default
`HEAD`) into a temporary directory, checks the contract and negative fixtures,
and runs locked metadata and workspace checks there. It never copies or
injects working-tree or untracked files, and it does not mutate the caller's
checkout. Therefore a newly added gate cannot claim a clean-clone pass until
the gate and its dependencies are committed.

`scripts/check-sm86-compile.sh` compiles every workspace example registered
behind `cuda-oxide-example` for `sm_86`. This is explicitly compile-only
evidence: it neither executes the binaries nor establishes SM86 correctness or
performance.

## Verification

```bash
scripts/test-gpu.sh
```

The gate checks the crate boundary, rejects RCCL proc-macro crates and exported
operation macros, keeps architecture selection inside the CUB policy crate, compiles every
registered public example, runs host/GPU correctness tests, and runs the
upstream-corresponding result and performance comparisons.
