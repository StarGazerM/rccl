# RCCL compile-time boundary

RCCL does not have, and must not grow, an RCCL compiler. Rustc and
cuda-oxide compile the program. RCCL contributes ordinary generic algorithms,
device kernels, and ordinary host bindings for those kernels.

The selection rule is:

```text
ordinary Rust function / trait / generic
    algorithm control, validation, ownership, type behavior, and policy math

macro_rules!
    private source repetition or deliberate compile-time loop unrolling only

procedural macro
    no RCCL-owned use
```

A C++ template is not evidence that Rust needs a macro. Type behavior maps to
generics and traits. Compile-time integers map to const generics. A closed
primitive family maps to a sealed trait. Architecture is compiler
configuration, never a public type or strategy parameter.

## Architecture selection

There is one source of truth:

```text
cargo oxide build --arch sm_89 ...
                 |
                 +-- exact cfg: cuda_arch="sm_89"
                 +-- cumulative cfg: cuda_arch_ge_80
```

For SM90 and newer, cuda-oxide also emits `cuda_arch_ge_90`; for SM100 and
newer it emits `cuda_arch_ge_100`. RCCL policy code reads only those cfgs.
There is no `CUDA_OXIDE_TARGET` parser, SM type argument, public architecture
object, runtime device-policy registry, or second target flag.

The pinned cuda-oxide patch makes this reproducible: its frontend appends cfgs
derived from the already parsed `--arch` value to `CARGO_ENCODED_RUSTFLAGS` for
both host and device compilation. The exact target remains compiler-owned;
RCCL only reads the resulting cfgs.

The same patch gives every Cargo target a target-specific device-artifact
anchor. A library and one of its examples can therefore contribute separate
device bundles even though they share a package name. The legacy package-only
anchor remains a weak compatibility alias; users do not supply an owner list
or another compilation flag.

A `build.rs` is the wrong boundary. It would run before the final cuda-oxide
invocation, add a second configuration channel, and cannot make an RCCL crate
authoritative over the compiler driver's selected GPU target.

## Public API

Users select an architecture in the compiler command, not in an algorithm
call:

```rust,ignore
keys.sort_unstable(&exec)?;
keys.sort_by(&exec, |left, right| right.cmp(left))?;

first
    .view()
    .merge_by(second.view(), |left, right| left.cmp(right))
    .write_into(output.view_mut(), &exec)?;

let aggregate = rccl_cub::warp_reduce::<i32, 32>(
    thread_value,
    32,
    |left, right| left + right,
);
```

The API carries data types, borrows, capabilities, values, and user behavior.
It does not carry an architecture type, policy type, target marker, generated
kernel set, runtime strategy object, or a public “type hole” whose only purpose
is compiler policy selection.

Rust range capabilities (`ReadRange`, `WriteRange`, `RandomAccessRange`, and
borrowed views) replace C++ iterator categories. `Fn` bounds replace function
objects. Opaque workspace values retain the allocation/lifetime contract.

## Why local declarative macros remain

Stable Rust cannot yet write this fully generic relationship:

```rust,ignore
kernel::<T, { size_of::<T>() }>();
```

Public operations remain ordinary functions. Internally they match the
runtime-observable Rust layout and call a finite set of literal const-generic
specializations:

```rust,ignore
match core::mem::size_of::<T>() {
    4 => launch::<T, 4, { policy_items(4) }>(...),
    8 => launch::<T, 8, { policy_items(8) }>(...),
    _ => Err(unsupported_layout()),
}
```

Local `macro_rules!` definitions only avoid duplicating those literal match
arms, repetitive primitive implementations, or fixed GPU loop-unrolling
steps. They are not exported, do not expand in a user's crate, do not parse a
call, and do not select architecture. Replacing them with handwritten copies
would enlarge the source without changing type checking or compilation.

## Completed proc-macro removals

The refactor started with 51 RCCL proc-macro entry points. The current source
has zero, and the `rccl-cub-macros` crate is deleted.

| Family removed from procedural expansion | Entries removed | Replacement |
|---|---:|---|
| warp reduce | 2 | ordinary public const-generic device functions |
| merge workspace/count | 4 | ordinary generic workspace functions |
| merge operation | 2 | ordinary validation/control plus a concrete kernel module |
| merge-sort workspace/count | 3 | ordinary generic workspace functions |
| merge-sort operations | 7 | ordinary pass loop plus a concrete kernel module |
| radix workspace/count | 6 | sealed `RadixKey` policy and ordinary allocation |
| radix operations | 8 | ordinary pass control plus a concrete kernel module |
| set workspace/count | 1 | ordinary generic policy/allocation |
| set operations | 8 | ordinary validation/control plus index-generic kernel functions |
| histogram operations/workspaces | 8 | cfg-selected const policy and ordinary control |
| program/item generation | 2 | checked-in cuda-oxide modules and ordinary `Program::load` constructors |
| **Total** | **51** | **51 -> 0 entries** |

The remaining local declarative macros fall into three implementation-only
categories:

| Local macro role | Why it remains local source repetition |
|---|---|
| literal layout dispatch arms | stable Rust rejects generic `size_of::<T>()` in const arguments |
| fixed GPU loop unrolling | the device algorithm requires compile-time materialization of each step |
| repetitive primitive/API implementations | each expansion is ordinary checked Rust with the same implementation |

The concrete histogram, ordering, and set modules live in ordinary library
source and are loaded once by `rccl_cub::cuda::Program`. A user neither declares
a generated module nor supplies a module type. The only procedural attributes
in this path are cuda-oxide's own `#[cuda_device::cuda_module]`, `#[kernel]`, and
launch-contract attributes, which define the actual CUDA compilation boundary.

The progress number measures entry-point removal, not semantic completion.
GPU correctness, sanitizer, differential, determinism, code-size, register,
and performance gates remain separate acceptance evidence.

## Compile-unit cost

Whole-program CUDA compilation monomorphizes the reachable kernel catalog.
The original histogram probe combined ten scalar layouts, four counter
layouts, and all behavioral/fault cases in one executable. After 26 minutes
23 seconds it was still compiling and its rustc child had reached roughly
3.4--3.6 GiB RSS. This was a test-binary topology problem, not a reason to put
policy back in a macro.

The same cases are now partitioned into one behavioral probe, four scalar
layout probes, and two counter-layout probes. A coldish SM89 debug build of all
seven from the final post-cleanup source with two jobs completed in 12 minutes
47.76 seconds at 1,586,600 KiB peak RSS in one run and 14 minutes 7.98 seconds
at 1,595,436 KiB in the final run; every resulting GPU executable
passed. A focused existing histogram benchmark built in 2 minutes 33 seconds
at 783,076 KiB peak RSS. These are compile-resource observations, not
cold/warm acceptance, binary-size, runtime performance, or sanitizer
completion claims.

After the complete proc-macro deletion, normal owner-filter-free SM89 builds
and executions passed for the combined ordering probe (6 minutes 4 seconds),
the Thrust set-operation example (1 minute 5 seconds), the Thrust histogram
example (2 minutes 15 seconds), and the constant/transform example (0.57
seconds). Those timings include different amounts of cached work and are not a
controlled cold/warm comparison.

Current measurable refactor status:

| Boundary | Result |
|---|---:|
| RCCL proc-macro entries removed | 51 / 51 (100%) |
| RCCL proc-macro crates remaining | 0 |
| exported CUB/Thrust operation macros remaining | 0 |
| ordinary fixed `Program::load` constructors | 2 / 2 |
| focused post-removal SM89 result probes | 4 / 4 passed |
| public architecture parameters introduced | 0 |

## Ownership boundary

Device algorithms use typed buffers, borrowed ranges, checked disjoint
regions, opaque workspaces, and cuda-oxide prepared launches. They do not use
raw-pointer views, erased comparators, `Arc` module registries, manual kernel
ABI vectors, PTX loaders, or an RCCL runtime compiler.

The three internal unsafe expressions bind the fixed histogram, ordering, and
set CUDA artifacts emitted from the adjacent checked-in modules. The public
`Program::load` constructors are safe; users neither write nor receive an
unsafe loader API.

## Review rule

A host compile proves only Rust type consistency. Each migrated family needs
an upstream-corresponding GPU result test, and completion claims additionally
require the relevant sanitizer, differential, determinism, compile-size,
register/spill, and exclusive-performance gates. Tests may be corrected when
they encode the deleted architecture/TMP design; thresholds and results may
not be weakened to conceal a regression.
