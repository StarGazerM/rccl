# RCCL compile-time DSL

RCCL treats procedural-macro expansion as its AOT boundary:

```text
CCCL semantic family
    -> rccl-cub-macros typed DSL
    -> concrete cuda-oxide device Rust
    -> rccl-thrust-macros typed lowering
    -> ordinary Rust host facade
```

This is not a runtime compilation pipeline. The expanded program contains no
template registry, specialization descriptor, target-policy object, kernel
catalog, erased type map, PTX text, or JIT fallback.

## Crate boundary

```text
                         compile time only
       +----------------------+-------------------------+
       |                                                |
rccl-cub-macros ------------------------------> rccl-thrust-macros
       | emits CUB device Rust                    | emits Thrust facade Rust
       v                                          v
rccl-memory ----------------> rccl-cub -------> rccl-thrust -------> rccl
                         runtime values only
```

The macro crates own every authored compile-time fact:

- the public CCCL-family inventory and dependency graph;
- support-type mappings and source/sink capabilities;
- thread, warp, block, agent, grid, and device kernel families;
- type and callable specialization rules;
- target schedules, launch bounds, tile shapes, unroll factors, shared-memory
  formulas, and fallback selection;
- Thrust-to-CUB lowering rules;
- the finite compiler expansion requested by one generated program.

The runtime crates may own only values that exist while a program runs:

- borrowed contexts and streams;
- allocation owners, buffers, ranges, lengths, and temporary storage;
- opaque CUDA module/function handles emitted by cuda-oxide for the expanded
  program, once cuda-oxide supports the required static generic binding;
- launch dimensions derived from runtime lengths;
- completion and error values.

In particular, `Execution` is only a borrowed stream plus an allocator. It
does not contain an `AotKernelSet`, module family, schedule, architecture,
type map, or specialization cache.

## Two typed DSLs

Both DSLs are private files inside their proc-macro crate. A runtime crate uses
a zero-input expansion macro, so it cannot restate or override compiler facts.
The macro parses its private DSL with `syn` 2 into a typed AST, validates the
whole graph, and emits Rust. Raw token copying is not the compiler model.

### CUB DSL

The CUB DSL declares support mappings, schedules, and primitive families:

```text
support span {
    rust: slice;
    ownership: borrowed;
    capability: contiguous;
}

schedule radix_sort {
    sm80 { block_threads: 256; items_per_thread: 12; min_blocks_per_sm: 2; }
    sm89 { block_threads: 256; items_per_thread: 16; min_blocks_per_sm: 2; }
}

primitive device_radix_sort {
    status: implemented;
    scope: device;
    requires: [block_radix_sort, device_scan];
    schedule: radix_sort;
    kernels: [histogram, prefix, scatter];
}
```

Its typed model is:

```text
CubProgram = Support* x Schedule* x Primitive*
Primitive  = Name x Status x Scope x Dependency* x Schedule? x Kernel*
Schedule   = Name x (Target -> FieldMap)
Scope      = thread | warp | block | agent | grid | device | iterator
Status     = planned | implemented | native_rust | backend_blocked
```

Validation rejects duplicate names, unknown dependencies, dependency cycles,
missing schedules, incomplete target rows, nonliteral schedule values, and a
runtime-selection declaration.

### Thrust DSL

The Thrust DSL declares Rust support vocabulary and lowering rules:

```text
algorithm stable_sort {
    status: implemented;
    form: in_place;
    requires: [device_merge_sort, device_radix_sort];
    select {
        natural_radix_key => device_radix_sort;
        otherwise => device_merge_sort;
    }
}
```

Its typed model is:

```text
ThrustProgram = Support* x Adaptor* x Algorithm*
Algorithm     = Name x Status x Form x Dependency* x SelectionRule*
SelectionRule = CompileTimePredicate -> CubPrimitive
```

Selection predicates are resolved while emitting Rust. They may become
ordinary Rust generic bounds or direct monomorphized calls in the expansion,
but never a runtime enum, match, trait object, `TypeId`, or cache.

## Expansion contract

For an algorithm call shape `C`, inferred Rust type facts `T`, and compiler
target `G`, expansion implements the partial function

```text
lower(C, T, G) = (concrete CUB operation, literal schedule, oxide kernel set)
```

The result is compiler input, not a Rust value. There is no inverse operation
that discovers `T` or `G` at runtime.

The CUB expansion emits:

1. native support/value/range code required by the selected primitives;
2. concrete device helpers and cuda-oxide kernel items;
3. direct literal schedule operands and compiler attributes;
4. generated typed launch code for the compiler-expanded kernels.

Before returning tokens, each proc macro parses its own output back into a
`syn::File` and runs a structural safety audit. Generated algorithm code is
rejected if it contains `Arc`; raw pointer types, casts, extraction, or raw
address expressions; `NonNull`/`UnsafeCell`/`ptr`/`transmute` plumbing; unsafe
blocks, signatures, traits, or impls; FFI blocks, unions, or `static mut`; or
opaque nested macros and unaudited attributes. The normal generated
representation is ownership plus borrows, slices, checked disjoint slices,
`MaybeUninit`, closures, and typed cuda-oxide launch arguments. A genuinely
unavoidable CUDA FFI operation belongs in a tiny hand-audited backend
primitive, not in generated algorithm code and not behind a general codegen
waiver.

The Thrust expansion emits:

1. ordinary Rust functions such as `sort`, `sort_by`, and `merge`;
2. private compile-time lowering code generated from the Thrust DSL;
3. calls to the generated CUB surface using borrowed ranges and runtime
   lengths only.

Generated code must not use `cuda_host::cuda_module` while its generic path
loads merged PTX at runtime. Until cuda-oxide provides static binding for the
generic expanded program, the compiler boundary remains an explicit build
blocker. RCCL does not replace it with a runtime error, PTX loader, JIT link,
symbol-name assembly, or erased ABI argument vector.

## Rust mappings

The macro DSL records mappings but does not generate compatibility wrappers
when native Rust already carries the semantics:

| CCCL concept | Generated Rust contract |
|---|---|
| `span` | kernel `&[T]`, `&mut [T]`, or checked disjoint slice; host allocation borrow |
| `array` | `[T; N]` |
| `pair`, `tuple` | Rust tuples; named `repr(C)` record only for a required ABI |
| type traits/TMP | Rust bounds, associated types, const generics, macro validation |
| `optional`, `expected` | `Option`, `Result` |
| `device_vector` | `rccl_memory::DeviceBuffer` |
| iterator facade/categories | sealed source/sink capability traits |
| fancy iterator | zero-copy indexed source/sink adaptor |
| comparator/function object | ordinary copyable Rust closure returning `Ordering` or the operation's result |
| temporary storage query | typed layout plus a borrowed aligned byte range |
| execution policy | borrowed stream and allocator runtime values only |

There are no `Less`, `Greater`, `Comparator`, `DeviceCallable`, C++ iterator
pair, policy, schedule, or legacy module aliases.

## Non-negotiable rejection rules

Outside the two macro crates, structural tests reject authored occurrences of:

- target schedules or algorithm policy constants;
- specialization/dispatch traits and proof witnesses;
- kernel-family or compiler-manifest structs;
- `TypeId`, `Any`, downcasting, or typed global registries;
- `Arc<CudaModule>`, module caches, or cross-context kernel caches;
- runtime target or type selection;
- PTX strings, PTX bundle merging, driver JIT, nvJitLink, or runtime compiler
  APIs;
- manual kernel symbol construction and raw ABI argument vectors;
- generated `Arc`, raw pointers, broad `unsafe`, or pointer-manufacturing
  casts;
- compatibility aliases for deleted RCCL APIs.

Tests and examples receive no exception. Test-only kernels and fault-injection
variants are declared in a macro crate test DSL and expanded by a test macro.

## Verification stages

1. Parse and validate both complete catalogs with `syn` 2.
2. Snapshot normalized typed ASTs and selected expansions.
3. Use compile-pass/compile-fail consumers to prove ordinary Rust API and
   ownership boundaries.
4. Scan all non-macro Rust sources for forbidden compile-time metadata and all
   generated binaries for PTX/JIT paths.
5. Add public examples and focused unit, boundary, randomized, and negative
   tests for each implemented family.
6. Seed implementation and launch faults and prove the intended tests fail;
   inspect for vacuous passes, stale binaries, shared oracles, skipped work,
   CPU fallbacks, partial-output checks, and swallowed errors.
7. Run the cuda-oxide compiler gate. A missing static generic binding is
   reported as a compiler blocker, never converted into a runnable fallback.
