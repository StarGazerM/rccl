# RCCL shallow compile-time DSL

RCCL's project-owned compile-time boundary is procedural-macro expansion:

```text
ordinary Rust call tokens
    -> rccl-thrust-macros `syn` call AST
    -> typed Thrust semantic selection
    -> rccl-cub-macros `syn` call AST
    -> typed CUB schedule/kernel selection
    -> concrete cuda-oxide Rust
    -> rustc type checking and monomorphization
```

The DSL is shallow and user-call-facing. There is no separate language file,
parser input file, or runtime compiler manifest. Rust expressions, closures,
paths, borrows, and generic arguments stay Rust syntax; the operation macro
changes only the meaning and lowering of the call. Any type fact that changes
a CCCL TMP selection is an explicit macro generic, because a stable proc macro
cannot query rustc's later type inference.

## Public call shape

An operation that needs compile-time CUB/Thrust selection is a macro, like
`println!` is a macro when a normal function call cannot express its required
compile-time work:

```rust,ignore
sort_by!(::<i64>(
    &exec,
    &mut keys,
    |left: &i64, right: &i64| right.cmp(left),
))?;

stable_sort_by_key!(::<i64, u32>(
    &exec,
    &mut keys,
    &mut values,
    |left: &i64, right: &i64| left.cmp(right),
))?;

device_reduce!(
    &exec,
    &values,
    &mut result,
    |left, right| left + right,
    0_i64,
)?;

let aggregate = warp_reduce!(::<i32, 32>(
    thread_value,
    |left, right| left + right,
));

let aggregate = warp_reduce_items!(::<i32, 32, 4>(
    thread_items,
    |left, right| left + right,
));

transform!(
    &exec,
    &mut values,
    repeated,
    |left, right| left + right,
)?;
```

Thrust call shapes expose only semantic operands. Radix scratch, merge-sort
ping-pong buffers, bins, lookback state, counters, and partition arrays are not
public Thrust arguments. Their layout formulas are selected by the proc macro;
their runtime extents come from the borrowed ranges; and the generated Rust
acquires the concrete temporary buffers through `Execution`. Direct CUB keeps
its separate caller-storage contract.

`warp_reduce!`, the shown finite-constant `transform!` intersection, and the
primitive numeric sort/merge slice are implemented and exported. Natural
integer/`f32`/`f64` sort lowers through CUB radix sort; custom ordering lowers
through CUB merge sort. Floating radix transformation, including stable
signed-zero equivalence, is selected from the explicit key type during macro
expansion. The ordering parity gate remains failing until the kernel schedules
and throughput match pinned CCCL. There is no CPU fallback.

`sort_by!` has one mutable range. `stable_sort_by_key!` has separate key and
value ranges and preserves the relative order of equal keys. The `_by_key`
suffix therefore denotes a different semantic input shape, not a comparator
kind. `compare` is an ordinary Rust closure expression; RCCL has no
`Comparator`, `Less`, or `Greater` object hierarchy.

The macro parser stores runtime operands as `syn::Expr`; TMP types and constants
are explicit `syn::Type`/literal syntax. It cannot inspect the inferred type or
array extent of an operand expression. Any such fact needed by lowering is
therefore a template input, not ordinary later type information:
`sort!(::<u32>(...))`, `warp_reduce!(::<i32, 32>(...))`, and
`warp_reduce_items!(::<i32, 32, 4>(...))`. Untyped forms are rejected. Rustc
checks that the explicit TMP facts agree with the concrete expressions; it
does not select the RCCL algorithm or policy.

## What is and is not the DSL

The user-facing macro invocation is the DSL. The files
`rccl-cub-macros/src/catalog.rs` and
`rccl-thrust-macros/src/catalog.rs` are private typed compiler data, not a
language:

```text
CubCall       = operation-specific Rust Type* x Expr* fields
CubPrimitive  = Name x Status x Scope x Dependency* x Schedule? x Kernel* x Contract
Contract      = Capability* x CallableShape x StorageClass
Schedule      = Name x (Target -> literal tuning row)

ThrustCall    = operation-specific Rust Type* x Expr* fields
Algorithm     = Name x Status x Form x CubDependency* x SelectionRule*
SelectionRule = CompileTimePredicate -> CubPrimitive
```

Closed CCCL TMP/CRTP policy becomes enum/struct data and direct lowering in
the macro crate. It does not become a public Rust policy, backend, strategy,
proof type, trait object, `TypeId`, registry, or cache. Rust traits remain only
where callers genuinely supply open behavior or capabilities.

The macro crates validate the complete typed catalogs for duplicate names,
unknown dependencies, dependency cycles, incomplete target schedules,
duplicate capabilities, illegal storage scope, and invalid selection targets.
Each CUB primitive contains exactly one semantic contract by construction;
there is no separately parsed binding table that can drift.

## Crate boundary

```text
                         compile time only
       +----------------------+-------------------------+
       |                                                |
rccl-cub-macros ------------------------------> rccl-thrust-macros
       | operation parsing/catalog/lowering       | call parsing/CUB selection
       v                                          v
rccl-memory ----------------> rccl-cub -------> rccl-thrust -------> rccl
                         runtime values only
```

There is no support-generation macro. Slices, buffers, tuples, range
descriptors, storage layouts, and `Execution` are ordinary handwritten Rust.
Only operation calls that require closed CUB/Thrust selection and concrete
kernel construction are proc macros.

The macro crates own every authored compile-time fact:

- the complete CCCL-family inventory and dependency graph;
- thread, warp, block, agent, grid, and device kernel families;
- type/callable selection rules;
- target schedules, launch bounds, tile shapes, unroll factors, shared-memory
  formulas, and fallback selection;
- Thrust-to-CUB lowering rules; and
- the concrete source generated for each macro invocation.

Runtime crates own only values that exist while a program runs:

- ordinary support types, range capabilities, and range adaptors;
- borrowed contexts, streams, and allocators;
- allocation owners, buffers, views, lengths, and temporary storage;
- launch dimensions derived from runtime lengths; and
- completion and error values.

`Execution` is only borrowed runtime values: stream, allocator where required,
and one statically typed program-local cuda-oxide module. It contains no module
set, schedule, architecture, type map, specialization cache, or compiler policy.

## Expansion contract

For call syntax `C`, macro-visible TMP type facts `T`, and compiler target `G`,
the generated generic program realizes the partial function

```text
lower(C, T, G) = (concrete CUB operation, literal schedule, oxide kernel set)
```

The proc macro sees `C` and `T` directly. It emits source for `G` that rustc and
cuda-oxide type-check and monomorphize. RCCL never discovers a strategy through
`TypeId`, `Any`, a runtime registry, or a fallback branch.

For a device-wide operation, the CUB macro layer emits:

1. concrete cuda-oxide kernel items and operation-specific device helpers;
2. literal schedule operands/compiler attributes;
3. the selected temporary-storage formula and launch graph; and
4. typed launch code with safe generated arguments.

For a cooperative device operation used inside a caller-authored kernel, such
as `warp_reduce!`, it instead emits a safe expression selecting the concrete
tile implementation. The caller's kernel remains ordinary cuda-oxide Rust;
the macro owns scalar representation, tile width, per-thread item count,
closure typing, and operation selection. Rustc checks the resulting expression.

The Thrust layer parses the public algorithm shape and lowers directly to its
CUB operation. It does not recreate C++ iterator-pair or execution-policy
scaffolding when a Rust borrow/range already carries the semantics.

Before returning tokens, each proc macro parses generated output into a
`syn::File` and runs a structural audit. Generated algorithm code is rejected
if it contains `Arc`; raw pointer types, casts, extraction, or raw-address
expressions; `NonNull`/`UnsafeCell`/`ptr`/`transmute`; unsafe blocks,
signatures, traits, or impls; FFI blocks, unions, `static mut`; opaque nested
macros; or unaudited attributes. The exact hidden Thrust-to-CUB operation
edges are admitted only because the receiving CUB macros audit their final
Rust with the identical pass. A genuinely unavoidable CUDA FFI action
belongs in a tiny hand-audited backend primitive, never in generated algorithm
code.

RCCL-generated code performs no compiler invocation, compiler configuration,
manual symbol construction, or erased ABI argument-vector assembly. It emits
pure Rust; cuda-oxide owns every later compilation decision. If cuda-oxide
lacks a required Rust/device binding, that operation stays unexported.

## Rust-native support mapping

Ordinary runtime code uses Rust vocabulary rather than generated compatibility
wrappers:

| CCCL concept | Rust representation |
|---|---|
| `span` | kernel `&[T]`, `&mut [T]`, or a checked disjoint slice |
| `array` | `[T; N]` |
| `pair`, `tuple` | Rust tuples; named records only when field names matter |
| type traits/TMP | bounds, associated types, const generics, macro semantic checks |
| `optional`, `expected` | `Option`, `Result` |
| `device_vector` | `rccl_memory::DeviceBuffer` |
| iterator categories/facade | minimal read/write/random-access range capabilities |
| fancy iterator | zero-copy range-expression descriptor |
| comparator/function object | ordinary Rust closure expression |
| temporary storage query | checked `Layout` and exclusive allocation borrow |
| execution policy | borrowed stream, allocator, and statically typed program-local module values only |

Generated range descriptors contain no host `get`/`write` execution path.
Construction records shape and closures but never evaluates device closures
on the host, preventing CPU behavior from masking a missing GPU lowering.

## Rejection and verification

Outside the macro crates, structural tests reject authored schedules,
specialization/dispatch traits, compiler manifests, `TypeId`/`Any`, module
caches, RCCL-owned compiler/loader paths, manual symbols/ABI vectors, generated `Arc`, raw
pointers, broad unsafe, and compatibility aliases.

Verification proceeds in these stages:

1. Validate both complete typed Rust catalogs.
2. Parse real shallow call syntax with `syn` 3.0.4 and compile-pass/compile-fail
   consumers once each operation is exported.
3. Audit generated syntax and ownership/capability boundaries.
4. Add focused examples, unit tests, GPU reference tests, randomized tests,
   and negative cases for each implemented family.
5. Seed parser, lowering, launch, and output faults and prove the intended test
   fails for the intended reason; inspect for vacuous passes, stale binaries,
   CPU execution, shared oracles, skipped work, partial-output checks, and
   swallowed errors.
6. Run the cuda-oxide compiler/GPU/reference gate. Missing static support is a
   compiler blocker and never a fallback.
