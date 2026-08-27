# Strict RCCL compiler-boundary audit

Baseline: NVIDIA CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`.

This audit describes the current rewrite. Deleted registry/JIT/erased-comparator
implementations are not retained as evidence for the current macro-generated
sort, merge, radix, or warp code.

## Fixed architecture

| ID | State | Contract |
|---|---|---|
| A01 | fixed | `rccl-cub-macros` is the only author of CUB primitive dependencies, target schedules, kernel families, and specialization facts. Ordinary CUB support vocabulary is handwritten in `rccl-cub`. |
| A02 | fixed | `rccl-thrust-macros` is the only author of Thrust-to-CUB selection rules. Ordinary range-adaptor and execution values are handwritten in `rccl-thrust`. |
| A03 | fixed | Both user-facing operation DSLs are shallow call-site macros parsed as operation-specific `syn` 3.0.4 ASTs. Runtime operands remain `syn::Expr`; every type that changes TMP selection is a macro-visible `syn::Type`. The complete inventories are private typed Rust catalogs, not separate language files. |
| A04 | fixed | `rccl-cub` and `rccl-thrust` contain ordinary Rust runtime values but no target schedule, kernel family, specialization table, or compiler registry. |
| A05 | fixed | `Execution` borrows a stream, allocator, and one statically typed cuda-oxide module value. It owns no module set, policy, type map, target, schedule, registry, or cache. |
| A06 | fixed | `AotKernelSet`, `TargetSchedule`, build-script schedule generation, `TypeId`/`Any` registries, module caches, and handwritten dispatch traits are deleted. |
| A07 | fixed | RCCL owns no compiler invocation/configuration, manual symbol lookup, or raw ABI argument-vector path; it emits Rust and cuda-oxide owns compilation. |
| A08 | fixed | There are no compatibility `Comparator`, `Less`, `Greater`, `DeviceSpan`, legacy execution, or stable-sort aliases. |
| A09 | fixed | An example is registered only with a generated GPU operation, a pinned upstream correspondence, and result validation. The exact pinned input is independently tracked as 19 programs, 378 embedded example regions, 49 API-test files, and 105 exercised public methods. |
| A10 | fixed | The compiler gate does not run a GPU program before a safe proc-macro-generated launch path exists. |
| A11 | fixed | Every one of the 59 typed CUB catalog entries contains exactly one operation contract declaring its range capabilities, callable shape, and storage class. |
| A12 | fixed | Closed TMP/CRTP backend policy is lowered by the DSL; generated Rust has no policy/backend/strategy/proof type hierarchy. |

## Generated Rust safety

Each proc macro parses its output back as a `syn::File` before returning it.
Expansion rejects:

- `Arc` paths;
- raw pointer types and casts, `&raw`, `NonNull`, pointer extraction such as
  `as_ptr`, and `ptr`/`transmute` paths;
- unsafe blocks, all unsafe function/method signatures, traits, impls, and
  bare function types;
- FFI blocks, unions, `static mut`, and `UnsafeCell` plumbing;
- opaque nested macro calls and unaudited attributes that could generate code
  after the structural pass. The exact hidden Thrust-to-CUB kernel/operation
  macro edges are allowlisted; the receiving CUB macros independently audit
  their final Rust with the identical pass.

Support code is not generated. A future generated CUDA operation that
genuinely requires unsafe code must first be isolated as a small reviewed
backend primitive; generated algorithms call only its safe contract. There is
no broad generator waiver.

This guarantee is deliberately scoped to proc-macro output. `rccl-memory` is
the hand-written CUDA FFI/resource boundary and still contains `Arc`, unsafe,
and raw-pointer operations. Those occurrences are red flags requiring a
separate line-by-line ownership/necessity audit; they are not evidence that
generated algorithms may use the same mechanisms. The compiler gate runs both
identical structural auditors so this boundary cannot silently widen.

## Current public state

| Surface | State | Evidence |
|---|---|---|
| complete CUB header catalog | implemented compiler input | `rccl-cub-macros/src/catalog.rs` contains 59 validated typed primitives and exactly one semantic contract per primitive; its 105 typed exercised methods are independently matched to pinned examples/API tests |
| complete Thrust algorithm catalog | implemented compiler input | `rccl-thrust-macros/src/catalog.rs` contains all 34 algorithm headers and typed CUB selections |
| complete Thrust iterator catalog | implemented compiler input | the same typed catalog contains all 19 iterator/adaptor headers |
| pinned CUB example/API-test input | implemented audit input | `CCCL_CUB_EXAMPLE_TEST_CONTRACT.md` and its checker track all 19 programs, 378 embedded regions, 49 API-test files, and 105 exercised methods |
| shallow operation syntax | implemented parser boundary | `call.rs` in both macro crates retains positional Rust `Expr` nodes and macro-visible ordering `Type` nodes; parser tests reject missing TMP types, wrong arity, and deleted named-field syntax |
| Rust support mappings | partial | handwritten `rccl-cub` code defines `KeyValue`, generic `DoubleBuffer`, checked `TempStorageLayout`/`TempStorage`, and range capabilities; `RepeatN` is operational only in the implemented transform intersection, while other handwritten descriptors remain non-operational planning vocabulary |
| `cuda::Execution` | implemented runtime value | borrows the stream, allocator, and program-local typed cuda-oxide module; contains no runtime compiler or dispatch metadata |
| CUB collectives/device algorithms | partial | `warp_reduce!` and `warp_reduce_items!` cover the documented overview and 150 supported pinned Catch2 rows; DeviceMerge, DeviceMergeSort, and primitive integer/`f32`/`f64` DeviceRadixSort have safe GPU result coverage, including bitwise signed-zero checks, but the ordering performance gate intentionally fails until their CCCL schedules are reproduced |
| Thrust algorithms | partial | binary in-place `transform!`, primitive numeric sort/stable-sort/by-key forms, merge, and merge-by-key are GPU exported; every ordering form lowers through its actual CUB dependency, while unsupported half/bfloat/decomposer radix cases fail closed |
| support tests | implemented | ordinary unit tests cover extent/bounds composition, capability propagation, invalid layouts, and prove adaptor construction never executes user closures on the host; test-only descriptors prevent host slices from masquerading as device memory |
| public algorithm examples/runtime parity | partial | pinned WarpReduce, constant-iterator transform, merge, merge-by-key, sort, stable-sort, and by-key executables validate exact upstream-derived results; equal-work ordering references are in the default gate and currently fail with explicit measured ratios rather than masking missing performance |

`backend-blocked` never means a runtime `CompilerUnavailable` error. It means
the function is not exported and cannot be mistaken for a runnable operation.

## Required implementation order

1. Complete safe handwritten read/write/random-access/product/storage vocabulary.
2. Complete handwritten Thrust adaptors and compile-fail ownership tests.
3. Generate CUB thread, warp, then block primitives without raw-pointer
   algorithm code.
4. Generate agents/grid/device operations and typed temporary-storage forms.
5. Generate Thrust functions only when their CUB lowering and launch path are
   present.
6. Restore examples and focused tests, then seed implementation/launch faults
   to prove those tests fail for the intended reason.

The full dependency order and error-masking checklist are in
`CCCL_THRUST_CUB_COVERAGE_PLAN.md`.
