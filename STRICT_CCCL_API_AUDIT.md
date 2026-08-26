# Strict RCCL compiler-boundary audit

Baseline: NVIDIA CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`.

This audit describes the current rewrite. Removed sort, merge, radix, and warp
implementations are not retained evidence.

## Fixed architecture

| ID | State | Contract |
|---|---|---|
| A01 | fixed | `rccl-cub-macros` is the only author of CUB support mappings, primitive dependencies, target schedules, kernel families, and specialization facts. |
| A02 | fixed | `rccl-thrust-macros` is the only author of Thrust support/adaptor mappings and Thrust-to-CUB selection rules. |
| A03 | fixed | Both private DSLs use custom `syn` 2 typed ASTs and whole-catalog validation. |
| A04 | fixed | `rccl-cub/src/lib.rs` and `rccl-thrust/src/lib.rs` are zero-input expansion sites; a runtime crate cannot restate compiler facts. |
| A05 | fixed | `Execution` borrows a stream and allocator only. It owns no module set, policy, type map, target, schedule, registry, or cache. |
| A06 | fixed | `AotKernelSet`, `TargetSchedule`, build-script schedule generation, `TypeId`/`Any` registries, module caches, and handwritten dispatch traits are deleted. |
| A07 | fixed | Runtime PTX merge, driver JIT, nvJitLink, manual symbol lookup, and raw ABI argument-vector paths are absent. |
| A08 | fixed | There are no compatibility `Comparator`, `Less`, `Greater`, `DeviceSpan`, legacy execution, or stable-sort aliases. |
| A09 | fixed | Old examples, benchmarks, and compile-only fixtures were removed because they could report API coverage without a generated algorithm. |
| A10 | fixed | The compiler gate does not run a GPU program before a safe proc-macro-generated launch path exists. |

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
  after the structural pass.

The generated support surface currently uses ownership, borrows, slices,
`MaybeUninit`, and ordinary `Result`. There is no generator waiver. A future
CUDA operation that genuinely requires unsafe code must first be isolated as
a small reviewed backend primitive; generated algorithms call only its safe
contract.

This guarantee is deliberately scoped to proc-macro output. `rccl-memory` is
the hand-written CUDA FFI/resource boundary and still contains `Arc`, unsafe,
and raw-pointer operations. Those occurrences are red flags requiring a
separate line-by-line ownership/necessity audit; they are not evidence that
generated algorithms may use the same mechanisms. The compiler gate scans
both expansion bodies and both identical structural auditors so this boundary
cannot silently widen.

## Current public state

| Surface | State | Evidence |
|---|---|---|
| complete CUB public catalog | implemented compiler input | `rccl-cub-macros/src/cub.dsl` contains 59 validated public primitives |
| complete Thrust algorithm catalog | implemented compiler input | `rccl-thrust-macros/src/thrust.dsl` contains all 34 algorithm headers |
| complete Thrust iterator catalog | implemented compiler input | the same DSL contains all 19 iterator headers |
| Rust support mappings | partial | CUB expansion emits `KeyValue`, `DoubleBuffer`, `TempStorage`, `DeviceSource`, and `DeviceSink`; Thrust expansion emits checked constant/counting/discard/subrange/reverse/stride/map/map-output/map-in-out/zip/permutation/tabulate adaptors |
| `cuda::Execution` | implemented runtime value | generated borrowed stream and allocator only |
| CUB collectives/device algorithms | missing or backend-blocked | old pointer-heavy code deleted; no safe generated kernels yet |
| Thrust algorithms | backend-blocked | deliberately unexported until their generated CUB lowering exists |
| generated support tests | implemented | macro output tests ownership, bounds, composition, failure propagation, closure capture, and rejected inputs; the zero-stride test exposed and repaired eager `then_some` evaluation |
| public algorithm examples/runtime parity | missing | must be rebuilt after each public algorithm exists |

`backend-blocked` never means a runtime `CompilerUnavailable` error. It means
the function is not exported and cannot be mistaken for a runnable operation.

## Required implementation order

1. Generate safe source/sink/product/storage vocabulary.
2. Generate Thrust adaptors and compile-fail ownership tests.
3. Generate CUB thread, warp, then block primitives without raw-pointer
   algorithm code.
4. Generate agents/grid/device operations and typed temporary-storage forms.
5. Generate Thrust functions only when their CUB lowering and launch path are
   present.
6. Restore examples and focused tests, then seed implementation/launch faults
   to prove those tests fail for the intended reason.

The full dependency order and error-masking checklist are in
`CCCL_THRUST_CUB_COVERAGE_PLAN.md`.
