# SRDatalog and HISA Rust GPU Migration TODO

Last reviewed: 2026-09-04

This is the canonical workload-driven tracker for replacing the CUDA C++,
Thrust, CUB, and cuCollections layer used by SRDatalog and HISA with Rust RCCL
and cuda-oxide. It does **not** claim to replace the NVIDIA CUDA driver, PTX,
NVVM, or the GPU runtime.
For the semantic capability classification and the audited reachable external
callsites, use [`CCCL_RUST_CAPABILITY_CHECKLIST.md`](CCCL_RUST_CAPABILITY_CHECKLIST.md)
as the source of truth. This is a focused companion to this tracker; it does
not rewrite the evidence log or checkbox state.

## Update rules

- Every RCCL task related to this research migration must review this file.
- Update the `Last reviewed` date, affected checkboxes, and evidence log in the
  same change.
- A checked item means its complete acceptance evidence is recorded below.
- Compilation-only, focused, interrupted, or shared-GPU validation is not a
  correctness, safety, parity, determinism, or performance completion claim.
- Preserve exact Datalog set semantics. Do not accept probabilistic tuple loss.
- Keep generic GPU mechanisms in RCCL and SRDatalog-specific relational,
  semiring, provenance, fixpoint, and code-generation logic in SRDatalog.

## Current baseline

- [x] Audit production HISA/SRDatalog Thrust, CUB, and cuCollections usage.
- [x] Establish Rust-native ownership direction: stream-bound buffers, borrowed
  shared views, affine mutable views, and typed temporary storage.
- [ ] Complete `rccl-cuco` workload coverage for SRDatalog. A packed `u32`/`i32`
  static-map slice and minimal host-orchestrated segmented dynamic-map slice are
  implemented, but Bloom filter parity, bit-compatible `XxHash64`,
  exact `StaticSet<Row<N>>` tuple set coverage, and forced-collision proof remain
  open.
- [ ] Demonstrate an end-to-end HISA or SRDatalog workload using only the Rust
  application/runtime layer.

## P0: execution safety and reproducibility

- [x] Pin and commit the cuda-oxide source/fork and all backend changes.
- [x] Pin the Rust nightly and CUDA/NVVM/libdevice compatibility matrix.
- [ ] Add a safe `Pending<'a>` or equivalent completion object that retains
  input/output borrows, callable state, module state, and temporary storage
  until stream completion.
- [ ] Define safe drop, wait, error, and explicit unsafe detach behavior for
  pending GPU work.
- [ ] Add owned CUDA event wrappers and cross-stream wait/dependency APIs.
- [ ] Prove mutable outputs cannot be reused while an operation is pending.
- [x] Add clean-clone build and test coverage without undeclared local files.
- [ ] Add SM86 compile and native correctness coverage for the HISA target.

## P0: ranges and linear algorithms

- [ ] Finish Rust-style lowering and composition coverage for zero-copy counting, constant, transform, zip, permutation/gather,
  discard, offset, and writable-output ranges (C++-facade/API cloning is not the target; initial Rust-style range APIs exist, remaining work is lowering/coverage breadth).
- [ ] Implement `for_each` as a Rust-style method/free-function API plus SRDatalog-oriented backend lowering (no C++ linear facade is required).
- [ ] Extend Rust-style `DeviceTransform`/`tabulate` backend coverage for SRDatalog beyond current repeated-value/in-place slice paths.
- [x] Extend Rust-style `copy`, `copy_n`, `fill`, `sequence`, and `tabulate` coverage to the SRDatalog range/composition behavior (validated range slicing is the Rust `copy_n` form).
- [x] Implement `gather` and the permutation-range bounds contract.
- [x] Implement inclusive and exclusive device scan coverage in Rust method/free-function lowering.
- [x] Implement workload-needed sorted adjacent-key `reduce_by_key` lowering with explicit key equivalence and the concrete SRDatalog provenance additions.
- [ ] Extend `DeviceReduce` beyond contiguous `u32`/`i32`/`f32` sum when another concrete workload requires it; broader reduction performance parity remains open.
- [x] Implement copy-if/device select as Rust-style coverage/lowering (`filter_map`-style path), not C++ facade parity.
- [x] Implement unique and unique-copy as Rust-style coverage/lowering (API shape parity, not C++ clone).
- [x] Implement stable partition.
- [ ] Validate contiguous and fancy-range forms with 32-bit and 64-bit counts.

## P0: ordered relations and HISA lookup

- [x] Implement the audited workload AoS record slice: padding-free lexicographic `u32` rows of arity 2 through 6, including stable provenance payload sorting.
- [ ] Extend comparison sorting to arbitrary caller-defined `#[repr(C)]` records and directly zipped structure-of-arrays columns when a concrete migration caller requires those forms.
- [ ] Complete stable sort-by-key with borrowed values and affine output views.
- [ ] Complete merge and merge-by-key for the required fancy ranges.
- [ ] Complete set union and set difference for SRDatalog relation keys.
- [ ] Implement device lower-bound and binary-search operations used by HISA.
- [ ] Close current ordering correctness, code-shape, register, and exclusive
  performance gates on every supported target architecture.
- [ ] Backend-blocked on pinned cuda-oxide: eliminate explicit local-array lowering in the supported `Row<6>` key-only block/merge kernels; matched NVCC/CUB scalarizes the same records with zero local frame, while the Rust kernels now have zero ptxas spills but retain 96-byte block and 48-byte merge frames.

## P0: cuCollections research subset

- [ ] Implement bit-compatible `XxHash64` for bitwise-stable `#[repr(C)]`
  HISA key records, with pinned host/device test vectors.
- [ ] Implement an owning stream-bound Bloom filter with clear, bulk add, bulk
  contains, and a read-only device reference.
- [ ] Match the required atomic scope, filter policy, capacity calculation, and
  no-false-negative semantics.
- [ ] Test Bloom false positives as permitted and false negatives as forbidden.
- [x] Implement an exact concurrent `StaticSet<Row<N>>`, or equivalent hash
  table that stores enough tuple data to perform full equality checks.
- [ ] Replace SRDatalog's hash-only dedup rule that treats equal 64-bit hashes
  as equal tuples.
- [x] Add forced-collision tests proving distinct tuples are never discarded.
- [x] Define and test fail-open behavior for probe or capacity exhaustion.

## P0: SRDatalog delta and provenance semantics

- [ ] Specify the custom delta operation over sorted key/value relations:
  compare new keys with full keys, merge matching provenance, mutate full
  provenance safely, and emit only new or changed tuples.
- [ ] Decide and document the boundary: generic merge/scan mechanisms in RCCL,
  semiring/provenance match behavior in SRDatalog.
- [ ] Implement the operation without exposing aliased mutable full values.
- [ ] Differential-test keys, emitted deltas, in-place full provenance, and
  fixpoint termination against the current implementation.
- [ ] Specify deterministic ordering for non-associative or floating provenance
  operations instead of relying on incidental GPU scheduling.

## P1: reusable CUB substrate

- [ ] Publish Rust-native BlockLoad and BlockStore contracts.
- [ ] Publish BlockScan, including prefix callback behavior.
- [ ] Publish BlockRadixSort for SRDatalog's small relation paths.
- [ ] Add typed device-wide temporary-storage query/execute APIs independent of
  the Thrust execution facade.
- [ ] Add resource-owned convenience paths for device-wide primitives.
- [x] Implement the first concrete workload segmented reduction (`reduce_by_key`) with an acceptance test.
- [ ] Implement segmented scan when recursive aggregation first requires it.

## P1: SRDatalog Rust integration

- [ ] Add an SRDatalog compiler backend that lowers MIR to Rust/cuda-oxide
  kernels or another documented Rust-owned compiled artifact.
- [ ] Cache generated artifacts by program, compiler/backend version, and SM.
- [ ] Provide opaque Rust-owned C ABI handles for Python instead of exposing
  untracked raw device ownership.
- [ ] Preserve current Python build/load error reporting and relation schemas.
- [ ] Port custom join, search, semiring, fixpoint, and JIT kernels in the
  SRDatalog repository rather than placing domain logic in RCCL.

## Required acceptance gates

- [ ] Differential HISA parity for build, lookup, duplicate checking, join
  counts, prefix offsets, and materialized rows.
- [ ] Differential SRDatalog parity at every relation and fixpoint boundary,
  including provenance values and changed/full relation state.
- [ ] Randomized and adversarial empty, singleton, duplicate-heavy, skewed,
  maximum-capacity, overflow, hash-collision, and large-offset cases.
- [ ] Compute Sanitizer memcheck and racecheck for all concurrent primitives.
- [ ] Compile matrix for all claimed SM targets and native execution where
  hardware is available, including SM86.
- [ ] Clean cold and warm builds from a fresh checkout.
- [ ] Exclusive-GPU performance comparison with identical allocation,
  synchronization, input, and warmup boundaries.
- [ ] Record compile time, binary size, registers, spills, temporary storage,
  throughput, and end-to-end workload time.

## Explicit non-goals for the initial research cut

- Do not port cuCollections `dynamic_map`, multimap, multiset,
  HyperLogLog, or roaring bitmap without a concrete SRDatalog/HISA use site.
- Do not port unrelated Thrust random, complex, host-policy, or C++
  compatibility machinery merely for header-count parity.
- Do not duplicate Rust slices, tuples, `Option`, `Result`, ownership, or
  allocator concepts with C++-shaped compatibility wrappers.
- Do not claim that Rust replaces the NVIDIA driver, PTX, NVVM, or CUDA
  platform dependencies.

## Evidence log

| Date | Scope | Status change | Evidence and remaining limitation |
|---|---|---|---|
| 2026-09-01 | Workload audit | Initial tracker created | Production source inventory identified the HISA Bloom/xxHash64 contract, SRDatalog's scan/reduce/select/block dependencies, the custom provenance-aware delta operation, multi-stream safety requirements, and the hash-only dedup correctness defect. Current RCCL ordering support remains partial/performance-incomplete; most linear, scan, reduce, compaction, block, and cuCollections requirements remain open. |
| 2026-09-03 | SRDATALOG tracker refresh | No checkbox changed; no new verification run | Missing-API checklist rows were refined to distinguish existing Rust-style API shape from still-missing Rust backend/lowering/composition coverage in `P0: ranges and linear algorithms` (e.g., for_each/scan/select/reduce_by_key/unique), while preserving cuco missing-item scope.
| 2026-09-01 | CUB/Thrust histogram and cuda-oxide root-cause isolation | No checkbox changed; histogram performance remains open | On an exclusive RTX 6000 Ada, `scripts/test-histogram.sh` passes the complete pinned result, regression, and injected-fault matrix; Compute Sanitizer reports 0 memcheck errors and 0 racecheck hazards. The unchanged five-round 1.001 gate fails honestly at 1.186523/1.053700/1.006392/1.047995 for single-even/single-range/multi-even/multi-range. `cargo oxide build --arch sm_89 --features cuda-oxide-example -- --release --package rccl-thrust --example histogram` and the resulting example pass. PTX/SASS inspection shows the required fully safe slice-backed `ThreadTile` is materialized by cuda-oxide as a 1,224-byte local stack frame with 76 registers in single-even. Earlier compact-address diagnostics preserved exact CUB load/atomic topology and reduced the remaining gaps, but used an internal representation forbidden by the Rust safety contract; branch folding removed its spill without changing timing, and batched timing ruled out host launch/event overhead. Closure therefore requires cuda-oxide to lower the lifetime-bearing bounds proof to compact SSA address/count values, followed by safe-range and predicate/control-flow optimization; raw pointers, unchecked indexing, a different algorithm, or a relaxed gate are not accepted. Host evidence also passes `cargo test --workspace --all-targets`, strict workspace clippy/docs, all vendored cuda-host/cuda-macros/cuda-device suites, rustfmt, and `git diff --check`. The old aggregate-gate false positive from the word `unsafe` in a documentation sentence was removed without weakening the code check. |
| 2026-09-03 | Compiler-owned architecture selection and complete RCCL proc-macro removal | No checkbox changed; architecture reproducibility and ordering acceptance gates remain open | Removed all 51 former RCCL proc-macro entries and deleted the entire `rccl-cub-macros` crate, along with the exported CUB/Thrust operation-macro layers. Warp, merge, merge-sort, radix, set, histogram, workspace sizing, validation, allocation, program loading, and pass control now use ordinary functions, const functions, sealed traits, generics, checked-in kernel modules, and private repetition/unrolling-only `macro_rules!`. Public APIs expose no architecture, policy, module, generated kernel-set, or compiler-only type parameter. Architecture comes only from `cargo oxide build --arch`; the repository patch emits exact and cumulative architecture cfgs into host and device rustc invocations. The cuda-oxide patch also makes target-specific v2 device-artifact anchors the default, with a weak legacy alias, preventing a same-package library/example anchor collision that previously dropped `device_radix_sort_init` from the final executable. Normal builds require no device-code owner list. Removed the C++-shaped `ReadOnlySlices`/`ReadOnlyRange` descriptor and its special launcher/compiler handling; retained only the genuine device cache-load intrinsic. Disabled Rust host ThinLTO because cuda-oxide deliberately embeds native `.oxart` rlib members without LLVM bitcode; device NVVM optimization remains enabled. The patch applies to pinned cuda-oxide `97f8b2b7`, `scripts/setup-cuda-oxide.sh /tmp/rccl-cuda-oxide-patch-check` rebuilt the compiler successfully, and `cargo +nightly-2026-08-28 test --manifest-path /tmp/rccl-cuda-oxide-patch-check/Cargo.toml -p cargo-oxide apply_output_mode_sets -- --nocapture` passed 3 tests. Strict host validation passed: `RUSTFLAGS='-D warnings' cargo check --workspace --all-targets`; `RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets -- -D warnings`; `RUSTFLAGS='-D warnings' cargo test --workspace --no-fail-fast` (63+5 rccl-cub, 14+29 rccl-memory, 11+4 rccl-thrust tests plus 6 compile-fail doctests); and `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps`. Vendored cuda-macros (102 tests), cuda-host (69 passed, 1 toolkit-path live test ignored, 7 doctests), and cuda-device (104 tests, device-only doctests ignored) suites passed. `RCCL_STRUCTURE_ONLY=1 scripts/test-gpu.sh` passed the pinned source contract (19 programs, 378 regions, 49 API tests, 105 exercised methods), compiler-cfg/API boundary checks, shell syntax, rustfmt, and `git diff --check`. Focused owner-filter-free SM89 build/runtime probes passed: combined merge/merge-sort/radix ordering in 6m04s with `device_radix_sort_init` present in the final binary; Thrust set operations in 1m05s; Thrust histogram in 2m15s; and constant/transform in 0.57s. These timings are not a controlled cold/warm comparison. Remaining limitations: this change did not run the complete sanitizer, differential-parity, determinism, register/spill, SM86, clean-clone, binary-size, or exclusive-performance matrix, so no corresponding acceptance checkbox is complete. |
| 2026-09-03 | Thrust generation basics and Rust facade polish | No checkbox changed; linear/reduce workload rows remain open | Added additive `DeviceSlice`/`DeviceSliceMut` aliases plus `as_device_slice`/`as_device_slice_mut`, Rust method syntax for the implemented repeated-value transform, and the first DeviceTransform-family generation slice: full-buffer `fill`, `tabulate`, `sequence`, `sequence_from`, and `sequence_by`. `tabulate` follows pinned Thrust CUDA's counting-index transform dependency; `sequence` composes through tabulate like generic Thrust. `cargo check --workspace`, `cargo check --workspace --examples --features cuda-oxide-example`, `cargo test --workspace api_tests`, `scripts/test-thrust-constant-iterator.sh`, and `scripts/test-thrust-histogram.sh` pass. The following evidence row records the first CUB `DeviceReduce::Sum`/Thrust `reduce` slice; broader reduce families remain open. |
| 2026-09-03 | CUB DeviceReduce::Sum and Thrust reduce slice | No checkbox changed; scan/select/reduce-by-key and custom reductions remain open | Added a fixed `reduce_kernels` module and CUB-level `DeviceReduce::Sum` workspace/facade for contiguous `u32`, `i32`, and `f32`. The implementation preserves pinned CUB's single-tile versus two-pass split, SM occupancy multiplied by subscription factor 5 for temporary partial capacity, strip-mined first-pass tile distribution, and synchronous one-element device-result readback. `rccl_thrust::reduce` and `DeviceReduceExt::reduce` now expose the default Thrust plus/zero-init path, with single-tile, two-pass, and empty-buffer assertions in the pinned `constant_iterator` Rust port. Verified with `cargo fmt --all`, `cargo check --workspace`, `cargo check --workspace --examples --features cuda-oxide-example`, `cargo test --workspace` (140 passed), `cargo test --workspace api_tests` (17 passed), `RCCL_STRUCTURE_ONLY=1 scripts/test-gpu.sh`, `scripts/test-thrust-constant-iterator.sh`, and `compute-sanitizer --tool memcheck target/release/examples/constant_iterator` (0 errors). |
| 2026-09-03 | Runpod reduce parity checkpoint | No checkbox changed; no new API coverage claim | Final exclusive-GPU evidence was collected on Runpod `vj3icxcclyqg8f` for the implemented contiguous `DeviceReduce::Sum` slice at 67,108,864 items: RCCL/CCCL median ratios were `u32=1.000365`, `i32=1.000784`, and `f32=1.000182` with `temp_bytes=17407`; this is timing parity evidence only and does **not** claim full reduce, scan, or `reduce_by_key` parity. |
| 2026-09-03 | Restored `rccl-cuco` static map | No checkbox changed; cuCollections workload item remains partial | Recovered the prior `rccl-cuco` packed static-map implementation from a local backup snapshot, added it back to the workspace and umbrella `rccl::cuco` export, moved kernel ABI adapters local to `rccl-cuco`, and added `StaticMapConfig` plus Rust-style `insert`, `get`, `contains_key`, and `remove` wrappers over the existing checked kernels. Restored pinned cuCollections C++ reference and correctness/performance scripts; static-map insertion performance parity was then rerun on the exclusive Runpod pod. Broader cuCollections work remained partial pending the subsequent dynamic-map slice and workload-specific Bloom/exact-row-set work. |
| 2026-09-03 | SRDatalog hash-only dedup integration search | No checkbox changed; integration blocked outside this repository | The local RCCL checkout contains no SRDatalog Nim/JIT source or production HISA callsite (`**/*.nim`, `**/join_include.cuh`, `**/jit_complete_runner.nim`, and `**/jit_emit_helpers.nim` all have no matches); its only workload references are tracker text and cuCollections oracle/examples/scripts. The tracker-referenced external checkout has the real defect in `<SRDATALOG_CHECKOUT>/src/srdatalog/codegen/target_jit/jit_complete_runner.nim`: generated `DedupTable::try_insert` lines 165-205 stores only a 64-bit FNV hash and line 201 treats `old == h` as duplicate, while `jit_emit_helpers.nim` lines 135-143 guards observable emission. |
| 2026-09-03 | Minimal `rccl-cuco` dynamic map | No checkbox changed; cuCollections workload item remains partial | Added `DynamicMapConfig`/`DynamicHashMap` as a real non-empty vector of `StaticHashMap` segments with host-side growth/rehash orchestration. The slice reuses static-map kernels for segment insert, lookup, erase, retrieve, clear, and rehash; `insert` grows by appending a segment when the active segment would exceed max load, `get`/`contains_key` search active segments newest-to-oldest, and `remove`/`clear`/`reserve`/compaction `rehash` are implemented. `rccl-cuco/examples/dynamic_map.rs` covers growth, lookup across segments, replacement, remove across segments, reserve/grow, rehash compaction, and clear. This remains partial versus cuCollections `dynamic_map`: no device-side concurrent segment-selection/growth publication or upstream policy/equality breadth. |
| 2026-09-03 | Wave 0 reproducibility gates | No checkbox changed; clean committed-tree and complete SM86 compile evidence remain open | Added a repository-rooted cuda-oxide identity check for nightly `nightly-2026-08-28`, commit `97f8b2b7882f0c15ad9ce9b53abed5553920caa8`, the checked-in patch and SHA-256, all manifest and lockfile source pins, and the CUDA/NVVM/libdevice matrix. Temporary negative fixtures rejected a manifest mismatch and both missing and modified patches through the contract and setup paths. `cargo fmt --all`, `cargo check --workspace`, and `cargo test --workspace api_tests` completed (17 passed, 125 filtered). The clean-clone gate honestly stopped because HEAD `d066af6bdb35c0a44a9c31b9948bf5f2e8d0f1bf` does not contain the three new gate scripts; no working-tree files were injected and no pass is claimed. The SM86 compile-only gate began all 23 `rccl-cub` CUDA examples, but `rustc` remained on `device_histogram_probe` at 99.9% CPU for more than 11 minutes; that run was cancelled, `rccl-cuco`/`rccl-thrust` were not reached, and no complete SM86 compile or runtime claim is made. |
| 2026-09-03 | Exact HISA row set | Checked exact `Row<N>` set, forced-collision, and capacity fail-open rows; SRDatalog call-site replacement remains open | Added stream-bound `ExactRowSet<N>` storage with padding-free HISA `u32` column layout, pinned `XxHash64` double hashing, full-row equality, checked sentinel rejection, duplicate/new/exhausted accounting, and `CudaCompletion`-retained async insert/contains borrows. `Row<2>` uses the pinned cuCollections-compatible packed 64-bit CAS path; other arities use release/acquire state publication plus stored hash and every row word. `scripts/test-cuco-exact-row-set.sh` passed CPU-oracle randomized/duplicate/absent/boundary cases, a forced-identical-hash negative query (`collision_false_positive=0`), pinned cuCollections capacity/contains differential, Compute Sanitizer memcheck (0 errors), racecheck (0 hazards), and synccheck (0 errors). SM89 PTX/SASS evidence for packed insertion is one 128-thread kernel with device-scope 64-bit CAS, 25 registers, 0-byte stack, 0 spills, and no local-memory instructions. The strict alternating five-round, 1.001 equal-work performance script covers insert/contains at 0.50/0.75/0.90 occupancy, but performance is unverified on this workstation because `VLLM::EngineCore` owns the GPU; no cuda-oxide/backend attribution is made. |
| 2026-09-03 | SRDatalog/HISA copy, gather, and generation lane | Checked copy/fill/sequence/tabulate composition and gather bounds rows | Added native contiguous D2D `copy`, zero-materialization range copy (Rust range slicing replaces `copy_n`), `u32`/`u64`/`usize` indexed gather, and affine range fill/tabulate/sequence, all retaining resources through `CudaCompletion`. Safe APIs reject overlapping source/output borrows; permutation indices are checked on-device and OOB deterministically traps with no partial-output guarantee. `scripts/test-copy-gather-generation.sh` passed empty/singleton/257-item/subrange/strided/reversed, length-mismatch, repeated-index, 32/64-bit index, and OOB-fault behavior on SM89; memcheck reported 0 errors and racecheck 0 hazards. PTX/SASS audit found direct accesses, 8 registers for common `u32` entries, 16-18 for `usize` fallbacks, and zero stack/spills. Both equal-work RCCL and pinned CCCL endpoint-oracle benchmark runs passed; their shared-GPU timings are not performance evidence. `scripts/test-copy-gather-generation-performance.sh` correctly exited 77 because VLLM owned the GPU, so performance is unverified and no cuda-oxide ownership claim is made. |
| 2026-09-03 | Private BlockScan and workload device scan | Checked inclusive/exclusive device scan method/free-function lowering; public BlockScan/prefix-callback row remains open | Added a private fixed 256-thread `BLOCK_SCAN_WARP_SCANS`-shape sum primitive with converged full-block participation, explicit partial-prefix zero fill, inclusive/exclusive/aggregate results, wrapping `u32`/`u64` counter semantics, two CTA barriers, and no allocation, atomics, or grid-wide synchronization. The workload-facing CUB layer provides typed reusable hierarchical workspace and in-place device scans; Thrust exposes synchronous/asynchronous free functions and `DeviceScanExt`. `scripts/test-block-scan.sh` passed direct aggregate and 0/1/31/32/33/255/256/257/65537 boundaries, multi-level `u32` and `u64`, wrapping and sentinel-total cases, an injected invalid-logical-extent trap, pinned CUB oracle, SM89 compilation, memcheck (0 errors), racecheck (0 hazards), synccheck (0 errors), and PTX/SASS resource audit (30–40 registers for scan entries, zero stack/spills, shuffle-up and CTA barriers present). The strict equal-work CUB performance gate correctly exited 77 because PID 1527736 (`VLLM::EngineCore`) owns the GPU, so no cuda-oxide performance ownership or parity claim is made. The generic/public P1 BlockScan row stays unchecked because no current workload requires arbitrary operations or prefix callbacks. |
| 2026-09-03 | SRDatalog/HISA stable compaction lane | Checked select/copy-if, adjacent unique/unique-copy, and stable partition rows | The workload surface remains limited to the four audited stable forms: predicate select over zero-materialization ranges, nonzero-`u32` flagged select, explicit-equivalence adjacent unique-copy, and scratch-backed stable two-way partition. Count, capacity, overlap, stability, and non-narrowing `u64` large-extent semantics are unchanged. The common `u32` path now follows pinned SM86+ `DeviceScan`: a 128-thread × 15-item warp-transpose tile, 32 padded transaction states, one state-init launch, and one decoupled-lookback scan launch with the pinned 450/350-cycle delay policy. The scan uses 8,192 bytes of static shared memory, 40 registers, zero stack/spills, and replaces the former recursive scan stages, reducing each compaction operation from seven launches to four. The source-owned common mark/store kernels use one item per thread while retaining the grid-stride `u64` fallback; the strict stable-partition scatter is 14 registers (other instantiated stable scatters 10–20), zero-frame/zero-spill. `scripts/test-compaction.sh` passed locally and on the exclusive SM89 pod: differential tails/predicates/flags/equivalence, count and range-fault contracts, native build, memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors, pinned CUB differential, and PTX/SASS no-local/no-atomic resource audit. Final local NSYS shows exactly mark + init + single scan + scatter; stable-partition component medians are 4.128, 0.896, 5.408, and 4.992 microseconds, respectively, versus the retained pinned-CUB source profile's 3.584-microsecond mark, 5.568-microsecond scan, and 5.472-microsecond scatter. Exactly one exclusive Runpod strict gate was run with identical 8,422,910-byte temporary storage; it did not meet 1.001: select-if 0.018739/0.016691 ms (1.122700857), flagged select 0.018723/0.016691 (1.121742256), unique-copy 0.018739/0.016790 (1.116081001), and stable partition 0.019046/0.017725 (1.074527504). Because the final device-stage sum and resource shape are already at or better than the pinned CUB stages while the residual is between launches, this is classified as generated/host-launch backend overhead rather than a remaining production scan/scatter algorithm defect; no threshold or benchmark policy was relaxed. The emitted raw gate log is preserved at `/tmp/compaction-performance-runpod-strict.log`, the finalized correctness log at `/tmp/compaction-correctness-runpod-final.log`, and the gate now prints both implementations' seven raw rounds before aggregation for future runs. Pod `lt0ehkxhhvds1o` was deleted and `runpodctl pod get` returned `not_found`/404. Broad CUB facade, unstable partition, arbitrary flag types, and non-adjacent unique remain intentionally absent. |
| 2026-09-03 | SRDatalog sorted adjacent-key grouped reduction | Checked workload `reduce_by_key` and first concrete segmented-reduction rows; general `DeviceReduce` and segmented scan remain open | External SRDatalog inspection found sorted `u32` segment IDs reduced with `SR::add` over NaturalBag `u64`, Boolean, and independent-probability provenance, while TVJoin groups sorted tuple keys and keeps the first tuple object. Added only stable adjacent-key grouping with explicit `equivalent(previous, current)`, first-key/provenance retention, `u64` group count, sealed NaturalBag wrapping add, normalized one-byte Boolean OR, and deterministic left-fold probability OR. The private implementation composes head marking, exclusive scan, and one disjoint fold per head without atomics or grid synchronization; inputs may be zero-materialization ranges, extents stay `usize`, and the common path uses `u32` positions through `u32::MAX`. Outputs require full input capacity because the count is asynchronous, and common `CudaCompletion` retains every input/output/count/operator/workspace borrow. `scripts/test-grouped-reduction.sh` passed CPU differentials for empty/singleton/31/32/33/255/256/257/513/65537 tails, all-one/all-distinct/mixed/non-adjacent groups, generated inputs, zipped first-key provenance, all three workload operators, wrapping overflow, capacity/length/count errors, injected range OOB, SM89, memcheck (0 errors), racecheck (0 hazards), synccheck (0 errors), pinned CUB differential, and PTX/SASS audit (20-28 reduction registers, no stack/spills/local traffic/out-of-line calls/atomics). The equal-work direct pinned `cub::DeviceReduce::ReduceByKey` benchmark matches u32 keys/u64 values/wrapping addition/results, 1,048,589-item partial tail, 256 threads, allocation exclusion, 8,422,910-byte temp parity, 10 warmups, 21 samples, and 10 launches/sample. `scripts/test-grouped-reduction-performance.sh` stopped with exit 77 before timing because foreign `VLLM::EngineCore` PID 1527736 owned the shared GPU, so no performance-parity or cuda-oxide attribution claim is made. |
| 2026-09-03 | SRDatalog/HISA workload record sort | Checked the audited `u32` AoS record and stable provenance-payload slice; arbitrary caller records, direct zipped-SoA sorting, affine outputs, and full ordering closure remain open | The public comparison-sort surface is restricted to audited `RelationRow<N> = [u32; N]` arities 2–6; dormant 7/8 dispatch was removed, scalar arity 1 remains on radix sort, and the sealed payload set is NaturalBag `u64`, independent-probability `f64`, and normalized byte Boolean provenance. In-place and exact-length out-of-place key/key-payload APIs remain stable and retain typed scratch, partitions, borrows, and execution through `CudaCompletion`. The Row<4>/`u64` comparator now spells out compile-time fields rather than runtime-indexed arrays; merge stages keys before value preload like pinned CUB, and reusable safe shared-tile proofs remove redundant key/value launch checks without unsafe or per-arity representation workarounds. The post-change correctness run passed its CPU, API, GPU, memcheck, racecheck, synccheck, and pinned-CUB differential phases before the resource gate caught one Row<3>/`u64` in-place pair spill; the final safe in-place shared-read proof removed it. A complete final SM89 ptxas matrix has zero stack frames and zero spill loads/stores for every instantiated record-sort entry. The strict Row<4>/`u64` benchmark entries are block out-of-place 58 registers, block in-place 58, merge 48, and partition 24, all zero-frame/zero-spill. The retained source NSYS profile measured old CUB/RCCL block-sort medians of 65.664/457.971 microseconds and merge medians of 23.745/56.769 microseconds; NCU was unavailable with `ERR_NVGPUCTRPERM`, so no unsupported counter attribution was made. The exclusive SM89 strict gate preserved all seven alternating raw rounds and passed with CUB 0.445030 ms, RCCL 0.437555 ms, ratio 0.983203380, identical 25,174,344-byte temporary storage, and the original stability, duplicate-tie, reset, synchronization, launch-count, and allocation-exclusion contract. |
| 2026-09-04 | Static-map lazy ranges, wide-value insertion parity, and cuda-oxide fork cutover | Checked cuda-oxide source/fork, toolchain compatibility, and clean-clone rows; broader range breadth remains open | Added static-map terminals for arithmetic/transform/entry/permutation ranges, checked `find_range`/`contains_range`, stencil-gated `contains_if`, and async completion guards retaining map, input, output, predicate, and status borrows. Permutation failures preserve the failing source extent, including independently fallible key and stencil ranges. The `u32`→`u64` no-tombstone insertion protocol now uses direct empty-control CAS plus relaxed payload publication while erased maps retain the tombstone-aware protocol; five alternating 8,000,000-item rounds passed the 1.050 parity gate at RCCL/cuCollections median ratios 1.017737, 1.017092, and 1.017937 for 0.50/0.75/0.90 occupancy. Migrated all cuda-oxide dependencies to `StarGazerM/cuda-oxide` commit `5f7c34fddadcf250f2e45d0cffe83a3d27f2bea3`, deleted the local vendor tree and patch, validated clean setup plus positive/negative identity contracts, built the `v1.0.2` Runpod image locally with its self-check, and passed the SM89 example, memcheck (0 errors), racecheck (0 hazards), targeted SM86 compilation, workspace check, 180 host tests, and clean committed-checkout gate at `4ee1b2751bcf46cb0f26ea838ab933211bfcb64f`. |

## Supporting references

- `CCCL_THRUST_CUB_COVERAGE_PLAN.md`
- `CCCL_CUB_EXAMPLE_TEST_CONTRACT.md`
- `reviews/ordering-cccl-source.md`
- `reviews/set-operations-cccl-source.md`
- SRDatalog production sources under `<SRDATALOG_CHECKOUT>`
- SRDatalog Python frontend under `<SRDATALOG_PYTHON_CHECKOUT>`
