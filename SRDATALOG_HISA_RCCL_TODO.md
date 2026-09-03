# SRDatalog and HISA Rust GPU Migration TODO

Last reviewed: 2026-09-03

This is the canonical workload-driven tracker for replacing the CUDA C++,
Thrust, CUB, and cuCollections layer used by SRDatalog and HISA with Rust RCCL
and cuda-oxide. It does **not** claim to replace the NVIDIA CUDA driver, PTX,
NVVM, or the GPU runtime.

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
- [ ] Complete `rccl-cuco` workload coverage. A usable packed `u32`/`i32`
  static-map slice is integrated, but its exclusive performance gate is still
  blocked and it is not yet the required Bloom filter or exact tuple set.
- [ ] Demonstrate an end-to-end HISA or SRDatalog workload using only the Rust
  application/runtime layer.

## P0: execution safety and reproducibility

- [ ] Pin and commit the cuda-oxide source/fork and all local backend patches.
- [ ] Pin the Rust nightly and CUDA/NVVM/libdevice compatibility matrix.
- [ ] Add a safe `Pending<'a>` or equivalent completion object that retains
  input/output borrows, callable state, module state, and temporary storage
  until stream completion.
- [ ] Define safe drop, wait, error, and explicit unsafe detach behavior for
  pending GPU work.
- [ ] Add owned CUDA event wrappers and cross-stream wait/dependency APIs.
- [ ] Prove mutable outputs cannot be reused while an operation is pending.
- [ ] Add clean-clone build and test coverage without undeclared local files.
- [ ] Add SM86 compile and native correctness coverage for the HISA target.

## P0: ranges and linear algorithms

- [ ] Finish zero-copy counting, constant, transform, zip, permutation/gather,
  discard, offset, and writable-output range lowering.
- [ ] Implement device `for_each`, general `transform`, and `tabulate`.
- [ ] Implement `copy`, `copy_n`, `fill`, and `sequence`.
- [ ] Implement `gather` and the permutation-range bounds contract.
- [ ] Implement inclusive and exclusive device scan.
- [ ] Implement device reduce and reduce-by-key.
- [ ] Implement copy-if/device select.
- [ ] Implement unique and unique-copy.
- [ ] Implement stable partition.
- [ ] Validate contiguous and fancy-range forms with 32-bit and 64-bit counts.

## P0: ordered relations and HISA lookup

- [ ] Make sort and stable sort operational over arbitrary `#[repr(C)]` record
  keys and lexicographic zipped structure-of-arrays columns.
- [ ] Complete stable sort-by-key with borrowed values and affine output views.
- [ ] Complete merge and merge-by-key for the required fancy ranges.
- [ ] Complete set union and set difference for SRDatalog relation keys.
- [ ] Implement device lower-bound and binary-search operations used by HISA.
- [ ] Close current ordering correctness, code-shape, register, and exclusive
  performance gates on every supported target architecture.

## P0: cuCollections research subset

- [ ] Implement bit-compatible `XxHash64` for bitwise-stable `#[repr(C)]`
  HISA key records, with pinned host/device test vectors.
- [ ] Implement an owning stream-bound Bloom filter with clear, bulk add, bulk
  contains, and a read-only device reference.
- [ ] Match the required atomic scope, filter policy, capacity calculation, and
  no-false-negative semantics.
- [ ] Test Bloom false positives as permitted and false negatives as forbidden.
- [ ] Implement an exact concurrent `StaticSet<Row<N>>`, or equivalent hash
  table that stores enough tuple data to perform full equality checks.
- [ ] Replace SRDatalog's hash-only dedup rule that treats equal 64-bit hashes
  as equal tuples.
- [ ] Add forced-collision tests proving distinct tuples are never discarded.
- [ ] Define and test fail-open behavior for probe or capacity exhaustion.

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
- [ ] Implement segmented scan/reduce when recursive aggregation first requires
  it, with a concrete workload and acceptance test.

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
| 2026-09-01 | Packed `rccl-cuco` static map | Partial implementation recorded; workload item remains open | Added stream-bound packed `StaticHashMap` with sealed 32-bit key/value words, xxHash32 linear/double probing, checked and prevalidated insertion, lookup/presence, erase, count, retrieval, rehash, device refs, and a borrow-retaining pending insertion guard. `scripts/test-cuco-static-map.sh` passes host tests, SM89 cuda-oxide correctness, memcheck, racecheck, synccheck, PTX schedule checks, and the pinned cuCollections/CCCL reference build/run. `RUSTFLAGS='-D warnings' cargo check -p rccl --all-targets`, focused clippy, and `RUSTDOCFLAGS='-D warnings'` focused docs pass. The issue-606 erase/reinsert regression is corrected rather than cloned. `scripts/test-cuco-static-map-performance.sh` refuses to time while PID 2614460 `VLLM::EngineCore` owns the GPU, so the strict 1.001 exclusive-performance claim remains open. The aggregate `scripts/test-gpu.sh` also stops before CUCO validation because the pre-existing dirty histogram work contains the forbidden word `unsafe` in `rccl-cub/src/histogram_backend.rs:808`. Bloom, XxHash64 records, and exact tuple-set support remain open. |
| 2026-09-01 | CUB/Thrust histogram and cuda-oxide root-cause isolation | No checkbox changed; histogram performance remains open | On an exclusive RTX 6000 Ada, `scripts/test-histogram.sh` passes the complete pinned result, regression, and injected-fault matrix; Compute Sanitizer reports 0 memcheck errors and 0 racecheck hazards. The unchanged five-round 1.001 gate fails honestly at 1.186523/1.053700/1.006392/1.047995 for single-even/single-range/multi-even/multi-range. `cargo oxide build --arch sm_89 --features cuda-oxide-example -- --release --package rccl-thrust --example histogram` and the resulting example pass. PTX/SASS inspection shows the required fully safe slice-backed `ThreadTile` is materialized by cuda-oxide as a 1,224-byte local stack frame with 76 registers in single-even. Earlier compact-address diagnostics preserved exact CUB load/atomic topology and reduced the remaining gaps, but used an internal representation forbidden by the Rust safety contract; branch folding removed its spill without changing timing, and batched timing ruled out host launch/event overhead. Closure therefore requires cuda-oxide to lower the lifetime-bearing bounds proof to compact SSA address/count values, followed by safe-range and predicate/control-flow optimization; raw pointers, unchecked indexing, a different algorithm, or a relaxed gate are not accepted. Host evidence also passes `cargo test --workspace --all-targets`, strict workspace clippy/docs, all vendored cuda-host/cuda-macros/cuda-device suites, rustfmt, and `git diff --check`. The old aggregate-gate false positive from the word `unsafe` in a documentation sentence was removed without weakening the code check. |
| 2026-09-03 | Compiler-owned architecture selection and complete RCCL proc-macro removal | No checkbox changed; architecture reproducibility and ordering acceptance gates remain open | Removed all 51 former RCCL proc-macro entries and deleted the entire `rccl-cub-macros` crate, along with the exported CUB/Thrust operation-macro layers. Warp, merge, merge-sort, radix, set, histogram, workspace sizing, validation, allocation, program loading, and pass control now use ordinary functions, const functions, sealed traits, generics, checked-in kernel modules, and private repetition/unrolling-only `macro_rules!`. Public APIs expose no architecture, policy, module, generated kernel-set, or compiler-only type parameter. Architecture comes only from `cargo oxide build --arch`; the repository patch emits exact and cumulative architecture cfgs into host and device rustc invocations. The cuda-oxide patch also makes target-specific v2 device-artifact anchors the default, with a weak legacy alias, preventing a same-package library/example anchor collision that previously dropped `device_radix_sort_init` from the final executable. Normal builds require no device-code owner list. Removed the C++-shaped `ReadOnlySlices`/`ReadOnlyRange` descriptor and its special launcher/compiler handling; retained only the genuine device cache-load intrinsic. Disabled Rust host ThinLTO because cuda-oxide deliberately embeds native `.oxart` rlib members without LLVM bitcode; device NVVM optimization remains enabled. The patch applies to pinned cuda-oxide `97f8b2b7`, `scripts/setup-cuda-oxide.sh /tmp/rccl-cuda-oxide-patch-check` rebuilt the compiler successfully, and `cargo +nightly-2026-08-28 test --manifest-path /tmp/rccl-cuda-oxide-patch-check/Cargo.toml -p cargo-oxide apply_output_mode_sets -- --nocapture` passed 3 tests. Strict host validation passed: `RUSTFLAGS='-D warnings' cargo check --workspace --all-targets`; `RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets -- -D warnings`; `RUSTFLAGS='-D warnings' cargo test --workspace --no-fail-fast` (63+5 rccl-cub, 14+29 rccl-memory, 11+4 rccl-thrust tests plus 6 compile-fail doctests); and `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps`. Vendored cuda-macros (102 tests), cuda-host (69 passed, 1 toolkit-path live test ignored, 7 doctests), and cuda-device (104 tests, device-only doctests ignored) suites passed. `RCCL_STRUCTURE_ONLY=1 scripts/test-gpu.sh` passed the pinned source contract (19 programs, 378 regions, 49 API tests, 105 exercised methods), compiler-cfg/API boundary checks, shell syntax, rustfmt, and `git diff --check`. Focused owner-filter-free SM89 build/runtime probes passed: combined merge/merge-sort/radix ordering in 6m04s with `device_radix_sort_init` present in the final binary; Thrust set operations in 1m05s; Thrust histogram in 2m15s; and constant/transform in 0.57s. These timings are not a controlled cold/warm comparison. Remaining limitations: this change did not run the complete sanitizer, differential-parity, determinism, register/spill, SM86, clean-clone, binary-size, or exclusive-performance matrix, so no corresponding acceptance checkbox is complete. |

## Supporting references

- `CCCL_THRUST_CUB_COVERAGE_PLAN.md`
- `CCCL_CUB_EXAMPLE_TEST_CONTRACT.md`
- `reviews/ordering-cccl-source.md`
- `reviews/set-operations-cccl-source.md`
- SRDatalog production sources under
  `/home/stargazermiao/workspace/SRDatalog`
- SRDatalog Python frontend under
  `/home/stargazermiao/workspace/srdatalog-python`
