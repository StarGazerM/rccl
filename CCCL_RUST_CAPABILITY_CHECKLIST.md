# CCCL/Rust semantic capability checklist (SRDatalog-only scope)

**Source of truth.** This checklist classifies observable Rust capabilities, not
C++ header or API-name counts. Scope is limited to SRDatalog sources:
`<SRDATALOG_CHECKOUT>/generalized_datalog` and
`<SRDATALOG_CHECKOUT>/integration_tests/examples/polonius/vflog`.
It does not claim external Rust integration exists; these callers remain
external C++/CUDA/Nim/JIT workloads.

## 1. Classification legend

- **Complete** — public Rust semantics and executable device-boundary evidence.
- **Complete-performance-open** — behavior is complete, but measured performance
  parity remains open.
- **Partial-lowering** — the Rust shape and some lowering exist, but a reachable
  shape/composition is unproven.
- **Genuinely-missing-reachable** — a reachable workload needs semantics not
  currently provided.
- **Reusable-substrate** — implementation foundation exists, not a caller claim.
- **No-caller-non-goal** — no reachable caller justifies this surface.

## 2. Reachable adaptor use

The xmake target declarations make the generalized_datalog GPU paths reachable:
`<SRDATALOG_CHECKOUT>/xmake/targets/xmake_tests.lua:50-138,180-220`.
The `build_test_tmp` additions are at `:223-299`; commented targets are not
reachable (`:171-177,246-262`). The polonius/vflog benchmark is an external
reachable workload (`<SRDATALOG_CHECKOUT>/benchmarks/scripts/bench_polonius.sh:5-20,66-76`).

| Source file:line | Adaptor/operation | Semantic purpose |
|---|---|---|
| `test/SRDatalog/generalized_datalog/gpu/deduplication.h:149-153` | binary zip | Compare complete SoA rows as one key range. |
| `.../deduplication.h:226-249` | discard, counting input | Reduce provenance by key and compact unique rows. |
| `.../deduplication.h:263-265` | gather/permutation | Compact each column from selected row indices. |
| `.../gpu/device_sorted_array_index.ipp:415-457` | binary zip + merge-by-key | Merge indexed SoA columns and provenance. |
| `.../device_sorted_array_index.ipp:560-597` | binary zip + set-difference-by-key | Produce delta rows while updating provenance. |
| `.../gpu/device_lsm_index.ipp:123-180` | sequence, sort, gather | Build sorted LSM storage and provenance permutation. |
| `.../device_lsm_index.ipp:207-214` | tabulate | Construct row-pointer records. |
| `.../device_lsm_index.ipp:749-798` | counting + for_each, copy_if | Mark rows absent from the full index and compact delta. |
| `SRDatalog/integration_tests/examples/polonius/vflog/include/column.cuh:6-14,33-42` | static map | Map `uint32_t` column keys to `uint64_t` compressed ranges. |
| `.../vflog/include/column.cuh:106-115` | `contains_if` | Stencil-filter map membership results. |
| `.../vflog/include/column.cuh:118-133` | transform(counting) | Create map key/value pairs from compressed index positions. |
| `.../vflog/src/indexing.cu:39-61,195-214` | counting, permutation, zip | Build indexed values and map entries. |
| `.../vflog/src/deduplication.cu:66-69,129-132,201-206,226-229` | counting transform, permutation | Deduplicate and look up indexed rows. |
| `.../vflog/src/join.cu:41-45,94-99` | permutation + `map_find`/`map_contains_if` | Join and membership filtering. |
| `.../vflog/src/join_filtered.cu:44-48,103-106` | permutation, counting + for_each | Filtered joins and materialization. |
| `.../vflog/src/ram/cartesian.cu:14-23` | counting transform | Generate Cartesian-product indices. |

## 3. Rust-native capabilities and executable evidence

These symbols are verified against `rccl-thrust/src/lib.rs:25-94` and
`rccl-thrust/src/range.rs`.

| Capability | Public symbols | Status/evidence |
|---|---|---|
| Arithmetic/counting | `arithmetic_range`, `ArithmeticRange` | Complete; exported by `lib.rs:53-57,90-94`, with range/device API evidence. |
| Repeat | `repeat_n`, `RepeatN` | Complete; `lib.rs:90-94`, `range.rs:56-58,216-243`. |
| Binary zip/key-value | `DeviceRangeExt::zip`, `DeviceRangeExt::zip_eq`, `KeyValueRange`, `KeyValueRangeMut` | Complete for binary SoA composition; `lib.rs:90-94`, `range.rs:273-373,776-834`. |
| Gather/permutation | `gather`, `gather_async`, `DeviceRangeExt::gather` | Complete for checked indexed gather; `lib.rs:34-37`, `copy.rs:173-199`. `Permutation` is not a root public export. |
| Discard | `discard`, `range::Discard` | Complete as explicit output range; `lib.rs:90-94`, `range.rs:60-62,245-271`, consumed by transform/output lowering. |
| Subrange/reverse/stride | `DeviceRangeExt::slice`, `rev`, `step_by` | Complete through `LowerReadRange`/`LowerWriteRange`; `range.rs:775-834,852-875`. |
| Affine/indexed output | `indexed_output`, `IndexedOutput`, `tabulate_range`, `sequence_range`, `sequence_range_from`, `sequence_range_by` | Complete; `lib.rs:39-43,90-94`, `range.rs:64-66,714-775`, `generation.rs:139-166,228-258`. |
| Copy/fill/sequence/tabulate | `copy`, `copy_range`, `fill`, `fill_range`, `sequence`, `sequence_from`, `sequence_by`, `tabulate` | Complete current contiguous/affine contract; `lib.rs:34-43`, `generation.rs:70-180,228-300`. |
| Transform/for_each | `transform`, `transform_into`, `for_each`, async variants | Complete for supported range lowering; `lib.rs:76-79`, `transform.rs:58-165`. |
| Inclusive/exclusive scan | `inclusive_scan`, `exclusive_scan`, async variants | Complete for workload counters; `lib.rs:68-70`, `scan.rs:23-95`. |
| Select/flagged/unique/stable partition | `select_if`, `select_flagged`, `unique_by`, `unique_copy_by`, `stable_partition_by` | Complete for audited stable forms; `lib.rs:29-33`, `compaction.rs:118-222,381-432`. |
| Reduce by key | `reduce_by_key`, `reduce_by_key_async` | Complete for sorted adjacent-key reduction; `lib.rs:44-46`, `grouped_reduction.rs:125-216`. |
| Merge keys/key-values | `Merge`, `MergeBy`, CUB `DeviceMerge::MergeKeys/MergePairs` substrate | Complete basic keys/pairs; lazy methods in `merge.rs:29-70` and CUB dispatch. |
| Set operations | `Union`, `UnionBy`, `Difference`, `DifferenceBy`, `Intersection`, `IntersectionBy`, `SymmetricDifference`, `SymmetricDifferenceBy`, `ArithmeticSetRangeExt` | Complete basic contiguous-key families; `lib.rs:72-75`, `set.rs`. |
| Sorting | `DeviceSortExt`, `DeviceKeyValueSortExt`, `sort_rows_lexicographic*`, payload variants | Complete-performance-open for audited relation rows/payloads; `lib.rs:52,60-67`. |
| Bloom/hash/exact set | `BloomFilter<K>`, `XxHash64`, `ExactRowSet` | Complete for current stated contracts; any throughput tuning is performance-open, not semantic missingness. |
| Static map | `StaticHashMap`, `StaticMap`, `MapKey`, `MapValue`, `entries` | Partial / genuinely-missing-reachable for the audited vflog caller: static-map shape exists, but `MapValue<u64>`, bulk range composition, and `contains_if` are not complete. |

Reusable substrate: `DeviceView`/`DeviceViewMut`, `LowerReadRange`,
`LowerWriteRange`, `DeviceBuffer`, typed workspaces, CUB merge/set/sort kernels,
scan/compaction kernels, `KeyValue`, and affine split borrows. These foundations
do not require C++ iterator facade names.

## 4. 2026-09 reachable C++ API audit (excluding TVJoin)

This table is based on operational (non-commented) calls found under the two
requested roots, excluding vendor/shim headers, comments, dormant targets, and
transitive implementation internals. A `NOT NECESSARY` marker means the C++
facade/adaptor name is not itself a required Rust deliverable because the
observable operation is already represented by a Rust-native public symbol.

No `.nim`, JIT-source, or generated-JIT source file under either requested root
contained an operational direct Thrust/CUB/cuCollections call; therefore no
Nim/JIT API row is inferred from build artifacts or stale plans.

| Reachable C++ API/adaptor | Exact external evidence | Current Rust public symbol / executable evidence | Classification |
|---|---|---|---|
| `thrust::copy`, `copy_n` | `generalized_datalog/gpu/device_array.h:259-280`; `gpu/nd_device_array.h:190-350`; `gpu/device_sorted_array_index.ipp:366-381`; `vflog/src/ram/copy.cu:44-46,109-110` | `rccl_thrust::copy`, `copy_range`; `rccl-thrust/src/copy.rs:24-92`; `scripts/test-copy-gather-generation.sh` | Rust semantic equivalent complete → **NOT NECESSARY** to reproduce C++ names |
| `thrust::copy_if` | `generalized_datalog/gpu/deduplication.h:247-249`; `benchmark/tc_lower_device_benchmark.cpp:448-449`; `gpu/device_lsm_index.ipp:796-798` | `rccl_thrust::select_if`, `select_flagged`; `rccl-thrust/src/compaction.rs:118-222`; `scripts/test-compaction.sh` | Complete flagged/predicate semantics → **NOT NECESSARY** |
| `thrust::gather` and permutation iterators | `gpu/deduplication.h:263-265`; `gpu/device_lsm_index.ipp:175-180`; `gpu/device_sorted_array_index.ipp:280-281`; `vflog/src/ram/copy.cu:27-28`; `vflog/src/ram/id_join.cu:32-37` | `rccl_thrust::gather`, `gather_async`; `rccl-thrust/src/copy.rs:173-199`; `scripts/test-copy-gather-generation.sh` | Checked indexed gather complete → **NOT NECESSARY** for `permutation_iterator` facade |
| `thrust::fill` | `generalized_datalog/benchmark/merge_overhead_benchmark.cpp:52-76`; `gpu/deduplication.h:213-216`; `gpu/device_sorted_array_index.ipp:217`; `gpu/device_lsm_index.ipp:601`; `vflog/src/hisa.cu:191-204` | `rccl_thrust::fill`, `fill_range`; `rccl-thrust/src/generation.rs:70-112`; `cargo oxide ... rccl-thrust --example constant_iterator` | Complete → **NOT NECESSARY** |
| `thrust::sequence`, counting iterator | `generalized_datalog/benchmark/merge_overhead_benchmark.cpp:57-90`; `benchmark/tc_lower_device_benchmark.cpp:270-345`; `gpu/device_lsm_index.ipp:124`; `vflog/src/indexing.cu:24-25`; `vflog/src/persistence.cu:131-139` | `rccl_thrust::sequence`, `sequence_from`, `sequence_by`, `arithmetic_range`; `rccl-thrust/src/generation.rs:228-300`; `scripts/test-thrust-constant-iterator.sh` | Complete arithmetic/counting semantics → **NOT NECESSARY** for C++ iterator name |
| `thrust::transform`, `tabulate`, `for_each` | `gpu/deduplication.h:163-165`; `gpu/device_lsm_index.ipp:211-213,315-341,749-753`; `vflog/src/ram/cartesian.cu:14-23`; `vflog/src/ram/materialization.cu:48-155`; `vflog/src/ram/copy.cu:65-67` | `rccl_thrust::transform`, `transform_into`, `for_each`, `tabulate`; `rccl-thrust/src/transform.rs:58-165`, `generation.rs:139-166`; `scripts/test-copy-gather-generation.sh` | Complete for audited contiguous/affine/device-callable shapes → **NOT NECESSARY**; arbitrary custom iterator composition remains partial |
| `thrust::exclusive_scan`, `inclusive_scan` | `generalized_datalog/benchmark/tc_lower_device_benchmark.cpp:270-271,344-345`; `benchmark/tc_lsm_device_benchmark.cpp:293-294,371-372`; `gpu/deduplication.h:180-181,199-200`; `runtime/launcher.h:78-79`; `runtime/executor_impl/executor_execute_pipeline.h:213-214,276-277` | `rccl_thrust::exclusive_scan`, `inclusive_scan`; `rccl-thrust/src/scan.rs:23-95`; `scripts/test-block-scan.sh` and scan examples | Complete for reachable counter scans → **NOT NECESSARY** |
| `thrust::reduce` | `generalized_datalog/gpu/deduplication.h:185-186` | `rccl_thrust::reduce`; `rccl-thrust/src/reduce.rs:12-54`; `cargo oxide ... device_reduce_sum_benchmark` | Complete for contiguous primitive sum; custom/general reductions remain partial |
| `thrust::reduce_by_key` | `generalized_datalog/gpu/deduplication.h:229-236` | `rccl_thrust::reduce_by_key`; `rccl-thrust/src/grouped_reduction.rs:125-216`; `scripts/test-grouped-reduction.sh` | Complete for sorted adjacent keys; this exact custom semiring composition is partial (public binary op is generic, semiring/device closure reachability is not independently proven) |
| `thrust::unique_count`, `unique_by_key_copy`, `unique_copy` | `vflog/src/indexing.cu:35-61`; `generalized_datalog/gpu/deduplication.h:359-362`; `gpu/device_lsm_index.ipp:690-692`; `gpu/device_sorted_array_index.ipp:632-635` | `rccl_thrust::unique_by`, `unique_copy_by`; `rccl-thrust/src/compaction.rs:224-330`; `scripts/test-compaction.sh` | Complete contiguous audited forms → **NOT NECESSARY** for C++ names |
| `thrust::remove`, `remove_if` | `vflog/src/deduplication.cu:86-88,177-179,276-278`; `vflog/src/ram/join.cu:60-66,103-106`; `vflog/src/ram/negate.cu:186-189`; `vflog/src/ram/id_join.cu:56-58` | `rccl_thrust::select_if` / flagged selection; `rccl-thrust/src/compaction.rs:118-222`; `scripts/test-compaction.sh` | Complete observable compaction semantics → **NOT NECESSARY** |
| `thrust::sort`, `stable_sort`, `stable_sort_by_key` | `generalized_datalog/benchmark/cuda_test.cpp:121-160`; `gpu/device_lsm_index.ipp:143,175-180`; `vflog/src/indexing.cu:30-32`; `vflog/src/persistence.cu:133-139` | `DeviceSortExt::{sort,sort_unstable,sort_by,sort_unstable_by}`, `DeviceKeyValueSortExt::{sort,sort_unstable,sort_by,sort_unstable_by}`; `rccl-thrust/src/ordering.rs:8-128`; row forms in `rccl-thrust/src/record_sort.rs:168-233`; `scripts/test-record-sort.sh` | Device relation-row forms complete/benchmark-backed; there is no root free `rccl_thrust::sort`; host TBB policy in `cuda_test.cpp:145-160` is not a GPU workload → **NOT NECESSARY** for TBB facade |
| `thrust::merge`, `merge_by_key` | `generalized_datalog/benchmark/merge_overhead_benchmark.cpp:120-121`; `gpu/device_lsm_index.ipp:569-571`; `gpu/device_sorted_array_index.ipp:455-457`; `vflog/src/persistence.cu:28-34,149-153` | `DeviceView::merge`, `DeviceView::merge_by`, `KeyValueRange::merge`, `KeyValueRange::merge_by`, returning `Merge`/`MergeBy`; `rccl-thrust/src/merge.rs:12-80`; ordering probes and `scripts/test-gpu.sh` | Basic keys/key-value semantics complete; there is no root free `rccl_thrust::merge`; complex SoA/custom comparator paths are partial concrete composition |
| `thrust::set_union`, `set_difference` | `gpu/device_sorted_array_index.ipp:491-493,591-597`; `test/test_reproduce_set_diff_bug.cpp:101-109` | `rccl_thrust::{Union,Difference,...}` and `ArithmeticSetRangeExt`; `rccl-thrust/src/set.rs`; `scripts/test-set-operations.sh` | Ordinary set operations complete → **NOT NECESSARY**; mutable-provenance `set_difference_by_key_with_match_custom` remains genuinely missing reachable behavior |
| `thrust::lower_bound` | `vflog/src/column.cu:20-27`; `vflog/src/ram/join_filtered.cu:125-127,175-177,210-212,379-381,438-440,483-485`; `vflog/src/ram/materialization.cu:69-71,120-122,155-157` | `rccl_thrust::{lower_bound,upper_bound,equal_range}` with comparator/async variants in `src/search.rs`; SM89 oracle/sanitizer coverage in `scripts/test-sorted-search.sh` | Implemented primitive; external caller migration and performance parity remain open |
| `thrust::adjacent_difference` | Header included by `vflog/src/hisa.cu:8` but no operational call found in scanned non-comment source | No caller contract | **NOT NECESSARY**; include-only surface is not a reachable behavior |
| `thrust::extrema` | Header included by `vflog/src/hisa.cu:11` but no operational call found in scanned non-comment source | No caller contract | **NOT NECESSARY** |
| `thrust::device_ptr`, `device_vector`, host/device execution policies | `generalized_datalog/gpu/device_array.h:79-80,138-142`; `gpu/deduplication.h:137-265`; vflog uses RMM vectors and `EXE_POLICY` (`utils.cuh:96-99`) | `DeviceBuffer`, `DeviceSlice`, `Execution`, allocator-bearing ranges; `rccl-memory/src/buffer.rs`; `rccl-thrust/src/cuda.rs` | Rust ownership/execution model is the semantic replacement → **NOT NECESSARY** to clone C++ containers/policies |
| cuCollections `static_map` bulk insert/find/contains/contains_if | `vflog/include/column.cuh:36-42,86-115,118-150`; calls in `vflog/src/indexing.cu:74,218-219`, `deduplication.cu:117-119,201-213`, `hisa.cu:308-309`, `ram/join.cu:41-45,94-99`, `ram/id_join.cu:43-45`, `ram/join_filtered.cu:275-279`, `multi_arity_join.cu:73-78`, `negate.cu:37-41,109-114` | `rccl_cuco::StaticHashMap::{insert,find,contains,contains_key}`; `rccl-cuco/src/map.rs:386-648`; `contains_if` has no public equivalent; `MapValue<u64>` absent (`rccl-cuco/src/word.rs:47-87`) | Genuinely missing reachable `MapValue<u64>`/16-byte packed-range map and `contains_if`; ordinary 32-bit map semantics are complete only for the narrower RCCL contract |

## 5. Exact remaining reachable gaps

1. **Static-map `u64` values — Genuinely-missing-reachable.** vflog defines
   `comp_range_t`/`offset_type` as `uint64_t` (`include/utils.cuh:102-105`) and
   uses `GpuMap` (`include/column.cuh:33-42`). RCCL `MapValue` currently covers
   only `u32/i32/f32`; the 16-byte `(uint32_t,uint64_t)` slot and its atomic
   publication protocol are not provided.
2. **Static-map bulk range composition — Partial-lowering.** Reachable pair
   construction uses transform/counting iterators (`include/column.cuh:118-150`);
   RCCL does not yet prove this exact composed input/output range contract.
3. **Static-map `contains_if` — Genuinely-missing-reachable.** The operation is
   called at `include/column.cuh:106-115` and `src/ram/join.cu:94-99`; RCCL has
   no public stencil-filtered map-membership operation.
4. **Lower-bound search — primitive implemented, integration/performance open.**
   vflog calls `thrust::lower_bound` in `src/column.cu:20-27`,
   `src/ram/join_filtered.cu:125-127,175-177,210-212,379-381,438-440,483-485`,
   and `src/ram/materialization.cu:69-71,120-122,155-157`. RCCL now exports
   batched lower/upper/equal-range search; these external callsites are not
   migrated or verified by the standalone R1 correctness example.
5. **Custom mutable-provenance set difference — Partial-lowering.**
   `device_sorted_array_index.ipp:560-597` performs set-difference-by-key with
   mutable provenance matching; the Rust public path does not yet prove this
   in-place delta contract.
6. **Custom semiring `reduce_by_key` composition — Partial-lowering.**
   `generalized_datalog/gpu/deduplication.h:229-236` supplies a device semiring
   closure; the generic Rust operation exists, but this exact semiring/device
   closure composition has no independent executable proof.
7. **Higher-arity zipped SoA composition — Partial-lowering.**
   `generalized_datalog/gpu/deduplication.h:151-153` and
   `device_sorted_array_index.ipp:416-426,561-570` construct parameter-pack
   zip ranges; RCCL's proven public key/value range is binary.
8. **Scan-by-key/transform-scan — No-caller-non-goal.** No operational caller
   was found in the audited non-TVJoin sources.
9. **Dynamic map — No-caller-non-goal.** The vflog `dynamic_map` alias is
   commented in `include/column.cuh:40-42`; static-map coverage is the actual
   reachable requirement.

## 6. Performance-open (not semantic missingness)

- Ordering merge/merge-sort/radix/set throughput and register/occupancy parity.
- Broader reduction and arbitrary-record sorting performance.
- Multi-stream and allocator/resource propagation tuning outside correctness.

## 7. Remove from roadmap

Remove C++ facade-name rows for `counting_iterator`, `constant_iterator`,
`transform_iterator`, `zip_iterator`, `permutation_iterator`,
`discard_iterator`, iterator categories/facades, `device_ptr`, and
`device_reference`. Remove speculative scan-by-key and transform-scan rows as
no-caller non-goals. Do not remove existing basic merge or set families.

## 8. Ordered next checklist by workload value

1. Add `MapValue<u64>` and a 16-byte static-map slot/publication path; validate
   compressed-range values and the `UINT32_MAX` miss sentinel against vflog.
2. Complete static-map bulk transform/counting range input/output composition.
3. Add public stencil-filtered map membership (`contains_if`).
4. Integrate public lower-bound search and exercise all vflog join/materialization
   callsites listed in section 5; close the separate search performance gate.
5. Implement the mutable-provenance set-difference/delta contract used by
   `device_sorted_array_index.ipp:560-597`.
6. Prove the custom semiring `reduce_by_key` composition at
   `generalized_datalog/gpu/deduplication.h:229-236`.
7. Expand SoA zip only for the cited higher-arity callers at
   `generalized_datalog/gpu/deduplication.h:151-153`; retain binary
   `KeyValueRange` as complete otherwise.
8. Keep scan-by-key, transform-scan, and dynamic map out of the active plan as
   no-caller surfaces.
9. Keep ordering performance gates in a separate benchmark lane.
