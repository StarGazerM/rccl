# CUB/Thrust ordering upstream correspondence and masking review

Date: 2026-08-28

Pinned CCCL revision: `571f2fc3bc53cd710e306ad43d58995c1fe4219f`.

Sources read for the currently exported intersection:

- `cub/test/catch2_test_device_merge_api.cu`;
- `cub/test/catch2_test_device_merge.cu`;
- `cub/test/catch2_test_device_merge_sort_iterators.cu`;
- `cub/test/catch2_test_device_radix_sort_keys.cu`;
- `cub/test/catch2_test_device_radix_sort_pairs.cu`;
- `cub/examples/device/example_device_radix_sort.cu`;
- `cub/cub/device/device_radix_sort.cuh`,
  `cub/cub/device/dispatch/dispatch_radix_sort.cuh`, and
  `cub/cub/device/dispatch/kernels/kernel_radix_sort.cuh`;
- `cub/cub/device/dispatch/tuning/tuning_radix_sort.cuh`;
- `cub/cub/agent/agent_radix_sort_onesweep.cuh`,
  `cub/cub/block/block_radix_sort.cuh`,
  `cub/cub/block/block_radix_rank.cuh`,
  `cub/cub/block/block_exchange.cuh`, and
  `cub/cub/util_type.cuh`;
- `thrust/examples/sort.cu`;
- `thrust/testing/sort.cu`, `stable_sort.cu`, `sort_by_key.cu`, and
  `stable_sort_by_key.cu`;
- `thrust/thrust/system/cuda/detail/sort.h`;
- `thrust/testing/merge.cu` and `merge_key_value.cu`.

The direct public Rust calls are the `device_merge_*!`,
`device_merge_sort_*!`, and `device_radix_sort_*!` CUB macros. Thrust
`sort*!` and `merge*!` macros lower through those exact CUB operation macros;
Thrust owns no ordering kernel.

## TMP boundary

C++ deduces `KeyT`, `ValueT`, iterator value types, and policy inputs before
instantiating dispatch and agents. A stable Rust proc macro cannot query
rustc's later inference. Every fact that changes this lowering is therefore
macro input:

- key and value types select layout-dependent merge policy;
- primitive radix category selects unsigned, sign-bit, or floating bit
  transformation;
- radix width selects the stable digit-pass tree;
- a custom decomposer and its component types will be required explicitly
  before custom radix keys can be exported; and
- fixed extents and offset category must likewise be explicit when their forms
  are implemented. Primitive radix bit windows are runtime `Range<u32>`
  values, matching CUB's runtime `begin_bit`/`end_bit`; only the maximum
  policy shape is macro-selected.

Primitive aliases and custom records currently fail closed because their
layout/decomposition is opaque to the macro. No `TypeId`, runtime trait
object, type registry, inferred fallback, or comparison-sort substitution is
used to hide that missing TMP input.

Temporary storage is not a Thrust semantic operand. Pinned CUDA Thrust first
queries CUB for `storage_size`, creates `temporary_array<std::uint8_t,
Derived>` from the execution policy, and then runs the CUB operation. RCCL's
Thrust macro therefore selects the CUB workspace/partition formula, derives
its runtime extent from the ranges, and acquires the concrete buffers through
`Execution`. Direct CUB retains the explicit caller-storage boundary as one
typed, opaque, reusable workspace. Its fields are private, matching CUB's
single `d_temp_storage` operand without importing byte pointers into Rust.

CUDA modules and architecture tags are not algorithm operands either. Pinned
Thrust uses `cuda::par.on(stream)` and pinned CUB receives the stream as an
ordinary call operand; neither API receives a module or an `sm_XX` token.
RCCL now constructs one typed program owner with `thrust_program!` or
`cub_program!`, binds CUB with `program.on(&stream)`, and constructs the
ordinary Thrust policy with `program.par(&stream)`. The CUB proc macro reads
the target selected by cuda-oxide. Parser tests reject a caller-provided
architecture token rather than ignoring it.

## Public call-shape comparison

| Pinned CCCL call boundary | Rust call boundary | Deliberate Rust translation |
|---|---|---|
| `thrust::cuda::par.on(stream)` | `let exec = thrust.par(&stream)?` | the typed `Program` owns the cuda-oxide program artifact; the execution value contains only runtime stream/allocation state |
| `thrust::sort(policy, first, last, comp)` | `sort_by!(::<T>(&exec, &mut range, closure))` | one mutable device range replaces the iterator pair; a Rust closure replaces the comparator object |
| `thrust::merge(policy, first1, last1, first2, last2, result, comp)` | `merge_by!(::<T>(&exec, &first, &second, &mut output, closure))` | range lengths and output exclusivity come from borrows |
| CUB query/execute with one `d_temp_storage` and `temp_storage_bytes` pair | one `device_*_workspace!` construction followed by `device_*!(..., &mut workspace, ...)` | a typed opaque owner replaces the byte pointer/size pair; internal scratch and agent state are private |
| CUB stream/environment operand | `let launch = cub.on(&stream)?`, then `&launch` | the stream binding is reusable runtime state; no invented CUB execution policy, module argument, or architecture token |
| CUB block/warp primitive inside a user `__global__` kernel | `#[rccl_cub::program] mod kernels { #[kernel] fn ... { warp_reduce!(...) } }`, then one `kernels::program!(&context)` bind | cuda-oxide requires an explicit Rust program boundary; the paired item/expression macros prove artifact provenance and keep its unsafe loader out of user code |

The extra `Program` construction is a cuda-oxide program-lifetime boundary,
not an algorithm operand. It occurs once before example or benchmark work;
neither the benchmarked operation nor its comparator receives a module.

## Result masking audit

- `rccl-cub/examples/device_ordering_probe.rs` uses cuda-oxide GPU launches
  and checks complete outputs. It contains the exact `MergeKeys` and
  `MergePairs` example input/output from pinned CCCL.
- Merge-pair payloads independently expose first-range precedence for equal
  keys. Checking sorted keys alone would miss this error.
- Merge-sort key/value cases check complete permutations and stable payload
  movement, not only sortedness.
- Radix integer cases include signed minima/maxima, duplicates, ascending and
  descending forms, and payload tags.
- The primitive radix `DoubleBuffer` probes sort the pinned u32 `0..24` bit
  window in both directions. Its three passes force the selected result into
  the alternate allocation, so reading a fixed allocation cannot pass. Pair
  payloads use an independently stable CPU reference, so key-only agreement
  cannot hide a scatter or selector error.
- A zero-width window starts with different contents in the two allocations
  and uses the macro-selected zero-sized single-tile workspace. It must
  preserve the original selected range without launching. Reversed ranges
  return their specific error and leave selection and contents unchanged.
- The exact Policy800 single-tile boundaries are tested on both sides: 4,864
  `u32` keys and key/value pairs receive zero one-sweep state, while 4,865
  keys receive the exact bins/lookback/counter extents and run the one-sweep
  route. Macro expansion tests assert both sides of that conditional, so an
  unconditional single-tile route or unconditional workspace allocation
  cannot pass the current fixtures.
- Single-tile probes require exactly one selector transition and compare every
  key and stable payload against independent CPU references. Direct owning
  buffer calls materialize that transition with safe `mem::swap`, without a
  device copy or a public C++-style numeric selector.
- Pair sorting uses one `DoubleBuffer<(keys, values)>`, not two mutable numeric
  selectors. The constructor rejects any of the four range extents differing,
  and one safe split borrow supplies disjoint current/alternate tuples.
- Floating radix checks `to_bits()` for every output. Rust floating equality
  would consider `-0.0 == +0.0` and mask the required stable signed-zero
  behavior, so it is not used as the oracle.
- Single-tile invalid lanes use the ordered integer sentinel (zero or all
  ones), not a synthetic floating value. The latter allowed the GPU to
  canonicalize an all-ones NaN and displaced a valid NaN payload; the complete
  bitwise NaN matrices now exercise the single-tile route and detect that
  failure.
- Primitive transformation unit tests independently cover ascending and
  descending round trips for all six supported representations, every valid
  one-byte digit-window shape, and signed-zero digit equivalence. The proc
  macro selects named function items rather than call-site closures, so
  repeated public calls reuse one cuda-oxide specialization without changing
  the digit semantics.
- Copy forms verify both the full output and that the full input remains
  unchanged.
- All calls propagate `Result`; no launch error is discarded. None of the
  semantic fixtures is empty, so the zero-length fast path cannot make them
  vacuously pass.
- CUB and Thrust parser tests reject every former scratch-bearing arity.
  Thrust expansion tests require the internal radix-workspace,
  merge-sort-partition, or merge-partition formula plus an `Execution`
  temporary acquisition; direct CUB examples require one opaque workspace
  constructor. This keeps a hand-sized allocation or silently ignored legacy
  operand from passing the current examples.
- The executable is built with `cargo oxide build` immediately before
  execution. Both fault iterations reported recompilation of the macro and
  runtime crates, ruling out a stale pre-fault executable.
- `benchmarks/ordering-reference.cu` and the Rust benchmarks use complete
  device-resident validation. The strict 1.001 performance ratio is still
  failing, so result parity cannot be misreported as CCCL performance parity.
  The direct bit-window and Policy800 single-tile additions bring the gate to
  fourteen paths. Its latest median-of-medians run still fails, including
  key-only radix at 1.064799x, bit-window keys at 1.083551x, single-tile keys
  at 1.049953x, and current internal-storage Thrust sort at 1.263096x; the
  combined gate still exits nonzero with
  `implementation gap is not masked`. The complete ratios and backend-boundary
  diagnosis are recorded in
  `reviews/current-implemented-performance.md`.

## Injected faults

Both faults were temporary and removed after observing the intended failure.

| Fault | Intended detector | Observed failure |
|---|---|---|
| Give the second merge input precedence when keys compare equal | exact pinned `MergePairs` payload assertion | GPU executable exited 101: output bytes `[65, 97, 98, 66, 67, 68, 99]`, expected `[97, 65, 98, 66, 67, 68, 99]` (`Aa...` instead of `aA...`) |
| Disable floating negative-zero normalization in both digit encoders | descending bitwise floating assertion | GPU executable exited 101: zero bit patterns were `[0, 2147483648]`, expected stable `[2147483648, 0]` |
| Reset packed radix counters using the later raking layout | single-tile compute-sanitizer memcheck | invalid shared-memory write; corrected by keeping CUB's counter-major reset before the barrier and raking reinterpretation after it |
| Materialize the invalid-lane float sentinel as `f32`/`f64` | complete bitwise NaN matrix | positive quiet NaN canonicalization changed the sentinel payload and displaced a valid NaN; retaining the ordered integer representation fixes the semantic boundary |

The first performance review also found that the Rust benchmark generated one
stateful xorshift stream while the C++ reference generated each value from its
index. Identical counts were therefore not identical input work. Both sides
now use the same per-index formula, and both Rust programs compare every output
against an independently CPU-sorted vector. The C++ reference separately
copies and compares the complete CUB radix, CUB merge-sort, and Thrust outputs;
sortedness alone is no longer accepted.

The focused key and pair single-tile probes pass both compute-sanitizer
memcheck and racecheck. The aggregate result gate runs those focused probes in
addition to the complete primitive matrices.

The Policy800 source mapping also retains CUB's `__launch_bounds__(256, 1)`,
the 32-byte alignment induced by `BlockScanWarpScans`, the u16 counter-update
phase before the u32 raking reinterpretation, the lane special register, and
separate const-specialized blocked/striped scatter calls. A runtime scatter
layout flag is rejected by the source test. Two performance experiments were
also rejected and removed: widening all hot shared-memory indices through a
new u32 access surface, and replacing the safe shuffle validity branch with an
arithmetic mask. Neither may be reintroduced merely because result tests pass.

The multi-block Policy800 mapping now also fixes histogram width at the
macro-selected 128 threads, reads the CUDA lane register instead of deriving a
lane from `threadIdx.x`, computes one pass digit mask for the named
`ShiftDigitExtractor` equivalent, keeps key and rank register arrays
uninitialized until their fully unrolled writes, and uses the exact guarded
`bin = threadIdx.x` coordinate rather than clamping inactive threads to bin
255. Full and partial warp-striped input capabilities consume a borrowed slice
and the checked lane coordinate: the partial form narrows the final valid
prefix once and preserves CUB's per-register activity predicate without
exposing a pointer view in the algorithm. Macro-generated entry variants own
the exact CUB static union extents:
34,304/32,768 bytes for supported key policies and
28,160/26,624/32,768 bytes for supported pair policies. Macro-visible layouts
select the entry; neither a runtime type registry nor a maximum-sized fallback
exists. Histogram and exclusive-sum storage are likewise static at 4,096 or
8,192 bytes and 2,336 bytes. The operation macros end `DoubleBuffer` borrows
with lexical scopes; a forced `drop` on a non-`Drop` carrier is
source-rejected. Generated CUDA entry points may carry only the exact
`clippy::too_many_arguments` ABI annotation, and the macro safety audit rejects
broader lint suppression such as `clippy::all`.

Two further code-shape experiments were measured and removed: constructing
the register key array with `core::array::from_fn`, and narrowing the uniform
bin/counter offsets from `usize` to `u32`. Each increased the key one-sweep
kernel from 99 to 101 registers while leaving the pair kernel at 80, so neither
is allowed back as a result-only cleanup.

Nsight launch-resource checks keep the remaining performance failure precise.
The common u32 Rust/CUB one-sweep key specializations use 99/75 registers with
the same 34,304-byte static union, limiting Rust to one CTA per SM. Pair
specializations use 80/65 registers with the same 28,160-byte static union and
both admit two CTAs per SM. The strict performance gate still exits nonzero;
static storage specialization is not reported as performance parity merely
because all result probes pass.

## Remaining upstream product

This review is partial. The implemented SM89 primitive agent topology is now
source-mapped. Primitive 32/64-bit SM80-policy borrowed-double-buffer bit
windows now execute the pinned runtime pass graph, including partial final
digits and selector updates. `ORDER-PERF-01` records the known unresolved
cuda-oxide/LLVM ballot-complement code-generation gap for direct CUB
Policy800 one-sweep radix and dependent Thrust sort paths; it is
`backend-blocked`, not an accepted difference. Exact single-tile and one-sweep
performance, pointer-input/output bit windows, range-specific temp-storage queries,
multi-portion inputs, direct-CUB resource-owned allocation, non-default Thrust
temporary allocators beyond cuda-oxide's concrete synchronous buffer ABI,
8/16-bit keys, half/bfloat,
decomposers/custom records, 32/64-bit offset selection,
general iterator/adaptor products, segmented sorting, and all other
architecture rows remain. Thrust permutation/zip/fancy-iterator ordering and
device-callable sequential forms also remain. None is represented by a passing
stub or CPU fallback.
