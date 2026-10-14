# Implementation request: lookup-expand primitives for MiniLinq

Requester: MiniLinq (`/data/mini_linq`), a Rust CQ compiler whose Physical IR
is binary lookup-expand (`probe`/`semi`/`anti`/`point`/`range`). A planned
external GPU printer (`features/gpu`) will emit ordinary Rust calls into RCCL,
consistent with `RCCL_COMPILER_DSL.md` (no RCCL compiler, no proc macros).
SRDatalog's unported `ProbeJoin`/`GatherColumn` path needs the same pieces.

Generic GPU mechanisms only; relational logic stays in MiniLinq
(`SRDATALOG_HISA_RCCL_TODO.md:23-24`).

## R1 Public sorted search

Currently private (`ordering_backend/set.rs:302`, `histogram_backend.rs:803`;
TODO:76, checklist §4).

- `lower_bound`, `upper_bound`, `equal_range` of a needle range against a
  sorted haystack view: Thrust vectorized-search shape, writing indices into a
  `view_mut()`; `_by` comparator variants; sync and `_async`.
- Key types: existing lowered read ranges, including packed primitives and row
  keys; do not impose `TemporaryValue` solely to implement search.
- Acceptance: CPU-oracle tests on duplicates, empty haystack/needles, all-miss,
  all-hit; `cargo oxide` example; README entry.

## R2 Load-balanced expand

Given per-probe match ranges `(begin, end)`:

- checked counts = `end - begin` → exclusive scan → representable total
  (reuse the scan, but reject overflow rather than accepting wrapping offsets);
- expand: for every output slot `o`, find owning probe
  `p = upper_bound(offsets, o) - 1`, emit
  `(p, begin[p] + o - offset[p])` row-id pairs into output views;
- then existing `gather` for payload columns.

Expose as a composable range terminal, e.g.
`probes.view().expand_ranges(begins.view(), ends.view()).write_into(out, &exec)`
and `.count(&exec)`. Acceptance: skewed fanout test (one key × 10⁶ matches
plus many singletons) matches CPU oracle; throughput independent of skew
within a stated factor.

## R3 Row keys beyond primitives

Scalar sorting and Thrust merge have primitive restrictions, but key-only set
operations already accept `DeviceCopy` keys in supported 1/2/4/8/16-byte
layouts. Row sorting supports `[u32; N]`, N=2..6; adjacent unique already
accepts generic lowered ranges. This is partial support, not a verified
end-to-end row pipeline.

- Minimum: documented, tested packing of 2–4 column keys into `u64`/`u128`
  usable by R1/R2, sort, unique, set ops.
- Preferred: lexicographic `Row<N>` (or `#[repr(C)]` DeviceCopy records)
  accepted by R1, merge and set operations, not only sort and `ExactRowSet`.

## R4 Multi-column hash lookup (lower priority)

Static map keys are currently `u32`/`i32`. The `static_map_u64` example has
`u32` keys and `u64` values, not wide keys. Packed `(first, count)` values
already provide lookup into pre-grouped rows. Wide or packed-row keys would
provide a hash alternative to R1 feeding the same R2 expand; they remain a
separate, lower-priority implementation.

## R5 Dedup of result rows

`sort` + `unique_by` over R3 rows (or `ExactRowSet` insert-if-absent) as a
documented pipeline for distinct answers; acceptance against CPU `BTreeSet`.

## Non-goals here

WCOJ k-way intersection, fixpoint/delta maintenance, strings (MiniLinq
interns at load), query-specific codegen.

## Open questions for RCCL owner

- Compile cost: generated code builds under `cargo oxide` (minutes per
  binary, `RCCL_COMPILER_DSL.md:150-178`). Any guidance for many small
  generated kernels per query (shared monomorphizations, caching, TODO:125)?
- Toolchain: MiniLinq pins stable 1.97.1; RCCL needs nightly-2026-08-28. The
  GPU feature workspace will pin RCCL's toolchain; confirm no host-side APIs
  require nightly in the caller beyond the build.

## Implementation and performance plan

Prepared 2026-10-14. R1 implementation and SM89 correctness are exercised below.
Performance gates and R2–R5 integration remain open.

### Sequence and ownership

1. **R1: sorted search.** Add the ordinary range-facing API in `rccl-thrust`,
   backed by library-owned kernels using existing range lowering and
   `CudaCompletion`. Follow execution-first free-function conventions, with
   natural-order and comparator forms and synchronous/asynchronous execution.
   `equal_range` writes separate begin/end index ranges usable directly by R2.
2. **R2: expansion.** Reuse typed scan workspace and gather. Build checked
   counts and a `P + 1` offset array once; retain offsets and total for
   repeated materialization. Output separate probe-ID and row-ID ranges (or
   their existing writable zip), not a new query-runtime object. Expansion
   consumes begin/end ranges; it does not need probe payloads as a receiver.
   Start with output-parallel owner search as a correctness/reference path;
   use tiled/merge-path work distribution if required to pass the performance
   gate. Do not label binary search per output as merge-path.
3. **R3/R5: integration.** Document and verify order-preserving packing of
   2–4 full-width `u32` columns: two fit in `u64`, three/four require `u128`,
   most-significant column first. Use comparison sorting for `u128`; current
   radix sorting does not support it. Exercise search, expand, gather, sort,
   unique and supported key-only set operations together against a CPU oracle.
4. **R4: deferred lane.** Wide-key hashing does not block the sorted path.
   No multimap or wide-key support is claimed by the existing packed-value map.

Relational selection, semi/anti interpretation, query code generation and
fixpoint semantics stay in MiniLinq/SRDatalog. Do not add an RCCL compiler.

### Contracts to freeze before kernels

- Search indices are `u64` insertion positions relative to the supplied
  haystack view, in `[0, len]`; output lengths equal the needle count.
  Any later narrow-index path must reject nonrepresentable positions.
- Haystack order and search comparator must agree. Comparator equivalence
  defines equal ranges. Floating-point callers must supply a consistent
  total-order comparator; do not silently choose a NaN policy.
- Expansion offsets start at zero, have length `P + 1`, and end at total `M`.
  Upper-bound owner selection must skip repeated offsets from zero fanout.
  Output order is probe-major, then ascending matched-row index.
- Reject mismatched begin/end lengths, `begin > end`, overflowing total or
  index arithmetic, and insufficient output capacity. Existing wrapping
  scans alone do not satisfy the total-count contract. Validation/status
  handling must remain part of the measured production path.
- Async completion retains input/output, comparator, workspace and execution
  borrows. Reading a host total is an explicit completion boundary; device
  totals and caller-provided output capacity must not require a hidden
  allocation/readback for each expansion.

### Correctness gates

- Search: empty/singleton inputs, duplicates, all-hit/all-miss, boundary
  misses, ascending/descending comparators, comparator captures, subranges,
  packed keys and row keys; CPU lower/upper-bound oracle.
- Expansion: empty/all-zero fanout, leading/interior/trailing zero counts,
  repeated ranges, nonzero begins, malformed ranges, overflow and capacity
  failures. Include one probe with 1,000,000 matches plus many singletons.
- Integration: emitted pairs and gathered columns match a CPU join oracle;
  distinct rows match `BTreeSet`, without hash-only tuple equality.
- Preserve regressions for consumer-visible behavior; run GPU examples,
  memcheck/racecheck and pending-borrow/error-path checks. Compilation alone
  does not close a GPU acceptance row.

### Equal-work performance gates

Use pinned CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`,
the existing benchmark/script conventions, and an exclusive GPU. Record
GPU/SM, compiler/backend identity, CUDA versions, inputs and seeds.

| Stage | Reference | Required measurements |
|---|---|---|
| R1 search | Pinned Thrust batched lower/upper bound; compose both for equal-range | Each operation separately, identical comparator, index width, output and stream |
| R2 offsets | Pinned CUB scan plus equivalent checked count/overflow handling | Preparation time, temporary bytes, total publication |
| R2 expansion | Explicit C++ output-parallel reference; optimized tiled/merge-path comparison before claiming optimized parity | Expansion-only and preparation-plus-expansion, identical stable pair outputs |
| R3/R5 pipeline | Equivalent Thrust/CUB packed-key sort/unique/gather composition | End-to-end time and component breakdown, equal output semantics |

There is no assumed CCCL `expand_ranges` primitive: label custom C++
references as such, and audit their algorithm before using them as a parity
target.

- Cover small latency-bound inputs and large throughput-bound inputs,
  uniform, duplicate-heavy, miss-heavy and skewed fanout. Fix probe count,
  total output count, key/index widths and gather work across skew comparisons.
- Preallocate reusable storage for both paths; report cold allocation/setup
  separately. Use identical warmups, samples, synchronization and reset work.
  Alternate reference/RCCL execution order across seven rounds; retain all
  results and report the paired-round median ratio and spread.
- Initial parity gate: RCCL/reference time ratio <= 1.001 for each required
  case, consistent with the existing strict parity policy. Record failures as
  performance gaps; do not average away a failing case or weaken semantics.
- Additional expansion skew gate: skewed/uniform expansion time <= 1.10 at
  fixed probe/output counts, including the million-match case. This is a
  chosen acceptance budget, not an observed result.
- Inspect generated PTX/SASS, register counts, local memory, spills, launch
  counts and temporary bytes when a gate fails. Optimize production code,
  not only the benchmark. Shared-GPU runs cannot establish parity.
- First target: SM89, matching existing performance references. SM86 requires
  its own native correctness/performance evidence; compile-only results
  remain explicitly separate.

### Build and artifact preparation

RCCL's `.cargo/config.toml` sets `CUDA_OXIDE_PTX_DIR` to workspace-relative
`target/oxide-ir`, overridable by the environment. Use a distinct absolute
directory per concurrent build configuration. This organizes intermediates;
it does not implement a compiled-query cache or change the compiler pin.

The GPU build and its host dependencies require the pinned nightly; ordinary
caller syntax does not imply stable-toolchain compatibility. Keep benchmark
compile units focused on required key/comparator families. Record cold/warm
compile time and peak memory separately from GPU performance; no shared
monomorphization or query-cache benefit is assumed.

### R1 execution evidence (2026-10-14)

Implemented all natural/comparator synchronous/asynchronous lower-bound,
upper-bound and equal-range functions in `rccl-thrust/src/search.rs`, using
allocation-free range kernels. Outputs are exact-length `u64` ranges;
completion retains ranges and comparator state. The correctness example
covers packed `u128`, array/zipped rows, floating total order, captured and
descending comparators, composed/subranges, empty/duplicate/miss inputs,
length/lowering errors and completion drop.

- `cargo test -p rccl-thrust --doc`: 9 passed, including pending-borrow
  compile-fail coverage.
- `bash scripts/test-sorted-search.sh`: SM89 CPU differential matrix passed;
  memcheck 0 errors, racecheck 0 hazards; isolated range OOB surfaced through
  completion as CUDA launch error 719.
- Initial seven-round comparison retained in
  `target/sorted-search-performance.fp3XXV`: 23/36 cases failed the unchanged
  1.001 gate. Large lower/upper ratios were approximately 1.04–1.38; large
  equal-range ratios 1.95–2.14. These local measurements are diagnostic:
  desktop graphics were active, so they do not establish exclusive-GPU parity.
- SM89 `ptxas -v` for both benchmark kernels: 22 registers, zero stack frame,
  zero spill stores/loads. Spills do not explain the observed gap.

### R1 production performance correction (2026-10-14)

Search launches now use 512-item tiles (256 threads, two contiguous stripes)
with a bounded grid-stride fallback. Both bounds in fused equal-range search
start at zero instead of narrowing the upper search to each lane's lower
bound. Partial-tile additions are guarded by subtraction to avoid overflow.

- Expanded correctness boundaries include 511/512/513 and 1023/1024/1025.
  The SM89 oracle matrix, memcheck (0 errors), racecheck (0 hazards) and
  isolated async OOB propagation passed again.
- Direct execution of the production PTX passed 48 additional CPU-oracle
  cases with forced 1/2-block grids, testing repeated tiles and partial tails.
- Seven-round results: `target/sorted-search-performance.HKcOaZ`. Large
  lower-bound ratios are 0.972–1.003, upper-bound 0.988–1.000, and equal-range
  0.810–0.950. All 12 equal-range cases pass the numerical gate.
- The unchanged 1.001 overall gate still fails 12/36 cases: 11 small bound
  cases and large descending-uniform lower bound (1.002963). Desktop graphics
  remain active; no exclusive-GPU parity claim is made.
- Updated SM89 resources: bound kernel 24 registers, equal-range 28;
  both have zero stack frames and zero spill stores/loads.

Next performance work: establish exclusive-GPU timing for the remaining
latency-bound cases before attributing their small differences to kernels.
No threshold or benchmark semantics were relaxed. SM86 native behavior,
packed/row performance and LE expansion remain unverified.
