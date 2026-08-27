# Current implemented-subset performance

Date: 2026-08-27

Reference: CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`  
Backend: pinned cuda-oxide `b22efa99e8383e2a349c4a3af60d8aa083973890`  
Runtime device: SM89

`scripts/test-current-performance-parity.sh` builds the Rust programs through
the standard cuda-oxide workflow and independently compiles the two pinned
CCCL reference programs. Each pair has identical element type, element count,
initialization, launch work, warm-ups, samples, repeated launches, event
synchronization, and full result oracle.

| Implemented intersection | RCCL minimum median | CCCL minimum median | RCCL / CCCL | Gate |
|---|---:|---:|---:|---:|
| CUB 32-lane `i32` WarpReduce sum, four independent reductions per value | 0.017395 ms | 0.017379 ms | 1.000921 | at most 1.001 |
| Thrust binary in-place transform plus constant range | 0.013955 ms | 0.017046 ms | 0.818667 | at most 1.001 |

The same strict gate now measures every implemented ordering path. The table
records the median of five independently measured per-process medians for
RCCL and CCCL, not a favorable single sample. The bit-window rows exercise
the public borrowed `DoubleBuffer` forms directly.

| Implemented ordering intersection | RCCL / CCCL | Gate result |
|---|---:|---:|
| CUB DeviceMerge keys | 1.000000 | pass |
| CUB DeviceMerge pairs | 1.009718 | **fail** |
| CUB DeviceMergeSort keys | 1.006900 | **fail** |
| CUB DeviceMergeSort pairs | 0.998203 | pass |
| CUB DeviceRadixSort keys | 1.001751 | **fail** |
| CUB DeviceRadixSort pairs | 1.017864 | **fail** |
| CUB DeviceRadixSort keys `DoubleBuffer` 0..24 | 1.005168 | **fail** |
| CUB DeviceRadixSort pairs `DoubleBuffer` 0..24 | 1.001041 | **fail** |
| Thrust merge | 1.000398 | pass |
| Thrust merge-by-key | 1.000000 | pass |
| Thrust sort | 1.019154 | **fail** |
| Thrust sort-by-key | 0.999147 | pass |

Five of twelve ordering paths are inside the 1.001 measurement-equivalence
band in this run. Some repeated-run ratios move across the boundary even when
the normalized cuda-oxide kernel instructions are unchanged, but the gate
does not classify that observation as a pass or widen its band. The key-only
radix and Thrust-sort gaps remain repeatable, and Thrust preserves its required
`par` synchronization. Neither the threshold nor that synchronization is
weakened to make the current test set pass.

The full-range and 0..24 operations now select the same named primitive Rust
function items. Before that change, identical zero-capture closure expressions
at different call sites had distinct Rust types and produced duplicate
cuda-oxide kernel specializations. The mixed benchmark now has one histogram,
one keys one-sweep, and one pairs one-sweep specialization for its common u32
policy. Round-trip, descending, every valid digit-window shape, and floating
signed-zero tests prevent that structural deduplication from masking changed
semantics.

The WarpReduce operation macro selects the same hardware integer-add
collective as pinned CUB. Four independent reductions per loaded value keep
the benchmark sensitive to that selection while amortizing the unavoidable
checked-slice boundary. Replacing the specialization with the correct but
slower shuffle tree preserves every result and changes the ratio to 2.011273,
so the 0.1% measurement band cannot mask that implementation fault.

The Thrust workload uses 8,388,608 `u32` values and the same eight-items-per-
thread work granularity as the pinned transform policy. RCCL's safe
grid-stride mapping is faster on the tested device. Replacing it with a
one-item-per-thread schedule crosses the performance gate even though the
result remains correct.

These are measured equivalence/no-regression claims, not a claim that two
independent event samples have bit-identical elapsed-time values. Result
parity is exact: both benchmarks validate every output element after timing.

## Remaining key-only radix code-generation gap

The Rust kernel follows pinned CUB's Policy800 topology: 384 threads, 21 keys
per thread, warp-private histograms, early counts, raking-memoize scan,
eight-ballot `MatchAny`, decoupled lookback, short-circuiting, and direct
scatter. Its accepted SM89 one-sweep key kernel uses 77 registers, 34,304
bytes of shared storage, no spills, and one barrier; pinned CCCL uses the same
shared-storage extent and 75 registers.

The remaining emitted-code differences have safe-backend ownership boundaries:

- pinned CUB's `MatchAny<8>` uses an inline-PTX predicated `not`; the same
  eight-ballot algorithm expressed in safe Rust lowers through cuda-oxide to
  2,568 SASS instructions with 156 `SEL`s, versus 2,392 instructions for CCCL;
- pinned CUB uses `__syncthreads_or`; the current cuda-oxide safe device API
  exposes only the plain CTA barrier, so RCCL uses one cell from CUB's otherwise
  idle guard allocation plus a barrier; and
- pinned CUB offsets its bins/counter pointers before launch, while the current
  synchronous cuda-oxide launcher accepts whole `DeviceBuffer`s, so RCCL keeps
  two uniform offsets as kernel arguments.

The rejected alternatives are part of the regression evidence. A branch-free
Rust arithmetic mask reduced `SEL` count from 156 to 9, but raised register use
from 77 to 79, replaced the selects with 168 additional integer additions, and
measured 0.587936 ms versus the accepted form's 0.586752 ms sample. Hardware
`match.any`, handwritten assembly, raw pointer views, async/`Arc` launch
plumbing, deleting the short-circuit, and relaxing the ratio are all rejected:
they change the pinned algorithm, the Rust safety boundary, or the test rather
than repairing the implementation.
