# Current implemented-subset performance

Date: 2026-08-26

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
