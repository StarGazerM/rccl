# WarpReduce upstream correspondence and masking review

Date: 2026-08-26

Pinned sources:

- CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`;
- `cub/cub/warp/warp_reduce.cuh`;
- `cub/test/warp/catch2_test_warp_reduce.cu`.

Public Rust calls under test are `rccl_cub::warp_reduce!` and
`rccl_cub::warp_reduce_items!`. They expand to safe Rust expressions calling a
closure-based device helper. The generated expressions contain no `unsafe`,
raw pointer, `static mut`, `Arc`, erased callable, operator policy, or compiler
artifact control.

## Masking audit

- The GPU executable asserts results; it cannot pass after only compiling or
  launching.
- No zero-length, skipped-launch, unsupported-device early return, CPU
  fallback, or ignored `Result` exists.
- Inputs and expected aggregates are deliberately nonzero. The `i64` values
  exercise both 32-bit halves, and the narrow integer case uses explicit
  wrapping arithmetic.
- All valid logical-warp lane-zero outputs are checked: four groups for width
  32, eight groups for width 16, and all 128 groups for width 1. Other lanes
  are not checked because pinned CUB promises the aggregate only in logical
  lane zero.
- The Catch2 matrix executable performs 150 GPU comparisons: all supported
  `u8`/`u16`/`i32`/`i64` sum/max/min full, two-partial-extent, and
  four-items-per-thread rows, plus supported `f32` max/min rows.
- The host oracle is an independent sequential fold and does not call RCCL.
  The overview also asserts CCCL's literal documented values `496`, `1520`,
  `2544`, and `3568` before comparing with the general oracle.
- The feature-gated cuda-oxide example is always enabled by
  `scripts/test-warp-reduce.sh`; ordinary Cargo tests execute the separately
  registered upstream-oracle unit target.
- Every injected implementation fault caused cargo-oxide to report that it
  recompiled `rccl-cub`, ruling out a stale executable for these checks.
- RCCL emits Rust only. The gate uses cuda-oxide's standard build workflow and
  does not configure its compiler or output.

## Injected faults

All faults were temporary and removed before the final gate.

| Fault | Intended detector | Observed failure |
|---|---|---|
| Change partial inclusion from `rank + offset < valid_items` to `<=` | partial GPU comparison | exit 101; got `[91, 539, 987, 1435]`, expected `[78, 494, 910, 1326]` |
| Start the fixed-array local fold at item 2, skipping item 1 | multiple-items GPU comparison | exit 101; all four logical-warp aggregates differed (`6112` vs `8128` for the first) |
| Shuffle the low 32 bits twice when reconstructing a 64-bit value | `i64` GPU comparison | exit 101; all four aggregates were corrupted, including `-6390930489259177520` vs `-1552004656` |
| Truncate each `u16` lane to `u8` before shuffling | new builtin-type matrix | overview executable passed, then matrix exited 101 in case 0; first group was `1008` instead of `36080` |
| Return the local `f32` value instead of shuffling a neighbor | new floating redux matrix | overview executable passed, then matrix exited 101 in floating case 0; first group was `-95.0` instead of `99.0` |

Parser negative tests independently reject runtime tile widths, unsupported
widths `9` and `7`, missing operands, and extra operands. Generated-expression
audits independently reject unsafe/raw-pointer syntax. A parser regression test
also proves that an integer literal forwarded through `macro_rules!` remains
accepted rather than being mistaken for a runtime width.

## Remaining upstream cases

This is intentionally partial, not full `WarpReduce` parity. The pinned test's
non-power-of-two widths `9` and `7`, vector/custom records, half/bfloat types,
segmented operations, and batched family remain unimplemented. They are not
represented by stubs, disabled assertions, or compile-only examples.
