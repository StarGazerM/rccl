# Pinned CCCL WarpReduce port

This slice is tied to CCCL commit
`571f2fc3bc53cd710e306ad43d58995c1fe4219f` and these exact sources:

- `cub/cub/warp/warp_reduce.cuh`
  (`92e55aa35a74c0faca2de3ba609b27167ead04062533c20976dc20fd16734a95`);
- `cub/test/warp/catch2_test_warp_reduce.cu`
  (`25bec1964bd85f8e9b1000af379b636218aebd713c42719cdf23e03b502c5295`).

The runnable Rust ports are `examples/warp_reduce.rs` for the documented
overview and focused cases, `examples/warp_reduce_catch2.rs` for the 144
integer Catch2 executions, and `examples/warp_reduce_catch2_f32.rs` for the
six floating executions. The ordinary Cargo unit target is
`tests/warp_reduce_upstream.rs`. The GPU ports use only the public operation
macros:

```rust,ignore
let aggregate = rccl_cub::warp_reduce!(::<i32, 32>(
    thread_value,
    |left, right| left + right,
));
```

The macro-visible arguments are the C++ template facts: scalar type and
logical-warp width. `warp_reduce_items!(::<T, WIDTH, ITEMS>(...))` also makes
the C++ array extent explicit, because a proc macro cannot perform the type
deduction that C++ performs for `T (&)[ITEMS]`. The proc macro annotates the
ordinary Rust closure with `T`, selects the scalar-specific shuffle/redux
form, and emits the whole safe expression. There is no public operator type,
algorithm policy, temporary-storage type, runtime selection trait, registry,
or device-lambda substitute. Literal widths forwarded through an ordinary
declarative Rust macro are accepted; runtime width expressions are rejected.

## Correspondence

| Pinned CCCL evidence | Rust executable evidence |
|---|---|
| `warp_reduce.cuh` overview: four 32-lane sums over `0..128` | `overview`; asserts exactly `496, 1520, 2544, 3568` |
| `WarpReduce::Sum`, full logical warp | `overview`, `i64_sum`, `u8_sum` |
| custom binary operation | closures passed directly to both public macros |
| partial logical warp with uniform `valid_items` | `partial_sum`; independently folds only the first 13 lanes |
| logical warp smaller than 32 | `logical_16_max`; checks all eight logical warps |
| four items per thread | `multiple_items`; `[T; 4]` replaces the C++ fixed-size range |
| built-in scalar matrix | `warp_reduce_catch2`: all `u8`, `u16`, `i32`, and `i64` sum/max/min rows at supported widths `32`, `16`, and `1` |
| two generated partial extents | `warp_reduce_catch2`: two executions for every supported builtin type/operation/width row |
| multiple-items matrix | `warp_reduce_catch2`: all supported builtin type/operation/width rows with `[T; 4]` |
| floating redux matrix | `warp_reduce_catch2_f32`: exact `f32` max/min comparisons at supported widths |

The host oracle does not call RCCL code. Every GPU output checked here is read
from a real device buffer, and every logical lane-zero result is compared, not
a sample or checksum. RCCL emits only Rust; cuda-oxide exclusively owns
everything after that Rust source.

`examples/warp_reduce_benchmark.rs` and
`benchmarks/warp-reduce-reference.cu` execute an identical 8,388,608-value,
256-thread-block, four-independent-reductions-per-value, 20-warm-up,
21-sample, 100-launch-per-sample workload. The
addition closure is recognized structurally by the operation macro and the
safe Rust backend lowers the full 32-lane `i32` sum to the same single
`redux.sync.add.s32` collective used by pinned CUB on SM80 and newer. Both
benchmarks independently validate every lane-zero result after timing.

## Explicitly not claimed yet

The pinned Catch2 file also covers logical widths `9` and `7`, custom/vector
records, and `half`/`bfloat16`. cuda-oxide's safe
`WarpTile` supports only `1, 2, 4, 8, 16, 32`; CUB implements non-power-of-two
widths through shared memory. Those cases remain unimplemented until shared
memory can be owned without emitting `static mut`, raw pointers, or unsafe
code. Head/tail segmented reduction and `WarpReduceBatched` are separate
unimplemented public operations. No compatibility stub or compile-only example
stands in for them.

Use `scripts/test-warp-reduce.sh` to build the feature-gated target through the
standard cuda-oxide workflow and execute it. The feature keeps the device
example out of ordinary host-only Cargo target discovery; it is always enabled
by the GPU gate and is not a skip path. The equal-work comparison is part of
`scripts/test-current-performance-parity.sh`.
