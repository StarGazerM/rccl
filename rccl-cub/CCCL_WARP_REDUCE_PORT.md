# Pinned CCCL WarpReduce port

This slice is tied to CCCL commit
`571f2fc3bc53cd710e306ad43d58995c1fe4219f` and these exact sources:

- `cub/cub/warp/warp_reduce.cuh`
  (`92e55aa35a74c0faca2de3ba609b27167ead04062533c20976dc20fd16734a95`);
- `cub/test/warp/catch2_test_warp_reduce.cu`
  (`25bec1964bd85f8e9b1000af379b636218aebd713c42719cdf23e03b502c5295`).

The runnable Rust port is `examples/warp_reduce.rs`; the ordinary Cargo unit
target is `tests/warp_reduce_upstream.rs`. The GPU port uses only the public
operation macros:

```rust,ignore
let aggregate = rccl_cub::warp_reduce!(
    32,
    thread_value,
    |left, right| left + right,
);
```

The first operand is the closed compile-time logical-warp width. The remaining
operands are ordinary Rust expressions; rustc infers their types and
monomorphizes the closure. There is no public operator type, algorithm policy,
temporary-storage type, runtime registry, or device lambda substitute.

## Correspondence

| Pinned CCCL evidence | Rust executable evidence |
|---|---|
| `warp_reduce.cuh` overview: four 32-lane sums over `0..128` | `overview`; asserts exactly `496, 1520, 2544, 3568` |
| `WarpReduce::Sum`, full logical warp | `overview`, `i64_sum`, `u8_sum` |
| custom binary operation | closures passed directly to both public macros |
| partial logical warp with uniform `valid_items` | `partial_sum`; independently folds only the first 13 lanes |
| logical warp smaller than 32 | `logical_16_max`; checks all eight logical warps |
| four items per thread | `multiple_items`; `[T; 4]` replaces the C++ fixed-size range |
| built-in scalar matrix | focused `u8`, `i32`, and `i64` GPU cases exercise narrow, native-32-bit, and two-register exchange |

The host oracle does not call RCCL code. Every GPU output checked here is read
from a real device buffer, and every logical lane-zero result is compared, not
a sample or checksum. RCCL emits only Rust; cuda-oxide exclusively owns
everything after that Rust source.

## Explicitly not claimed yet

The pinned Catch2 file also covers logical widths `9` and `7`, custom/vector
records, `half`/`bfloat16`, and a wider combinatorial product. cuda-oxide's safe
`WarpTile` supports only `1, 2, 4, 8, 16, 32`; CUB implements non-power-of-two
widths through shared memory. Those cases remain unimplemented until shared
memory can be owned without emitting `static mut`, raw pointers, or unsafe
code. Head/tail segmented reduction and `WarpReduceBatched` are separate
unimplemented public operations. No compatibility stub or compile-only example
stands in for them.

Use `scripts/test-warp-reduce.sh` to build the feature-gated target through the
standard cuda-oxide workflow and execute it. The feature keeps the device
example out of ordinary host-only Cargo target discovery; it is always enabled
by the GPU gate and is not a skip path.
