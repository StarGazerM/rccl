# Pinned CCCL constant-iterator transform port

This executable slice is tied to CCCL commit
`571f2fc3bc53cd710e306ad43d58995c1fe4219f` and these exact sources:

- `thrust/examples/constant_iterator.cu`
  (`fd7953b9bde0ce933106cbe54fab7434b25903d6db774c6d8b8cc14c5cfd242b`);
- `thrust/testing/constant_iterator.cu`
  (`40fdf9632920c0c89fab7c2ddcd2a5d524c09dbccce3ea6302e7b544f61be88a`).

The runnable Rust port is `examples/constant_iterator.rs`. It keeps the
upstream input, repeated value, binary addition, in-place output, and exact
result:

```rust,ignore
let mut data = DeviceBuffer::from_host(&stream, &[3_i32, 7, 2, 5])?;
let repeated = RepeatN::new(10_i32, data.len());
rccl_thrust::transform!(&exec, &mut data, repeated, |left, right| left + right)?;
assert_eq!(data.to_host_vec(&stream)?, [13, 17, 12, 15]);
```

`transform!` is a shallow Rust operation macro because its inferred element
and closure types select and instantiate device code. It validates the Thrust
range shape and lowers to the hidden CUB device-transform operation macro;
CUB owns the kernel and launch shape. `RepeatN` is ordinary Rust support, not
a macro. The operation receives a normal Rust closure; RCCL has no
function-object compatibility type or runtime callable erasure.

## Correspondence

| Pinned CCCL evidence | Rust evidence |
|---|---|
| example input `{3, 7, 2, 5}` | the same four `i32` values in the GPU executable |
| `constant_iterator<int>(10)` | finite `RepeatN::new(10_i32, data.len())` |
| binary in-place `transform` with addition | `transform!` with `|left, right| left + right` |
| result `{13, 17, 12, 15}` | every copied-back element is asserted exactly |
| constant iterator is trivially copyable | `repeat_n_is_copy_and_keeps_value_and_extent` |
| input/random-access iterator traits | `ReadRange<Item = i32> + RandomAccessRange` compile-time test |
| distance `10_000_000_000` | full-width `usize` extent test on the supported 64-bit host |

`examples/constant_iterator_benchmark.rs` and
`benchmarks/thrust-transform-constant-reference.cu` execute the same
8,388,608-element, 20-warm-up, 21-sample, 100-launch-per-sample workload. Both
validate every result after the timed launches. The Rust kernel uses 256
threads and eight grid-stride items per thread, matching the pinned Thrust
transform policy's work granularity without copying its C++ policy types into
the runtime API.

## Explicitly not claimed yet

This is the implemented intersection of binary in-place transform and one
finite constant input range. The pinned unit file also tests iterator
positions/comparisons, constant-to-output copy, unary transform, two constant
inputs, and reduce. RCCL does not export those operations yet. `RepeatN` is a
finite device-range expression, not a C++ iterator-position compatibility
object, so unsupported rows are not represented by host-only lookalike tests.

Use `scripts/test-thrust-constant-iterator.sh` for the pinned source/result
gate and `scripts/test-current-performance-parity.sh` for the equal-work
performance gate.
