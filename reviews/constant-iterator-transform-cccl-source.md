# Constant-iterator transform correspondence and masking review

Date: 2026-08-26

Pinned sources:

- CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`;
- `thrust/examples/constant_iterator.cu`;
- `thrust/testing/constant_iterator.cu`.

The public Rust operation is `rccl_thrust::transform!`; `RepeatN` is its
ordinary finite constant-range operand. The Thrust call macro preserves the
output borrow and closure expression and checks equal extents, then lowers to
the hidden CUB device-transform macro. CUB owns the closed launch shape,
kernel, and cuda-oxide typed prepared launcher. The generated Rust
contains no `unsafe`, raw pointer, `Arc`, runtime type registry, erased
callable, compiler invocation, or output-format choice.

## Masking review

- The exact pinned example runs on the GPU and asserts all four values after a
  device-to-host copy; compilation or a successful launch alone cannot pass.
- The benchmark starts from nonconstant data, performs 2,120 transforms, and
  checks all 8,388,608 outputs against the accumulated increment.
- The benchmark and CCCL reference use identical element type, count,
  initialization, in-place update, repeated operand, warm-up count, sample
  count, launches per sample, synchronization points, and per-element oracle.
- A zero-length early return cannot mask either executable because both use
  nonempty ranges and verify their complete result.
- `RepeatN` construction never executes the user closure on the host. The only
  operational path is the fixed CUB cuda-oxide kernel.
- The catalog marks only this intersection as partial. Other range adaptors
  and transform forms remain planned rather than gaining compile-only or CPU
  examples.
- RCCL emits Rust only. The gate invokes cuda-oxide's standard workflow and
  does not configure its compiler, target, device-code format, or loader.
- The gate rejects Thrust-owned CUDA kernels; the ordinary Thrust function
  lowers through the hidden CUB transform edge.

## Injected faults

All faults are temporary and absent from the final tree.

| Fault | Intended detector | Observed failure |
|---|---|---|
| write the original value instead of `op(value, repeated)` | pinned four-element executable | exact result assertion fails (`[3, 7, 2, 5]` instead of `[13, 17, 12, 15]`) |
| use one item per thread while retaining the equal-work benchmark | performance comparison | result remains correct, but the RCCL/CCCL timing ratio exceeds the measurement-equivalence band |

The final eight-items-per-thread grid-stride form remains safe Rust and is
faster than the pinned Thrust reference on the tested SM89 device. The one
one-tenth-percent event-timing band accounts for device clock variation; the seeded
schedule fault is larger than that band, so it cannot be hidden by the
tolerance.

## Remaining upstream cases

Constant-to-output copy, unary transform, two-constant-input transform,
constant-range reduction, and iterator-position compatibility are not
implemented. The review therefore makes no claim about their result or
performance.
