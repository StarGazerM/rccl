# CCCL merge compatibility audit

Audit baseline: NVIDIA/cccl commit
`571f2fc3bc53cd710e306ad43d58995c1fe4219f` (2026-08-22).

RCCL follows CUDA `thrust::merge` lowering through
`thrust/system/cuda/detail/merge.h` into `cub::DeviceMerge`. It is a native
Rust/cuda-oxide implementation and does not call or link CUDA C++. This file
records the implemented boundary and measured cases; it is not a claim of
complete CCCL parity. The release-level defect ledger is
[`STRICT_CCCL_API_AUDIT.md`](STRICT_CCCL_API_AUDIT.md).

## Public Thrust facade

Applications call ordinary functions with one shared Thrust execution policy:

```rust
let exec = rccl::cuda::par(&stream)?;
let end = rccl::merge(&exec, &first, &second, &mut output)?;
```

The contiguous-range facade consists of `merge`, `merge_by`, `merge_by_key`,
and `merge_by_key_with`. Rust cannot overload by arity, so `_with` denotes the
key/value operation with an explicit ordering closure. Descending order passes
`|left, right| right.cmp(left)` to the closure-taking forms.

`Execution<'stream>` belongs to `rccl-thrust` and borrows
`&'stream CudaStream`. `par_in`
also carries a concrete allocator, including a borrowed allocator handle. It is
the same value accepted by sort and contains no merge policy, target schedule,
or strategy type.

Public operations synchronize before returning. If a later allocation or
launch fails after earlier work was queued, Thrust still attempts
synchronization before releasing the borrowed inputs and outputs. The functions
return the logical output end, or both key and value ends for a key/value merge.
Equivalent records from the first range precede equivalent records from the
second range. Output ranges must not overlap either input; contiguous calls
diagnose overlap.

## CUB dispatch boundary

`rccl-cub` owns the device-wide merge kernels, target schedule, module cache,
and host dispatch. It does not import or accept the Thrust execution policy.
Its hidden enqueue functions receive a CUB kernel context, stream, and
allocator directly and are unsafe: callers must retain every range until all
queued work completes. `rccl-thrust` is the safe wrapper that enforces that
condition with synchronization.

One cargo-oxide target produces one CUB schedule. `rccl-cub/build.rs` validates
`CUDA_OXIDE_TARGET` and emits literal normal and fallback geometry. Host-only
Cargo checks deterministically use SM80. Key/value shape and ordering closure
remain ordinary Rust generics and are monomorphized at the call site.

There is no public schedule language, runtime architecture probe, breakpoint
table, policy object, selected-policy wrapper, launch callback, or
function-pointer table. Scalar register slots above a concrete record shape's
item count disappear during monomorphization; RCCL does not package a matrix of
candidate kernels.

## Kernel and temporary-storage behavior

The implementation has the same broad two-stage device-wide structure as the
pinned CUB merge:

1. A Merge Path partition kernel emits one first-range offset per tile
   boundary.
2. One merge block owns each tile, performs stable per-thread serial merge, and
   writes the coalesced result.

The default SM80+ shape is 512 threads with
`clamp(floor(15 * 4 / (sizeof(key) + sizeof(value))), 1, 15)` items per thread.
When its tile exceeds the portable shared-memory ceiling, the 64x1
specialization is used if that smaller tile fits. Otherwise the normal shape
uses 128-byte-padded global virtual shared memory.

This mirrors the branch order in pinned CUB `choose_merge_agent`, but not yet
its exact storage predicate. RCCL sizes the staged key/value arrays directly;
CUB tests the full `sizeof(Agent::TempStorage)`, including unions, padding, and
block-load/store alternatives. Some wide records can therefore choose fallback
earlier than pinned CUB.

Temporary storage is allocated through the allocator retained by Thrust
execution. The workspace belongs to one dispatch call and is not a public
executor object. CUB shares one merged CUDA module and typed namespace views
per live CUDA context through weak registry entries; the cache stores CUDA
resources, not schedule values.

## Iterator and proxy status

Merge currently accepts contiguous `DeviceSlice` and `DeviceSliceMut` ranges.
Counting, discard, strided, reverse, transform, zip, and permutation adapters
remain missing. A correct Rust design needs typed associated device views and
proxy references whose lifetimes retain every backing range and whose
transitions preserve stream-ordering dependencies. Crate-private raw device
inputs at the final kernel ABI are not a public substitute for that contract.

## GPU matrix

`src/bin/cccl-merge-tests.rs` constructs and validates fixtures in CUDA memory.
It reports exactly three contiguous-range suites:

1. ascending, descending, empty, and state-carrying-closure key merges on a
   non-default stream, including exact output-end values;
2. stable mixed-type key/value provenance;
3. 512-byte fallback keys and 1024-byte virtual-shared-memory keys.

It contains no iterator suite and makes no counting, discard, strided, reverse,
permutation, or `2^32` logical-range claim. The default gate executes the whole
GPU binary and example inventory twice.

## Historical low-level measurement

RTX 6000 Ada (`sm_89`), CUDA 12.9, 16,777,216 sorted elements per input,
50 measured iterations after three warmups. These values predate the current
safe host boundary: the RCCL and CUB rows reused temporary storage, while the
Thrust row invoked its public merge call. Output was validated on the GPU.

| operation | implementation | best | throughput |
|---|---|---:|---:|
| `u32` keys | RCCL low-level merge engine | 0.348 ms | 96.376 Gitem/s |
| | CCCL CUB `DeviceMerge` | 0.348 ms | 96.394 Gitem/s |
| | CCCL `thrust::merge` | 0.357 ms | 94.102 Gitem/s |
| `u32`/`u64` pairs | RCCL low-level merge engine | 1.037 ms | 32.347 Gpair/s |
| | CCCL CUB `DeviceMerge` | 1.046 ms | 32.094 Gpair/s |
| | CCCL `thrust::merge_by_key` | 1.053 ms | 31.875 Gpair/s |

This is optimization evidence for retained kernels, not a measurement of the
current allocation and synchronization boundary. A replacement table needs
fresh RCCL, CUB, and Thrust measurements under one timing contract.

```bash
source scripts/oxide-env.sh
cargo oxide build --arch sm_89 --features gpu-examples
target/release/merge-benchmark --elements-per-input 16777216 --iterations 50
```

## Explicit scope differences

- RCCL's device-wide merge targets SM80 and newer; pinned CUB retains older
  architecture policies.
- One cargo-oxide build contains one target specialization, not a CUB-style
  multi-target fat binary.
- CUDA device-side dynamic-parallelism invocation is not exposed by
  cuda-oxide.
- The complete upstream randomized Catch2/Thrust parameter product has not
  been ported.
- Only SM89 has runtime measurements. Compilation for other targets is not
  hardware evidence.

Run the retained gate with `scripts/test-gpu.sh`.
