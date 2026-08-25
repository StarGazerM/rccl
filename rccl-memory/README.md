# rccl-memory

`rccl-memory` is the stream-ordered device-memory layer shared by RCCL
algorithms and applications. Its design follows RMM's resource composition but
uses Rust ownership for resource and allocation lifetimes.

## Resource model

| type | allocation path | pooling behavior |
| --- | --- | --- |
| `CudaMemoryResource` | `cuMemAlloc` / `cuMemFree` | none; conservative synchronous fallback |
| `ManagedMemoryResource` | `cuMemAllocManaged` / `cuMemFree` | CUDA Unified Memory |
| `PinnedHostMemoryResource` | `cuMemAllocHost` / `cuMemFreeHost` | page-locked host memory with device access |
| `SystemMemoryResource` | Rust aligned system allocation | pageable HMM/ATS memory with capability checks |
| `SamHeadroomMemoryResource` | Rust aligned system allocation plus CUDA advice | HMM migration while reserving GPU headroom |
| `CudaAsyncMemoryResource` | `cuMemAllocAsync` / `cuMemFreeAsync` | CUDA device's current/default pool |
| `CudaPoolMemoryResource` | `cuMemAllocFromPoolAsync` | private configurable CUDA driver pool |
| `CudaPoolViewMemoryResource` | shared existing CUDA pool | lifetime-safe view retaining the source pool |
| `PoolMemoryResource` | arbitrary upstream resource | stream-ordered coalescing best-fit suballocation |
| `FixedSizeMemoryResource` | arbitrary upstream resource | preallocated uniform blocks |
| `BinningMemoryResource` | size-class resource dispatch | fixed-size bins plus upstream fallback |
| `TrackingMemoryResource` | delegates to an upstream resource | byte/allocation counters around any resource |
| `LimitingMemoryResource` | delegates to an upstream resource | atomic live-byte cap |
| `FailureCallbackMemoryResource` | delegates to an upstream resource | callback-controlled allocation retry |
| `CallbackMemoryResource` | captured Rust closures | application/FFI-defined allocation policy |
| `AlignedMemoryResource` | delegates with over-allocation | alignments stronger than CUDA's 256-byte guarantee |
| `StatisticsMemoryResource` | delegates to an upstream resource | nested RMM-style counters |
| `LoggingMemoryResource` | delegates to an upstream resource | RMM-compatible CSV allocation actions |
| `PrefetchMemoryResource` | delegates to an upstream resource | managed-memory prefetch on the allocation stream |
| `ThreadSafeMemoryResource` | delegates under a mutex | serialization for non-thread-safe FFI resources |

Every implementation satisfies `DeviceAllocator`. Containers and adaptors are
generic over a concrete allocator handle, so normal composition uses static
dispatch: `TrackingMemoryResource<PoolMemoryResource<CudaMemoryResource>>`.
The trait does not impose `Debug`, `Send`, or `Sync`; APIs add those bounds only
when their operation actually needs them.

`DeviceBuffer<T, A>` is a typed allocation with RMM-style reserve, resize,
shrink, and fallible deep-copy operations. Like `Vec<T, A>`, it stores `A` by
value. Passing `&'pool Pool` makes the allocator lifetime part of the buffer's
type and prevents the pool from being dropped or moved while the buffer is
alive. Passing `Arc<Pool>` explicitly opts into shared ownership for a
`'static` or cross-task buffer; it is not required by the API. The buffer also
stores its last-use stream, frees in stream order on `Drop`, and implements
`DeviceSlice<T>` / `DeviceSliceMut<T>` for algorithm interoperation. The same
traits are implemented for `cuda_core::DeviceBuffer<T>`.

`DeviceUVector<T, A>` names the uninitialized typed-vector contract and
`DeviceScalar<T, A>` supplies one-value set/get ownership. The optional
process-wide compatibility registry lives under `rccl_memory::registry`. Its
`SharedDeviceAllocator` is the sole intentionally erased allocator handle,
because an open runtime registry cannot store arbitrary concrete types.

```rust,ignore
use cuda_core::CudaContext;
use rccl::memory::{
    CudaPoolMemoryResource, DeviceBuffer, PoolConfig, TrackingMemoryResource,
};

let context = CudaContext::new(0)?;
let stream = context.default_stream();
let cuda_pool = CudaPoolMemoryResource::new(
    context.clone(),
    PoolConfig::default(),
)?;
let tracked = TrackingMemoryResource::new(&cuda_pool);

let exec = rccl::cuda::par_in(&stream, &tracked)?;
let mut keys = DeviceBuffer::<u32, _>::zeroed_in(
    &tracked,
    stream.clone(),
    1 << 20,
)?;

rccl::sort(&exec, &mut keys)?;
```

The compilable end-to-end form is
[`examples/resource_sort.rs`](../examples/resource_sort.rs). The standalone
private-pool/statistics smoke test is
[`examples/private_pool.rs`](examples/private_pool.rs).

## Stream ownership

An asynchronous allocation is only immediately usable in the stream that
ordered it. A resource-owned buffer therefore remembers its current stream.
Before using it from another stream, call `buffer.transition_to(new_stream)`.
That records an event in the old stream and orders the new stream after it.
RCCL checks this invariant at its trait boundary. RCCL workspaces transition
their internal buffers automatically because the workspace is mutably borrowed.

## Why a custom trait instead of `std::alloc::Allocator`?

Rust's `Layout` is reused because it is the right representation of byte size
and alignment. `std::alloc::Allocator` itself is not used: it models
host-addressable pointers, has no CUDA context or stream argument, and cannot
express stream-ordered allocation/deallocation. `DeviceAllocator` is the
corresponding device-space contract.

No macro is required for allocation or algorithm interoperation. Traits express
the open extension points, generic RCCL functions monomorphize normal buffer
calls, and borrowed allocator handles carry their resource lifetime in the
container type. `Arc<A>` is available when shared ownership is required. Only
the optional process-wide registry uses `SharedDeviceAllocator`, an explicitly
type-erased handle needed for open-ended runtime replacement. Macros remain
appropriate for closed compile-time policy generation (for example, RCCL's
declarative per-SM sort schedule), not memory ownership.

## RMM parity and benchmarks

[`RMM_PARITY.md`](RMM_PARITY.md) pins the upstream RMM commit and records every
resource, container, test, and performance gate. Run the deterministic
allocation trace in release mode, preserving its one-line JSON output as a
benchmark artifact:

```bash
cargo run --release -p rccl-memory --example allocation_trace -- \
  --resource=pool --allocations=10000 --max-allocation=1048576 \
  --max-live-bytes=268435456 --warmups=2 --repetitions=7 --seed=24301
```

Valid resources are `cuda`, `cuda_async`, `cuda_pool`, and `pool`. The seed,
trace shape, warm-up count, GPU, and driver must be identical when comparing
against the pinned RMM baseline.
