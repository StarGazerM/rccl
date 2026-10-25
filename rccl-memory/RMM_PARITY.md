# RMM parity contract

`rccl-memory` treats RAPIDS Memory Manager (RMM) commit
`c65202f52766e12c73e293b16d6a520a069d4afc` as its behavioral and performance
reference. A row is only marked complete when an automated Rust test or
benchmark exercises the same contract. Similar names or CUDA API calls are not
enough.

The upstream snapshot contains 54 C++ test translation units and about 331
GoogleTest declarations under `cpp/tests`. This crate currently has 3 host
unit tests and 29 GPU contract tests. The Rust suite is intentionally
split into host tests, GPU contract tests, and opt-in performance comparisons so
machines without a CUDA device can still run `cargo test`.

## Resource parity

| RMM facility | Rust facility | Status | Required gate |
| --- | --- | --- | --- |
| `cuda_memory_resource` | `CudaMemoryResource` | implemented | allocation, zero-byte, alignment, wrong-context tests |
| `managed_memory_resource` | `ManagedMemoryResource` | implemented | host/device visibility and teardown tests |
| `cuda_async_memory_resource` | `CudaAsyncMemoryResource` | implemented | same/cross-stream reuse tests |
| `cuda_async_view_memory_resource` | `&CudaPoolMemoryResource` | native Rust borrow | source lifetime and allocation tests |
| `cuda_async_managed_memory_resource` | none | missing | managed native-pool tests |
| `cuda_memory_resource` with fabric | none | missing | fabric handle tests on supported hardware |
| CUDA private pool | `CudaPoolMemoryResource` | implemented | configuration, stats, trim, max-size tests |
| `pool_memory_resource` | `PoolMemoryResource` | partial | best-fit, growth, maximum-size, reclaim, cross-stream, deleted-stream, and multi-thread tests implemented; multi-device and full fragmentation matrix pending |
| `arena_memory_resource` | none | missing | arena fragmentation and thread tests |
| `fixed_size_memory_resource` | `FixedSizeMemoryResource` | implemented | block-size, preallocation, growth, stream-reuse tests |
| `binning_memory_resource` | `BinningMemoryResource` | implemented | bin selection, explicit bin, and upstream fallback tests |
| `limiting_resource_adaptor` | `LimitingMemoryResource` | implemented | exact-limit, free/reuse, upstream-failure tests |
| `failure_callback_resource_adaptor` | `FailureCallbackMemoryResource` | implemented | retry, stop, and forwarding tests |
| `callback_memory_resource` | `CallbackMemoryResource` | implemented | captured-state, allocation/deallocation, and serialized callback tests |
| `statistics_resource_adaptor` | `StatisticsMemoryResource` | implemented | nested scope, peak, total, and counter-stack tests; negative and multithread cases pending |
| `tracking_resource_adaptor` | `TrackingMemoryResource` | implemented | live/total byte and allocation accounting tests; failure and multithread matrix pending |
| `logging_resource_adaptor` | `LoggingMemoryResource` | implemented | CSV header/action test; failure and sink-error matrix pending |
| `thread_safe_resource_adaptor` | `ThreadSafeMemoryResource` | implemented | concurrent callback serialization test |
| `prefetch_resource_adaptor` | `PrefetchMemoryResource` | implemented | managed-memory location and non-managed no-op tests |
| aligned resource | `AlignedMemoryResource` | implemented | over-alignment, threshold, size, and multithread tests |
| per-device resource registry | intentionally omitted | explicit allocator arguments avoid process-global mutable policy |
| pinned host resource | `PinnedHostMemoryResource` | implemented | host/device visibility, pool, zero-byte, and alignment tests |
| `system_memory_resource` | `SystemMemoryResource` | implemented | capability rejection, alignment, host/device visibility, and stream-ordered free tests |
| `sam_headroom_memory_resource` | `SamHeadroomMemoryResource` | implemented | all-host headroom and host/device visibility tests; split-range pressure test pending |
| `polymorphic_allocator` / Thrust allocator | Rust collection/CCCL interop traits | design pending | algorithm integration without copies |

`PoolMemoryResource::release_unused` returns only wholly free upstream chunks,
preserving recorded stream dependencies and leaving partially live chunks
reserved. Reclamation also caps future geometric growth by the remaining pool
size (with a 1-MiB floor, subject to the configured maximum), rather than keeping
the historical high-water growth target across allocate/release cycles.

## Container parity

| RMM container contract | Rust status |
| --- | --- |
| `device_buffer` byte ownership, copy/move, reserve, resize, shrink | typed `DeviceBuffer<'stream, T, A>` borrows its stream, stores its allocator handle, and has deep copy and capacity operations; byte-erased facade missing |
| allocator lifetime and dispatch | `A`, `&'pool A`, `Rc<A>`, and `Arc<A>` handles use static dispatch; compile-fail coverage prevents a borrowed pool from being destroyed first |
| `device_uvector<T>` uninitialized typed vector | `DeviceUninitBuffer<'stream, T, A>` uses `MaybeUninit<T>` and requires unsafe `assume_init`; broader parity tests pending |
| `device_scalar<T>` set/get and stream ordering | implemented as `DeviceScalar<'stream, T, A>`; broader parity tests pending |
| destruction after active-device change | context is retained; automated multi-device coverage missing |
| explicit stream transitions | intentionally replaced by compile-time fixed stream affinity; cross-stream use is rejected instead of mutating buffer state |
| RCCL/CCCL zero-copy input traits | `ContiguousDeviceRange` and `ContiguousDeviceRangeMut` implemented; broader CCCL traits pending |

## Test gates

1. `cargo test -p rccl-memory` must pass everywhere. GPU tests report a clear
   skip only when CUDA device 0 cannot be opened.
2. GPU contract tests mirror upstream test groups, including same-stream,
   cross-stream, deleted-stream, and multi-thread cases. The multi-device case
   is present but skips on the one-GPU reference machine.
3. Miri/loom-style host tests cover adaptor state machines where CUDA calls are
   not required.
4. Sanitizer and `compute-sanitizer` jobs cover use-after-free, races, and
   stream-order mistakes.
5. No row reaches performance parity until the comparison benchmark meets the
   thresholds below on the same GPU, CUDA driver, warm-up, seed, and allocation
   trace.

On the RTX 6000 Ada reference machine, CUDA memcheck currently reports zero
errors across the 28 runnable GPU contracts. The GPU-1 context-rebinding probe
is filtered from memcheck because the machine has one device; the ordinary test
suite still runs it and records the hardware skip.

## Performance gates

The comparison harness must run both pinned RMM and `rccl-memory` for RMM's
`random_allocations`, `multi_stream_allocations`, `replay`, `device_uvector`,
and `synchronization` benchmark families.

- warmed median operations/second: Rust must be at least 95% of RMM;
- p99 allocation latency: Rust must be no more than 110% of RMM;
- peak reserved bytes and fragmentation: Rust must be no more than 105% of RMM;
- no additional device-wide synchronization in an equivalent trace;
- results must include confidence intervals and the raw trace/JSON artifact.

The single-stream random-allocation gate passes on the RTX 6000 Ada reference
machine. Five paired runs measured median Rust throughput at 147.6% of RMM and
median Rust p99 at 60.3% of RMM. See
[`benchmarks/results/2026-08-23-rtx6000-ada.md`](benchmarks/results/2026-08-23-rtx6000-ada.md).
Multi-stream, replay, container, and synchronization performance gates remain
open.

These thresholds are release gates, not current claims. Until all relevant rows
and benchmark gates are green, the crate is RMM-inspired rather than
feature/performance-identical.
