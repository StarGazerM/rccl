# Ordering closure, raw pointer, and `Arc` adversarial review

- Baseline: CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`
- Public surface: Thrust sort/merge functions and CUB `WarpMergeSort`
- Focused host test: `tests/sort_api.rs::ordering_arguments_are_ordinary_copy_closures`
- GPU tests: `cccl-sort-tests`, `cccl-comparison-tests`, `cccl-merge-tests`, and `cccl-warp-merge-sort-tests`
- Full gate: `scripts/test-gpu.sh`

## Ordering callable result

The public and device-level APIs accept an ordinary
`Copy + 'static + Fn(&K, &K) -> Ordering`. RCCL has no `DeviceComparator`,
hidden callable trait, `Less`/`Greater` marker, or compatibility alias. Default
natural-ascending `u32` calls are the explicit radix specialization point;
arbitrary closures use the comparison path because their semantics are opaque.

The matrix covers function items, noncapturing descending closures, custom
record functions, and move-capturing mask closures. The captured-mask GPU
oracles do not receive the closure. They receive the mask as an independent
scalar kernel argument and calculate masked order directly, preventing one
closure-lowering defect from being repeated by both implementation and test.

The pinned cuda-oxide collector and its generated typed module launcher derive
different PTX type-id suffixes for the comparison kernel when its generic
callable is a closure defined in another crate. RCCL does not expose a
compatibility callable to hide that backend defect. The private comparison
dispatcher asks cuda-oxide's generated instantiation helper for the collector's
actual symbol, loads that exact symbol, and marshals the same by-value kernel
ABI. Lookup and launch failures retain the concrete symbol name in the error.

## Mutable raw-pointer result

The baseline commit contains 174 `*mut` spellings. The retained scan has
46. Concrete examples, benchmarks, and ordinary typed test kernels now use
cuda-oxide `&[T]` and `DisjointSlice<T>` launch arguments. `WarpMergeSort` no
longer uses a nullable `*mut V` to encode a keys-only call: all pair register
storage is an ordinary mutable array borrow, with a local `[(); N]` for the
keys-only specialization.

The comparison, radix, and merge workspaces own typed buffers and lower them
through affine, length-bearing `DeviceRangeMut<T>` launch views. Merge keeps
separate typed key and value spill allocations rather than recovering their
types from one byte buffer. Fixture kernels likewise receive private by-value
length-bearing ranges; only the immediate host constructor sees the device
address.

The remaining sites were inspected by role:

| Location | Count | Why retained |
|---|---:|---|
| `rccl-memory/src/lib.rs` | 5 | CUDA driver/host ABI pointer and out-parameter conversions |
| `rccl-memory/tests/gpu_contract.rs` | 1 | deliberate managed-memory host dereference contract test |
| `rccl-cub/src/warp_merge_sort.rs` | 5 | type-aligned views over caller-owned dynamic shared storage |
| `rccl-cub/src/device.rs` | 3 | the private pointer/length fields and constructor of the typed device-range ABI |
| `rccl-cub/src/comparison.rs` | 6 | typed range construction plus cuda-oxide's raw argument vector for the cross-crate closure-symbol workaround |
| `rccl-cub/src/merge.rs` | 8 | zero-copy generic source/sink adapters and typed workspace/range construction |
| `rccl-cub/src/radix.rs` | 11 | typed workspace/range construction and two dynamic-shared atomic-array views |
| `examples/gpu_support.rs` | 3 | private writable fixture-range field, accessor, and host construction from any RCCL `DeviceSliceMut` implementor |
| `src/bin/cccl-merge-tests.rs` | 4 | deliberate `u32` word reinterpretation of 512/1024-byte record fixtures and their independent validators |

The private pointer/count lowering in `gpu_support` and the CUB dispatchers is
not considered an ideal endpoint. cuda-oxide already has generic
`KernelSliceArg` traits for its asynchronous launcher; equivalent generic
synchronous lowering would allow RCCL spans and allocator-backed buffers to
retain typed slice arguments. Until that exists, replacing the private pointer
with a slice would silently drop the public span/resource-buffer coverage.

## `Arc` result

The baseline has 157 `Arc<...>` type spellings; the retained Rust source has
75. Each remaining ownership role is listed below rather than treating `Arc`
as an ordinary convenience wrapper.

No `Arc<CudaStream>` remains. `Execution`, workspaces, and allocator-backed
buffers borrow `&CudaStream` with an explicit lifetime. The gate rejects any
reintroduction of an Arc-wrapped stream.

The audit also removed Arc where a Rust lifetime or single owner expresses the
relationship:

- `CudaPoolViewMemoryResource<'pool>` borrows its source pool, matching RMM's
  non-owning view while statically preventing premature pool destruction;
- pool and fixed-size free blocks retain only a predecessor stream id; each
  pool owns one `CudaEvent` per stream instead of cloning one Arc into every
  free block;
- tests use borrowed allocator composition and scoped threads, not Arc merely
  to satisfy a `'static` thread-spawn boundary;
- a non-threaded log capture uses `Rc<RefCell<_>>`, and a non-threaded retry
  counter is borrowed directly.

Retained `Arc` uses have an ownership reason:

- cuda-core constructs contexts as `Arc<CudaContext>` and independent streams,
  modules, allocations, and memory resources must keep that context alive;
- CUB's weak context registry and typed module cache share one loaded CUDA
  module per live context;
- the per-device allocator registry intentionally returns a replaceable shared
  trait object, and its test verifies Arc identity because identity is part of
  that registry contract;
- the generic allocator implementation for `Arc<A>` remains an explicit caller
  opt-in, parallel to the implementations for `&A`, `Box<A>`, and `Rc<A>`.

An `&Arc<CudaContext>` remains only where cuda-core's embedded-module loader or
the allocator trait must expose the cloneable context ownership handle. It is
not propagated as the algorithm execution handle.

## Masking checks

- A captured closure is not validated by replaying that same closure.
- GPU fixtures are nonempty and nonuniform; natural order cannot satisfy the
  masked-order oracle accidentally.
- Stable key/value validation checks original iota provenance inside every
  equal masked-key class, not sortedness alone.
- Host API tests execute a real test and check closure results; a zero-test
  filter is invalid evidence.
- The gate rejects removed callable names and Arc-wrapped streams before
  runtime tests.
- Device compilation is required; host compilation alone cannot prove closure,
  `Ordering`, slice, or captured-state lowering.
- Cold and warm runtime rounds must both pass so a stale module cache or dirty
  device buffer cannot mask a defect.

## Seeded fault

The review temporarily changed the captured masked ordering in
`cccl-comparison-tests` to natural `i64::cmp`, rebuilt device code, and ran the
binary. The independent scalar-mask oracle failed at the intended assertion
with `left: 500324`, `right: 0`, and process exit 101. The real captured closure
was restored and the comparison, sort, merge, warp, resource-buffer, and span
runtime matrices passed afterward. The final `scripts/test-gpu.sh` cold/warm
gate also passed, including the 239-header ledger and pinned CUB/Thrust
references. A future passing result under the same fault must block completion
and trigger another oracle repair iteration.
