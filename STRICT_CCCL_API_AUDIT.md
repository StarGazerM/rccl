# Strict CCCL/Rust API audit

Baseline: NVIDIA CCCL
`571f2fc3bc53cd710e306ad43d58995c1fe4219f`.

This audit records the observable Rust contract and the exact upstream
operation represented by each name. Hidden visibility, compatibility aliases,
and tests that instantiate no real call are not evidence of a correct design.

Status vocabulary:

- **fixed**: corrected in the retained implementation;
- **accepted**: intentional, source-backed Rust adaptation;
- **replace**: an operation is required but the current representation is not
  sufficient;
- **missing**: pinned CCCL functionality is not implemented;
- **backend**: the correction belongs in cuda-oxide rather than an RCCL
  signature disguise.

## A. Ownership, lifetime, and safety

| ID | Status | Current contract |
|---|---|---|
| A01 | fixed | Safe Thrust sort and merge calls always attempt stream synchronization after a path that may have queued work, including when a later launch fails. Borrowed ranges are not accessed after return. |
| A02 | backend | `Execution` borrows `&Arc<CudaStream>` because that is cuda-core's handle representation. Replacing the `Arc` belongs in cuda-oxide. |
| A03 | fixed | `Execution` is defined only by `rccl-thrust`. It holds the borrowed stream, allocator, and an opaque CUB kernel-context handle; it contains no schedule or algorithm type. |
| A04 | accepted | CUB owns a context-scoped typed module cache because cuda-oxide does not provide one. The global registry stores weak entries, so the final live CUB handle can release the module. |
| A05 | fixed | Sort and merge bind once with `par(&stream)` or `par_in(&stream, allocator)`. There is no algorithm-specific execution wrapper. |
| A06 | fixed | CUB dispatch receives its own context handle, stream, and allocator. It never accepts or imports the Thrust execution policy. |
| A07 | fixed | Radix, comparison, and merge workspaces allocate through the allocator held by Thrust execution. |
| A08 | replace | `DeviceComparator` can capture raw device pointers with only a prose lifetime obligation. A future typed device-reference wrapper should encode that state. |
| A09 | fixed | Public sort and merge consume typed borrowed device ranges; erased pointer/boolean iterator mappings are gone. |
| A10 | missing | A safe no-sync policy needs a completion object that retains every borrowed range. An unchecked safe asynchronous alias would be unsound. |

## B. Crate and dispatch boundary

The dependency order is exact:

```text
rccl-memory -> rccl-cub -> rccl-thrust -> rccl
```

`rccl-memory` is also used directly by Thrust, and the umbrella directly
re-exports both public layers. There is no CUB-to-Thrust edge.

| crate | owns | must not own |
|---|---|---|
| `rccl-memory` | stream-ordered resources, buffers, contiguous device-range ownership | CUB or Thrust dispatch |
| `rccl-cub` | device scalar/comparator vocabulary, cooperative collectives, device-wide kernels and dispatch, target schedule, context module cache | Thrust execution or host facade |
| `rccl-thrust` | `Execution`, `par`, `par_in`, safe synchronous host algorithms, compile-time backend specialization | CUB kernel bodies, a duplicate module cache, public schedule language |
| `rccl` | thin re-exports for Thrust, memory, and CUB | kernels, dispatch engines, another execution type |

Target schedules are private CUB constants generated for one
`CUDA_OXIDE_TARGET`. Host-only Cargo checks use deterministic SM80 constants.
There is no runtime GPU probe, policy object, generated strategy type,
breakpoint table, launch callback, or function-pointer dispatch table.

Rust generics for element, comparator, allocator, and range types are ordinary
parametric polymorphism. They are not a schedule/control-flow language.

## C. Thrust-facing API

| ID | Status | Current contract or gap |
|---|---|---|
| C01 | fixed | Ordinary root functions provide `sort`, `sort_by`, stable variants, key/value variants, and the four merge operations. |
| C02 | fixed | Descending convenience names are absent; callers pass `Greater`. |
| C03 | accepted | Rust cannot overload by arity, so `_with` denotes the key/value comparator form. |
| C04 | fixed | Safe operations synchronize like pinned Thrust `par`; only a future explicit no-sync policy may omit that boundary. |
| C05 | accepted | Rust returns `Result` instead of CUDA status output parameters or C++ exception/termination machinery. |
| C06 | accepted | A contiguous Rust range replaces `(first, last)` pointer pairs while preserving the same logical range. |
| C07 | missing | General Thrust fancy/proxy iterators do not participate in the ordinary APIs. |
| C08 | missing | Device-side Thrust entry points are absent. Pinned Thrust selects its sequential implementation in device code because device-side launches are unsupported. |
| C09 | replace | `sort` currently shares the stable backend. Stability is guaranteed only by the explicitly stable operations so future optimization remains possible. |

## D. Sort fidelity

| ID | Status | Current contract or gap |
|---|---|---|
| D01 | fixed | Exact built-in `Less`/`Greater` over `u32` keys reaches the four-pass stable OneSweep radix path. |
| D02 | fixed | Exact built-in `Less`/`Greater` over `u32` keys and `u32` values reaches the specialized pair path. |
| D03 | fixed | Custom records and comparators use a typed comparison backend and are not mislabeled as radix. |
| D04 | replace | The comparison fallback performs rank/binary-search merge passes and is a correctness path, not the claimed CUB-radix performance path. |
| D05 | missing | Full ascending/descending optimized coverage for every CCCL arithmetic, half, and bfloat combination is absent. |
| D06 | missing | CUB single-tile versus multi-pass selection, large-offset portioning, `DoubleBuffer`, selector, bit-range, and decomposer contracts are absent. |
| D07 | missing | Offset type is not selected from the caller's range difference type; the optimized path currently converts counts to `u32`. |
| D08 | missing | Policy tuning does not yet cover the complete CUB product of key size, value size, offset width, phase, and architecture. |
| D09 | missing | One SM89 runtime matrix is not evidence for SM80, SM90, or SM100 behavior. |
| D10 | replace | The recorded 4 GiB OneSweep measurement used an internal reusable workspace, not an identical public allocation/synchronization boundary. |

## E. Merge fidelity

| ID | Status | Current contract or gap |
|---|---|---|
| E01 | fixed | The retained host functions perform stable key-only and key/value merge with built-in or state-carrying comparators. |
| E02 | fixed | CUB owns the asynchronous device-wide dispatch. The hidden entry points are unsafe and state the stream-completion lifetime requirement; Thrust adds the safe synchronization boundary. |
| E03 | fixed | Normal and wide-record fallback geometry is selected at monomorphization, not through a runtime strategy object. |
| E04 | missing | Merge accepts contiguous device slices only; counting, discard, transform, zip, permutation, and other proxy iterators remain missing. |
| E05 | missing | The public CUB temporary-storage query/execute and environment-allocation forms are not represented. |
| E06 | replace | Fallback selection measures RCCL's staged key/value bytes rather than pinned CUB's full agent `TempStorage` union and can switch too early for some record shapes. |
| E07 | replace | Public performance parity needs fresh CUB/Thrust measurements with identical allocation and synchronization timing. |

## F. CUB device layer

Pinned CUB distinguishes device-wide host dispatch from in-kernel collectives:

| Pinned primitive | Ordering customization | RCCL state |
|---|---|---|
| thread stable odd-even sort | comparator | missing |
| `WarpBitonicSort` | comparator | missing |
| `WarpMergeSort` | comparator | implemented and public |
| `BlockMergeSort` | comparator | missing |
| `BlockRadixSort` | decomposer and bit range | missing |
| `DeviceRadixSort` | device-wide host dispatch | private specialized backend only |

The public `WarpMergeSort` is device-callable, has compile-time geometry, and
uses explicit temporary storage. Missing collectives must follow the same CUB
shape rather than introducing host execution objects into the device layer.

## G. Iterator and range gaps

The deleted transitional iterator designs are not retained under compatibility
names. A replacement needs associated typed device views/proxy references,
must retain every backing range, must propagate stream-ordering dependencies,
and must preserve zero-copy contiguous behavior. Zip/tuple, transform,
counting, discard, reverse, strided, and permutation views remain missing.

`DeviceSlice::cu_deviceptr` is still the universal low-level interop escape
hatch. Safe algorithms should eventually consume a more strongly typed device
view instead of asking implementations for an untyped integer address.

## H. Test and example integrity

| ID | Status | Evidence rule |
|---|---|---|
| H01 | fixed | Integration tests instantiate ordinary root functions with real buffer and allocator types. Renamed-root and direct-Thrust fixtures each contain an actual test. |
| H02 | fixed | Every retained sort example uses the ordinary host facade; the collective example uses the real CUB `WarpMergeSort`. |
| H03 | fixed | The gate derives every binary and example from Cargo metadata and fails if a target has no runtime route. |
| H04 | fixed | Known consumer inventories are non-empty and each declared consumer must contain an ordinary algorithm call. |
| H05 | fixed | The gate rejects obsolete crate directories, dependencies, names, and generated call structures rather than allowing compatibility shims. |
| H06 | fixed | Host tests and the complete GPU runtime/example inventory run twice, exercising cache reuse and teardown paths. |
| H07 | fixed | CUDA C++ reference sources are inventoried and executed against the exact pinned CCCL commit. |
| H08 | missing | The complete randomized upstream Thrust/CUB test product is not ported; selected matrices must not be called all CCCL tests. |

Passing the gate proves only the retained RCCL surface and the recorded pinned
reference cases. It does not prove full upstream product parity.

## I. Accepted Rust adaptations

- `Result` replaces CUDA status output parameters and C++ exceptions.
- Safe functions borrow typed device ranges rather than accepting naked
  pointer/count pairs.
- `DeviceComparator<K>` is implemented by a concrete `Copy` value rather than
  C++ callable-object syntax.
- Target schedule constants are selected during the CUB build for one explicit
  cargo-oxide architecture.
- Raw pointers remain confined to unsafe CUDA kernel and hidden asynchronous
  dispatch boundaries.
- Temporary storage and collective shared storage have explicit Rust ownership.

Any additional difference needs a source-backed justification rather than the
claim that C++ used a pointer, class, overload, or template.
