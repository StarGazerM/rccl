# CCCL Thrust/CUB coverage plan

Baseline: NVIDIA CCCL commit
`571f2fc3bc53cd710e306ad43d58995c1fe4219f`.

This is the product plan for the complete Thrust/CUB subset, including the
support vocabulary needed to express the algorithms. It is not a plan to
transliterate C++ headers. A row is covered only when the observable CCCL
operation exists, or when this document records an intentional Rust-native
replacement and tests that replacement at the host/device boundary.

## Boundary and counting rule

The pinned public entry-point inventory is:

- 64 top-level Thrust headers: 34 algorithm headers and 30 support headers;
- 19 Thrust iterator headers, 16 memory-resource headers, and 6 public
  type-trait headers;
- 59 top-level CUB primitive headers: 17 block, 21 device, 9 warp, 8 thread,
  and 4 iterator headers;
- the CUB implementation substrate: 26 agent, 3 grid, 22 detail, and 16
  top-level configuration/utility headers.

The counts are a drift detector, not a completion metric. Completion is
tracked per public operation and overload family. Internal agent, dispatch,
tuning, and temporary-storage code is covered transitively by the public
primitive that needs it.

The boundary includes:

1. all public Thrust algorithm families in the 64 top-level headers;
2. Thrust containers, execution policies, allocators, memory resources,
   pointers/references, function objects, random types, complex values, and
   fancy iterators;
3. all public CUB thread, warp, block, device, and iterator primitives;
4. the CUB agent/grid/dispatch/schedule/temp-storage machinery required by
   those primitives;
5. the libcu++ concepts actually required by this surface, expressed using
   Rust `core`, `alloc`, `std`, and cuda-oxide rather than copied as another
   standard library.

The boundary does not include unrelated libcu++ facilities merely because
they share the CCCL repository. For example, a CUB implementation may need
atomics, `span`, and `mdspan`; that does not make the complete libcu++ text,
locale, or host threading library part of RCCL.

## Rust design rules

The runtime dependency direction is:

```text
rccl-memory -> rccl-cub -> rccl-thrust -> rccl
```

All authored compile-time information is above that graph:

```text
rccl-cub-macros    -> generated rccl-cub Rust
rccl-thrust-macros -> generated rccl-thrust Rust
```

`rccl-memory` is also consumed directly by `rccl-thrust`. Support code goes in
the lowest existing owner that needs it:

- `rccl-memory`: owning allocations, allocator handles, host-side borrowed
  device ranges, context and stream-order validation;
- `rccl-cub-macros`: CUB primitives, agents, schedules, specialization, and
  cuda-oxide kernel families;
- `rccl-cub`: handwritten support/runtime values plus macro-emitted operation
  code only after an operation is implemented;
- `rccl-thrust-macros`: every Thrust-to-CUB compile-time lowering rule;
- `rccl-thrust`: handwritten adaptors, execution/completion values, and
  macro-emitted algorithm facades only after implementation;
- `rccl`: re-exports only.

Do not create a monolithic compatibility crate. If real implementation work
would otherwise create a dependency cycle, extract only the pure value/range
vocabulary into a small lower crate. It must not own an execution policy,
module cache, allocator, dispatch engine, or schedule selection.

Other rules:

- use native Rust values and traits where they preserve the contract;
- keep host device pointers non-dereferenceable and lifetime-carrying;
- lower read-only kernel ranges to `&[T]` and exclusive kernel ranges to
  cuda-oxide's checked mutable/disjoint slice forms where possible;
- represent fancy iterators as zero-copy indexed range expressions, not as
  host `Iterator<Item = &T>` values;
- map template type computation to proc-macro-emitted generics, traits,
  associated types, const generics, and `Fn` callables;
- map repeated kernel families and whole-program specialization collection to
  declarative/procedural macros and the cuda-oxide compiler collector, never
  runtime reflection or type erasure;
- keep target schedules exclusively in `rccl-cub-macros`; expansion inserts
  literal operands without emitting a runtime schedule value;
- retain GPU buffers, views, and temporary storage until stream completion;
- never add CPU staging or host scratch to make a GPU algorithm easier.

### Compiler DSL contract

The complete contract is in `RCCL_COMPILER_DSL.md`. The CUB and Thrust DSLs
are shallow user call-site macros parsed into operation-specific `syn` 2 ASTs.
Their closed inventories and lowering tables are typed Rust compiler catalogs,
not separate `.dsl` files and not user-authored syntax.
Expansion must:

1. expose every operation requiring TMP, schedule selection, kernel
   registration, or static shared-storage syntax as a shallow macro such as
   `sort_by!`, while leaving ordinary Rust support operations as normal types
   and functions;
2. pass resolved schedule values as literal const-generic operands and direct
   cuda-oxide compiler attributes;
3. let Rust monomorphize key, value, callable, and range types without a
   `TypeId`, `Any`, erased comparator, strategy object, or proof trait leaking
   across the generated API;
4. emit concrete cuda-oxide Rust kernels and typed launch code directly; and
5. leave no manifest, schedule, specialization map, or compiler capability
   value in a runtime struct.

Declarative macros may cover repeated compiler implementation syntax, but the
two procedural macro crates own call parsing, typed catalog validation,
wrapper generation, and all compile-time facts. Proc-macro expansion owns
RCCL's compile-time semantic lowering and emits pure Rust; cuda-oxide owns all
subsequent compilation decisions. RCCL may not invoke or configure a device
compiler, perform symbol-name lookup, or build an erased ABI argument-vector
path of its own.

The earlier direct-literal rule remains: do not reintroduce
`DispatchEngine`, `SortConstInvocation`, `MergeConstInvocation`, or another
generic-const witness layer merely to make macro-expanded code type-check.
The old runtime loader and handwritten sort/merge kernel layer have been
deleted. Algorithms remain unexported until a safe proc-macro expansion
exists; they do not receive a runtime error, panic, CPU fallback, or fake
implementation.

## Support-library mapping

| CCCL support surface | Rust-native representation | Owner | Current state |
|---|---|---|---|
| `cuda::std::span<T>` | Kernel-side `&[T]`, `&mut [T]`, or `DisjointSlice<T>` | Rust + cuda-oxide | native; the removed host `DeviceSpan` hierarchy was not dereferenceable and duplicated the slice vocabulary; public host subranges remain unimplemented until a real consumer justifies `Range` or native launcher support |
| `cuda::std::array<T, N>` | `[T; N]` | Rust/cuda-oxide | native; add ABI and device-code matrices |
| `thrust::pair`, `thrust::tuple` | `(A, B)` and native tuples for ordinary products; named `#[repr(C)]` records where field names or external ABI matter | Rust + CUB value types | partial; cuda-core implements `DeviceCopy` for tuples through arity 8 and handwritten support code provides checked two-way zip range descriptors, but device compilation and higher-arity composition remain |
| `tuple_size`, `tuple_element`, concepts and type traits | associated types, trait bounds, const generics, and compiler type checking | Rust | native; do not recreate template metafunctions |
| `optional`, `expected`, variant-like results | `Option<T>`, `Result<T, E>`, and Rust enums | Rust | native; test device lowering for every used shape |
| CUDA half and bfloat values | canonical `half::f16` / `half::bf16`, which cuda-core already marks `DeviceCopy`; ordering/radix transforms belong to operations, not replacement scalar types | CUB + cuda-core | native scalar types retained; RCCL storage-only wrappers removed; optimized sort coverage remains |
| CUDA Boolean values | Rust `bool`; generated writers must only construct valid Boolean values | Rust + cuda-oxide | native; the duplicate byte-backed wrapper was removed |
| `thrust::host_vector` | `Vec<T>` | `std` | native compatibility mapping; not part of the default GPU-resident data path |
| `thrust::device_vector` / `vector_base` | `rccl_memory::DeviceBuffer<'stream, T, A>` with Rust `Vec`-like reserve/resize/truncate semantics and a borrowed stream lifetime | memory | partial and usable; iterator/adaptor integration remains |
| universal and pinned-host vectors | `DeviceBuffer<T, ManagedMemoryResource>` and pinned-host resource/buffer types | memory | resources exist; container aliases and parity tests remain |
| `device_ptr`, `universal_ptr`, raw pointer casts | unsafe typed device address plus safe lifetime-bearing range/view; raw address only at FFI/kernel internals | memory | replace current universal `cu_deviceptr` escape hatch |
| `device_reference` and proxy references | no free-standing fake `&T`; range/read/write capabilities and adaptor-specific device lowering | CUB + Thrust | partial; minimal host capability traits cover cuda-core and allocator-aware device buffers but deliberately not host slices, while generated kernels remain |
| iterator traits/categories/facade/adaptor | descriptor structs compose range/read/write/random-access capabilities without host execution methods | CUB + Thrust | partial; constant/counting/discard/subrange/permutation/reverse/stride/tabulate/transform/zip forms and semantic host tests are handwritten, while the cuda-oxide device composition matrix remains |
| `mdspan`, extents, layouts | typed multidimensional device view with shape, strides, and layout marker; contiguous 1-D case lowers to slice | CUB | missing; cuda-oxide checked 2-D views are reusable backend pieces |
| functional objects | ordinary `Fn`/`FnMut` bounds plus `Copy` device-callable closures; specialization is emitted by `rccl-thrust-macros` | CUB + Thrust macros | catalogued; algorithm expansion is backend-blocked; no RCCL callable compatibility trait |
| `complex<T>` | RCCL-owned `#[repr(C)] Complex<T>` or an upstream cuda-core `DeviceCopy` implementation for a standard Rust complex type | CUB value layer | missing |
| random engines/distributions | small copyable RNG state values with Rust methods; exact pinned sequences tested against Thrust | Thrust | missing |
| memory construction/destruction and unique allocation | `MaybeUninit<T>`, RAII owners, `Drop`, and explicit stream-ordered destruction/release | memory + Thrust | partial; unique object/array helpers and nontrivial device construction remain |
| Thrust memory resources and allocators | `DeviceAllocator` values/references plus existing concrete/adaptor resources | memory | strong partial coverage; public mapping and allocator propagation matrix remain |
| CUB `Uninitialized`, `KeyValuePair`, `DoubleBuffer`, `FutureValue`, `NullType` | `MaybeUninit`, named key/value record, typed two-buffer selector, device scalar/future input, and `()`/enum absence markers | CUB | partial; handwritten `MaybeUninit` storage, `KeyValue`, and borrowed `DoubleBuffer` are public and tested, while future/device-scalar and absence forms remain |
| CUB temporary-storage query/execute | checked `Layout` plus a borrow of caller-owned `DeviceBlock` whose recorded size/alignment satisfies it; collective shared storage stays explicit and typed | CUB macros + runtime | safe handwritten layout/storage protocol exists; macro-emitted operation formulas and public request forms are missing |
| CUB chained policies and architecture traits | typed target rows owned only by `rccl-cub-macros`, then emitted as literal kernel operands | CUB macros | catalogued; no runtime or public policy language |
| CUDA atomics, barriers, warp operations, shared memory, vector loads | reuse canonical `cuda-device` types and intrinsics | cuda-oxide backend | available in part; missing backend operations should be fixed upstream, not re-exported through RCCL paths |
| status/system errors | opaque Rust error values and `Result` | each owning crate | established; backend choice is private |

### libcu++ dependency closure

The pinned Thrust/CUB sources directly include these public libcu++ domains:
`array`, assertions and integer limits/types, `concepts`, `expected`,
`functional`, `initializer_list`, `limits`, `mdspan`, `numbers`, `optional`,
`span`, `tuple`, and `type_traits`. They also use internal algorithm, bit,
cmath, execution, iterator, memory, numeric, tuple, type-trait, and utility
facilities, plus CUDA argument, atomic, device, execution, memory-resource,
PTX, stream, and warp support.

| libcu++ dependency class | Rust implementation decision |
|---|---|
| scalar types, sizes, limits, arrays, initialization | Rust primitives, `core::mem`, `core::num`, `[T; N]`, and `MaybeUninit<T>` |
| traits, concepts, forwarding/move, integer sequences | Rust traits, associated types, const generics, ownership, and compiler checks; no runtime library |
| span, iterators, mdspan | native kernel slices plus minimal read/write/random-access range and multidimensional-view contracts |
| tuple, optional, expected | Rust tuple, `Option`, and `Result`, subject to cuda-oxide device-ABI tests |
| functional, algorithms, numeric, bit, math, limits | Rust operators/closures and `core` methods; use canonical cuda-device/libdevice intrinsics when GPU semantics differ |
| atomics, barriers, warp/PTX, access/cache operations | canonical cuda-device APIs; add missing instructions to cuda-oxide rather than cloning intrinsics in RCCL |
| execution, streams, arguments, memory resources | cuda-core/cuda-host launch types plus RCCL `Execution`, `Pending`, and `DeviceAllocator` |
| assertions/exceptions/system errors | device assertions/traps where supported and host `Result`; never unwind through a CUDA kernel boundary |

This is an equivalence closure, not a promise to publish an `rccl::std`
namespace. A new libcu++ facility is added to RCCL only when a public
Thrust/CUB row needs observable behavior that Rust or cuda-oxide does not
already provide.

### Current RCCL baseline

- `rccl-memory` already has allocator-aware `DeviceBuffer`, `DeviceScalar`,
  managed/pinned/device resources, pool resources, and resource adaptors.
- `rccl-cub-macros` validates the complete 59-primitive public CUB catalog;
  `rccl-thrust-macros` validates all 34 algorithms and 19 iterator/adaptor
  headers. Support families are ordinary-library coverage rows, not compiler
  catalog entries.
- `rccl-cub` currently exposes only handwritten safe support values. The old
  pointer-heavy `WarpMergeSort`, radix, comparison, and merge implementations
  were removed rather than hidden behind a runtime loader.
- `rccl-thrust` currently exposes only handwritten `cuda::Execution`, borrowing
  a stream and allocator. Sort and merge are backend-blocked and unexported.
- Descriptor-only fancy-range algebra and the checked device temporary-storage
  value protocol exist; device access lowering and every public CUB
  thread/warp/block/device algorithm remain absent.

### Slice and device-range contract

A host `&[T]` is host-addressable and cannot stand for device memory. RCCL does
not publish a second span type. Generated launch code must lower a borrowed
device allocation directly to cuda-oxide slices or checked disjoint views; it
may not recreate pointer/length structs. No generated algorithm currently
claims this boundary, so the relevant algorithms remain unexported. Public
subranges should use an operation-specific checked `Range<usize>` or native
cuda-oxide launcher support.

### Tuple, zip, and proxy contract

Ordinary values should use Rust tuples. Zip is different: a zipped output is
not a Rust tuple of `&mut` references that can be freely copied. It is an
indexed product of range descriptors. The public host-side contract is
deliberately non-operational:

```rust,ignore
trait DeviceRange {
    fn len(&self) -> usize;
}

trait ReadRange: DeviceRange {
    type Item: DeviceCopy;
}

trait WriteRange: DeviceRange {
    type Item: DeviceCopy;
}
```

`RandomAccessRange` and `ContiguousRange` refine it. They contain no host
`get`/`write` method: the macro compiler lowers each descriptor to concrete
device operations only after rustc/cuda-oxide has inferred its type. `Zip`,
`Transform`, `Permutation`, `Discard`, and `TransformOutput` remain zero-copy.

For product arities above cuda-core's current tuple `DeviceCopy` coverage,
first extend cuda-oxide's tuple implementations. A local `#[repr(C)]` product
record is the fallback when a stable named layout is required; nested tuples
must not silently become a public ABI convention.

### Vector contract

Do not add a second `device_vector` owner. `DeviceBuffer<'stream, T, A>` is the Rust
container. Complete it as an allocator-aware, stream-ordered vector:

- `len`, `capacity`, `reserve`, `resize`, `truncate`, and moves;
- explicit uninitialized construction using `MaybeUninit` semantics;
- allocator propagation and context/stream transition rules;
- operation-specific checked ranges and range adaptor entry points when a real
  consumer requires them;
- managed and pinned variants through allocator choice, not subclasses.

`cuda-device::vector` is a different concept: aligned scalar packs for wide
memory transactions. Reuse those types for CUB load/store implementations; do
not confuse them with an owning vector container.

## Complete Thrust inventory

### Algorithm headers

| Family | Pinned headers and operation families | Current state | Main dependency |
|---|---|---|---|
| Elementwise traversal | `for_each`, `transform`, `tabulate`, `generate`, `sequence` | missing | range-expression algebra + DeviceFor/Transform |
| Copy and initialization | `copy`, `fill`, `uninitialized_copy`, `uninitialized_fill` | missing | contiguous fast path + general read/write range path |
| Gather/scatter | `gather`, `scatter` including conditional forms | missing | permutation/index range + bounds contract |
| Reordering | `reverse`, `shuffle`, `swap`/`swap_ranges` | missing | indexed ranges + random engine |
| Search and predicates | `find`, `count`, `logical`, `equal`, `mismatch` | missing | transform/reduce/select foundations |
| Ordered queries | `binary_search`, `extrema` | missing | ordering callables + reduce/search agents |
| Reductions | `reduce`, `inner_product`, `transform_reduce` including reduce-by-key | missing | thread/warp/block/device reduce |
| Scans/differences | `scan`, `transform_scan`, `adjacent_difference` including scan-by-key | missing | thread/warp/block/device scan |
| Compaction | `remove`, `unique`, `partition` including copy, stable, by-key, and point/query forms | missing | scan + select/partition agents |
| Replacement | `replace` and copy/predicate variants | missing | elementwise traversal |
| Merge and sets | `merge`, `set_operations` including key/value forms | merge backend-blocked; sets missing | general iterators + generated device merge |
| Sorting | `sort` including stable/key/value/query forms | backend-blocked | generated radix/merge sort, offsets, decomposers, iterators |

This table covers exactly these 34 headers:

```text
adjacent_difference.h  binary_search.h    copy.h
count.h                equal.h            extrema.h
fill.h                 find.h             for_each.h
gather.h               generate.h         inner_product.h
logical.h              merge.h            mismatch.h
partition.h            reduce.h           remove.h
replace.h              reverse.h          scan.h
scatter.h              sequence.h         set_operations.h
shuffle.h              sort.h             swap.h
tabulate.h             transform.h        transform_reduce.h
transform_scan.h       uninitialized_copy.h
uninitialized_fill.h   unique.h
```

### Support headers

| Family | Exact pinned entry points | Rust treatment |
|---|---|---|
| Values | `complex.h`, `pair.h`, `tuple.h`, `functional.h`, `zip_function.h`, `random.h` and `random/*` | native products/callables plus RCCL value types only where layout/state matters |
| Iterator utilities | `addressof.h`, `advance.h`, `distance.h` | pointer methods and indexed range operations |
| Allocation/lifetime | `allocate_unique.h`, `device_delete.h`, `device_free.h`, `device_make_unique.h`, `device_malloc.h`, `device_new.h`, `memory.h` | RAII allocation and `MaybeUninit`; no C++ new/delete facade |
| Allocators | `device_allocator.h`, `device_malloc_allocator.h`, `device_new_allocator.h`, `universal_allocator.h`, `per_device_resource.h` | `DeviceAllocator` and concrete resource values |
| Pointers/references | `device_ptr.h`, `device_reference.h`, `universal_ptr.h` | typed device address internally, lifetime views publicly, indexed proxy capabilities |
| Containers | `device_vector.h`, `host_vector.h`, `universal_vector.h` | `DeviceBuffer`, `Vec`, and allocator-selected managed/pinned buffers |
| Execution/errors | `execution_policy.h`, `system_error.h`, `version.h` | `Execution`, completion values, `Result`, crate version |

The 19 iterator headers are all in scope:

```text
constant_iterator.h                 counting_iterator.h
discard_iterator.h                  iterator_adaptor.h
iterator_categories.h               iterator_facade.h
iterator_traits.h                   iterator_traversal_tags.h
offset_iterator.h                   permutation_iterator.h
retag.h                             reverse_iterator.h
shuffle_iterator.h                  strided_iterator.h
tabulate_output_iterator.h          transform_input_output_iterator.h
transform_iterator.h                transform_output_iterator.h
zip_iterator.h
```

`iterator_adaptor`, `iterator_facade`, categories, traversal tags, traits, and
`retag` are not copied as public C++-shaped machinery. Their capabilities must
still be represented by Rust traits so every remaining adaptor has a place.

All 16 `thrust/mr` headers are in scope through their behavior:

```text
allocator.h               device_memory_resource.h
disjoint_pool.h           disjoint_sync_pool.h
disjoint_tls_pool.h       fancy_pointer_resource.h
host_memory_resource.h    memory_resource.h
new.h                     polymorphic_adaptor.h
pool.h                    pool_options.h
sync_pool.h               tls_pool.h
universal_memory_resource.h
validator.h
```

The six public Thrust type-trait headers map to Rust traits/compiler facts:
`integer_sequence`, contiguous-iterator recognition, execution-policy
recognition, trivial relocation, logical metafunctions, and contiguous
unwrapping. Only runtime- or ABI-relevant distinctions get RCCL traits.

```text
integer_sequence.h              is_contiguous_iterator.h
is_execution_policy.h           is_trivially_relocatable.h
logical_metafunctions.h         unwrap_contiguous_iterator.h
```

The public random subheaders are also individually in scope:

```text
discard_block_engine.h            linear_congruential_engine.h
linear_feedback_shift_engine.h    normal_distribution.h
subtract_with_carry_engine.h      uniform_int_distribution.h
uniform_real_distribution.h       xor_combine_engine.h
```

## Complete CUB inventory

### Public primitives

| Layer | Exact pinned entry points | Current state |
|---|---|---|
| Thread | `thread_load`, `thread_store`, `thread_reduce`, `thread_scan`, `thread_search`, `thread_sort`, `thread_simd`, `thread_operators` | missing as public CUB layer |
| Warp | `warp_load`, `warp_store`, `warp_exchange`, `warp_reduce`, `warp_reduce_batched`, `warp_scan`, `warp_bitonic_sort`, `warp_merge_sort`, `warp_utils` | catalogued; `warp_reduce` has a safe power-of-two scalar/fixed-array slice with pinned GPU evidence; non-power-of-two, record, segmented, and batched forms remain missing/backend-blocked |
| Block data movement | `block_load`, `block_load_to_shared`, `block_store`, `block_exchange`, `block_shuffle` | missing |
| Block scan/reduce | `block_reduce`, `block_scan`, `block_adjacent_difference`, `block_discontinuity` | missing |
| Block ordering | `block_merge_sort`, `block_radix_rank`, `block_radix_sort`, `radix_rank_sort_operations` | missing |
| Block specialized | `block_histogram`, `block_run_length_decode`, `block_topk`, `block_raking_layout` | missing |
| Device traversal/movement | `device_for`, `device_transform`, `device_copy`, `device_memcpy` | missing |
| Device queries/reduction | `device_find`, `device_reduce`, `device_scan`, `device_segmented_reduce`, `device_segmented_scan`, `device_adjacent_difference` | missing |
| Device compaction | `device_select`, `device_partition`, `device_run_length_encode` | missing |
| Device histogram | `device_histogram` | missing |
| Device ordered | `device_merge`, `device_merge_sort`, `device_radix_sort`, `device_segmented_radix_sort`, `device_segmented_sort` | catalogued; merge/radix families backend-blocked, segmented families missing |
| Device top-k | `device_topk`, `device_batched_topk` | missing |
| CUB iterators | `arg_index_input_iterator`, `cache_modified_input_iterator`, `cache_modified_output_iterator`, `tex_obj_input_iterator` | missing; cache/texture semantics need cuda-oxide backend checks |

### Internal substrate that must not be skipped

| Layer | Pinned source inventory | Rust implementation rule |
|---|---|---|
| Agents (26) | adjacent difference, batch memcpy, batched top-k and cluster top-k, find and sorted-bound find, for, histogram, merge, merge sort, radix histogram/up/down/OneSweep, reduce/reduce-by-key, RLE, scan/scan-by-key, segmented radix sort, select-if, sub-warp merge sort, three-way partition, top-k, unique-by-key, single-pass scan operators | private kernel-level structs/functions shared by device APIs; one implementation may serve several public operations |
| Grid (3) | `grid_even_share`, `grid_mapping`, `grid_queue` | typed partition/grid mapping and decoupled-lookback state; use 64-bit global offsets when required |
| Detail (22) | array utilities, binary search helpers, compute-capability dispatch, offset selection, deferred parameters, delays, runtime detection, device double buffer, device memory resource, environment dispatch, fast division, mdspan helpers, prefetch, RFA, segmented params/helpers, strong load/store, temporary storage, type traits, uninitialized copy, bitcast | implement behavior at its Rust owner; do not expose a mirror `detail` namespace merely to match filenames |
| Root utilities (16) | config, umbrella/version, allocator, architecture, dialect, debug, device, macro, math, namespace, policy wrapper, PTX, temporary storage, type, virtual shared memory | reuse Rust/cuda-oxide facilities; retain only semantic types such as temp layout, double buffer, offset selection, and schedules |

The device-wide APIs must expose both of CUB's relevant allocation forms:

1. query required temporary-storage layout, then execute with caller storage;
2. an environment/resource convenience path that owns allocation policy.

Thrust calls the resource form internally. Direct CUB users must not be forced
through a Thrust `Execution`.

## Execution and asynchronous safety

The existing synchronous `cuda::par` contract stays safe. A safe asynchronous
form must return a completion value whose lifetime is tied to all borrowed
inputs, outputs, callable state, and temporary storage:

```text
submit(&'a input, &'a mut output) -> Pending<'a>
Pending::wait(self)              -> completed result
Drop for Pending                 -> waits or otherwise preserves safety
unsafe Pending::detach(self)     -> explicit lifetime responsibility
```

This is the Rust replacement for casually exposing `par_nosync`. Merely
launching and returning `Ok(())` while borrowed ranges may be dropped or
mutated is not an acceptable compatibility alias.

Device-side Thrust calls are a separate surface. Pinned Thrust selects a
sequential implementation in device code because nested launches are not the
contract. Implement those only after the range algebra can execute without
host dispatch.

## Dependency-ordered delivery plan

| Phase | Deliverable | Unlocks | Exit evidence |
|---|---|---|---|
| 0 | Machine-readable coverage ledger, exact pinned example/API-test inventory, typed private catalogs, shallow call parsers, and generated-algorithm safety audit | honest progress, drift checks, and one compile-time owner for every later primitive | every inventory item has an owner/status; all 19 programs, 378 embedded regions, 49 API tests, and 105 exercised CUB methods are pinned; runtime crates contain no closed compiler metadata; generated code rejects `Arc`, raw pointers, and unsafe; no RCCL-owned compiler or loader route exists |
| 1 | Read/write/random-access range traits, callable bounds, product/key-value types, double buffer, offset selection, temp-storage layout; rely on native cuda-oxide slices | all general algorithms and CUB public forms | compile-fail alias/lifetime tests, host/device ABI tests, and zero-copy ownership checks |
| 2 | Thrust adaptors: counting, constant, discard, offset, reverse, strided, transform, transform-output, transform-in/out, zip, permutation, shuffle, tabulate-output, retag/capability conversion | fancy-iterator parity for later algorithms | adaptor composition matrix on device, including read/write zip and non-contiguous outputs |
| 3 | CUB thread then warp then block load/store/exchange/reduce/scan/search/sort/radix/histogram building blocks | reusable implementation base for device-wide algorithms | upstream example/test matrices, exact temp-storage sizes/alignment, and GPU synchronization stress tests |
| 4 | CUB agents, grid partitioning/lookback, public temp-storage query/resource forms, 32/64-bit offset selection | complete device-wide API contracts | query/execute parity, insufficient-storage errors, large-count boundary tests |
| 5 | DeviceFor/Transform/Copy/Memcpy/Find and Thrust traversal, copy, generation, gather/scatter, replace, reverse | broad linear algorithm surface | contiguous and fancy-range differential matrices |
| 6 | DeviceReduce/Scan/Select/Partition/Histogram/RLE/adjacent difference and corresponding Thrust reductions, scans, logical/query, compaction | analytics and fundamental data-parallel substrate | operation/type/operator/offset/segment matrices plus SM performance gates |
| 7 | Complete radix/merge/segmented sorts, merge sort, set operations, binary search, top-k; repair existing sort/merge gaps | all ordered algorithms | scalar/decomposer/bit-range/double-buffer/iterator/stability matrices and equal-boundary benchmarks |
| 8 | Complete container/unique-allocation API, managed/pinned aliases, random, complex, device-side sequential Thrust entry points | remaining support and user ergonomics | construction/destruction, allocator propagation, exact RNG sequence, complex math, device-call tests |
| 9 | Related examples, unit/negative/property tests, then adversarial review and repair iteration | trustworthy evidence rather than tests that merely agree with the current implementation | every public row has a runnable public-API example and focused tests; seeded implementation/launch faults make the relevant tests fail; the masking-error checklist below is closed with evidence; cold and warm full gates pass after repairs |
| 10 | Product hardening across supported SM targets | release-quality full subset | pinned upstream randomized product, compile matrix for SM80/89/90/100, runtime where hardware exists, sanitizer and performance reports |

Implement phases in dependency order, but keep each vertical slice usable. For
example, Phase 3 should land `WarpReduce` with its public matrix and examples,
not wait for every block primitive. Phase numbers prevent building more
high-level one-off kernels that bypass missing foundations.

### Phase 9: adversarial test review and iteration

Examples and tests are deliverables, not evidence merely because they return
success. For every coverage-ledger row, Phase 9 performs this loop:

1. add at least one runnable example through the ordinary public API and
   focused unit, negative/compile-fail, boundary, randomized, and property or
   metamorphic tests as applicable;
2. review how each test could pass while the operation is wrong;
3. seed a temporary fault in the implementation or launch route and prove the
   relevant test fails for the intended reason;
4. add or strengthen tests and fix implementation errors exposed by the
   review;
5. remove the temporary fault, run the focused suite, then run the complete
   host/GPU/pinned-reference gate once cold and once with warm caches;
6. repeat review after repairs until every plausible masking route is either
   tested or recorded as a specific backend limitation.

The review must explicitly check for:

- an example that compiles or launches but never asserts its result;
- zero-length, skipped-launch, disabled-assertion, ignored-error, or early
  return paths that make the test vacuously pass;
- output buffers that retain a zero/default/sentinel value which accidentally
  equals the expected value;
- checking only sortedness, a prefix, a checksum, or a sample while missing
  permutation, stability, key/value association, tail, or exact output errors;
- expected results computed by the implementation under test or by shared
  helper logic with the same bug; the pinned CCCL result must be an independent
  oracle where parity is claimed;
- a CPU fallback, host staging, synchronization side effect, or internal-only
  path masking failure of the intended GPU/public path;
- stale binaries, cached modules, or an unregistered Cargo target causing the
  gate to execute an older artifact;
- compile-only matrix entries being reported as runtime coverage;
- one type, size, alignment, offset width, architecture, or iterator shape
  standing in for a wider product without an explicit limitation;
- grep/guard checks matching comments, documentation, or negative fixtures
  instead of the live implementation;
- benchmark-only or test-only entry points bypassing allocation,
  synchronization, dispatch, or lifetime behavior of the public API;
- repeated tests sharing dirty device state, leaked allocations, or cache state
  that hides initialization, teardown, or reuse defects.

The review result is itself recorded in the coverage ledger: example path,
focused test path, independent oracle, temporary fault used, observed failing
assertion, masking risks checked, repairs made, and cold/warm gate results. A
simple `PASS` field is insufficient.

## Coverage-ledger dimensions

Each public operation row must record applicable dimensions instead of a
single yes/no:

- host Thrust, direct CUB host dispatch, or device-callable collective;
- keys-only, key/value, input/output, in-place, double-buffer, segmented;
- default and custom callable/decomposer, state-carrying callable;
- contiguous and every supported adaptor composition;
- empty/single/tail/exact-tile/multi-tile and very large ranges;
- 32-bit and 64-bit offsets where pinned CCCL selects them;
- stable/unstable, ascending/descending, bit range, NaN/signed-zero behavior;
- caller temp storage and resource-owned storage;
- synchronous and lifetime-safe asynchronous execution;
- target architecture and selected private schedule.

Statuses are `missing`, `partial`, `implemented`, `native-rust`, and
`backend-blocked`. `implemented` requires all applicable dimensions or an
explicit accepted-difference entry.

## Definition of done for one row

A row becomes `implemented` only when all of the following hold:

1. the pinned declaration, CUDA detail route, CUB dispatch/agent route, tuning,
   examples, and tests were read;
2. the Rust API documents its mapping and any accepted difference;
3. safe host APIs encode ownership, aliasing, context, stream, and completion;
4. no host scratch or host data path is introduced;
5. host tests, compile-fail tests, cuda-oxide builds, and runtime tests pass;
6. randomized differential tests run against the exact pinned CCCL checkout;
7. generated Rust audits and GPU runtime tests cover launch bounds,
   synchronization, ownership, and specialization selection where relevant;
8. performance compares identical allocation, synchronization, type,
   distribution, size, and stream boundaries;
9. the aggregate `scripts/test-gpu.sh` gate includes the new consumer, example,
   and reference route;
10. Phase 9 adversarial review demonstrates that temporary implementation or
    launch faults are detected, closes the masking-error checklist, and records
    the final cold/warm gate evidence.

Passing the current gate proves only the existing RCCL surface. Full subset
coverage is reached only when the Phase 0 ledger has no `missing`, `partial`,
or unexplained `backend-blocked` public rows.
