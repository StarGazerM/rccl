# cuCollections hash-map coverage

RCCL pins NVIDIA cuCollections commit
`1d538b6befc286ab450d9e27d7ffbc4f1415c641` as the source and performance
oracle. The default branch is not a reproducible contract. The corresponding
source checkout is `target/cuco-reference-<commit>` and the differential entry
point is `scripts/test-cuco-static-map.sh`.

## Crate and ownership boundary

`rccl-cuco` is a peer of `rccl-cub` and `rccl-thrust`. It owns container
semantics and depends directly on `rccl-memory`; it does not route a container
through a Thrust execution policy. Library-owned `cuda_module` kernels are
materialized inside `rccl-cuco`, while the root `rccl::cuco` module is only a
re-export.

The owning `StaticHashMap<'stream, K, V, A, P, CG>` stores a
`rccl_memory::DeviceBuffer<'stream, u64, A>`. Construction and operations borrow
the generated program and stream. Mutations require `&mut StaticHashMap`, reads
require `&StaticHashMap`, and all ordinary terminals synchronize before input
or output borrows are released. `PendingInsert` is the enqueue-only exception:
it retains the map, input, stream, and program borrows and synchronizes on
`wait` or `Drop`.

Raw pointers, kernel ABI reinterpretation, and checked atomic suballocation
views remain in `rccl-cuco`'s private `kernel_args` boundary over
`rccl-memory` contiguous ranges. Runtime container code uses safe prepared
launch contracts with an exact `(128, 1, 1)` block shape.

## Production slice

The current packed implementation intentionally supports only the CUDA path
which the pinned toolchain can publish atomically:

| Contract | Implemented |
|---|---|
| key words | `u32`, `i32` |
| mapped words | `u32`, `i32`, bit-preserving `f32` |
| slot | one aligned `u64`, key low and value high |
| publication | one relaxed device-scope 64-bit CAS |
| default probing | `LinearProbing`, xxHash32 seed 0, `CG = 4`, bucket 1 |
| alternate probing | `DoubleHashing`, xxHash32 seeds 0 and 1 |
| capacity | exact linear stride rounding and double-hash prime-group rounding |
| bulk block | 128 threads |
| storage | stream-affine allocator-owned device buffer |

The key/value traits are sealed. A wider type cannot accidentally enter the
packed path under a generic `DeviceCopy` bound. SM90 `atom.cas.b128`, split-slot
publication, larger keys/values, bucket size 2, and heterogeneous equality are
not represented as implemented.

## Implemented semantic surface

- construction by `StaticMapConfig` builder from requested capacity or
  `(element count, load factor)`;
- explicit empty key/value and optional erased-key sentinels;
- checked Rust-style `insert`, `get`, `contains_key`, and `remove` terminals
  returning `Result`;
- compatibility names for `try_insert`, sentinel-output `find`,
  explicit-presence `find_with_presence`, `contains`, and `erase`;
- insert-or-assign;
- uncounted insertion matching the upstream `insert_async` timing boundary;
- O(capacity) `len`, compact `retrieve_all`, clear;
- same-capacity tombstone purge and resizing rehash;
- copyable device read/mutation references with bounded full-table probing;
- structure-of-arrays device input through borrowed contiguous ranges.

The minimal `DynamicHashMap` slice is implemented as host-side orchestration
over a non-empty vector of verified `StaticHashMap` segments. `insert` requires
an erased-key sentinel, removes existing keys across active segments, grows by
appending a new static segment when the active segment would exceed the
configured maximum load, then inserts/replaces in the newest segment. `get` and
`contains_key` search active segments newest-to-oldest with reused static-map
lookup kernels and merge explicit presence masks on the host. `remove`, `clear`,
`reserve`, and compaction `rehash` are implemented over the same segments.

Checked insertion rejects reserved key or value sentinels with `Result` rather
than inheriting upstream undefined behavior. It reports that non-reserved
entries may already have been inserted. The uncounted benchmark path assumes
sentinels were excluded and does not add validation or a success counter.

Every probe is bounded by one complete table traversal, so insertion into a
full table terminates. Tombstone insertion records the first erased slot,
continues until it proves the key absent at an empty slot or after a full
cycle, then CASes the remembered tombstone and restarts on a race. This is the
corrected behavior for the still-open NVIDIA/cuCollections issue 606; cloning
the pinned behavior would permit duplicate keys after erase and reinsert.

## Remaining upstream surface

The following pinned `static_map` operations remain explicit coverage work:

- `insert_if`, `insert_and_find`, `insert_or_apply`, and user-defined equality;
- conditional and heterogeneous `contains`/`find`;
- `for_each`, key count, partial `retrieve`, and found-slot views;
- scalar and CG 1/2/8 policies, bucket size 2, and additional hash functions;
- 8/16-bit words, 64-bit words, split publication, and gated SM90 b128 CAS;
- shared-output lookup buffering, warp/shared retrieve aggregation, and
  shared-memory rehash gathering;
- full cuCollections `dynamic_map` parity: device-side segment selection,
  concurrent growth publication, cooperative insert-and-grow, and upstream
  policy/equality variations.

The implemented `DynamicHashMap` is a minimal Rust-owned dynamic-map slice, not
an alias: it owns a non-empty segment vector, performs real host-side
growth/rehash orchestration, preserves lookup across active segments, and tests
growth, replacement, remove, clear, reserve, and compaction. It remains partial
against cuCollections `dynamic_map` because growth is synchronized host
orchestration around static kernels rather than device-side concurrent
segmentation.

## Gates

`scripts/test-cuco-static-map.sh` requires the exact cuCollections and CCCL
pins, then runs host unit tests, the cuda-oxide SM89 build, the GPU correctness
matrix, Compute Sanitizer memcheck/racecheck/synccheck, embedded-PTX shape
checks, and an independently compiled C++ reference. The matrix covers exact
hash vectors, zero/linear/prime extent rounding, hits and misses, assignment,
reserved sentinels, retrieval, duplicates, full-table termination, erase,
issue-606 tombstone reuse, rehash, and both probing policies. The PTX gate
requires the 128-thread, 32-bit cached-key probe, one elected-lane relaxed
device-scope b64 CAS, one status shuffle, and no coherent/wide probe, system
atomic, fence, 64-bit probe-loop remainder, spill, or helper call.

`scripts/test-cuco-static-map-performance.sh` refuses to run with any foreign
GPU compute process. It compares equal SoA inputs, capacity, hash seed, stream,
warmups, clear placement, event boundary, and synchronization. For each of
50%, 75%, and 90% occupancy it alternates five C++/Rust rounds, takes ten event
samples per process, compares the median of process medians, and requires
`RCCL / cuCollections <= 1.001`.

On 2026-09-01 the production insertion slice passed every strict cell on an
exclusive Runpod RTX 4090 (SM89, CUDA 12.8, driver 580.159.04) with no foreign
compute process. The median-of-medians `RCCL / cuCollections` ratios were
`0.999428` at 50% occupancy, `0.997711` at 75%, and `0.974912` at 90%; equal
capacities were checked in every cell. This closes performance parity for the
implemented packed `u32 -> u32` static-map insertion slice only. The later
API/type surfaces above remain open and require their own correctness and
performance evidence. A shared-GPU diagnostic remains non-evidence.
