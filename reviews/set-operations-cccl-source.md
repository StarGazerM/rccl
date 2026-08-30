# Set operations: pinned CCCL source contract

Pinned source: CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`.

## Public behavior

The supported family is `set_union`, `set_intersection`, `set_difference`, and
`set_symmetric_difference`, each with the natural ordering or a user ordering.
The key/value forms retain values from the selected key source.  Equality is
left-biased: union and intersection take the key and value from the first
input.  Intersection has no second value input in the public CCCL API.

Rust does not reproduce iterator-pair overloads or returned end iterators.
The execution environment owns the symmetric operation, inputs are borrowed
device buffers (or structure-of-arrays key/value ranges), output is an
exclusive borrow, the comparator is an ordinary `Fn(&K, &K) -> Ordering`, and
the result is the initialized output-prefix length.  The `_by_key` distinction
is represented by the `KeyValueRange` input/output shape rather than by a
compatibility suffix.

CCCL accepts an output iterator sized to the actual result; it does not
require worst-case capacity. Rust therefore performs no invented
maximum-result precheck. A caller may use an exactly sized `DeviceBuffer`, and
cuda-oxide's checked writable region traps an actual out-of-bounds device
write. `Discard<T>` removes the final global scatter and returns the output
count, matching `discard_iterator`.

The pinned `key_value<U, U>` helper compares ordering by `key` but equality by
both fields. Implementing that as Rust `Ord` would violate the language
contract. RCCL instead uses the named `KeyValue` product with an explicit
key-ordering closure; the generated kernel and CCCL comparator-equivalence
behavior remain identical.

## CUDA lowering

The CUDA backend in `thrust/system/cuda/detail/set_operations.h` uses:

1. a 128-thread `InitAgent` for padded scan-tile descriptors;
2. a 256-thread `PartitionAgent` using `balanced_path`;
3. one `SetOpAgent` block per tile.

`balanced_path` is the pinned merge-path plus biased lower-bound levels
`9, 7, 5, 4`, a final binary search, duplicate-run stair-step balancing, and
the star adjustment.  Each set block loads the concatenated input tile in a
blocked arrangement, computes one balanced local partition per thread, runs
the operation-specific serial set routine, scans each thread's active mask
with `BLOCK_SCAN_WARP_SCANS`, scatters the compacted keys and values through
shared memory, and has the final tile publish the output count.

Tiles after tile zero use `ScanTileState` and `TilePrefixCallbackOp`'s classic
32-lane decoupled lookback.  Primitive 32-bit indices use one 8-byte
status/value transaction word; 64-bit indices use one 16-byte transaction
word.  Thirty-two padded descriptors precede tile zero.  Relaxed publication,
the pinned 450-cycle initial delay, and the 350-cycle retry policy are part of
the implementation contract.

## Compile-time policy

The proc-macro-owned policy is exactly:

- SM 5.2: 256 threads, nominal 15 four-byte items;
- SM 6.0 and newer: 512 threads, nominal 19 four-byte items;
- `items = min(nominal, max(1, ceil(nominal * 4 / sizeof(K))))`;
- `items_per_tile = threads * items - 1`.

Value size deliberately does not affect item tuning.  It does affect the
shared union: a `threads * sizeof(i32)` offset array followed by the larger of
the key and value arrays, each with `items_per_tile + threads` elements.  The
complete union has 16-byte alignment.  These values, operation
monomorphization, kernel entry selection, and launch bounds are compile-time
facts.  Input lengths and the 32/64-bit index branch are runtime facts.

The Rust program setup can emit a set-only trait implementation catalog. This
is capability collection, not another schedule table: the same CUB proc macro
still owns every architecture/type decision and every kernel body, and the
public set operations remain ordinary Rust methods. The pinned example adds
the independent `merge` capability because it demonstrates merge before the
four set operations; the focused set tests and benchmark do not compile it.
Rust comparator captures must be owned (`move`) and `'static`; this is the
language-level replacement for copying a C++ comparison object into a launch.

## Temporary storage and virtual shared memory

Thrust makes one temporary byte allocation.  Its nested 256-byte aliases are:

- output count;
- scan tile state (itself a nested padded alias);
- `(num_tiles + 1)` balanced partitions;
- virtual shared memory.

The Rust owner must preserve both offsets and the otherwise easy-to-miss
nested tail padding.  If agent shared storage exceeds the device's portable
per-block shared-memory attribute, CCCL allocates one agent-storage region per
tile and launches the virtual-shared-memory kernel path.  This is a required
policy path, not an unsupported-type escape hatch.

## Rust safety boundary

Generated kernels and the algorithm module contain no raw pointers or
`unsafe`.  cuda-oxide address recovery required for dynamic or virtual block
storage and data-dependent scatter remains in `rccl-memory::device` behind
checked capabilities.  C++'s serial routines speculatively read padded,
uninitialized key/value registers.  Rust may not form such values; inactive
lanes use an already-valid key/value as their ignored padding value, preserving
the operation and schedule without introducing an invalid Rust value.

## Corresponding evidence and open gate

`rccl-thrust/examples/set_operations.rs` ports the pinned example's merge,
four set operations, and discard-output intersection-size case. The focused
GPU test ports ascending/descending, duplicate/multiset, equivalent-range,
discard, by-key source selection, all eight `key_value` types, and counting
iterator magnitudes 30 through 33. Exact-result buffers deliberately use the
known result length in the simple cases so an accidental worst-case-capacity
restriction fails the suite.

The proc-macro test pins SM52 and SM89 record item counts and shared-byte
layouts, including the rule that value size changes the shared union but not
items per thread. Memcheck/racecheck skip only the magnitude-30-through-33
launches after running a smaller instance of the identical arithmetic agent;
the native wide-index result tests remain mandatory.

The pinned SM52 policy is therefore preserved and tested, but the current
cuda-oxide revision cannot emit any Rust CUDA module below SM70: its own target
validator assigns `DetectedFeatures::Basic` an SM70 floor. A direct SM52 build
reaches that validator and is rejected before PTX generation; the same
set/merge capability bundle builds on SM70 and SM89. Consequently SM52 device
execution and performance are not claimed until cuda-oxide lowers its basic
target floor. This is an upstream backend limitation, not a substituted RCCL
schedule.

The review also deliberately changed the SM60+ nominal item count from 19 to
18. The focused policy test failed with `(512, 9)` instead of `(512, 10)` and
passed again only after restoring 19, demonstrating that the test detects a
wrong TMP policy rather than accepting the current implementation.

The strict benchmark rejects any foreign compute process before measuring and
uses a 0.1% slowdown ceiling. Its default is the large pinned `i32/i64`,
entropy-1.000, ratio-50, `2^24` row. `RCCL_SET_FULL_MATRIX=1` expands this to
the upstream signed 8/16/32/64-bit key types, 8/64-bit values, both entropy
rows, all three input ratios, and sizes `2^16`, `2^20`, `2^24`, and `2^28`.
Correctness, memcheck, and racecheck are passing, but performance is not
recorded as passing while PID
1128380 (`VLLM::EngineCore`) occupies the GPU; this external exclusivity
condition is intentionally not masked by relaxed thresholds or selected
samples.

The compiler-resource comparison is also intentionally recorded as a gap,
not converted into a parity claim.  For the isolated SM89 `i32` key-only
union, the pinned CCCL 32-bit-index agent uses 64 registers with no stack or
spill traffic; the corresponding cuda-oxide agent uses 83 registers with no
stack or spill traffic.  Their partition agents use 18 and 16 registers,
respectively.  Both use one barrier and the pinned 512-thread/19-item policy,
so this is a code-generation resource mismatch after matching the algorithm
and TMP path. Exact performance remains open until this difference and the
exclusive-GPU throughput gate are resolved.
