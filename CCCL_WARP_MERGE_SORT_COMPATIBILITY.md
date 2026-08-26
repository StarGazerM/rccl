# CUB WarpMergeSort compatibility

Audit baseline: NVIDIA/cccl commit
`571f2fc3bc53cd710e306ad43d58995c1fe4219f` (2026-08-22).

`rccl::cub::WarpMergeSort` is a native Rust/cuda-oxide port of
`cub::WarpMergeSort`; it does not call or link CUDA C++. The implementation was
audited against:

- `cub/cub/warp/warp_merge_sort.cuh`;
- its inherited implementation in `cub/cub/block/block_merge_sort.cuh`;
- `cub/test/catch2_test_warp_merge_sort.cu`;
- the default-template/header smoke use in `cub/test/catch2_test_nvrtc.cu`;
- the complete example embedded in the public warp header.

## Public API and traits

The Rust specialization preserves CUB's generic order and defaults:

```rust
rccl::cub::WarpMergeSort<
    Key,
    ITEMS_PER_THREAD,
    LOGICAL_WARP_THREADS = 32,
    Value = (),
>
```

Rust cannot overload methods by argument list, so pair and partial forms have
explicit names:

| CUB surface | RCCL surface |
| --- | --- |
| `TempStorage` | `WarpMergeSortTempStorage<K, ITEMS, LOGICAL, V>` created by `warp_merge_sort_temp_storage!` |
| constructor from `TempStorage&` | `WarpMergeSort::new(storage)` |
| `get_linear_tid()` | `get_linear_tid()` |
| `get_member_mask()` | `get_member_mask()` |
| `Sort(keys, comp)` | `sort(keys, comp)` |
| `StableSort(keys, comp)` | `stable_sort(keys, comp)` |
| `Sort(keys, comp, valid, oob)` | `sort_partial(keys, comp, valid, oob)` |
| `StableSort(keys, comp, valid, oob)` | `stable_sort_partial(keys, comp, valid, oob)` |
| `Sort(keys, comp, valid)` | `sort_partial_without_sentinel(keys, comp, valid)` |
| `StableSort(keys, comp, valid)` | `stable_sort_partial_without_sentinel(keys, comp, valid)` |
| the six key/value overloads | the corresponding `*_pairs*` methods |

Ordering is an ordinary `Copy + 'static` `Fn(&K, &K) -> Ordering`. The closure
can be zero-sized or carry scalar state and is monomorphized into the calling
kernel; RCCL defines no callable compatibility trait. `WarpMergeSortConfig`
exposes only specialization facts:
key/value types, items per thread, logical warp size, tile size, and exact
temporary-storage size/alignment. This separation avoids making users
implement cuda-oxide policy traits and avoids conflating a runtime ordering
value with a compile-time tile policy.

The storage token encodes key type, value type, items per thread, and logical
warp size. It is deliberately not `Copy`; storage for one specialization
cannot be passed to another specialization by type accident.

## Required temporary-storage macro

CUB requires caller-owned shared `TempStorage` per logical warp. Rust generic
statics cannot capture a function's type or const parameters, so RCCL requires
a concrete allocation at the final CUDA kernel call site:

```rust
let storage = unsafe {
    rccl::cub::warp_merge_sort_temp_storage!(
        STORAGE,
        key = i32,
        items_per_thread = 4,
        logical_warp_threads = 16,
        warps = 16,
        warp_index = warp_id,
    )
};
let sorter = rccl::cub::WarpMergeSort::<i32, 4, 16>::new(storage);
```

The macro emits a statically sized and aligned `cuda_device::SharedArray` and
returns one typed warp view. It rejects zero items, zero warps, and logical
warp sizes that are not powers of two in `1..=32` during compilation. The
specialization constructor repeats the geometry check, so bypassing the macro
does not turn an invalid geometry into a runtime branch.

The CUDA kernel crate must list the same `cuda-device` revision as a direct
dependency. This is intentional: cuda-oxide recognizes shared-memory and
thread intrinsics through their canonical `::cuda_device` paths. Routing them
through an RCCL re-export produces ordinary unresolved device symbols at this
collective's user-authored kernel call site. This requirement is specific to
device-callable collectives; ordinary Thrust host algorithms do not require an
application to name `cuda-device` directly.

The storage size matches CUB's union exactly: the maximum of
`Key[tile + 1]` and `Value[tile + 1]`, aligned and padded to form an array of
per-warp objects. The extra key is required because serial merge prefetches one
position past a run.

## Algorithm and runtime behavior

The port follows the inherited CUB implementation:

1. stable odd-even sort of each lane's blocked register array;
2. logical-subwarp mask and warp barriers;
3. one shared-memory exchange per merge round;
4. Merge Path partitioning and stable serial merge;
5. aliased value exchange through the same union storage for pair sorts;
6. clamped merge runs for partial tiles, including the sentinel-free CUB 3.6
   overloads and the same unspecified suffix contract.

All tile-local counts and partitions use 32-bit arithmetic as in CUB; storage
byte sizing remains `usize` on the host. `Sort` is currently stable in both
implementations, but only `StableSort` publishes that guarantee.

## Upstream test and example coverage

`src/bin/cccl-warp-merge-sort-tests.rs` ports the complete parameter product
from `catch2_test_warp_merge_sort.cu`:

- keys: `u8`, `i32`, `i64`, and a lexicographic custom record;
- values: `i32` and the custom record;
- logical warp threads: 32 and 4;
- items per thread: 1, 4, and 7;
- two logical warps per block;
- keys and pairs, full and partial tiles, `Sort` and `StableSort`.

That is 288 pinned upstream cases. The matrix adds 144 cases for the newer
sentinel-free partial overloads. Duplicate keys and distinct values make pair
stability observable. A separate 32-lane smoke kernel uses
`WarpMergeSort::<i32, 4>` with the default logical-warp argument, matching the
NVRTC instantiation.

`examples/warp-merge-sort.rs` ports the public header's 16-thread by 4-item
example and runs it across a 256-thread block with explicitly allocated
per-warp storage. Both executables are registered in `scripts/test-gpu.sh`.

## Measured performance

RTX 6000 Ada (`sm_89`), CUDA 12.9.41, 268,435,456 `u32` items, 100 measured
iterations after five warmups. Both kernels used 256 threads, eight 32-lane
warps per block, four items per thread, the same descending 128-item tiles,
and the input-reset kernel outside the timed interval. Outputs were validated
on the GPU. The C++ reference was compiled from the pinned checkout with
`-O3 -arch=sm_89`.

| implementation | best | median | best throughput |
| --- | ---: | ---: | ---: |
| RCCL `WarpMergeSort<u32, 4>` | 2.647 ms | 2.668 ms | 101.410 Gitem/s |
| CCCL CUB `WarpMergeSort<u32, 4>` | 2.644 ms | 2.666 ms | 101.530 Gitem/s |

The recorded best-throughput difference is 0.12%; the median-time difference
is 0.08%. This is measured performance parity on this specialization and GPU,
not a claim of identical SASS or results for unmeasured types/architectures.
The SM89 assembler reports 4,128 shared bytes for both kernels and 26 registers
for RCCL versus 22 for CUB.

```bash
source scripts/oxide-env.sh
cargo oxide run --arch sm_89 --features gpu-examples \
  --bin warp-merge-sort-benchmark
target/release/warp-merge-sort-benchmark \
  --elements 268435456 --iterations 100

CCCL_ROOT=/path/to/cccl-at-571f2fc3bc53cd710e306ad43d58995c1fe4219f
nvcc -std=c++20 -O3 -arch=sm_89 \
  -I"$CCCL_ROOT/cub" -I"$CCCL_ROOT/thrust" \
  -I"$CCCL_ROOT/libcudacxx/include" \
  benchmarks/warp-merge-sort-reference.cu \
  -o target/warp-merge-sort-reference
target/warp-merge-sort-reference 268435456 100
```

Only SM89 has runtime evidence. The primitive is device-callable rather than
NVRTC-callable because cuda-oxide compiles Rust MIR to PTX; the NVRTC test's
relevant public contract—default template arguments, storage formation, and
kernel execution—is covered by the Rust smoke kernel.
