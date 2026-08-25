# RCCL: pure-Rust CUDA primitives

RCCL implements CUDA primitives in Rust with NVIDIA Research's
[`cuda-oxide`](https://github.com/NVlabs/cuda-oxide) compiler. Kernel bodies are
ordinary Rust compiled through MIR to PTX. RCCL does not call Thrust, CUB, CUDA
C++, NVRTC, or a CPU sorting library at runtime.

The host API consists of ordinary functions:

```rust
use cuda_core::{CudaContext, DeviceBuffer};

let context = CudaContext::new(0)?;
let stream = context.default_stream();
let exec = rccl::cuda::par(&stream)?;
let mut keys = DeviceBuffer::<u32>::zeroed(&stream, count)?;

rccl::sort(&exec, &mut keys)?;
```

`rccl::cuda::par` borrows the stream and uses its context. Each safe host
algorithm synchronizes before returning, including after a failure on a path
that may already have queued work. Temporary storage stays on the GPU; the GPU
examples generate and validate their data on the GPU and copy only scalar
validation results to the CPU.

## Crate boundary

The workspace follows CCCL's CUB-below-Thrust dependency direction:

```text
rccl-memory
     |
     v
  rccl-cub       device-callable collectives, device-wide dispatch,
     |           target schedules, kernels, context module cache
     v
 rccl-thrust     Execution, par/par_in, safe host algorithms
     |
     v
    rccl         thin re-export crate
```

There is no shared algorithm runtime crate and no generated facade crate.
`rccl-cub` never accepts a Thrust `Execution`. Its hidden asynchronous merge
dispatch accepts a CUB kernel context, stream, and allocator directly.
`rccl-thrust` owns the safe borrowed-range boundary, invokes CUB, and performs
the required synchronization.

Applications can depend directly on `rccl-thrust`. Device-level users can
depend directly on `rccl-cub`. The `rccl` umbrella re-exports both surfaces but
contains no kernels or dispatch implementation.

## Thrust host algorithms

The sort surface contains:

- `sort`, `sort_by`, `stable_sort`, `stable_sort_by`
- `sort_by_key`, `sort_by_key_with`
- `stable_sort_by_key`, `stable_sort_by_key_with`

The merge surface contains:

- `merge`, `merge_by`
- `merge_by_key`, `merge_by_key_with`

Rust cannot overload functions by arity, so `_with` denotes the key/value form
with an explicit comparator. Descending order uses the same operation with
`Greater`.

Range parameters implement `rccl::memory::DeviceSlice<T>` or
`DeviceSliceMut<T>`. Both `cuda_core::DeviceBuffer<T>` and allocator-aware
`rccl::memory::DeviceBuffer<T, A>` implement those traits. General Thrust-style
fancy/proxy iterators are not implemented yet.

A custom comparator is a `Copy` Rust value whose concrete type and state are
monomorphized into the GPU kernels:

```rust
#[derive(Clone, Copy)]
struct MaskedOrder {
    mask: i64,
}

unsafe impl rccl::DeviceComparator<i64> for MaskedOrder {
    unsafe fn less(&self, left: &i64, right: &i64) -> bool {
        (left & self.mask) < (right & self.mask)
    }
}

rccl::stable_sort_by(&exec, &mut keys, MaskedOrder { mask: 0xff })?;
```

The unsafe implementation promises that `less` is GPU-callable, defines a
strict weak order, and keeps captured device state valid through stream
completion.

## Compile-target schedules

CUB owns its target schedule. A cargo-oxide build with `--arch sm_XX` supplies
`CUDA_OXIDE_TARGET=sm_XX`; `rccl-cub/build.rs` validates that target and writes
one set of internal constants. Sort and merge pass those constants directly to
const-generic kernels. There is no public schedule language, policy type,
runtime architecture probe, breakpoint table, or function-pointer dispatch.

Ordinary host-only Cargo commands do not set a CUDA target, so CUB uses a
deterministic SM80 specialization for type checking. GPU builds must supply an
explicit architecture.

## Current sort dispatch

Compile-time specialization selects the backend from the concrete key, value,
and comparator types:

| Call shape | Comparator | Current backend |
|---|---|---|
| `u32` keys | exact `Less` or `Greater` | four-pass stable OneSweep radix |
| `u32` keys and `u32` values | exact `Less` or `Greater` | four-pass stable OneSweep radix |
| other supported key/value shapes | built-in or custom comparator | typed stable rank/merge fallback |

The optimized boundary is therefore `u32` and `u32/u32`, not every arithmetic
type supported by Thrust smart sort. The rank/merge fallback is a correctness
path, not a CUB-radix-performance claim. `sort` currently uses the stable
backend, but callers should rely on stability only through `stable_sort`.

Important remaining parity work includes optimized radix coverage for all CCCL
scalar combinations, large-offset portioning, `DoubleBuffer`, bit-range and
custom decomposer contracts, general typed iterators, and the complete pinned
upstream randomized test product. The exact gaps are tracked in
[`STRICT_CCCL_API_AUDIT.md`](STRICT_CCCL_API_AUDIT.md).

## Execution and memory resources

`rccl::cuda::Execution` belongs to Thrust and can be shared by sort and merge.
The default constructor uses `CudaAsyncMemoryResource`; applications can bind a
concrete allocator with `par_in`:

```rust
let pool = rccl::memory::CudaPoolMemoryResource::new(
    context.clone(),
    rccl::memory::PoolConfig::default(),
)?;
let tracked = rccl::memory::TrackingMemoryResource::new(&pool);
let exec = rccl::cuda::par_in(&stream, &tracked)?;
```

Allocator handles are ordinary Rust type parameters, so a borrowed resource
cannot be outlived by a buffer or execution policy. CUB maintains one merged
CUDA module and typed namespace views for each live CUDA context. Its registry
stores weak handles; the module can be released after the last execution drops
its CUB context handle. See [`rccl-memory/README.md`](rccl-memory/README.md) for
the resource matrix.

## CUB device primitives

`rccl::cub::WarpMergeSort` is a real device-callable warp collective with
explicit temporary storage. Its pinned-source and overload mapping is recorded
in
[`CCCL_WARP_MERGE_SORT_COMPATIBILITY.md`](CCCL_WARP_MERGE_SORT_COMPATIBILITY.md).
Device-wide merge parity is recorded in
[`CCCL_MERGE_COMPATIBILITY.md`](CCCL_MERGE_COMPATIBILITY.md).

## Verification

The full gate is:

```bash
scripts/test-gpu.sh
```

It derives the complete binary/example inventory from Cargo metadata, rejects
obsolete crates and generated call paths, verifies the exact dependency graph,
and fails if a declared consumer contains zero ordinary algorithm calls. It
runs formatting, warning-denied Clippy, workspace tests, warning-denied docs,
the oxide doctor/build, every GPU compatibility binary, every registered RCCL
example, bounded benchmark smoke runs, both `rccl-memory` examples, and the
pinned CUDA C++ references.

Host tests and the complete GPU runtime/example set each run for two rounds.
The second round exercises reuse and teardown paths instead of treating one
fresh-process pass as sufficient evidence.

The reference phase verifies pinned CCCL commit
`571f2fc3bc53cd710e306ad43d58995c1fe4219f`. It uses `CCCL_ROOT` when provided;
otherwise it creates a cached checkout below ignored `target/`. A manual GPU
build must also specify its architecture:

```bash
cargo oxide build --arch sm_89 --features gpu-examples
```

Passing the gate proves the retained RCCL surface and recorded pinned-reference
cases. It does not claim that the complete upstream Thrust/CUB product has been
ported.
