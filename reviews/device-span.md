# Device span red-flag decision

- Ledger row: `device-span`
- Baseline: CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`
- Decision: use native Rust/cuda-oxide slices; do not publish an RCCL span type

## Why the earlier type was removed

`DeviceSpan<T>` and `DeviceSpanMut<T>` copied the C++ `span` vocabulary into
the host API, but they were not Rust slices: the CPU could not dereference
them. Device code already receives ordinary cuda-oxide `&[T]`, `&mut [T]`, or
`DisjointSlice<T>` values. Keeping another public span hierarchy therefore
created two meanings of “slice” without adding a kernel capability.

The retained `ContiguousDeviceRange<T>` and `ContiguousDeviceRangeMut<T>` traits are deliberately
smaller. They are unsafe host-side allocation capabilities that let one
algorithm accept both `cuda_core::DeviceBuffer<T>` and allocator-aware
`rccl_memory::DeviceBuffer<T, A>`. They do not manufacture references, provide
subspans, or compete with device-side slices.

## Subranges

RCCL currently accepts complete buffer ranges. A future host subrange API must
be justified by a real algorithm consumer. Its preferred forms are:

1. an operation-specific checked `Range<usize>` parameter for synchronous
   algorithms; or
2. native generic slice-argument lowering in cuda-oxide when a reusable view
   is required.

RCCL should not reintroduce a `span`, iterator-pair, pointer-wrapper, or proxy
reference compatibility layer merely to match a CCCL name.

## Safety boundary

Private CUB launch ABI values still lower an exclusively borrowed allocation
to pointer plus length inside an unsafe enqueue boundary. They are not public
views and cannot outlive the safe synchronous call. Kernel bodies use typed
range operations or cuda-oxide slices; raw addresses remain confined to CUDA
ABI, atomics, and shared-memory boundaries.
