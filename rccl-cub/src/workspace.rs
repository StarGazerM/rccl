use cuda_core::{DeviceBuffer, DeviceCopy};
use rccl_memory::TemporaryValue;

#[doc(hidden)]
pub struct RadixSortWorkspaceCounts {
    pub scratch_keys: usize,
    pub bins: usize,
    pub lookback: usize,
    pub counters: usize,
}

#[doc(hidden)]
pub struct RadixSortPairsWorkspaceCounts {
    pub scratch_keys: usize,
    pub scratch_values: usize,
    pub bins: usize,
    pub lookback: usize,
    pub counters: usize,
}

/// Runtime extents selected by the compile-time set-operation policy.
#[doc(hidden)]
pub struct SetOperationWorkspaceCounts {
    pub num_tiles: usize,
    pub tile_descriptors: usize,
    pub partitions: usize,
    pub virtual_shared_bytes: usize,
    pub wide_indices: bool,
}
/// Normalize an owned direct buffer and a borrowed aliased temporary region to
/// the same mutable range surface.
#[doc(hidden)]
pub trait WorkspaceBuffer<T: DeviceCopy> {
    type Range;

    fn range_mut(&mut self) -> &mut Self::Range;
}

impl<T: DeviceCopy> WorkspaceBuffer<T> for DeviceBuffer<T> {
    type Range = DeviceBuffer<T>;

    fn range_mut(&mut self) -> &mut Self::Range {
        self
    }
}

impl<'stream, T: TemporaryValue> WorkspaceBuffer<T>
    for &mut rccl_memory::TemporaryRegion<'stream, T>
{
    type Range = rccl_memory::TemporaryRegion<'stream, T>;

    fn range_mut(&mut self) -> &mut Self::Range {
        self
    }
}

/// Opaque temporary storage for one direct DeviceMerge operation.
pub struct DeviceMergeWorkspace<P = DeviceBuffer<usize>> {
    partitions: P,
}

impl<P: WorkspaceBuffer<usize>> DeviceMergeWorkspace<P> {
    #[doc(hidden)]
    pub fn __from_partitions(partitions: P) -> Self {
        Self { partitions }
    }

    #[doc(hidden)]
    pub fn __partitions_mut(&mut self) -> &mut P::Range {
        self.partitions.range_mut()
    }
}

/// Opaque regions inside Thrust's one nested set-operation allocation.
#[doc(hidden)]
pub struct DeviceSetOperationWorkspace<O, D, P, V> {
    output_count: O,
    tile_descriptors: D,
    partitions: P,
    virtual_shared: V,
}

impl<O, D, P, V> DeviceSetOperationWorkspace<O, D, P, V> {
    #[doc(hidden)]
    pub fn __from_parts(
        output_count: O,
        tile_descriptors: D,
        partitions: P,
        virtual_shared: V,
    ) -> Self {
        Self {
            output_count,
            tile_descriptors,
            partitions,
            virtual_shared,
        }
    }
}

impl<O, D, P, V> DeviceSetOperationWorkspace<O, D, P, V>
where
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<u64>,
    P: WorkspaceBuffer<u64>,
    V: WorkspaceBuffer<u8>,
{
    #[doc(hidden)]
    pub fn __narrow_parts_mut(
        &mut self,
    ) -> (&mut O::Range, &mut D::Range, &mut P::Range, &mut V::Range) {
        (
            self.output_count.range_mut(),
            self.tile_descriptors.range_mut(),
            self.partitions.range_mut(),
            self.virtual_shared.range_mut(),
        )
    }
}

impl<O, D, P, V> DeviceSetOperationWorkspace<O, D, P, V>
where
    O: WorkspaceBuffer<usize>,
    D: WorkspaceBuffer<u128>,
    P: WorkspaceBuffer<u128>,
    V: WorkspaceBuffer<u8>,
{
    #[doc(hidden)]
    pub fn __wide_parts_mut(
        &mut self,
    ) -> (&mut O::Range, &mut D::Range, &mut P::Range, &mut V::Range) {
        (
            self.output_count.range_mut(),
            self.tile_descriptors.range_mut(),
            self.partitions.range_mut(),
            self.virtual_shared.range_mut(),
        )
    }
}

/// Opaque temporary storage for direct key-only DeviceMergeSort operations.
pub struct DeviceMergeSortKeysWorkspace<K: DeviceCopy, S = DeviceBuffer<K>, P = DeviceBuffer<usize>>
{
    scratch_keys: S,
    partitions: P,
    key: core::marker::PhantomData<fn() -> K>,
}

impl<K: DeviceCopy, S: WorkspaceBuffer<K>, P: WorkspaceBuffer<usize>>
    DeviceMergeSortKeysWorkspace<K, S, P>
{
    #[doc(hidden)]
    pub fn __from_parts(scratch_keys: S, partitions: P) -> Self {
        Self {
            scratch_keys,
            partitions,
            key: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut S::Range, &mut P::Range) {
        (self.scratch_keys.range_mut(), self.partitions.range_mut())
    }
}

/// Opaque temporary storage for direct key/value DeviceMergeSort operations.
pub struct DeviceMergeSortPairsWorkspace<
    K: DeviceCopy,
    V: DeviceCopy,
    S = DeviceBuffer<K>,
    U = DeviceBuffer<V>,
    P = DeviceBuffer<usize>,
> {
    scratch_keys: S,
    scratch_values: U,
    partitions: P,
    types: core::marker::PhantomData<fn() -> (K, V)>,
}

impl<
    K: DeviceCopy,
    V: DeviceCopy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    P: WorkspaceBuffer<usize>,
> DeviceMergeSortPairsWorkspace<K, V, S, U, P>
{
    #[doc(hidden)]
    pub fn __from_parts(scratch_keys: S, scratch_values: U, partitions: P) -> Self {
        Self {
            scratch_keys,
            scratch_values,
            partitions,
            types: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut S::Range, &mut U::Range, &mut P::Range) {
        (
            self.scratch_keys.range_mut(),
            self.scratch_values.range_mut(),
            self.partitions.range_mut(),
        )
    }
}

/// Opaque CUB-owned state used with caller-owned radix double buffers.
pub struct DeviceRadixSortBufferWorkspace<
    B = DeviceBuffer<u64>,
    L = DeviceBuffer<i32>,
    C = DeviceBuffer<i32>,
> {
    bins: B,
    lookback: L,
    counters: C,
}

impl<B: WorkspaceBuffer<u64>, L: WorkspaceBuffer<i32>, C: WorkspaceBuffer<i32>>
    DeviceRadixSortBufferWorkspace<B, L, C>
{
    #[doc(hidden)]
    pub fn __from_parts(bins: B, lookback: L, counters: C) -> Self {
        Self {
            bins,
            lookback,
            counters,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut B::Range, &mut L::Range, &mut C::Range) {
        (
            self.bins.range_mut(),
            self.lookback.range_mut(),
            self.counters.range_mut(),
        )
    }
}

/// Opaque temporary storage for direct key-only DeviceRadixSort operations.
pub struct DeviceRadixSortKeysWorkspace<
    K: DeviceCopy,
    S = DeviceBuffer<K>,
    B = DeviceBuffer<u64>,
    L = DeviceBuffer<i32>,
    C = DeviceBuffer<i32>,
> {
    scratch_keys: S,
    controls: DeviceRadixSortBufferWorkspace<B, L, C>,
    key: core::marker::PhantomData<fn() -> K>,
}

impl<
    K: DeviceCopy,
    S: WorkspaceBuffer<K>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
> DeviceRadixSortKeysWorkspace<K, S, B, L, C>
{
    #[doc(hidden)]
    pub fn __from_parts(scratch_keys: S, bins: B, lookback: L, counters: C) -> Self {
        Self {
            scratch_keys,
            controls: DeviceRadixSortBufferWorkspace::__from_parts(bins, lookback, counters),
            key: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut S::Range, &mut DeviceRadixSortBufferWorkspace<B, L, C>) {
        (self.scratch_keys.range_mut(), &mut self.controls)
    }
}

/// Opaque temporary storage for direct key/value DeviceRadixSort operations.
pub struct DeviceRadixSortPairsWorkspace<
    K: DeviceCopy,
    V: DeviceCopy,
    S = DeviceBuffer<K>,
    U = DeviceBuffer<V>,
    B = DeviceBuffer<u64>,
    L = DeviceBuffer<i32>,
    C = DeviceBuffer<i32>,
> {
    scratch_keys: S,
    scratch_values: U,
    controls: DeviceRadixSortBufferWorkspace<B, L, C>,
    types: core::marker::PhantomData<fn() -> (K, V)>,
}

impl<
    K: DeviceCopy,
    V: DeviceCopy,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
> DeviceRadixSortPairsWorkspace<K, V, S, U, B, L, C>
{
    #[doc(hidden)]
    pub fn __from_parts(
        scratch_keys: S,
        scratch_values: U,
        bins: B,
        lookback: L,
        counters: C,
    ) -> Self {
        Self {
            scratch_keys,
            scratch_values,
            controls: DeviceRadixSortBufferWorkspace::__from_parts(bins, lookback, counters),
            types: core::marker::PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn __parts_mut(
        &mut self,
    ) -> (
        &mut S::Range,
        &mut U::Range,
        &mut DeviceRadixSortBufferWorkspace<B, L, C>,
    ) {
        (
            self.scratch_keys.range_mut(),
            self.scratch_values.range_mut(),
            &mut self.controls,
        )
    }
}
