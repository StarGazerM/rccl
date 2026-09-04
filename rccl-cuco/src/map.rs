use core::marker::PhantomData;

use cuda_core::{
    CudaStream,
    memory::{memcpy_dtoh_async, memset_d8_async},
};
use cuda_device::{
    atomic::DeviceAtomicU64,
    cooperative_groups::{ThreadGroup, WarpCollective, this_thread_block},
};
use rccl_memory::{
    BorrowedCudaAsyncMemoryResource, ContiguousDeviceRange, ContiguousDeviceRangeMut,
    CudaCompletion, DeviceAllocator, DeviceBuffer,
};

use crate::{
    Entries, Launch, LinearProbing, MapError, MapKey, MapValue, ProbingScheme,
    backend::{InsertCall, StaticMapBackend},
    probing::valid_capacity,
};

const METADATA_WORDS: usize = 2;

/// Bitwise sentinels which define empty and erased table slots.
///
/// `empty_key` is always reserved. Calling [`Self::with_erased_key`] also
/// reserves a distinct tombstone key and enables deletion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sentinels<K, V> {
    empty_key: K,
    empty_value: V,
    erased_key: Option<K>,
}

impl<K: MapKey, V: MapValue> Sentinels<K, V> {
    pub const fn new(empty_key: K, empty_value: V) -> Self {
        Self {
            empty_key,
            empty_value,
            erased_key: None,
        }
    }

    pub fn with_erased_key(mut self, erased_key: K) -> Result<Self, MapError> {
        if erased_key == self.empty_key {
            return Err(MapError::EmptyKeyEqualsErasedKey);
        }
        self.erased_key = Some(erased_key);
        Ok(self)
    }

    pub const fn empty_key(self) -> K {
        self.empty_key
    }

    pub const fn empty_value(self) -> V {
        self.empty_value
    }

    pub const fn erased_key(self) -> Option<K> {
        self.erased_key
    }

    #[inline(always)]
    fn device_erased_key(self) -> K {
        self.erased_key.unwrap_or(self.empty_key)
    }
}

/// Fixed-capacity GPU hash map with stream-affine storage.
///
/// Four-byte values use the original packed slot: one `u64` CAS word holding
/// key and value. `u64` values use a two-word slot with a release-published
/// key/control word and a full-width value word. Mutation requires an affine
/// `&mut self`; lookup only borrows the map.
pub struct StaticMap<'stream, K, V, A, P, const CG: u32 = 4>
where
    K: MapKey,
    V: MapValue,
    A: DeviceAllocator,
    P: ProbingScheme<K>,
{
    storage: DeviceBuffer<'stream, u64, A>,
    capacity: usize,
    sentinels: Sentinels<K, V>,
    probing: P,
}

/// Rust-style name for [`StaticMap`].
pub type StaticHashMap<'stream, K, V, A, P, const CG: u32 = 4> = StaticMap<'stream, K, V, A, P, CG>;

#[derive(Clone, Copy, Debug, PartialEq)]
enum CapacitySpec {
    Slots(usize),
    InsertCount {
        elements: usize,
        desired_load_factor: f64,
    },
}

/// Builder/configuration object for an owned stream-bound [`StaticHashMap`].
///
/// The configuration is ordinary Rust state: it names the capacity rule,
/// sentinels, probing policy, and cooperative-group size, then `build` allocates
/// and initializes the device table on the supplied CUDA stream launch context.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticMapConfig<K, V, P = LinearProbing, const CG: u32 = 4>
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    capacity: CapacitySpec,
    sentinels: Sentinels<K, V>,
    probing: P,
}

impl<K, V> StaticMapConfig<K, V, LinearProbing, 4>
where
    K: MapKey,
    V: MapValue,
{
    /// Request a fixed number of slots before cuCollections-compatible
    /// cooperative-group extent rounding.
    pub fn with_capacity(requested_capacity: usize, sentinels: Sentinels<K, V>) -> Self {
        Self {
            capacity: CapacitySpec::Slots(requested_capacity),
            sentinels,
            probing: LinearProbing::default(),
        }
    }

    /// Request enough slots for `elements` at `desired_load_factor`.
    pub fn for_insert_count(
        elements: usize,
        desired_load_factor: f64,
        sentinels: Sentinels<K, V>,
    ) -> Self {
        Self {
            capacity: CapacitySpec::InsertCount {
                elements,
                desired_load_factor,
            },
            sentinels,
            probing: LinearProbing::default(),
        }
    }
}

impl<K, V, P, const CG: u32> StaticMapConfig<K, V, P, CG>
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    pub fn with_probing<Q>(self, probing: Q) -> StaticMapConfig<K, V, Q, CG>
    where
        Q: ProbingScheme<K>,
    {
        StaticMapConfig {
            capacity: self.capacity,
            sentinels: self.sentinels,
            probing,
        }
    }

    pub fn with_cooperative_group<const NEXT_CG: u32>(self) -> StaticMapConfig<K, V, P, NEXT_CG> {
        StaticMapConfig {
            capacity: self.capacity,
            sentinels: self.sentinels,
            probing: self.probing,
        }
    }

    pub const fn sentinels(&self) -> Sentinels<K, V> {
        self.sentinels
    }

    pub const fn probing_scheme(&self) -> P {
        self.probing
    }

    pub fn build<'stream, M>(
        self,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<
        StaticHashMap<'stream, K, V, BorrowedCudaAsyncMemoryResource<'stream>, P, CG>,
        MapError,
    >
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        let allocator = BorrowedCudaAsyncMemoryResource::new(launch.stream().context());
        self.build_in(allocator, launch)
    }

    pub fn build_in<'stream, A, M>(
        self,
        allocator: A,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<StaticHashMap<'stream, K, V, A, P, CG>, MapError>
    where
        A: DeviceAllocator,
        M: StaticMapBackend<K, V, P, CG>,
    {
        match self.capacity {
            CapacitySpec::Slots(requested_capacity) => StaticHashMap::with_capacity_in(
                requested_capacity,
                self.sentinels,
                self.probing,
                allocator,
                launch,
            ),
            CapacitySpec::InsertCount {
                elements,
                desired_load_factor,
            } => StaticHashMap::for_insert_count_in(
                elements,
                desired_load_factor,
                self.sentinels,
                self.probing,
                allocator,
                launch,
            ),
        }
    }
}

/// In-flight CUDA insertion which retains the map and input borrows.
///
/// This is a CUDA completion guard, not a Rust `Future`. [`Self::wait`]
/// surfaces execution errors; dropping it still synchronizes so no borrowed
/// allocation can be released while the kernel is running.
struct PendingInsertResources<'borrow>(PhantomData<&'borrow mut ()>);

pub struct PendingInsert<'borrow>(
    CudaCompletion<'borrow, (), MapError, PendingInsertResources<'borrow>>,
);

impl PendingInsert<'_> {
    pub fn wait(self) -> Result<(), MapError> {
        self.0.wait()
    }
}

impl<'stream, K, V, P, const CG: u32>
    StaticMap<'stream, K, V, BorrowedCudaAsyncMemoryResource<'stream>, P, CG>
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    /// Allocate with the stream's default CUDA asynchronous memory resource.
    pub fn with_capacity<M>(
        requested_capacity: usize,
        sentinels: Sentinels<K, V>,
        probing: P,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<Self, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        let allocator = BorrowedCudaAsyncMemoryResource::new(launch.stream().context());
        Self::with_capacity_in(requested_capacity, sentinels, probing, allocator, launch)
    }

    /// Allocate enough slots for `elements` at the requested occupancy.
    pub fn for_insert_count<M>(
        elements: usize,
        desired_load_factor: f64,
        sentinels: Sentinels<K, V>,
        probing: P,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<Self, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        let allocator = BorrowedCudaAsyncMemoryResource::new(launch.stream().context());
        Self::for_insert_count_in(
            elements,
            desired_load_factor,
            sentinels,
            probing,
            allocator,
            launch,
        )
    }
}

impl<'stream, K, V, A, P, const CG: u32> StaticMap<'stream, K, V, A, P, CG>
where
    K: MapKey,
    V: MapValue,
    A: DeviceAllocator,
    P: ProbingScheme<K>,
{
    pub fn with_capacity_in<M>(
        requested_capacity: usize,
        sentinels: Sentinels<K, V>,
        probing: P,
        allocator: A,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<Self, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        validate_group_size::<CG>()?;
        if sentinels.erased_key == Some(sentinels.empty_key) {
            return Err(MapError::EmptyKeyEqualsErasedKey);
        }
        let capacity = valid_capacity::<K, P>(requested_capacity, CG as usize)
            .ok_or(MapError::CapacityOverflow)?;
        let _ = u32::try_from(capacity).map_err(|_| MapError::CapacityOverflow)?;
        let slot_words = slot_storage_words::<V>(capacity)?;
        let storage_len = slot_words
            .checked_add(METADATA_WORDS)
            .ok_or(MapError::CapacityOverflow)?;
        let mut storage = DeviceBuffer::zeroed_in(allocator, launch.stream(), storage_len)?;
        let empty_slot = empty_slot_word(sentinels.empty_key, sentinels.empty_value);
        let submitted = launch
            .module()
            .clear(launch.stream(), &mut storage, capacity, empty_slot);
        finish_submission(submitted, launch.stream())?;
        Ok(Self {
            storage,
            capacity,
            sentinels,
            probing,
        })
    }

    pub fn for_insert_count_in<M>(
        elements: usize,
        desired_load_factor: f64,
        sentinels: Sentinels<K, V>,
        probing: P,
        allocator: A,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<Self, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        if !desired_load_factor.is_finite()
            || desired_load_factor <= 0.0
            || desired_load_factor > 1.0
        {
            return Err(MapError::InvalidLoadFactor(desired_load_factor));
        }
        let requested = ((elements as f64) / desired_load_factor).ceil();
        if requested > usize::MAX as f64 {
            return Err(MapError::CapacityOverflow);
        }
        Self::with_capacity_in(requested as usize, sentinels, probing, allocator, launch)
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub const fn sentinels(&self) -> Sentinels<K, V> {
        self.sentinels
    }

    pub const fn probing_scheme(&self) -> P {
        self.probing
    }

    pub fn stream(&self) -> &'stream CudaStream {
        self.storage.stream()
    }

    /// Restore every slot to the empty sentinel.
    pub fn clear<M>(&mut self, launch: &Launch<'stream, '_, M>) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        self.ensure_launch(launch)?;
        let submitted = launch.module().clear(
            launch.stream(),
            &mut self.storage,
            self.capacity,
            empty_slot_word(self.sentinels.empty_key, self.sentinels.empty_value),
        );
        finish_submission(submitted, launch.stream())
    }

    /// Insert unique keys and return the number of newly occupied slots.
    ///
    /// This is the Rust-style checked insertion terminal. It reports invalid
    /// sentinel inputs with [`MapError`] instead of requiring callers to infer
    /// status from device-side sentinel values.
    pub fn insert<M, KR, VR>(
        &mut self,
        input: Entries<'_, K, V, KR, VR>,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<usize, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        self.try_insert(input, launch)
    }

    /// Insert unique keys and return the number of newly occupied slots.
    ///
    /// This checked terminal also detects reserved keys. It intentionally has
    /// counting overhead; [`Self::insert_prevalidated`] is the kernel-parity
    /// path.
    pub fn try_insert<M, KR, VR>(
        &mut self,
        input: Entries<'_, K, V, KR, VR>,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<usize, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        self.insert_counted::<false, _, _, _>(input, launch)
    }

    /// Insert or replace values and return the number of newly occupied slots.
    pub fn insert_or_assign<M, KR, VR>(
        &mut self,
        input: Entries<'_, K, V, KR, VR>,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<usize, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        self.insert_counted::<true, _, _, _>(input, launch)
    }

    fn insert_counted<const ASSIGN: bool, M, KR, VR>(
        &mut self,
        input: Entries<'_, K, V, KR, VR>,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<usize, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        self.ensure_launch(launch)?;
        self.ensure_input(&input, launch.stream())?;
        if input.is_empty() {
            return Ok(0);
        }
        self.reset_metadata()?;
        let call = self.insert_call(input, launch.stream());
        let submitted = if ASSIGN {
            launch.module().insert_or_assign(call)
        } else {
            launch.module().insert(call)
        };
        if let Err(error) = submitted {
            let _ = launch.stream().synchronize();
            return Err(error);
        }
        let [inserted, invalid] = self.read_metadata()?;
        if invalid != 0 {
            return Err(MapError::ReservedSentinels {
                count: invalid as usize,
            });
        }
        usize::try_from(inserted).map_err(|_| MapError::CapacityOverflow)
    }

    /// Enqueue the same uncounted kernel timed by cuCollections' insert
    /// benchmark, then synchronize before releasing the input borrows.
    ///
    /// Reserved key/value sentinels are outside this operation's contract and
    /// are not checked in the kernel; use [`Self::try_insert`] when they have
    /// not already been excluded.
    pub fn insert_prevalidated<M, KR, VR>(
        &mut self,
        input: Entries<'_, K, V, KR, VR>,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        self.insert_prevalidated_async(input, launch)?.wait()
    }

    /// Enqueue the uncounted insertion kernel and retain every borrow in a
    /// completion guard. This exposes cuCollections' benchmark timing boundary
    /// without permitting safe fire-and-forget CUDA work.
    pub fn insert_prevalidated_async<'borrow, 'program, M, KR, VR>(
        &'borrow mut self,
        input: Entries<'borrow, K, V, KR, VR>,
        launch: &'borrow Launch<'stream, 'program, M>,
    ) -> Result<PendingInsert<'borrow>, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        self.ensure_launch(launch)?;
        self.ensure_input(&input, launch.stream())?;
        if input.is_empty() {
            // SAFETY: the empty-input path enqueues no GPU work.
            return Ok(PendingInsert(unsafe {
                CudaCompletion::ready(launch.stream(), PendingInsertResources(PhantomData), Ok(()))
            }));
        }
        let supports_erase = self.sentinels.erased_key.is_some();
        let call = self.insert_call(input, launch.stream());
        let submitted = if supports_erase {
            launch.module().insert_prevalidated_with_erase(call)
        } else {
            launch.module().insert_prevalidated(call)
        };
        if let Err(error) = submitted {
            let _ = launch.stream().synchronize();
            return Err(error);
        }
        // SAFETY: the returned guard's `'borrow` marker retains the exclusive
        // map borrow plus both input and launch borrows supplied to this call.
        // The insertion was enqueued only on `launch.stream()`.
        Ok(PendingInsert(unsafe {
            CudaCompletion::from_submission(
                launch.stream(),
                PendingInsertResources(PhantomData),
                |_| Ok(()),
            )
        }))
    }

    /// Exact cuCollections-style find: misses receive the empty-value sentinel.
    pub fn find<M, KR, OR>(
        &self,
        keys: &KR,
        output: &mut OR,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        OR: ContiguousDeviceRangeMut<V> + ?Sized,
    {
        self.ensure_launch(launch)?;
        ensure_query_output(keys, output, launch.stream())?;
        if keys.is_empty() {
            return Ok(());
        }
        let submitted = launch.module().find(
            launch.stream(),
            &self.storage,
            self.capacity,
            keys,
            output,
            self.sentinels.empty_key,
            self.sentinels.empty_value,
            self.sentinels.device_erased_key(),
            self.probing,
        );
        finish_submission(submitted, launch.stream())
    }

    /// Write one `u32` Boolean (0 or 1) for each query key.
    pub fn contains<M, KR, OR>(
        &self,
        keys: &KR,
        output: &mut OR,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        OR: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        self.ensure_launch(launch)?;
        ensure_query_output(keys, output, launch.stream())?;
        if keys.is_empty() {
            return Ok(());
        }
        let submitted = launch.module().contains(
            launch.stream(),
            &self.storage,
            self.capacity,
            keys,
            output,
            self.sentinels.empty_key,
            self.sentinels.empty_value,
            self.sentinels.device_erased_key(),
            self.probing,
        );
        finish_submission(submitted, launch.stream())
    }

    /// Rust-native lookup which keeps the entire mapped-value domain
    /// expressible by writing an explicit presence mask beside the values.
    pub fn find_with_presence<M, KR, VO, PO>(
        &self,
        keys: &KR,
        values: &mut VO,
        present: &mut PO,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VO: ContiguousDeviceRangeMut<V> + ?Sized,
        PO: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        self.find(keys, values, launch)?;
        self.contains(keys, present, launch)
    }

    /// Rust-style lookup: write values plus an explicit 0/1 presence mask.
    ///
    /// Misses still receive the configured empty-value bits in `values`, but
    /// callers should use `present` rather than treating a value sentinel as the
    /// only status channel.
    pub fn get<M, KR, VO, PO>(
        &self,
        keys: &KR,
        values: &mut VO,
        present: &mut PO,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VO: ContiguousDeviceRangeMut<V> + ?Sized,
        PO: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        self.find_with_presence(keys, values, present, launch)
    }

    /// Rust-style presence query alias for [`Self::contains`].
    pub fn contains_key<M, KR, OR>(
        &self,
        keys: &KR,
        output: &mut OR,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
        OR: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        self.contains(keys, output, launch)
    }

    pub fn erase<M, KR>(
        &mut self,
        keys: &KR,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<usize, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
    {
        let erased_key = self.sentinels.erased_key.ok_or(MapError::EraseDisabled)?;
        self.ensure_launch(launch)?;
        keys.ensure_usable_on(launch.stream())?;
        if keys.is_empty() {
            return Ok(0);
        }
        self.reset_metadata()?;
        let submitted = launch.module().erase(
            launch.stream(),
            &mut self.storage,
            self.capacity,
            keys,
            self.sentinels.empty_key,
            self.sentinels.empty_value,
            erased_key,
            self.probing,
        );
        if let Err(error) = submitted {
            let _ = launch.stream().synchronize();
            return Err(error);
        }
        let [erased, _] = self.read_metadata()?;
        usize::try_from(erased).map_err(|_| MapError::CapacityOverflow)
    }

    /// Rust-style deletion alias for [`Self::erase`].
    pub fn remove<M, KR>(
        &mut self,
        keys: &KR,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<usize, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KR: ContiguousDeviceRange<K> + ?Sized,
    {
        self.erase(keys, launch)
    }

    /// Count occupied slots. Like upstream `size()`, this is O(capacity).
    pub fn len<M>(&mut self, launch: &Launch<'stream, '_, M>) -> Result<usize, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
    {
        self.ensure_launch(launch)?;
        self.reset_metadata()?;
        let submitted = launch.module().count(
            launch.stream(),
            &mut self.storage,
            self.capacity,
            self.sentinels.empty_key,
            self.sentinels.device_erased_key(),
        );
        if let Err(error) = submitted {
            let _ = launch.stream().synchronize();
            return Err(error);
        }
        let [count, _] = self.read_metadata()?;
        usize::try_from(count).map_err(|_| MapError::CapacityOverflow)
    }

    /// Compact all entries into caller-owned key and value buffers.
    ///
    /// Outputs conservatively need `capacity()` elements because the map does
    /// not maintain a contended size counter on its fast insertion path.
    pub fn retrieve_all<M, KO, VO>(
        &mut self,
        keys: &mut KO,
        values: &mut VO,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<usize, MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        KO: ContiguousDeviceRangeMut<K> + ?Sized,
        VO: ContiguousDeviceRangeMut<V> + ?Sized,
    {
        self.ensure_launch(launch)?;
        keys.ensure_usable_on(launch.stream())?;
        values.ensure_usable_on(launch.stream())?;
        if keys.len() < self.capacity || values.len() < self.capacity {
            return Err(MapError::OutputTooSmall {
                needed: self.capacity,
                available: keys.len().min(values.len()),
            });
        }
        self.reset_metadata()?;
        let submitted = launch.module().retrieve_all(
            launch.stream(),
            &mut self.storage,
            self.capacity,
            keys,
            values,
            self.sentinels.empty_key,
            self.sentinels.device_erased_key(),
        );
        if let Err(error) = submitted {
            let _ = launch.stream().synchronize();
            return Err(error);
        }
        let [count, _] = self.read_metadata()?;
        usize::try_from(count).map_err(|_| MapError::CapacityOverflow)
    }

    /// Rebuild the table at a new valid capacity, purging all tombstones.
    ///
    /// The allocator is cloned only for this operation; ordinary map use does
    /// not impose a cloneable-resource requirement.
    pub fn rehash<M>(
        &mut self,
        requested_capacity: usize,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        A: Clone,
    {
        self.ensure_launch(launch)?;
        let destination_capacity = valid_capacity::<K, P>(requested_capacity, CG as usize)
            .ok_or(MapError::CapacityOverflow)?;
        let _ = u32::try_from(destination_capacity).map_err(|_| MapError::CapacityOverflow)?;
        let entries = self.len(launch)?;
        if entries > destination_capacity {
            return Err(MapError::InsufficientCapacity {
                entries,
                capacity: destination_capacity,
            });
        }
        let allocation_len = slot_storage_words::<V>(destination_capacity)?
            .checked_add(METADATA_WORDS)
            .ok_or(MapError::CapacityOverflow)?;
        let mut destination = DeviceBuffer::zeroed_in(
            self.storage.allocator().clone(),
            self.storage.stream(),
            allocation_len,
        )?;
        let cleared = launch.module().clear(
            launch.stream(),
            &mut destination,
            destination_capacity,
            empty_slot_word(self.sentinels.empty_key, self.sentinels.empty_value),
        );
        if let Err(error) = cleared {
            let _ = launch.stream().synchronize();
            return Err(error);
        }
        let submitted = launch.module().rehash(
            launch.stream(),
            &self.storage,
            self.capacity,
            &mut destination,
            destination_capacity,
            self.sentinels.empty_key,
            self.sentinels.empty_value,
            self.sentinels.device_erased_key(),
            self.probing,
        );
        finish_submission(submitted, launch.stream())?;
        self.storage = destination;
        self.capacity = destination_capacity;
        Ok(())
    }

    /// Rebuild without changing capacity.
    pub fn rehash_in_place<M>(&mut self, launch: &Launch<'stream, '_, M>) -> Result<(), MapError>
    where
        M: StaticMapBackend<K, V, P, CG>,
        A: Clone,
    {
        self.rehash(self.capacity, launch)
    }

    fn ensure_launch<M>(&self, launch: &Launch<'stream, '_, M>) -> Result<(), MapError> {
        self.storage.ensure_usable_on(launch.stream())?;
        Ok(())
    }

    fn ensure_input<KR, VR>(
        &self,
        input: &Entries<'_, K, V, KR, VR>,
        stream: &CudaStream,
    ) -> Result<(), MapError>
    where
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        input.keys.ensure_usable_on(stream)?;
        input.values.ensure_usable_on(stream)?;
        Ok(())
    }

    fn insert_call<'borrow, KR, VR>(
        &'borrow mut self,
        input: Entries<'borrow, K, V, KR, VR>,
        stream: &'borrow CudaStream,
    ) -> InsertCall<'borrow, 'stream, K, V, P, A, KR, VR>
    where
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        InsertCall {
            stream,
            storage: &mut self.storage,
            capacity: self.capacity,
            keys: input.keys,
            values: input.values,
            empty_key: self.sentinels.empty_key,
            empty_value: self.sentinels.empty_value,
            erased_key: self.sentinels.device_erased_key(),
            probing: self.probing,
        }
    }

    fn metadata_offset(&self) -> Result<usize, MapError> {
        slot_storage_words::<V>(self.capacity)
    }

    fn reset_metadata(&mut self) -> Result<(), MapError> {
        let metadata_offset = self.metadata_offset()?;
        let byte_offset = metadata_offset
            .checked_mul(core::mem::size_of::<u64>())
            .ok_or(MapError::CapacityOverflow)?;
        let byte_len = METADATA_WORDS
            .checked_mul(core::mem::size_of::<u64>())
            .ok_or(MapError::CapacityOverflow)?;
        if byte_len != 0 {
            unsafe {
                memset_d8_async(
                    self.storage.cu_deviceptr() + byte_offset as u64,
                    0,
                    byte_len,
                    self.storage.stream().cu_stream(),
                )?;
            }
        }
        Ok(())
    }

    fn read_metadata(&self) -> Result<[u64; METADATA_WORDS], MapError> {
        let metadata_offset = self.metadata_offset()?;
        let byte_offset = metadata_offset
            .checked_mul(core::mem::size_of::<u64>())
            .ok_or(MapError::CapacityOverflow)?;
        let byte_len = METADATA_WORDS
            .checked_mul(core::mem::size_of::<u64>())
            .ok_or(MapError::CapacityOverflow)?;
        let mut values = [0u64; METADATA_WORDS];
        if byte_len != 0 {
            unsafe {
                memcpy_dtoh_async(
                    values.as_mut_ptr(),
                    self.storage.cu_deviceptr() + byte_offset as u64,
                    byte_len,
                    self.storage.stream().cu_stream(),
                )?;
            }
            self.storage.stream().synchronize()?;
        }
        Ok(values)
    }
}

fn validate_group_size<const CG: u32>() -> Result<(), MapError> {
    if matches!(CG, 1 | 2 | 4 | 8 | 16 | 32) && 128 % CG == 0 {
        Ok(())
    } else {
        Err(MapError::InvalidCooperativeGroup(CG))
    }
}

fn finish_submission(submitted: Result<(), MapError>, stream: &CudaStream) -> Result<(), MapError> {
    let synchronized = stream.synchronize().map_err(MapError::Launch);
    submitted.and(synchronized)
}

fn ensure_query_output<K, O, KR, OR>(
    keys: &KR,
    output: &mut OR,
    stream: &CudaStream,
) -> Result<(), MapError>
where
    KR: ContiguousDeviceRange<K> + ?Sized,
    OR: ContiguousDeviceRangeMut<O> + ?Sized,
{
    keys.ensure_usable_on(stream)?;
    output.ensure_usable_on(stream)?;
    if output.len() < keys.len() {
        return Err(MapError::OutputTooSmall {
            needed: keys.len(),
            available: output.len(),
        });
    }
    Ok(())
}

fn slot_storage_words<V: MapValue>(capacity: usize) -> Result<usize, MapError> {
    capacity
        .checked_mul(V::SLOT_WORDS)
        .ok_or(MapError::CapacityOverflow)
}

#[inline(always)]
fn pack<K: MapKey, V: MapValue>(key: K, value: V) -> u64 {
    key.to_bits() as u64 | ((value.to_bits() & 0xffff_ffff) << 32)
}

#[inline(always)]
fn empty_slot_word<K: MapKey, V: MapValue>(key: K, value: V) -> u64 {
    if V::SLOT_WORDS == 1 {
        pack(key, value)
    } else {
        wide_control(key, WIDE_EMPTY)
    }
}

const WIDE_EMPTY: u32 = 0;
const WIDE_FULL: u32 = 1;
const WIDE_BUSY: u32 = 2;

#[inline(always)]
fn wide_control<K: MapKey>(key: K, state: u32) -> u64 {
    key.to_bits() as u64 | ((state as u64) << 32)
}

#[inline(always)]
fn wide_state(control: u64) -> u32 {
    (control >> 32) as u32
}

#[inline(always)]
fn packed_key<K: MapKey>(slot: u64) -> K {
    K::from_bits(slot as u32)
}

#[inline(always)]
fn packed_value<V: MapValue>(slot: u64) -> V {
    V::from_bits((slot >> 32) & 0xffff_ffff)
}

#[inline(always)]
fn shuffle_u64<const CG: u32>(
    tile: &cuda_device::cooperative_groups::WarpTile<CG>,
    value: u64,
    source: u32,
) -> u64 {
    let low = tile.shfl(value as u32, source) as u64;
    let high = tile.shfl((value >> 32) as u32, source) as u64;
    low | (high << 32)
}

/// Copyable device-side mutation capability.
///
/// Every lane in one `WarpTile<CG>` must call each method with identical
/// arguments. Host-side mutation remains affine and all safe terminals wait
/// for completion before releasing their borrows.
#[derive(Clone, Copy)]
pub struct StaticMapRef<'a, K, V, P, const CG: u32>
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    slots: &'a [DeviceAtomicU64],
    empty_key: K,
    empty_value: V,
    erased_key: K,
    probing: P,
}

impl<'a, K, V, P, const CG: u32> StaticMapRef<'a, K, V, P, CG>
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    /// Construct a cooperative device-side mutation reference.
    ///
    /// # Safety
    ///
    /// `CG` must be one of `1, 2, 4, 8, 16, 32`; `slots` must describe a
    /// non-empty slot-word allocation whose logical slot count is divisible by
    /// `CG`; and every lane in each participating `WarpTile<CG>` must invoke
    /// the map operation convergently with identical arguments. The allocation
    /// must remain under the static-map atomic protocol for `'a`.
    #[inline(always)]
    pub const unsafe fn new(
        slots: &'a [DeviceAtomicU64],
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Self {
        Self {
            slots,
            empty_key,
            empty_value,
            erased_key,
            probing,
        }
    }

    #[inline(always)]
    fn capacity(&self) -> usize {
        self.slots.len() / V::SLOT_WORDS
    }

    #[inline(always)]
    fn word_index(index: usize) -> usize {
        index * V::SLOT_WORDS
    }

    #[inline(always)]
    fn value_word_index(index: usize) -> usize {
        Self::word_index(index) + 1
    }

    #[inline(always)]
    fn optimistic_slot_snapshot(&self, index: usize) -> u64 {
        self.slots[Self::word_index(index)].cached_snapshot()
    }

    #[inline(always)]
    fn optimistic_key_snapshot(&self, index: usize) -> u32 {
        self.slots[Self::word_index(index)].cached_low_u32()
    }

    #[inline(always)]
    fn wide_control_acquire(&self, index: usize) -> u64 {
        self.slots[Self::word_index(index)].load(cuda_device::atomic::AtomicOrdering::Acquire)
    }

    #[inline(always)]
    pub fn insert(&self, key: K, value: V) -> bool {
        self.insert_counted::<false>(key, value)
    }

    #[inline(always)]
    pub fn insert_or_assign(&self, key: K, value: V) -> bool {
        self.insert_counted::<true>(key, value)
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn insert_counted<const ASSIGN: bool>(&self, key: K, value: V) -> bool {
        self.insert_impl::<true, true, ASSIGN>(key, value)
    }

    #[inline(always)]
    pub fn insert_without_tombstones(&self, key: K, value: V) -> bool {
        self.insert_impl::<false, true, false>(key, value)
    }

    /// Insert a prevalidated entry through the compile-time tombstone policy.
    #[doc(hidden)]
    #[inline(always)]
    pub fn insert_prevalidated<const SUPPORTS_ERASE: bool>(&self, key: K, value: V) -> bool {
        self.insert_impl::<SUPPORTS_ERASE, false, false>(key, value)
    }

    #[inline(always)]
    fn insert_impl<const SUPPORTS_ERASE: bool, const CHECK_SENTINELS: bool, const ASSIGN: bool>(
        &self,
        key: K,
        value: V,
    ) -> bool {
        if V::SLOT_WORDS == 1 {
            self.insert_packed::<SUPPORTS_ERASE, CHECK_SENTINELS, ASSIGN>(key, value)
        } else {
            self.insert_wide::<SUPPORTS_ERASE, CHECK_SENTINELS, ASSIGN>(key, value)
        }
    }

    #[inline(always)]
    fn insert_packed<
        const SUPPORTS_ERASE: bool,
        const CHECK_SENTINELS: bool,
        const ASSIGN: bool,
    >(
        &self,
        key: K,
        value: V,
    ) -> bool {
        if CHECK_SENTINELS
            && (key == self.empty_key
                || (SUPPORTS_ERASE && key == self.erased_key)
                || value.to_bits() == self.empty_value.to_bits())
        {
            return false;
        }
        let tile = this_thread_block().tiled_partition::<CG>();
        let rank = tile.thread_rank();
        let groups = self.capacity() / CG as usize;
        let start = self.probing.initial_group(key, groups);
        let step = self.probing.step_groups(key, groups);
        let desired = pack(key, value);
        let empty_slot = pack(self.empty_key, self.empty_value);
        let erased_slot = pack(self.erased_key, self.empty_value);

        loop {
            let mut group = start;
            let mut first_erased = usize::MAX;
            let mut visited = 0_usize;
            while visited < groups {
                let index = group * CG as usize + rank as usize;
                let observed = if SUPPORTS_ERASE {
                    self.optimistic_slot_snapshot(index)
                } else {
                    self.optimistic_key_snapshot(index) as u64
                };
                let observed_key = packed_key::<K>(observed);
                let matching = tile.ballot(observed_key == key);
                if matching != 0 {
                    if !ASSIGN {
                        return false;
                    }
                    let owner = matching.trailing_zeros();
                    let previous = shuffle_u64(&tile, observed, owner);
                    let owner_index = group * CG as usize + owner as usize;
                    let status = if rank == 0 {
                        match self.slots[Self::word_index(owner_index)]
                            .compare_exchange_relaxed(previous, desired)
                        {
                            Ok(_) => 1,
                            Err(actual) if packed_key::<K>(actual) == key => 2,
                            Err(_) => 0,
                        }
                    } else {
                        0
                    };
                    match tile.shfl(status, 0) {
                        1 | 2 => return false,
                        _ => break,
                    }
                }

                if SUPPORTS_ERASE && first_erased == usize::MAX {
                    let erased = tile.ballot(observed_key == self.erased_key);
                    if erased != 0 {
                        first_erased = group * CG as usize + erased.trailing_zeros() as usize;
                    }
                }

                let empty = tile.ballot(observed_key == self.empty_key);
                if empty != 0 {
                    let empty_owner = empty.trailing_zeros();
                    if !SUPPORTS_ERASE {
                        let status = if rank == empty_owner {
                            match self.slots[Self::word_index(index)]
                                .compare_exchange_relaxed(empty_slot, desired)
                            {
                                Ok(_) => 1,
                                Err(actual) if packed_key::<K>(actual) == key => 2,
                                Err(_) => 0,
                            }
                        } else {
                            0
                        };
                        match tile.shfl(status, empty_owner) {
                            1 => return true,
                            2 => return false,
                            _ => break,
                        }
                    }
                    let target = if SUPPORTS_ERASE && first_erased != usize::MAX {
                        first_erased
                    } else {
                        group * CG as usize + empty_owner as usize
                    };
                    let expected = if SUPPORTS_ERASE && first_erased != usize::MAX {
                        erased_slot
                    } else {
                        shuffle_u64(&tile, observed, empty_owner)
                    };
                    let status = if rank == 0 {
                        match self.slots[Self::word_index(target)]
                            .compare_exchange_relaxed(expected, desired)
                        {
                            Ok(_) => 1,
                            Err(actual) if packed_key::<K>(actual) == key => 2,
                            Err(_) => 0,
                        }
                    } else {
                        0
                    };
                    match tile.shfl(status, 0) {
                        1 => return true,
                        2 => return false,
                        _ => break,
                    }
                }

                visited += 1;
                group += step;
                if group >= groups {
                    group -= groups;
                }
            }

            if visited == groups {
                if !SUPPORTS_ERASE || first_erased == usize::MAX {
                    return false;
                }
                let status = if rank == 0 {
                    match self.slots[Self::word_index(first_erased)]
                        .compare_exchange_relaxed(erased_slot, desired)
                    {
                        Ok(_) => 1,
                        Err(actual) if packed_key::<K>(actual) == key => 2,
                        Err(_) => 0,
                    }
                } else {
                    0
                };
                match tile.shfl(status, 0) {
                    1 => return true,
                    2 => return false,
                    _ => {}
                }
            }
        }
    }

    #[inline(always)]
    fn insert_wide<const SUPPORTS_ERASE: bool, const CHECK_SENTINELS: bool, const ASSIGN: bool>(
        &self,
        key: K,
        value: V,
    ) -> bool {
        if CHECK_SENTINELS
            && (key == self.empty_key
                || (SUPPORTS_ERASE && key == self.erased_key)
                || value.to_bits() == self.empty_value.to_bits())
        {
            return false;
        }
        let tile = this_thread_block().tiled_partition::<CG>();
        let rank = tile.thread_rank();
        let groups = self.capacity() / CG as usize;
        let start = self.probing.initial_group(key, groups);
        let step = self.probing.step_groups(key, groups);
        let empty_slot = wide_control(self.empty_key, WIDE_EMPTY);
        let erased_slot = wide_control(self.erased_key, WIDE_EMPTY);
        let busy_slot = wide_control(key, WIDE_BUSY);
        let full_slot = wide_control(key, WIDE_FULL);

        loop {
            let mut group = start;
            let mut first_erased = usize::MAX;
            let mut visited = 0_usize;
            while visited < groups {
                let index = group * CG as usize + rank as usize;
                let observed = self.wide_control_acquire(index);
                let observed_key = packed_key::<K>(observed);
                let observed_full = wide_state(observed) == WIDE_FULL;
                let busy = tile.ballot(wide_state(observed) == WIDE_BUSY);
                if busy != 0 {
                    break;
                }
                let matching = tile.ballot(observed_full && observed_key == key);
                if matching != 0 {
                    if !ASSIGN {
                        return false;
                    }
                    let owner = matching.trailing_zeros();
                    let owner_index = group * CG as usize + owner as usize;
                    let status = if rank == 0 {
                        match self.slots[Self::word_index(owner_index)].compare_exchange(
                            wide_control(key, WIDE_FULL),
                            wide_control(key, WIDE_BUSY),
                            cuda_device::atomic::AtomicOrdering::AcqRel,
                            cuda_device::atomic::AtomicOrdering::Acquire,
                        ) {
                            Ok(_) => {
                                self.slots[Self::value_word_index(owner_index)].store(
                                    value.to_bits(),
                                    cuda_device::atomic::AtomicOrdering::Relaxed,
                                );
                                self.slots[Self::word_index(owner_index)]
                                    .store(full_slot, cuda_device::atomic::AtomicOrdering::Release);
                                1
                            }
                            Err(actual)
                                if packed_key::<K>(actual) == key
                                    && wide_state(actual) == WIDE_FULL =>
                            {
                                2
                            }
                            Err(_) => 0,
                        }
                    } else {
                        0
                    };
                    match tile.shfl(status, 0) {
                        1 | 2 => return false,
                        _ => break,
                    }
                }

                if SUPPORTS_ERASE && first_erased == usize::MAX {
                    let erased = tile.ballot(observed_key == self.erased_key);
                    if erased != 0 {
                        first_erased = group * CG as usize + erased.trailing_zeros() as usize;
                    }
                }

                let empty = tile.ballot(observed_key == self.empty_key);
                if empty != 0 {
                    let empty_owner = empty.trailing_zeros();
                    let target = if SUPPORTS_ERASE && first_erased != usize::MAX {
                        first_erased
                    } else {
                        group * CG as usize + empty_owner as usize
                    };
                    let expected = if SUPPORTS_ERASE && first_erased != usize::MAX {
                        erased_slot
                    } else {
                        empty_slot
                    };
                    let status = if rank == 0 {
                        match self.slots[Self::word_index(target)].compare_exchange(
                            expected,
                            busy_slot,
                            cuda_device::atomic::AtomicOrdering::AcqRel,
                            cuda_device::atomic::AtomicOrdering::Acquire,
                        ) {
                            Ok(_) => {
                                self.slots[Self::value_word_index(target)].store(
                                    value.to_bits(),
                                    cuda_device::atomic::AtomicOrdering::Relaxed,
                                );
                                self.slots[Self::word_index(target)]
                                    .store(full_slot, cuda_device::atomic::AtomicOrdering::Release);
                                1
                            }
                            Err(actual)
                                if packed_key::<K>(actual) == key
                                    && wide_state(actual) == WIDE_FULL =>
                            {
                                2
                            }
                            Err(_) => 0,
                        }
                    } else {
                        0
                    };
                    match tile.shfl(status, 0) {
                        1 => return true,
                        2 => return false,
                        _ => break,
                    }
                }

                visited += 1;
                group += step;
                if group >= groups {
                    group -= groups;
                }
            }

            if visited == groups {
                if !SUPPORTS_ERASE || first_erased == usize::MAX {
                    return false;
                }
                let status = if rank == 0 {
                    match self.slots[Self::word_index(first_erased)].compare_exchange(
                        erased_slot,
                        busy_slot,
                        cuda_device::atomic::AtomicOrdering::AcqRel,
                        cuda_device::atomic::AtomicOrdering::Acquire,
                    ) {
                        Ok(_) => {
                            self.slots[Self::value_word_index(first_erased)].store(
                                value.to_bits(),
                                cuda_device::atomic::AtomicOrdering::Relaxed,
                            );
                            self.slots[Self::word_index(first_erased)]
                                .store(full_slot, cuda_device::atomic::AtomicOrdering::Release);
                            1
                        }
                        Err(actual)
                            if packed_key::<K>(actual) == key
                                && wide_state(actual) == WIDE_FULL =>
                        {
                            2
                        }
                        Err(_) => 0,
                    }
                } else {
                    0
                };
                match tile.shfl(status, 0) {
                    1 => return true,
                    2 => return false,
                    _ => {}
                }
            }
        }
    }

    #[inline(always)]
    pub fn erase(&self, key: K) -> bool {
        if V::SLOT_WORDS == 1 {
            self.erase_packed(key)
        } else {
            self.erase_wide(key)
        }
    }

    #[inline(always)]
    fn erase_packed(&self, key: K) -> bool {
        if self.erased_key == self.empty_key || key == self.empty_key || key == self.erased_key {
            return false;
        }
        let tile = this_thread_block().tiled_partition::<CG>();
        let rank = tile.thread_rank();
        let groups = self.capacity() / CG as usize;
        let start = self.probing.initial_group(key, groups);
        let step = self.probing.step_groups(key, groups);
        let erased_slot = pack(self.erased_key, self.empty_value);
        loop {
            let mut group = start;
            let mut visited = 0_usize;
            while visited < groups {
                let index = group * CG as usize + rank as usize;
                let observed = self.optimistic_slot_snapshot(index);
                let observed_key = packed_key::<K>(observed);
                let matching = tile.ballot(observed_key == key);
                if matching != 0 {
                    let owner = matching.trailing_zeros();
                    let previous = shuffle_u64(&tile, observed, owner);
                    let owner_index = group * CG as usize + owner as usize;
                    let status = if rank == 0 {
                        self.slots[Self::word_index(owner_index)]
                            .compare_exchange_relaxed(previous, erased_slot)
                            .is_ok() as u32
                    } else {
                        0
                    };
                    if tile.shfl(status, 0) != 0 {
                        return true;
                    }
                    break;
                }
                if tile.ballot(observed_key == self.empty_key) != 0 {
                    return false;
                }
                visited += 1;
                group = (group + step) % groups;
            }
            if visited == groups {
                return false;
            }
        }
    }

    #[inline(always)]
    fn erase_wide(&self, key: K) -> bool {
        if self.erased_key == self.empty_key || key == self.empty_key || key == self.erased_key {
            return false;
        }
        let tile = this_thread_block().tiled_partition::<CG>();
        let rank = tile.thread_rank();
        let groups = self.capacity() / CG as usize;
        let start = self.probing.initial_group(key, groups);
        let step = self.probing.step_groups(key, groups);
        let erased_slot = wide_control(self.erased_key, WIDE_EMPTY);
        loop {
            let mut group = start;
            let mut visited = 0_usize;
            while visited < groups {
                let index = group * CG as usize + rank as usize;
                let observed = self.wide_control_acquire(index);
                if wide_state(observed) == WIDE_BUSY {
                    break;
                }
                let observed_key = packed_key::<K>(observed);
                let matching =
                    tile.ballot(wide_state(observed) == WIDE_FULL && observed_key == key);
                if matching != 0 {
                    let owner = matching.trailing_zeros();
                    let owner_index = group * CG as usize + owner as usize;
                    let status = if rank == 0 {
                        self.slots[Self::word_index(owner_index)]
                            .compare_exchange(
                                wide_control(key, WIDE_FULL),
                                erased_slot,
                                cuda_device::atomic::AtomicOrdering::AcqRel,
                                cuda_device::atomic::AtomicOrdering::Acquire,
                            )
                            .is_ok() as u32
                    } else {
                        0
                    };
                    if tile.shfl(status, 0) != 0 {
                        return true;
                    }
                    break;
                }
                if tile.ballot(observed_key == self.empty_key) != 0 {
                    return false;
                }
                visited += 1;
                group = (group + step) % groups;
            }
            if visited == groups {
                return false;
            }
        }
    }
}

/// Copyable lookup-only device reference.
#[derive(Clone, Copy)]
pub struct StaticMapReadRef<'a, K, V, P, const CG: u32>
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    slots: &'a [u64],
    empty_key: K,
    erased_key: K,
    probing: P,
    marker: PhantomData<V>,
}

impl<'a, K, V, P, const CG: u32> StaticMapReadRef<'a, K, V, P, CG>
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    #[inline(always)]
    pub const fn new(
        slots: &'a [u64],
        empty_key: K,
        _empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Self {
        Self {
            slots,
            empty_key,
            erased_key,
            probing,
            marker: PhantomData,
        }
    }

    #[inline(always)]
    fn capacity(&self) -> usize {
        self.slots.len() / V::SLOT_WORDS
    }

    #[inline(always)]
    fn word_index(index: usize) -> usize {
        index * V::SLOT_WORDS
    }

    #[inline(always)]
    fn value_word_index(index: usize) -> usize {
        Self::word_index(index) + 1
    }

    #[inline(always)]
    pub fn find(&self, key: K) -> Option<V> {
        if key == self.empty_key || key == self.erased_key {
            return None;
        }
        if V::SLOT_WORDS == 1 {
            return self.find_packed(key);
        }
        self.find_wide(key)
    }

    #[inline(always)]
    fn find_packed(&self, key: K) -> Option<V> {
        let tile = this_thread_block().tiled_partition::<CG>();
        let rank = tile.thread_rank();
        let groups = self.capacity() / CG as usize;
        let mut group = self.probing.initial_group(key, groups);
        let step = self.probing.step_groups(key, groups);
        let mut visited = 0_usize;
        while visited < groups {
            let observed = self.slots[Self::word_index(group * CG as usize + rank as usize)];
            let observed_key = packed_key::<K>(observed);
            let matching = tile.ballot(observed_key == key);
            if matching != 0 {
                let packed = shuffle_u64(&tile, observed, matching.trailing_zeros());
                return Some(packed_value(packed));
            }
            if tile.ballot(observed_key == self.empty_key) != 0 {
                return None;
            }
            visited += 1;
            group = (group + step) % groups;
        }
        None
    }

    #[inline(always)]
    fn find_wide(&self, key: K) -> Option<V> {
        let tile = this_thread_block().tiled_partition::<CG>();
        let rank = tile.thread_rank();
        let groups = self.capacity() / CG as usize;
        let start = self.probing.initial_group(key, groups);
        let step = self.probing.step_groups(key, groups);
        loop {
            let mut group = start;
            let mut visited = 0_usize;
            while visited < groups {
                let index = group * CG as usize + rank as usize;
                let control = self.slots[Self::word_index(index)];
                if wide_state(control) == WIDE_BUSY {
                    break;
                }
                let observed_key = packed_key::<K>(control);
                let matching = tile.ballot(wide_state(control) == WIDE_FULL && observed_key == key);
                if matching != 0 {
                    let owner = matching.trailing_zeros();
                    let value =
                        self.slots[Self::value_word_index(group * CG as usize + owner as usize)];
                    let value = shuffle_u64(&tile, value, owner);
                    return Some(V::from_bits(value));
                }
                if tile.ballot(observed_key == self.empty_key) != 0 {
                    return None;
                }
                visited += 1;
                group = (group + step) % groups;
            }
            if visited == groups {
                return None;
            }
        }
    }

    #[inline(always)]
    pub fn contains(&self, key: K) -> bool {
        self.find(key).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinels_reject_equal_erased_key() {
        assert!(matches!(
            Sentinels::new(u32::MAX, u32::MAX).with_erased_key(u32::MAX),
            Err(MapError::EmptyKeyEqualsErasedKey)
        ));
    }

    #[test]
    fn packed_words_preserve_signed_and_float_bits() {
        let word = pack(-7_i32, f32::from_bits(0x7fc0_1234));
        assert_eq!(packed_key::<i32>(word), -7);
        assert_eq!(packed_value::<f32>(word).to_bits(), 0x7fc0_1234);
    }

    #[test]
    fn slot_words_reserve_two_words_for_u64_values() {
        assert_eq!(slot_storage_words::<u32>(7).unwrap(), 7);
        assert_eq!(slot_storage_words::<u64>(7).unwrap(), 14);
        assert_eq!(empty_slot_word(u32::MAX, u32::MAX as u64), u32::MAX as u64);
        assert_eq!(pack(1_u32, u64::MAX), 0xffff_ffff_0000_0001);
    }
}
