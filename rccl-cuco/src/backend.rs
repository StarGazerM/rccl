use cuda_core::{CudaStream, LaunchConfig1D, LaunchContractError};
use cuda_device::atomic::{DeviceAtomicU32, DeviceAtomicU64};
use cuda_host::{KernelSliceArg, KernelSliceArgMut};
use rccl_memory::{ContiguousDeviceRange, ContiguousDeviceRangeMut, DeviceAllocator, DeviceBuffer};

use crate::{MapError, MapKey, MapValue, ProbingScheme};

/// Affine host packet shared by the distinct bulk insertion terminals.
pub struct InsertCall<'borrow, 'stream, K, V, P, A, KR: ?Sized, VR: ?Sized>
where
    A: DeviceAllocator,
{
    pub(crate) stream: &'borrow CudaStream,
    pub(crate) storage: &'borrow mut DeviceBuffer<'stream, u64, A>,
    pub(crate) capacity: usize,
    pub(crate) keys: &'borrow KR,
    pub(crate) values: &'borrow VR,
    pub(crate) empty_key: K,
    pub(crate) empty_value: V,
    pub(crate) erased_key: K,
    pub(crate) probing: P,
}

#[doc(hidden)]
pub trait StaticMapBackend<K, V, P, const CG: u32>: Sized
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    fn clear<A: DeviceAllocator>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        empty_slot: u64,
    ) -> Result<(), MapError>;

    fn insert<A, KR, VR>(
        &self,
        call: InsertCall<'_, '_, K, V, P, A, KR, VR>,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized;

    fn insert_or_assign<A, KR, VR>(
        &self,
        call: InsertCall<'_, '_, K, V, P, A, KR, VR>,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized;

    fn insert_prevalidated<A, KR, VR>(
        &self,
        call: InsertCall<'_, '_, K, V, P, A, KR, VR>,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized;

    fn insert_prevalidated_with_erase<A, KR, VR>(
        &self,
        call: InsertCall<'_, '_, K, V, P, A, KR, VR>,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized;

    #[allow(clippy::too_many_arguments)]
    fn find<A, KR, OR>(
        &self,
        stream: &CudaStream,
        storage: &DeviceBuffer<'_, u64, A>,
        capacity: usize,
        keys: &KR,
        output: &mut OR,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        OR: ContiguousDeviceRangeMut<V> + ?Sized;

    #[allow(clippy::too_many_arguments)]
    fn contains<A, KR, OR>(
        &self,
        stream: &CudaStream,
        storage: &DeviceBuffer<'_, u64, A>,
        capacity: usize,
        keys: &KR,
        output: &mut OR,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        OR: ContiguousDeviceRangeMut<u32> + ?Sized;

    #[allow(clippy::too_many_arguments)]
    fn erase<A, KR>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        keys: &KR,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized;

    #[allow(clippy::too_many_arguments)]
    fn retrieve_all<A, KO, VO>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        keys: &mut KO,
        values: &mut VO,
        empty_key: K,
        erased_key: K,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KO: ContiguousDeviceRangeMut<K> + ?Sized,
        VO: ContiguousDeviceRangeMut<V> + ?Sized;

    fn count<A: DeviceAllocator>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        empty_key: K,
        erased_key: K,
    ) -> Result<(), MapError>;

    #[allow(clippy::too_many_arguments)]
    fn rehash<SA: DeviceAllocator, DA: DeviceAllocator>(
        &self,
        stream: &CudaStream,
        source: &DeviceBuffer<'_, u64, SA>,
        source_capacity: usize,
        destination: &mut DeviceBuffer<'_, u64, DA>,
        destination_capacity: usize,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>;
}

#[doc(hidden)]
pub trait StaticMapRangeBackend<K, V, P, const CG: u32>: Sized
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    #[allow(clippy::too_many_arguments)]
    fn insert_range<const ASSIGN: bool, A, SA, Input>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        input: Input,
        len: usize,
        status: &mut DeviceBuffer<'_, u64, SA>,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        SA: DeviceAllocator,
        Input: crate::range::DeviceMapRead<Item = crate::KeyValue<K, V>>;

    #[allow(clippy::too_many_arguments)]
    fn insert_prevalidated_range<const SUPPORTS_ERASE: bool, A, SA, Input>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        input: Input,
        len: usize,
        status: &mut DeviceBuffer<'_, u64, SA>,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        SA: DeviceAllocator,
        Input: crate::range::DeviceMapRead<Item = crate::KeyValue<K, V>>;

    #[allow(clippy::too_many_arguments)]
    fn find_range<A, SA, Input, Output>(
        &self,
        stream: &CudaStream,
        storage: &DeviceBuffer<'_, u64, A>,
        capacity: usize,
        input: Input,
        output: Output,
        len: usize,
        status: &mut DeviceBuffer<'_, u64, SA>,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        SA: DeviceAllocator,
        Input: crate::range::DeviceMapRead<Item = K>,
        Output: crate::range::DeviceMapWrite<Item = V>;

    #[allow(clippy::too_many_arguments)]
    fn contains_range<A, SA, Input, Output>(
        &self,
        stream: &CudaStream,
        storage: &DeviceBuffer<'_, u64, A>,
        capacity: usize,
        input: Input,
        output: Output,
        len: usize,
        status: &mut DeviceBuffer<'_, u64, SA>,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        SA: DeviceAllocator,
        Input: crate::range::DeviceMapRead<Item = K>,
        Output: crate::range::DeviceMapWrite<Item = u32>;

    #[allow(clippy::too_many_arguments)]
    fn contains_if_range<A, SA, Input, Stencil, Output, Predicate>(
        &self,
        stream: &CudaStream,
        storage: &DeviceBuffer<'_, u64, A>,
        capacity: usize,
        input: Input,
        stencil: Stencil,
        output: Output,
        predicate: Predicate,
        len: usize,
        status: &mut DeviceBuffer<'_, u64, SA>,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        SA: DeviceAllocator,
        Input: crate::range::DeviceMapRead<Item = K>,
        Stencil: crate::range::DeviceMapRead,
        Output: crate::range::DeviceMapWrite<Item = u32>,
        Predicate: Fn(Stencil::Item) -> bool + Copy;
}

#[doc(hidden)]
pub fn launch_config(elements: u32) -> LaunchConfig1D {
    const BLOCK: u32 = 128;
    LaunchConfig1D::new(elements.div_ceil(BLOCK), BLOCK, 0)
}

/// Direct bindings to one cuda-oxide generated static-map kernel family.
///
/// The declarative macro implements only this trait. Runtime validation,
/// configuration, and memory-adapter construction remain in the ordinary
/// [`StaticMapBackend`] implementation below.
#[doc(hidden)]
pub trait StaticMapKernelModule {
    fn launch_clear<S>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &mut S,
        empty_slot: u64,
    ) -> Result<(), LaunchContractError>
    where
        S: KernelSliceArgMut<Elem = u64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_insert<K, V, P, const CG: u32, S, KR, VR, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        keys: &KR,
        values: &VR,
        inserted: &C,
        invalid: &C,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        KR: KernelSliceArg<Elem = K>,
        VR: KernelSliceArg<Elem = V>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_insert_or_assign<K, V, P, const CG: u32, S, KR, VR, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        keys: &KR,
        values: &VR,
        inserted: &C,
        invalid: &C,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        KR: KernelSliceArg<Elem = K>,
        VR: KernelSliceArg<Elem = V>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_insert_prevalidated<K, V, P, const CG: u32, S, KR, VR>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        keys: &KR,
        values: &VR,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        KR: KernelSliceArg<Elem = K>,
        VR: KernelSliceArg<Elem = V>;

    #[allow(clippy::too_many_arguments)]
    fn launch_insert_prevalidated_with_erase<K, V, P, const CG: u32, S, KR, VR>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        keys: &KR,
        values: &VR,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        KR: KernelSliceArg<Elem = K>,
        VR: KernelSliceArg<Elem = V>;

    #[allow(clippy::too_many_arguments)]
    fn launch_find<K, V, P, const CG: u32, S, KR, OR>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        keys: &KR,
        output: &OR,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        S: KernelSliceArg<Elem = u64>,
        KR: KernelSliceArg<Elem = K>,
        OR: KernelSliceArg<Elem = V::Atomic>;

    #[allow(clippy::too_many_arguments)]
    fn launch_contains<K, V, P, const CG: u32, S, KR, OR>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        keys: &KR,
        output: &OR,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        S: KernelSliceArg<Elem = u64>,
        KR: KernelSliceArg<Elem = K>,
        OR: KernelSliceArg<Elem = DeviceAtomicU32>;

    #[allow(clippy::too_many_arguments)]
    fn launch_erase<K, V, P, const CG: u32, S, KR, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        keys: &KR,
        erased: &C,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        KR: KernelSliceArg<Elem = K>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_retrieve_all<K, V, S, KO, VO, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        keys: &KO,
        values: &VO,
        count: &C,
        empty_key: K,
        erased_key: K,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        S: KernelSliceArg<Elem = u64>,
        KO: KernelSliceArg<Elem = DeviceAtomicU32>,
        VO: KernelSliceArg<Elem = V::Atomic>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;

    fn launch_count<K, V, S, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        count: &C,
        empty_key: K,
        erased_key: K,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        S: KernelSliceArg<Elem = u64>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_rehash<K, V, P, const CG: u32, S, D>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        source: &S,
        destination: &D,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        S: KernelSliceArg<Elem = u64>,
        D: KernelSliceArg<Elem = DeviceAtomicU64>;
}

#[doc(hidden)]
pub trait StaticMapRangeKernelModule {
    #[allow(clippy::too_many_arguments)]
    fn launch_insert_range<K, V, P, Input, const CG: u32, S, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        input: Input,
        len: usize,
        inserted: &C,
        invalid: &C,
        range_error: &C,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        Input: crate::range::DeviceMapRead<Item = crate::KeyValue<K, V>>,
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_insert_or_assign_range<K, V, P, Input, const CG: u32, S, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        input: Input,
        len: usize,
        inserted: &C,
        invalid: &C,
        range_error: &C,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        Input: crate::range::DeviceMapRead<Item = crate::KeyValue<K, V>>,
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_insert_prevalidated_range<K, V, P, Input, const CG: u32, S, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        input: Input,
        len: usize,
        range_error: &C,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        Input: crate::range::DeviceMapRead<Item = crate::KeyValue<K, V>>,
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_insert_prevalidated_range_with_erase<K, V, P, Input, const CG: u32, S, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        input: Input,
        len: usize,
        range_error: &C,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        Input: crate::range::DeviceMapRead<Item = crate::KeyValue<K, V>>,
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_find_range<K, V, P, Input, Output, const CG: u32, S, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        input: Input,
        output: Output,
        len: usize,
        range_error: &C,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        Input: crate::range::DeviceMapRead<Item = K>,
        Output: crate::range::DeviceMapWrite<Item = V>,
        S: KernelSliceArg<Elem = u64>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_contains_range<K, V, P, Input, Output, const CG: u32, S, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        input: Input,
        output: Output,
        len: usize,
        range_error: &C,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        Input: crate::range::DeviceMapRead<Item = K>,
        Output: crate::range::DeviceMapWrite<Item = u32>,
        S: KernelSliceArg<Elem = u64>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_contains_if_range<K, V, P, Input, Stencil, Output, Predicate, const CG: u32, S, C>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        input: Input,
        stencil: Stencil,
        output: Output,
        predicate: Predicate,
        len: usize,
        range_error: &C,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), LaunchContractError>
    where
        K: MapKey,
        V: MapValue,
        P: ProbingScheme<K>,
        Input: crate::range::DeviceMapRead<Item = K>,
        Stencil: crate::range::DeviceMapRead,
        Output: crate::range::DeviceMapWrite<Item = u32>,
        Predicate: Fn(Stencil::Item) -> bool + Copy,
        S: KernelSliceArg<Elem = u64>,
        C: KernelSliceArg<Elem = DeviceAtomicU64>;
}

impl<M, K, V, P, const CG: u32> StaticMapBackend<K, V, P, CG> for M
where
    M: StaticMapKernelModule,
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    fn clear<A: DeviceAllocator>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        empty_slot: u64,
    ) -> Result<(), MapError> {
        storage.ensure_usable_on(stream)?;
        let slot_words = slot_storage_words::<V>(capacity)?;
        let mut slots = crate::kernel_args::RangeMutArg::subrange(storage, 0, slot_words);
        let config =
            launch_config(u32::try_from(slot_words).map_err(|_| MapError::CapacityOverflow)?);
        self.launch_clear(stream, config, &mut slots, empty_slot)
            .map_err(MapError::Contract)
    }

    fn insert<A, KR, VR>(
        &self,
        call: InsertCall<'_, '_, K, V, P, A, KR, VR>,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        call.launch_counted::<false, _, CG>(self)
    }

    fn insert_or_assign<A, KR, VR>(
        &self,
        call: InsertCall<'_, '_, K, V, P, A, KR, VR>,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        call.launch_counted::<true, _, CG>(self)
    }

    fn insert_prevalidated<A, KR, VR>(
        &self,
        call: InsertCall<'_, '_, K, V, P, A, KR, VR>,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        call.launch_prevalidated::<false, _, CG>(self)
    }

    fn insert_prevalidated_with_erase<A, KR, VR>(
        &self,
        call: InsertCall<'_, '_, K, V, P, A, KR, VR>,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        VR: ContiguousDeviceRange<V> + ?Sized,
    {
        call.launch_prevalidated::<true, _, CG>(self)
    }

    fn find<A, KR, OR>(
        &self,
        stream: &CudaStream,
        storage: &DeviceBuffer<'_, u64, A>,
        capacity: usize,
        keys: &KR,
        output: &mut OR,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        OR: ContiguousDeviceRangeMut<V> + ?Sized,
    {
        let config = bulk_launch_config::<CG>(keys.len())?;
        let prefix = crate::kernel_args::Prefix::new(storage, slot_storage_words::<V>(capacity)?);
        let slots = crate::kernel_args::RangeArg::new(&prefix);
        let keys = crate::kernel_args::RangeArg::new(keys);
        let output = crate::kernel_args::ValueAtomicRangeMutArg::<V, _>::new(output);
        self.launch_find::<K, V, P, CG, _, _, _>(
            stream,
            config,
            &slots,
            &keys,
            &output,
            empty_key,
            empty_value,
            erased_key,
            probing,
        )
        .map_err(MapError::Contract)
    }

    fn contains<A, KR, OR>(
        &self,
        stream: &CudaStream,
        storage: &DeviceBuffer<'_, u64, A>,
        capacity: usize,
        keys: &KR,
        output: &mut OR,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
        OR: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        let config = bulk_launch_config::<CG>(keys.len())?;
        let prefix = crate::kernel_args::Prefix::new(storage, slot_storage_words::<V>(capacity)?);
        let slots = crate::kernel_args::RangeArg::new(&prefix);
        let keys = crate::kernel_args::RangeArg::new(keys);
        let output = crate::kernel_args::AtomicU32RangeMutArg::new(output);
        self.launch_contains::<K, V, P, CG, _, _, _>(
            stream,
            config,
            &slots,
            &keys,
            &output,
            empty_key,
            empty_value,
            erased_key,
            probing,
        )
        .map_err(MapError::Contract)
    }

    fn erase<A, KR>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        keys: &KR,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KR: ContiguousDeviceRange<K> + ?Sized,
    {
        let config = bulk_launch_config::<CG>(keys.len())?;
        let storage = crate::kernel_args::AtomicU64StorageArg::new(storage);
        let slot_words = slot_storage_words::<V>(capacity)?;
        let slots = storage.subrange(0, slot_words);
        let erased = storage.subrange(slot_words, 1);
        let keys = crate::kernel_args::RangeArg::new(keys);
        self.launch_erase::<K, V, P, CG, _, _, _>(
            stream,
            config,
            &slots,
            &keys,
            &erased,
            empty_key,
            empty_value,
            erased_key,
            probing,
        )
        .map_err(MapError::Contract)
    }

    fn retrieve_all<A, KO, VO>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        keys: &mut KO,
        values: &mut VO,
        empty_key: K,
        erased_key: K,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        KO: ContiguousDeviceRangeMut<K> + ?Sized,
        VO: ContiguousDeviceRangeMut<V> + ?Sized,
    {
        let config =
            launch_config(u32::try_from(capacity).map_err(|_| MapError::CapacityOverflow)?);
        let storage = crate::kernel_args::AtomicU64StorageArg::new(storage);
        let slots = storage.plain_subrange(0, slot_storage_words::<V>(capacity)?);
        let keys = crate::kernel_args::AtomicU32RangeMutArg::new(keys);
        let values = crate::kernel_args::ValueAtomicRangeMutArg::<V, _>::new(values);
        let count = storage.subrange(slot_storage_words::<V>(capacity)?, 1);
        self.launch_retrieve_all::<K, V, _, _, _, _>(
            stream, config, &slots, &keys, &values, &count, empty_key, erased_key,
        )
        .map_err(MapError::Contract)
    }

    fn count<A: DeviceAllocator>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        empty_key: K,
        erased_key: K,
    ) -> Result<(), MapError> {
        let config =
            launch_config(u32::try_from(capacity).map_err(|_| MapError::CapacityOverflow)?);
        let storage = crate::kernel_args::AtomicU64StorageArg::new(storage);
        let slots = storage.plain_subrange(0, slot_storage_words::<V>(capacity)?);
        let count = storage.subrange(slot_storage_words::<V>(capacity)?, 1);
        self.launch_count::<K, V, _, _>(stream, config, &slots, &count, empty_key, erased_key)
            .map_err(MapError::Contract)
    }

    fn rehash<SA: DeviceAllocator, DA: DeviceAllocator>(
        &self,
        stream: &CudaStream,
        source: &DeviceBuffer<'_, u64, SA>,
        source_capacity: usize,
        destination: &mut DeviceBuffer<'_, u64, DA>,
        destination_capacity: usize,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError> {
        let config = bulk_launch_config::<CG>(source_capacity)?;
        let source_prefix =
            crate::kernel_args::Prefix::new(source, slot_storage_words::<V>(source_capacity)?);
        let source = crate::kernel_args::RangeArg::new(&source_prefix);
        let destination = crate::kernel_args::AtomicU64StorageArg::new(destination);
        let destination = destination.subrange(0, slot_storage_words::<V>(destination_capacity)?);
        self.launch_rehash::<K, V, P, CG, _, _>(
            stream,
            config,
            &source,
            &destination,
            empty_key,
            empty_value,
            erased_key,
            probing,
        )
        .map_err(MapError::Contract)
    }
}

impl<M, K, V, P, const CG: u32> StaticMapRangeBackend<K, V, P, CG> for M
where
    M: StaticMapRangeKernelModule,
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
{
    fn insert_range<const ASSIGN: bool, A, SA, Input>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        input: Input,
        len: usize,
        status: &mut DeviceBuffer<'_, u64, SA>,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        SA: DeviceAllocator,
        Input: crate::range::DeviceMapRead<Item = crate::KeyValue<K, V>>,
    {
        let config = bulk_launch_config::<CG>(len)?;
        let storage = crate::kernel_args::AtomicU64StorageArg::new(storage);
        let slot_words = slot_storage_words::<V>(capacity)?;
        let slots = storage.subrange(0, slot_words);
        let inserted = storage.subrange(slot_words, 1);
        let invalid = storage.subrange(slot_words + 1, 1);
        let status = crate::kernel_args::AtomicU64StorageArg::new(status);
        let range_error = status.subrange(0, 1);
        let result = if ASSIGN {
            self.launch_insert_or_assign_range::<K, V, P, Input, CG, _, _>(
                stream,
                config,
                &slots,
                input,
                len,
                &inserted,
                &invalid,
                &range_error,
                empty_key,
                empty_value,
                erased_key,
                probing,
            )
        } else {
            self.launch_insert_range::<K, V, P, Input, CG, _, _>(
                stream,
                config,
                &slots,
                input,
                len,
                &inserted,
                &invalid,
                &range_error,
                empty_key,
                empty_value,
                erased_key,
                probing,
            )
        };
        result.map_err(MapError::Contract)
    }

    fn insert_prevalidated_range<const SUPPORTS_ERASE: bool, A, SA, Input>(
        &self,
        stream: &CudaStream,
        storage: &mut DeviceBuffer<'_, u64, A>,
        capacity: usize,
        input: Input,
        len: usize,
        status: &mut DeviceBuffer<'_, u64, SA>,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        SA: DeviceAllocator,
        Input: crate::range::DeviceMapRead<Item = crate::KeyValue<K, V>>,
    {
        let config = bulk_launch_config::<CG>(len)?;
        let storage = crate::kernel_args::AtomicU64StorageArg::new(storage);
        let slots = storage.subrange(0, slot_storage_words::<V>(capacity)?);
        let status = crate::kernel_args::AtomicU64StorageArg::new(status);
        let range_error = status.subrange(0, 1);
        let result = if SUPPORTS_ERASE {
            self.launch_insert_prevalidated_range_with_erase::<K, V, P, Input, CG, _, _>(
                stream,
                config,
                &slots,
                input,
                len,
                &range_error,
                empty_key,
                empty_value,
                erased_key,
                probing,
            )
        } else {
            self.launch_insert_prevalidated_range::<K, V, P, Input, CG, _, _>(
                stream,
                config,
                &slots,
                input,
                len,
                &range_error,
                empty_key,
                empty_value,
                erased_key,
                probing,
            )
        };
        result.map_err(MapError::Contract)
    }

    fn find_range<A, SA, Input, Output>(
        &self,
        stream: &CudaStream,
        storage: &DeviceBuffer<'_, u64, A>,
        capacity: usize,
        input: Input,
        output: Output,
        len: usize,
        status: &mut DeviceBuffer<'_, u64, SA>,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        SA: DeviceAllocator,
        Input: crate::range::DeviceMapRead<Item = K>,
        Output: crate::range::DeviceMapWrite<Item = V>,
    {
        let config = bulk_launch_config::<CG>(len)?;
        let prefix = crate::kernel_args::Prefix::new(storage, slot_storage_words::<V>(capacity)?);
        let slots = crate::kernel_args::RangeArg::new(&prefix);
        let status = crate::kernel_args::AtomicU64StorageArg::new(status);
        let range_error = status.subrange(0, 1);
        self.launch_find_range::<K, V, P, Input, Output, CG, _, _>(
            stream,
            config,
            &slots,
            input,
            output,
            len,
            &range_error,
            empty_key,
            empty_value,
            erased_key,
            probing,
        )
        .map_err(MapError::Contract)
    }

    fn contains_range<A, SA, Input, Output>(
        &self,
        stream: &CudaStream,
        storage: &DeviceBuffer<'_, u64, A>,
        capacity: usize,
        input: Input,
        output: Output,
        len: usize,
        status: &mut DeviceBuffer<'_, u64, SA>,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        SA: DeviceAllocator,
        Input: crate::range::DeviceMapRead<Item = K>,
        Output: crate::range::DeviceMapWrite<Item = u32>,
    {
        let config = bulk_launch_config::<CG>(len)?;
        let prefix = crate::kernel_args::Prefix::new(storage, slot_storage_words::<V>(capacity)?);
        let slots = crate::kernel_args::RangeArg::new(&prefix);
        let status = crate::kernel_args::AtomicU64StorageArg::new(status);
        let range_error = status.subrange(0, 1);
        self.launch_contains_range::<K, V, P, Input, Output, CG, _, _>(
            stream,
            config,
            &slots,
            input,
            output,
            len,
            &range_error,
            empty_key,
            empty_value,
            erased_key,
            probing,
        )
        .map_err(MapError::Contract)
    }

    fn contains_if_range<A, SA, Input, Stencil, Output, Predicate>(
        &self,
        stream: &CudaStream,
        storage: &DeviceBuffer<'_, u64, A>,
        capacity: usize,
        input: Input,
        stencil: Stencil,
        output: Output,
        predicate: Predicate,
        len: usize,
        status: &mut DeviceBuffer<'_, u64, SA>,
        empty_key: K,
        empty_value: V,
        erased_key: K,
        probing: P,
    ) -> Result<(), MapError>
    where
        A: DeviceAllocator,
        SA: DeviceAllocator,
        Input: crate::range::DeviceMapRead<Item = K>,
        Stencil: crate::range::DeviceMapRead,
        Output: crate::range::DeviceMapWrite<Item = u32>,
        Predicate: Fn(Stencil::Item) -> bool + Copy,
    {
        let config = bulk_launch_config::<CG>(len)?;
        let prefix = crate::kernel_args::Prefix::new(storage, slot_storage_words::<V>(capacity)?);
        let slots = crate::kernel_args::RangeArg::new(&prefix);
        let status = crate::kernel_args::AtomicU64StorageArg::new(status);
        let range_error = status.subrange(0, 1);
        self.launch_contains_if_range::<K, V, P, Input, Stencil, Output, Predicate, CG, _, _>(
            stream,
            config,
            &slots,
            input,
            stencil,
            output,
            predicate,
            len,
            &range_error,
            empty_key,
            empty_value,
            erased_key,
            probing,
        )
        .map_err(MapError::Contract)
    }
}

impl<'borrow, 'stream, K, V, P, A, KR: ?Sized, VR: ?Sized>
    InsertCall<'borrow, 'stream, K, V, P, A, KR, VR>
where
    K: MapKey,
    V: MapValue,
    P: ProbingScheme<K>,
    A: DeviceAllocator,
    KR: ContiguousDeviceRange<K>,
    VR: ContiguousDeviceRange<V>,
{
    fn launch_counted<const ASSIGN: bool, M, const CG: u32>(
        self,
        module: &M,
    ) -> Result<(), MapError>
    where
        M: StaticMapKernelModule,
    {
        let InsertCall {
            stream,
            storage,
            capacity,
            keys,
            values,
            empty_key,
            empty_value,
            erased_key,
            probing,
        } = self;
        let config = bulk_launch_config::<CG>(keys.len())?;
        let storage = crate::kernel_args::AtomicU64StorageArg::new(storage);
        let slot_words = slot_storage_words::<V>(capacity)?;
        let slots = storage.subrange(0, slot_words);
        let inserted = storage.subrange(slot_words, 1);
        let invalid = storage.subrange(slot_words + 1, 1);
        let keys = crate::kernel_args::RangeArg::new(keys);
        let values = crate::kernel_args::RangeArg::new(values);
        let result = if ASSIGN {
            module.launch_insert_or_assign::<K, V, P, CG, _, _, _, _>(
                stream,
                config,
                &slots,
                &keys,
                &values,
                &inserted,
                &invalid,
                empty_key,
                empty_value,
                erased_key,
                probing,
            )
        } else {
            module.launch_insert::<K, V, P, CG, _, _, _, _>(
                stream,
                config,
                &slots,
                &keys,
                &values,
                &inserted,
                &invalid,
                empty_key,
                empty_value,
                erased_key,
                probing,
            )
        };
        result.map_err(MapError::Contract)
    }

    fn launch_prevalidated<const SUPPORTS_ERASE: bool, M, const CG: u32>(
        self,
        module: &M,
    ) -> Result<(), MapError>
    where
        M: StaticMapKernelModule,
    {
        let InsertCall {
            stream,
            storage,
            capacity,
            keys,
            values,
            empty_key,
            empty_value,
            erased_key,
            probing,
        } = self;
        let config = bulk_launch_config::<CG>(keys.len())?;
        let storage = crate::kernel_args::AtomicU64StorageArg::new(storage);
        let slot_words = slot_storage_words::<V>(capacity)?;
        let slots = storage.subrange(0, slot_words);
        let keys = crate::kernel_args::RangeArg::new(keys);
        let values = crate::kernel_args::RangeArg::new(values);
        let result = if SUPPORTS_ERASE {
            module.launch_insert_prevalidated_with_erase::<K, V, P, CG, _, _, _>(
                stream,
                config,
                &slots,
                &keys,
                &values,
                empty_key,
                empty_value,
                erased_key,
                probing,
            )
        } else {
            module.launch_insert_prevalidated::<K, V, P, CG, _, _, _>(
                stream,
                config,
                &slots,
                &keys,
                &values,
                empty_key,
                empty_value,
                erased_key,
                probing,
            )
        };
        result.map_err(MapError::Contract)
    }
}

fn slot_storage_words<V: MapValue>(capacity: usize) -> Result<usize, MapError> {
    capacity
        .checked_mul(V::SLOT_WORDS)
        .ok_or(MapError::CapacityOverflow)
}

fn bulk_launch_config<const CG: u32>(items: usize) -> Result<LaunchConfig1D, MapError> {
    let threads = items
        .checked_mul(CG as usize)
        .ok_or(MapError::CapacityOverflow)?;
    let threads = u32::try_from(threads).map_err(|_| MapError::CapacityOverflow)?;
    Ok(launch_config(threads))
}

#[doc(hidden)]
#[macro_export]
macro_rules! __cuco_backend {
    ($module:ident) => {
        impl $crate::__private::StaticMapKernelModule for $module::LoadedModule {
            fn launch_clear<S>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &mut S,
                empty_slot: u64,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                S: $crate::__private::cuda_host::KernelSliceArgMut<Elem = u64>,
            {
                let prepared = self.prepare_static_map_clear(config)?;
                self.static_map_clear(stream, &prepared, slots, empty_slot)
            }

            fn launch_insert<K, V, P, const CG: u32, S, KR, VR, C>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                keys: &KR,
                values: &VR,
                inserted: &C,
                invalid: &C,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                KR: $crate::__private::cuda_host::KernelSliceArg<Elem = K>,
                VR: $crate::__private::cuda_host::KernelSliceArg<Elem = V>,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared = self.prepare_static_map_insert::<K, V, P, CG>(config)?;
                self.static_map_insert::<K, V, P, CG>(
                    stream,
                    &prepared,
                    slots,
                    keys,
                    values,
                    inserted,
                    invalid,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_insert_or_assign<K, V, P, const CG: u32, S, KR, VR, C>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                keys: &KR,
                values: &VR,
                inserted: &C,
                invalid: &C,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                KR: $crate::__private::cuda_host::KernelSliceArg<Elem = K>,
                VR: $crate::__private::cuda_host::KernelSliceArg<Elem = V>,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared = self.prepare_static_map_insert_or_assign::<K, V, P, CG>(config)?;
                self.static_map_insert_or_assign::<K, V, P, CG>(
                    stream,
                    &prepared,
                    slots,
                    keys,
                    values,
                    inserted,
                    invalid,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_insert_prevalidated<K, V, P, const CG: u32, S, KR, VR>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                keys: &KR,
                values: &VR,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                KR: $crate::__private::cuda_host::KernelSliceArg<Elem = K>,
                VR: $crate::__private::cuda_host::KernelSliceArg<Elem = V>,
            {
                let prepared = self.prepare_static_map_insert_fast::<K, V, P, CG>(config)?;
                self.static_map_insert_fast::<K, V, P, CG>(
                    stream,
                    &prepared,
                    slots,
                    keys,
                    values,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_insert_prevalidated_with_erase<K, V, P, const CG: u32, S, KR, VR>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                keys: &KR,
                values: &VR,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                KR: $crate::__private::cuda_host::KernelSliceArg<Elem = K>,
                VR: $crate::__private::cuda_host::KernelSliceArg<Elem = V>,
            {
                let prepared =
                    self.prepare_static_map_insert_fast_with_erase::<K, V, P, CG>(config)?;
                self.static_map_insert_fast_with_erase::<K, V, P, CG>(
                    stream,
                    &prepared,
                    slots,
                    keys,
                    values,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_find<K, V, P, const CG: u32, S, KR, OR>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                keys: &KR,
                output: &OR,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                S: $crate::__private::cuda_host::KernelSliceArg<Elem = u64>,
                KR: $crate::__private::cuda_host::KernelSliceArg<Elem = K>,
                OR: $crate::__private::cuda_host::KernelSliceArg<Elem = V::Atomic>,
            {
                let prepared = self.prepare_static_map_find::<K, V, P, CG>(config)?;
                self.static_map_find::<K, V, P, CG>(
                    stream,
                    &prepared,
                    slots,
                    keys,
                    output,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_contains<K, V, P, const CG: u32, S, KR, OR>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                keys: &KR,
                output: &OR,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                S: $crate::__private::cuda_host::KernelSliceArg<Elem = u64>,
                KR: $crate::__private::cuda_host::KernelSliceArg<Elem = K>,
                OR: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
            {
                let prepared = self.prepare_static_map_contains::<K, V, P, CG>(config)?;
                self.static_map_contains::<K, V, P, CG>(
                    stream,
                    &prepared,
                    slots,
                    keys,
                    output,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_erase<K, V, P, const CG: u32, S, KR, C>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                keys: &KR,
                erased: &C,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                KR: $crate::__private::cuda_host::KernelSliceArg<Elem = K>,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared = self.prepare_static_map_erase::<K, V, P, CG>(config)?;
                self.static_map_erase::<K, V, P, CG>(
                    stream,
                    &prepared,
                    slots,
                    keys,
                    erased,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_retrieve_all<K, V, S, KO, VO, C>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                keys: &KO,
                values: &VO,
                count: &C,
                empty_key: K,
                erased_key: K,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                S: $crate::__private::cuda_host::KernelSliceArg<Elem = u64>,
                KO: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
                VO: $crate::__private::cuda_host::KernelSliceArg<Elem = V::Atomic>,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared = self.prepare_static_map_retrieve_all::<K, V>(config)?;
                self.static_map_retrieve_all::<K, V>(
                    stream, &prepared, slots, keys, values, count, empty_key, erased_key,
                )
            }

            fn launch_count<K, V, S, C>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                count: &C,
                empty_key: K,
                erased_key: K,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                S: $crate::__private::cuda_host::KernelSliceArg<Elem = u64>,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared = self.prepare_static_map_count::<K, V>(config)?;
                self.static_map_count::<K, V>(
                    stream, &prepared, slots, count, empty_key, erased_key,
                )
            }

            fn launch_rehash<K, V, P, const CG: u32, S, D>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                source: &S,
                destination: &D,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                S: $crate::__private::cuda_host::KernelSliceArg<Elem = u64>,
                D: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared = self.prepare_static_map_rehash::<K, V, P, CG>(config)?;
                self.static_map_rehash::<K, V, P, CG>(
                    stream,
                    &prepared,
                    source,
                    destination,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }
        }

        impl $crate::__private::StaticMapRangeKernelModule for $module::LoadedModule {
            fn launch_insert_range<K, V, P, Input, const CG: u32, S, C>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                input: Input,
                len: usize,
                inserted: &C,
                invalid: &C,
                range_error: &C,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = $crate::KeyValue<K, V>>,
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared =
                    self.prepare_static_map_insert_range::<K, V, P, Input, CG>(config)?;
                self.static_map_insert_range::<K, V, P, Input, CG>(
                    stream,
                    &prepared,
                    slots,
                    input,
                    len,
                    inserted,
                    invalid,
                    range_error,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_insert_or_assign_range<K, V, P, Input, const CG: u32, S, C>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                input: Input,
                len: usize,
                inserted: &C,
                invalid: &C,
                range_error: &C,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = $crate::KeyValue<K, V>>,
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared =
                    self.prepare_static_map_insert_or_assign_range::<K, V, P, Input, CG>(config)?;
                self.static_map_insert_or_assign_range::<K, V, P, Input, CG>(
                    stream,
                    &prepared,
                    slots,
                    input,
                    len,
                    inserted,
                    invalid,
                    range_error,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_insert_prevalidated_range<K, V, P, Input, const CG: u32, S, C>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                input: Input,
                len: usize,
                range_error: &C,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = $crate::KeyValue<K, V>>,
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared = self
                    .prepare_static_map_insert_prevalidated_range::<K, V, P, Input, CG>(config)?;
                self.static_map_insert_prevalidated_range::<K, V, P, Input, CG>(
                    stream,
                    &prepared,
                    slots,
                    input,
                    len,
                    range_error,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_insert_prevalidated_range_with_erase<K, V, P, Input, const CG: u32, S, C>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                input: Input,
                len: usize,
                range_error: &C,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = $crate::KeyValue<K, V>>,
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared = self
                    .prepare_static_map_insert_prevalidated_range_with_erase::<K, V, P, Input, CG>(
                        config,
                    )?;
                self.static_map_insert_prevalidated_range_with_erase::<K, V, P, Input, CG>(
                    stream,
                    &prepared,
                    slots,
                    input,
                    len,
                    range_error,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_find_range<K, V, P, Input, Output, const CG: u32, S, C>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                input: Input,
                output: Output,
                len: usize,
                range_error: &C,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = K>,
                Output: $crate::__private::DeviceMapWrite<Item = V>,
                S: $crate::__private::cuda_host::KernelSliceArg<Elem = u64>,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared =
                    self.prepare_static_map_find_range::<K, V, P, Input, Output, CG>(config)?;
                self.static_map_find_range::<K, V, P, Input, Output, CG>(
                    stream,
                    &prepared,
                    slots,
                    input,
                    output,
                    len,
                    range_error,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_contains_range<K, V, P, Input, Output, const CG: u32, S, C>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                input: Input,
                output: Output,
                len: usize,
                range_error: &C,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = K>,
                Output: $crate::__private::DeviceMapWrite<Item = u32>,
                S: $crate::__private::cuda_host::KernelSliceArg<Elem = u64>,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared =
                    self.prepare_static_map_contains_range::<K, V, P, Input, Output, CG>(config)?;
                self.static_map_contains_range::<K, V, P, Input, Output, CG>(
                    stream,
                    &prepared,
                    slots,
                    input,
                    output,
                    len,
                    range_error,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }

            fn launch_contains_if_range<
                K,
                V,
                P,
                Input,
                Stencil,
                Output,
                Predicate,
                const CG: u32,
                S,
                C,
            >(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                input: Input,
                stencil: Stencil,
                output: Output,
                predicate: Predicate,
                len: usize,
                range_error: &C,
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::MapKey,
                V: $crate::MapValue,
                P: $crate::ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = K>,
                Stencil: $crate::__private::DeviceMapRead,
                Output: $crate::__private::DeviceMapWrite<Item = u32>,
                Predicate: Fn(Stencil::Item) -> bool + Copy,
                S: $crate::__private::cuda_host::KernelSliceArg<Elem = u64>,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
            {
                let prepared = self.prepare_static_map_contains_if_range(config)?;
                self.static_map_contains_if_range::<K, V, P, Input, Stencil, Output, Predicate, CG>(
                    stream,
                    &prepared,
                    slots,
                    input,
                    stencil,
                    output,
                    predicate,
                    len,
                    range_error,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                )
            }
        }
    };
}

#[cfg(test)]
mod tests {
    const DEVICE: &str = include_str!("device.rs");
    const SOURCE: &str = include_str!("backend.rs");

    #[test]
    fn generated_backend_binds_leaf_kernels_without_runtime_policy_branches() {
        let (runtime, generated) = SOURCE
            .split_once("macro_rules! __cuco_backend")
            .expect("backend source must contain its generated-module binding");
        let generated = generated
            .split_once("#[cfg(test)]")
            .map_or(generated, |(macro_source, _)| macro_source);

        assert!(!runtime.contains("assign: bool"));
        assert!(!runtime.contains("supports_erase: bool"));
        assert!(runtime.contains("fn insert_or_assign"));
        assert!(runtime.contains("fn insert_prevalidated_with_erase"));
        assert!(!generated.contains("if assign"));
        assert!(!generated.contains("if supports_erase"));
        assert!(generated.contains("fn launch_insert_or_assign"));
        assert!(generated.contains("fn launch_insert_prevalidated_with_erase"));
        assert!(!DEVICE.contains("if ASSIGN"));
        assert!(!DEVICE.contains("if SUPPORTS_ERASE"));
        assert!(DEVICE.contains("insert_counted::<ASSIGN>"));
        assert!(DEVICE.contains("insert_prevalidated::<SUPPORTS_ERASE>"));
    }
}
