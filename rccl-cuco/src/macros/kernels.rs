/// Materialize the fixed CUDA entry-point family in the consuming crate.
///
/// This declarative macro performs only item construction required by
/// cuda-oxide. Every algorithm body is an ordinary function in `device.rs`.
#[macro_export]
#[doc(hidden)]
macro_rules! __cuco_device_kernels {
    ($module:ident) => {
        #[doc(hidden)]
        use ::cuda_core;
        #[doc(hidden)]
        use ::cuda_host;

        #[::cuda_device::cuda_module]
        #[doc(hidden)]
        #[allow(clippy::too_many_arguments)]
        mod $module {
            use ::cuda_device::atomic::{DeviceAtomicU32, DeviceAtomicU64};
            use ::cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
            use $crate::{MapKey, MapValue, ProbingScheme, Row, StableHashBytes};

            #[kernel]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn bloom_clear(words: DisjointSlice<u32>) {
                $crate::__private::device_bloom_clear(thread::index_1d(), words);
            }

            #[kernel]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn bloom_add<K: StableHashBytes>(
                words: &[DeviceAtomicU32],
                keys: &[K],
                num_blocks: usize,
                seed: u64,
            ) {
                $crate::__private::device_bloom_add(
                    thread::index_1d().get(),
                    words,
                    keys,
                    num_blocks,
                    seed,
                );
            }

            #[kernel]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn bloom_contains<K: StableHashBytes>(
                words: &[u32],
                keys: &[K],
                output: DisjointSlice<u8>,
                num_blocks: usize,
                seed: u64,
            ) {
                $crate::__private::device_bloom_contains(
                    thread::index_1d(),
                    words,
                    keys,
                    output,
                    num_blocks,
                    seed,
                );
            }

            #[kernel]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn bloom_add_fault<K: StableHashBytes>(
                words: &[DeviceAtomicU32],
                keys: &[K],
                num_blocks: usize,
                seed: u64,
                omitted_word: u32,
            ) {
                $crate::__private::device_bloom_add_fault(
                    thread::index_1d().get(),
                    words,
                    keys,
                    num_blocks,
                    seed,
                    omitted_word,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn exact_set_clear(states: DisjointSlice<u32>) {
                $crate::__private::device_exact_set_clear(thread::index_1d(), states);
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn exact_set_clear_direct(slots: DisjointSlice<u64>, empty_slot: u64) {
                $crate::__private::device_exact_set_clear_direct(
                    thread::index_1d(),
                    slots,
                    empty_slot,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn exact_set_insert<const N: usize>(
                states: &[DeviceAtomicU32],
                hashes: &[DeviceAtomicU64],
                words: &[DeviceAtomicU32],
                metadata: &[DeviceAtomicU64],
                rows: &[Row<N>],
                empty_row: Row<N>,
                seed: u64,
                force_hash: u32,
                forced_hash: u64,
            ) {
                $crate::__private::device_exact_set_insert::<N, true, true>(
                    thread::index_1d().get(),
                    states,
                    hashes,
                    words,
                    metadata,
                    rows,
                    empty_row,
                    seed,
                    force_hash,
                    forced_hash,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn exact_set_insert_direct<const N: usize>(
                slots: &[DeviceAtomicU64],
                metadata: &[DeviceAtomicU64],
                rows: &[Row<N>],
                empty_slot: u64,
                seed: u64,
                force_hash: u32,
                forced_hash: u64,
            ) {
                $crate::__private::device_exact_set_insert_direct::<N, true, true>(
                    thread::index_1d().get(),
                    slots,
                    metadata,
                    rows,
                    empty_slot,
                    seed,
                    force_hash,
                    forced_hash,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn exact_set_insert_prevalidated<const N: usize>(
                states: &[DeviceAtomicU32],
                hashes: &[DeviceAtomicU64],
                words: &[DeviceAtomicU32],
                rows: &[Row<N>],
                seed: u64,
            ) {
                $crate::__private::device_exact_set_insert::<N, false, false>(
                    thread::index_1d().get(),
                    states,
                    hashes,
                    words,
                    hashes,
                    rows,
                    Row::new([0; N]),
                    seed,
                    0,
                    0,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn exact_set_insert_prevalidated_direct<const N: usize>(
                slots: &[DeviceAtomicU64],
                rows: &[Row<N>],
                empty_slot: u64,
                seed: u64,
            ) {
                $crate::__private::device_exact_set_insert_direct::<N, false, false>(
                    thread::index_1d().get(),
                    slots,
                    slots,
                    rows,
                    empty_slot,
                    seed,
                    0,
                    0,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn exact_set_contains<const N: usize>(
                states: &[DeviceAtomicU32],
                hashes: &[DeviceAtomicU64],
                words: &[DeviceAtomicU32],
                rows: &[Row<N>],
                output: &[DeviceAtomicU32],
                seed: u64,
                force_hash: u32,
                forced_hash: u64,
            ) {
                $crate::__private::device_exact_set_contains(
                    thread::index_1d().get(),
                    states,
                    hashes,
                    words,
                    rows,
                    output,
                    seed,
                    force_hash,
                    forced_hash,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn exact_set_contains_direct<const N: usize>(
                slots: &[DeviceAtomicU64],
                rows: &[Row<N>],
                output: &[DeviceAtomicU32],
                empty_slot: u64,
                seed: u64,
                force_hash: u32,
                forced_hash: u64,
            ) {
                $crate::__private::device_exact_set_contains_direct(
                    thread::index_1d().get(),
                    slots,
                    rows,
                    output,
                    empty_slot,
                    seed,
                    force_hash,
                    forced_hash,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn static_map_clear(slots: DisjointSlice<u64>, empty_slot: u64) {
                $crate::__private::device_clear(thread::index_1d(), slots, empty_slot);
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn static_map_insert<K, V, P, const CG: u32>(
                slots: &[DeviceAtomicU64],
                keys: &[K],
                values: &[V],
                inserted: &[DeviceAtomicU64],
                invalid: &[DeviceAtomicU64],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
            {
                $crate::__private::device_insert::<K, V, P, CG, false>(
                    thread::index_1d().get(),
                    slots,
                    keys,
                    values,
                    inserted,
                    invalid,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn static_map_insert_or_assign<K, V, P, const CG: u32>(
                slots: &[DeviceAtomicU64],
                keys: &[K],
                values: &[V],
                inserted: &[DeviceAtomicU64],
                invalid: &[DeviceAtomicU64],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
            {
                $crate::__private::device_insert::<K, V, P, CG, true>(
                    thread::index_1d().get(),
                    slots,
                    keys,
                    values,
                    inserted,
                    invalid,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn static_map_insert_fast<K, V, P, const CG: u32>(
                slots: &[DeviceAtomicU64],
                keys: &[K],
                values: &[V],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
            {
                $crate::__private::device_insert_prevalidated::<K, V, P, CG, false>(
                    thread::index_1d().get(),
                    slots,
                    keys,
                    values,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn static_map_insert_fast_with_erase<K, V, P, const CG: u32>(
                slots: &[DeviceAtomicU64],
                keys: &[K],
                values: &[V],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
            {
                $crate::__private::device_insert_prevalidated::<K, V, P, CG, true>(
                    thread::index_1d().get(),
                    slots,
                    keys,
                    values,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn static_map_insert_range<K, V, P, Input, const CG: u32>(
                slots: &[DeviceAtomicU64],
                input: Input,
                len: usize,
                inserted: &[DeviceAtomicU64],
                invalid: &[DeviceAtomicU64],
                range_error: &[DeviceAtomicU64],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = $crate::KeyValue<K, V>>,
            {
                $crate::__private::device_insert_range::<K, V, P, Input, CG, false>(
                    thread::index_1d().get(),
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
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn static_map_insert_or_assign_range<K, V, P, Input, const CG: u32>(
                slots: &[DeviceAtomicU64],
                input: Input,
                len: usize,
                inserted: &[DeviceAtomicU64],
                invalid: &[DeviceAtomicU64],
                range_error: &[DeviceAtomicU64],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = $crate::KeyValue<K, V>>,
            {
                $crate::__private::device_insert_range::<K, V, P, Input, CG, true>(
                    thread::index_1d().get(),
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
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn static_map_insert_prevalidated_range<K, V, P, Input, const CG: u32>(
                slots: &[DeviceAtomicU64],
                input: Input,
                len: usize,
                range_error: &[DeviceAtomicU64],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = $crate::KeyValue<K, V>>,
            {
                $crate::__private::device_insert_range_prevalidated::<K, V, P, Input, CG, false>(
                    thread::index_1d().get(),
                    slots,
                    input,
                    len,
                    range_error,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn static_map_insert_prevalidated_range_with_erase<K, V, P, Input, const CG: u32>(
                slots: &[DeviceAtomicU64],
                input: Input,
                len: usize,
                range_error: &[DeviceAtomicU64],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = $crate::KeyValue<K, V>>,
            {
                $crate::__private::device_insert_range_prevalidated::<K, V, P, Input, CG, true>(
                    thread::index_1d().get(),
                    slots,
                    input,
                    len,
                    range_error,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn static_map_find_range<K, V, P, Input, Output, const CG: u32>(
                slots: &[u64],
                input: Input,
                output: Output,
                len: usize,
                range_error: &[DeviceAtomicU64],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = K>,
                Output: $crate::__private::DeviceMapWrite<Item = V>,
            {
                $crate::__private::device_find_range::<K, V, P, Input, Output, CG>(
                    thread::index_1d().get(),
                    slots,
                    input,
                    output,
                    len,
                    range_error,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn static_map_contains_range<K, V, P, Input, Output, const CG: u32>(
                slots: &[u64],
                input: Input,
                output: Output,
                len: usize,
                range_error: &[DeviceAtomicU64],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = K>,
                Output: $crate::__private::DeviceMapWrite<Item = u32>,
            {
                $crate::__private::device_contains_range::<K, V, P, Input, Output, CG>(
                    thread::index_1d().get(),
                    slots,
                    input,
                    output,
                    len,
                    range_error,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            #[allow(clippy::too_many_arguments)]
            pub fn static_map_contains_if_range<
                K,
                V,
                P,
                Input,
                Stencil,
                Output,
                Predicate,
                const CG: u32,
            >(
                slots: &[u64],
                input: Input,
                stencil: Stencil,
                output: Output,
                predicate: Predicate,
                len: usize,
                range_error: &[DeviceAtomicU64],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
                Input: $crate::__private::DeviceMapRead<Item = K>,
                Stencil: $crate::__private::DeviceMapRead,
                Output: $crate::__private::DeviceMapWrite<Item = u32>,
                Predicate: Fn(Stencil::Item) -> bool + Copy,
            {
                $crate::__private::device_contains_if_range::<
                    K,
                    V,
                    P,
                    Input,
                    Stencil,
                    Output,
                    Predicate,
                    CG,
                >(
                    thread::index_1d().get(),
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
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn static_map_find<K, V, P, const CG: u32>(
                slots: &[u64],
                keys: &[K],
                output: &[V::Atomic],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
            {
                $crate::__private::device_find::<K, V, P, CG>(
                    thread::index_1d().get(),
                    slots,
                    keys,
                    output,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn static_map_contains<K, V, P, const CG: u32>(
                slots: &[u64],
                keys: &[K],
                output: &[DeviceAtomicU32],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
            {
                $crate::__private::device_contains::<K, V, P, CG>(
                    thread::index_1d().get(),
                    slots,
                    keys,
                    output,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn static_map_erase<K, V, P, const CG: u32>(
                slots: &[DeviceAtomicU64],
                keys: &[K],
                erased: &[DeviceAtomicU64],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
            {
                $crate::__private::device_erase::<K, V, P, CG>(
                    thread::index_1d().get(),
                    slots,
                    keys,
                    erased,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn static_map_retrieve_all<K, V>(
                slots: &[u64],
                keys: &[DeviceAtomicU32],
                values: &[V::Atomic],
                count: &[DeviceAtomicU64],
                empty_key: K,
                erased_key: K,
            ) where
                K: MapKey,
                V: MapValue,
            {
                $crate::__private::device_retrieve_all::<K, V>(
                    thread::index_1d().get(),
                    slots,
                    keys,
                    values,
                    count,
                    empty_key,
                    erased_key,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn static_map_count<K, V>(
                slots: &[u64],
                count: &[DeviceAtomicU64],
                empty_key: K,
                erased_key: K,
            ) where
                K: MapKey,
                V: MapValue,
            {
                $crate::__private::device_count::<K, V>(
                    thread::index_1d().get(),
                    slots,
                    count,
                    empty_key,
                    erased_key,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn static_map_rehash<K, V, P, const CG: u32>(
                source: &[u64],
                destination: &[DeviceAtomicU64],
                empty_key: K,
                empty_value: V,
                erased_key: K,
                probing: P,
            ) where
                K: MapKey,
                V: MapValue,
                P: ProbingScheme<K>,
            {
                $crate::__private::device_rehash::<K, V, P, CG>(
                    thread::index_1d().get(),
                    source,
                    destination,
                    empty_key,
                    empty_value,
                    erased_key,
                    probing,
                );
            }
        }
    };
}
