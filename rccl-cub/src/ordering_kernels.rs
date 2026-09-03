//! Library-owned CUDA entries for merge, merge-sort, radix-sort, and transform.

use cuda_device::cuda_module;

#[cuda_module]
#[doc(hidden)]
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA entry points preserve the algorithm launch ABI"
)]
mod device {
    use cuda_device::{
        DisjointSlice, LinearTiles, Uniform, kernel, launch_bounds, launch_contract, thread,
    };

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , block = (256 , 1 , 1))]
    pub fn device_partition_merge_path<K, F, const ITEMS: usize>(
        first: &[K],
        second: &[K],
        order: F,
        mut partitions: DisjointSlice<usize>,
    ) where
        K: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    {
        let diagonal_index = thread::index_1d(launch_context);
        let raw_index = diagonal_index.get();
        if let ::core::option::Option::Some(partition) = partitions.get_mut(diagonal_index) {
            let tile_items = 512usize * ITEMS;
            let total = first.len() + second.len();
            let diagonal = (raw_index * tile_items).min(total);
            *partition = crate::__private::global_merge_path(first, second, diagonal, &order);
        }
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(512)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (512 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_merge_keys_agent<
        K,
        F,
        const ITEMS: usize,
        const K_SIZE: usize,
        const USE_LOAD_TO_SHARED: bool,
        const SHARED_BYTES: usize,
    >(
        first: &[K],
        second: &[K],
        partitions: &[usize],
        order: F,
        output: DisjointSlice<K, LinearTiles<ITEMS>>,
    ) where
        K: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    {
        crate::__private::merge_keys_agent::<
            K,
            F,
            512,
            ITEMS,
            K_SIZE,
            USE_LOAD_TO_SHARED,
            SHARED_BYTES,
        >(first, second, partitions, order, output, launch_context);
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(512)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (512 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_merge_pairs_agent<
        K,
        V,
        F,
        const ITEMS: usize,
        const K_SIZE: usize,
        const V_SIZE: usize,
        const USE_LOAD_TO_SHARED_KEYS: bool,
        const USE_LOAD_TO_SHARED_VALUES: bool,
        const SHARED_BYTES: usize,
    >(
        first_keys: &[K],
        first_values: &[V],
        second_keys: &[K],
        second_values: &[V],
        partitions: &[usize],
        order: F,
        output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
        output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
    ) where
        K: Copy,
        V: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    {
        crate::__private::merge_pairs_agent::<
            K,
            V,
            F,
            512,
            ITEMS,
            K_SIZE,
            V_SIZE,
            USE_LOAD_TO_SHARED_KEYS,
            USE_LOAD_TO_SHARED_VALUES,
            SHARED_BYTES,
        >(
            (first_keys, first_values),
            (second_keys, second_values),
            partitions,
            order,
            (output_keys, output_values),
            launch_context,
        );
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_merge_sort_block_keys_256<K, F, const ITEMS: usize, const SHARED_BYTES: usize>(
        input: &[K],
        order: F,
        output: DisjointSlice<K, LinearTiles<ITEMS>>,
    ) where
        K: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    {
        crate::__private::merge_sort_block_keys_agent::<K, F, 256, ITEMS, SHARED_BYTES>(
            input,
            order,
            output,
            launch_context,
        );
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_merge_sort_block_keys_in_place_256<
        K,
        F,
        const ITEMS: usize,
        const SHARED_BYTES: usize,
    >(
        order: F,
        output: DisjointSlice<K, LinearTiles<ITEMS>>,
    ) where
        K: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    {
        crate::__private::merge_sort_block_keys_in_place_agent::<K, F, 256, ITEMS, SHARED_BYTES>(
            order,
            output,
            launch_context,
        );
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_merge_sort_block_pairs_256<
        K,
        V,
        F,
        const ITEMS: usize,
        const SHARED_BYTES: usize,
    >(
        input_keys: &[K],
        input_values: &[V],
        order: F,
        output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
        output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
    ) where
        K: Copy,
        V: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    {
        crate::__private::merge_sort_block_pairs_agent::<K, V, F, 256, ITEMS, SHARED_BYTES>(
            input_keys,
            input_values,
            order,
            output_keys,
            output_values,
            launch_context,
        );
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_merge_sort_block_pairs_in_place_256<
        K,
        V,
        F,
        const ITEMS: usize,
        const SHARED_BYTES: usize,
    >(
        order: F,
        output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
        output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
    ) where
        K: Copy,
        V: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    {
        crate::__private::merge_sort_block_pairs_in_place_agent::<K, V, F, 256, ITEMS, SHARED_BYTES>(
            order,
            output_keys,
            output_values,
            launch_context,
        );
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , block = (256 , 1 , 1))]
    pub fn device_merge_sort_partition<K, F, const THREADS: usize, const ITEMS: usize>(
        source: &[K],
        target_merged_tiles: usize,
        order: F,
        mut partitions: DisjointSlice<usize>,
    ) where
        K: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    {
        let index = thread::index_1d(launch_context);
        let partition_index = index.get();
        let partition_count = partitions.len();
        if let ::core::option::Option::Some(partition) = partitions.get_mut(index) {
            *partition = crate::__private::merge_sort_partition_point::<K, F, THREADS, ITEMS>(
                source,
                partition_index,
                partition_count,
                target_merged_tiles,
                &order,
            );
        }
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_merge_sort_merge_keys_256<K, F, const ITEMS: usize, const SHARED_BYTES: usize>(
        source: &[K],
        partitions: &[usize],
        target_merged_tiles: usize,
        order: F,
        output: DisjointSlice<K, LinearTiles<ITEMS>>,
    ) where
        K: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    {
        crate::__private::merge_sort_merge_keys_agent::<K, F, 256, ITEMS, SHARED_BYTES>(
            source,
            partitions,
            target_merged_tiles,
            order,
            output,
            launch_context,
        );
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1) , dynamic_shared_range = (0 , 49152) , dynamic_shared_alignment = 16 ,)]
    pub fn device_merge_sort_merge_pairs_256<
        K,
        V,
        F,
        const ITEMS: usize,
        const SHARED_BYTES: usize,
    >(
        source_keys: &[K],
        source_values: &[V],
        partitions: &[usize],
        target_merged_tiles: usize,
        order: F,
        output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
        output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
    ) where
        K: Copy,
        V: Copy,
        F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
    {
        crate::__private::merge_sort_merge_pairs_agent::<K, V, F, 256, ITEMS, SHARED_BYTES>(
            (source_keys, source_values),
            partitions,
            target_merged_tiles,
            order,
            (output_keys, output_values),
            launch_context,
        );
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1))]
    pub fn device_radix_sort_init(
        counters: DisjointSlice<i32, LinearTiles<1>>,
        bins: DisjointSlice<u64, LinearTiles<1>>,
    ) {
        let counter_thread = thread::index_1d_u32(launch_context);
        let bin_thread = thread::index_1d_u32(launch_context);
        crate::__private::radix_init_bins_and_counters_agent(
            counters,
            bins,
            counter_thread,
            bin_thread,
        );
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256, 1)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1))]
    pub fn device_radix_sort_single_tile_keys<
        K,
        B,
        DefaultOrdered,
        ToOrdered,
        FromOrdered,
        RankDigit,
        const ITEMS: usize,
        const DESCENDING: bool,
    >(
        input: &[K],
        output: DisjointSlice<K, LinearTiles<ITEMS>>,
        begin_bit: Uniform<u32>,
        end_bit: Uniform<u32>,
        default_ordered: DefaultOrdered,
        to_ordered: ToOrdered,
        from_ordered: FromOrdered,
        rank_digit: RankDigit,
    ) where
        K: Copy,
        B: Copy,
        DefaultOrdered: Fn() -> B + Copy,
        ToOrdered: Fn(K) -> B + Copy,
        FromOrdered: Fn(B) -> K + Copy,
        RankDigit: Fn(B, u32, u32) -> u32 + Copy,
    {
        let shared_memory = crate::__private::static_block_shared_memory!(
            __RCCL_RADIX_SINGLE_TILE_KEYS,
            33_856usize,
            32usize
        );
        crate::__private::radix_single_tile_keys_agent::<
            K,
            B,
            DefaultOrdered,
            ToOrdered,
            FromOrdered,
            RankDigit,
            256,
            ITEMS,
            DESCENDING,
        >(
            input,
            output,
            begin_bit.get()..end_bit.get(),
            (default_ordered, to_ordered, from_ordered, rank_digit),
            launch_context,
            shared_memory,
        );
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256, 1)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1))]
    pub fn device_radix_sort_single_tile_pairs<
        K,
        V,
        B,
        DefaultOrdered,
        ToOrdered,
        FromOrdered,
        RankDigit,
        const ITEMS: usize,
        const DESCENDING: bool,
    >(
        input_keys: &[K],
        input_values: &[V],
        output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
        output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
        begin_bit: Uniform<u32>,
        end_bit: Uniform<u32>,
        default_ordered: DefaultOrdered,
        to_ordered: ToOrdered,
        from_ordered: FromOrdered,
        rank_digit: RankDigit,
    ) where
        K: Copy,
        V: Copy,
        B: Copy,
        DefaultOrdered: Fn() -> B + Copy,
        ToOrdered: Fn(K) -> B + Copy,
        FromOrdered: Fn(B) -> K + Copy,
        RankDigit: Fn(B, u32, u32) -> u32 + Copy,
    {
        let shared_memory = crate::__private::static_block_shared_memory!(
            __RCCL_RADIX_SINGLE_TILE_PAIRS,
            33_856usize,
            32usize
        );
        crate::__private::radix_single_tile_pairs_agent::<
            K,
            V,
            B,
            DefaultOrdered,
            ToOrdered,
            FromOrdered,
            RankDigit,
            256,
            ITEMS,
            DESCENDING,
        >(
            (input_keys, input_values),
            (output_keys, output_values),
            begin_bit.get()..end_bit.get(),
            (default_ordered, to_ordered, from_ordered, rank_digit),
            launch_context,
            shared_memory,
        );
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (128 , 1 , 1))]
    pub fn device_radix_sort_histogram_4<K, B, DefaultKey, ToOrdered, Digit, const ITEMS: usize>(
        keys: &[K],
        begin_bit: u32,
        end_bit: u32,
        default_key: DefaultKey,
        to_ordered: ToOrdered,
        digit: Digit,
        bins: DisjointSlice<u64>,
    ) where
        K: Copy,
        B: Copy,
        DefaultKey: Fn() -> B + Copy,
        ToOrdered: Fn(K) -> B + Copy,
        Digit: Fn(B, u32, u32) -> u32 + Copy,
    {
        let shared_memory = crate::__private::static_block_shared_memory!(
            __RCCL_RADIX_HISTOGRAM_4,
            4_096usize,
            16usize
        );
        crate::__private::radix_histogram_agent::<
            K,
            B,
            DefaultKey,
            ToOrdered,
            Digit,
            128,
            ITEMS,
            4usize,
        >(
            keys,
            begin_bit..end_bit,
            default_key,
            to_ordered,
            digit,
            bins,
            shared_memory,
        );
    }

    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (128 , 1 , 1))]
    pub fn device_radix_sort_histogram_8<K, B, DefaultKey, ToOrdered, Digit, const ITEMS: usize>(
        keys: &[K],
        begin_bit: u32,
        end_bit: u32,
        default_key: DefaultKey,
        to_ordered: ToOrdered,
        digit: Digit,
        bins: DisjointSlice<u64>,
    ) where
        K: Copy,
        B: Copy,
        DefaultKey: Fn() -> B + Copy,
        ToOrdered: Fn(K) -> B + Copy,
        Digit: Fn(B, u32, u32) -> u32 + Copy,
    {
        let shared_memory = crate::__private::static_block_shared_memory!(
            __RCCL_RADIX_HISTOGRAM_8,
            8_192usize,
            16usize
        );
        crate::__private::radix_histogram_agent::<
            K,
            B,
            DefaultKey,
            ToOrdered,
            Digit,
            128,
            ITEMS,
            8usize,
        >(
            keys,
            begin_bit..end_bit,
            default_key,
            to_ordered,
            digit,
            bins,
            shared_memory,
        );
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1))]
    pub fn device_radix_sort_exclusive_sum<const PASSES: usize>(bins: DisjointSlice<u64>) {
        let shared_memory = crate::__private::static_block_shared_memory!(
            __RCCL_RADIX_EXCLUSIVE_SUM,
            2_336usize,
            16usize
        );
        let bin_thread = thread::index_1d_u32(launch_context);
        crate::__private::radix_exclusive_sum_agent::<PASSES>(bins, bin_thread, shared_memory);
    }

    #[kernel]
    #[launch_bounds(384)]
    #[launch_contract(domain = 1, coordinates = u32, block = (384, 1, 1))]
    pub fn device_radix_sort_onesweep_keys<
        K,
        B,
        DefaultKey,
        ToOrdered,
        FromOrdered,
        Digit,
        const ITEMS: usize,
        const UNION_BYTES: usize,
        const SHARED_BYTES: usize,
    >(
        lookback: DisjointSlice<i32>,
        counter: DisjointSlice<i32>,
        bins: &[u64],
        input: &[K],
        output: DisjointSlice<K>,
        bin_offset: Uniform<usize>,
        counter_offset: Uniform<usize>,
        current_bit: Uniform<u32>,
        num_bits: Uniform<u32>,
        default_key: DefaultKey,
        to_ordered: ToOrdered,
        from_ordered: FromOrdered,
        digit: Digit,
    ) where
        K: Copy,
        B: Copy,
        DefaultKey: Fn() -> B + Copy,
        ToOrdered: Fn(K) -> B + Copy,
        FromOrdered: Fn(B) -> K + Copy,
        Digit: Fn(B, u32, u32) -> u32 + Copy,
    {
        let shared_memory = crate::__private::static_block_shared_memory_from_size!(
            SHARED_BYTES;
            32768, 34304,
        );
        crate::__private::radix_onesweep_keys_agent::<
            K,
            B,
            DefaultKey,
            ToOrdered,
            FromOrdered,
            Digit,
            384,
            ITEMS,
            UNION_BYTES,
        >(
            (lookback, counter, bins),
            (input, output),
            (
                bin_offset.get(),
                counter_offset.get(),
                current_bit.get(),
                num_bits.get(),
            ),
            (default_key, to_ordered, from_ordered, digit),
            shared_memory,
        );
    }

    #[kernel]
    #[launch_bounds(384)]
    #[launch_contract(domain = 1, coordinates = u32, block = (384, 1, 1))]
    pub fn device_radix_sort_onesweep_pairs<
        K,
        V,
        B,
        DefaultKey,
        ToOrdered,
        FromOrdered,
        Digit,
        const ITEMS: usize,
        const UNION_BYTES: usize,
        const SHARED_BYTES: usize,
    >(
        lookback: DisjointSlice<i32>,
        counter: DisjointSlice<i32>,
        bins: &[u64],
        input_keys: &[K],
        input_values: &[V],
        output_keys: DisjointSlice<K>,
        output_values: DisjointSlice<V>,
        bin_offset: Uniform<usize>,
        counter_offset: Uniform<usize>,
        current_bit: Uniform<u32>,
        num_bits: Uniform<u32>,
        default_key: DefaultKey,
        to_ordered: ToOrdered,
        from_ordered: FromOrdered,
        digit: Digit,
    ) where
        K: Copy,
        V: Copy,
        B: Copy,
        DefaultKey: Fn() -> B + Copy,
        ToOrdered: Fn(K) -> B + Copy,
        FromOrdered: Fn(B) -> K + Copy,
        Digit: Fn(B, u32, u32) -> u32 + Copy,
    {
        let shared_memory = crate::__private::static_block_shared_memory_from_size!(
            SHARED_BYTES;
            26624, 28160, 32768,
        );
        crate::__private::radix_onesweep_pairs_agent::<
            K,
            V,
            B,
            DefaultKey,
            ToOrdered,
            FromOrdered,
            Digit,
            384,
            ITEMS,
            UNION_BYTES,
        >(
            (lookback, counter, bins),
            (input_keys, input_values, output_keys, output_values),
            (
                bin_offset.get(),
                counter_offset.get(),
                current_bit.get(),
                num_bits.get(),
            ),
            (default_key, to_ordered, from_ordered, digit),
            shared_memory,
        );
    }

    #[kernel (launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract (domain = 1 , coordinates = u32 , block = (256 , 1 , 1))]
    pub fn transform_repeat_in_place<T, F>(
        repeated: T,
        op: F,
        mut output: DisjointSlice<T, LinearTiles<1>>,
    ) where
        T: Copy,
        F: Fn(T, T) -> T + Copy,
    {
        let index = thread::index_1d_u32(launch_context);
        let mut runs = output.grid_stride_runs32(index);
        while let Some(mut run) = runs.next_run() {
            if let Some(mut slot) = run.at(0) {
                let value = slot.read();
                slot.write(op(value, repeated));
            }
        }
    }
}

#[inline(always)]
fn radix_single_default<K: crate::__private::RadixKey, const DESC: bool>() -> K::Bits {
    if DESC {
        K::minimum_ordered()
    } else {
        K::default_ordered()
    }
}

#[inline(always)]
fn radix_to_ordered<K: crate::__private::RadixKey, const DESC: bool>(key: K) -> K::Bits {
    if DESC {
        K::to_ordered_descending(key)
    } else {
        K::to_ordered_ascending(key)
    }
}

#[inline(always)]
fn radix_from_ordered<K: crate::__private::RadixKey, const DESC: bool>(bits: K::Bits) -> K {
    if DESC {
        K::from_ordered_descending(bits)
    } else {
        K::from_ordered_ascending(bits)
    }
}

// Rust traits require the two named histogram methods, while cuda-oxide emits
// distinct 4-pass and 8-pass launchers. This local macro only spells that
// duplicated method body; it does not select architecture, types, or policy.
macro_rules! radix_histogram_stage {
    ($method:ident, $prepare:ident, $kernel:ident, $passes:literal, $ignored:ident) => {
        fn $method<Keys, Bins, const ITEMS: usize>(
            &self,
            stream: &crate::__private::cuda_core::CudaStream,
            keys: &Keys,
            bins: &mut Bins,
            begin_bit: u32,
            end_bit: u32,
        ) -> Result<(), crate::DeviceAlgorithmError>
        where
            Keys: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
            Bins: crate::__private::cuda_host::KernelSliceArgMut<Elem = u64>,
        {
            let key_witness = K::zero();
            let bit_witness = K::default_ordered();
            let default_key = K::default_ordered;
            let to_ordered = radix_to_ordered::<K, DESC>;
            let digit = K::digit;
            let probe_launch = crate::__private::cuda_core::LaunchConfig1D::new(1, 128, 0);
            let probe = self
                .$prepare::<K, K::Bits, _, _, _, ITEMS>(
                    &key_witness,
                    &bit_witness,
                    &default_key,
                    &to_ordered,
                    &digit,
                    probe_launch,
                )
                .map_err(crate::DeviceAlgorithmError::Launch)?;
            let blocks_per_sm = probe
                .function()
                .max_active_blocks_per_multiprocessor(128, 0)
                .map_err(crate::DeviceAlgorithmError::Driver)?;
            let multiprocessors = stream
                .context()
                .multiprocessor_count()
                .map_err(crate::DeviceAlgorithmError::Driver)?;
            let launch = crate::__private::cuda_core::LaunchConfig1D::new(
                blocks_per_sm * multiprocessors,
                128,
                0,
            );
            let prepared = self
                .$prepare::<K, K::Bits, _, _, _, ITEMS>(
                    &key_witness,
                    &bit_witness,
                    &default_key,
                    &to_ordered,
                    &digit,
                    launch,
                )
                .map_err(crate::DeviceAlgorithmError::Launch)?;
            self.$kernel::<K, K::Bits, _, _, _, ITEMS>(
                stream,
                &prepared,
                keys,
                begin_bit,
                end_bit,
                default_key,
                to_ordered,
                digit,
                bins,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)
        }
    };
}

impl<K, F> crate::__private::MergeKeysKernel<K, F> for OrderingModule
where
    K: crate::__private::cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
{
    fn launch_merge_keys<
        First,
        Second,
        Partitions,
        Output,
        const ITEMS: usize,
        const K_SIZE: usize,
        const LOAD: bool,
        const SHARED: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        first: &First,
        second: &Second,
        partitions: &mut Partitions,
        output: &mut Output,
        order: F,
        num_tiles: usize,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        First: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        Second: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        Partitions: crate::__private::cuda_host::KernelSliceArgMut<Elem = usize>,
        Output: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
    {
        let partition_launch = crate::__private::merge_partition_launch(num_tiles);
        let partition = self
            .prepare_device_partition_merge_path::<K, F, ITEMS>(partition_launch)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_partition_merge_path::<K, F, ITEMS>(
            stream,
            &partition,
            first,
            second,
            order,
            &mut *partitions,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)?;

        let agent_launch = crate::__private::merge_agent_launch(num_tiles, SHARED);
        let agent = self
            .prepare_device_merge_keys_agent::<K, F, ITEMS, K_SIZE, LOAD, SHARED>(agent_launch)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_merge_keys_agent::<K, F, ITEMS, K_SIZE, LOAD, SHARED>(
            stream,
            &agent,
            first,
            second,
            &*partitions,
            order,
            output,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }
}

impl<K, V, F> crate::__private::MergePairsKernel<K, V, F> for OrderingModule
where
    K: crate::__private::cuda_core::DeviceCopy,
    V: crate::__private::cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
{
    fn launch_merge_pairs<
        FirstKeys,
        FirstValues,
        SecondKeys,
        SecondValues,
        Partitions,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
        const K_SIZE: usize,
        const V_SIZE: usize,
        const LOAD_KEYS: bool,
        const LOAD_VALUES: bool,
        const SHARED: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        first_keys: &FirstKeys,
        first_values: &FirstValues,
        second_keys: &SecondKeys,
        second_values: &SecondValues,
        partitions: &mut Partitions,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
        order: F,
        num_tiles: usize,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        FirstKeys: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        FirstValues: crate::__private::cuda_host::KernelSliceArg<Elem = V>,
        SecondKeys: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        SecondValues: crate::__private::cuda_host::KernelSliceArg<Elem = V>,
        Partitions: crate::__private::cuda_host::KernelSliceArgMut<Elem = usize>,
        OutputKeys: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
        OutputValues: crate::__private::cuda_host::KernelSliceArgMut<Elem = V>,
    {
        let partition_launch = crate::__private::merge_partition_launch(num_tiles);
        let partition = self
            .prepare_device_partition_merge_path::<K, F, ITEMS>(partition_launch)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_partition_merge_path::<K, F, ITEMS>(
            stream,
            &partition,
            first_keys,
            second_keys,
            order,
            &mut *partitions,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)?;

        let agent_launch = crate::__private::merge_agent_launch(num_tiles, SHARED);
        let agent = self
                                            .prepare_device_merge_pairs_agent::<
                                                K,
                                                V,
                                                F,
                                                ITEMS,
                                                K_SIZE,
                                                V_SIZE,
                                                LOAD_KEYS,
                                                LOAD_VALUES,
                                                SHARED,
                                            >(agent_launch)
                                            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_merge_pairs_agent::<
                                            K,
                                            V,
                                            F,
                                            ITEMS,
                                            K_SIZE,
                                            V_SIZE,
                                            LOAD_KEYS,
                                            LOAD_VALUES,
                                            SHARED,
                                        >(
                                            stream,
                                            &agent,
                                            first_keys,
                                            first_values,
                                            second_keys,
                                            second_values,
                                            &*partitions,
                                            order,
                                            output_keys,
                                            output_values,
                                        )
                                        .map_err(crate::DeviceAlgorithmError::Launch)
    }
}

impl<K, F> crate::__private::MergeSortKeysKernel<K, F> for OrderingModule
where
    K: crate::__private::cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
{
    fn launch_merge_sort_block_keys<Input, Output, const ITEMS: usize, const SHARED: usize>(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        config: crate::__private::cuda_core::LaunchConfig1D,
        input: &Input,
        order: F,
        output: &mut Output,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        Input: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        Output: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
    {
        let prepared = self
            .prepare_device_merge_sort_block_keys_256::<K, F, ITEMS, SHARED>(config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_merge_sort_block_keys_256::<K, F, ITEMS, SHARED>(
            stream, &prepared, input, order, output,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }

    fn launch_merge_sort_block_keys_in_place<Keys, const ITEMS: usize, const SHARED: usize>(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        config: crate::__private::cuda_core::LaunchConfig1D,
        order: F,
        keys: &mut Keys,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        Keys: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
    {
        let prepared = self
            .prepare_device_merge_sort_block_keys_in_place_256::<K, F, ITEMS, SHARED>(config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_merge_sort_block_keys_in_place_256::<K, F, ITEMS, SHARED>(
            stream, &prepared, order, keys,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }

    fn launch_merge_sort_partition<Input, Partitions, const ITEMS: usize>(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        config: crate::__private::cuda_core::LaunchConfig1D,
        input: &Input,
        target_merged_tiles: usize,
        order: F,
        partitions: &mut Partitions,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        Input: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        Partitions: crate::__private::cuda_host::KernelSliceArgMut<Elem = usize>,
    {
        let prepared = self
            .prepare_device_merge_sort_partition::<K, F, 256, ITEMS>(config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_merge_sort_partition::<K, F, 256, ITEMS>(
            stream,
            &prepared,
            input,
            target_merged_tiles,
            order,
            partitions,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }

    fn launch_merge_sort_merge_keys<
        Input,
        Partitions,
        Output,
        const ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        config: crate::__private::cuda_core::LaunchConfig1D,
        input: &Input,
        partitions: &Partitions,
        target_merged_tiles: usize,
        order: F,
        output: &mut Output,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        Input: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        Partitions: crate::__private::cuda_host::KernelSliceArg<Elem = usize>,
        Output: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
    {
        let prepared = self
            .prepare_device_merge_sort_merge_keys_256::<K, F, ITEMS, SHARED>(config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_merge_sort_merge_keys_256::<K, F, ITEMS, SHARED>(
            stream,
            &prepared,
            input,
            partitions,
            target_merged_tiles,
            order,
            output,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }
}

impl<K, V, F> crate::__private::MergeSortPairsKernel<K, V, F> for OrderingModule
where
    K: crate::__private::cuda_core::DeviceCopy,
    V: crate::__private::cuda_core::DeviceCopy,
    F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
{
    fn launch_merge_sort_block_pairs<
        InputKeys,
        InputValues,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        config: crate::__private::cuda_core::LaunchConfig1D,
        input_keys: &InputKeys,
        input_values: &InputValues,
        order: F,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        InputKeys: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        InputValues: crate::__private::cuda_host::KernelSliceArg<Elem = V>,
        OutputKeys: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
        OutputValues: crate::__private::cuda_host::KernelSliceArgMut<Elem = V>,
    {
        let prepared = self
            .prepare_device_merge_sort_block_pairs_256::<K, V, F, ITEMS, SHARED>(config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_merge_sort_block_pairs_256::<K, V, F, ITEMS, SHARED>(
            stream,
            &prepared,
            input_keys,
            input_values,
            order,
            output_keys,
            output_values,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }

    fn launch_merge_sort_block_pairs_in_place<
        Keys,
        Values,
        const ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        config: crate::__private::cuda_core::LaunchConfig1D,
        order: F,
        keys: &mut Keys,
        values: &mut Values,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        Keys: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
        Values: crate::__private::cuda_host::KernelSliceArgMut<Elem = V>,
    {
        let prepared = self
            .prepare_device_merge_sort_block_pairs_in_place_256::<K, V, F, ITEMS, SHARED>(config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_merge_sort_block_pairs_in_place_256::<K, V, F, ITEMS, SHARED>(
            stream, &prepared, order, keys, values,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }

    fn launch_merge_sort_merge_pairs<
        InputKeys,
        InputValues,
        Partitions,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
        const SHARED: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        config: crate::__private::cuda_core::LaunchConfig1D,
        input_keys: &InputKeys,
        input_values: &InputValues,
        partitions: &Partitions,
        target_merged_tiles: usize,
        order: F,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        InputKeys: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        InputValues: crate::__private::cuda_host::KernelSliceArg<Elem = V>,
        Partitions: crate::__private::cuda_host::KernelSliceArg<Elem = usize>,
        OutputKeys: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
        OutputValues: crate::__private::cuda_host::KernelSliceArgMut<Elem = V>,
    {
        let prepared = self
            .prepare_device_merge_sort_merge_pairs_256::<K, V, F, ITEMS, SHARED>(config)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_merge_sort_merge_pairs_256::<K, V, F, ITEMS, SHARED>(
            stream,
            &prepared,
            input_keys,
            input_values,
            partitions,
            target_merged_tiles,
            order,
            output_keys,
            output_values,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }
}

impl<K, const DESC: bool> crate::__private::RadixKernel<K, DESC> for OrderingModule
where
    K: crate::__private::RadixKey,
{
    fn launch_radix_init<C, B>(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        counters: &mut C,
        bins: &mut B,
        passes: usize,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        C: crate::__private::cuda_host::KernelSliceArgMut<Elem = i32>,
        B: crate::__private::cuda_host::KernelSliceArgMut<Elem = u64>,
    {
        let launch = crate::__private::cuda_core::LaunchConfig1D::new(passes as u32, 256, 0);
        let prepared = self
            .prepare_device_radix_sort_init(launch)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_radix_sort_init(stream, &prepared, counters, bins)
            .map_err(crate::DeviceAlgorithmError::Launch)
    }

    radix_histogram_stage!(
        launch_radix_histogram_4,
        prepare_device_radix_sort_histogram_4_for,
        device_radix_sort_histogram_4,
        4,
        ignored
    );
    radix_histogram_stage!(
        launch_radix_histogram_8,
        prepare_device_radix_sort_histogram_8_for,
        device_radix_sort_histogram_8,
        8,
        ignored
    );

    fn launch_radix_exclusive_sum<Bins, const PASSES: usize>(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        bins: &mut Bins,
        passes: usize,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        Bins: crate::__private::cuda_host::KernelSliceArgMut<Elem = u64>,
    {
        let launch = crate::__private::cuda_core::LaunchConfig1D::new(passes as u32, 256, 0);
        let prepared = self
            .prepare_device_radix_sort_exclusive_sum::<PASSES>(launch)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_radix_sort_exclusive_sum::<PASSES>(stream, &prepared, bins)
            .map_err(crate::DeviceAlgorithmError::Launch)
    }
}

impl<K, const DESC: bool> crate::__private::RadixKeysKernel<K, DESC> for OrderingModule
where
    K: crate::__private::RadixKey,
{
    fn launch_radix_single_keys<Input, Output, const ITEMS: usize>(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        input: &Input,
        output: &mut Output,
        begin_bit: u32,
        end_bit: u32,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        Input: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        Output: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
    {
        let key_witness = <K as crate::__private::RadixKey>::zero();
        let bit_witness = <K as crate::__private::RadixKey>::default_ordered();
        let default_ordered = radix_single_default::<K, DESC>;
        let to_ordered = <K as crate::__private::RadixKey>::to_ordered_ascending;
        let from_ordered = <K as crate::__private::RadixKey>::from_ordered_ascending;
        let rank_digit = <K as crate::__private::RadixKey>::rank_digit;
        let launch = crate::__private::cuda_core::LaunchConfig1D::new(1, 256, 0);
        let prepared = self
            .prepare_device_radix_sort_single_tile_keys_for::<
                K,
                <K as crate::__private::RadixKey>::Bits,
                _,
                _,
                _,
                _,
                ITEMS,
                DESC,
            >(
                &key_witness,
                &bit_witness,
                &default_ordered,
                &to_ordered,
                &from_ordered,
                &rank_digit,
                launch,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_radix_sort_single_tile_keys::<
            K,
            <K as crate::__private::RadixKey>::Bits,
            _,
            _,
            _,
            _,
            ITEMS,
            DESC,
        >(
            stream,
            &prepared,
            input,
            output,
            begin_bit,
            end_bit,
            default_ordered,
            to_ordered,
            from_ordered,
            rank_digit,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }

    fn launch_radix_onesweep_keys<
        Lookback,
        Counters,
        Bins,
        Input,
        Output,
        const ITEMS: usize,
        const UNION_BYTES: usize,
        const SHARED_BYTES: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        lookback: &mut Lookback,
        counters: &mut Counters,
        bins: &Bins,
        input: &Input,
        output: &mut Output,
        bin_offset: usize,
        pass: usize,
        current_bit: u32,
        num_bits: u32,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        Lookback: crate::__private::cuda_host::KernelSliceArgMut<Elem = i32>,
        Counters: crate::__private::cuda_host::KernelSliceArgMut<Elem = i32>,
        Bins: crate::__private::cuda_host::KernelSliceArg<Elem = u64>,
        Input: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        Output: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
    {
        let key_witness = <K as crate::__private::RadixKey>::zero();
        let bit_witness = <K as crate::__private::RadixKey>::default_ordered();
        let default_key = <K as crate::__private::RadixKey>::default_ordered;
        let to_ordered = radix_to_ordered::<K, DESC>;
        let from_ordered = radix_from_ordered::<K, DESC>;
        let digit = <K as crate::__private::RadixKey>::digit;
        let blocks = input.len().div_ceil(384 * ITEMS);
        let launch = crate::__private::cuda_core::LaunchConfig1D::new(blocks as u32, 384, 0);
        debug_assert_eq!(UNION_BYTES + 2_048, SHARED_BYTES);
        let prepared = self
            .prepare_device_radix_sort_onesweep_keys_for::<
                K,
                <K as crate::__private::RadixKey>::Bits,
                _,
                _,
                _,
                _,
                ITEMS,
                UNION_BYTES,
                SHARED_BYTES,
            >(
                &key_witness,
                &bit_witness,
                &default_key,
                &to_ordered,
                &from_ordered,
                &digit,
                launch,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_radix_sort_onesweep_keys::<
            K,
            <K as crate::__private::RadixKey>::Bits,
            _,
            _,
            _,
            _,
            ITEMS,
            UNION_BYTES,
            SHARED_BYTES,
        >(
            stream,
            &prepared,
            lookback,
            counters,
            bins,
            input,
            output,
            bin_offset,
            pass,
            current_bit,
            num_bits,
            default_key,
            to_ordered,
            from_ordered,
            digit,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }
}

impl<K, V, const DESC: bool> crate::__private::RadixPairsKernel<K, V, DESC> for OrderingModule
where
    K: crate::__private::RadixKey,
    V: crate::__private::RadixValue,
{
    fn launch_radix_single_pairs<
        InputKeys,
        InputValues,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        input_keys: &InputKeys,
        input_values: &InputValues,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
        begin_bit: u32,
        end_bit: u32,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        InputKeys: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        InputValues: crate::__private::cuda_host::KernelSliceArg<Elem = V>,
        OutputKeys: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
        OutputValues: crate::__private::cuda_host::KernelSliceArgMut<Elem = V>,
    {
        let key_witness = <K as crate::__private::RadixKey>::zero();
        let value_witness = V::default();
        let bit_witness = <K as crate::__private::RadixKey>::default_ordered();
        let default_ordered = radix_single_default::<K, DESC>;
        let to_ordered = <K as crate::__private::RadixKey>::to_ordered_ascending;
        let from_ordered = <K as crate::__private::RadixKey>::from_ordered_ascending;
        let rank_digit = <K as crate::__private::RadixKey>::rank_digit;
        let launch = crate::__private::cuda_core::LaunchConfig1D::new(1, 256, 0);
        let prepared = self
            .prepare_device_radix_sort_single_tile_pairs_for::<
                K,
                V,
                <K as crate::__private::RadixKey>::Bits,
                _,
                _,
                _,
                _,
                ITEMS,
                DESC,
            >(
                &key_witness,
                &value_witness,
                &bit_witness,
                &default_ordered,
                &to_ordered,
                &from_ordered,
                &rank_digit,
                launch,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_radix_sort_single_tile_pairs::<
            K,
            V,
            <K as crate::__private::RadixKey>::Bits,
            _,
            _,
            _,
            _,
            ITEMS,
            DESC,
        >(
            stream,
            &prepared,
            input_keys,
            input_values,
            output_keys,
            output_values,
            begin_bit,
            end_bit,
            default_ordered,
            to_ordered,
            from_ordered,
            rank_digit,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }

    fn launch_radix_onesweep_pairs<
        Lookback,
        Counters,
        Bins,
        InputKeys,
        InputValues,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
        const UNION_BYTES: usize,
        const SHARED_BYTES: usize,
    >(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        lookback: &mut Lookback,
        counters: &mut Counters,
        bins: &Bins,
        input_keys: &InputKeys,
        input_values: &InputValues,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
        bin_offset: usize,
        pass: usize,
        current_bit: u32,
        num_bits: u32,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        Lookback: crate::__private::cuda_host::KernelSliceArgMut<Elem = i32>,
        Counters: crate::__private::cuda_host::KernelSliceArgMut<Elem = i32>,
        Bins: crate::__private::cuda_host::KernelSliceArg<Elem = u64>,
        InputKeys: crate::__private::cuda_host::KernelSliceArg<Elem = K>,
        InputValues: crate::__private::cuda_host::KernelSliceArg<Elem = V>,
        OutputKeys: crate::__private::cuda_host::KernelSliceArgMut<Elem = K>,
        OutputValues: crate::__private::cuda_host::KernelSliceArgMut<Elem = V>,
    {
        let key_witness = <K as crate::__private::RadixKey>::zero();
        let value_witness = V::default();
        let bit_witness = <K as crate::__private::RadixKey>::default_ordered();
        let default_key = <K as crate::__private::RadixKey>::default_ordered;
        let to_ordered = radix_to_ordered::<K, DESC>;
        let from_ordered = radix_from_ordered::<K, DESC>;
        let digit = <K as crate::__private::RadixKey>::digit;
        let blocks = input_keys.len().div_ceil(384 * ITEMS);
        let launch = crate::__private::cuda_core::LaunchConfig1D::new(blocks as u32, 384, 0);
        debug_assert_eq!(UNION_BYTES + 2_048, SHARED_BYTES);
        let prepared = self
            .prepare_device_radix_sort_onesweep_pairs_for::<
                K,
                V,
                <K as crate::__private::RadixKey>::Bits,
                _,
                _,
                _,
                _,
                ITEMS,
                UNION_BYTES,
                SHARED_BYTES,
            >(
                &key_witness,
                &value_witness,
                &bit_witness,
                &default_key,
                &to_ordered,
                &from_ordered,
                &digit,
                launch,
            )
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_radix_sort_onesweep_pairs::<
            K,
            V,
            <K as crate::__private::RadixKey>::Bits,
            _,
            _,
            _,
            _,
            ITEMS,
            UNION_BYTES,
            SHARED_BYTES,
        >(
            stream,
            &prepared,
            lookback,
            counters,
            bins,
            input_keys,
            input_values,
            output_keys,
            output_values,
            bin_offset,
            pass,
            current_bit,
            num_bits,
            default_key,
            to_ordered,
            from_ordered,
            digit,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }
}

#[doc(hidden)]
pub type OrderingModule = device::LoadedModule;

#[doc(hidden)]
pub fn load(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> Result<OrderingModule, cuda_host::EmbeddedModuleError> {
    unsafe { device::load(context) }
}
