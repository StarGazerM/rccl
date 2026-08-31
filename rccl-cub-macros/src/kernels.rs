use quote::{format_ident, quote};

fn set_kernels_for_index(
    index: &syn::Type,
    transaction: &syn::Type,
    suffix: &str,
) -> proc_macro2::TokenStream {
    let init = format_ident!("device_set_init_{suffix}");
    let partition = format_ident!("device_set_partition_{suffix}");
    let arithmetic_partition = format_ident!("device_set_arithmetic_partition_{suffix}");
    let agents = [256usize, 512usize].into_iter().map(|threads| {
        let keys = format_ident!("device_set_keys_{suffix}_{threads}");
        let keys_virtual = format_ident!("device_set_keys_virtual_{suffix}_{threads}");
        let discard = format_ident!("device_set_discard_{suffix}_{threads}");
        let discard_virtual = format_ident!("device_set_discard_virtual_{suffix}_{threads}");
        let arithmetic = format_ident!("device_set_arithmetic_{suffix}_{threads}");
        let arithmetic_virtual = format_ident!("device_set_arithmetic_virtual_{suffix}_{threads}");
        let pairs = format_ident!("device_set_pairs_{suffix}_{threads}");
        let pairs_virtual = format_ident!("device_set_pairs_virtual_{suffix}_{threads}");
        // `quote` preserves a primitive integer's suffix (`512usize`).  CUDA's
        // launch-contract coordinates are `u32`, while the agent const generic
        // is `usize`, so emit one unsuffixed literal and let each Rust context
        // assign the correct type.
        let threads = proc_macro2::Literal::usize_unsuffixed(threads);
        quote! {
            #[kernel]
            #[launch_bounds(#threads)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (#threads, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn #keys<K, F, Op, const ITEMS: usize, const SHARED_BYTES: usize, const SM70: bool>(
                first: &[K],
                second: &[K],
                partitions: &[#transaction],
                order: F,
                tile_state: DisjointSlice<#transaction>,
                output_count: DisjointSlice<usize>,
                output: DisjointSlice<K>,
            ) where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
                Op: ::rccl_cub::__private::SerialSetOperation,
            {
                let shared = ::rccl_cub::__private::BlockSharedMemory::new(SHARED_BYTES);
                let state = ::rccl_cub::__private::SetAgentState::new(
                    partitions,
                    tile_state,
                    output_count,
                    shared,
                );
                ::rccl_cub::__private::set_keys_agent::<
                    K,
                    #index,
                    F,
                    Op,
                    #threads,
                    ITEMS,
                    SM70,
                >(
                    first,
                    second,
                    order,
                    state,
                    output,
                );
            }

            #[kernel]
            #[launch_bounds(#threads)]
            #[launch_contract(domain = 1, coordinates = u32, block = (#threads, 1, 1))]
            pub fn #keys_virtual<
                K,
                F,
                Op,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
                const SM70: bool,
            >(
                first: &[K],
                second: &[K],
                partitions: &[#transaction],
                order: F,
                tile_state: DisjointSlice<#transaction>,
                output_count: DisjointSlice<usize>,
                output: DisjointSlice<K>,
                virtual_shared: DisjointSlice<u8>,
            ) where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
                Op: ::rccl_cub::__private::SerialSetOperation,
            {
                let shared = ::rccl_cub::__private::BlockSharedMemory::from_virtual_block(
                    virtual_shared,
                    thread::blockIdx_x(),
                    SHARED_BYTES,
                );
                let state = ::rccl_cub::__private::SetAgentState::new(
                    partitions,
                    tile_state,
                    output_count,
                    shared,
                );
                ::rccl_cub::__private::set_keys_agent::<
                    K,
                    #index,
                    F,
                    Op,
                    #threads,
                    ITEMS,
                    SM70,
                >(
                    first,
                    second,
                    order,
                    state,
                    output,
                );
            }

            #[kernel]
            #[launch_bounds(#threads)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (#threads, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn #discard<
                K,
                F,
                Op,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
                const SM70: bool,
            >(
                first: &[K],
                second: &[K],
                partitions: &[#transaction],
                order: F,
                tile_state: DisjointSlice<#transaction>,
                output_count: DisjointSlice<usize>,
            ) where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
                Op: ::rccl_cub::__private::SerialSetOperation,
            {
                let shared = ::rccl_cub::__private::BlockSharedMemory::new(SHARED_BYTES);
                let state = ::rccl_cub::__private::SetAgentState::new(
                    partitions,
                    tile_state,
                    output_count,
                    shared,
                );
                ::rccl_cub::__private::set_keys_discard_agent::<
                    K,
                    #index,
                    F,
                    Op,
                    #threads,
                    ITEMS,
                    SM70,
                >(
                    first,
                    second,
                    order,
                    state,
                );
            }

            #[kernel]
            #[launch_bounds(#threads)]
            #[launch_contract(domain = 1, coordinates = u32, block = (#threads, 1, 1))]
            pub fn #discard_virtual<
                K,
                F,
                Op,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
                const SM70: bool,
            >(
                first: &[K],
                second: &[K],
                partitions: &[#transaction],
                order: F,
                tile_state: DisjointSlice<#transaction>,
                output_count: DisjointSlice<usize>,
                virtual_shared: DisjointSlice<u8>,
            ) where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
                Op: ::rccl_cub::__private::SerialSetOperation,
            {
                let shared = ::rccl_cub::__private::BlockSharedMemory::from_virtual_block(
                    virtual_shared,
                    thread::blockIdx_x(),
                    SHARED_BYTES,
                );
                let state = ::rccl_cub::__private::SetAgentState::new(
                    partitions,
                    tile_state,
                    output_count,
                    shared,
                );
                ::rccl_cub::__private::set_keys_discard_agent::<
                    K,
                    #index,
                    F,
                    Op,
                    #threads,
                    ITEMS,
                    SM70,
                >(
                    first,
                    second,
                    order,
                    state,
                );
            }

            #[kernel]
            #[launch_bounds(#threads)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (#threads, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn #arithmetic<
                K,
                F,
                Op,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
                const SM70: bool,
            >(
                first_start: Uniform<K>,
                first_step: Uniform<K>,
                first_len: Uniform<usize>,
                second_start: Uniform<K>,
                second_step: Uniform<K>,
                second_len: Uniform<usize>,
                partitions: &[#transaction],
                order: F,
                tile_state: DisjointSlice<#transaction>,
                output_count: DisjointSlice<usize>,
                output: DisjointSlice<K>,
            ) where
                K: Copy
                    + ::core::convert::TryFrom<usize>
                    + ::core::ops::Add<Output = K>
                    + ::core::ops::Mul<Output = K>,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
                Op: ::rccl_cub::__private::SerialSetOperation,
            {
                let shared = ::rccl_cub::__private::BlockSharedMemory::new(SHARED_BYTES);
                let first = ::rccl_cub::__private::ArithmeticSetRange::new(
                    first_start.get(), first_step.get(), first_len.get(),
                );
                let second = ::rccl_cub::__private::ArithmeticSetRange::new(
                    second_start.get(), second_step.get(), second_len.get(),
                );
                let state = ::rccl_cub::__private::SetAgentState::new(
                    partitions,
                    tile_state,
                    output_count,
                    shared,
                );
                ::rccl_cub::__private::set_arithmetic_keys_agent::<
                    K, #index, F, Op, #threads, ITEMS, SM70,
                >(
                    first, second, order, state, output,
                );
            }

            #[kernel]
            #[launch_bounds(#threads)]
            #[launch_contract(domain = 1, coordinates = u32, block = (#threads, 1, 1))]
            pub fn #arithmetic_virtual<
                K,
                F,
                Op,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
                const SM70: bool,
            >(
                first_start: Uniform<K>,
                first_step: Uniform<K>,
                first_len: Uniform<usize>,
                second_start: Uniform<K>,
                second_step: Uniform<K>,
                second_len: Uniform<usize>,
                partitions: &[#transaction],
                order: F,
                tile_state: DisjointSlice<#transaction>,
                output_count: DisjointSlice<usize>,
                output: DisjointSlice<K>,
                virtual_shared: DisjointSlice<u8>,
            ) where
                K: Copy
                    + ::core::convert::TryFrom<usize>
                    + ::core::ops::Add<Output = K>
                    + ::core::ops::Mul<Output = K>,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
                Op: ::rccl_cub::__private::SerialSetOperation,
            {
                let shared = ::rccl_cub::__private::BlockSharedMemory::from_virtual_block(
                    virtual_shared, thread::blockIdx_x(), SHARED_BYTES,
                );
                let first = ::rccl_cub::__private::ArithmeticSetRange::new(
                    first_start.get(), first_step.get(), first_len.get(),
                );
                let second = ::rccl_cub::__private::ArithmeticSetRange::new(
                    second_start.get(), second_step.get(), second_len.get(),
                );
                let state = ::rccl_cub::__private::SetAgentState::new(
                    partitions,
                    tile_state,
                    output_count,
                    shared,
                );
                ::rccl_cub::__private::set_arithmetic_keys_agent::<
                    K, #index, F, Op, #threads, ITEMS, SM70,
                >(
                    first, second, order, state, output,
                );
            }

            #[kernel]
            #[launch_bounds(#threads)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (#threads, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn #pairs<
                K,
                V,
                F,
                Op,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
                const SM70: bool,
            >(
                first_keys: &[K],
                second_keys: &[K],
                first_values: &[V],
                second_values: &[V],
                partitions: &[#transaction],
                order: F,
                tile_state: DisjointSlice<#transaction>,
                output_count: DisjointSlice<usize>,
                output_keys: DisjointSlice<K>,
                output_values: DisjointSlice<V>,
            ) where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
                Op: ::rccl_cub::__private::SerialSetOperation,
            {
                let shared = ::rccl_cub::__private::BlockSharedMemory::new(SHARED_BYTES);
                let state = ::rccl_cub::__private::SetAgentState::new(
                    partitions,
                    tile_state,
                    output_count,
                    shared,
                );
                let output = ::rccl_cub::__private::SetPairOutput::new(
                    output_keys,
                    output_values,
                );
                ::rccl_cub::__private::set_pairs_agent::<
                    K,
                    V,
                    #index,
                    F,
                    Op,
                    #threads,
                    ITEMS,
                    SM70,
                >(
                    first_keys,
                    second_keys,
                    first_values,
                    second_values,
                    order,
                    state,
                    output,
                );
            }

            #[kernel]
            #[launch_bounds(#threads)]
            #[launch_contract(domain = 1, coordinates = u32, block = (#threads, 1, 1))]
            pub fn #pairs_virtual<
                K,
                V,
                F,
                Op,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
                const SM70: bool,
            >(
                first_keys: &[K],
                second_keys: &[K],
                first_values: &[V],
                second_values: &[V],
                partitions: &[#transaction],
                order: F,
                tile_state: DisjointSlice<#transaction>,
                output_count: DisjointSlice<usize>,
                output_keys: DisjointSlice<K>,
                output_values: DisjointSlice<V>,
                virtual_shared: DisjointSlice<u8>,
            ) where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
                Op: ::rccl_cub::__private::SerialSetOperation,
            {
                let shared = ::rccl_cub::__private::BlockSharedMemory::from_virtual_block(
                    virtual_shared,
                    thread::blockIdx_x(),
                    SHARED_BYTES,
                );
                let state = ::rccl_cub::__private::SetAgentState::new(
                    partitions,
                    tile_state,
                    output_count,
                    shared,
                );
                let output = ::rccl_cub::__private::SetPairOutput::new(
                    output_keys,
                    output_values,
                );
                ::rccl_cub::__private::set_pairs_agent::<
                    K,
                    V,
                    #index,
                    F,
                    Op,
                    #threads,
                    ITEMS,
                    SM70,
                >(
                    first_keys,
                    second_keys,
                    first_values,
                    second_values,
                    order,
                    state,
                    output,
                );
            }
        }
    });
    quote! {
        #[kernel]
        #[launch_bounds(128)]
        #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
        pub fn #init(tile_state: DisjointSlice<#transaction>, num_tiles: Uniform<usize>) {
            ::rccl_cub::__private::set_init_agent::<#index>(tile_state, num_tiles.get());
        }

        #[kernel(launch_context = launch_context)]
        #[launch_bounds(256)]
        #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
        pub fn #partition<K, F, const ITEMS_PER_TILE: usize>(
            first: &[K],
            second: &[K],
            order: F,
            partitions: DisjointSlice<#transaction>,
        ) where
            K: Copy,
            F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
        {
            ::rccl_cub::__private::set_partition_agent::<
                #index,
                K,
                F,
                _,
                _,
                ITEMS_PER_TILE,
            >(first, second, order, partitions, launch_context);
        }

        #[kernel(launch_context = launch_context)]
        #[launch_bounds(256)]
        #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
        pub fn #arithmetic_partition<K, F, const ITEMS_PER_TILE: usize>(
            first_start: Uniform<K>,
            first_step: Uniform<K>,
            first_len: Uniform<usize>,
            second_start: Uniform<K>,
            second_step: Uniform<K>,
            second_len: Uniform<usize>,
            order: F,
            partitions: DisjointSlice<#transaction>,
        ) where
            K: Copy
                + ::core::convert::TryFrom<usize>
                + ::core::ops::Add<Output = K>
                + ::core::ops::Mul<Output = K>,
            F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
        {
            let first = ::rccl_cub::__private::ArithmeticSetRange::new(
                first_start.get(), first_step.get(), first_len.get(),
            );
            let second = ::rccl_cub::__private::ArithmeticSetRange::new(
                second_start.get(), second_step.get(), second_len.get(),
            );
            ::rccl_cub::__private::set_partition_agent::<
                #index, K, F, _, _, ITEMS_PER_TILE,
            >(first, second, order, partitions, launch_context);
        }

        #(#agents)*
    }
}

fn radix_onesweep_keys_kernel(shared_bytes: usize) -> proc_macro2::TokenStream {
    let kernel = format_ident!("device_radix_sort_onesweep_keys_{shared_bytes}");
    let storage = format_ident!("__RCCL_RADIX_ONESWEEP_KEYS_{shared_bytes}");
    let union_bytes = shared_bytes - 256 * core::mem::size_of::<u64>();
    quote! {
        #[kernel]
        #[launch_bounds(384)]
        #[launch_contract(domain = 1, coordinates = u32, block = (384, 1, 1))]
        pub fn #kernel<
            K,
            B,
            DefaultKey,
            ToOrdered,
            FromOrdered,
            Digit,
            const ITEMS: usize,
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
            let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                #storage,
                #shared_bytes,
                16usize
            );
            ::rccl_cub::__private::radix_onesweep_keys_agent::<
                K,
                B,
                DefaultKey,
                ToOrdered,
                FromOrdered,
                Digit,
                384,
                ITEMS,
                #union_bytes,
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
    }
}

fn radix_onesweep_pairs_kernel(shared_bytes: usize) -> proc_macro2::TokenStream {
    let kernel = format_ident!("device_radix_sort_onesweep_pairs_{shared_bytes}");
    let storage = format_ident!("__RCCL_RADIX_ONESWEEP_PAIRS_{shared_bytes}");
    let union_bytes = shared_bytes - 256 * core::mem::size_of::<u64>();
    quote! {
        #[kernel]
        #[launch_bounds(384)]
        #[launch_contract(domain = 1, coordinates = u32, block = (384, 1, 1))]
        pub fn #kernel<
            K,
            V,
            B,
            DefaultKey,
            ToOrdered,
            FromOrdered,
            Digit,
            const ITEMS: usize,
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
            let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                #storage,
                #shared_bytes,
                16usize
            );
            ::rccl_cub::__private::radix_onesweep_pairs_agent::<
                K,
                V,
                B,
                DefaultKey,
                ToOrdered,
                FromOrdered,
                Digit,
                384,
                ITEMS,
                #union_bytes,
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
    }
}

pub(super) fn expand_device_kernels(module: syn::Ident) -> syn::Result<proc_macro2::TokenStream> {
    let radix_onesweep_keys = [34_304usize, 32_768usize]
        .into_iter()
        .map(radix_onesweep_keys_kernel);
    let radix_onesweep_pairs = [28_160usize, 26_624usize, 32_768usize]
        .into_iter()
        .map(radix_onesweep_pairs_kernel);
    let set_narrow = set_kernels_for_index(&syn::parse_quote!(i32), &syn::parse_quote!(u64), "i32");
    let set_wide = set_kernels_for_index(&syn::parse_quote!(i64), &syn::parse_quote!(u128), "i64");
    Ok(quote! {
        #[doc(hidden)]
        use ::rccl_cub::__private::cuda_core;
        #[doc(hidden)]
        use ::rccl_cub::__private::cuda_host;

        #[::rccl_cub::__private::cuda_device::cuda_module]
        #[doc(hidden)]
        #[expect(
            clippy::too_many_arguments,
            reason = "CUDA entry points preserve the pinned CUB launch ABI as separate uniform parameters"
        )]
        mod #module {
            use ::rccl_cub::__private::cuda_device::{
                DisjointSlice, LinearTiles, Uniform, kernel, launch_bounds, launch_contract, thread,
            };

            #set_narrow
            #set_wide

            #[inline(always)]
            fn ordered_before<K, F>(order: &F, left: &K, right: &K) -> bool
            where
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                order(left, right) == ::core::cmp::Ordering::Less
            }

            /// Return the stable co-rank of `diagonal` in two sorted runs.
            ///
            /// Equality is assigned to the left run first. This is the exact
            /// semantic distinction between CUB's stable merge and a merely
            /// sorted permutation; no source index or pointer escapes.
            #[inline(always)]
            fn merge_path<K, F>(
                first: &[K],
                first_begin: usize,
                first_count: usize,
                second: &[K],
                second_begin: usize,
                second_count: usize,
                diagonal: usize,
                order: &F,
            ) -> usize
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let mut low = if diagonal > second_count {
                    diagonal - second_count
                } else {
                    0
                };
                let mut high = if diagonal < first_count {
                    diagonal
                } else {
                    first_count
                };
                while low < high {
                    let middle = low + (high - low) / 2;
                    let second_taken = diagonal - middle;
                    let partition_is_high_enough = middle == first_count
                        || second_taken == 0
                        || ordered_before(
                            order,
                            &second[second_begin + second_taken - 1],
                            &first[first_begin + middle],
                        );
                    if partition_is_high_enough {
                        high = middle;
                    } else {
                        low = middle + 1;
                    }
                }
                low
            }

            #[inline(always)]
            fn merged_source<K, F>(
                first: &[K],
                first_begin: usize,
                first_count: usize,
                second: &[K],
                second_begin: usize,
                second_count: usize,
                diagonal: usize,
                order: &F,
            ) -> (bool, usize)
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let first_taken = merge_path(
                    first,
                    first_begin,
                    first_count,
                    second,
                    second_begin,
                    second_count,
                    diagonal,
                    order,
                );
                let second_taken = diagonal - first_taken;
                let take_first = first_taken < first_count
                    && (second_taken == second_count
                        || !ordered_before(
                            order,
                            &second[second_begin + second_taken],
                            &first[first_begin + first_taken],
                        ));
                if take_first {
                    (true, first_begin + first_taken)
                } else {
                    (false, second_begin + second_taken)
                }
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, block = (256, 1, 1))]
            pub fn device_partition_merge_path<K, F, const ITEMS: usize>(
                first: &[K],
                second: &[K],
                order: F,
                mut partitions: DisjointSlice<usize>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let diagonal_index = thread::index_1d(launch_context);
                let raw_index = diagonal_index.get();
                if let ::core::option::Option::Some(partition) =
                    partitions.get_mut(diagonal_index)
                {
                    let tile_items = 512usize * ITEMS;
                    let total = first.len() + second.len();
                    let diagonal = (raw_index * tile_items).min(total);
                    *partition = ::rccl_cub::__private::global_merge_path(
                        first, second, diagonal, &order,
                    );
                }
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
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
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_keys_agent::<
                    K,
                    F,
                    512,
                    ITEMS,
                    K_SIZE,
                    USE_LOAD_TO_SHARED,
                    SHARED_BYTES,
                >(
                    first,
                    second,
                    partitions,
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
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
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_pairs_agent::<
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

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_keys_256<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                input: &[K],
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_keys_agent::<K, F, 256, ITEMS, SHARED_BYTES>(
                    input,
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_keys_in_place_256<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_keys_in_place_agent::<K, F, 256, ITEMS, SHARED_BYTES>(
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_keys_512<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                input: &[K],
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_keys_agent::<K, F, 512, ITEMS, SHARED_BYTES>(
                    input,
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_keys_in_place_512<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_keys_in_place_agent::<K, F, 512, ITEMS, SHARED_BYTES>(
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
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
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_pairs_agent::<K, V, F, 256, ITEMS, SHARED_BYTES>(
                    input_keys,
                    input_values,
                    order,
                    output_keys,
                    output_values,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
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
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_pairs_in_place_agent::<
                    K,
                    V,
                    F,
                    256,
                    ITEMS,
                    SHARED_BYTES,
                >(order, output_keys, output_values, launch_context);
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_pairs_512<
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
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_pairs_agent::<K, V, F, 512, ITEMS, SHARED_BYTES>(
                    input_keys,
                    input_values,
                    order,
                    output_keys,
                    output_values,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_pairs_in_place_512<
                K,
                V,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                order: F,
                output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
                output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_pairs_in_place_agent::<
                    K,
                    V,
                    F,
                    512,
                    ITEMS,
                    SHARED_BYTES,
                >(order, output_keys, output_values, launch_context);
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, block = (256, 1, 1))]
            pub fn device_merge_sort_partition<K, F, const THREADS: usize, const ITEMS: usize>(
                source: &[K],
                target_merged_tiles: usize,
                order: F,
                mut partitions: DisjointSlice<usize>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let index = thread::index_1d(launch_context);
                let partition_index = index.get();
                let partition_count = partitions.len();
                if let ::core::option::Option::Some(partition) = partitions.get_mut(index) {
                    *partition = ::rccl_cub::__private::merge_sort_partition_point::<
                        K,
                        F,
                        THREADS,
                        ITEMS,
                    >(
                        source,
                        partition_index,
                        partition_count,
                        target_merged_tiles,
                        &order,
                    );
                }
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_merge_keys_256<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                source: &[K],
                partitions: &[usize],
                target_merged_tiles: usize,
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_merge_keys_agent::<K, F, 256, ITEMS, SHARED_BYTES>(
                    source,
                    partitions,
                    target_merged_tiles,
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_merge_keys_512<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                source: &[K],
                partitions: &[usize],
                target_merged_tiles: usize,
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_merge_keys_agent::<K, F, 512, ITEMS, SHARED_BYTES>(
                    source,
                    partitions,
                    target_merged_tiles,
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
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
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_merge_pairs_agent::<K, V, F, 256, ITEMS, SHARED_BYTES>(
                    (source_keys, source_values),
                    partitions,
                    target_merged_tiles,
                    order,
                    (output_keys, output_values),
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_merge_pairs_512<
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
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_merge_pairs_agent::<K, V, F, 512, ITEMS, SHARED_BYTES>(
                    (source_keys, source_values),
                    partitions,
                    target_merged_tiles,
                    order,
                    (output_keys, output_values),
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn device_radix_sort_init(
                counters: DisjointSlice<i32, LinearTiles<1>>,
                bins: DisjointSlice<u64, LinearTiles<1>>,
            ) {
                let counter_thread = thread::index_1d_u32(launch_context);
                let bin_thread = thread::index_1d_u32(launch_context);
                ::rccl_cub::__private::radix_init_bins_and_counters_agent(
                    counters,
                    bins,
                    counter_thread,
                    bin_thread,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256, 1)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
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
                let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                    __RCCL_RADIX_SINGLE_TILE_KEYS,
                    33_856usize,
                    32usize
                );
                ::rccl_cub::__private::radix_single_tile_keys_agent::<
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

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256, 1)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
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
                let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                    __RCCL_RADIX_SINGLE_TILE_PAIRS,
                    33_856usize,
                    32usize
                );
                ::rccl_cub::__private::radix_single_tile_pairs_agent::<
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
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn device_radix_sort_histogram_4<
                K,
                B,
                DefaultKey,
                ToOrdered,
                Digit,
                const ITEMS: usize,
            >(
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
                let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                    __RCCL_RADIX_HISTOGRAM_4,
                    4_096usize,
                    16usize
                );
                ::rccl_cub::__private::radix_histogram_agent::<
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
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn device_radix_sort_histogram_8<
                K,
                B,
                DefaultKey,
                ToOrdered,
                Digit,
                const ITEMS: usize,
            >(
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
                let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                    __RCCL_RADIX_HISTOGRAM_8,
                    8_192usize,
                    16usize
                );
                ::rccl_cub::__private::radix_histogram_agent::<
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

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn device_radix_sort_exclusive_sum<const PASSES: usize>(
                bins: DisjointSlice<u64>,
            ) {
                let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                    __RCCL_RADIX_EXCLUSIVE_SUM,
                    2_336usize,
                    16usize
                );
                let bin_thread = thread::index_1d_u32(launch_context);
                ::rccl_cub::__private::radix_exclusive_sum_agent::<PASSES>(
                    bins,
                    bin_thread,
                    shared_memory,
                );
            }

            #(#radix_onesweep_keys)*
            #(#radix_onesweep_pairs)*

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn transform_repeat_in_place<T, F>(
                repeated: T,
                op: F,
                mut output: DisjointSlice<T, LinearTiles<1>>,
            )
            where
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
    })
}
