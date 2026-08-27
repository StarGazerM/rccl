mod audit;
mod call;
#[cfg(test)]
mod catalog;

use proc_macro::TokenStream;
use quote::{format_ident, quote};

#[proc_macro]
pub fn warp_reduce(input: TokenStream) -> TokenStream {
    let call = syn::parse_macro_input!(input as call::WarpReduceCall);
    expand_warp_reduce(call)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro]
pub fn warp_reduce_items(input: TokenStream) -> TokenStream {
    let call = syn::parse_macro_input!(input as call::WarpReduceItemsCall);
    expand_warp_reduce_items(call)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Emit the CUB device-wide kernel family into an ordinary cuda-oxide module.
///
/// The module is generated at the consuming program's item boundary. Every
/// fact needed by TMP lowering must be explicit macro input; later rustc type
/// inference is only used to check the emitted Rust. RCCL emits no PTX, symbol
/// table, or erased launch ABI of its own.
#[proc_macro]
pub fn cub_kernels(input: TokenStream) -> TokenStream {
    let module = if input.is_empty() {
        syn::parse_quote!(__rccl_cub_kernels)
    } else {
        syn::parse_macro_input!(input as syn::Ident)
    };
    expand_device_kernels(module)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Internal item lowering used by the Thrust macro layer.
#[doc(hidden)]
#[proc_macro]
pub fn __device_kernels(input: TokenStream) -> TokenStream {
    let module = syn::parse_macro_input!(input as syn::Ident);
    expand_device_kernels(module)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Internal operation lowering used by the Thrust macro layer.
#[doc(hidden)]
#[proc_macro]
pub fn __device_transform_repeat_in_place(input: TokenStream) -> TokenStream {
    let call = syn::parse_macro_input!(input as call::DeviceTransformRepeatInPlaceCall);
    expand_device_transform_repeat_in_place(call)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

macro_rules! operation_macro {
    ($macro_name:ident, $call:ty, $expand:ident) => {
        #[proc_macro]
        pub fn $macro_name(input: TokenStream) -> TokenStream {
            let call = syn::parse_macro_input!(input as $call);
            $expand(call)
                .unwrap_or_else(syn::Error::into_compile_error)
                .into()
        }
    };
}

operation_macro!(
    device_merge_keys,
    call::DeviceMergeKeysCall,
    expand_device_merge_keys
);
operation_macro!(
    device_merge_keys_partition_count,
    call::DeviceMergeKeysPartitionCountCall,
    expand_device_merge_keys_partition_count
);
operation_macro!(
    device_merge_pairs_partition_count,
    call::DeviceMergePairsPartitionCountCall,
    expand_device_merge_pairs_partition_count
);
operation_macro!(
    device_merge_pairs,
    call::DeviceMergePairsCall,
    expand_device_merge_pairs
);
operation_macro!(
    device_merge_sort_keys,
    call::DeviceMergeSortKeysCall,
    expand_device_merge_sort_keys
);
operation_macro!(
    device_merge_sort_partition_count,
    call::DeviceMergeSortPartitionCountCall,
    expand_device_merge_sort_partition_count
);
operation_macro!(
    device_stable_merge_sort_keys,
    call::DeviceMergeSortKeysCall,
    expand_device_merge_sort_keys
);
operation_macro!(
    device_merge_sort_keys_copy,
    call::DeviceMergeSortKeysCopyCall,
    expand_device_merge_sort_keys_copy
);
operation_macro!(
    device_stable_merge_sort_keys_copy,
    call::DeviceMergeSortKeysCopyCall,
    expand_device_merge_sort_keys_copy
);
operation_macro!(
    device_merge_sort_pairs,
    call::DeviceMergeSortPairsCall,
    expand_device_merge_sort_pairs
);
operation_macro!(
    device_stable_merge_sort_pairs,
    call::DeviceMergeSortPairsCall,
    expand_device_merge_sort_pairs
);
operation_macro!(
    device_merge_sort_pairs_copy,
    call::DeviceMergeSortPairsCopyCall,
    expand_device_merge_sort_pairs_copy
);
operation_macro!(
    device_radix_sort_keys,
    call::DeviceRadixSortKeysCall,
    expand_device_radix_sort_keys
);
operation_macro!(
    device_radix_sort_keys_buffer,
    call::DeviceRadixSortKeysBufferCall,
    expand_device_radix_sort_keys_buffer
);
operation_macro!(
    device_radix_sort_keys_buffer_descending,
    call::DeviceRadixSortKeysBufferCall,
    expand_device_radix_sort_keys_buffer_descending
);
operation_macro!(
    device_radix_sort_workspace,
    call::DeviceRadixSortWorkspaceCall,
    expand_device_radix_sort_workspace
);
operation_macro!(
    device_radix_sort_pairs_workspace,
    call::DeviceRadixSortPairsWorkspaceCall,
    expand_device_radix_sort_pairs_workspace
);
operation_macro!(
    device_radix_sort_keys_descending,
    call::DeviceRadixSortKeysCall,
    expand_device_radix_sort_keys_descending
);
operation_macro!(
    device_radix_sort_pairs,
    call::DeviceRadixSortPairsCall,
    expand_device_radix_sort_pairs
);
operation_macro!(
    device_radix_sort_pairs_buffer,
    call::DeviceRadixSortPairsBufferCall,
    expand_device_radix_sort_pairs_buffer
);
operation_macro!(
    device_radix_sort_pairs_buffer_descending,
    call::DeviceRadixSortPairsBufferCall,
    expand_device_radix_sort_pairs_buffer_descending
);
operation_macro!(
    device_radix_sort_pairs_descending,
    call::DeviceRadixSortPairsCall,
    expand_device_radix_sort_pairs_descending
);

#[derive(Clone, Copy)]
enum PrimitiveRadixKind {
    Unsigned,
    Signed,
    Floating,
}

#[derive(Clone, Copy)]
struct PrimitiveRadix {
    bits: u32,
    kind: PrimitiveRadixKind,
}

struct PrimitiveRadixLowering {
    bit_type: syn::Type,
    sentinel: proc_macro2::TokenStream,
    default_key: proc_macro2::TokenStream,
    to_ordered: proc_macro2::TokenStream,
    from_ordered: proc_macro2::TokenStream,
    digit: proc_macro2::TokenStream,
    rank_digit: proc_macro2::TokenStream,
}

/// Pinned CUB `detail::RegBoundScaling` evaluated by the proc macro.
///
/// The result is TMP policy state in C++; it must not be reconstructed by the
/// running Rust program or selected from a runtime device property.
fn reg_bound_policy(
    nominal_threads: usize,
    nominal_items: usize,
    type_size: usize,
) -> (usize, usize) {
    let items = (nominal_items * 4 / type_size.max(4)).max(1);
    let static_shared_words = (48 * 1024) / (type_size * items);
    let shared_limited_threads = static_shared_words.div_ceil(32) * 32;
    (nominal_threads.min(shared_limited_threads), items)
}

fn primitive_radix_lowering(
    key_type: &syn::Type,
    primitive: PrimitiveRadix,
    descending: bool,
) -> syn::Result<PrimitiveRadixLowering> {
    match (primitive.bits, primitive.kind) {
        (32, PrimitiveRadixKind::Unsigned) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_u32_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_u32_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_u32_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_u32_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u32),
                sentinel: quote!(u32::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u32),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_digit_u32),
                rank_digit: quote!(::rccl_cub::__private::radix_rank_digit_u32),
            })
        }
        (32, PrimitiveRadixKind::Signed) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_i32_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_i32_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_i32_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_i32_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u32),
                sentinel: quote!(u32::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u32),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_digit_u32),
                rank_digit: quote!(::rccl_cub::__private::radix_rank_digit_u32),
            })
        }
        (32, PrimitiveRadixKind::Floating) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_f32_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_f32_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_f32_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_f32_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u32),
                sentinel: quote!(u32::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u32),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_float_digit_u32),
                rank_digit: quote!(::rccl_cub::__private::radix_float_rank_digit_u32),
            })
        }
        (64, PrimitiveRadixKind::Unsigned) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_u64_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_u64_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_u64_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_u64_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u64),
                sentinel: quote!(u64::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u64),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_digit_u64),
                rank_digit: quote!(::rccl_cub::__private::radix_rank_digit_u64),
            })
        }
        (64, PrimitiveRadixKind::Signed) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_i64_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_i64_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_i64_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_i64_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u64),
                sentinel: quote!(u64::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u64),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_digit_u64),
                rank_digit: quote!(::rccl_cub::__private::radix_rank_digit_u64),
            })
        }
        (64, PrimitiveRadixKind::Floating) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_f64_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_f64_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_f64_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_f64_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u64),
                sentinel: quote!(u64::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u64),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_float_digit_u64),
                rank_digit: quote!(::rccl_cub::__private::radix_float_rank_digit_u64),
            })
        }
        _ => Err(syn::Error::new_spanned(
            key_type,
            "the exact pinned SM80 one-sweep path requires a 32-bit or 64-bit primitive key; smaller keys use CUB's different multi-pass policy",
        )),
    }
}

/// Re-express CUB's second digit extraction as a distinct Rust operation.
///
fn primitive_radix(key_type: &syn::Type) -> syn::Result<PrimitiveRadix> {
    let syn::Type::Path(path) = key_type else {
        return Err(syn::Error::new_spanned(
            key_type,
            "DeviceRadixSort requires a macro-visible primitive numeric type; aliases need explicit radix metadata",
        ));
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return Err(syn::Error::new_spanned(
            key_type,
            "DeviceRadixSort cannot resolve a qualified or aliased key type before rustc type inference",
        ));
    }
    let name = path.path.segments[0].ident.to_string();
    let (bits, kind) = match name.as_str() {
        "u8" => (8, PrimitiveRadixKind::Unsigned),
        "u16" => (16, PrimitiveRadixKind::Unsigned),
        "u32" => (32, PrimitiveRadixKind::Unsigned),
        "u64" => (64, PrimitiveRadixKind::Unsigned),
        "i8" => (8, PrimitiveRadixKind::Signed),
        "i16" => (16, PrimitiveRadixKind::Signed),
        "i32" => (32, PrimitiveRadixKind::Signed),
        "i64" => (64, PrimitiveRadixKind::Signed),
        "f32" => (32, PrimitiveRadixKind::Floating),
        "f64" => (64, PrimitiveRadixKind::Floating),
        _ => {
            return Err(syn::Error::new_spanned(
                key_type,
                "custom and aliased radix keys need macro-visible decomposer metadata",
            ));
        }
    };
    Ok(PrimitiveRadix { bits, kind })
}

fn expand_device_kernels(module: syn::Ident) -> syn::Result<proc_macro2::TokenStream> {
    audit::generated_rust(quote! {
        #[doc(hidden)]
        use ::rccl_cub::__private::cuda_core;
        #[doc(hidden)]
        use ::rccl_cub::__private::cuda_host;

        #[::rccl_cub::__private::cuda_device::cuda_module]
        #[doc(hidden)]
        mod #module {
            use ::rccl_cub::__private::cuda_device::{
                DisjointSlice, LinearTiles, Uniform, kernel, launch_bounds, launch_contract, thread,
            };

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

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (128, 1, 1),
                dynamic_shared_range = (4096, 8192),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_radix_sort_histogram<
                K,
                B,
                DefaultKey,
                ToOrdered,
                Digit,
                const ITEMS: usize,
                const MAX_PASSES: usize,
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
                ::rccl_cub::__private::radix_histogram_agent::<
                    K,
                    B,
                    DefaultKey,
                    ToOrdered,
                    Digit,
                    ITEMS,
                    MAX_PASSES,
                >(keys, begin_bit, end_bit, default_key, to_ordered, digit, bins);
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (2336, 2336),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_radix_sort_exclusive_sum<const PASSES: usize>(
                bins: DisjointSlice<u64>,
            ) {
                let bin_thread = thread::index_1d_u32(launch_context);
                ::rccl_cub::__private::radix_exclusive_sum_agent::<PASSES>(bins, bin_thread);
            }

            #[kernel]
            #[launch_bounds(384)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (384, 1, 1),
                dynamic_shared_range = (0, 101376),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_radix_sort_onesweep_keys<
                K,
                B,
                DefaultKey,
                ToOrdered,
                FromOrdered,
                Digit,
                RankDigit,
                const ITEMS: usize,
                const UNION_BYTES: usize,
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
                mut digit: Digit,
                mut rank_digit: RankDigit,
            ) where
                K: Copy,
                B: Copy,
                DefaultKey: Fn() -> B + Copy,
                ToOrdered: Fn(K) -> B + Copy,
                FromOrdered: Fn(B) -> K + Copy,
                Digit: FnMut(B, u32, u32) -> u32 + Copy,
                RankDigit: FnMut(B, u32, u32) -> u32 + Copy,
            {
                ::rccl_cub::__private::radix_onesweep_keys_agent::<
                    K,
                    B,
                    DefaultKey,
                    ToOrdered,
                    FromOrdered,
                    Digit,
                    RankDigit,
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
                    (default_key, to_ordered, from_ordered, digit, rank_digit),
                );
            }

            #[kernel]
            #[launch_bounds(384)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (384, 1, 1),
                dynamic_shared_range = (0, 101376),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_radix_sort_onesweep_pairs<
                K,
                V,
                B,
                DefaultKey,
                ToOrdered,
                FromOrdered,
                Digit,
                RankDigit,
                const ITEMS: usize,
                const UNION_BYTES: usize,
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
                mut digit: Digit,
                mut rank_digit: RankDigit,
            ) where
                K: Copy,
                V: Copy,
                B: Copy,
                DefaultKey: Fn() -> B + Copy,
                ToOrdered: Fn(K) -> B + Copy,
                FromOrdered: Fn(B) -> K + Copy,
                Digit: FnMut(B, u32, u32) -> u32 + Copy,
                RankDigit: FnMut(B, u32, u32) -> u32 + Copy,
            {
                ::rccl_cub::__private::radix_onesweep_pairs_agent::<
                    K,
                    V,
                    B,
                    DefaultKey,
                    ToOrdered,
                    FromOrdered,
                    Digit,
                    RankDigit,
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
                    (default_key, to_ordered, from_ordered, digit, rank_digit),
                );
            }

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

fn expand_device_transform_repeat_in_place(
    call: call::DeviceTransformRepeatInPlaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let module = call.module;
    let stream = call.stream;
    let output = call.output;
    let repeated = call.repeated;
    let op = call.op;
    audit::generated_expression(quote! {{
        let __rccl_module = #module;
        let __rccl_stream = #stream;
        let __rccl_output = #output;
        let __rccl_repeated = #repeated;
        let __rccl_op = #op;
        let __rccl_output_len = __rccl_output.len();
        if __rccl_output_len == 0 {
            ::core::result::Result::Ok(())
        } else {
            let __rccl_launch = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                __rccl_output_len.div_ceil(8).div_ceil(256) as u32,
                256,
                0,
            );
            match __rccl_module.prepare_transform_repeat_in_place_for(
                &__rccl_repeated,
                &__rccl_op,
                __rccl_launch,
            ) {
                ::core::result::Result::Ok(__rccl_prepared) => __rccl_module
                    .transform_repeat_in_place(
                        __rccl_stream,
                        &__rccl_prepared,
                        __rccl_repeated,
                        __rccl_op,
                        __rccl_output,
                    ),
                ::core::result::Result::Err(__rccl_error) => {
                    ::core::result::Result::Err(__rccl_error)
                }
            }
        }
    }})
}

fn tmp_primitive_size(value_type: &syn::Type) -> syn::Result<usize> {
    let semantic_type = ungroup_type(value_type);
    let syn::Type::Path(path) = semantic_type else {
        return Err(syn::Error::new_spanned(
            value_type,
            "this CUB schedule needs macro-visible type layout metadata",
        ));
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return Err(syn::Error::new_spanned(
            value_type,
            "aliases and custom values need explicit macro-visible size/alignment metadata",
        ));
    }
    match path.path.segments[0].ident.to_string().as_str() {
        "bool" | "u8" | "i8" => Ok(1),
        "u16" | "i16" => Ok(2),
        "u32" | "i32" | "f32" => Ok(4),
        "u64" | "i64" | "usize" | "isize" | "f64" => Ok(8),
        "u128" | "i128" => Ok(16),
        _ => Err(syn::Error::new_spanned(
            value_type,
            "aliases and custom values need explicit macro-visible size/alignment metadata",
        )),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CudaTarget {
    capability: u16,
}

/// Parse the macro-only architecture argument used for CUB policy selection.
///
/// The `sm_XX` token deliberately looks like a Rust type argument so the call
/// retains ordinary turbofish syntax, but it is consumed here and is never
/// emitted as a Rust type or runtime value.
fn tmp_cuda_target(target: &syn::Type) -> syn::Result<CudaTarget> {
    let syn::Type::Path(path) = ungroup_type(target) else {
        return Err(syn::Error::new_spanned(
            target,
            "the CUB target must be a macro-only architecture such as `sm_89`",
        ));
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return Err(syn::Error::new_spanned(
            target,
            "the CUB target must be a single architecture token such as `sm_89`",
        ));
    }
    let spelling = path.path.segments[0].ident.to_string();
    let Some(numeric_with_suffix) = spelling.strip_prefix("sm_") else {
        return Err(syn::Error::new_spanned(
            target,
            "the CUB target must use CUDA's `sm_XX` spelling",
        ));
    };
    let numeric = numeric_with_suffix.trim_end_matches(['a', 'f']);
    if numeric.len() < 2
        || numeric.len() > 3
        || numeric_with_suffix[numeric.len()..]
            .chars()
            .any(|suffix| !matches!(suffix, 'a' | 'f'))
    {
        return Err(syn::Error::new_spanned(
            target,
            "invalid CUDA architecture; expected `sm_XX`, `sm_XXX`, or an `a`/`f` suffix",
        ));
    }
    let capability = numeric
        .parse::<u16>()
        .map_err(|_| syn::Error::new_spanned(target, "CUDA architecture digits must be decimal"))?;
    if capability < 52 {
        return Err(syn::Error::new_spanned(
            target,
            "the pinned CCCL ordering policies require sm_52 or newer",
        ));
    }
    Ok(CudaTarget { capability })
}

fn nominal_4b_items(nominal: usize, item_size: usize) -> usize {
    (nominal * 4 / item_size).clamp(1, nominal)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MergeSortPolicy {
    threads: usize,
    items: usize,
}

fn merge_sort_policy(target: CudaTarget, key_size: usize) -> MergeSortPolicy {
    if target.capability >= 60 {
        MergeSortPolicy {
            threads: 256,
            items: nominal_4b_items(17, key_size),
        }
    } else {
        MergeSortPolicy {
            threads: 512,
            items: nominal_4b_items(15, key_size),
        }
    }
}

fn align_16(bytes: usize) -> Option<usize> {
    bytes.checked_add(15).map(|value| value / 16 * 16)
}

/// Size of the exact CUB `AgentBlockSort`/`AgentMerge` shared-memory union.
///
/// `BlockExchange` inserts one item after each 32 logical items when the
/// per-thread item count is a power of two greater than four. `BlockStore`
/// derives from that 16-byte-aligned exchange storage and appends one volatile
/// `int` for the partial-tile guard. The block-merge member remains a dense
/// `T[ITEMS_PER_TILE + 1]`; the enclosing union takes the larger member.
fn merge_sort_shared_bytes(
    threads: usize,
    items: usize,
    key_size: usize,
    value_size: Option<usize>,
) -> Option<usize> {
    let tile_items = threads.checked_mul(items)?;
    let exchange_items = if items > 4 && items.is_power_of_two() {
        tile_items.checked_add(tile_items / 32)?
    } else {
        tile_items
    };
    let representation = |item_size: usize| -> Option<usize> {
        let merge = tile_items.checked_add(1)?.checked_mul(item_size)?;
        let exchange = align_16(exchange_items.checked_mul(item_size)?)?;
        let store = align_16(exchange.checked_add(core::mem::size_of::<i32>())?)?;
        Some(merge.max(store))
    };
    let mut bytes = representation(key_size)?;
    if let Some(value_size) = value_size {
        bytes = bytes.max(representation(value_size)?);
    }
    align_16(bytes)
}

fn merge_load_to_shared_policy(
    target: CudaTarget,
    key_size: usize,
    value_size: Option<usize>,
) -> (bool, bool) {
    if target.capability >= 100 {
        return (true, value_size.is_some());
    }
    if target.capability >= 90 {
        let selected = match value_size {
            None => key_size != 8,
            Some(value_size) => match key_size {
                8 => value_size != 8,
                16 => matches!(value_size, 1 | 8),
                _ => true,
            },
        };
        return (selected, value_size.is_some() && selected);
    }
    if target.capability >= 80 {
        let selected = match value_size {
            None => key_size < 4,
            Some(value_size) => {
                key_size == 1
                    || (key_size == 2 && value_size < 4)
                    || (key_size == 4 && value_size == 1)
            }
        };
        return (selected, value_size.is_some() && selected);
    }
    (false, false)
}

fn load_to_shared_buffer_bytes(items: usize, item_size: usize) -> usize {
    let bytes = items * item_size;
    if item_size >= 16 || bytes == 0 {
        bytes
    } else {
        bytes.div_ceil(16) * 16 + 16
    }
}

fn merge_shared_bytes(
    threads: usize,
    items: usize,
    key_size: usize,
    value_size: Option<usize>,
    use_load_to_shared_keys: bool,
    use_load_to_shared_values: bool,
) -> usize {
    let tile_items = threads * items;
    let exchange_items = if items > 4 && items.is_power_of_two() {
        tile_items + tile_items / 32
    } else {
        tile_items
    };
    let representation = |item_size: usize, use_load_to_shared: bool| {
        if use_load_to_shared {
            load_to_shared_buffer_bytes(tile_items + 1, item_size)
                + usize::from(item_size < 16) * 32
        } else {
            (tile_items + 1) * item_size
        }
    };
    let block_store = |item_size: usize| {
        align_16(align_16(exchange_items * item_size).unwrap() + core::mem::size_of::<i32>())
            .unwrap()
    };
    let mut union_bytes =
        representation(key_size, use_load_to_shared_keys).max(block_store(key_size));
    if let Some(value_size) = value_size {
        union_bytes = union_bytes
            .max(representation(value_size, use_load_to_shared_values))
            .max(block_store(value_size));
    }
    let union_bytes = align_16(union_bytes).unwrap();
    if use_load_to_shared_keys || use_load_to_shared_values {
        // `temp_storages_with_bl2sh` derives from the 16-byte-aligned union,
        // appends BlockLoadToShared's u64 mbarrier cell, then rounds the whole
        // derived object back to its inherited 16-byte alignment.
        align_16(union_bytes + core::mem::size_of::<u64>()).unwrap()
    } else {
        union_bytes
    }
}

fn merge_keys_launch(
    key_type: &syn::Type,
    key_size: usize,
    use_load_to_shared: bool,
    items: usize,
) -> proc_macro2::TokenStream {
    let tile_items = 512usize * items;
    let shared_bytes = merge_shared_bytes(512, items, key_size, None, use_load_to_shared, false);
    quote! {{
        let __rccl_num_tiles = __rccl_output_len.div_ceil(#tile_items);
        let __rccl_num_diagonals = __rccl_num_tiles + 1;
        let __rccl_partition_launch = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
            __rccl_num_diagonals.div_ceil(256).min(u32::MAX as usize) as u32,
            256,
            0,
        );
        match __rccl_execution
            .module()
            .prepare_device_partition_merge_path::<#key_type, _, #items>(__rccl_partition_launch)
        {
            ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
            ),
            ::core::result::Result::Ok(__rccl_partition_prepared) => match __rccl_execution
                .module()
                .device_partition_merge_path::<#key_type, _, #items>(
                    __rccl_execution.stream(),
                    &__rccl_partition_prepared,
                    __rccl_first,
                    __rccl_second,
                    __rccl_order,
                    &mut *__rccl_partitions,
                )
            {
                ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                ),
                ::core::result::Result::Ok(()) => {
                    let __rccl_agent_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_tiles.min(u32::MAX as usize) as u32,
                            512,
                            #shared_bytes as u32,
                        );
                    match __rccl_execution
                        .module()
                        .prepare_device_merge_keys_agent::<#key_type, _, #items, #key_size, #use_load_to_shared, #shared_bytes>(
                            __rccl_agent_launch,
                        )
                    {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(__rccl_agent_prepared) => __rccl_execution
                            .module()
                            .device_merge_keys_agent::<#key_type, _, #items, #key_size, #use_load_to_shared, #shared_bytes>(
                                __rccl_execution.stream(),
                                &__rccl_agent_prepared,
                                __rccl_first,
                                __rccl_second,
                                &*__rccl_partitions,
                                __rccl_order,
                                __rccl_output,
                            )
                            .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                    }
                }
            },
        }
    }}
}

fn merge_pairs_launch(
    key_type: &syn::Type,
    value_type: &syn::Type,
    key_size: usize,
    value_size: usize,
    use_load_to_shared_keys: bool,
    use_load_to_shared_values: bool,
    items: usize,
) -> proc_macro2::TokenStream {
    let tile_items = 512usize * items;
    let shared_bytes = merge_shared_bytes(
        512,
        items,
        key_size,
        Some(value_size),
        use_load_to_shared_keys,
        use_load_to_shared_values,
    );
    quote! {{
        let __rccl_num_tiles = __rccl_output_len.div_ceil(#tile_items);
        let __rccl_num_diagonals = __rccl_num_tiles + 1;
        let __rccl_partition_launch = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
            __rccl_num_diagonals.div_ceil(256).min(u32::MAX as usize) as u32,
            256,
            0,
        );
        match __rccl_execution
            .module()
            .prepare_device_partition_merge_path::<#key_type, _, #items>(__rccl_partition_launch)
        {
            ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
            ),
            ::core::result::Result::Ok(__rccl_partition_prepared) => match __rccl_execution
                .module()
                .device_partition_merge_path::<#key_type, _, #items>(
                    __rccl_execution.stream(),
                    &__rccl_partition_prepared,
                    __rccl_first_keys,
                    __rccl_second_keys,
                    __rccl_order,
                    &mut *__rccl_partitions,
                )
            {
                ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                ),
                ::core::result::Result::Ok(()) => {
                    let __rccl_agent_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_tiles.min(u32::MAX as usize) as u32,
                            512,
                            #shared_bytes as u32,
                        );
                    match __rccl_execution
                        .module()
                        .prepare_device_merge_pairs_agent::<#key_type, #value_type, _, #items, #key_size, #value_size, #use_load_to_shared_keys, #use_load_to_shared_values, #shared_bytes>(
                            __rccl_agent_launch,
                        )
                    {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(__rccl_agent_prepared) => __rccl_execution
                            .module()
                            .device_merge_pairs_agent::<#key_type, #value_type, _, #items, #key_size, #value_size, #use_load_to_shared_keys, #use_load_to_shared_values, #shared_bytes>(
                                __rccl_execution.stream(),
                                &__rccl_agent_prepared,
                                __rccl_first_keys,
                                __rccl_first_values,
                                __rccl_second_keys,
                                __rccl_second_values,
                                &*__rccl_partitions,
                                __rccl_order,
                                __rccl_output_keys,
                                __rccl_output_values,
                            )
                            .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                    }
                }
            },
        }
    }}
}

fn expand_device_merge_keys(
    call: call::DeviceMergeKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = tmp_cuda_target(&call.target)?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let items = nominal_4b_items(nominal_items, key_size);
    let (use_load_to_shared, _) = merge_load_to_shared_policy(target, key_size, None);
    let launch = merge_keys_launch(&key_type, key_size, use_load_to_shared, items);
    let execution = call.execution;
    let first = call.first;
    let second = call.second;
    let partitions = call.partitions;
    let output = call.output;
    let order = call.order;
    audit::generated_algorithm_expression(
        audit::Algorithm::Merge,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_first = #first;
            let __rccl_second = #second;
            let __rccl_partitions = #partitions;
            let __rccl_output = #output;
            let __rccl_order = #order;
            match __rccl_first.len().checked_add(__rccl_second.len()) {
                ::core::option::Option::None => ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::SizeOverflow,
                ),
                ::core::option::Option::Some(__rccl_output_len)
                    if __rccl_output.len() != __rccl_output_len =>
                {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "output",
                        expected: __rccl_output_len,
                        actual: __rccl_output.len(),
                    })
                }
                ::core::option::Option::Some(0) => ::core::result::Result::Ok(()),
                ::core::option::Option::Some(__rccl_output_len) => {
                    let __rccl_required_partitions = __rccl_output_len.div_ceil(#items * 512) + 1;
                    if __rccl_partitions.len() != __rccl_required_partitions {
                        ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                role: "merge partitions",
                                expected: __rccl_required_partitions,
                                actual: __rccl_partitions.len(),
                            },
                        )
                    } else {
                        #launch
                    }
                }
            }
        }},
    )
}

fn expand_device_merge_keys_partition_count(
    call: call::DeviceMergeKeysPartitionCountCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let target = tmp_cuda_target(&call.target)?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let items = nominal_4b_items(nominal_items, key_size);
    let tile_items = 512usize * items;
    let first_len = call.first_len;
    let second_len = call.second_len;
    audit::generated_expression(quote! {{
        match (#first_len).checked_add(#second_len) {
            ::core::option::Option::Some(0) => ::core::result::Result::Ok(0usize),
            ::core::option::Option::Some(__rccl_total) => {
                ::core::result::Result::Ok(__rccl_total.div_ceil(#tile_items) + 1)
            }
            ::core::option::Option::None => ::core::result::Result::Err(
                ::rccl_cub::DeviceAlgorithmError::SizeOverflow,
            ),
        }
    }})
}

fn expand_device_merge_pairs_partition_count(
    call: call::DeviceMergePairsPartitionCountCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let value_size = tmp_primitive_size(&call.value_type)?;
    let target = tmp_cuda_target(&call.target)?;
    let tune_size = key_size
        .checked_add(value_size)
        .ok_or_else(|| syn::Error::new_spanned(&call.value_type, "merge tuning size overflowed"))?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let items = nominal_4b_items(nominal_items, tune_size);
    let tile_items = 512usize * items;
    let first_len = call.first_len;
    let second_len = call.second_len;
    audit::generated_expression(quote! {{
        match (#first_len).checked_add(#second_len) {
            ::core::option::Option::Some(0) => ::core::result::Result::Ok(0usize),
            ::core::option::Option::Some(__rccl_total) => {
                ::core::result::Result::Ok(__rccl_total.div_ceil(#tile_items) + 1)
            }
            ::core::option::Option::None => ::core::result::Result::Err(
                ::rccl_cub::DeviceAlgorithmError::SizeOverflow,
            ),
        }
    }})
}

fn expand_device_merge_pairs(
    call: call::DeviceMergePairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let target = tmp_cuda_target(&call.target)?;
    let key_size = tmp_primitive_size(&key_type)?;
    let value_size = tmp_primitive_size(&value_type)?;
    let tune_size = key_size
        .checked_add(value_size)
        .ok_or_else(|| syn::Error::new_spanned(&value_type, "merge tuning size overflowed"))?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let items = nominal_4b_items(nominal_items, tune_size);
    let (use_load_to_shared_keys, use_load_to_shared_values) =
        merge_load_to_shared_policy(target, key_size, Some(value_size));
    let launch = merge_pairs_launch(
        &key_type,
        &value_type,
        key_size,
        value_size,
        use_load_to_shared_keys,
        use_load_to_shared_values,
        items,
    );
    let execution = call.execution;
    let first_keys = call.first_keys;
    let first_values = call.first_values;
    let second_keys = call.second_keys;
    let second_values = call.second_values;
    let partitions = call.partitions;
    let output_keys = call.output_keys;
    let output_values = call.output_values;
    let order = call.order;
    audit::generated_algorithm_expression(
        audit::Algorithm::Merge,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_first_keys = #first_keys;
            let __rccl_first_values = #first_values;
            let __rccl_second_keys = #second_keys;
            let __rccl_second_values = #second_values;
            let __rccl_partitions = #partitions;
            let __rccl_output_keys = #output_keys;
            let __rccl_output_values = #output_values;
            let __rccl_order = #order;
            let __rccl_first_len = __rccl_first_keys.len();
            let __rccl_second_len = __rccl_second_keys.len();
            let __rccl_shape_error = if __rccl_first_values.len() != __rccl_first_len {
                ::core::option::Option::Some((
                    "first values",
                    __rccl_first_len,
                    __rccl_first_values.len(),
                ))
            } else if __rccl_second_values.len() != __rccl_second_len {
                ::core::option::Option::Some((
                    "second values",
                    __rccl_second_len,
                    __rccl_second_values.len(),
                ))
            } else {
                ::core::option::Option::None
            };
            if let ::core::option::Option::Some((__rccl_role, __rccl_expected, __rccl_actual)) =
                __rccl_shape_error
            {
                ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                    role: __rccl_role,
                    expected: __rccl_expected,
                    actual: __rccl_actual,
                })
            } else {
                match __rccl_first_len.checked_add(__rccl_second_len) {
                    ::core::option::Option::None => ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::SizeOverflow,
                    ),
                    ::core::option::Option::Some(__rccl_output_len)
                        if __rccl_output_keys.len() != __rccl_output_len =>
                    {
                        ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                role: "output keys",
                                expected: __rccl_output_len,
                                actual: __rccl_output_keys.len(),
                            },
                        )
                    }
                    ::core::option::Option::Some(__rccl_output_len)
                        if __rccl_output_values.len() != __rccl_output_len =>
                    {
                        ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                role: "output values",
                                expected: __rccl_output_len,
                                actual: __rccl_output_values.len(),
                            },
                        )
                    }
                    ::core::option::Option::Some(0) => ::core::result::Result::Ok(()),
                    ::core::option::Option::Some(__rccl_output_len) => {
                        let __rccl_required_partitions =
                            __rccl_output_len.div_ceil(#items * 512) + 1;
                        if __rccl_partitions.len() != __rccl_required_partitions {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                    role: "merge partitions",
                                    expected: __rccl_required_partitions,
                                    actual: __rccl_partitions.len(),
                                },
                            )
                        } else {
                            #launch
                        }
                    }
                }
            }
        }},
    )
}

fn expand_device_merge_sort_partition_count(
    call: call::DeviceMergeSortPartitionCountCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let target = tmp_cuda_target(&call.target)?;
    let policy = merge_sort_policy(target, key_size);
    let tile_items = policy.threads * policy.items;
    let len = call.len;
    audit::generated_expression(quote! {{
        let __rccl_len: usize = #len;
        if __rccl_len == 0 {
            ::core::result::Result::<usize, ::rccl_cub::DeviceAlgorithmError>::Ok(0usize)
        } else {
            ::core::result::Result::<usize, ::rccl_cub::DeviceAlgorithmError>::Ok(
                __rccl_len.div_ceil(#tile_items) + 1,
            )
        }
    }})
}

fn expand_device_merge_sort_keys(
    call: call::DeviceMergeSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = tmp_cuda_target(&call.target)?;
    let policy = merge_sort_policy(target, key_size);
    let threads = policy.threads;
    let items = policy.items;
    let tile_items = threads * items;
    let shared_bytes =
        merge_sort_shared_bytes(threads, items, key_size, None).ok_or_else(|| {
            syn::Error::new_spanned(&key_type, "merge-sort shared storage overflowed")
        })?;
    let prepare_block = format_ident!("prepare_device_merge_sort_block_keys_{threads}");
    let launch_block = format_ident!("device_merge_sort_block_keys_{threads}");
    let prepare_block_in_place =
        format_ident!("prepare_device_merge_sort_block_keys_in_place_{threads}");
    let launch_block_in_place = format_ident!("device_merge_sort_block_keys_in_place_{threads}");
    let prepare_merge = format_ident!("prepare_device_merge_sort_merge_keys_{threads}");
    let launch_merge = format_ident!("device_merge_sort_merge_keys_{threads}");
    let execution = call.execution;
    let keys = call.keys;
    let scratch_keys = call.scratch_keys;
    let partitions = call.partitions;
    let order = call.order;
    audit::generated_algorithm_expression(
        audit::Algorithm::MergeSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_keys = #keys;
            let __rccl_scratch_keys = #scratch_keys;
            let __rccl_partitions = #partitions;
            let __rccl_order = #order;
            let __rccl_len = __rccl_keys.len();
            if __rccl_scratch_keys.len() != __rccl_len {
                ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                    role: "merge-sort scratch keys",
                    expected: __rccl_len,
                    actual: __rccl_scratch_keys.len(),
                })
            } else if __rccl_len == 0 {
                if __rccl_partitions.len() == 0 {
                    ::core::result::Result::Ok(())
                } else {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: 0,
                        actual: __rccl_partitions.len(),
                    })
                }
            } else {
                let __rccl_num_tiles = __rccl_len.div_ceil(#tile_items);
                let __rccl_num_partitions = __rccl_num_tiles + 1;
                if __rccl_partitions.len() != __rccl_num_partitions {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: __rccl_num_partitions,
                        actual: __rccl_partitions.len(),
                    })
                } else {
                    let __rccl_num_passes = usize::BITS as usize
                        - (__rccl_num_tiles - 1).leading_zeros() as usize;
                    let mut __rccl_ping = (__rccl_num_passes & 1) == 0;
                    let __rccl_agent_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_tiles.min(u32::MAX as usize) as u32,
                            #threads as u32,
                            #shared_bytes as u32,
                        );
                    let mut __rccl_result = if __rccl_ping {
                        match __rccl_execution
                            .module()
                            .#prepare_block_in_place::<#key_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                        {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                                .module()
                                .#launch_block_in_place::<#key_type, _, #items, #shared_bytes>(
                                    __rccl_execution.stream(),
                                    &__rccl_prepared,
                                    __rccl_order,
                                    &mut *__rccl_keys,
                                )
                                .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                        }
                    } else {
                        match __rccl_execution
                            .module()
                            .#prepare_block::<#key_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                        {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                                .module()
                                .#launch_block::<#key_type, _, #items, #shared_bytes>(
                                    __rccl_execution.stream(),
                                    &__rccl_prepared,
                                    &*__rccl_keys,
                                    __rccl_order,
                                    &mut *__rccl_scratch_keys,
                                )
                                .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                        }
                    };
                    let __rccl_partition_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_partitions.div_ceil(256).min(u32::MAX as usize) as u32,
                            256,
                            0,
                        );
                    let mut __rccl_pass = 0usize;
                    while __rccl_pass < __rccl_num_passes && __rccl_result.is_ok() {
                        let __rccl_target_merged_tiles = 2usize << __rccl_pass;
                        __rccl_result = match __rccl_execution
                            .module()
                            .prepare_device_merge_sort_partition::<#key_type, _, #threads, #items>(
                                __rccl_partition_launch,
                            )
                        {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => {
                                let __rccl_launch_result = if __rccl_ping {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_keys,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                } else {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_scratch_keys,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                };
                                __rccl_launch_result
                                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                            }
                        };
                        if __rccl_result.is_ok() {
                            __rccl_result = match __rccl_execution
                                .module()
                                .#prepare_merge::<#key_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                            {
                                ::core::result::Result::Err(__rccl_error) => {
                                    ::core::result::Result::Err(
                                        ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                    )
                                }
                                ::core::result::Result::Ok(__rccl_prepared) => {
                                    let __rccl_launch_result = if __rccl_ping {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_keys,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_scratch_keys,
                                        )
                                    } else {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_scratch_keys,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_keys,
                                        )
                                    };
                                    __rccl_launch_result
                                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                                }
                            };
                        }
                        __rccl_ping = !__rccl_ping;
                        __rccl_pass += 1;
                    }
                    __rccl_result
                }
            }
        }},
    )
}

fn expand_device_merge_sort_keys_copy(
    call: call::DeviceMergeSortKeysCopyCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = tmp_cuda_target(&call.target)?;
    let policy = merge_sort_policy(target, key_size);
    let threads = policy.threads;
    let items = policy.items;
    let tile_items = threads * items;
    let shared_bytes =
        merge_sort_shared_bytes(threads, items, key_size, None).ok_or_else(|| {
            syn::Error::new_spanned(&key_type, "merge-sort shared storage overflowed")
        })?;
    let prepare_block = format_ident!("prepare_device_merge_sort_block_keys_{threads}");
    let launch_block = format_ident!("device_merge_sort_block_keys_{threads}");
    let prepare_merge = format_ident!("prepare_device_merge_sort_merge_keys_{threads}");
    let launch_merge = format_ident!("device_merge_sort_merge_keys_{threads}");
    let execution = call.execution;
    let input = call.input;
    let scratch_keys = call.scratch_keys;
    let partitions = call.partitions;
    let output = call.output;
    let order = call.order;
    audit::generated_algorithm_expression(
        audit::Algorithm::MergeSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_input = #input;
            let __rccl_scratch_keys = #scratch_keys;
            let __rccl_partitions = #partitions;
            let __rccl_output = #output;
            let __rccl_order = #order;
            let __rccl_len = __rccl_input.len();
            let __rccl_shape_error = if __rccl_output.len() != __rccl_len {
                ::core::option::Option::Some(("output keys", __rccl_output.len()))
            } else if __rccl_scratch_keys.len() != __rccl_len {
                ::core::option::Option::Some((
                    "merge-sort scratch keys",
                    __rccl_scratch_keys.len(),
                ))
            } else {
                ::core::option::Option::None
            };
            if let ::core::option::Option::Some((__rccl_role, __rccl_actual)) = __rccl_shape_error {
                ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                    role: __rccl_role,
                    expected: __rccl_len,
                    actual: __rccl_actual,
                })
            } else if __rccl_len == 0 {
                if __rccl_partitions.len() == 0 {
                    ::core::result::Result::Ok(())
                } else {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: 0,
                        actual: __rccl_partitions.len(),
                    })
                }
            } else {
                let __rccl_num_tiles = __rccl_len.div_ceil(#tile_items);
                let __rccl_num_partitions = __rccl_num_tiles + 1;
                if __rccl_partitions.len() != __rccl_num_partitions {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: __rccl_num_partitions,
                        actual: __rccl_partitions.len(),
                    })
                } else {
                    let __rccl_num_passes = usize::BITS as usize
                        - (__rccl_num_tiles - 1).leading_zeros() as usize;
                    let mut __rccl_ping = (__rccl_num_passes & 1) == 0;
                    let __rccl_agent_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_tiles.min(u32::MAX as usize) as u32,
                            #threads as u32,
                            #shared_bytes as u32,
                        );
                    let mut __rccl_result = match __rccl_execution
                        .module()
                        .#prepare_block::<#key_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                    {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(__rccl_prepared) => {
                            let __rccl_launch_result = if __rccl_ping {
                                __rccl_execution.module().#launch_block::<#key_type, _, #items, #shared_bytes>(
                                    __rccl_execution.stream(),
                                    &__rccl_prepared,
                                    __rccl_input,
                                    __rccl_order,
                                    &mut *__rccl_output,
                                )
                            } else {
                                __rccl_execution.module().#launch_block::<#key_type, _, #items, #shared_bytes>(
                                    __rccl_execution.stream(),
                                    &__rccl_prepared,
                                    __rccl_input,
                                    __rccl_order,
                                    &mut *__rccl_scratch_keys,
                                )
                            };
                            __rccl_launch_result.map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                        }
                    };
                    let __rccl_partition_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_partitions.div_ceil(256).min(u32::MAX as usize) as u32,
                            256,
                            0,
                        );
                    let mut __rccl_pass = 0usize;
                    while __rccl_pass < __rccl_num_passes && __rccl_result.is_ok() {
                        let __rccl_target_merged_tiles = 2usize << __rccl_pass;
                        __rccl_result = match __rccl_execution
                            .module()
                            .prepare_device_merge_sort_partition::<#key_type, _, #threads, #items>(
                                __rccl_partition_launch,
                            )
                        {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => {
                                let __rccl_launch_result = if __rccl_ping {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_output,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                } else {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_scratch_keys,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                };
                                __rccl_launch_result
                                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                            }
                        };
                        if __rccl_result.is_ok() {
                            __rccl_result = match __rccl_execution
                                .module()
                                .#prepare_merge::<#key_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                            {
                                ::core::result::Result::Err(__rccl_error) => {
                                    ::core::result::Result::Err(
                                        ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                    )
                                }
                                ::core::result::Result::Ok(__rccl_prepared) => {
                                    let __rccl_launch_result = if __rccl_ping {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_output,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_scratch_keys,
                                        )
                                    } else {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_scratch_keys,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_output,
                                        )
                                    };
                                    __rccl_launch_result
                                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                                }
                            };
                        }
                        __rccl_ping = !__rccl_ping;
                        __rccl_pass += 1;
                    }
                    __rccl_result
                }
            }
        }},
    )
}

fn expand_device_merge_sort_pairs(
    call: call::DeviceMergeSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_merge_sort_pairs_operation(
        call.key_type,
        call.value_type,
        call.target,
        call.execution,
        None,
        None,
        call.keys,
        call.values,
        call.scratch_keys,
        call.scratch_values,
        call.partitions,
        call.order,
    )
}

fn expand_device_merge_sort_pairs_copy(
    call: call::DeviceMergeSortPairsCopyCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_merge_sort_pairs_operation(
        call.key_type,
        call.value_type,
        call.target,
        call.execution,
        Some(call.input_keys),
        Some(call.input_values),
        call.output_keys,
        call.output_values,
        call.scratch_keys,
        call.scratch_values,
        call.partitions,
        call.order,
    )
}

#[allow(clippy::too_many_arguments)]
fn expand_device_merge_sort_pairs_operation(
    key_type: syn::Type,
    value_type: syn::Type,
    target: syn::Type,
    execution: syn::Expr,
    input_keys: Option<syn::Expr>,
    input_values: Option<syn::Expr>,
    primary_keys: syn::Expr,
    primary_values: syn::Expr,
    scratch_keys: syn::Expr,
    scratch_values: syn::Expr,
    partitions: syn::Expr,
    order: syn::Expr,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&key_type)?;
    let value_size = tmp_primitive_size(&value_type)?;
    let target = tmp_cuda_target(&target)?;
    let policy = merge_sort_policy(target, key_size);
    let threads = policy.threads;
    let items = policy.items;
    let tile_items = threads * items;
    let shared_bytes = merge_sort_shared_bytes(threads, items, key_size, Some(value_size))
        .ok_or_else(|| {
            syn::Error::new_spanned(&value_type, "merge-sort shared storage overflowed")
        })?;
    if shared_bytes > 49_152 {
        return Err(syn::Error::new_spanned(
            &value_type,
            "this pinned CUB pair policy requires virtual shared memory, whose exact agent path is not implemented; refusing a smaller substitute",
        ));
    }

    let prepare_block = format_ident!("prepare_device_merge_sort_block_pairs_{threads}");
    let launch_block = format_ident!("device_merge_sort_block_pairs_{threads}");
    let prepare_block_in_place =
        format_ident!("prepare_device_merge_sort_block_pairs_in_place_{threads}");
    let launch_block_in_place = format_ident!("device_merge_sort_block_pairs_in_place_{threads}");
    let prepare_merge = format_ident!("prepare_device_merge_sort_merge_pairs_{threads}");
    let launch_merge = format_ident!("device_merge_sort_merge_pairs_{threads}");

    let (mode_bindings, shape_error, initial_block) = match (input_keys, input_values) {
        (Some(input_keys), Some(input_values)) => (
            quote! {
                let __rccl_input_keys = #input_keys;
                let __rccl_input_values = #input_values;
                let __rccl_primary_keys = #primary_keys;
                let __rccl_primary_values = #primary_values;
            },
            quote! {
                if __rccl_input_values.len() != __rccl_len {
                    ::core::option::Option::Some(("input values", __rccl_input_values.len()))
                } else if __rccl_primary_keys.len() != __rccl_len {
                    ::core::option::Option::Some(("output keys", __rccl_primary_keys.len()))
                } else if __rccl_primary_values.len() != __rccl_len {
                    ::core::option::Option::Some(("output values", __rccl_primary_values.len()))
                } else if __rccl_scratch_keys.len() != __rccl_len {
                    ::core::option::Option::Some(("merge-sort scratch keys", __rccl_scratch_keys.len()))
                } else if __rccl_scratch_values.len() != __rccl_len {
                    ::core::option::Option::Some(("merge-sort scratch values", __rccl_scratch_values.len()))
                } else {
                    ::core::option::Option::None
                }
            },
            quote! {
                match __rccl_execution
                    .module()
                    .#prepare_block::<#key_type, #value_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                {
                    ::core::result::Result::Err(__rccl_error) => {
                        ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                        )
                    }
                    ::core::result::Result::Ok(__rccl_prepared) => {
                        let __rccl_launch_result = if __rccl_ping {
                            __rccl_execution.module().#launch_block::<
                                #key_type,
                                #value_type,
                                _,
                                #items,
                                #shared_bytes,
                            >(
                                __rccl_execution.stream(),
                                &__rccl_prepared,
                                __rccl_input_keys,
                                __rccl_input_values,
                                __rccl_order,
                                &mut *__rccl_primary_keys,
                                &mut *__rccl_primary_values,
                            )
                        } else {
                            __rccl_execution.module().#launch_block::<
                                #key_type,
                                #value_type,
                                _,
                                #items,
                                #shared_bytes,
                            >(
                                __rccl_execution.stream(),
                                &__rccl_prepared,
                                __rccl_input_keys,
                                __rccl_input_values,
                                __rccl_order,
                                &mut *__rccl_scratch_keys,
                                &mut *__rccl_scratch_values,
                            )
                        };
                        __rccl_launch_result.map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                    }
                }
            },
        ),
        (None, None) => (
            quote! {
                let __rccl_primary_keys = #primary_keys;
                let __rccl_primary_values = #primary_values;
            },
            quote! {
                if __rccl_primary_values.len() != __rccl_len {
                    ::core::option::Option::Some(("values", __rccl_primary_values.len()))
                } else if __rccl_scratch_keys.len() != __rccl_len {
                    ::core::option::Option::Some(("merge-sort scratch keys", __rccl_scratch_keys.len()))
                } else if __rccl_scratch_values.len() != __rccl_len {
                    ::core::option::Option::Some(("merge-sort scratch values", __rccl_scratch_values.len()))
                } else {
                    ::core::option::Option::None
                }
            },
            quote! {
                if __rccl_ping {
                    match __rccl_execution
                        .module()
                        .#prepare_block_in_place::<#key_type, #value_type, _, #items, #shared_bytes>(
                            __rccl_agent_launch,
                        )
                    {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                            .module()
                            .#launch_block_in_place::<#key_type, #value_type, _, #items, #shared_bytes>(
                                __rccl_execution.stream(),
                                &__rccl_prepared,
                                __rccl_order,
                                &mut *__rccl_primary_keys,
                                &mut *__rccl_primary_values,
                            )
                            .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                    }
                } else {
                    match __rccl_execution
                        .module()
                        .#prepare_block::<#key_type, #value_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                    {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                            .module()
                            .#launch_block::<#key_type, #value_type, _, #items, #shared_bytes>(
                                __rccl_execution.stream(),
                                &__rccl_prepared,
                                &*__rccl_primary_keys,
                                &*__rccl_primary_values,
                                __rccl_order,
                                &mut *__rccl_scratch_keys,
                                &mut *__rccl_scratch_values,
                            )
                            .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                    }
                }
            },
        ),
        _ => unreachable!("copy mode always supplies both input ranges"),
    };

    audit::generated_algorithm_expression(
        audit::Algorithm::MergeSort,
        quote! {{
            let __rccl_execution = #execution;
            #mode_bindings
            let __rccl_scratch_keys = #scratch_keys;
            let __rccl_scratch_values = #scratch_values;
            let __rccl_partitions = #partitions;
            let __rccl_order = #order;
            let __rccl_len = __rccl_primary_keys.len();
            let __rccl_shape_error = #shape_error;
            if let ::core::option::Option::Some((__rccl_role, __rccl_actual)) = __rccl_shape_error {
                ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                    role: __rccl_role,
                    expected: __rccl_len,
                    actual: __rccl_actual,
                })
            } else if __rccl_len == 0 {
                if __rccl_partitions.len() == 0 {
                    ::core::result::Result::Ok(())
                } else {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: 0,
                        actual: __rccl_partitions.len(),
                    })
                }
            } else {
                let __rccl_num_tiles = __rccl_len.div_ceil(#tile_items);
                let __rccl_num_partitions = __rccl_num_tiles + 1;
                if __rccl_partitions.len() != __rccl_num_partitions {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: __rccl_num_partitions,
                        actual: __rccl_partitions.len(),
                    })
                } else {
                    let __rccl_num_passes = usize::BITS as usize
                        - (__rccl_num_tiles - 1).leading_zeros() as usize;
                    let mut __rccl_ping = (__rccl_num_passes & 1) == 0;
                    let __rccl_agent_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_tiles.min(u32::MAX as usize) as u32,
                            #threads as u32,
                            #shared_bytes as u32,
                        );
                    let mut __rccl_result = #initial_block;
                    let __rccl_partition_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_partitions.div_ceil(256).min(u32::MAX as usize) as u32,
                            256,
                            0,
                        );
                    let mut __rccl_pass = 0usize;
                    while __rccl_pass < __rccl_num_passes && __rccl_result.is_ok() {
                        let __rccl_target_merged_tiles = 2usize << __rccl_pass;
                        __rccl_result = match __rccl_execution
                            .module()
                            .prepare_device_merge_sort_partition::<#key_type, _, #threads, #items>(
                                __rccl_partition_launch,
                            )
                        {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => {
                                let __rccl_launch_result = if __rccl_ping {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_primary_keys,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                } else {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_scratch_keys,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                };
                                __rccl_launch_result
                                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                            }
                        };
                        if __rccl_result.is_ok() {
                            __rccl_result = match __rccl_execution
                                .module()
                                .#prepare_merge::<#key_type, #value_type, _, #items, #shared_bytes>(
                                    __rccl_agent_launch,
                                )
                            {
                                ::core::result::Result::Err(__rccl_error) => {
                                    ::core::result::Result::Err(
                                        ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                    )
                                }
                                ::core::result::Result::Ok(__rccl_prepared) => {
                                    let __rccl_launch_result = if __rccl_ping {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            #value_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_primary_keys,
                                            &*__rccl_primary_values,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_scratch_keys,
                                            &mut *__rccl_scratch_values,
                                        )
                                    } else {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            #value_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_scratch_keys,
                                            &*__rccl_scratch_values,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_primary_keys,
                                            &mut *__rccl_primary_values,
                                        )
                                    };
                                    __rccl_launch_result
                                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                                }
                            };
                        }
                        __rccl_ping = !__rccl_ping;
                        __rccl_pass += 1;
                    }
                    __rccl_result
                }
            }
        }},
    )
}

fn expand_device_radix_sort_keys(
    call: call::DeviceRadixSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_via_buffer(call, false)
}

fn expand_device_radix_sort_workspace(
    call: call::DeviceRadixSortWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let target = tmp_cuda_target(&call.target)?;
    if !(80..90).contains(&target.capability) {
        return Err(syn::Error::new_spanned(
            call.target,
            "this workspace macro contains the pinned Policy800 layout only",
        ));
    }
    let primitive = primitive_radix(&call.key_type)?;
    let _ = primitive_radix_lowering(&call.key_type, primitive, false)?;
    let len = call.len;
    let passes = primitive.bits.div_ceil(8) as usize;
    let bit_size = (primitive.bits / 8) as usize;
    let (onesweep_threads, onesweep_items) = reg_bound_policy(384, 21, bit_size);
    let tile_items = onesweep_threads * onesweep_items;
    audit::generated_expression(quote! {{
        let __rccl_len: usize = #len;
        let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
        ::rccl_cub::RadixSortWorkspaceCounts {
            scratch_keys: __rccl_len,
            bins: #passes * 256usize,
            lookback: __rccl_blocks * 256usize,
            counters: #passes,
        }
    }})
}

fn expand_device_radix_sort_pairs_workspace(
    call: call::DeviceRadixSortPairsWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let target = tmp_cuda_target(&call.target)?;
    if !(80..90).contains(&target.capability) {
        return Err(syn::Error::new_spanned(
            call.target,
            "this workspace macro contains the pinned Policy800 layout only",
        ));
    }
    let primitive = primitive_radix(&call.key_type)?;
    let _ = primitive_radix_lowering(&call.key_type, primitive, false)?;
    let key_size = tmp_primitive_size(&call.key_type)?;
    let value_size = tmp_primitive_size(&call.value_type)?;
    let passes = primitive.bits.div_ceil(8) as usize;
    let nominal_items = if key_size == 4 { 17 } else { 21 };
    let (onesweep_threads, onesweep_items) =
        reg_bound_policy(384, nominal_items, key_size.max(value_size));
    let tile_items = onesweep_threads * onesweep_items;
    let len = call.len;
    audit::generated_expression(quote! {{
        let __rccl_len: usize = #len;
        let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
        ::rccl_cub::RadixSortPairsWorkspaceCounts {
            scratch_keys: __rccl_len,
            scratch_values: __rccl_len,
            bins: #passes * 256usize,
            lookback: __rccl_blocks * 256usize,
            counters: #passes,
        }
    }})
}

fn expand_device_radix_sort_keys_descending(
    call: call::DeviceRadixSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_via_buffer(call, true)
}

fn expand_device_radix_sort_keys_buffer(
    call: call::DeviceRadixSortKeysBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_buffer_order(call, false)
}

fn expand_device_radix_sort_keys_buffer_descending(
    call: call::DeviceRadixSortKeysBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_buffer_order(call, true)
}

/// Lower CUB's `DoubleBuffer` overload without importing its pointer/selector
/// representation. The maximum policy shape is TMP; the requested bit window,
/// pass count, and selected side are ordinary runtime state.
fn expand_device_radix_sort_keys_buffer_order(
    call: call::DeviceRadixSortKeysBufferCall,
    descending: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let target = tmp_cuda_target(&call.target)?;
    if !(80..90).contains(&target.capability) {
        return Err(syn::Error::new_spanned(
            call.target,
            "this lowering contains the exact pinned Policy800 one-sweep graph only; another architecture must add its own macro-selected CUB policy",
        ));
    }
    let primitive = primitive_radix(&call.key_type)?;
    let lowering = primitive_radix_lowering(&call.key_type, primitive, descending)?;
    let key_type = call.key_type;
    let bit_type = lowering.bit_type;
    let sentinel = lowering.sentinel;
    let default_key = lowering.default_key;
    let to_ordered = lowering.to_ordered;
    let from_ordered = lowering.from_ordered;
    let digit = lowering.digit;
    let rank_digit = lowering.rank_digit;
    let execution = call.execution;
    let buffers = call.buffers;
    let bit_range = call.bit_range;
    let bins = call.bins;
    let lookback = call.lookback;
    let counters = call.counters;
    let max_passes = primitive.bits.div_ceil(8) as usize;
    let key_bits = primitive.bits;
    let histogram_items = 16usize;
    let bit_size = (primitive.bits / 8) as usize;
    let (onesweep_threads, onesweep_items) = reg_bound_policy(384, 21, bit_size);
    let tile_items = onesweep_threads * onesweep_items;
    let histogram_shared_bytes = max_passes * 256usize * core::mem::size_of::<u32>();
    let union_bytes = tile_items * bit_size;
    let shared_bytes = union_bytes + 256usize * core::mem::size_of::<u64>();
    let portion_size = ((1usize << 28) - 1) / tile_items * tile_items;
    audit::generated_algorithm_expression(
        audit::Algorithm::RadixSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_buffers = #buffers;
            let __rccl_bit_range: ::core::ops::Range<u32> = #bit_range;
            let __rccl_bins = #bins;
            let __rccl_lookback = #lookback;
            let __rccl_counters = #counters;
            let __rccl_len = __rccl_buffers.len();
            (|| -> ::core::result::Result<(), ::rccl_cub::DeviceAlgorithmError> {
                let __rccl_begin_bit = __rccl_bit_range.start;
                let __rccl_end_bit = __rccl_bit_range.end;
                if __rccl_begin_bit > __rccl_end_bit || __rccl_end_bit > #key_bits {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::InvalidBitRange {
                            begin: __rccl_begin_bit,
                            end: __rccl_end_bit,
                            key_bits: #key_bits,
                        },
                    );
                }
                if __rccl_len > #portion_size {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "Policy800 one-portion maximum",
                            expected: #portion_size,
                            actual: __rccl_len,
                        },
                    );
                }
                let __rccl_passes =
                    ((__rccl_end_bit - __rccl_begin_bit) as usize).div_ceil(8usize);
                if __rccl_len == 0 || __rccl_passes == 0 {
                    return ::core::result::Result::Ok(());
                }
                let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
                let __rccl_required_bins = __rccl_passes * 256usize;
                if __rccl_bins.len() < __rccl_required_bins {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::WorkspaceTooSmall {
                            role: "radix bins",
                            required: __rccl_required_bins,
                            actual: __rccl_bins.len(),
                        },
                    );
                }
                if __rccl_lookback.len() != __rccl_blocks * 256usize {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "radix lookback",
                            expected: __rccl_blocks * 256usize,
                            actual: __rccl_lookback.len(),
                        },
                    );
                }
                if __rccl_counters.len() < __rccl_passes {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::WorkspaceTooSmall {
                            role: "radix pass counters",
                            required: __rccl_passes,
                            actual: __rccl_counters.len(),
                        },
                    );
                }

                let __rccl_init_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_passes as u32,
                        256,
                        0,
                    );
                let __rccl_init = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_init(__rccl_init_launch)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .device_radix_sort_init(
                        __rccl_execution.stream(),
                        &__rccl_init,
                        &mut *__rccl_counters,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_default_key = #default_key;
                let __rccl_to_ordered = #to_ordered;
                let __rccl_from_ordered = #from_ordered;
                let __rccl_digit = #digit;
                let __rccl_rank_digit = #rank_digit;
                let __rccl_key_witness: #key_type = 0 as #key_type;
                let __rccl_bit_witness: #bit_type = #sentinel;
                let __rccl_histogram_probe_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        1,
                        128,
                        #histogram_shared_bytes as u32,
                    );
                let __rccl_histogram_probe = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_histogram_for::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                        #max_passes,
                    >(
                        &__rccl_key_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_digit,
                        __rccl_histogram_probe_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                let __rccl_histogram_blocks_per_sm = __rccl_histogram_probe
                    .function()
                    .max_active_blocks_per_multiprocessor(128, #histogram_shared_bytes as u32)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
                let __rccl_multiprocessors = __rccl_execution
                    .stream()
                    .context()
                    .multiprocessor_count()
                    .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
                let __rccl_histogram_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_histogram_blocks_per_sm * __rccl_multiprocessors,
                        128,
                        #histogram_shared_bytes as u32,
                    );
                let __rccl_histogram = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_histogram_for::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                        #max_passes,
                    >(
                        &__rccl_key_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_digit,
                        __rccl_histogram_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .device_radix_sort_histogram::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                        #max_passes,
                    >(
                        __rccl_execution.stream(),
                        &__rccl_histogram,
                        &**__rccl_buffers.current(),
                        __rccl_begin_bit,
                        __rccl_end_bit,
                        __rccl_default_key,
                        __rccl_to_ordered,
                        __rccl_digit,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_scan_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_passes as u32,
                        256,
                        2336,
                    );
                let __rccl_scan = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_exclusive_sum::<#max_passes>(__rccl_scan_launch)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .device_radix_sort_exclusive_sum::<#max_passes>(
                        __rccl_execution.stream(),
                        &__rccl_scan,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_onesweep_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_blocks as u32,
                        #onesweep_threads as u32,
                        #shared_bytes as u32,
                    );
                let __rccl_onesweep = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_onesweep_keys_for::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        _,
                        _,
                        #onesweep_items,
                        #union_bytes,
                    >(
                        &__rccl_key_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_from_ordered,
                        &__rccl_digit,
                        &__rccl_rank_digit,
                        __rccl_onesweep_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                let mut __rccl_pass = 0usize;
                while __rccl_pass < __rccl_passes {
                    ::rccl_cub::__private::ZeroRadixLookback::zero_radix_lookback(
                        &mut *__rccl_lookback,
                        __rccl_execution.stream(),
                    )?;
                    let __rccl_bin_offset = __rccl_pass * 256usize;
                    let __rccl_current_bit =
                        __rccl_begin_bit + (__rccl_pass * 8usize) as u32;
                    let __rccl_num_bits =
                        (__rccl_end_bit - __rccl_current_bit).min(8u32);
                    let __rccl_pass_result = {
                        let (__rccl_input, __rccl_output) =
                            __rccl_buffers.current_and_alternate_mut();
                        __rccl_execution
                            .module()
                            .device_radix_sort_onesweep_keys::<
                                #key_type,
                                #bit_type,
                                _,
                                _,
                                _,
                                _,
                                _,
                                #onesweep_items,
                                #union_bytes,
                            >(
                                __rccl_execution.stream(),
                                &__rccl_onesweep,
                                &mut *__rccl_lookback,
                                &mut *__rccl_counters,
                                &*__rccl_bins,
                                &**__rccl_input,
                                &mut **__rccl_output,
                                __rccl_bin_offset,
                                __rccl_pass,
                                __rccl_current_bit,
                                __rccl_num_bits,
                                __rccl_default_key,
                                __rccl_to_ordered,
                                __rccl_from_ordered,
                                __rccl_digit,
                                __rccl_rank_digit,
                            )
                    };
                    __rccl_pass_result
                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                    __rccl_buffers.swap();
                    __rccl_pass += 1usize;
                }
                ::core::result::Result::Ok(())
            })()
        }},
    )
}

fn expand_device_radix_sort_keys_via_buffer(
    call: call::DeviceRadixSortKeysCall,
    descending: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let primitive = primitive_radix(&call.key_type)?;
    let max_passes = primitive.bits.div_ceil(8) as usize;
    let key_bits = primitive.bits;
    let bit_size = (primitive.bits / 8) as usize;
    let (onesweep_threads, onesweep_items) = reg_bound_policy(384, 21, bit_size);
    let tile_items = onesweep_threads * onesweep_items;
    let call::DeviceRadixSortKeysCall {
        key_type,
        target,
        execution,
        keys,
        scratch_keys,
        bins,
        lookback,
        counters,
    } = call;
    let operation = expand_device_radix_sort_keys_buffer_order(
        call::DeviceRadixSortKeysBufferCall {
            key_type,
            target,
            execution: syn::parse_quote!(__rccl_execution),
            buffers: syn::parse_quote!(&mut __rccl_buffers),
            bit_range: syn::parse_quote!(0u32..#key_bits),
            bins: syn::parse_quote!(__rccl_bins),
            lookback: syn::parse_quote!(__rccl_lookback),
            counters: syn::parse_quote!(__rccl_counters),
        },
        descending,
    )?;
    audit::generated_algorithm_expression(
        audit::Algorithm::RadixSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_keys = #keys;
            let __rccl_scratch_keys = #scratch_keys;
            let __rccl_bins = #bins;
            let __rccl_lookback = #lookback;
            let __rccl_counters = #counters;
            let __rccl_len = __rccl_keys.len();
            (|| -> ::core::result::Result<(), ::rccl_cub::DeviceAlgorithmError> {
                let __rccl_scratch_len = __rccl_scratch_keys.len();
                let mut __rccl_buffers = ::rccl_cub::DoubleBuffer::new(
                    &mut *__rccl_keys,
                    &mut *__rccl_scratch_keys,
                )
                .ok_or(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                    role: "radix scratch keys",
                    expected: __rccl_len,
                    actual: __rccl_scratch_len,
                })?;
                let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
                for (__rccl_role, __rccl_expected, __rccl_actual) in [
                    ("radix bins", #max_passes * 256usize, __rccl_bins.len()),
                    (
                        "radix lookback",
                        __rccl_blocks * 256usize,
                        __rccl_lookback.len(),
                    ),
                    (
                        "radix pass counters",
                        #max_passes,
                        __rccl_counters.len(),
                    ),
                ] {
                    if __rccl_actual != __rccl_expected {
                        return ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                role: __rccl_role,
                                expected: __rccl_expected,
                                actual: __rccl_actual,
                            },
                        );
                    }
                }
                #operation
            })()
        }},
    )
}

fn expand_device_radix_sort_pairs(
    call: call::DeviceRadixSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_via_buffer(call, false)
}

fn expand_device_radix_sort_pairs_descending(
    call: call::DeviceRadixSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_via_buffer(call, true)
}

fn expand_device_radix_sort_pairs_buffer(
    call: call::DeviceRadixSortPairsBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_buffer_order(call, false)
}

fn expand_device_radix_sort_pairs_buffer_descending(
    call: call::DeviceRadixSortPairsBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_buffer_order(call, true)
}

fn expand_device_radix_sort_pairs_buffer_order(
    call: call::DeviceRadixSortPairsBufferCall,
    descending: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let target = tmp_cuda_target(&call.target)?;
    if !(80..90).contains(&target.capability) {
        return Err(syn::Error::new_spanned(
            call.target,
            "this lowering contains the exact pinned Policy800 one-sweep graph only; another architecture must add its own macro-selected CUB policy",
        ));
    }
    let primitive = primitive_radix(&call.key_type)?;
    let lowering = primitive_radix_lowering(&call.key_type, primitive, descending)?;
    let key_type = call.key_type;
    let value_type = call.value_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let value_size = tmp_primitive_size(&value_type)?;
    let bit_type = lowering.bit_type;
    let sentinel = lowering.sentinel;
    let default_key = lowering.default_key;
    let to_ordered = lowering.to_ordered;
    let from_ordered = lowering.from_ordered;
    let digit = lowering.digit;
    let rank_digit = lowering.rank_digit;
    let execution = call.execution;
    let buffers = call.buffers;
    let bit_range = call.bit_range;
    let bins = call.bins;
    let lookback = call.lookback;
    let counters = call.counters;
    let max_passes = primitive.bits.div_ceil(8) as usize;
    let key_bits = primitive.bits;
    let histogram_items = 16usize;
    let nominal_items = if key_size == 4 { 17 } else { 21 };
    let (onesweep_threads, onesweep_items) =
        reg_bound_policy(384, nominal_items, key_size.max(value_size));
    let tile_items = onesweep_threads * onesweep_items;
    let histogram_shared_bytes = max_passes * 256usize * core::mem::size_of::<u32>();
    let union_bytes = tile_items * key_size.max(value_size);
    let shared_bytes = union_bytes + 256usize * core::mem::size_of::<u64>();
    let portion_size = ((1usize << 28) - 1) / tile_items * tile_items;
    audit::generated_algorithm_expression(
        audit::Algorithm::RadixSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_buffers = #buffers;
            let __rccl_bit_range: ::core::ops::Range<u32> = #bit_range;
            let __rccl_bins = #bins;
            let __rccl_lookback = #lookback;
            let __rccl_counters = #counters;
            let __rccl_len = __rccl_buffers.len();
            (|| -> ::core::result::Result<(), ::rccl_cub::DeviceAlgorithmError> {
                let __rccl_begin_bit = __rccl_bit_range.start;
                let __rccl_end_bit = __rccl_bit_range.end;
                if __rccl_begin_bit > __rccl_end_bit || __rccl_end_bit > #key_bits {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::InvalidBitRange {
                            begin: __rccl_begin_bit,
                            end: __rccl_end_bit,
                            key_bits: #key_bits,
                        },
                    );
                }
                if __rccl_len > #portion_size {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "Policy800 one-portion maximum",
                            expected: #portion_size,
                            actual: __rccl_len,
                        },
                    );
                }
                let __rccl_passes =
                    ((__rccl_end_bit - __rccl_begin_bit) as usize).div_ceil(8usize);
                if __rccl_len == 0 || __rccl_passes == 0 {
                    return ::core::result::Result::Ok(());
                }
                let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
                let __rccl_required_bins = __rccl_passes * 256usize;
                if __rccl_bins.len() < __rccl_required_bins {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::WorkspaceTooSmall {
                            role: "radix bins",
                            required: __rccl_required_bins,
                            actual: __rccl_bins.len(),
                        },
                    );
                }
                if __rccl_lookback.len() != __rccl_blocks * 256usize {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "radix lookback",
                            expected: __rccl_blocks * 256usize,
                            actual: __rccl_lookback.len(),
                        },
                    );
                }
                if __rccl_counters.len() < __rccl_passes {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::WorkspaceTooSmall {
                            role: "radix pass counters",
                            required: __rccl_passes,
                            actual: __rccl_counters.len(),
                        },
                    );
                }

                let __rccl_init_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_passes as u32,
                        256,
                        0,
                    );
                let __rccl_init = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_init(__rccl_init_launch)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .device_radix_sort_init(
                        __rccl_execution.stream(),
                        &__rccl_init,
                        &mut *__rccl_counters,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_default_key = #default_key;
                let __rccl_to_ordered = #to_ordered;
                let __rccl_from_ordered = #from_ordered;
                let __rccl_digit = #digit;
                let __rccl_rank_digit = #rank_digit;
                let __rccl_key_witness: #key_type = 0 as #key_type;
                let __rccl_value_witness: #value_type = ::core::default::Default::default();
                let __rccl_bit_witness: #bit_type = #sentinel;
                let __rccl_histogram_probe_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        1,
                        128,
                        #histogram_shared_bytes as u32,
                    );
                let __rccl_histogram_probe = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_histogram_for::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                        #max_passes,
                    >(
                        &__rccl_key_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_digit,
                        __rccl_histogram_probe_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                let __rccl_histogram_blocks_per_sm = __rccl_histogram_probe
                    .function()
                    .max_active_blocks_per_multiprocessor(128, #histogram_shared_bytes as u32)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
                let __rccl_multiprocessors = __rccl_execution
                    .stream()
                    .context()
                    .multiprocessor_count()
                    .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
                let __rccl_histogram_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_histogram_blocks_per_sm * __rccl_multiprocessors,
                        128,
                        #histogram_shared_bytes as u32,
                    );
                let __rccl_histogram = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_histogram_for::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                        #max_passes,
                    >(
                        &__rccl_key_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_digit,
                        __rccl_histogram_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .device_radix_sort_histogram::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                        #max_passes,
                    >(
                        __rccl_execution.stream(),
                        &__rccl_histogram,
                        &**&__rccl_buffers.current().0,
                        __rccl_begin_bit,
                        __rccl_end_bit,
                        __rccl_default_key,
                        __rccl_to_ordered,
                        __rccl_digit,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_scan_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_passes as u32,
                        256,
                        2336,
                    );
                let __rccl_scan = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_exclusive_sum::<#max_passes>(__rccl_scan_launch)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .device_radix_sort_exclusive_sum::<#max_passes>(
                        __rccl_execution.stream(),
                        &__rccl_scan,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_onesweep_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_blocks as u32,
                        #onesweep_threads as u32,
                        #shared_bytes as u32,
                    );
                let __rccl_onesweep = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_onesweep_pairs_for::<
                        #key_type,
                        #value_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        _,
                        _,
                        #onesweep_items,
                        #union_bytes,
                    >(
                        &__rccl_key_witness,
                        &__rccl_value_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_from_ordered,
                        &__rccl_digit,
                        &__rccl_rank_digit,
                        __rccl_onesweep_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                let mut __rccl_pass = 0usize;
                while __rccl_pass < __rccl_passes {
                    ::rccl_cub::__private::ZeroRadixLookback::zero_radix_lookback(
                        &mut *__rccl_lookback,
                        __rccl_execution.stream(),
                    )?;
                    let __rccl_bin_offset = __rccl_pass * 256usize;
                    let __rccl_current_bit =
                        __rccl_begin_bit + (__rccl_pass * 8usize) as u32;
                    let __rccl_num_bits =
                        (__rccl_end_bit - __rccl_current_bit).min(8u32);
                    let __rccl_pass_result = {
                        let (__rccl_input, __rccl_output) =
                            __rccl_buffers.current_and_alternate_mut();
                        let (__rccl_input_keys, __rccl_input_values) = __rccl_input;
                        let (__rccl_output_keys, __rccl_output_values) = __rccl_output;
                        __rccl_execution
                            .module()
                            .device_radix_sort_onesweep_pairs::<
                                #key_type,
                                #value_type,
                                #bit_type,
                                _,
                                _,
                                _,
                                _,
                                _,
                                #onesweep_items,
                                #union_bytes,
                            >(
                                __rccl_execution.stream(),
                                &__rccl_onesweep,
                                &mut *__rccl_lookback,
                                &mut *__rccl_counters,
                                &*__rccl_bins,
                                &**__rccl_input_keys,
                                &**__rccl_input_values,
                                &mut **__rccl_output_keys,
                                &mut **__rccl_output_values,
                                __rccl_bin_offset,
                                __rccl_pass,
                                __rccl_current_bit,
                                __rccl_num_bits,
                                __rccl_default_key,
                                __rccl_to_ordered,
                                __rccl_from_ordered,
                                __rccl_digit,
                                __rccl_rank_digit,
                            )
                    };
                    __rccl_pass_result
                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                    __rccl_buffers.swap();
                    __rccl_pass += 1usize;
                }
                ::core::result::Result::Ok(())
            })()
        }},
    )
}

fn expand_device_radix_sort_pairs_via_buffer(
    call: call::DeviceRadixSortPairsCall,
    descending: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let primitive = primitive_radix(&call.key_type)?;
    let max_passes = primitive.bits.div_ceil(8) as usize;
    let key_bits = primitive.bits;
    let key_size = tmp_primitive_size(&call.key_type)?;
    let value_size = tmp_primitive_size(&call.value_type)?;
    let nominal_items = if key_size == 4 { 17 } else { 21 };
    let (onesweep_threads, onesweep_items) =
        reg_bound_policy(384, nominal_items, key_size.max(value_size));
    let tile_items = onesweep_threads * onesweep_items;
    let call::DeviceRadixSortPairsCall {
        key_type,
        value_type,
        target,
        execution,
        keys,
        values,
        scratch_keys,
        scratch_values,
        bins,
        lookback,
        counters,
    } = call;
    let operation = expand_device_radix_sort_pairs_buffer_order(
        call::DeviceRadixSortPairsBufferCall {
            key_type,
            value_type,
            target,
            execution: syn::parse_quote!(__rccl_execution),
            buffers: syn::parse_quote!(&mut __rccl_buffers),
            bit_range: syn::parse_quote!(0u32..#key_bits),
            bins: syn::parse_quote!(__rccl_bins),
            lookback: syn::parse_quote!(__rccl_lookback),
            counters: syn::parse_quote!(__rccl_counters),
        },
        descending,
    )?;
    audit::generated_algorithm_expression(
        audit::Algorithm::RadixSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_keys = #keys;
            let __rccl_values = #values;
            let __rccl_scratch_keys = #scratch_keys;
            let __rccl_scratch_values = #scratch_values;
            let __rccl_bins = #bins;
            let __rccl_lookback = #lookback;
            let __rccl_counters = #counters;
            let __rccl_len = __rccl_keys.len();
            (|| -> ::core::result::Result<(), ::rccl_cub::DeviceAlgorithmError> {
                for (__rccl_role, __rccl_actual) in [
                    ("radix values", __rccl_values.len()),
                    ("radix scratch keys", __rccl_scratch_keys.len()),
                    ("radix scratch values", __rccl_scratch_values.len()),
                ] {
                    if __rccl_actual != __rccl_len {
                        return ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                role: __rccl_role,
                                expected: __rccl_len,
                                actual: __rccl_actual,
                            },
                        );
                    }
                }
                let mut __rccl_buffers = ::rccl_cub::DoubleBuffer::paired(
                    (&mut *__rccl_keys, &mut *__rccl_values),
                    (&mut *__rccl_scratch_keys, &mut *__rccl_scratch_values),
                )
                .ok_or(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                    role: "radix key/value buffers",
                    expected: __rccl_len,
                    actual: __rccl_len,
                })?;
                let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
                for (__rccl_role, __rccl_expected, __rccl_actual) in [
                    ("radix bins", #max_passes * 256usize, __rccl_bins.len()),
                    (
                        "radix lookback",
                        __rccl_blocks * 256usize,
                        __rccl_lookback.len(),
                    ),
                    (
                        "radix pass counters",
                        #max_passes,
                        __rccl_counters.len(),
                    ),
                ] {
                    if __rccl_actual != __rccl_expected {
                        return ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                role: __rccl_role,
                                expected: __rccl_expected,
                                actual: __rccl_actual,
                            },
                        );
                    }
                }
                #operation
            })()
        }},
    )
}

fn expand_warp_reduce(call: call::WarpReduceCall) -> syn::Result<proc_macro2::TokenStream> {
    let value_type = call.value_type;
    let scalar = warp_scalar(&value_type)?;
    let width = call.logical_warp_threads;
    let value = call.input;
    let is_addition = call::is_addition_closure(&call.op);
    let op = type_binary_closure(call.op, &value_type);
    let is_builtin_sum = call.valid_items.is_none() && is_addition && scalar.redux_sum;
    let reduction = if is_builtin_sum {
        warp_redux_sum(scalar, &value_type, &width)
    } else {
        warp_shuffle_reduction(scalar, &value_type, &width, call.valid_items.as_ref())
    };
    audit::generated_expression(quote! {{
        let __rccl_op = #op;
        let mut __rccl_aggregate: #value_type = #value;
        #reduction
    }})
}

fn expand_warp_reduce_items(
    call: call::WarpReduceItemsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let value_type = call.value_type;
    let scalar = warp_scalar(&value_type)?;
    let width = call.logical_warp_threads;
    let items_per_thread = call.items_per_thread;
    let items = call.items;
    let is_addition = call::is_addition_closure(&call.op);
    let op = type_binary_closure(call.op, &value_type);
    let reduction = if is_addition && scalar.redux_sum {
        warp_redux_sum(scalar, &value_type, &width)
    } else {
        warp_shuffle_reduction(scalar, &value_type, &width, None)
    };
    audit::generated_expression(quote! {{
        let __rccl_op = #op;
        let __rccl_items: [#value_type; #items_per_thread] = #items;
        let mut __rccl_aggregate: #value_type = __rccl_items[0];
        let mut __rccl_item = 1usize;
        while __rccl_item < #items_per_thread {
            __rccl_aggregate = __rccl_op(__rccl_aggregate, __rccl_items[__rccl_item]);
            __rccl_item += 1;
        }
        #reduction
    }})
}

fn type_binary_closure(mut operation: syn::Expr, value_type: &syn::Type) -> syn::Expr {
    if let syn::Expr::Closure(closure) = &mut operation {
        for input in &mut closure.inputs {
            if !matches!(input, syn::Pat::Type(_)) {
                let pattern = input.clone();
                *input = syn::Pat::Type(syn::PatType {
                    attrs: Vec::new(),
                    pat: Box::new(pattern),
                    colon_token: Default::default(),
                    ty: Box::new(value_type.clone()),
                });
            }
        }
    }
    operation
}

#[derive(Clone, Copy)]
struct WarpScalar {
    kind: WarpScalarKind,
    redux_sum: bool,
}

#[derive(Clone, Copy)]
enum WarpScalarKind {
    Unsigned32,
    Signed32,
    Unsigned64,
    Signed64,
    Float32,
    Float64,
}

fn warp_scalar(value_type: &syn::Type) -> syn::Result<WarpScalar> {
    let semantic_type = ungroup_type(value_type);
    let syn::Type::Path(path) = semantic_type else {
        return Err(syn::Error::new_spanned(
            value_type,
            "WarpReduce needs a macro-visible primitive scalar type",
        ));
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return Err(syn::Error::new_spanned(
            value_type,
            "WarpReduce aliases and custom values need explicit macro-visible shuffle metadata",
        ));
    }
    let (kind, redux_sum) = match path.path.segments[0].ident.to_string().as_str() {
        "u8" | "u16" | "u32" => (WarpScalarKind::Unsigned32, true),
        "i8" | "i16" | "i32" => (WarpScalarKind::Signed32, true),
        "u64" => (WarpScalarKind::Unsigned64, false),
        "i64" => (WarpScalarKind::Signed64, false),
        "f32" => (WarpScalarKind::Float32, false),
        "f64" => (WarpScalarKind::Float64, false),
        _ => {
            return Err(syn::Error::new_spanned(
                value_type,
                "WarpReduce aliases and custom values need explicit macro-visible shuffle metadata",
            ));
        }
    };
    Ok(WarpScalar { kind, redux_sum })
}

fn ungroup_type(value_type: &syn::Type) -> &syn::Type {
    match value_type {
        syn::Type::Group(group) => ungroup_type(&group.elem),
        syn::Type::Paren(paren) => ungroup_type(&paren.elem),
        other => other,
    }
}

fn warp_redux_sum(
    scalar: WarpScalar,
    value_type: &syn::Type,
    width: &syn::LitInt,
) -> proc_macro2::TokenStream {
    let result = match scalar.kind {
        WarpScalarKind::Unsigned32 => quote! {
            ::rccl_cub::__private::cuda_device::warp::redux_sync_add(
                __rccl_mask,
                __rccl_aggregate as u32,
            ) as #value_type
        },
        WarpScalarKind::Signed32 => quote! {
            ::rccl_cub::__private::cuda_device::warp::redux_sync_add(
                __rccl_mask,
                __rccl_aggregate as i32 as u32,
            ) as i32 as #value_type
        },
        WarpScalarKind::Unsigned64
        | WarpScalarKind::Signed64
        | WarpScalarKind::Float32
        | WarpScalarKind::Float64 => unreachable!("redux_sum is false for this scalar"),
    };
    quote! {{
        if false {
            let _: #value_type = __rccl_op(__rccl_aggregate, __rccl_aggregate);
        }
        let __rccl_first_lane =
            ::rccl_cub::__private::cuda_device::warp::lane_id() & !(#width - 1);
        let __rccl_mask = if #width == 32 {
            u32::MAX
        } else {
            ((1u32 << #width) - 1) << __rccl_first_lane
        };
        if #width == 1 {
            __rccl_aggregate
        } else {
            #result
        }
    }}
}

fn warp_shuffle_reduction(
    scalar: WarpScalar,
    value_type: &syn::Type,
    width: &syn::LitInt,
    valid_items: Option<&syn::Expr>,
) -> proc_macro2::TokenStream {
    let shuffled = warp_shuffle_down(scalar, value_type);
    let combine = if let Some(valid_items) = valid_items {
        quote! {
            let __rccl_valid_items = #valid_items;
            let __rccl_rank =
                ::rccl_cub::__private::cuda_device::cooperative_groups::ThreadGroup::thread_rank(
                    &__rccl_tile,
                );
            if __rccl_rank < __rccl_valid_items
                && __rccl_rank + __rccl_offset < __rccl_valid_items
            {
                __rccl_aggregate = __rccl_op(__rccl_aggregate, __rccl_other);
            }
        }
    } else {
        quote! {
            __rccl_aggregate = __rccl_op(__rccl_aggregate, __rccl_other);
        }
    };
    quote! {{
        let __rccl_tile =
            ::rccl_cub::__private::cuda_device::cooperative_groups::this_thread_block()
                .tiled_partition::<#width>();
        let mut __rccl_offset = 1u32;
        while __rccl_offset < #width {
            let __rccl_other: #value_type = #shuffled;
            #combine
            __rccl_offset <<= 1;
        }
        __rccl_aggregate
    }}
}

fn warp_shuffle_down(scalar: WarpScalar, value_type: &syn::Type) -> proc_macro2::TokenStream {
    match scalar.kind {
        WarpScalarKind::Unsigned32 | WarpScalarKind::Signed32 => quote! {
            ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down(
                &__rccl_tile,
                __rccl_aggregate as u32,
                __rccl_offset,
            ) as #value_type
        },
        WarpScalarKind::Unsigned64 | WarpScalarKind::Signed64 => quote! {{
            let __rccl_bits = __rccl_aggregate as u64;
            let __rccl_low =
                ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down(
                    &__rccl_tile,
                    __rccl_bits as u32,
                    __rccl_offset,
                ) as u64;
            let __rccl_high =
                ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down(
                    &__rccl_tile,
                    (__rccl_bits >> 32) as u32,
                    __rccl_offset,
                ) as u64;
            (__rccl_low | (__rccl_high << 32)) as #value_type
        }},
        WarpScalarKind::Float32 => quote! {
            ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down_f32(
                &__rccl_tile,
                __rccl_aggregate,
                __rccl_offset,
            )
        },
        WarpScalarKind::Float64 => quote! {{
            let __rccl_bits = __rccl_aggregate.to_bits();
            let __rccl_low =
                ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down(
                    &__rccl_tile,
                    __rccl_bits as u32,
                    __rccl_offset,
                ) as u64;
            let __rccl_high =
                ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down(
                    &__rccl_tile,
                    (__rccl_bits >> 32) as u32,
                    __rccl_offset,
                ) as u64;
            f64::from_bits(__rccl_low | (__rccl_high << 32))
        }},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_catalog_is_complete_and_valid() {
        catalog::validate().expect("the typed CUB compiler catalog must be valid");
        assert_eq!(catalog::TARGETS.len(), 6);
        assert_eq!(catalog::PRIMITIVES.len(), 59);
        assert_eq!(catalog::EXERCISED_OPERATIONS.len(), 105);
        assert!(
            catalog::PRIMITIVES
                .iter()
                .any(|primitive| primitive.name == "warp_merge_sort")
        );
        assert!(
            catalog::PRIMITIVES
                .iter()
                .any(|primitive| primitive.name == "device_radix_sort")
        );
    }

    #[test]
    fn generated_algorithm_audit_accepts_safe_rust() {
        let generated = audit::generated_rust(quote::quote! {
            fn generated_helper(value: usize) -> usize { value + 1 }
        })
        .expect("safe generated Rust must pass its audit");
        syn::parse2::<syn::File>(generated).expect("audited output must remain valid Rust");
    }

    #[test]
    fn warp_reduce_expansion_is_a_safe_rust_expression() {
        let call: call::WarpReduceCall = syn::parse_quote! {
            ::<i32, 32>(value, |left, right| left + right)
        };
        let output = expand_warp_reduce(call).expect("the operation must lower");
        let text = output.to_string();
        assert!(text.contains("warp :: redux_sync_add"));
        assert!(text.contains("let mut __rccl_aggregate : i32"));
        assert!(!text.contains("WarpReduceValue"));
        assert!(!text.contains("TypeId"));
        assert!(!text.contains("unsafe"));
        assert!(!text.contains("* mut"));
    }

    #[test]
    fn device_transform_lowering_owns_kernel_and_launch_rust() {
        let kernels = expand_device_kernels(syn::parse_quote!(__test_kernels))
            .expect("CUB must generate the transform kernel");
        let kernel_text = kernels.to_string();
        assert!(kernel_text.contains("transform_repeat_in_place"));
        assert!(kernel_text.contains("grid_stride_runs32"));

        let call: call::DeviceTransformRepeatInPlaceCall = syn::parse_quote! {
            module, stream, &mut output, repeated, |left, right| left + right
        };
        let operation = expand_device_transform_repeat_in_place(call)
            .expect("CUB must lower the transform launch");
        let operation_text = operation.to_string();
        assert!(operation_text.contains("prepare_transform_repeat_in_place_for"));
        assert!(!operation_text.contains("unsafe"));
        assert!(!operation_text.contains("* mut"));
    }

    #[test]
    fn radix_default_key_is_macro_specialization_not_runtime_data() {
        let kernels = expand_device_kernels(syn::parse_quote!(__test_kernels))
            .expect("CUB must generate the radix kernels")
            .to_string();
        assert!(kernels.contains("DefaultKey : Fn () -> B + Copy"));
        assert!(kernels.contains("default_key : DefaultKey"));
        assert!(!kernels.contains("sentinel : Uniform < B >"));

        let operation = expand_device_radix_sort_keys(syn::parse_quote! {
            ::<i32, sm_89>(
                &exec,
                &mut keys,
                &mut scratch_keys,
                &mut bins,
                &mut lookback,
                &mut counters
            )
        })
        .expect("the macro must specialize the default key")
        .to_string();
        assert!(
            operation
                .contains("let __rccl_default_key = :: rccl_cub :: __private :: radix_default_u32")
        );
        assert!(operation.contains("& __rccl_default_key"));
        assert!(!operation.contains("let __rccl_default_key = ||"));
        assert!(!operation.contains("sentinel"));
    }

    #[test]
    fn radix_keys_emit_the_exact_policy800_onesweep_graph() {
        let call: call::DeviceRadixSortKeysCall = syn::parse_quote! {
            ::<i32, sm_89>(
                &exec,
                &mut keys,
                &mut scratch_keys,
                &mut bins,
                &mut lookback,
                &mut counters
            )
        };
        let operation = expand_device_radix_sort_keys(call)
            .expect("the pinned Policy800 key graph must lower")
            .to_string();
        assert!(operation.contains("prepare_device_radix_sort_init"));
        assert!(operation.contains("prepare_device_radix_sort_histogram"));
        assert!(operation.contains("prepare_device_radix_sort_exclusive_sum"));
        assert!(operation.contains("prepare_device_radix_sort_onesweep_keys"));
        assert!(operation.contains(
            "prepare_device_radix_sort_histogram_for :: < i32 , u32 , _ , _ , _ , 16usize , 4usize , >"
        ));
        assert!(operation.contains(
            "device_radix_sort_histogram :: < i32 , u32 , _ , _ , _ , 16usize , 4usize , >"
        ));
        assert!(operation.contains(
            "prepare_device_radix_sort_onesweep_keys_for :: < i32 , u32 , _ , _ , _ , _ , _ , 21usize"
        ));
        assert!(operation.contains("let __rccl_rank_digit ="));
        assert!(operation.contains("radix_rank_digit_u32"));
        assert!(!operation.contains("| __rccl_ordered"));
        assert!(operation.contains("LaunchConfig1D :: new (__rccl_passes as u32 , 256 , 0"));
        assert!(operation.contains("128 , 4096usize as u32"));
        assert!(
            !operation.contains("device_radix_sort_histogram :: < i32 , u32 , _ , _ , 21usize")
        );
        assert_eq!(operation.matches("zero_radix_lookback").count(), 1);
        assert!(operation.contains("while __rccl_pass < __rccl_passes"));
        assert!(operation.contains("let __rccl_bin_offset = __rccl_pass * 256usize"));
        assert!(
            operation.contains(
                "let __rccl_current_bit = __rccl_begin_bit + (__rccl_pass * 8usize) as u32"
            )
        );
        assert_eq!(
            operation
                .matches("device_radix_sort_onesweep_keys ::")
                .count(),
            1,
            "one affine split borrow must replace duplicated selector branches"
        );
        assert!(operation.contains("DoubleBuffer :: new"));
        assert!(operation.contains("current_and_alternate_mut"));
        assert!(operation.contains("__rccl_buffers . swap"));
        assert!(!operation.contains("if (__rccl_pass & 1usize) == 0"));
        assert!(!operation.contains("device_merge_sort"));
        assert!(!operation.contains("DeviceBuffer :: zeroed"));
        assert!(!operation.contains("compute_capability"));
        assert!(!operation.contains("__rccl_len as u32"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));
    }

    #[test]
    fn radix_key_buffer_keeps_bit_window_and_selection_runtime() {
        let operation = expand_device_radix_sort_keys_buffer(syn::parse_quote! {
            ::<u32, sm_89>(
                &exec,
                &mut buffers,
                begin_bit..end_bit,
                &mut bins,
                &mut lookback,
                &mut counters
            )
        })
        .expect("the DoubleBuffer bit-window graph must lower")
        .to_string();

        assert!(operation.contains("Range < u32 >"));
        assert!(operation.contains("let __rccl_passes ="));
        assert!(operation.contains("div_ceil (8usize)"));
        assert!(operation.contains("__rccl_passes as u32"));
        assert!(operation.contains("__rccl_begin_bit , __rccl_end_bit"));
        assert!(operation.contains("min (8u32)"));
        assert!(operation.contains("current_and_alternate_mut"));
        assert_eq!(
            operation
                .matches("device_radix_sort_onesweep_keys ::")
                .count(),
            1
        );
        assert_eq!(operation.matches("__rccl_buffers . swap").count(), 1);
        let launch_check = operation
            .rfind("map_err (:: rccl_cub :: DeviceAlgorithmError :: Launch) ?")
            .expect("launch failure must be propagated");
        let swap = operation
            .rfind("__rccl_buffers . swap")
            .expect("a successful pass must update selection");
        assert!(launch_check < swap);
        assert!(operation.contains("InvalidBitRange"));
        assert!(operation.contains("WorkspaceTooSmall"));
        assert!(!operation.contains("#max_passes"));
        assert!(!operation.contains("selector"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));
    }

    #[test]
    fn radix_policy800_reg_bound_scaling_is_macro_selected_for_key_width() {
        assert_eq!(reg_bound_policy(384, 21, 4), (384, 21));
        assert_eq!(reg_bound_policy(384, 21, 8), (384, 10));

        let call: call::DeviceRadixSortKeysCall = syn::parse_quote! {
            ::<u64, sm_89>(
                &exec,
                &mut keys,
                &mut scratch_keys,
                &mut bins,
                &mut lookback,
                &mut counters
            )
        };
        let operation = expand_device_radix_sort_keys(call)
            .expect("the pinned 64-bit Policy800 graph must lower")
            .to_string();
        assert!(operation.contains(
            "prepare_device_radix_sort_onesweep_keys_for :: < u64 , u64 , _ , _ , _ , _ , _ , 10usize"
        ));
        assert!(operation.contains("let __rccl_blocks = __rccl_len . div_ceil (3840usize)"));
        assert!(operation.contains("while __rccl_pass < __rccl_passes"));
        assert!(
            operation.contains("LaunchConfig1D :: new (__rccl_blocks as u32 , 384usize as u32")
        );
        assert!(operation.contains("128 , 8192usize as u32"));
        assert!(!operation.contains("compute_capability"));

        let workspace_call: call::DeviceRadixSortWorkspaceCall =
            syn::parse_quote! { ::<u64, sm_89>(len) };
        let workspace = expand_device_radix_sort_workspace(workspace_call)
            .expect("the 64-bit workspace must use the same macro-selected tile")
            .to_string();
        assert!(workspace.contains("let __rccl_blocks = __rccl_len . div_ceil (3840usize)"));
    }

    #[test]
    fn radix_pairs_emit_the_exact_policy800_pair_graph_and_workspace() {
        let call: call::DeviceRadixSortPairsCall = syn::parse_quote! {
            ::<f32, i32, sm_89>(
                &exec,
                &mut keys,
                &mut values,
                &mut scratch_keys,
                &mut scratch_values,
                &mut bins,
                &mut lookback,
                &mut counters
            )
        };
        let operation = expand_device_radix_sort_pairs(call)
            .expect("the pinned Policy800 pair graph must lower")
            .to_string();
        assert!(operation.contains("prepare_device_radix_sort_histogram"));
        assert!(operation.contains("prepare_device_radix_sort_exclusive_sum"));
        assert!(operation.contains(
            "prepare_device_radix_sort_onesweep_pairs_for :: < f32 , i32 , u32 , _ , _ , _ , _ , _ , 17usize , 26112usize"
        ));
        assert!(operation.contains("let __rccl_blocks = __rccl_len . div_ceil (6528usize)"));
        assert!(operation.contains("28160usize as u32"));
        assert!(operation.contains("radix_float_digit_u32"));
        assert!(operation.contains("radix_float_rank_digit_u32"));
        assert!(!operation.contains("| __rccl_ordered"));
        assert_eq!(
            operation
                .matches("device_radix_sort_onesweep_pairs ::")
                .count(),
            1
        );
        assert!(operation.contains("DoubleBuffer :: paired"));
        assert!(operation.contains("current_and_alternate_mut"));
        assert!(!operation.contains("device_merge_sort"));
        assert!(!operation.contains("DeviceBuffer :: zeroed"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));

        let workspace = expand_device_radix_sort_pairs_workspace(syn::parse_quote! {
            ::<f32, i32, sm_89>(len)
        })
        .expect("pair workspace must use the same macro-selected tile")
        .to_string();
        assert!(workspace.contains("scratch_values : __rccl_len"));
        assert!(workspace.contains("div_ceil (6528usize)"));
    }

    #[test]
    fn radix_pair_buffer_has_one_structural_selector() {
        let operation = expand_device_radix_sort_pairs_buffer(syn::parse_quote! {
            ::<u32, i32, sm_89>(
                &exec,
                &mut buffers,
                3u32..19u32,
                &mut bins,
                &mut lookback,
                &mut counters
            )
        })
        .expect("the paired DoubleBuffer graph must lower")
        .to_string();

        assert!(operation.contains("current_and_alternate_mut"));
        assert!(operation.contains("let (__rccl_input_keys , __rccl_input_values)"));
        assert!(operation.contains("let (__rccl_output_keys , __rccl_output_values)"));
        assert_eq!(
            operation
                .matches("device_radix_sort_onesweep_pairs ::")
                .count(),
            1
        );
        assert_eq!(operation.matches("__rccl_buffers . swap").count(), 1);
        assert!(!operation.contains("key_buffers"));
        assert!(!operation.contains("value_buffers"));
        assert!(!operation.contains("selector"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));
    }

    #[test]
    fn merge_sort_keys_emit_the_exact_block_partition_merge_topology() {
        let call: call::DeviceMergeSortKeysCall = syn::parse_quote! {
            ::<u32, sm_89>(
                &exec,
                &mut keys,
                &mut scratch_keys,
                &mut partitions,
                |left: &u32, right: &u32| left.cmp(right)
            )
        };
        let operation = expand_device_merge_sort_keys(call)
            .expect("the pinned SM60+ merge-sort policy must lower")
            .to_string();
        assert!(operation.contains("prepare_device_merge_sort_block_keys_256"));
        assert!(operation.contains("device_merge_sort_block_keys_in_place_256"));
        assert!(
            operation.contains(
                "prepare_device_merge_sort_partition :: < u32 , _ , 256usize , 17usize >"
            )
        );
        assert!(operation.contains("prepare_device_merge_sort_merge_keys_256"));
        assert!(operation.contains("17424usize as u32"));
        assert!(!operation.contains("DeviceBuffer :: zeroed"));
        assert!(!operation.contains("device_merge_sort_keys_pass"));
        assert!(!operation.contains("compute_capability"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));

        let count: call::DeviceMergeSortPartitionCountCall =
            syn::parse_quote! { ::<u32, sm_89>(len) };
        let count = expand_device_merge_sort_partition_count(count)
            .expect("partition storage must use the same macro-selected tile")
            .to_string();
        assert!(count.contains("div_ceil (4352usize)"));

        let padded_call: call::DeviceMergeSortKeysCall = syn::parse_quote! {
            ::<u64, sm_89>(
                &exec,
                &mut keys,
                &mut scratch_keys,
                &mut partitions,
                |left: &u64, right: &u64| left.cmp(right)
            )
        };
        let padded = expand_device_merge_sort_keys(padded_call)
            .expect("u64 must lower through the padded Policy600 BlockExchange path")
            .to_string();
        assert!(
            padded
                .contains("prepare_device_merge_sort_partition :: < u64 , _ , 256usize , 8usize >")
        );
        assert!(padded.contains("16912usize as u32"));
    }

    #[test]
    fn merge_sort_pairs_emit_the_exact_agents_and_borrow_all_workspace() {
        let call: call::DeviceMergeSortPairsCall = syn::parse_quote! {
            ::<u32, u32, sm_89>(
                &exec,
                &mut keys,
                &mut values,
                &mut scratch_keys,
                &mut scratch_values,
                &mut partitions,
                |left: &u32, right: &u32| left.cmp(right)
            )
        };
        let operation = expand_device_merge_sort_pairs(call)
            .expect("the pinned key/value graph must lower")
            .to_string();
        assert!(operation.contains("prepare_device_merge_sort_block_pairs_256"));
        assert!(operation.contains("device_merge_sort_block_pairs_in_place_256"));
        assert!(operation.contains("prepare_device_merge_sort_partition"));
        assert!(operation.contains("prepare_device_merge_sort_merge_pairs_256"));
        assert!(operation.contains("17424usize as u32"));
        assert!(!operation.contains("DeviceBuffer :: zeroed"));
        assert!(!operation.contains("device_merge_sort_pairs_pass"));
        assert!(!operation.contains("compute_capability"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));
    }

    #[test]
    fn merge_tmp_layout_selects_exact_cccl_nominal_items_and_aliases_fail_closed() {
        let keys: call::DeviceMergeKeysCall = syn::parse_quote! {
            ::<u32, sm_89>(
                &exec, &first, &second, &mut partitions, &mut output,
                |left, right| left.cmp(right)
            )
        };
        let key_text = expand_device_merge_keys(keys)
            .expect("primitive layout must select the pinned merge policies")
            .to_string();
        assert!(key_text.contains("prepare_device_partition_merge_path :: < u32 , _ , 15usize >"));
        assert!(key_text.contains(
            "prepare_device_merge_keys_agent :: < u32 , _ , 15usize , 4usize , false , 30736usize >"
        ));
        assert!(key_text.contains("30736usize as u32"));
        assert!(!key_text.contains("prepare_device_merge_keys ::"));
        assert!(!key_text.contains("compute_capability"));

        let sm52_keys: call::DeviceMergeKeysCall = syn::parse_quote! {
            ::<u32, sm_52>(
                &exec, &first, &second, &mut partitions, &mut output,
                |left, right| left.cmp(right)
            )
        };
        let sm52_text = expand_device_merge_keys(sm52_keys)
            .expect("sm_52 must select the pinned fallback policy at expansion time")
            .to_string();
        assert!(sm52_text.contains("prepare_device_partition_merge_path :: < u32 , _ , 13usize >"));
        assert!(sm52_text.contains(
            "prepare_device_merge_keys_agent :: < u32 , _ , 13usize , 4usize , false , 26640usize >"
        ));
        assert!(!sm52_text.contains("compute_capability"));

        let pairs: call::DeviceMergePairsCall = syn::parse_quote! {
            ::<i32, u64, sm_89>(
                &exec,
                &first_keys,
                &first_values,
                &second_keys,
                &second_values,
                &mut partitions,
                &mut output_keys,
                &mut output_values,
                |left, right| left.cmp(right),
            )
        };
        let pair_text = expand_device_merge_pairs(pairs)
            .expect("combined key/value size must be TMP input")
            .to_string();
        assert!(pair_text.contains("prepare_device_partition_merge_path :: < i32 , _ , 5usize >"));
        assert!(pair_text.contains(
            "prepare_device_merge_pairs_agent :: < i32 , u64 , _ , 5usize , 4usize , 8usize , false , false , 20496usize >"
        ));
        assert!(pair_text.contains("20496usize as u32"));

        let async_keys: call::DeviceMergeKeysCall = syn::parse_quote! {
            ::<u8, sm_89>(
                &exec, &first, &second, &mut partitions, &mut output,
                |left, right| left.cmp(right)
            )
        };
        let async_key_text = expand_device_merge_keys(async_keys)
            .expect("the SM80 byte-key policy must select BlockLoadToShared")
            .to_string();
        assert!(async_key_text.contains(
            "prepare_device_merge_keys_agent :: < u8 , _ , 15usize , 1usize , true , 7760usize >"
        ));
        assert!(async_key_text.contains("7760usize as u32"));

        let async_pairs: call::DeviceMergePairsCall = syn::parse_quote! {
            ::<i32, u8, sm_89>(
                &exec,
                &first_keys,
                &first_values,
                &second_keys,
                &second_values,
                &mut partitions,
                &mut output_keys,
                &mut output_values,
                |left, right| left.cmp(right),
            )
        };
        let async_pair_text = expand_device_merge_pairs(async_pairs)
            .expect("the SM80 i32/u8 policy must stage both arrays through shared memory")
            .to_string();
        assert!(async_pair_text.contains(
            "prepare_device_merge_pairs_agent :: < i32 , u8 , _ , 12usize , 4usize , 1usize , true , true , 24656usize >"
        ));
        assert!(async_pair_text.contains("24656usize as u32"));

        let padded_pairs: call::DeviceMergePairsCall = syn::parse_quote! {
            ::<u16, u32, sm_52>(
                &exec,
                &first_keys,
                &first_values,
                &second_keys,
                &second_values,
                &mut partitions,
                &mut output_keys,
                &mut output_values,
                |left, right| left.cmp(right),
            )
        };
        let padded_pair_text = expand_device_merge_pairs(padded_pairs)
            .expect("the SM52 pair policy must retain BlockExchange padding")
            .to_string();
        assert!(padded_pair_text.contains(
            "prepare_device_merge_pairs_agent :: < u16 , u32 , _ , 8usize , 2usize , 4usize , false , false , 16912usize >"
        ));
        assert!(padded_pair_text.contains("16912usize as u32"));

        let alias: call::DeviceMergeKeysCall = syn::parse_quote! {
            ::<Key, sm_89>(
                &exec, &first, &second, &mut partitions, &mut output,
                |left, right| left.cmp(right)
            )
        };
        let error = expand_device_merge_keys(alias)
            .expect_err("an opaque alias must not silently receive an invented CUB schedule");
        assert!(error.to_string().contains("macro-visible size/alignment"));
    }
}
