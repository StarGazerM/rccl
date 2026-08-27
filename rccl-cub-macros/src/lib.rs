mod audit;
mod call;
#[cfg(test)]
mod catalog;

use proc_macro::TokenStream;
use quote::quote;

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

fn radix_digit_order(
    key_type: &syn::Type,
    radix: PrimitiveRadix,
    shift: u32,
    digit_bits: u32,
    descending: bool,
) -> syn::Expr {
    let value_mask = if radix.bits == 64 {
        u64::MAX
    } else {
        (1u64 << radix.bits) - 1
    };
    let sign_mask = if matches!(
        radix.kind,
        PrimitiveRadixKind::Signed | PrimitiveRadixKind::Floating
    ) {
        1u64 << (radix.bits - 1)
    } else {
        0
    };
    let digit_mask = (1u64 << digit_bits) - 1;
    let encode_left = match radix.kind {
        PrimitiveRadixKind::Unsigned => {
            quote!(((*__rccl_left as u64) & #value_mask))
        }
        PrimitiveRadixKind::Signed => {
            quote!((((*__rccl_left as u64) & #value_mask) ^ #sign_mask))
        }
        PrimitiveRadixKind::Floating => quote!({
            let __rccl_bits = __rccl_left.to_bits() as u64;
            let __rccl_bits = if (__rccl_bits & (#sign_mask - 1)) == 0 {
                0
            } else {
                __rccl_bits
            };
            if (__rccl_bits & #sign_mask) == 0 {
                __rccl_bits ^ #sign_mask
            } else {
                (!__rccl_bits) & #value_mask
            }
        }),
    };
    let encode_right = match radix.kind {
        PrimitiveRadixKind::Unsigned => {
            quote!(((*__rccl_right as u64) & #value_mask))
        }
        PrimitiveRadixKind::Signed => {
            quote!((((*__rccl_right as u64) & #value_mask) ^ #sign_mask))
        }
        PrimitiveRadixKind::Floating => quote!({
            let __rccl_bits = __rccl_right.to_bits() as u64;
            let __rccl_bits = if (__rccl_bits & (#sign_mask - 1)) == 0 {
                0
            } else {
                __rccl_bits
            };
            if (__rccl_bits & #sign_mask) == 0 {
                __rccl_bits ^ #sign_mask
            } else {
                (!__rccl_bits) & #value_mask
            }
        }),
    };
    if descending {
        syn::parse_quote! {
            |__rccl_left: &#key_type, __rccl_right: &#key_type| {
                let __rccl_left_encoded = #encode_left;
                let __rccl_right_encoded = #encode_right;
                let __rccl_left_digit = (__rccl_left_encoded >> #shift) & #digit_mask;
                let __rccl_right_digit = (__rccl_right_encoded >> #shift) & #digit_mask;
                __rccl_right_digit.cmp(&__rccl_left_digit)
            }
        }
    } else {
        syn::parse_quote! {
            |__rccl_left: &#key_type, __rccl_right: &#key_type| {
                let __rccl_left_encoded = #encode_left;
                let __rccl_right_encoded = #encode_right;
                let __rccl_left_digit = (__rccl_left_encoded >> #shift) & #digit_mask;
                let __rccl_right_digit = (__rccl_right_encoded >> #shift) & #digit_mask;
                __rccl_left_digit.cmp(&__rccl_right_digit)
            }
        }
    }
}

fn sequence_operations(
    operations: impl DoubleEndedIterator<Item = proc_macro2::TokenStream>,
) -> proc_macro2::TokenStream {
    operations.rev().fold(
        quote!(::core::result::Result::Ok(())),
        |continuation, operation| {
            quote! {
                match #operation {
                    ::core::result::Result::Err(__rccl_error) => {
                        ::core::result::Result::Err(__rccl_error)
                    }
                    ::core::result::Result::Ok(()) => #continuation,
                }
            }
        },
    )
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
                DisjointSlice, LinearTiles, kernel, launch_bounds, launch_contract, thread,
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
            fn stable_left_partition<K, F>(
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
                let first_taken = stable_left_partition(
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
            #[launch_bounds(512)]
            #[launch_contract(domain = 1, coordinates = u32, block = (512, 1, 1))]
            pub fn device_merge_keys<K, F, const ITEMS: usize>(
                first: &[K],
                second: &[K],
                order: F,
                mut output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let thread_index = thread::index_1d_u32(launch_context);
                let mut output_index = thread_index.get() as usize * ITEMS;
                let mut runs = output.grid_stride_runs32(thread_index);
                let period = runs.period() as usize * ITEMS;
                while let Some(mut run) = runs.next_run() {
                    let mut first_taken = stable_left_partition(
                        first,
                        0,
                        first.len(),
                        second,
                        0,
                        second.len(),
                        output_index,
                        &order,
                    );
                    let mut second_taken = output_index - first_taken;
                    let mut local = 0u32;
                    while (local as usize) < ITEMS {
                        let Some(mut slot) = run.at(local) else {
                            break;
                        };
                        let take_first = first_taken < first.len()
                            && (second_taken == second.len()
                                || !ordered_before(
                                    &order,
                                    &second[second_taken],
                                    &first[first_taken],
                                ));
                        if take_first {
                            slot.write(first[first_taken]);
                            first_taken += 1;
                        } else {
                            slot.write(second[second_taken]);
                            second_taken += 1;
                        }
                        local += 1;
                    }
                    output_index = output_index.wrapping_add(period);
                }
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(domain = 1, coordinates = u32, block = (512, 1, 1))]
            pub fn device_merge_pairs<K, V, F, const ITEMS: usize>(
                first_keys: &[K],
                first_values: &[V],
                second_keys: &[K],
                second_values: &[V],
                order: F,
                mut output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
                mut output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let key_thread = thread::index_1d_u32(launch_context);
                let value_thread = thread::index_1d_u32(launch_context);
                let mut output_index = key_thread.get() as usize * ITEMS;
                let mut key_runs = output_keys.grid_stride_runs32(key_thread);
                let mut value_runs = output_values.grid_stride_runs32(value_thread);
                let period = key_runs.period() as usize * ITEMS;
                while let Some(mut key_run) = key_runs.next_run() {
                    let Some(mut value_run) = value_runs.next_run() else {
                        return;
                    };
                    let mut first_taken = stable_left_partition(
                        first_keys,
                        0,
                        first_keys.len(),
                        second_keys,
                        0,
                        second_keys.len(),
                        output_index,
                        &order,
                    );
                    let mut second_taken = output_index - first_taken;
                    let mut local = 0u32;
                    while (local as usize) < ITEMS {
                        let Some(mut key_slot) = key_run.at(local) else {
                            break;
                        };
                        let Some(mut value_slot) = value_run.at(local) else {
                            break;
                        };
                        let take_first = first_taken < first_keys.len()
                            && (second_taken == second_keys.len()
                                || !ordered_before(
                                    &order,
                                    &second_keys[second_taken],
                                    &first_keys[first_taken],
                                ));
                        if take_first {
                            key_slot.write(first_keys[first_taken]);
                            value_slot.write(first_values[first_taken]);
                            first_taken += 1;
                        } else {
                            key_slot.write(second_keys[second_taken]);
                            value_slot.write(second_values[second_taken]);
                            second_taken += 1;
                        }
                        local += 1;
                    }
                    output_index = output_index.wrapping_add(period);
                }
            }

            #[inline(always)]
            fn merge_sort_source<K, F>(
                source: &[K],
                output_index: usize,
                run_length: usize,
                order: &F,
            ) -> usize
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let pair_length = run_length.saturating_mul(2);
                let pair_begin = (output_index / pair_length) * pair_length;
                let first_end_candidate = pair_begin.saturating_add(run_length);
                let pair_end_candidate = pair_begin.saturating_add(pair_length);
                let first_end = if first_end_candidate < source.len() {
                    first_end_candidate
                } else {
                    source.len()
                };
                let pair_end = if pair_end_candidate < source.len() {
                    pair_end_candidate
                } else {
                    source.len()
                };
                let diagonal = output_index - pair_begin;
                let (from_first, source_index) = merged_source(
                    source,
                    pair_begin,
                    first_end - pair_begin,
                    source,
                    first_end,
                    pair_end - first_end,
                    diagonal,
                    order,
                );
                let _ = from_first;
                source_index
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn device_merge_sort_keys_pass<K, F>(
                source: &[K],
                run_length: usize,
                order: F,
                mut output: DisjointSlice<K, LinearTiles<1>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let thread_index = thread::index_1d_u32(launch_context);
                let mut output_index = thread_index.get() as usize;
                let mut runs = output.grid_stride_runs32(thread_index);
                let period = runs.period() as usize;
                while let Some(mut run) = runs.next_run() {
                    if let Some(mut slot) = run.at(0) {
                        let source_index =
                            merge_sort_source(source, output_index, run_length, &order);
                        slot.write(source[source_index]);
                    }
                    output_index = output_index.wrapping_add(period);
                }
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn device_merge_sort_pairs_pass<K, V, F>(
                source_keys: &[K],
                source_values: &[V],
                run_length: usize,
                order: F,
                mut output_keys: DisjointSlice<K, LinearTiles<1>>,
                mut output_values: DisjointSlice<V, LinearTiles<1>>,
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let key_thread = thread::index_1d_u32(launch_context);
                let value_thread = thread::index_1d_u32(launch_context);
                let mut output_index = key_thread.get() as usize;
                let mut key_runs = output_keys.grid_stride_runs32(key_thread);
                let mut value_runs = output_values.grid_stride_runs32(value_thread);
                let period = key_runs.period() as usize;
                while let Some(mut key_run) = key_runs.next_run() {
                    let Some(mut value_run) = value_runs.next_run() else {
                        return;
                    };
                    if let Some(mut key_slot) = key_run.at(0) {
                        if let Some(mut value_slot) = value_run.at(0) {
                            let source_index =
                                merge_sort_source(source_keys, output_index, run_length, &order);
                            key_slot.write(source_keys[source_index]);
                            value_slot.write(source_values[source_index]);
                        }
                    }
                    output_index = output_index.wrapping_add(period);
                }
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

fn merge_launch(
    output_len: proc_macro2::TokenStream,
    threads: u32,
    items_per_thread: usize,
) -> proc_macro2::TokenStream {
    quote! {
        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
            (#output_len)
                .div_ceil(#items_per_thread)
                .div_ceil(#threads as usize)
                .min(u32::MAX as usize) as u32,
            #threads,
            0,
        )
    }
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

fn nominal_4b_items(nominal: usize, item_size: usize) -> usize {
    (nominal * 4 / item_size).clamp(1, nominal)
}

fn merge_keys_launch(key_type: &syn::Type, items: usize) -> proc_macro2::TokenStream {
    let launch = merge_launch(quote!(__rccl_output_len), 512, items);
    quote! {{
        let __rccl_launch = #launch;
        match __rccl_execution
            .module()
            .prepare_device_merge_keys::<#key_type, _, #items>(__rccl_launch)
        {
            ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
            ),
            ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                .module()
                .device_merge_keys::<#key_type, _, #items>(
                    __rccl_execution.stream(),
                    &__rccl_prepared,
                    __rccl_first,
                    __rccl_second,
                    __rccl_order,
                    __rccl_output,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
        }
    }}
}

fn merge_pairs_launch(
    key_type: &syn::Type,
    value_type: &syn::Type,
    items: usize,
) -> proc_macro2::TokenStream {
    let launch = merge_launch(quote!(__rccl_output_len), 512, items);
    quote! {{
        let __rccl_launch = #launch;
        match __rccl_execution
            .module()
            .prepare_device_merge_pairs::<#key_type, #value_type, _, #items>(__rccl_launch)
        {
            ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
            ),
            ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                .module()
                .device_merge_pairs::<#key_type, #value_type, _, #items>(
                    __rccl_execution.stream(),
                    &__rccl_prepared,
                    __rccl_first_keys,
                    __rccl_first_values,
                    __rccl_second_keys,
                    __rccl_second_values,
                    __rccl_order,
                    __rccl_output_keys,
                    __rccl_output_values,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
        }
    }}
}

fn expand_device_merge_keys(
    call: call::DeviceMergeKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let sm52_items = nominal_4b_items(13, key_size);
    let sm60_items = nominal_4b_items(15, key_size);
    let sm52_launch = merge_keys_launch(&key_type, sm52_items);
    let sm60_launch = merge_keys_launch(&key_type, sm60_items);
    let execution = call.execution;
    let first = call.first;
    let second = call.second;
    let output = call.output;
    let order = call.order;
    audit::generated_expression(quote! {{
        let __rccl_execution = #execution;
        let __rccl_first = #first;
        let __rccl_second = #second;
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
                match __rccl_execution.stream().context().compute_capability() {
                    ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::Driver(__rccl_error),
                    ),
                    ::core::result::Result::Ok((__rccl_major, _)) if __rccl_major >= 6 => {
                        #sm60_launch
                    }
                    ::core::result::Result::Ok(_) => #sm52_launch,
                }
            }
        }
    }})
}

fn expand_device_merge_pairs(
    call: call::DeviceMergePairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let tune_size = tmp_primitive_size(&key_type)?
        .checked_add(tmp_primitive_size(&value_type)?)
        .ok_or_else(|| syn::Error::new_spanned(&value_type, "merge tuning size overflowed"))?;
    let sm52_items = nominal_4b_items(13, tune_size);
    let sm60_items = nominal_4b_items(15, tune_size);
    let sm52_launch = merge_pairs_launch(&key_type, &value_type, sm52_items);
    let sm60_launch = merge_pairs_launch(&key_type, &value_type, sm60_items);
    let execution = call.execution;
    let first_keys = call.first_keys;
    let first_values = call.first_values;
    let second_keys = call.second_keys;
    let second_values = call.second_values;
    let output_keys = call.output_keys;
    let output_values = call.output_values;
    let order = call.order;
    audit::generated_expression(quote! {{
        let __rccl_execution = #execution;
        let __rccl_first_keys = #first_keys;
        let __rccl_first_values = #first_values;
        let __rccl_second_keys = #second_keys;
        let __rccl_second_values = #second_values;
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
                    match __rccl_execution.stream().context().compute_capability() {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Driver(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok((__rccl_major, _)) if __rccl_major >= 6 => {
                            #sm60_launch
                        }
                        ::core::result::Result::Ok(_) => #sm52_launch,
                    }
                }
            }
        }
    }})
}

fn expand_device_merge_sort_keys(
    call: call::DeviceMergeSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let execution = call.execution;
    let keys = call.keys;
    let order = call.order;
    let launch = merge_launch(quote!(__rccl_len), 256, 1);
    audit::generated_expression(quote! {{
        let __rccl_execution = #execution;
        let __rccl_keys = #keys;
        let __rccl_order = #order;
        let __rccl_len = __rccl_keys.len();
        if __rccl_len < 2 {
            ::core::result::Result::Ok(())
        } else {
            match ::rccl_cub::__private::cuda_core::DeviceBuffer::zeroed(
                __rccl_execution.stream(),
                __rccl_len,
            ) {
                ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::Driver(__rccl_error),
                ),
                ::core::result::Result::Ok(mut __rccl_scratch) => {
                    let __rccl_launch = #launch;
                    let mut __rccl_run_length = 1usize;
                    let mut __rccl_source_is_keys = true;
                    let mut __rccl_result = ::core::result::Result::Ok(());
                    while __rccl_run_length < __rccl_len && __rccl_result.is_ok() {
                        let __rccl_prepared = __rccl_execution
                            .module()
                            .prepare_device_merge_sort_keys_pass(__rccl_launch);
                        __rccl_result = match __rccl_prepared {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => {
                                let __rccl_launch_result = if __rccl_source_is_keys {
                                    __rccl_execution.module().device_merge_sort_keys_pass(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_keys,
                                        __rccl_run_length,
                                        __rccl_order,
                                        &mut __rccl_scratch,
                                    )
                                } else {
                                    __rccl_execution.module().device_merge_sort_keys_pass(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &__rccl_scratch,
                                        __rccl_run_length,
                                        __rccl_order,
                                        &mut *__rccl_keys,
                                    )
                                };
                                __rccl_launch_result
                                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                            }
                        };
                        __rccl_source_is_keys = !__rccl_source_is_keys;
                        __rccl_run_length = __rccl_run_length
                            .checked_mul(2)
                            .unwrap_or(__rccl_len);
                    }
                    if __rccl_result.is_ok() && !__rccl_source_is_keys {
                        let __rccl_prepared = __rccl_execution
                            .module()
                            .prepare_device_merge_sort_keys_pass(__rccl_launch);
                        __rccl_result = match __rccl_prepared {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                                .module()
                                .device_merge_sort_keys_pass(
                                    __rccl_execution.stream(),
                                    &__rccl_prepared,
                                    &__rccl_scratch,
                                    __rccl_len,
                                    __rccl_order,
                                    &mut *__rccl_keys,
                                )
                                .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                        };
                    }
                    __rccl_result
                }
            }
        }
    }})
}

fn expand_device_merge_sort_keys_copy(
    call: call::DeviceMergeSortKeysCopyCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let execution = call.execution;
    let input = call.input;
    let output = call.output;
    let order = call.order;
    let launch = merge_launch(quote!(__rccl_len), 256, 1);
    audit::generated_expression(quote! {{
        let __rccl_execution = #execution;
        let __rccl_input = #input;
        let __rccl_output = #output;
        let __rccl_order = #order;
        let __rccl_len = __rccl_input.len();
        if __rccl_output.len() != __rccl_len {
            ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                role: "output keys",
                expected: __rccl_len,
                actual: __rccl_output.len(),
            })
        } else if __rccl_len == 0 {
            ::core::result::Result::Ok(())
        } else {
            match ::rccl_cub::__private::cuda_core::DeviceBuffer::zeroed(
                __rccl_execution.stream(),
                __rccl_len,
            ) {
                ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::Driver(__rccl_error),
                ),
                ::core::result::Result::Ok(mut __rccl_scratch) => {
                    let __rccl_launch = #launch;
                    let __rccl_prepared = __rccl_execution
                        .module()
                        .prepare_device_merge_sort_keys_pass(__rccl_launch);
                    let mut __rccl_result = match __rccl_prepared {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                            .module()
                            .device_merge_sort_keys_pass(
                                __rccl_execution.stream(),
                                &__rccl_prepared,
                                __rccl_input,
                                1usize,
                                __rccl_order,
                                &mut *__rccl_output,
                            )
                            .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                    };
                    let mut __rccl_run_length = 2usize;
                    let mut __rccl_source_is_output = true;
                    while __rccl_run_length < __rccl_len && __rccl_result.is_ok() {
                        let __rccl_prepared = __rccl_execution
                            .module()
                            .prepare_device_merge_sort_keys_pass(__rccl_launch);
                        __rccl_result = match __rccl_prepared {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => {
                                let __rccl_launch_result = if __rccl_source_is_output {
                                    __rccl_execution.module().device_merge_sort_keys_pass(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_output,
                                        __rccl_run_length,
                                        __rccl_order,
                                        &mut __rccl_scratch,
                                    )
                                } else {
                                    __rccl_execution.module().device_merge_sort_keys_pass(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &__rccl_scratch,
                                        __rccl_run_length,
                                        __rccl_order,
                                        &mut *__rccl_output,
                                    )
                                };
                                __rccl_launch_result
                                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                            }
                        };
                        __rccl_source_is_output = !__rccl_source_is_output;
                        __rccl_run_length = __rccl_run_length
                            .checked_mul(2)
                            .unwrap_or(__rccl_len);
                    }
                    if __rccl_result.is_ok() && !__rccl_source_is_output {
                        let __rccl_prepared = __rccl_execution
                            .module()
                            .prepare_device_merge_sort_keys_pass(__rccl_launch);
                        __rccl_result = match __rccl_prepared {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                                .module()
                                .device_merge_sort_keys_pass(
                                    __rccl_execution.stream(),
                                    &__rccl_prepared,
                                    &__rccl_scratch,
                                    __rccl_len,
                                    __rccl_order,
                                    &mut *__rccl_output,
                                )
                                .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                        };
                    }
                    __rccl_result
                }
            }
        }
    }})
}

fn expand_device_merge_sort_pairs(
    call: call::DeviceMergeSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let execution = call.execution;
    let keys = call.keys;
    let values = call.values;
    let order = call.order;
    let launch = merge_launch(quote!(__rccl_len), 256, 1);
    audit::generated_expression(quote! {{
        let __rccl_execution = #execution;
        let __rccl_keys = #keys;
        let __rccl_values = #values;
        let __rccl_order = #order;
        let __rccl_len = __rccl_keys.len();
        if __rccl_values.len() != __rccl_len {
            ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                role: "values",
                expected: __rccl_len,
                actual: __rccl_values.len(),
            })
        } else if __rccl_len < 2 {
            ::core::result::Result::Ok(())
        } else {
            match ::rccl_cub::__private::cuda_core::DeviceBuffer::zeroed(
                __rccl_execution.stream(),
                __rccl_len,
            ) {
                ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::Driver(__rccl_error),
                ),
                ::core::result::Result::Ok(mut __rccl_key_scratch) => {
                    match ::rccl_cub::__private::cuda_core::DeviceBuffer::zeroed(
                        __rccl_execution.stream(),
                        __rccl_len,
                    ) {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Driver(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(mut __rccl_value_scratch) => {
                            let __rccl_launch = #launch;
                            let mut __rccl_run_length = 1usize;
                            let mut __rccl_source_is_input = true;
                            let mut __rccl_result = ::core::result::Result::Ok(());
                            while __rccl_run_length < __rccl_len && __rccl_result.is_ok() {
                                let __rccl_prepared = __rccl_execution
                                    .module()
                                    .prepare_device_merge_sort_pairs_pass(__rccl_launch);
                                __rccl_result = match __rccl_prepared {
                                    ::core::result::Result::Err(__rccl_error) => {
                                        ::core::result::Result::Err(
                                            ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                        )
                                    }
                                    ::core::result::Result::Ok(__rccl_prepared) => {
                                        let __rccl_launch_result = if __rccl_source_is_input {
                                            __rccl_execution.module().device_merge_sort_pairs_pass(
                                                __rccl_execution.stream(),
                                                &__rccl_prepared,
                                                &*__rccl_keys,
                                                &*__rccl_values,
                                                __rccl_run_length,
                                                __rccl_order,
                                                &mut __rccl_key_scratch,
                                                &mut __rccl_value_scratch,
                                            )
                                        } else {
                                            __rccl_execution.module().device_merge_sort_pairs_pass(
                                                __rccl_execution.stream(),
                                                &__rccl_prepared,
                                                &__rccl_key_scratch,
                                                &__rccl_value_scratch,
                                                __rccl_run_length,
                                                __rccl_order,
                                                &mut *__rccl_keys,
                                                &mut *__rccl_values,
                                            )
                                        };
                                        __rccl_launch_result
                                            .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                                    }
                                };
                                __rccl_source_is_input = !__rccl_source_is_input;
                                __rccl_run_length = __rccl_run_length
                                    .checked_mul(2)
                                    .unwrap_or(__rccl_len);
                            }
                            if __rccl_result.is_ok() && !__rccl_source_is_input {
                                let __rccl_prepared = __rccl_execution
                                    .module()
                                    .prepare_device_merge_sort_pairs_pass(__rccl_launch);
                                __rccl_result = match __rccl_prepared {
                                    ::core::result::Result::Err(__rccl_error) => {
                                        ::core::result::Result::Err(
                                            ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                        )
                                    }
                                    ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                                        .module()
                                        .device_merge_sort_pairs_pass(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &__rccl_key_scratch,
                                            &__rccl_value_scratch,
                                            __rccl_len,
                                            __rccl_order,
                                            &mut *__rccl_keys,
                                            &mut *__rccl_values,
                                        )
                                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                                };
                            }
                            __rccl_result
                        }
                    }
                }
            }
        }
    }})
}

fn expand_device_merge_sort_pairs_copy(
    call: call::DeviceMergeSortPairsCopyCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let execution = call.execution;
    let input_keys = call.input_keys;
    let input_values = call.input_values;
    let output_keys = call.output_keys;
    let output_values = call.output_values;
    let order = call.order;
    let launch = merge_launch(quote!(__rccl_len), 256, 1);
    audit::generated_expression(quote! {{
        let __rccl_execution = #execution;
        let __rccl_input_keys = #input_keys;
        let __rccl_input_values = #input_values;
        let __rccl_output_keys = #output_keys;
        let __rccl_output_values = #output_values;
        let __rccl_order = #order;
        let __rccl_len = __rccl_input_keys.len();
        let __rccl_shape_error = if __rccl_input_values.len() != __rccl_len {
            ::core::option::Option::Some((
                "input values",
                __rccl_len,
                __rccl_input_values.len(),
            ))
        } else if __rccl_output_keys.len() != __rccl_len {
            ::core::option::Option::Some((
                "output keys",
                __rccl_len,
                __rccl_output_keys.len(),
            ))
        } else if __rccl_output_values.len() != __rccl_len {
            ::core::option::Option::Some((
                "output values",
                __rccl_len,
                __rccl_output_values.len(),
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
        } else if __rccl_len == 0 {
            ::core::result::Result::Ok(())
        } else {
            match ::rccl_cub::__private::cuda_core::DeviceBuffer::zeroed(
                __rccl_execution.stream(),
                __rccl_len,
            ) {
                ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::Driver(__rccl_error),
                ),
                ::core::result::Result::Ok(mut __rccl_key_scratch) => {
                    match ::rccl_cub::__private::cuda_core::DeviceBuffer::zeroed(
                        __rccl_execution.stream(),
                        __rccl_len,
                    ) {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Driver(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(mut __rccl_value_scratch) => {
                            let __rccl_launch = #launch;
                            let __rccl_prepared = __rccl_execution
                                .module()
                                .prepare_device_merge_sort_pairs_pass(__rccl_launch);
                            let mut __rccl_result = match __rccl_prepared {
                                ::core::result::Result::Err(__rccl_error) => {
                                    ::core::result::Result::Err(
                                        ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                    )
                                }
                                ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                                    .module()
                                    .device_merge_sort_pairs_pass(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        __rccl_input_keys,
                                        __rccl_input_values,
                                        1usize,
                                        __rccl_order,
                                        &mut *__rccl_output_keys,
                                        &mut *__rccl_output_values,
                                    )
                                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                            };
                            let mut __rccl_run_length = 2usize;
                            let mut __rccl_source_is_output = true;
                            while __rccl_run_length < __rccl_len && __rccl_result.is_ok() {
                                let __rccl_prepared = __rccl_execution
                                    .module()
                                    .prepare_device_merge_sort_pairs_pass(__rccl_launch);
                                __rccl_result = match __rccl_prepared {
                                    ::core::result::Result::Err(__rccl_error) => {
                                        ::core::result::Result::Err(
                                            ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                        )
                                    }
                                    ::core::result::Result::Ok(__rccl_prepared) => {
                                        let __rccl_launch_result = if __rccl_source_is_output {
                                            __rccl_execution.module().device_merge_sort_pairs_pass(
                                                __rccl_execution.stream(),
                                                &__rccl_prepared,
                                                &*__rccl_output_keys,
                                                &*__rccl_output_values,
                                                __rccl_run_length,
                                                __rccl_order,
                                                &mut __rccl_key_scratch,
                                                &mut __rccl_value_scratch,
                                            )
                                        } else {
                                            __rccl_execution.module().device_merge_sort_pairs_pass(
                                                __rccl_execution.stream(),
                                                &__rccl_prepared,
                                                &__rccl_key_scratch,
                                                &__rccl_value_scratch,
                                                __rccl_run_length,
                                                __rccl_order,
                                                &mut *__rccl_output_keys,
                                                &mut *__rccl_output_values,
                                            )
                                        };
                                        __rccl_launch_result
                                            .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                                    }
                                };
                                __rccl_source_is_output = !__rccl_source_is_output;
                                __rccl_run_length = __rccl_run_length
                                    .checked_mul(2)
                                    .unwrap_or(__rccl_len);
                            }
                            if __rccl_result.is_ok() && !__rccl_source_is_output {
                                let __rccl_prepared = __rccl_execution
                                    .module()
                                    .prepare_device_merge_sort_pairs_pass(__rccl_launch);
                                __rccl_result = match __rccl_prepared {
                                    ::core::result::Result::Err(__rccl_error) => {
                                        ::core::result::Result::Err(
                                            ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                        )
                                    }
                                    ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                                        .module()
                                        .device_merge_sort_pairs_pass(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &__rccl_key_scratch,
                                            &__rccl_value_scratch,
                                            __rccl_len,
                                            __rccl_order,
                                            &mut *__rccl_output_keys,
                                            &mut *__rccl_output_values,
                                        )
                                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                                };
                            }
                            __rccl_result
                        }
                    }
                }
            }
        }
    }})
}

fn expand_device_radix_sort_keys(
    call: call::DeviceRadixSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_order(call, false)
}

fn expand_device_radix_sort_keys_descending(
    call: call::DeviceRadixSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_order(call, true)
}

fn expand_device_radix_sort_keys_order(
    call: call::DeviceRadixSortKeysCall,
    descending: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let radix = primitive_radix(&key_type)?;
    let execution = call.execution;
    let keys = call.keys;
    let mut operations = Vec::new();
    let mut shift = 0;
    while shift < radix.bits {
        let digit_bits = (radix.bits - shift).min(8);
        let order = radix_digit_order(&key_type, radix, shift, digit_bits, descending);
        operations.push(expand_device_merge_sort_keys(
            call::DeviceMergeSortKeysCall {
                key_type: key_type.clone(),
                execution: syn::parse_quote!(__rccl_execution),
                keys: syn::parse_quote!(&mut *__rccl_keys),
                order,
            },
        )?);
        shift += digit_bits;
    }
    let operations = sequence_operations(operations.into_iter());
    audit::generated_expression(quote! {{
        let __rccl_execution = #execution;
        let __rccl_keys = #keys;
        #operations
    }})
}

fn expand_device_radix_sort_pairs(
    call: call::DeviceRadixSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_order(call, false)
}

fn expand_device_radix_sort_pairs_descending(
    call: call::DeviceRadixSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_order(call, true)
}

fn expand_device_radix_sort_pairs_order(
    call: call::DeviceRadixSortPairsCall,
    descending: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let radix = primitive_radix(&key_type)?;
    let execution = call.execution;
    let keys = call.keys;
    let values = call.values;
    let mut operations = Vec::new();
    let mut shift = 0;
    while shift < radix.bits {
        let digit_bits = (radix.bits - shift).min(8);
        let order = radix_digit_order(&key_type, radix, shift, digit_bits, descending);
        operations.push(expand_device_merge_sort_pairs(
            call::DeviceMergeSortPairsCall {
                key_type: key_type.clone(),
                value_type: value_type.clone(),
                execution: syn::parse_quote!(__rccl_execution),
                keys: syn::parse_quote!(&mut *__rccl_keys),
                values: syn::parse_quote!(&mut *__rccl_values),
                order,
            },
        )?);
        shift += digit_bits;
    }
    let operations = sequence_operations(operations.into_iter());
    audit::generated_expression(quote! {{
        let __rccl_execution = #execution;
        let __rccl_keys = #keys;
        let __rccl_values = #values;
        #operations
    }})
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
    fn signed_radix_tmp_is_resolved_by_the_macro_and_emits_only_safe_rust() {
        let call: call::DeviceRadixSortKeysCall = syn::parse_quote! { ::<i32>(&exec, &mut keys) };
        let operation = expand_device_radix_sort_keys(call)
            .expect("the explicit primitive TMP argument must select signed radix lowering");
        let operation_text = operation.to_string();

        // i32 ordering is obtained by flipping its sign bit before each stable
        // digit pass. Four occurrences prove the 32-bit/8-bit pass tree was
        // chosen by this proc macro, not by runtime type dispatch.
        assert_eq!(operation_text.matches("2147483648").count(), 8);
        assert_eq!(operation_text.matches("let __rccl_execution").count(), 5);
        assert!(operation_text.contains("prepare_device_merge_sort_keys_pass"));
        assert!(!operation_text.contains("TypeId"));
        assert!(!operation_text.contains("unsafe"));
        assert!(!operation_text.contains("* mut"));
    }

    #[test]
    fn floating_radix_tmp_emits_cccl_bit_transform_and_zero_normalization() {
        let call: call::DeviceRadixSortPairsCall = syn::parse_quote! {
            ::<f32, i32>(&exec, &mut keys, &mut values)
        };
        let operation = expand_device_radix_sort_pairs(call)
            .expect("the explicit f32 TMP argument must select floating radix lowering");
        let operation_text = operation.to_string();

        // Two encoders in each of four stable 8-bit passes prove that f32
        // selected a 32-bit radix tree during macro expansion.
        assert_eq!(operation_text.matches("to_bits").count(), 8);
        assert!(operation_text.contains("2147483648"));
        assert!(operation_text.contains("4294967295"));
        assert!(operation_text.contains("prepare_device_merge_sort_pairs_pass"));
        assert!(!operation_text.contains("TypeId"));
        assert!(!operation_text.contains("unsafe"));
        assert!(!operation_text.contains("* mut"));
    }

    #[test]
    fn merge_tmp_layout_selects_exact_cccl_nominal_items_and_aliases_fail_closed() {
        let keys: call::DeviceMergeKeysCall = syn::parse_quote! {
            ::<u32>(&exec, &first, &second, &mut output, |left, right| left.cmp(right))
        };
        let key_text = expand_device_merge_keys(keys)
            .expect("primitive layout must select the pinned merge policies")
            .to_string();
        assert!(key_text.contains("prepare_device_merge_keys :: < u32 , _ , 13usize >"));
        assert!(key_text.contains("prepare_device_merge_keys :: < u32 , _ , 15usize >"));
        assert!(key_text.contains("compute_capability"));

        let pairs: call::DeviceMergePairsCall = syn::parse_quote! {
            ::<i32, u64>(
                &exec,
                &first_keys,
                &first_values,
                &second_keys,
                &second_values,
                &mut output_keys,
                &mut output_values,
                |left, right| left.cmp(right),
            )
        };
        let pair_text = expand_device_merge_pairs(pairs)
            .expect("combined key/value size must be TMP input")
            .to_string();
        assert!(pair_text.contains("prepare_device_merge_pairs :: < i32 , u64 , _ , 4usize >"));
        assert!(pair_text.contains("prepare_device_merge_pairs :: < i32 , u64 , _ , 5usize >"));

        let alias: call::DeviceMergeKeysCall = syn::parse_quote! {
            ::<Key>(&exec, &first, &second, &mut output, |left, right| left.cmp(right))
        };
        let error = expand_device_merge_keys(alias)
            .expect_err("an opaque alias must not silently receive an invented CUB schedule");
        assert!(error.to_string().contains("macro-visible size/alignment"));
    }
}
