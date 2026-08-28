mod audit;
mod call;
#[cfg(test)]
mod catalog;

use proc_macro::TokenStream;
use quote::quote;

#[proc_macro]
pub fn thrust_program(input: TokenStream) -> TokenStream {
    let context = syn::parse_macro_input!(input as syn::Expr);
    expand_thrust_program(context)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro]
pub fn transform(input: TokenStream) -> TokenStream {
    let call = syn::parse_macro_input!(input as call::TransformRepeatInPlaceCall);
    expand_transform(call)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

macro_rules! algorithm_macro {
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

algorithm_macro!(sort, call::SortCall, expand_sort);
algorithm_macro!(sort_by, call::SortByCall, expand_sort_by);
algorithm_macro!(stable_sort, call::SortCall, expand_stable_sort);
algorithm_macro!(stable_sort_by, call::SortByCall, expand_stable_sort_by);
algorithm_macro!(sort_by_key, call::SortByKeyCall, expand_sort_by_key);
algorithm_macro!(
    sort_by_key_with,
    call::SortByKeyWithCall,
    expand_sort_by_key_with
);
algorithm_macro!(
    stable_sort_by_key,
    call::SortByKeyCall,
    expand_stable_sort_by_key
);
algorithm_macro!(
    stable_sort_by_key_with,
    call::StableSortByKeyCall,
    expand_stable_sort_by_key_with
);
algorithm_macro!(merge, call::MergeCall, expand_merge);
algorithm_macro!(merge_by, call::MergeByCall, expand_merge_by);
algorithm_macro!(merge_by_key, call::MergeByKeyCall, expand_merge_by_key);
algorithm_macro!(
    merge_by_key_with,
    call::MergeByKeyWithCall,
    expand_merge_by_key_with
);

fn expand_thrust_program(context: syn::Expr) -> syn::Result<proc_macro2::TokenStream> {
    let module = syn::parse_quote!(__rccl_thrust_kernels);
    audit::generated_program_expression(
        quote! {{
            ::rccl_cub::__private::__device_kernels!(__rccl_thrust_kernels);
            let __rccl_context = #context;
            unsafe { __rccl_thrust_kernels::load(__rccl_context) }.map(|__rccl_module| {
                ::rccl_thrust::cuda::Program::__from_module(
                    __rccl_context.as_ref(),
                    __rccl_module,
                )
            })
        }},
        &module,
    )
}

fn expand_transform(
    call: call::TransformRepeatInPlaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let execution = call.execution;
    let output = call.output;
    let repeated = call.repeated;
    let op = call.op;
    audit::generated_expression(quote! {{
        let __rccl_execution = #execution;
        let __rccl_output = #output;
        let __rccl_repeated = #repeated;
        let __rccl_op = #op;
        let __rccl_output_len = ::rccl_thrust::DeviceRange::len(&*__rccl_output);
        if ::rccl_thrust::DeviceRange::len(&__rccl_repeated) != __rccl_output_len {
            ::core::result::Result::Err(::rccl_thrust::TransformError::LengthMismatch {
                input: ::rccl_thrust::DeviceRange::len(&__rccl_repeated),
                output: __rccl_output_len,
            })
        } else if __rccl_output_len == 0 {
            ::core::result::Result::Ok(())
        } else {
            ::rccl_cub::__private::__device_transform_repeat_in_place!(
                __rccl_execution.module(),
                __rccl_execution.stream(),
                __rccl_output,
                *__rccl_repeated.value(),
                __rccl_op,
            )
            .map_err(::rccl_thrust::TransformError::Launch)
        }
    }})
}

fn natural_order(key_type: &syn::Type) -> proc_macro2::TokenStream {
    quote! {
        |__rccl_left: &#key_type, __rccl_right: &#key_type| {
            if __rccl_left < __rccl_right {
                ::core::cmp::Ordering::Less
            } else if __rccl_right < __rccl_left {
                ::core::cmp::Ordering::Greater
            } else {
                ::core::cmp::Ordering::Equal
            }
        }
    }
}

fn synchronous_cub(
    execution: syn::Expr,
    operation: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    audit::generated_expression(quote! {{
        let __rccl_execution = #execution;
        #operation
    }})
}

fn expand_sort(call: call::SortCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let execution = call.execution;
    let range = call.range;
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_range = #range;
            let __rccl_len = ::rccl_thrust::DeviceRange::len(&*__rccl_range);
            let __rccl_workspace =
                ::rccl_cub::__private::__device_radix_sort_workspace_counts!(::<#key_type>(__rccl_len));
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace4::<_, #key_type, u64, i32, i32>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [
                        __rccl_workspace.scratch_keys,
                        __rccl_workspace.bins,
                        __rccl_workspace.lookback,
                        __rccl_workspace.counters,
                    ],
                    [128usize, 256usize, 256usize, 256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (__rccl_scratch_keys, __rccl_bins, __rccl_lookback, __rccl_counters) =
                __rccl_temporary.parts_mut();
            let mut __rccl_storage =
                ::rccl_cub::DeviceRadixSortKeysWorkspace::<#key_type, _, _, _, _>::__from_parts(
                __rccl_scratch_keys,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            );
            let __rccl_result = ::rccl_cub::device_radix_sort_keys!(::<#key_type>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_storage,
            ));
            ::core::mem::drop(__rccl_storage);
            ::core::mem::drop(__rccl_temporary);
            __rccl_execution.__synchronize(__rccl_result)
        })()},
    )
}

fn expand_sort_by(call: call::SortByCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let execution = call.execution;
    let range = call.range;
    let compare = call.compare;
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_range = #range;
            let __rccl_compare = #compare;
            let __rccl_len = ::rccl_thrust::DeviceRange::len(&*__rccl_range);
            let __rccl_partition_count =
                ::rccl_cub::__private::__device_merge_sort_partition_count!(::<#key_type>(
                    __rccl_len
                ))?;
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace2::<_, usize, #key_type>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count, __rccl_len],
                    [256usize, 256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions, __rccl_scratch_keys) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                ::rccl_cub::DeviceMergeSortKeysWorkspace::<#key_type, _, _>::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_partitions,
                );
            __rccl_execution.__synchronize(::rccl_cub::device_merge_sort_keys!(::<#key_type>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_workspace,
                __rccl_compare,
            )))
        })()},
    )
}

fn expand_stable_sort(call: call::SortCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let execution = call.execution;
    let range = call.range;
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_range = #range;
            let __rccl_len = ::rccl_thrust::DeviceRange::len(&*__rccl_range);
            let __rccl_workspace =
                ::rccl_cub::__private::__device_radix_sort_workspace_counts!(::<#key_type>(__rccl_len));
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace4::<_, #key_type, u64, i32, i32>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [
                        __rccl_workspace.scratch_keys,
                        __rccl_workspace.bins,
                        __rccl_workspace.lookback,
                        __rccl_workspace.counters,
                    ],
                    [128usize, 256usize, 256usize, 256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (__rccl_scratch_keys, __rccl_bins, __rccl_lookback, __rccl_counters) =
                __rccl_temporary.parts_mut();
            let mut __rccl_storage =
                ::rccl_cub::DeviceRadixSortKeysWorkspace::<#key_type, _, _, _, _>::__from_parts(
                __rccl_scratch_keys,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            );
            let __rccl_result = ::rccl_cub::device_radix_sort_keys!(::<#key_type>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_storage,
            ));
            ::core::mem::drop(__rccl_storage);
            ::core::mem::drop(__rccl_temporary);
            __rccl_execution.__synchronize(__rccl_result)
        })()},
    )
}

fn expand_stable_sort_by(call: call::SortByCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let execution = call.execution;
    let range = call.range;
    let compare = call.compare;
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_range = #range;
            let __rccl_compare = #compare;
            let __rccl_len = ::rccl_thrust::DeviceRange::len(&*__rccl_range);
            let __rccl_partition_count =
                ::rccl_cub::__private::__device_merge_sort_partition_count!(::<#key_type>(
                    __rccl_len
                ))?;
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace2::<_, usize, #key_type>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count, __rccl_len],
                    [256usize, 256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions, __rccl_scratch_keys) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                ::rccl_cub::DeviceMergeSortKeysWorkspace::<#key_type, _, _>::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_partitions,
                );
            __rccl_execution.__synchronize(::rccl_cub::device_stable_merge_sort_keys!(::<#key_type>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_workspace,
                __rccl_compare,
            )))
        })()},
    )
}

fn expand_sort_by_key(call: call::SortByKeyCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let execution = call.execution;
    let keys = call.keys;
    let values = call.values;
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_keys = #keys;
            let __rccl_values = #values;
            let __rccl_len = ::rccl_thrust::DeviceRange::len(&*__rccl_keys);
            let __rccl_workspace =
                ::rccl_cub::__private::__device_radix_sort_pairs_workspace_counts!(::<
                    #key_type,
                    #value_type
                >(__rccl_len));
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace5::<
                    _, #key_type, #value_type, u64, i32, i32
                >::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [
                        __rccl_workspace.scratch_keys,
                        __rccl_workspace.scratch_values,
                        __rccl_workspace.bins,
                        __rccl_workspace.lookback,
                        __rccl_workspace.counters,
                    ],
                    [128usize, 128usize, 256usize, 256usize, 256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (
                __rccl_scratch_keys,
                __rccl_scratch_values,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            ) = __rccl_temporary.parts_mut();
            let mut __rccl_storage =
                ::rccl_cub::DeviceRadixSortPairsWorkspace::<
                    #key_type, #value_type, _, _, _, _, _
                >::__from_parts(
                __rccl_scratch_keys,
                __rccl_scratch_values,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            );
            let __rccl_result = ::rccl_cub::device_radix_sort_pairs!(::<#key_type, #value_type>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_storage,
            ));
            ::core::mem::drop(__rccl_storage);
            ::core::mem::drop(__rccl_temporary);
            __rccl_execution.__synchronize(__rccl_result)
        })()},
    )
}

fn expand_sort_by_key_with(call: call::SortByKeyWithCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let execution = call.execution;
    let keys = call.keys;
    let values = call.values;
    let compare = call.compare;
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_keys = #keys;
            let __rccl_values = #values;
            let __rccl_compare = #compare;
            let __rccl_len = ::rccl_thrust::DeviceRange::len(&*__rccl_keys);
            let __rccl_partition_count =
                ::rccl_cub::__private::__device_merge_sort_partition_count!(::<#key_type>(
                    __rccl_len
                ))?;
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace3::<_, usize, #key_type, #value_type>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count, __rccl_len, __rccl_len],
                    [256usize, 256usize, 256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions, __rccl_scratch_keys, __rccl_scratch_values) =
                __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                ::rccl_cub::DeviceMergeSortPairsWorkspace::<#key_type, #value_type, _, _, _>::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_scratch_values,
                    __rccl_partitions,
                );
            __rccl_execution.__synchronize(::rccl_cub::device_merge_sort_pairs!(::<#key_type, #value_type>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_workspace,
                __rccl_compare,
            )))
        })()},
    )
}

fn expand_stable_sort_by_key(call: call::SortByKeyCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let execution = call.execution;
    let keys = call.keys;
    let values = call.values;
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_keys = #keys;
            let __rccl_values = #values;
            let __rccl_len = ::rccl_thrust::DeviceRange::len(&*__rccl_keys);
            let __rccl_workspace =
                ::rccl_cub::__private::__device_radix_sort_pairs_workspace_counts!(::<
                    #key_type,
                    #value_type
                >(__rccl_len));
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace5::<
                    _, #key_type, #value_type, u64, i32, i32
                >::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [
                        __rccl_workspace.scratch_keys,
                        __rccl_workspace.scratch_values,
                        __rccl_workspace.bins,
                        __rccl_workspace.lookback,
                        __rccl_workspace.counters,
                    ],
                    [128usize, 128usize, 256usize, 256usize, 256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (
                __rccl_scratch_keys,
                __rccl_scratch_values,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            ) = __rccl_temporary.parts_mut();
            let mut __rccl_storage =
                ::rccl_cub::DeviceRadixSortPairsWorkspace::<
                    #key_type, #value_type, _, _, _, _, _
                >::__from_parts(
                __rccl_scratch_keys,
                __rccl_scratch_values,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            );
            let __rccl_result = ::rccl_cub::device_radix_sort_pairs!(::<#key_type, #value_type>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_storage,
            ));
            ::core::mem::drop(__rccl_storage);
            ::core::mem::drop(__rccl_temporary);
            __rccl_execution.__synchronize(__rccl_result)
        })()},
    )
}

fn expand_stable_sort_by_key_with(
    call: call::StableSortByKeyCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let execution = call.execution;
    let keys = call.keys;
    let values = call.values;
    let compare = call.compare;
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_keys = #keys;
            let __rccl_values = #values;
            let __rccl_compare = #compare;
            let __rccl_len = ::rccl_thrust::DeviceRange::len(&*__rccl_keys);
            let __rccl_partition_count =
                ::rccl_cub::__private::__device_merge_sort_partition_count!(::<#key_type>(
                    __rccl_len
                ))?;
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace3::<_, usize, #key_type, #value_type>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count, __rccl_len, __rccl_len],
                    [256usize, 256usize, 256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions, __rccl_scratch_keys, __rccl_scratch_values) =
                __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                ::rccl_cub::DeviceMergeSortPairsWorkspace::<#key_type, #value_type, _, _, _>::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_scratch_values,
                    __rccl_partitions,
                );
            __rccl_execution.__synchronize(::rccl_cub::device_stable_merge_sort_pairs!(::<#key_type, #value_type>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_workspace,
                __rccl_compare,
            )))
        })()},
    )
}

fn expand_merge(call: call::MergeCall) -> syn::Result<proc_macro2::TokenStream> {
    let value_type = call.value_type;
    let execution = call.execution;
    let first = call.first;
    let second = call.second;
    let output = call.output;
    let order = natural_order(&value_type);
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_first = #first;
            let __rccl_second = #second;
            let __rccl_output = #output;
            let __rccl_partition_count =
                ::rccl_cub::__private::__device_merge_keys_partition_count!(::<#value_type>(
                    ::rccl_thrust::DeviceRange::len(&__rccl_first),
                    ::rccl_thrust::DeviceRange::len(&__rccl_second),
                ))?;
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace1::<_, usize>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count],
                    [256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions,) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                ::rccl_cub::DeviceMergeWorkspace::<_>::__from_partitions(__rccl_partitions);
            __rccl_execution.__synchronize(::rccl_cub::device_merge_keys!(::<#value_type>(
                __rccl_execution,
                __rccl_first,
                __rccl_second,
                &mut __rccl_workspace,
                __rccl_output,
                #order,
            )))
        })()},
    )
}

fn expand_merge_by(call: call::MergeByCall) -> syn::Result<proc_macro2::TokenStream> {
    let value_type = call.value_type;
    let execution = call.execution;
    let first = call.first;
    let second = call.second;
    let output = call.output;
    let compare = call.compare;
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_first = #first;
            let __rccl_second = #second;
            let __rccl_output = #output;
            let __rccl_compare = #compare;
            let __rccl_partition_count =
                ::rccl_cub::__private::__device_merge_keys_partition_count!(::<#value_type>(
                    ::rccl_thrust::DeviceRange::len(&__rccl_first),
                    ::rccl_thrust::DeviceRange::len(&__rccl_second),
                ))?;
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace1::<_, usize>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count],
                    [256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions,) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                ::rccl_cub::DeviceMergeWorkspace::<_>::__from_partitions(__rccl_partitions);
            __rccl_execution.__synchronize(::rccl_cub::device_merge_keys!(::<#value_type>(
                __rccl_execution,
                __rccl_first,
                __rccl_second,
                &mut __rccl_workspace,
                __rccl_output,
                __rccl_compare,
            )))
        })()},
    )
}

fn expand_merge_by_key(call: call::MergeByKeyCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let execution = call.execution;
    let first_keys = call.first_keys;
    let first_values = call.first_values;
    let second_keys = call.second_keys;
    let second_values = call.second_values;
    let output_keys = call.output_keys;
    let output_values = call.output_values;
    let order = natural_order(&key_type);
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_first_keys = #first_keys;
            let __rccl_first_values = #first_values;
            let __rccl_second_keys = #second_keys;
            let __rccl_second_values = #second_values;
            let __rccl_output_keys = #output_keys;
            let __rccl_output_values = #output_values;
            let __rccl_partition_count =
                ::rccl_cub::__private::__device_merge_pairs_partition_count!(::<
                    #key_type,
                    #value_type
                >(
                    ::rccl_thrust::DeviceRange::len(&__rccl_first_keys),
                    ::rccl_thrust::DeviceRange::len(&__rccl_second_keys),
                ))?;
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace1::<_, usize>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count],
                    [256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions,) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                ::rccl_cub::DeviceMergeWorkspace::<_>::__from_partitions(__rccl_partitions);
            __rccl_execution.__synchronize(::rccl_cub::device_merge_pairs!(::<#key_type, #value_type>(
                __rccl_execution,
                __rccl_first_keys,
                __rccl_first_values,
                __rccl_second_keys,
                __rccl_second_values,
                &mut __rccl_workspace,
                __rccl_output_keys,
                __rccl_output_values,
                #order,
            )))
        })()},
    )
}

fn expand_merge_by_key_with(
    call: call::MergeByKeyWithCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let execution = call.execution;
    let first_keys = call.first_keys;
    let first_values = call.first_values;
    let second_keys = call.second_keys;
    let second_values = call.second_values;
    let output_keys = call.output_keys;
    let output_values = call.output_values;
    let compare = call.compare;
    synchronous_cub(
        execution,
        quote! {(|| -> ::core::result::Result<
            (),
            ::rccl_thrust::DeviceAlgorithmError,
        > {
            let __rccl_first_keys = #first_keys;
            let __rccl_first_values = #first_values;
            let __rccl_second_keys = #second_keys;
            let __rccl_second_values = #second_values;
            let __rccl_output_keys = #output_keys;
            let __rccl_output_values = #output_values;
            let __rccl_compare = #compare;
            let __rccl_partition_count =
                ::rccl_cub::__private::__device_merge_pairs_partition_count!(::<
                    #key_type,
                    #value_type
                >(
                    ::rccl_thrust::DeviceRange::len(&__rccl_first_keys),
                    ::rccl_thrust::DeviceRange::len(&__rccl_second_keys),
                ))?;
            let mut __rccl_temporary =
                ::rccl_thrust::memory::TemporaryWorkspace1::<_, usize>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count],
                    [256usize],
                )
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions,) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                ::rccl_cub::DeviceMergeWorkspace::<_>::__from_partitions(__rccl_partitions);
            __rccl_execution.__synchronize(::rccl_cub::device_merge_pairs!(::<#key_type, #value_type>(
                __rccl_execution,
                __rccl_first_keys,
                __rccl_first_values,
                __rccl_second_keys,
                __rccl_second_values,
                &mut __rccl_workspace,
                __rccl_output_keys,
                __rccl_output_values,
                __rccl_compare,
            )))
        })()},
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_catalog_is_complete_and_valid() {
        catalog::validate().expect("the typed Thrust compiler catalog must be valid");
        assert_eq!(catalog::ADAPTORS.len(), 19);
        assert_eq!(catalog::ALGORITHMS.len(), 34);
        assert!(
            catalog::ALGORITHMS
                .iter()
                .any(|algorithm| algorithm.name == "sort")
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
    fn transform_expansion_is_safe_and_keeps_rust_borrows() {
        let call: call::TransformRepeatInPlaceCall = syn::parse_quote! {
            &exec, &mut values, repeated, |left, right| left + right
        };
        let output = expand_transform(call).expect("transform must lower");
        let text = output.to_string();
        assert!(text.contains("__device_transform_repeat_in_place"));
        assert!(text.contains("& mut values"));
        assert!(!text.contains("unsafe"));
        assert!(!text.contains("* mut"));
    }

    #[test]
    fn every_ordering_algorithm_lowers_through_its_cub_operation() {
        let expansions = [
            (
                "sort",
                expand_sort(syn::parse_quote!(::<i32>(
                    &exec,
                    &mut keys
                )))
                .expect("sort must lower"),
                "device_radix_sort_keys",
            ),
            (
                "sort_by",
                expand_sort_by(syn::parse_quote!(
                    ::<i32>(
                        &exec,
                        &mut keys,
                        |left: &i32, right: &i32| left.cmp(right)
                    )
                ))
                .expect("sort_by must lower"),
                "device_merge_sort_keys",
            ),
            (
                "stable_sort",
                expand_stable_sort(syn::parse_quote!(::<i32>(
                    &exec,
                    &mut keys
                )))
                .expect("stable_sort must lower"),
                "device_radix_sort_keys",
            ),
            (
                "stable_sort_by",
                expand_stable_sort_by(syn::parse_quote!(
                    ::<i32>(
                        &exec,
                        &mut keys,
                        |left: &i32, right: &i32| left.cmp(right)
                    )
                ))
                .expect("stable_sort_by must lower"),
                "device_stable_merge_sort_keys",
            ),
            (
                "sort_by_key",
                expand_sort_by_key(syn::parse_quote!(::<i32, u64>(
                    &exec,
                    &mut keys,
                    &mut values
                )))
                .expect("sort_by_key must lower"),
                "device_radix_sort_pairs",
            ),
            (
                "sort_by_key_with",
                expand_sort_by_key_with(syn::parse_quote!(
                    ::<i32, u64>(
                        &exec,
                        &mut keys,
                        &mut values,
                        |left: &i32, right: &i32| left.cmp(right)
                    )
                ))
                .expect("sort_by_key_with must lower"),
                "device_merge_sort_pairs",
            ),
            (
                "stable_sort_by_key",
                expand_stable_sort_by_key(syn::parse_quote!(::<i32, u64>(
                    &exec,
                    &mut keys,
                    &mut values
                )))
                .expect("stable_sort_by_key must lower"),
                "device_radix_sort_pairs",
            ),
            (
                "stable_sort_by_key_with",
                expand_stable_sort_by_key_with(syn::parse_quote!(
                    ::<i32, u64>(
                        &exec,
                        &mut keys,
                        &mut values,
                        |left: &i32, right: &i32| left.cmp(right)
                    )
                ))
                .expect("stable_sort_by_key_with must lower"),
                "device_stable_merge_sort_pairs",
            ),
            (
                "merge",
                expand_merge(syn::parse_quote!(::<i32>(
                    &exec,
                    &first,
                    &second,
                    &mut output
                )))
                .expect("merge must lower"),
                "device_merge_keys",
            ),
            (
                "merge_by",
                expand_merge_by(syn::parse_quote!(
                    ::<i32>(
                        &exec,
                        &first,
                        &second,
                        &mut output,
                        |left: &i32, right: &i32| left.cmp(right)
                    )
                ))
                .expect("merge_by must lower"),
                "device_merge_keys",
            ),
            (
                "merge_by_key",
                expand_merge_by_key(syn::parse_quote!(
                    ::<i32, u64>(
                        &exec,
                        &first_keys,
                        &first_values,
                        &second_keys,
                        &second_values,
                        &mut output_keys,
                        &mut output_values
                    )
                ))
                .expect("merge_by_key must lower"),
                "device_merge_pairs",
            ),
            (
                "merge_by_key_with",
                expand_merge_by_key_with(syn::parse_quote!(
                    ::<i32, u64>(
                        &exec,
                        &first_keys,
                        &first_values,
                        &second_keys,
                        &second_values,
                        &mut output_keys,
                        &mut output_values,
                        |left: &i32, right: &i32| left.cmp(right)
                    )
                ))
                .expect("merge_by_key_with must lower"),
                "device_merge_pairs",
            ),
        ];

        for (name, expansion, cub_operation) in expansions {
            let text = expansion.to_string();
            assert!(
                text.contains(&format!("rccl_cub :: {cub_operation} !")),
                "{name} bypassed its CUB lowering: {text}",
            );
            assert!(text.contains("synchronize"), "{name} lost Thrust sync");
            assert_eq!(
                text.matches("TemporaryWorkspace").count(),
                1,
                "{name} must perform exactly one temporary allocation transaction: {text}",
            );
            assert!(!text.contains("DeviceBuffer :: zeroed"));
            assert!(!text.contains(". temporary ::"));
            assert!(
                !text.contains("prepare_device"),
                "{name} contains a direct kernel preparation instead of CUB lowering",
            );
        }
    }

    #[test]
    fn ordering_storage_extent_is_selected_inside_the_thrust_lowering() {
        let expansions = [
            (
                "radix keys",
                expand_sort(syn::parse_quote!(::<u32>(&exec, &mut keys))).expect("sort must lower"),
                "__private :: __device_radix_sort_workspace_counts",
            ),
            (
                "radix pairs",
                expand_sort_by_key(syn::parse_quote!(::<u32, u64>(
                    &exec,
                    &mut keys,
                    &mut values
                )))
                .expect("sort_by_key must lower"),
                "__private :: __device_radix_sort_pairs_workspace_counts",
            ),
            (
                "merge sort",
                expand_sort_by(syn::parse_quote!(::<u32>(
                    &exec,
                    &mut keys,
                    |left: &u32, right: &u32| left.cmp(right)
                )))
                .expect("sort_by must lower"),
                "__private :: __device_merge_sort_partition_count",
            ),
            (
                "merge keys",
                expand_merge(syn::parse_quote!(::<u32>(
                    &exec,
                    &first,
                    &second,
                    &mut output
                )))
                .expect("merge must lower"),
                "__private :: __device_merge_keys_partition_count",
            ),
            (
                "merge pairs",
                expand_merge_by_key(syn::parse_quote!(::<u32, u64>(
                    &exec,
                    &first_keys,
                    &first_values,
                    &second_keys,
                    &second_values,
                    &mut output_keys,
                    &mut output_values
                )))
                .expect("merge_by_key must lower"),
                "__private :: __device_merge_pairs_partition_count",
            ),
        ];

        for (family, expansion, extent_macro) in expansions {
            let text = expansion.to_string();
            assert!(
                text.contains(&format!("rccl_cub :: {extent_macro} !")),
                "{family} lost its macro-selected temporary extent: {text}",
            );
            assert!(
                text.contains("TemporaryWorkspace"),
                "{family} stopped using its one owning temporary workspace: {text}",
            );
        }
    }

    #[test]
    fn ordering_storage_uses_cccl_alignment_and_never_masks_with_zero_fill() {
        let radix_keys = expand_sort(syn::parse_quote!(::<u32>(&exec, &mut keys)))
            .expect("radix keys must lower")
            .to_string();
        assert!(radix_keys.contains("[128usize , 256usize , 256usize , 256usize]"));
        assert!(
            radix_keys.find("drop (__rccl_temporary)").unwrap()
                < radix_keys.find("__synchronize (__rccl_result)").unwrap(),
            "pinned radix sort releases its inner temporary array before smart-sort synchronization",
        );

        let radix_pairs = expand_sort_by_key(syn::parse_quote!(::<u32, u64>(
            &exec,
            &mut keys,
            &mut values
        )))
        .expect("radix pairs must lower")
        .to_string();
        assert!(radix_pairs.contains("[128usize , 128usize , 256usize , 256usize , 256usize]"));
        assert!(
            radix_pairs.find("drop (__rccl_temporary)").unwrap()
                < radix_pairs.find("__synchronize (__rccl_result)").unwrap(),
            "pinned radix pair sort releases its inner temporary array before smart-sort synchronization",
        );

        let comparison = expand_sort_by(syn::parse_quote!(::<u32>(
            &exec,
            &mut keys,
            |left: &u32, right: &u32| left.cmp(right)
        )))
        .expect("comparison sort must lower")
        .to_string();
        assert!(comparison.contains("[256usize , 256usize]"));

        for expansion in [radix_keys, radix_pairs, comparison] {
            assert_eq!(expansion.matches("TemporaryWorkspace").count(), 1);
            assert!(!expansion.contains("DeviceBuffer :: zeroed"));
            assert!(!expansion.contains("zero_async"));
            assert!(!expansion.contains("unsafe"));
            assert!(!expansion.contains("* mut"));
        }
    }
}
