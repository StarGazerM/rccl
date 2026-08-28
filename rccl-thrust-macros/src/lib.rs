mod audit;
mod call;
#[cfg(test)]
mod catalog;

use proc_macro::TokenStream;
use quote::quote;

#[proc_macro]
pub fn thrust_kernels(input: TokenStream) -> TokenStream {
    if !input.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "thrust_kernels! takes no input",
        )
        .into_compile_error()
        .into();
    }
    expand_thrust_kernels()
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

fn expand_thrust_kernels() -> syn::Result<proc_macro2::TokenStream> {
    audit::generated_rust(quote! {
        ::rccl_cub::__private::__device_kernels!(__rccl_thrust_kernels);
    })
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
        let __rccl_result = #operation;
        match __rccl_result {
            ::core::result::Result::Err(__rccl_error) => {
                ::core::result::Result::Err(__rccl_error)
            }
            ::core::result::Result::Ok(()) => __rccl_execution
                .stream()
                .synchronize()
                .map_err(::rccl_thrust::DeviceAlgorithmError::Driver),
        }
    }})
}

fn expand_sort(call: call::SortCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let target = call.target;
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
                ::rccl_cub::device_radix_sort_workspace!(::<#key_type, #target>(__rccl_len));
            let mut __rccl_scratch_keys =
                __rccl_execution.temporary::<#key_type>(__rccl_workspace.scratch_keys)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_bins =
                __rccl_execution.temporary::<u64>(__rccl_workspace.bins)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_lookback =
                __rccl_execution.temporary::<i32>(__rccl_workspace.lookback)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_counters =
                __rccl_execution.temporary::<i32>(__rccl_workspace.counters)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_radix_sort_keys!(::<#key_type, #target>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_scratch_keys,
                &mut __rccl_bins,
                &mut __rccl_lookback,
                &mut __rccl_counters,
            ))
        })()},
    )
}

fn expand_sort_by(call: call::SortByCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let target = call.target;
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
                ::rccl_cub::device_merge_sort_partition_count!(::<#key_type, #target>(
                    __rccl_len
                ))?;
            let mut __rccl_scratch_keys =
                __rccl_execution.temporary::<#key_type>(__rccl_len)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_partitions =
                __rccl_execution.temporary::<usize>(__rccl_partition_count)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_merge_sort_keys!(::<#key_type, #target>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_scratch_keys,
                &mut __rccl_partitions,
                __rccl_compare,
            ))
        })()},
    )
}

fn expand_stable_sort(call: call::SortCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let target = call.target;
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
                ::rccl_cub::device_radix_sort_workspace!(::<#key_type, #target>(__rccl_len));
            let mut __rccl_scratch_keys =
                __rccl_execution.temporary::<#key_type>(__rccl_workspace.scratch_keys)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_bins =
                __rccl_execution.temporary::<u64>(__rccl_workspace.bins)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_lookback =
                __rccl_execution.temporary::<i32>(__rccl_workspace.lookback)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_counters =
                __rccl_execution.temporary::<i32>(__rccl_workspace.counters)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_radix_sort_keys!(::<#key_type, #target>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_scratch_keys,
                &mut __rccl_bins,
                &mut __rccl_lookback,
                &mut __rccl_counters,
            ))
        })()},
    )
}

fn expand_stable_sort_by(call: call::SortByCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let target = call.target;
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
                ::rccl_cub::device_merge_sort_partition_count!(::<#key_type, #target>(
                    __rccl_len
                ))?;
            let mut __rccl_scratch_keys =
                __rccl_execution.temporary::<#key_type>(__rccl_len)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_partitions =
                __rccl_execution.temporary::<usize>(__rccl_partition_count)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_stable_merge_sort_keys!(::<#key_type, #target>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_scratch_keys,
                &mut __rccl_partitions,
                __rccl_compare,
            ))
        })()},
    )
}

fn expand_sort_by_key(call: call::SortByKeyCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let target = call.target;
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
                ::rccl_cub::device_radix_sort_pairs_workspace!(::<
                    #key_type,
                    #value_type,
                    #target
                >(__rccl_len));
            let mut __rccl_scratch_keys =
                __rccl_execution.temporary::<#key_type>(__rccl_workspace.scratch_keys)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_scratch_values =
                __rccl_execution.temporary::<#value_type>(__rccl_workspace.scratch_values)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_bins =
                __rccl_execution.temporary::<u64>(__rccl_workspace.bins)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_lookback =
                __rccl_execution.temporary::<i32>(__rccl_workspace.lookback)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_counters =
                __rccl_execution.temporary::<i32>(__rccl_workspace.counters)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_radix_sort_pairs!(::<#key_type, #value_type, #target>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_scratch_keys,
                &mut __rccl_scratch_values,
                &mut __rccl_bins,
                &mut __rccl_lookback,
                &mut __rccl_counters,
            ))
        })()},
    )
}

fn expand_sort_by_key_with(call: call::SortByKeyWithCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let target = call.target;
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
                ::rccl_cub::device_merge_sort_partition_count!(::<#key_type, #target>(
                    __rccl_len
                ))?;
            let mut __rccl_scratch_keys =
                __rccl_execution.temporary::<#key_type>(__rccl_len)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_scratch_values =
                __rccl_execution.temporary::<#value_type>(__rccl_len)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_partitions =
                __rccl_execution.temporary::<usize>(__rccl_partition_count)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_merge_sort_pairs!(::<#key_type, #value_type, #target>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_scratch_keys,
                &mut __rccl_scratch_values,
                &mut __rccl_partitions,
                __rccl_compare,
            ))
        })()},
    )
}

fn expand_stable_sort_by_key(call: call::SortByKeyCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let target = call.target;
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
                ::rccl_cub::device_radix_sort_pairs_workspace!(::<
                    #key_type,
                    #value_type,
                    #target
                >(__rccl_len));
            let mut __rccl_scratch_keys =
                __rccl_execution.temporary::<#key_type>(__rccl_workspace.scratch_keys)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_scratch_values =
                __rccl_execution.temporary::<#value_type>(__rccl_workspace.scratch_values)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_bins =
                __rccl_execution.temporary::<u64>(__rccl_workspace.bins)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_lookback =
                __rccl_execution.temporary::<i32>(__rccl_workspace.lookback)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_counters =
                __rccl_execution.temporary::<i32>(__rccl_workspace.counters)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_radix_sort_pairs!(::<#key_type, #value_type, #target>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_scratch_keys,
                &mut __rccl_scratch_values,
                &mut __rccl_bins,
                &mut __rccl_lookback,
                &mut __rccl_counters,
            ))
        })()},
    )
}

fn expand_stable_sort_by_key_with(
    call: call::StableSortByKeyCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let target = call.target;
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
                ::rccl_cub::device_merge_sort_partition_count!(::<#key_type, #target>(
                    __rccl_len
                ))?;
            let mut __rccl_scratch_keys =
                __rccl_execution.temporary::<#key_type>(__rccl_len)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_scratch_values =
                __rccl_execution.temporary::<#value_type>(__rccl_len)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            let mut __rccl_partitions =
                __rccl_execution.temporary::<usize>(__rccl_partition_count)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_stable_merge_sort_pairs!(::<#key_type, #value_type, #target>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_scratch_keys,
                &mut __rccl_scratch_values,
                &mut __rccl_partitions,
                __rccl_compare,
            ))
        })()},
    )
}

fn expand_merge(call: call::MergeCall) -> syn::Result<proc_macro2::TokenStream> {
    let value_type = call.value_type;
    let target = call.target;
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
                ::rccl_cub::device_merge_keys_partition_count!(::<#value_type, #target>(
                    ::rccl_thrust::DeviceRange::len(&__rccl_first),
                    ::rccl_thrust::DeviceRange::len(&__rccl_second),
                ))?;
            let mut __rccl_partitions =
                __rccl_execution.temporary::<usize>(__rccl_partition_count)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_merge_keys!(::<#value_type, #target>(
                __rccl_execution,
                __rccl_first,
                __rccl_second,
                &mut __rccl_partitions,
                __rccl_output,
                #order,
            ))
        })()},
    )
}

fn expand_merge_by(call: call::MergeByCall) -> syn::Result<proc_macro2::TokenStream> {
    let value_type = call.value_type;
    let target = call.target;
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
                ::rccl_cub::device_merge_keys_partition_count!(::<#value_type, #target>(
                    ::rccl_thrust::DeviceRange::len(&__rccl_first),
                    ::rccl_thrust::DeviceRange::len(&__rccl_second),
                ))?;
            let mut __rccl_partitions =
                __rccl_execution.temporary::<usize>(__rccl_partition_count)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_merge_keys!(::<#value_type, #target>(
                __rccl_execution,
                __rccl_first,
                __rccl_second,
                &mut __rccl_partitions,
                __rccl_output,
                __rccl_compare,
            ))
        })()},
    )
}

fn expand_merge_by_key(call: call::MergeByKeyCall) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let target = call.target;
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
                ::rccl_cub::device_merge_pairs_partition_count!(::<
                    #key_type,
                    #value_type,
                    #target
                >(
                    ::rccl_thrust::DeviceRange::len(&__rccl_first_keys),
                    ::rccl_thrust::DeviceRange::len(&__rccl_second_keys),
                ))?;
            let mut __rccl_partitions =
                __rccl_execution.temporary::<usize>(__rccl_partition_count)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_merge_pairs!(::<#key_type, #value_type, #target>(
                __rccl_execution,
                __rccl_first_keys,
                __rccl_first_values,
                __rccl_second_keys,
                __rccl_second_values,
                &mut __rccl_partitions,
                __rccl_output_keys,
                __rccl_output_values,
                #order,
            ))
        })()},
    )
}

fn expand_merge_by_key_with(
    call: call::MergeByKeyWithCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let target = call.target;
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
                ::rccl_cub::device_merge_pairs_partition_count!(::<
                    #key_type,
                    #value_type,
                    #target
                >(
                    ::rccl_thrust::DeviceRange::len(&__rccl_first_keys),
                    ::rccl_thrust::DeviceRange::len(&__rccl_second_keys),
                ))?;
            let mut __rccl_partitions =
                __rccl_execution.temporary::<usize>(__rccl_partition_count)
                .map_err(::rccl_thrust::DeviceAlgorithmError::Memory)?;
            ::rccl_cub::device_merge_pairs!(::<#key_type, #value_type, #target>(
                __rccl_execution,
                __rccl_first_keys,
                __rccl_first_values,
                __rccl_second_keys,
                __rccl_second_values,
                &mut __rccl_partitions,
                __rccl_output_keys,
                __rccl_output_values,
                __rccl_compare,
            ))
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
                expand_sort(syn::parse_quote!(::<i32, sm_89>(
                    &exec,
                    &mut keys
                )))
                .expect("sort must lower"),
                "device_radix_sort_keys",
            ),
            (
                "sort_by",
                expand_sort_by(syn::parse_quote!(
                    ::<i32, sm_89>(
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
                expand_stable_sort(syn::parse_quote!(::<i32, sm_89>(
                    &exec,
                    &mut keys
                )))
                .expect("stable_sort must lower"),
                "device_radix_sort_keys",
            ),
            (
                "stable_sort_by",
                expand_stable_sort_by(syn::parse_quote!(
                    ::<i32, sm_89>(
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
                expand_sort_by_key(syn::parse_quote!(::<i32, u64, sm_89>(
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
                    ::<i32, u64, sm_89>(
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
                expand_stable_sort_by_key(syn::parse_quote!(::<i32, u64, sm_89>(
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
                    ::<i32, u64, sm_89>(
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
                expand_merge(syn::parse_quote!(::<i32, sm_89>(
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
                    ::<i32, sm_89>(
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
                    ::<i32, u64, sm_89>(
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
                    ::<i32, u64, sm_89>(
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
            assert!(
                text.contains("temporary"),
                "{name} did not acquire its temporary storage from the execution policy: {text}",
            );
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
                expand_sort(syn::parse_quote!(::<u32, sm_89>(&exec, &mut keys)))
                    .expect("sort must lower"),
                "device_radix_sort_workspace",
            ),
            (
                "radix pairs",
                expand_sort_by_key(syn::parse_quote!(::<u32, u64, sm_89>(
                    &exec,
                    &mut keys,
                    &mut values
                )))
                .expect("sort_by_key must lower"),
                "device_radix_sort_pairs_workspace",
            ),
            (
                "merge sort",
                expand_sort_by(syn::parse_quote!(::<u32, sm_89>(
                    &exec,
                    &mut keys,
                    |left: &u32, right: &u32| left.cmp(right)
                )))
                .expect("sort_by must lower"),
                "device_merge_sort_partition_count",
            ),
            (
                "merge keys",
                expand_merge(syn::parse_quote!(::<u32, sm_89>(
                    &exec,
                    &first,
                    &second,
                    &mut output
                )))
                .expect("merge must lower"),
                "device_merge_keys_partition_count",
            ),
            (
                "merge pairs",
                expand_merge_by_key(syn::parse_quote!(::<u32, u64, sm_89>(
                    &exec,
                    &first_keys,
                    &first_values,
                    &second_keys,
                    &second_values,
                    &mut output_keys,
                    &mut output_values
                )))
                .expect("merge_by_key must lower"),
                "device_merge_pairs_partition_count",
            ),
        ];

        for (family, expansion, extent_macro) in expansions {
            let text = expansion.to_string();
            assert!(
                text.contains(&format!("rccl_cub :: {extent_macro} !")),
                "{family} lost its macro-selected temporary extent: {text}",
            );
            assert!(
                text.contains("temporary"),
                "{family} stopped allocating through Execution: {text}",
            );
        }
    }
}
