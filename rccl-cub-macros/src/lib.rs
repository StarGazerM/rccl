#![cfg_attr(not(test), feature(proc_macro_tracked_env))]

mod call;
#[cfg(test)]
mod catalog;
mod kernels;
mod merge;
mod program;
mod radix;
mod set;
mod syntax;
mod target;
#[cfg(test)]
mod tests;
mod warp;

use kernels::expand_device_kernels;
use merge::*;
use proc_macro::TokenStream;
use program::expand_user_program;
use radix::*;
use set::*;
use warp::{expand_warp_reduce, expand_warp_reduce_items};

/// Declare a crate-root cuda-oxide program containing user kernels that call
/// CUB block/warp primitives.
///
/// The attribute owns the exact artifact-provenance bind, so the caller uses
/// `module_name::program!(context)` without spelling an unsafe module load.
#[proc_macro_attribute]
pub fn program(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[rccl_cub::program] does not take arguments",
        )
        .into_compile_error()
        .into();
    }
    let module = syn::parse_macro_input!(item as syn::ItemMod);
    expand_user_program(module)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

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

/// Internal item lowering used by declarative Thrust backend scaffolding.
#[doc(hidden)]
#[proc_macro]
pub fn __device_kernels(input: TokenStream) -> TokenStream {
    let module = syn::parse_macro_input!(input as syn::Ident);
    expand_device_kernels(module)
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
    __device_merge_keys_partition_count,
    call::DeviceMergeKeysPartitionCountCall,
    expand_device_merge_keys_partition_count
);
operation_macro!(
    device_merge_keys_workspace,
    call::DeviceMergeKeysWorkspaceCall,
    expand_device_merge_keys_workspace
);
operation_macro!(
    __device_merge_pairs_partition_count,
    call::DeviceMergePairsPartitionCountCall,
    expand_device_merge_pairs_partition_count
);
operation_macro!(
    device_merge_pairs_workspace,
    call::DeviceMergePairsWorkspaceCall,
    expand_device_merge_pairs_workspace
);
operation_macro!(
    device_merge_pairs,
    call::DeviceMergePairsCall,
    expand_device_merge_pairs
);
operation_macro!(
    __device_set_workspace_counts,
    call::DeviceSetWorkspaceCountsCall,
    expand_device_set_workspace_counts
);
operation_macro!(
    __device_set_keys_i32,
    call::DeviceSetKeysCall,
    expand_device_set_keys_i32
);
operation_macro!(
    __device_set_keys_i64,
    call::DeviceSetKeysCall,
    expand_device_set_keys_i64
);
operation_macro!(
    __device_set_discard_i32,
    call::DeviceSetDiscardCall,
    expand_device_set_discard_i32
);
operation_macro!(
    __device_set_discard_i64,
    call::DeviceSetDiscardCall,
    expand_device_set_discard_i64
);
operation_macro!(
    __device_set_arithmetic_i32,
    call::DeviceSetArithmeticCall,
    expand_device_set_arithmetic_i32
);
operation_macro!(
    __device_set_arithmetic_i64,
    call::DeviceSetArithmeticCall,
    expand_device_set_arithmetic_i64
);
operation_macro!(
    __device_set_pairs_i32,
    call::DeviceSetPairsCall,
    expand_device_set_pairs_i32
);
operation_macro!(
    __device_set_pairs_i64,
    call::DeviceSetPairsCall,
    expand_device_set_pairs_i64
);
operation_macro!(
    device_merge_sort_keys,
    call::DeviceMergeSortKeysCall,
    expand_device_merge_sort_keys
);
operation_macro!(
    __device_merge_sort_partition_count,
    call::DeviceMergeSortPartitionCountCall,
    expand_device_merge_sort_partition_count
);
operation_macro!(
    device_merge_sort_keys_workspace,
    call::DeviceMergeSortKeysWorkspaceCall,
    expand_device_merge_sort_keys_workspace
);
operation_macro!(
    device_merge_sort_pairs_workspace,
    call::DeviceMergeSortPairsWorkspaceCall,
    expand_device_merge_sort_pairs_workspace
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
    __device_radix_sort_workspace_counts,
    call::DeviceRadixSortWorkspaceCountsCall,
    expand_device_radix_sort_workspace_counts
);
operation_macro!(
    device_radix_sort_keys_buffer_workspace,
    call::DeviceRadixSortKeysBufferWorkspaceCall,
    expand_device_radix_sort_keys_buffer_workspace
);
operation_macro!(
    device_radix_sort_pairs_workspace,
    call::DeviceRadixSortPairsWorkspaceCall,
    expand_device_radix_sort_pairs_workspace
);
operation_macro!(
    __device_radix_sort_pairs_workspace_counts,
    call::DeviceRadixSortPairsWorkspaceCountsCall,
    expand_device_radix_sort_pairs_workspace_counts
);
operation_macro!(
    device_radix_sort_pairs_buffer_workspace,
    call::DeviceRadixSortPairsBufferWorkspaceCall,
    expand_device_radix_sort_pairs_buffer_workspace
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
