use quote::{format_ident, quote};

use crate::{
    call,
    target::{CudaTarget, configured_cuda_target, tmp_primitive_size},
};

#[derive(Clone, Copy)]
pub(crate) struct SetPolicy {
    pub(crate) threads: usize,
    pub(crate) items: usize,
    pub(crate) shared_bytes: usize,
    pub(crate) sm70: bool,
}

fn align_16(bytes: usize) -> syn::Result<usize> {
    bytes
        .checked_add(15)
        .map(|value| value / 16 * 16)
        .ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "set shared storage overflowed",
            )
        })
}

fn set_policy(key_size: usize, value_size: usize) -> syn::Result<SetPolicy> {
    let target = configured_cuda_target()?;
    set_policy_for(target, key_size, value_size)
}

pub(crate) fn set_policy_for(
    target: CudaTarget,
    key_size: usize,
    value_size: usize,
) -> syn::Result<SetPolicy> {
    let (threads, nominal) = if target.capability >= 60 {
        (512usize, 19usize)
    } else {
        (256usize, 15usize)
    };
    let items = ((nominal * 4).div_ceil(key_size)).clamp(1, nominal);
    let tile_items = threads
        .checked_mul(items)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            syn::Error::new(proc_macro2::Span::call_site(), "set tile size overflowed")
        })?;
    let shared_items = tile_items.checked_add(threads).ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "set shared item count overflowed",
        )
    })?;
    let union_bytes = shared_items
        .checked_mul(key_size.max(value_size))
        .ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "set shared union overflowed",
            )
        })?;
    let shared_bytes = align_16(
        threads
            .checked_mul(core::mem::size_of::<i32>())
            .and_then(|offsets| offsets.checked_add(union_bytes))
            .ok_or_else(|| {
                syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "set shared storage overflowed",
                )
            })?,
    )?;
    Ok(SetPolicy {
        threads,
        items,
        shared_bytes,
        sm70: target.capability >= 70,
    })
}

pub(super) fn expand_device_set_workspace_counts(
    call: call::DeviceSetWorkspaceCountsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let value_size = tmp_primitive_size(&call.value_type)?;
    let policy = set_policy(key_size, value_size)?;
    let tile_items = policy.threads * policy.items - 1;
    let shared_bytes = policy.shared_bytes;
    let execution = call.execution;
    let first_len = call.first_len;
    let second_len = call.second_len;
    Ok(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::SetOperationWorkspaceCounts,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_first_len = #first_len;
            let __rccl_second_len = #second_len;
            let __rccl_total = __rccl_first_len
                .checked_add(__rccl_second_len)
                .ok_or(::rccl_cub::DeviceAlgorithmError::SizeOverflow)?;
            if __rccl_total == 0 {
                return ::core::result::Result::Ok(
                    ::rccl_cub::SetOperationWorkspaceCounts {
                        num_tiles: 0,
                        tile_descriptors: 0,
                        partitions: 0,
                        virtual_shared_bytes: 0,
                        wide_indices: false,
                    },
                );
            }
            let __rccl_num_tiles = __rccl_total.div_ceil(#tile_items);
            let __rccl_max_shared = __rccl_execution
                .stream()
                .context()
                .launch_limits()
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?
                .max_shared_memory_per_block() as usize;
            let __rccl_virtual_shared_bytes = if #shared_bytes > __rccl_max_shared {
                #shared_bytes
                    .checked_mul(__rccl_num_tiles)
                    .ok_or(::rccl_cub::DeviceAlgorithmError::SizeOverflow)?
            } else {
                0usize
            };
            ::core::result::Result::Ok(::rccl_cub::SetOperationWorkspaceCounts {
                num_tiles: __rccl_num_tiles,
                tile_descriptors: __rccl_num_tiles + 32,
                partitions: __rccl_num_tiles + 1,
                virtual_shared_bytes: __rccl_virtual_shared_bytes,
                wide_indices: __rccl_total > i32::MAX as usize,
            })
        })()
    }})
}

fn set_launch(call: SetLaunchCall, wide: bool) -> syn::Result<proc_macro2::TokenStream> {
    let SetLaunchCall {
        key_type,
        value_type,
        operation_type,
        execution,
        first_keys,
        first_values,
        second_keys,
        second_values,
        workspace,
        output_keys,
        output_values,
        order,
        arithmetic,
    } = call;
    let key_size = tmp_primitive_size(&key_type)?;
    let value_size = tmp_primitive_size(&value_type)?;
    let policy = set_policy(key_size, value_size)?;
    let threads = policy.threads;
    let items = policy.items;
    let tile_items = threads * items - 1;
    let shared_bytes = policy.shared_bytes;
    let sm70 = policy.sm70;
    let suffix = if wide { "i64" } else { "i32" };
    let init_prepare = format_ident!("prepare_device_set_init_{suffix}");
    let init_launch = format_ident!("device_set_init_{suffix}");
    let partition_kind = if arithmetic {
        "arithmetic_partition"
    } else {
        "partition"
    };
    let partition_prepare = format_ident!("prepare_device_set_{partition_kind}_{suffix}");
    let partition_launch = format_ident!("device_set_{partition_kind}_{suffix}");
    let kind = if arithmetic {
        "arithmetic"
    } else if output_values.is_some() {
        "pairs"
    } else if output_keys.is_some() {
        "keys"
    } else {
        "discard"
    };
    let agent_prepare = format_ident!("prepare_device_set_{kind}_{suffix}_{threads}");
    let agent_launch = format_ident!("device_set_{kind}_{suffix}_{threads}");
    let virtual_prepare = format_ident!("prepare_device_set_{kind}_virtual_{suffix}_{threads}");
    let virtual_launch = format_ident!("device_set_{kind}_virtual_{suffix}_{threads}");
    let parts_method = if wide {
        format_ident!("__wide_parts_mut")
    } else {
        format_ident!("__narrow_parts_mut")
    };
    let (length_bindings, partition_input_args, agent_input_args) = if arithmetic {
        (
            quote! {
                let __rccl_first_len = ::rccl_cub::DeviceRange::len(&__rccl_first_keys);
                let __rccl_second_len = ::rccl_cub::DeviceRange::len(&__rccl_second_keys);
            },
            quote! {
                *__rccl_first_keys.start(),
                *__rccl_first_keys.step(),
                __rccl_first_len,
                *__rccl_second_keys.start(),
                *__rccl_second_keys.step(),
                __rccl_second_len,
            },
            quote! {
                *__rccl_first_keys.start(),
                *__rccl_first_keys.step(),
                __rccl_first_len,
                *__rccl_second_keys.start(),
                *__rccl_second_keys.step(),
                __rccl_second_len,
            },
        )
    } else {
        (
            quote! {
                let __rccl_first_len = __rccl_first_keys.len();
                let __rccl_second_len = __rccl_second_keys.len();
            },
            quote! { __rccl_first_keys, __rccl_second_keys, },
            quote! { __rccl_first_keys, __rccl_second_keys, },
        )
    };
    let output_binding = output_keys.as_ref().map_or_else(
        || quote! {},
        |output| quote! { let __rccl_output_keys = #output; },
    );
    let (value_bindings, value_validation, value_launch_args) = match (
        &first_values,
        &second_values,
        &output_values,
    ) {
        (Some(first_values), Some(second_values), Some(output_values)) => (
            quote! {
                let __rccl_first_values = #first_values;
                let __rccl_second_values = #second_values;
                let __rccl_output_values = #output_values;
            },
            quote! {
                if __rccl_first_values.len() != __rccl_first_keys.len() {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "first set values",
                            expected: __rccl_first_keys.len(),
                            actual: __rccl_first_values.len(),
                        },
                    );
                }
                if <#operation_type as ::rccl_cub::__private::SerialSetOperation>::READS_SECOND_VALUES
                    && __rccl_second_values.len() != __rccl_second_keys.len()
                {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "second set values",
                            expected: __rccl_second_keys.len(),
                            actual: __rccl_second_values.len(),
                        },
                    );
                }
                if __rccl_output_values.len() != __rccl_output_keys.len() {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "set output values",
                            expected: __rccl_output_keys.len(),
                            actual: __rccl_output_values.len(),
                        },
                    );
                }
            },
            quote! {
                __rccl_first_values,
                __rccl_second_values,
                &*__rccl_partitions,
                __rccl_order,
                &mut *__rccl_tile_state,
                &mut *__rccl_output_count,
                __rccl_output_keys,
                __rccl_output_values,
            },
        ),
        (None, None, None) => {
            let output_argument = if output_keys.is_some() {
                quote! { __rccl_output_keys, }
            } else {
                quote! {}
            };
            (
                quote! {},
                quote! {},
                quote! {
                    &*__rccl_partitions,
                    __rccl_order,
                    &mut *__rccl_tile_state,
                    &mut *__rccl_output_count,
                    #output_argument
                },
            )
        }
        _ => unreachable!("set launch shape is constructed internally"),
    };
    let agent_generics = if output_values.is_some() {
        quote!(::<#key_type, #value_type, _, #operation_type, #items, #shared_bytes, #sm70>)
    } else {
        quote!(::<#key_type, _, #operation_type, #items, #shared_bytes, #sm70>)
    };
    Ok(quote! {{
        (|| -> ::core::result::Result<(), ::rccl_cub::DeviceAlgorithmError> {
            let __rccl_execution = #execution;
            let __rccl_first_keys = #first_keys;
            let __rccl_second_keys = #second_keys;
            let __rccl_workspace = #workspace;
            #output_binding
            let __rccl_order = #order;
            #value_bindings
            #length_bindings
            let __rccl_total = __rccl_first_len
                .checked_add(__rccl_second_len)
                .ok_or(::rccl_cub::DeviceAlgorithmError::SizeOverflow)?;
            if __rccl_total == 0 {
                return ::core::result::Result::Ok(());
            }
            #value_validation
            let __rccl_num_tiles = __rccl_total.div_ceil(#tile_items);
            let (__rccl_output_count, __rccl_tile_state, __rccl_partitions, __rccl_virtual_shared) =
                __rccl_workspace.#parts_method();
            if __rccl_output_count.len() != 1 {
                return ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "set output count",
                        expected: 1,
                        actual: __rccl_output_count.len(),
                    },
                );
            }
            if __rccl_tile_state.len() != __rccl_num_tiles + 32 {
                return ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "set tile descriptors",
                        expected: __rccl_num_tiles + 32,
                        actual: __rccl_tile_state.len(),
                    },
                );
            }
            if __rccl_partitions.len() != __rccl_num_tiles + 1 {
                return ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "set balanced partitions",
                        expected: __rccl_num_tiles + 1,
                        actual: __rccl_partitions.len(),
                    },
                );
            }
            if !__rccl_virtual_shared.is_empty()
                && __rccl_virtual_shared.len() != #shared_bytes * __rccl_num_tiles
            {
                return ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "set virtual shared memory",
                        expected: #shared_bytes * __rccl_num_tiles,
                        actual: __rccl_virtual_shared.len(),
                    },
                );
            }

            let __rccl_init_config = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                __rccl_num_tiles.div_ceil(128).min(u32::MAX as usize) as u32,
                128,
                0,
            );
            let __rccl_init = __rccl_execution
                .module()
                .#init_prepare(__rccl_init_config)
                .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
            __rccl_execution
                .module()
                .#init_launch(
                    __rccl_execution.stream(),
                    &__rccl_init,
                    &mut *__rccl_tile_state,
                    __rccl_num_tiles,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

            let __rccl_partition_config = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                (__rccl_num_tiles + 1).div_ceil(256).min(u32::MAX as usize) as u32,
                256,
                0,
            );
            let __rccl_partition = __rccl_execution
                .module()
                .#partition_prepare::<#key_type, _, #tile_items>(__rccl_partition_config)
                .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
            __rccl_execution
                .module()
                .#partition_launch::<#key_type, _, #tile_items>(
                    __rccl_execution.stream(),
                    &__rccl_partition,
                    #partition_input_args
                    __rccl_order,
                    &mut *__rccl_partitions,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

            let __rccl_agent_config = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                __rccl_num_tiles.min(u32::MAX as usize) as u32,
                #threads as u32,
                if __rccl_virtual_shared.is_empty() {
                    #shared_bytes as u32
                } else {
                    0u32
                },
            );
            if __rccl_virtual_shared.is_empty() {
                let __rccl_agent = __rccl_execution
                    .module()
                    .#agent_prepare #agent_generics (__rccl_agent_config)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .#agent_launch #agent_generics (
                        __rccl_execution.stream(),
                        &__rccl_agent,
                        #agent_input_args
                        #value_launch_args
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
            } else {
                let __rccl_agent = __rccl_execution
                    .module()
                    .#virtual_prepare #agent_generics (__rccl_agent_config)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .#virtual_launch #agent_generics (
                        __rccl_execution.stream(),
                        &__rccl_agent,
                        #agent_input_args
                        #value_launch_args
                        &mut *__rccl_virtual_shared,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
            }
            ::core::result::Result::Ok(())
        })()
    }})
}

struct SetLaunchCall {
    key_type: syn::Type,
    value_type: syn::Type,
    operation_type: syn::Type,
    execution: syn::Expr,
    first_keys: syn::Expr,
    first_values: Option<syn::Expr>,
    second_keys: syn::Expr,
    second_values: Option<syn::Expr>,
    workspace: syn::Expr,
    output_keys: Option<syn::Expr>,
    output_values: Option<syn::Expr>,
    order: syn::Expr,
    arithmetic: bool,
}

fn device_set_keys(
    call: call::DeviceSetKeysCall,
    wide: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    set_launch(
        SetLaunchCall {
            value_type: key_type.clone(),
            key_type,
            operation_type: call.operation_type,
            execution: call.execution,
            first_keys: call.first,
            first_values: None,
            second_keys: call.second,
            second_values: None,
            workspace: call.workspace,
            output_keys: Some(call.output),
            output_values: None,
            order: call.order,
            arithmetic: false,
        },
        wide,
    )
}

fn device_set_pairs(
    call: call::DeviceSetPairsCall,
    wide: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    set_launch(
        SetLaunchCall {
            key_type: call.key_type,
            value_type: call.value_type,
            operation_type: call.operation_type,
            execution: call.execution,
            first_keys: call.first_keys,
            first_values: Some(call.first_values),
            second_keys: call.second_keys,
            second_values: Some(call.second_values),
            workspace: call.workspace,
            output_keys: Some(call.output_keys),
            output_values: Some(call.output_values),
            order: call.order,
            arithmetic: false,
        },
        wide,
    )
}

fn device_set_discard(
    call: call::DeviceSetDiscardCall,
    wide: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    set_launch(
        SetLaunchCall {
            value_type: key_type.clone(),
            key_type,
            operation_type: call.operation_type,
            execution: call.execution,
            first_keys: call.first,
            first_values: None,
            second_keys: call.second,
            second_values: None,
            workspace: call.workspace,
            output_keys: None,
            output_values: None,
            order: call.order,
            arithmetic: false,
        },
        wide,
    )
}

fn device_set_arithmetic(
    call: call::DeviceSetArithmeticCall,
    wide: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    set_launch(
        SetLaunchCall {
            value_type: key_type.clone(),
            key_type,
            operation_type: call.operation_type,
            execution: call.execution,
            first_keys: call.first,
            first_values: None,
            second_keys: call.second,
            second_values: None,
            workspace: call.workspace,
            output_keys: Some(call.output),
            output_values: None,
            order: call.order,
            arithmetic: true,
        },
        wide,
    )
}

pub(super) fn expand_device_set_keys_i32(
    call: call::DeviceSetKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    device_set_keys(call, false)
}

pub(super) fn expand_device_set_keys_i64(
    call: call::DeviceSetKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    device_set_keys(call, true)
}

pub(super) fn expand_device_set_discard_i32(
    call: call::DeviceSetDiscardCall,
) -> syn::Result<proc_macro2::TokenStream> {
    device_set_discard(call, false)
}

pub(super) fn expand_device_set_discard_i64(
    call: call::DeviceSetDiscardCall,
) -> syn::Result<proc_macro2::TokenStream> {
    device_set_discard(call, true)
}

pub(super) fn expand_device_set_arithmetic_i32(
    call: call::DeviceSetArithmeticCall,
) -> syn::Result<proc_macro2::TokenStream> {
    device_set_arithmetic(call, false)
}

pub(super) fn expand_device_set_arithmetic_i64(
    call: call::DeviceSetArithmeticCall,
) -> syn::Result<proc_macro2::TokenStream> {
    device_set_arithmetic(call, true)
}

pub(super) fn expand_device_set_pairs_i32(
    call: call::DeviceSetPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    device_set_pairs(call, false)
}

pub(super) fn expand_device_set_pairs_i64(
    call: call::DeviceSetPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    device_set_pairs(call, true)
}
