use crate::call;
use crate::target::{CudaTarget, configured_cuda_target, tmp_primitive_size};
use quote::{format_ident, quote};

pub(super) fn nominal_4b_items(nominal: usize, item_size: usize) -> usize {
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

pub(super) fn merge_load_to_shared_policy(
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

pub(super) fn merge_shared_bytes(
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

pub(super) fn expand_device_merge_keys(
    call: call::DeviceMergeKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = configured_cuda_target()?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let items = nominal_4b_items(nominal_items, key_size);
    let (use_load_to_shared, _) = merge_load_to_shared_policy(target, key_size, None);
    let launch = merge_keys_launch(&key_type, key_size, use_load_to_shared, items);
    let execution = call.execution;
    let first = call.first;
    let second = call.second;
    let workspace = call.workspace;
    let output = call.output;
    let order = call.order;
    Ok(quote! {{
        let __rccl_execution = #execution;
        let __rccl_first = #first;
        let __rccl_second = #second;
        let __rccl_workspace = #workspace;
        let __rccl_partitions = __rccl_workspace.__partitions_mut();
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
    }})
}

pub(super) fn expand_device_merge_keys_workspace(
    call: call::DeviceMergeKeysWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let target = configured_cuda_target()?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let tile_items = 512usize * nominal_4b_items(nominal_items, key_size);
    let execution = call.execution;
    let first_len = call.first_len;
    let second_len = call.second_len;
    Ok(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceMergeWorkspace,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_total = (#first_len)
                .checked_add(#second_len)
                .ok_or(::rccl_cub::DeviceAlgorithmError::SizeOverflow)?;
            let __rccl_count = if __rccl_total == 0 {
                0usize
            } else {
                __rccl_total.div_ceil(#tile_items) + 1
            };
            let __rccl_partitions =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<usize>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_count,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            ::core::result::Result::Ok(
                ::rccl_cub::DeviceMergeWorkspace::__from_partitions(__rccl_partitions),
            )
        })()
    }})
}

pub(super) fn expand_device_merge_keys_partition_count(
    call: call::DeviceMergeKeysPartitionCountCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let target = configured_cuda_target()?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let items = nominal_4b_items(nominal_items, key_size);
    let tile_items = 512usize * items;
    let first_len = call.first_len;
    let second_len = call.second_len;
    Ok(quote! {{
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

pub(super) fn expand_device_merge_pairs_partition_count(
    call: call::DeviceMergePairsPartitionCountCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let value_size = tmp_primitive_size(&call.value_type)?;
    let target = configured_cuda_target()?;
    let tune_size = key_size
        .checked_add(value_size)
        .ok_or_else(|| syn::Error::new_spanned(&call.value_type, "merge tuning size overflowed"))?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let items = nominal_4b_items(nominal_items, tune_size);
    let tile_items = 512usize * items;
    let first_len = call.first_len;
    let second_len = call.second_len;
    Ok(quote! {{
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

pub(super) fn expand_device_merge_pairs(
    call: call::DeviceMergePairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let target = configured_cuda_target()?;
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
    let workspace = call.workspace;
    let output_keys = call.output_keys;
    let output_values = call.output_values;
    let order = call.order;
    Ok(quote! {{
        let __rccl_execution = #execution;
        let __rccl_first_keys = #first_keys;
        let __rccl_first_values = #first_values;
        let __rccl_second_keys = #second_keys;
        let __rccl_second_values = #second_values;
        let __rccl_workspace = #workspace;
        let __rccl_partitions = __rccl_workspace.__partitions_mut();
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
    }})
}

pub(super) fn expand_device_merge_pairs_workspace(
    call: call::DeviceMergePairsWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let value_size = tmp_primitive_size(&call.value_type)?;
    let target = configured_cuda_target()?;
    let tune_size = key_size
        .checked_add(value_size)
        .ok_or_else(|| syn::Error::new_spanned(&call.value_type, "merge tuning size overflowed"))?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let tile_items = 512usize * nominal_4b_items(nominal_items, tune_size);
    let execution = call.execution;
    let first_len = call.first_len;
    let second_len = call.second_len;
    Ok(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceMergeWorkspace,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_total = (#first_len)
                .checked_add(#second_len)
                .ok_or(::rccl_cub::DeviceAlgorithmError::SizeOverflow)?;
            let __rccl_count = if __rccl_total == 0 {
                0usize
            } else {
                __rccl_total.div_ceil(#tile_items) + 1
            };
            let __rccl_partitions =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<usize>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_count,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            ::core::result::Result::Ok(
                ::rccl_cub::DeviceMergeWorkspace::__from_partitions(__rccl_partitions),
            )
        })()
    }})
}

pub(super) fn expand_device_merge_sort_partition_count(
    call: call::DeviceMergeSortPartitionCountCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let target = configured_cuda_target()?;
    let policy = merge_sort_policy(target, key_size);
    let tile_items = policy.threads * policy.items;
    let len = call.len;
    Ok(quote! {{
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

pub(super) fn expand_device_merge_sort_keys_workspace(
    call: call::DeviceMergeSortKeysWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = configured_cuda_target()?;
    let policy = merge_sort_policy(target, key_size);
    let tile_items = policy.threads * policy.items;
    let execution = call.execution;
    let len = call.len;
    Ok(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceMergeSortKeysWorkspace<#key_type>,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_len: usize = #len;
            let __rccl_partition_count = if __rccl_len == 0 {
                0usize
            } else {
                __rccl_len.div_ceil(#tile_items) + 1
            };
            let __rccl_scratch_keys =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<#key_type>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_len,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_partitions =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<usize>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_partition_count,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            ::core::result::Result::Ok(
                ::rccl_cub::DeviceMergeSortKeysWorkspace::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_partitions,
                ),
            )
        })()
    }})
}

pub(super) fn expand_device_merge_sort_pairs_workspace(
    call: call::DeviceMergeSortPairsWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = configured_cuda_target()?;
    let policy = merge_sort_policy(target, key_size);
    let tile_items = policy.threads * policy.items;
    let execution = call.execution;
    let len = call.len;
    Ok(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceMergeSortPairsWorkspace<#key_type, #value_type>,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_len: usize = #len;
            let __rccl_partition_count = if __rccl_len == 0 {
                0usize
            } else {
                __rccl_len.div_ceil(#tile_items) + 1
            };
            let __rccl_scratch_keys =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<#key_type>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_len,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_scratch_values =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<#value_type>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_len,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_partitions =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<usize>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_partition_count,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            ::core::result::Result::Ok(
                ::rccl_cub::DeviceMergeSortPairsWorkspace::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_scratch_values,
                    __rccl_partitions,
                ),
            )
        })()
    }})
}

pub(super) fn expand_device_merge_sort_keys(
    call: call::DeviceMergeSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = configured_cuda_target()?;
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
    let workspace = call.workspace;
    let order = call.order;
    Ok(quote! {{
        let __rccl_execution = #execution;
        let __rccl_keys = #keys;
        let __rccl_workspace = #workspace;
        let (__rccl_scratch_keys, __rccl_partitions) = __rccl_workspace.__parts_mut();
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
    }})
}

pub(super) fn expand_device_merge_sort_keys_copy(
    call: call::DeviceMergeSortKeysCopyCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = configured_cuda_target()?;
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
    let workspace = call.workspace;
    let output = call.output;
    let order = call.order;
    Ok(quote! {{
        let __rccl_execution = #execution;
        let __rccl_input = #input;
        let __rccl_workspace = #workspace;
        let (__rccl_scratch_keys, __rccl_partitions) = __rccl_workspace.__parts_mut();
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
    }})
}

pub(super) fn expand_device_merge_sort_pairs(
    call: call::DeviceMergeSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_merge_sort_pairs_operation(
        call.key_type,
        call.value_type,
        call.execution,
        None,
        None,
        call.keys,
        call.values,
        call.workspace,
        call.order,
    )
}

pub(super) fn expand_device_merge_sort_pairs_copy(
    call: call::DeviceMergeSortPairsCopyCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_merge_sort_pairs_operation(
        call.key_type,
        call.value_type,
        call.execution,
        Some(call.input_keys),
        Some(call.input_values),
        call.output_keys,
        call.output_values,
        call.workspace,
        call.order,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn expand_device_merge_sort_pairs_operation(
    key_type: syn::Type,
    value_type: syn::Type,
    execution: syn::Expr,
    input_keys: Option<syn::Expr>,
    input_values: Option<syn::Expr>,
    primary_keys: syn::Expr,
    primary_values: syn::Expr,
    workspace: syn::Expr,
    order: syn::Expr,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&key_type)?;
    let value_size = tmp_primitive_size(&value_type)?;
    let target = configured_cuda_target()?;
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

    Ok(quote! {{
        let __rccl_execution = #execution;
        #mode_bindings
        let __rccl_workspace = #workspace;
        let (__rccl_scratch_keys, __rccl_scratch_values, __rccl_partitions) =
            __rccl_workspace.__parts_mut();
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
    }})
}
