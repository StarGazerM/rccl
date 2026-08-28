#[macro_export]
macro_rules! sort {
    (::<$key_type:ident>($execution:expr, $range:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_range = $range;
            let __rccl_len = $crate::DeviceRange::len(&*__rccl_range);
            let __rccl_workspace =
                $crate::__private::rccl_cub::__private::__device_radix_sort_workspace_counts!(::<$key_type>(__rccl_len));
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace4::<_, $key_type, u64, i32, i32>::new_in(
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
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (__rccl_scratch_keys, __rccl_bins, __rccl_lookback, __rccl_counters) =
                __rccl_temporary.parts_mut();
            let mut __rccl_storage =
                $crate::__private::rccl_cub::DeviceRadixSortKeysWorkspace::<$key_type, _, _, _, _>::__from_parts(
                __rccl_scratch_keys,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            );
            let __rccl_result = $crate::__private::rccl_cub::device_radix_sort_keys!(::<$key_type>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_storage,
            ));
            ::core::mem::drop(__rccl_storage);
            ::core::mem::drop(__rccl_temporary);
            __rccl_execution.__synchronize(__rccl_result)
        })()
    }};
}

#[macro_export]
macro_rules! sort_by {
    (::<$key_type:ident>($execution:expr, $range:expr, $compare:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_range = $range;
            let __rccl_compare = $compare;
            let __rccl_len = $crate::DeviceRange::len(&*__rccl_range);
            let __rccl_partition_count =
                $crate::__private::rccl_cub::__private::__device_merge_sort_partition_count!(::<$key_type>(
                    __rccl_len
                ))?;
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace2::<_, usize, $key_type>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count, __rccl_len],
                    [256usize, 256usize],
                )
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions, __rccl_scratch_keys) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                $crate::__private::rccl_cub::DeviceMergeSortKeysWorkspace::<$key_type, _, _>::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_partitions,
                );
            __rccl_execution.__synchronize($crate::__private::rccl_cub::device_merge_sort_keys!(::<$key_type>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_workspace,
                __rccl_compare,
            )))
        })()
    }};
}

#[macro_export]
macro_rules! stable_sort {
    (::<$key_type:ident>($execution:expr, $range:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_range = $range;
            let __rccl_len = $crate::DeviceRange::len(&*__rccl_range);
            let __rccl_workspace =
                $crate::__private::rccl_cub::__private::__device_radix_sort_workspace_counts!(::<$key_type>(__rccl_len));
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace4::<_, $key_type, u64, i32, i32>::new_in(
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
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (__rccl_scratch_keys, __rccl_bins, __rccl_lookback, __rccl_counters) =
                __rccl_temporary.parts_mut();
            let mut __rccl_storage =
                $crate::__private::rccl_cub::DeviceRadixSortKeysWorkspace::<$key_type, _, _, _, _>::__from_parts(
                __rccl_scratch_keys,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            );
            let __rccl_result = $crate::__private::rccl_cub::device_radix_sort_keys!(::<$key_type>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_storage,
            ));
            ::core::mem::drop(__rccl_storage);
            ::core::mem::drop(__rccl_temporary);
            __rccl_execution.__synchronize(__rccl_result)
        })()
    }};
}

#[macro_export]
macro_rules! stable_sort_by {
    (::<$key_type:ident>($execution:expr, $range:expr, $compare:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_range = $range;
            let __rccl_compare = $compare;
            let __rccl_len = $crate::DeviceRange::len(&*__rccl_range);
            let __rccl_partition_count =
                $crate::__private::rccl_cub::__private::__device_merge_sort_partition_count!(::<$key_type>(
                    __rccl_len
                ))?;
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace2::<_, usize, $key_type>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count, __rccl_len],
                    [256usize, 256usize],
                )
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions, __rccl_scratch_keys) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                $crate::__private::rccl_cub::DeviceMergeSortKeysWorkspace::<$key_type, _, _>::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_partitions,
                );
            __rccl_execution.__synchronize($crate::__private::rccl_cub::device_stable_merge_sort_keys!(::<$key_type>(
                __rccl_execution,
                __rccl_range,
                &mut __rccl_workspace,
                __rccl_compare,
            )))
        })()
    }};
}

#[macro_export]
macro_rules! sort_by_key {
    (::<$key_type:ident, $value_type:ident>($execution:expr, $keys:expr, $values:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_keys = $keys;
            let __rccl_values = $values;
            let __rccl_len = $crate::DeviceRange::len(&*__rccl_keys);
            let __rccl_workspace =
                $crate::__private::rccl_cub::__private::__device_radix_sort_pairs_workspace_counts!(::<
                    $key_type,
                    $value_type
                >(__rccl_len));
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace5::<
                    _, $key_type, $value_type, u64, i32, i32
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
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (
                __rccl_scratch_keys,
                __rccl_scratch_values,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            ) = __rccl_temporary.parts_mut();
            let mut __rccl_storage =
                $crate::__private::rccl_cub::DeviceRadixSortPairsWorkspace::<
                    $key_type, $value_type, _, _, _, _, _
                >::__from_parts(
                __rccl_scratch_keys,
                __rccl_scratch_values,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            );
            let __rccl_result = $crate::__private::rccl_cub::device_radix_sort_pairs!(::<$key_type, $value_type>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_storage,
            ));
            ::core::mem::drop(__rccl_storage);
            ::core::mem::drop(__rccl_temporary);
            __rccl_execution.__synchronize(__rccl_result)
        })()
    }};
}

#[macro_export]
macro_rules! sort_by_key_with {
    (::<$key_type:ident, $value_type:ident>($execution:expr, $keys:expr, $values:expr, $compare:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_keys = $keys;
            let __rccl_values = $values;
            let __rccl_compare = $compare;
            let __rccl_len = $crate::DeviceRange::len(&*__rccl_keys);
            let __rccl_partition_count =
                $crate::__private::rccl_cub::__private::__device_merge_sort_partition_count!(::<$key_type>(
                    __rccl_len
                ))?;
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace3::<_, usize, $key_type, $value_type>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count, __rccl_len, __rccl_len],
                    [256usize, 256usize, 256usize],
                )
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions, __rccl_scratch_keys, __rccl_scratch_values) =
                __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                $crate::__private::rccl_cub::DeviceMergeSortPairsWorkspace::<$key_type, $value_type, _, _, _>::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_scratch_values,
                    __rccl_partitions,
                );
            __rccl_execution.__synchronize($crate::__private::rccl_cub::device_merge_sort_pairs!(::<$key_type, $value_type>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_workspace,
                __rccl_compare,
            )))
        })()
    }};
}

#[macro_export]
macro_rules! stable_sort_by_key {
    (::<$key_type:ident, $value_type:ident>($execution:expr, $keys:expr, $values:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_keys = $keys;
            let __rccl_values = $values;
            let __rccl_len = $crate::DeviceRange::len(&*__rccl_keys);
            let __rccl_workspace =
                $crate::__private::rccl_cub::__private::__device_radix_sort_pairs_workspace_counts!(::<
                    $key_type,
                    $value_type
                >(__rccl_len));
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace5::<
                    _, $key_type, $value_type, u64, i32, i32
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
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (
                __rccl_scratch_keys,
                __rccl_scratch_values,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            ) = __rccl_temporary.parts_mut();
            let mut __rccl_storage =
                $crate::__private::rccl_cub::DeviceRadixSortPairsWorkspace::<
                    $key_type, $value_type, _, _, _, _, _
                >::__from_parts(
                __rccl_scratch_keys,
                __rccl_scratch_values,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            );
            let __rccl_result = $crate::__private::rccl_cub::device_radix_sort_pairs!(::<$key_type, $value_type>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_storage,
            ));
            ::core::mem::drop(__rccl_storage);
            ::core::mem::drop(__rccl_temporary);
            __rccl_execution.__synchronize(__rccl_result)
        })()
    }};
}

#[macro_export]
macro_rules! stable_sort_by_key_with {
    (::<$key_type:ident, $value_type:ident>($execution:expr, $keys:expr, $values:expr, $compare:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_keys = $keys;
            let __rccl_values = $values;
            let __rccl_compare = $compare;
            let __rccl_len = $crate::DeviceRange::len(&*__rccl_keys);
            let __rccl_partition_count =
                $crate::__private::rccl_cub::__private::__device_merge_sort_partition_count!(::<$key_type>(
                    __rccl_len
                ))?;
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace3::<_, usize, $key_type, $value_type>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count, __rccl_len, __rccl_len],
                    [256usize, 256usize, 256usize],
                )
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions, __rccl_scratch_keys, __rccl_scratch_values) =
                __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                $crate::__private::rccl_cub::DeviceMergeSortPairsWorkspace::<$key_type, $value_type, _, _, _>::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_scratch_values,
                    __rccl_partitions,
                );
            __rccl_execution.__synchronize($crate::__private::rccl_cub::device_stable_merge_sort_pairs!(::<$key_type, $value_type>(
                __rccl_execution,
                __rccl_keys,
                __rccl_values,
                &mut __rccl_workspace,
                __rccl_compare,
            )))
        })()
    }};
}
