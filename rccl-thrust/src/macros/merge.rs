#[macro_export]
macro_rules! merge {
    (::<$value_type:ident>($execution:expr, $first:expr, $second:expr, $output:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_first = $first;
            let __rccl_second = $second;
            let __rccl_output = $output;
            let __rccl_partition_count =
                $crate::__private::rccl_cub::__private::__device_merge_keys_partition_count!(::<$value_type>(
                    $crate::DeviceRange::len(&__rccl_first),
                    $crate::DeviceRange::len(&__rccl_second),
                ))?;
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace1::<_, usize>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count],
                    [256usize],
                )
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions,) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                $crate::__private::rccl_cub::DeviceMergeWorkspace::<_>::__from_partitions(__rccl_partitions);
            __rccl_execution.__synchronize($crate::__private::rccl_cub::device_merge_keys!(::<$value_type>(
                __rccl_execution,
                __rccl_first,
                __rccl_second,
                &mut __rccl_workspace,
                __rccl_output,
                |__rccl_left: &$value_type, __rccl_right: &$value_type| {
                    if __rccl_left < __rccl_right {
                        ::core::cmp::Ordering::Less
                    } else if __rccl_right < __rccl_left {
                        ::core::cmp::Ordering::Greater
                    } else {
                        ::core::cmp::Ordering::Equal
                    }
                },
            )))
        })()
    }};
}

#[macro_export]
macro_rules! merge_by {
    (::<$value_type:ident>($execution:expr, $first:expr, $second:expr, $output:expr, $compare:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_first = $first;
            let __rccl_second = $second;
            let __rccl_output = $output;
            let __rccl_compare = $compare;
            let __rccl_partition_count =
                $crate::__private::rccl_cub::__private::__device_merge_keys_partition_count!(::<$value_type>(
                    $crate::DeviceRange::len(&__rccl_first),
                    $crate::DeviceRange::len(&__rccl_second),
                ))?;
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace1::<_, usize>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count],
                    [256usize],
                )
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions,) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                $crate::__private::rccl_cub::DeviceMergeWorkspace::<_>::__from_partitions(__rccl_partitions);
            __rccl_execution.__synchronize($crate::__private::rccl_cub::device_merge_keys!(::<$value_type>(
                __rccl_execution,
                __rccl_first,
                __rccl_second,
                &mut __rccl_workspace,
                __rccl_output,
                __rccl_compare,
            )))
        })()
    }};
}

#[macro_export]
macro_rules! merge_by_key {
    (::<$key_type:ident, $value_type:ident>($execution:expr, $first_keys:expr, $first_values:expr, $second_keys:expr, $second_values:expr, $output_keys:expr, $output_values:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_first_keys = $first_keys;
            let __rccl_first_values = $first_values;
            let __rccl_second_keys = $second_keys;
            let __rccl_second_values = $second_values;
            let __rccl_output_keys = $output_keys;
            let __rccl_output_values = $output_values;
            let __rccl_partition_count =
                $crate::__private::rccl_cub::__private::__device_merge_pairs_partition_count!(::<
                    $key_type,
                    $value_type
                >(
                    $crate::DeviceRange::len(&__rccl_first_keys),
                    $crate::DeviceRange::len(&__rccl_second_keys),
                ))?;
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace1::<_, usize>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count],
                    [256usize],
                )
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions,) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                $crate::__private::rccl_cub::DeviceMergeWorkspace::<_>::__from_partitions(__rccl_partitions);
            __rccl_execution.__synchronize($crate::__private::rccl_cub::device_merge_pairs!(::<$key_type, $value_type>(
                __rccl_execution,
                __rccl_first_keys,
                __rccl_first_values,
                __rccl_second_keys,
                __rccl_second_values,
                &mut __rccl_workspace,
                __rccl_output_keys,
                __rccl_output_values,
                |__rccl_left: &$key_type, __rccl_right: &$key_type| {
                    if __rccl_left < __rccl_right {
                        ::core::cmp::Ordering::Less
                    } else if __rccl_right < __rccl_left {
                        ::core::cmp::Ordering::Greater
                    } else {
                        ::core::cmp::Ordering::Equal
                    }
                },
            )))
        })()
    }};
}

#[macro_export]
macro_rules! merge_by_key_with {
    (::<$key_type:ident, $value_type:ident>($execution:expr, $first_keys:expr, $first_values:expr, $second_keys:expr, $second_values:expr, $output_keys:expr, $output_values:expr, $compare:expr $(,)?)) => {{
        let __rccl_execution = $execution;
(|| -> ::core::result::Result<
            (),
            $crate::DeviceAlgorithmError,
        > {
            let __rccl_first_keys = $first_keys;
            let __rccl_first_values = $first_values;
            let __rccl_second_keys = $second_keys;
            let __rccl_second_values = $second_values;
            let __rccl_output_keys = $output_keys;
            let __rccl_output_values = $output_values;
            let __rccl_compare = $compare;
            let __rccl_partition_count =
                $crate::__private::rccl_cub::__private::__device_merge_pairs_partition_count!(::<
                    $key_type,
                    $value_type
                >(
                    $crate::DeviceRange::len(&__rccl_first_keys),
                    $crate::DeviceRange::len(&__rccl_second_keys),
                ))?;
            let mut __rccl_temporary =
                $crate::memory::TemporaryWorkspace1::<_, usize>::new_in(
                    __rccl_execution.allocator(),
                    __rccl_execution.stream(),
                    [__rccl_partition_count],
                    [256usize],
                )
                .map_err($crate::DeviceAlgorithmError::Memory)?;
            let (__rccl_partitions,) = __rccl_temporary.parts_mut();
            let mut __rccl_workspace =
                $crate::__private::rccl_cub::DeviceMergeWorkspace::<_>::__from_partitions(__rccl_partitions);
            __rccl_execution.__synchronize($crate::__private::rccl_cub::device_merge_pairs!(::<$key_type, $value_type>(
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
        })()
    }};
}
