// These macros contain fixed Rust scaffolding only. The nested CUB operation
// macros retain ownership of target/type policy selection and exact launch
// graph expansion.

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_sort_backend {
    ($module:ident, $key:ident) => {
        impl $crate::__private::RadixSortBackend<$key> for $module::LoadedModule {
            fn radix_workspace_counts(
                &self,
                len: usize,
            ) -> $crate::__private::rccl_cub::RadixSortWorkspaceCounts {
                $crate::__private::rccl_cub::__private::__device_radix_sort_workspace_counts!(::<
                    $key
                >(len))
            }

            fn radix_sort<A: $crate::memory::DeviceAllocator>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                keys: &mut $crate::__private::cuda_core::DeviceBuffer<$key>,
                scratch_keys: &mut $crate::memory::TemporaryRegion<'_, $key>,
                bins: &mut $crate::memory::TemporaryRegion<'_, u64>,
                lookback: &mut $crate::memory::TemporaryRegion<'_, i32>,
                counters: &mut $crate::memory::TemporaryRegion<'_, i32>,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError> {
                let mut workspace = $crate::__private::rccl_cub::DeviceRadixSortKeysWorkspace::<
                    $key,
                    _,
                    _,
                    _,
                    _,
                >::__from_parts(scratch_keys, bins, lookback, counters);
                $crate::__private::rccl_cub::device_radix_sort_keys!(::<$key>(
                    execution,
                    keys,
                    &mut workspace,
                ))
            }

        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_comparison_sort_backend {
    ($module:ident, $key:ident) => {
        impl $crate::__private::ComparisonSortBackend<$key> for $module::LoadedModule {
            fn merge_sort_partition_count(
                &self,
                len: usize,
            ) -> ::core::result::Result<usize, $crate::DeviceAlgorithmError> {
                $crate::__private::rccl_cub::__private::__device_merge_sort_partition_count!(::<
                    $key
                >(len))
            }

            fn merge_sort<A, F>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                keys: &mut $crate::__private::cuda_core::DeviceBuffer<$key>,
                scratch_keys: &mut $crate::memory::TemporaryRegion<'_, $key>,
                partitions: &mut $crate::memory::TemporaryRegion<'_, usize>,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$key, &$key) -> ::core::cmp::Ordering + Copy,
            {
                let mut workspace =
                    $crate::__private::rccl_cub::DeviceMergeSortKeysWorkspace::<$key, _, _>::__from_parts(
                        scratch_keys,
                        partitions,
                    );
                $crate::__private::rccl_cub::device_merge_sort_keys!(::<$key>(
                    execution,
                    keys,
                    &mut workspace,
                    order,
                ))
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_sort_pairs_backend {
    ($module:ident, $key:ident, $value:ident) => {
        impl $crate::__private::RadixSortPairsBackend<$key, $value> for $module::LoadedModule {
            fn radix_pairs_workspace_counts(
                &self,
                len: usize,
            ) -> $crate::__private::rccl_cub::RadixSortPairsWorkspaceCounts {
                $crate::__private::rccl_cub::__private::__device_radix_sort_pairs_workspace_counts!(::<
                    $key,
                    $value
                >(len))
            }

            fn radix_sort_pairs<A: $crate::memory::DeviceAllocator>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                keys: &mut $crate::__private::cuda_core::DeviceBuffer<$key>,
                values: &mut $crate::__private::cuda_core::DeviceBuffer<$value>,
                scratch_keys: &mut $crate::memory::TemporaryRegion<'_, $key>,
                scratch_values: &mut $crate::memory::TemporaryRegion<'_, $value>,
                bins: &mut $crate::memory::TemporaryRegion<'_, u64>,
                lookback: &mut $crate::memory::TemporaryRegion<'_, i32>,
                counters: &mut $crate::memory::TemporaryRegion<'_, i32>,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError> {
                let mut workspace =
                    $crate::__private::rccl_cub::DeviceRadixSortPairsWorkspace::<
                        $key,
                        $value,
                        _,
                        _,
                        _,
                        _,
                        _,
                    >::__from_parts(
                        scratch_keys,
                        scratch_values,
                        bins,
                        lookback,
                        counters,
                    );
                $crate::__private::rccl_cub::device_radix_sort_pairs!(::<$key, $value>(
                    execution,
                    keys,
                    values,
                    &mut workspace,
                ))
            }

        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_comparison_sort_pairs_backend {
    ($module:ident, $key:ident, $value:ident) => {
        impl $crate::__private::ComparisonSortPairsBackend<$key, $value>
            for $module::LoadedModule
        {
            fn merge_sort_pairs_partition_count(
                &self,
                len: usize,
            ) -> ::core::result::Result<usize, $crate::DeviceAlgorithmError> {
                $crate::__private::rccl_cub::__private::__device_merge_sort_partition_count!(::<
                    $key
                >(len))
            }

            fn merge_sort_pairs<A, F>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                keys: &mut $crate::__private::cuda_core::DeviceBuffer<$key>,
                values: &mut $crate::__private::cuda_core::DeviceBuffer<$value>,
                scratch_keys: &mut $crate::memory::TemporaryRegion<'_, $key>,
                scratch_values: &mut $crate::memory::TemporaryRegion<'_, $value>,
                partitions: &mut $crate::memory::TemporaryRegion<'_, usize>,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$key, &$key) -> ::core::cmp::Ordering + Copy,
            {
                let mut workspace =
                    $crate::__private::rccl_cub::DeviceMergeSortPairsWorkspace::<
                        $key,
                        $value,
                        _,
                        _,
                        _,
                    >::__from_parts(scratch_keys, scratch_values, partitions);
                $crate::__private::rccl_cub::device_merge_sort_pairs!(::<$key, $value>(
                    execution,
                    keys,
                    values,
                    &mut workspace,
                    order,
                ))
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_merge_backend {
    ($module:ident, $key:ident) => {
        impl $crate::__private::MergeBackend<$key> for $module::LoadedModule {
            fn merge_partition_count(
                &self,
                first_len: usize,
                second_len: usize,
            ) -> ::core::result::Result<usize, $crate::DeviceAlgorithmError> {
                $crate::__private::rccl_cub::__private::__device_merge_keys_partition_count!(::<
                    $key
                >(first_len, second_len))
            }

            fn merge<A, F>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first: &$crate::__private::cuda_core::DeviceBuffer<$key>,
                second: &$crate::__private::cuda_core::DeviceBuffer<$key>,
                partitions: &mut $crate::memory::TemporaryRegion<'_, usize>,
                output: &mut $crate::__private::cuda_core::DeviceBuffer<$key>,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$key, &$key) -> ::core::cmp::Ordering + Copy,
            {
                let mut workspace =
                    $crate::__private::rccl_cub::DeviceMergeWorkspace::__from_partitions(
                        partitions,
                    );
                $crate::__private::rccl_cub::device_merge_keys!(::<$key>(
                    execution,
                    first,
                    second,
                    &mut workspace,
                    output,
                    order,
                ))
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_merge_pairs_backend {
    ($module:ident, $key:ident, $value:ident) => {
        impl $crate::__private::MergePairsBackend<$key, $value> for $module::LoadedModule {
            fn merge_pairs_partition_count(
                &self,
                first_len: usize,
                second_len: usize,
            ) -> ::core::result::Result<usize, $crate::DeviceAlgorithmError> {
                $crate::__private::rccl_cub::__private::__device_merge_pairs_partition_count!(::<
                    $key,
                    $value
                >(first_len, second_len))
            }

            fn merge_pairs<A, F>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first_keys: &$crate::__private::cuda_core::DeviceBuffer<$key>,
                first_values: &$crate::__private::cuda_core::DeviceBuffer<$value>,
                second_keys: &$crate::__private::cuda_core::DeviceBuffer<$key>,
                second_values: &$crate::__private::cuda_core::DeviceBuffer<$value>,
                partitions: &mut $crate::memory::TemporaryRegion<'_, usize>,
                output_keys: &mut $crate::__private::cuda_core::DeviceBuffer<$key>,
                output_values: &mut $crate::__private::cuda_core::DeviceBuffer<$value>,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$key, &$key) -> ::core::cmp::Ordering + Copy,
            {
                let mut workspace =
                    $crate::__private::rccl_cub::DeviceMergeWorkspace::__from_partitions(
                        partitions,
                    );
                $crate::__private::rccl_cub::device_merge_pairs!(::<$key, $value>(
                    execution,
                    first_keys,
                    first_values,
                    second_keys,
                    second_values,
                    &mut workspace,
                    output_keys,
                    output_values,
                    order,
                ))
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_set_backend {
    ($module:ident, $($key:tt)+) => {
        impl $crate::__private::SetBackend<$($key)+> for $module::LoadedModule {
            fn set_workspace_counts<A: $crate::memory::DeviceAllocator>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first_len: usize,
                second_len: usize,
            ) -> ::core::result::Result<
                $crate::__private::rccl_cub::SetOperationWorkspaceCounts,
                $crate::DeviceAlgorithmError,
            > {
                $crate::__private::rccl_cub::__private::__device_set_workspace_counts!(::<
                    $($key)+,
                    $($key)+
                >(execution, first_len, second_len))
            }

            fn set_i32<A, F, Op>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first: &$crate::__private::cuda_core::DeviceBuffer<$($key)+>,
                second: &$crate::__private::cuda_core::DeviceBuffer<$($key)+>,
                workspace: &mut $crate::__private::rccl_cub::DeviceSetOperationWorkspace<
                    &mut $crate::memory::TemporaryRegion<'_, usize>,
                    &mut $crate::memory::TemporaryRegion<'_, u64>,
                    &mut $crate::memory::TemporaryRegion<'_, u64>,
                    &mut $crate::memory::TemporaryRegion<'_, u8>,
                >,
                output: &mut $crate::__private::cuda_core::DeviceBuffer<$($key)+>,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$($key)+, &$($key)+) -> ::core::cmp::Ordering + Copy,
                Op: $crate::__private::rccl_cub::__private::SerialSetOperation,
            {
                $crate::__private::rccl_cub::__private::__device_set_keys_i32!(::<$($key)+, Op>(
                    execution, first, second, workspace, output, order,
                ))
            }

            fn set_i64<A, F, Op>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first: &$crate::__private::cuda_core::DeviceBuffer<$($key)+>,
                second: &$crate::__private::cuda_core::DeviceBuffer<$($key)+>,
                workspace: &mut $crate::__private::rccl_cub::DeviceSetOperationWorkspace<
                    &mut $crate::memory::TemporaryRegion<'_, usize>,
                    &mut $crate::memory::TemporaryRegion<'_, u128>,
                    &mut $crate::memory::TemporaryRegion<'_, u128>,
                    &mut $crate::memory::TemporaryRegion<'_, u8>,
                >,
                output: &mut $crate::__private::cuda_core::DeviceBuffer<$($key)+>,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$($key)+, &$($key)+) -> ::core::cmp::Ordering + Copy,
                Op: $crate::__private::rccl_cub::__private::SerialSetOperation,
            {
                $crate::__private::rccl_cub::__private::__device_set_keys_i64!(::<$($key)+, Op>(
                    execution, first, second, workspace, output, order,
                ))
            }

            fn set_discard_i32<A, F, Op>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first: &$crate::__private::cuda_core::DeviceBuffer<$($key)+>,
                second: &$crate::__private::cuda_core::DeviceBuffer<$($key)+>,
                workspace: &mut $crate::__private::rccl_cub::DeviceSetOperationWorkspace<
                    &mut $crate::memory::TemporaryRegion<'_, usize>,
                    &mut $crate::memory::TemporaryRegion<'_, u64>,
                    &mut $crate::memory::TemporaryRegion<'_, u64>,
                    &mut $crate::memory::TemporaryRegion<'_, u8>,
                >,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$($key)+, &$($key)+) -> ::core::cmp::Ordering + Copy,
                Op: $crate::__private::rccl_cub::__private::SerialSetOperation,
            {
                $crate::__private::rccl_cub::__private::__device_set_discard_i32!(::<$($key)+, Op>(
                    execution, first, second, workspace, order,
                ))
            }

            fn set_discard_i64<A, F, Op>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first: &$crate::__private::cuda_core::DeviceBuffer<$($key)+>,
                second: &$crate::__private::cuda_core::DeviceBuffer<$($key)+>,
                workspace: &mut $crate::__private::rccl_cub::DeviceSetOperationWorkspace<
                    &mut $crate::memory::TemporaryRegion<'_, usize>,
                    &mut $crate::memory::TemporaryRegion<'_, u128>,
                    &mut $crate::memory::TemporaryRegion<'_, u128>,
                    &mut $crate::memory::TemporaryRegion<'_, u8>,
                >,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$($key)+, &$($key)+) -> ::core::cmp::Ordering + Copy,
                Op: $crate::__private::rccl_cub::__private::SerialSetOperation,
            {
                $crate::__private::rccl_cub::__private::__device_set_discard_i64!(::<$($key)+, Op>(
                    execution, first, second, workspace, order,
                ))
            }

        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_arithmetic_set_backend {
    ($module:ident, $key:ident) => {
        impl $crate::__private::ArithmeticSetBackend<$key> for $module::LoadedModule {
            fn set_arithmetic_i32<A, F, Op>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first: $crate::range::ArithmeticRange<$key>,
                second: $crate::range::ArithmeticRange<$key>,
                workspace: &mut $crate::__private::rccl_cub::DeviceSetOperationWorkspace<
                    &mut $crate::memory::TemporaryRegion<'_, usize>,
                    &mut $crate::memory::TemporaryRegion<'_, u64>,
                    &mut $crate::memory::TemporaryRegion<'_, u64>,
                    &mut $crate::memory::TemporaryRegion<'_, u8>,
                >,
                output: &mut $crate::__private::cuda_core::DeviceBuffer<$key>,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$key, &$key) -> ::core::cmp::Ordering + Copy,
                Op: $crate::__private::rccl_cub::__private::SerialSetOperation,
            {
                $crate::__private::rccl_cub::__private::__device_set_arithmetic_i32!(::<$key, Op>(
                    execution, first, second, workspace, output, order,
                ))
            }

            fn set_arithmetic_i64<A, F, Op>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first: $crate::range::ArithmeticRange<$key>,
                second: $crate::range::ArithmeticRange<$key>,
                workspace: &mut $crate::__private::rccl_cub::DeviceSetOperationWorkspace<
                    &mut $crate::memory::TemporaryRegion<'_, usize>,
                    &mut $crate::memory::TemporaryRegion<'_, u128>,
                    &mut $crate::memory::TemporaryRegion<'_, u128>,
                    &mut $crate::memory::TemporaryRegion<'_, u8>,
                >,
                output: &mut $crate::__private::cuda_core::DeviceBuffer<$key>,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$key, &$key) -> ::core::cmp::Ordering + Copy,
                Op: $crate::__private::rccl_cub::__private::SerialSetOperation,
            {
                $crate::__private::rccl_cub::__private::__device_set_arithmetic_i64!(::<$key, Op>(
                    execution, first, second, workspace, output, order,
                ))
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_set_pairs_backend {
    ($module:ident, $key:ident, $value:ident) => {
        impl $crate::__private::SetPairsBackend<$key, $value> for $module::LoadedModule {
            fn set_pairs_workspace_counts<A: $crate::memory::DeviceAllocator>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first_len: usize,
                second_len: usize,
            ) -> ::core::result::Result<
                $crate::__private::rccl_cub::SetOperationWorkspaceCounts,
                $crate::DeviceAlgorithmError,
            > {
                $crate::__private::rccl_cub::__private::__device_set_workspace_counts!(::<
                    $key,
                    $value
                >(execution, first_len, second_len))
            }

            fn set_pairs_i32<A, F, Op>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first_keys: &$crate::__private::cuda_core::DeviceBuffer<$key>,
                first_values: &$crate::__private::cuda_core::DeviceBuffer<$value>,
                second_keys: &$crate::__private::cuda_core::DeviceBuffer<$key>,
                second_values: &$crate::__private::cuda_core::DeviceBuffer<$value>,
                workspace: &mut $crate::__private::rccl_cub::DeviceSetOperationWorkspace<
                    &mut $crate::memory::TemporaryRegion<'_, usize>,
                    &mut $crate::memory::TemporaryRegion<'_, u64>,
                    &mut $crate::memory::TemporaryRegion<'_, u64>,
                    &mut $crate::memory::TemporaryRegion<'_, u8>,
                >,
                output_keys: &mut $crate::__private::cuda_core::DeviceBuffer<$key>,
                output_values: &mut $crate::__private::cuda_core::DeviceBuffer<$value>,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$key, &$key) -> ::core::cmp::Ordering + Copy,
                Op: $crate::__private::rccl_cub::__private::SerialSetOperation,
            {
                $crate::__private::rccl_cub::__private::__device_set_pairs_i32!(::<
                    $key,
                    $value,
                    Op
                >(
                    execution,
                    first_keys,
                    first_values,
                    second_keys,
                    second_values,
                    workspace,
                    output_keys,
                    output_values,
                    order,
                ))
            }

            fn set_pairs_i64<A, F, Op>(
                &self,
                execution: &$crate::cuda::Execution<'_, '_, A, Self>,
                first_keys: &$crate::__private::cuda_core::DeviceBuffer<$key>,
                first_values: &$crate::__private::cuda_core::DeviceBuffer<$value>,
                second_keys: &$crate::__private::cuda_core::DeviceBuffer<$key>,
                second_values: &$crate::__private::cuda_core::DeviceBuffer<$value>,
                workspace: &mut $crate::__private::rccl_cub::DeviceSetOperationWorkspace<
                    &mut $crate::memory::TemporaryRegion<'_, usize>,
                    &mut $crate::memory::TemporaryRegion<'_, u128>,
                    &mut $crate::memory::TemporaryRegion<'_, u128>,
                    &mut $crate::memory::TemporaryRegion<'_, u8>,
                >,
                output_keys: &mut $crate::__private::cuda_core::DeviceBuffer<$key>,
                output_values: &mut $crate::__private::cuda_core::DeviceBuffer<$value>,
                order: F,
            ) -> ::core::result::Result<(), $crate::DeviceAlgorithmError>
            where
                A: $crate::memory::DeviceAllocator,
                F: Fn(&$key, &$key) -> ::core::cmp::Ordering + Copy,
                Op: $crate::__private::rccl_cub::__private::SerialSetOperation,
            {
                $crate::__private::rccl_cub::__private::__device_set_pairs_i64!(::<
                    $key,
                    $value,
                    Op
                >(
                    execution,
                    first_keys,
                    first_values,
                    second_keys,
                    second_values,
                    workspace,
                    output_keys,
                    output_values,
                    order,
                ))
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_sort_pairs_for_key {
    ($module:ident, $key:ident) => {
        $crate::__thrust_sort_pairs_backend!($module, $key, u8);
        $crate::__thrust_sort_pairs_backend!($module, $key, u16);
        $crate::__thrust_sort_pairs_backend!($module, $key, u32);
        $crate::__thrust_sort_pairs_backend!($module, $key, u64);
        $crate::__thrust_sort_pairs_backend!($module, $key, usize);
        $crate::__thrust_sort_pairs_backend!($module, $key, i8);
        $crate::__thrust_sort_pairs_backend!($module, $key, i16);
        $crate::__thrust_sort_pairs_backend!($module, $key, i32);
        $crate::__thrust_sort_pairs_backend!($module, $key, i64);
        $crate::__thrust_sort_pairs_backend!($module, $key, isize);
        $crate::__thrust_sort_pairs_backend!($module, $key, f32);
        $crate::__thrust_sort_pairs_backend!($module, $key, f64);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_comparison_sort_pairs_for_narrow_key {
    ($module:ident, $key:ident) => {
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, u8);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, u16);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, u32);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, u64);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, usize);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, i8);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, i16);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, i32);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, i64);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, isize);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, f32);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, f64);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_comparison_sort_pairs_for_wide_key {
    ($module:ident, $key:ident) => {
        $crate::__thrust_comparison_sort_pairs_for_narrow_key!($module, $key);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, u128);
        $crate::__thrust_comparison_sort_pairs_backend!($module, $key, i128);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_merge_pairs_for_key {
    ($module:ident, $key:ident) => {
        $crate::__thrust_merge_pairs_backend!($module, $key, u8);
        $crate::__thrust_merge_pairs_backend!($module, $key, u16);
        $crate::__thrust_merge_pairs_backend!($module, $key, u32);
        $crate::__thrust_merge_pairs_backend!($module, $key, u64);
        $crate::__thrust_merge_pairs_backend!($module, $key, u128);
        $crate::__thrust_merge_pairs_backend!($module, $key, usize);
        $crate::__thrust_merge_pairs_backend!($module, $key, i8);
        $crate::__thrust_merge_pairs_backend!($module, $key, i16);
        $crate::__thrust_merge_pairs_backend!($module, $key, i32);
        $crate::__thrust_merge_pairs_backend!($module, $key, i64);
        $crate::__thrust_merge_pairs_backend!($module, $key, i128);
        $crate::__thrust_merge_pairs_backend!($module, $key, isize);
        $crate::__thrust_merge_pairs_backend!($module, $key, f32);
        $crate::__thrust_merge_pairs_backend!($module, $key, f64);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_set_pairs_for_key {
    ($module:ident, $key:ident) => {
        $crate::__thrust_set_pairs_backend!($module, $key, u8);
        $crate::__thrust_set_pairs_backend!($module, $key, u16);
        $crate::__thrust_set_pairs_backend!($module, $key, u32);
        $crate::__thrust_set_pairs_backend!($module, $key, u64);
        $crate::__thrust_set_pairs_backend!($module, $key, u128);
        $crate::__thrust_set_pairs_backend!($module, $key, usize);
        $crate::__thrust_set_pairs_backend!($module, $key, i8);
        $crate::__thrust_set_pairs_backend!($module, $key, i16);
        $crate::__thrust_set_pairs_backend!($module, $key, i32);
        $crate::__thrust_set_pairs_backend!($module, $key, i64);
        $crate::__thrust_set_pairs_backend!($module, $key, i128);
        $crate::__thrust_set_pairs_backend!($module, $key, isize);
        $crate::__thrust_set_pairs_backend!($module, $key, f32);
        $crate::__thrust_set_pairs_backend!($module, $key, f64);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_merge_backends {
    ($module:ident) => {
        $crate::__thrust_merge_backend!($module, u8);
        $crate::__thrust_merge_backend!($module, u16);
        $crate::__thrust_merge_backend!($module, u32);
        $crate::__thrust_merge_backend!($module, u64);
        $crate::__thrust_merge_backend!($module, u128);
        $crate::__thrust_merge_backend!($module, usize);
        $crate::__thrust_merge_backend!($module, i8);
        $crate::__thrust_merge_backend!($module, i16);
        $crate::__thrust_merge_backend!($module, i32);
        $crate::__thrust_merge_backend!($module, i64);
        $crate::__thrust_merge_backend!($module, i128);
        $crate::__thrust_merge_backend!($module, isize);
        $crate::__thrust_merge_backend!($module, f32);
        $crate::__thrust_merge_backend!($module, f64);

        $crate::__thrust_merge_pairs_for_key!($module, u8);
        $crate::__thrust_merge_pairs_for_key!($module, u16);
        $crate::__thrust_merge_pairs_for_key!($module, u32);
        $crate::__thrust_merge_pairs_for_key!($module, u64);
        $crate::__thrust_merge_pairs_for_key!($module, u128);
        $crate::__thrust_merge_pairs_for_key!($module, usize);
        $crate::__thrust_merge_pairs_for_key!($module, i8);
        $crate::__thrust_merge_pairs_for_key!($module, i16);
        $crate::__thrust_merge_pairs_for_key!($module, i32);
        $crate::__thrust_merge_pairs_for_key!($module, i64);
        $crate::__thrust_merge_pairs_for_key!($module, i128);
        $crate::__thrust_merge_pairs_for_key!($module, isize);
        $crate::__thrust_merge_pairs_for_key!($module, f32);
        $crate::__thrust_merge_pairs_for_key!($module, f64);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_set_backends {
    ($module:ident) => {
        $crate::__thrust_set_backend!($module, u8);
        $crate::__thrust_set_backend!($module, u16);
        $crate::__thrust_set_backend!($module, u32);
        $crate::__thrust_set_backend!($module, u64);
        $crate::__thrust_set_backend!($module, u128);
        $crate::__thrust_set_backend!($module, usize);
        $crate::__thrust_set_backend!($module, i8);
        $crate::__thrust_set_backend!($module, i16);
        $crate::__thrust_set_backend!($module, i32);
        $crate::__thrust_set_backend!($module, i64);
        $crate::__thrust_set_backend!($module, i128);
        $crate::__thrust_set_backend!($module, isize);
        $crate::__thrust_set_backend!($module, f32);
        $crate::__thrust_set_backend!($module, f64);
        $crate::__thrust_set_backend!(
            $module,
            $crate::__private::rccl_cub::KeyValue<i8, i8>
        );
        $crate::__thrust_set_backend!(
            $module,
            $crate::__private::rccl_cub::KeyValue<u8, u8>
        );
        $crate::__thrust_set_backend!(
            $module,
            $crate::__private::rccl_cub::KeyValue<i16, i16>
        );
        $crate::__thrust_set_backend!(
            $module,
            $crate::__private::rccl_cub::KeyValue<u16, u16>
        );
        $crate::__thrust_set_backend!(
            $module,
            $crate::__private::rccl_cub::KeyValue<i32, i32>
        );
        $crate::__thrust_set_backend!(
            $module,
            $crate::__private::rccl_cub::KeyValue<u32, u32>
        );
        $crate::__thrust_set_backend!(
            $module,
            $crate::__private::rccl_cub::KeyValue<f32, f32>
        );
        $crate::__thrust_set_backend!(
            $module,
            $crate::__private::rccl_cub::KeyValue<f64, f64>
        );

        $crate::__thrust_arithmetic_set_backend!($module, u8);
        $crate::__thrust_arithmetic_set_backend!($module, u16);
        $crate::__thrust_arithmetic_set_backend!($module, u32);
        $crate::__thrust_arithmetic_set_backend!($module, u64);
        $crate::__thrust_arithmetic_set_backend!($module, u128);
        $crate::__thrust_arithmetic_set_backend!($module, usize);
        $crate::__thrust_arithmetic_set_backend!($module, i8);
        $crate::__thrust_arithmetic_set_backend!($module, i16);
        $crate::__thrust_arithmetic_set_backend!($module, i32);
        $crate::__thrust_arithmetic_set_backend!($module, i64);
        $crate::__thrust_arithmetic_set_backend!($module, i128);
        $crate::__thrust_arithmetic_set_backend!($module, isize);

        $crate::__thrust_set_pairs_for_key!($module, u8);
        $crate::__thrust_set_pairs_for_key!($module, u16);
        $crate::__thrust_set_pairs_for_key!($module, u32);
        $crate::__thrust_set_pairs_for_key!($module, u64);
        $crate::__thrust_set_pairs_for_key!($module, u128);
        $crate::__thrust_set_pairs_for_key!($module, usize);
        $crate::__thrust_set_pairs_for_key!($module, i8);
        $crate::__thrust_set_pairs_for_key!($module, i16);
        $crate::__thrust_set_pairs_for_key!($module, i32);
        $crate::__thrust_set_pairs_for_key!($module, i64);
        $crate::__thrust_set_pairs_for_key!($module, i128);
        $crate::__thrust_set_pairs_for_key!($module, isize);
        $crate::__thrust_set_pairs_for_key!($module, f32);
        $crate::__thrust_set_pairs_for_key!($module, f64);
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __thrust_backends {
    ($module:ident) => {
        $crate::__thrust_sort_backend!($module, u32);
        $crate::__thrust_sort_backend!($module, u64);
        $crate::__thrust_sort_backend!($module, i32);
        $crate::__thrust_sort_backend!($module, i64);
        $crate::__thrust_sort_backend!($module, f32);
        $crate::__thrust_sort_backend!($module, f64);

        $crate::__thrust_comparison_sort_backend!($module, u8);
        $crate::__thrust_comparison_sort_backend!($module, u16);
        $crate::__thrust_comparison_sort_backend!($module, u32);
        $crate::__thrust_comparison_sort_backend!($module, u64);
        $crate::__thrust_comparison_sort_backend!($module, u128);
        $crate::__thrust_comparison_sort_backend!($module, usize);
        $crate::__thrust_comparison_sort_backend!($module, i8);
        $crate::__thrust_comparison_sort_backend!($module, i16);
        $crate::__thrust_comparison_sort_backend!($module, i32);
        $crate::__thrust_comparison_sort_backend!($module, i64);
        $crate::__thrust_comparison_sort_backend!($module, i128);
        $crate::__thrust_comparison_sort_backend!($module, isize);
        $crate::__thrust_comparison_sort_backend!($module, f32);
        $crate::__thrust_comparison_sort_backend!($module, f64);

        $crate::__thrust_merge_backends!($module);

        $crate::__thrust_set_backends!($module);

        $crate::__thrust_sort_pairs_for_key!($module, u32);
        $crate::__thrust_sort_pairs_for_key!($module, u64);
        $crate::__thrust_sort_pairs_for_key!($module, i32);
        $crate::__thrust_sort_pairs_for_key!($module, i64);
        $crate::__thrust_sort_pairs_for_key!($module, f32);
        $crate::__thrust_sort_pairs_for_key!($module, f64);

        $crate::__thrust_comparison_sort_pairs_for_narrow_key!($module, u8);
        $crate::__thrust_comparison_sort_pairs_for_narrow_key!($module, u16);
        $crate::__thrust_comparison_sort_pairs_for_narrow_key!($module, u32);
        $crate::__thrust_comparison_sort_pairs_for_wide_key!($module, u64);
        $crate::__thrust_comparison_sort_pairs_for_wide_key!($module, u128);
        $crate::__thrust_comparison_sort_pairs_for_wide_key!($module, usize);
        $crate::__thrust_comparison_sort_pairs_for_narrow_key!($module, i8);
        $crate::__thrust_comparison_sort_pairs_for_narrow_key!($module, i16);
        $crate::__thrust_comparison_sort_pairs_for_narrow_key!($module, i32);
        $crate::__thrust_comparison_sort_pairs_for_wide_key!($module, i64);
        $crate::__thrust_comparison_sort_pairs_for_wide_key!($module, i128);
        $crate::__thrust_comparison_sort_pairs_for_wide_key!($module, isize);
        $crate::__thrust_comparison_sort_pairs_for_narrow_key!($module, f32);
        $crate::__thrust_comparison_sort_pairs_for_wide_key!($module, f64);
    };
}
