pub use crate::histogram::DeviceHistogramWorkspaceParts;
pub use crate::histogram::histogram_uses_i32_offsets;
pub use crate::histogram_backend::{
    ArithmeticHistogramRange, HistogramOffset, HistogramRead, HistogramSample,
    histogram_init_agent, histogram_range_bin, histogram_sweep_agent,
};
/// Implementation details used by RCCL's operation macros.
///
/// This is public only because a proc-macro expansion is compiled in the
/// caller's crate. It is not a compatibility API.
#[doc(hidden)]
pub use crate::ordering_backend::{
    ArithmeticSetRange, SerialSetOperation, SetAgentState, SetDifference, SetIndex,
    SetIntersection, SetPairOutput, SetReadRange, SetSymmetricDifference, SetUnion,
    global_merge_path, merge_keys_agent, merge_pairs_agent, merge_sort_block_keys_agent,
    merge_sort_block_keys_in_place_agent, merge_sort_block_pairs_agent,
    merge_sort_block_pairs_in_place_agent, merge_sort_merge_keys_agent,
    merge_sort_merge_pairs_agent, merge_sort_partition_point, radix_default_u32, radix_default_u64,
    radix_digit_masked_u32, radix_digit_masked_u64, radix_digit_u32, radix_digit_u64,
    radix_exclusive_sum_agent, radix_f32_from_ordered_ascending, radix_f32_from_ordered_descending,
    radix_f32_to_ordered_ascending, radix_f32_to_ordered_descending,
    radix_f64_from_ordered_ascending, radix_f64_from_ordered_descending,
    radix_f64_to_ordered_ascending, radix_f64_to_ordered_descending, radix_float_digit_masked_u32,
    radix_float_digit_masked_u64, radix_float_digit_u32, radix_float_digit_u64,
    radix_float_rank_digit_u32, radix_float_rank_digit_u64, radix_histogram_agent,
    radix_i32_from_ordered_ascending, radix_i32_from_ordered_descending,
    radix_i32_to_ordered_ascending, radix_i32_to_ordered_descending,
    radix_i64_from_ordered_ascending, radix_i64_from_ordered_descending,
    radix_i64_to_ordered_ascending, radix_i64_to_ordered_descending,
    radix_init_bins_and_counters_agent, radix_min_ordered_u32, radix_min_ordered_u64,
    radix_onesweep_keys_agent, radix_onesweep_pairs_agent, radix_rank_digit_u32,
    radix_rank_digit_u64, radix_single_tile_keys_agent, radix_single_tile_pairs_agent,
    radix_u32_from_ordered_ascending, radix_u32_from_ordered_descending,
    radix_u32_to_ordered_ascending, radix_u32_to_ordered_descending,
    radix_u64_from_ordered_ascending, radix_u64_from_ordered_descending,
    radix_u64_to_ordered_ascending, radix_u64_to_ordered_descending, set_arithmetic_keys_agent,
    set_init_agent, set_keys_agent, set_keys_discard_agent, set_pairs_agent, set_partition_agent,
};
pub use cuda_core;
pub use cuda_device;
pub use cuda_host;
#[doc(hidden)]
pub use rccl_cub_macros::{
    __device_kernels, __device_merge_keys_partition_count, __device_merge_pairs_partition_count,
    __device_merge_sort_partition_count, __device_radix_sort_pairs_workspace_counts,
    __device_radix_sort_workspace_counts, __device_set_arithmetic_i32, __device_set_arithmetic_i64,
    __device_set_discard_i32, __device_set_discard_i64, __device_set_keys_i32,
    __device_set_keys_i64, __device_set_pairs_i32, __device_set_pairs_i64,
    __device_set_workspace_counts,
};
pub use rccl_memory::device::{BlockSharedMemory, HistogramCounter};
pub use rccl_memory::{copy_device_range, static_block_shared_memory};
/// Normalize cuda-oxide and RCCL allocator buffers for CUB's required
/// stream-ordered lookback memset. Implemented only for the signed
/// 32-bit protocol word, so byte-zeroing cannot create an invalid value.
pub trait ZeroRadixLookback {
    fn zero_radix_lookback(
        &mut self,
        stream: &cuda_core::CudaStream,
    ) -> Result<(), crate::DeviceAlgorithmError>;
}

impl ZeroRadixLookback for cuda_core::DeviceBuffer<i32> {
    fn zero_radix_lookback(
        &mut self,
        stream: &cuda_core::CudaStream,
    ) -> Result<(), crate::DeviceAlgorithmError> {
        self.zero_async(stream)
            .map_err(crate::DeviceAlgorithmError::Driver)
    }
}

impl<A: rccl_memory::DeviceAllocator> ZeroRadixLookback for rccl_memory::DeviceBuffer<'_, i32, A> {
    fn zero_radix_lookback(
        &mut self,
        stream: &cuda_core::CudaStream,
    ) -> Result<(), crate::DeviceAlgorithmError> {
        rccl_memory::ContiguousDeviceRange::ensure_usable_on(self, stream)
            .map_err(crate::DeviceAlgorithmError::Memory)?;
        self.zero_async()
            .map_err(crate::DeviceAlgorithmError::Memory)
    }
}

impl ZeroRadixLookback for rccl_memory::TemporaryRegion<'_, i32> {
    fn zero_radix_lookback(
        &mut self,
        stream: &cuda_core::CudaStream,
    ) -> Result<(), crate::DeviceAlgorithmError> {
        rccl_memory::ContiguousDeviceRange::ensure_usable_on(self, stream)
            .map_err(crate::DeviceAlgorithmError::Memory)?;
        self.zero_async()
            .map_err(crate::DeviceAlgorithmError::Memory)
    }
}
