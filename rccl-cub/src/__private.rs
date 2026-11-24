pub use crate::block_scan::{
    BlockScanResult, BlockScanValue, block_scan_sum_128, block_scan_sum_256,
};
pub use crate::compaction::{CompactionIndex, CompactionWorkspace};
pub use crate::compaction_kernels::CompactionModule;
pub use crate::grouped_reduction::{
    BooleanOr, BooleanProvenance, GroupReduceOperator, IndependentProbabilityOr, NaturalBagAdd,
};
pub use crate::histogram::{
    DeviceHistogramWorkspaceParts, HistogramInitKernel, HistogramKernel, HistogramLaunchGeometry,
    HistogramLaunchStorage, HistogramLevelStorage, HistogramLevels, HistogramPolicy,
    HistogramPolicyKernels, HistogramScalar, histogram_policy, histogram_shared_bytes,
    histogram_uses_i32_offsets,
};
pub use crate::histogram_backend::{
    ArithmeticHistogramRange, EvenHistogramTransforms, HistogramDecode, HistogramFourReader,
    HistogramFourSampleLoad, HistogramOffset, HistogramRead, HistogramSample,
    RangeHistogramTransform, RangeHistogramTransforms, histogram_init_agent, histogram_range_bin,
    histogram_sweep_agent,
};
pub use crate::histogram_kernels::HistogramModule;
pub use crate::merge::{
    MergeKeysKernel, MergePairsKernel, merge_agent_launch, merge_items, merge_load_to_shared_keys,
    merge_partition_launch, merge_shared_bytes,
};
pub use crate::merge_sort::{
    MergeSortKeysKernel, MergeSortPairsKernel, merge_sort_items, merge_sort_shared_bytes,
};
/// Implementation details used by RCCL's separately compiled CUDA modules.
///
/// cuda-oxide compiles those modules as distinct crates, so their device code
/// needs a public path back to these items. This is not a compatibility API.
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
pub use crate::ordering_kernels::OrderingModule;
pub use crate::radix::{
    RadixKernel, RadixKey, RadixKeysKernel, RadixPairsKernel, RadixValue, radix_items,
    radix_shared_bytes, radix_single_items,
};
pub use crate::reduce_kernels::ReduceModule;
pub use crate::scan_kernels::ScanModule;
pub use crate::set_kernels::SetModule;
pub use crate::set_operation::{ArithmeticSetKernel, SetKernel, set_items, set_shared_bytes};
pub use crate::warp::{WarpValue, warp_reduce, warp_reduce_items};
pub use crate::workspace::{
    device_merge_pairs_partition_count, device_merge_pairs_workspace, device_merge_partition_count,
    device_merge_sort_keys_workspace, device_merge_sort_pairs_workspace,
    device_merge_sort_partition_count, device_merge_workspace,
    device_radix_sort_keys_buffer_workspace, device_radix_sort_keys_workspace,
    device_radix_sort_pairs_buffer_workspace, device_radix_sort_pairs_workspace,
    device_radix_sort_pairs_workspace_counts, device_radix_sort_workspace_counts,
    device_set_workspace_counts,
};
pub use cuda_core;
pub use cuda_host;
pub use rccl_memory::device::{
    BlockSharedMemory, DeviceRegion, HistogramCounter, set_lookback_initial_delay,
    set_lookback_retry_delay,
};
pub use rccl_memory::{
    copy_device_range, static_block_shared_memory, static_block_shared_memory_from_size,
};
/// Normalize cuda-oxide and RCCL allocator buffers for CUB's required
/// stream-ordered lookback memset. Implemented only for the signed
/// 32-bit protocol word, so byte-zeroing cannot create an invalid value.
pub trait ZeroRadixLookback {
    fn zero_radix_lookback(
        &mut self,
        stream: &cuda_core::CudaStream,
        words: usize,
    ) -> Result<(), crate::DeviceAlgorithmError>;
}

impl<B: rccl_memory::ContiguousDeviceRangeMut<i32>> ZeroRadixLookback for B {
    fn zero_radix_lookback(
        &mut self,
        stream: &cuda_core::CudaStream,
        words: usize,
    ) -> Result<(), crate::DeviceAlgorithmError> {
        self.ensure_usable_on(stream).map_err(crate::DeviceAlgorithmError::Memory)?;
        if words > self.len() {
            return Err(crate::DeviceAlgorithmError::WorkspaceTooSmall {
                role: "radix lookback", required: words, actual: self.len(),
            });
        }
        let bytes = words.checked_mul(size_of::<i32>()).ok_or(crate::DeviceAlgorithmError::SizeOverflow)?;
        if bytes != 0 {
            // SAFETY: the mutable range covers this checked prefix of i32
            // protocol words; zero is the unpublished state for every slot.
            unsafe {
                cuda_core::simt::memory::memset_d8_async(self.cu_deviceptr(), 0, bytes, stream.cu_stream())
                    .map_err(crate::DeviceAlgorithmError::Driver)?;
            }
        }
        Ok(())
    }
}
