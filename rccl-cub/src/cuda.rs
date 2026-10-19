//! CUDA module ownership and stream binding for direct CUB operations.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy};
use rccl_memory::{CudaCompletion, MemoryError};

use crate::{
    compaction_kernels, histogram_kernels, ordering_kernels, range_kernels, reduce_kernels,
    scan_kernels, set_kernels,
};

/// The fixed RCCL kernel set compiled for cargo-oxide's selected target.
///
/// This is a runtime resource owner, not an algorithm or architecture policy.
/// Its concrete type keeps module implementation details out of operation
/// signatures.
pub(crate) struct KernelModules {
    compaction: compaction_kernels::CompactionModule,
    histogram: histogram_kernels::HistogramModule,
    ordering: ordering_kernels::OrderingModule,
    range: range_kernels::RangeModule,
    reduce: reduce_kernels::ReduceModule,
    scan: scan_kernels::ScanModule,
    set: set_kernels::SetModule,
}

impl KernelModules {
    fn load(context: &Arc<CudaContext>) -> Result<Self, cuda_host::EmbeddedModuleError> {
        Ok(Self {
            compaction: compaction_kernels::load(context)?,
            histogram: histogram_kernels::load(context)?,
            ordering: ordering_kernels::load(context)?,
            range: range_kernels::load(context)?,
            reduce: reduce_kernels::load(context)?,
            scan: scan_kernels::load(context)?,
            set: set_kernels::load(context)?,
        })
    }

    pub(crate) fn compaction(&self) -> &compaction_kernels::CompactionModule {
        &self.compaction
    }

    pub(crate) fn histogram(&self) -> &histogram_kernels::HistogramModule {
        &self.histogram
    }

    pub(crate) fn ordering(&self) -> &ordering_kernels::OrderingModule {
        &self.ordering
    }
    pub(crate) fn range(&self) -> &range_kernels::RangeModule {
        &self.range
    }
    pub(crate) fn reduce(&self) -> &reduce_kernels::ReduceModule {
        &self.reduce
    }

    pub(crate) fn scan(&self) -> &scan_kernels::ScanModule {
        &self.scan
    }

    pub(crate) fn set(&self) -> &set_kernels::SetModule {
        &self.set
    }
}

#[derive(Clone, Copy)]
pub struct Launch<'stream, 'program> {
    stream: &'stream CudaStream,
    modules: &'program KernelModules,
}

impl<'stream, 'program> Launch<'stream, 'program> {
    pub fn stream(&self) -> &'stream CudaStream {
        self.stream
    }

    /// Submit CUDA work while retaining this launch (and therefore its stream
    /// and kernel program) together with every operation resource until
    /// completion.
    ///
    /// # Safety
    ///
    /// Every resource accessed by the submitted GPU work must be present in
    /// `resources`, and the work must be enqueued only on this launch's stream.
    pub unsafe fn submit<'borrow, Resources, T, E, F>(
        &'borrow self,
        resources: Resources,
        submit: F,
    ) -> CudaCompletion<'borrow, T, E, (&'borrow Self, Resources)>
    where
        F: FnOnce(&Self, &mut Resources) -> Result<T, E>,
    {
        // SAFETY: the caller supplies the resource-retention contract.
        unsafe {
            CudaCompletion::from_submission(self.stream, (self, resources), |retained| {
                submit(retained.0, &mut retained.1)
            })
        }
    }

    pub(crate) fn module(&self) -> &KernelModules {
        self.modules
    }

    pub(crate) fn compaction_module(&self) -> &compaction_kernels::CompactionModule {
        self.modules.compaction()
    }

    pub(crate) fn histogram_module(&self) -> &histogram_kernels::HistogramModule {
        self.modules.histogram()
    }

    pub(crate) fn ordering_module(&self) -> &ordering_kernels::OrderingModule {
        self.modules.ordering()
    }

    fn range_module(&self) -> &range_kernels::RangeModule {
        self.modules.range()
    }
    pub(crate) fn reduce_module(&self) -> &reduce_kernels::ReduceModule {
        self.modules.reduce()
    }
    pub(crate) fn scan_module(&self) -> &scan_kernels::ScanModule {
        self.modules.scan()
    }

    /// Apply a binary device function between every output item and one
    /// repeated value.
    ///
    /// The operation is an ordinary generic argument. It does not require a
    /// macro because cuda-oxide can monomorphize this method at the call site.
    pub(crate) fn transform_repeat_in_place<T, F>(
        &self,
        output: &mut DeviceBuffer<T>,
        repeated: T,
        op: F,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        T: DeviceCopy,
        F: Fn(T, T) -> T + Copy,
    {
        if output.is_empty() {
            return Ok(());
        }
        let launch =
            cuda_core::LaunchConfig1D::new(output.len().div_ceil(8).div_ceil(256) as u32, 256, 0);
        let prepared = self
            .ordering_module()
            .prepare_transform_repeat_in_place_for(&repeated, &op, launch)?;
        self.ordering_module().transform_repeat_in_place(
            self.stream,
            &prepared,
            repeated,
            op,
            output,
        )
    }

    pub(crate) fn reduce_sum_blocks<T, Input, Partials>(
        &self,
        input: &Input,
        len: usize,
        partials: &mut Partials,
        blocks: u32,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        T: reduce_kernels::ReduceSumValue + crate::warp::WarpValue,
        Input: cuda_host::KernelSliceArg<Elem = T>,
        Partials: cuda_host::KernelSliceArgMut<Elem = T>,
    {
        if blocks == 0 {
            return Ok(());
        }
        let shared_bytes = (8 * core::mem::size_of::<T>()) as u32;
        let launch =
            cuda_core::LaunchConfig1D::new(blocks, reduce_kernels::REDUCE_THREADS, shared_bytes);
        let witness = T::default();
        let prepared = self
            .reduce_module()
            .prepare_device_reduce_sum_blocks_for::<T, { reduce_kernels::REDUCE_NOMINAL_4B_ITEMS }>(
                &witness, launch,
            )?;
        self.reduce_module()
            .device_reduce_sum_blocks::<T, { reduce_kernels::REDUCE_NOMINAL_4B_ITEMS }>(
                self.stream,
                &prepared,
                input,
                len,
                partials,
            )
    }

    pub(crate) fn reduce_sum_single_tile<T, Input, Output>(
        &self,
        input: &Input,
        len: usize,
        output: &mut Output,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        T: reduce_kernels::ReduceSumValue + crate::warp::WarpValue,
        Input: cuda_host::KernelSliceArg<Elem = T>,
        Output: cuda_host::KernelSliceArgMut<Elem = T>,
    {
        let shared_bytes = (8 * core::mem::size_of::<T>()) as u32;
        let launch =
            cuda_core::LaunchConfig1D::new(1, reduce_kernels::REDUCE_THREADS, shared_bytes);
        let witness = T::default();
        let prepared = self
            .reduce_module()
            .prepare_device_reduce_sum_single_tile_for::<
                T,
                { reduce_kernels::REDUCE_NOMINAL_4B_ITEMS },
            >(&witness, launch)?;
        self.reduce_module()
            .device_reduce_sum_single_tile::<T, { reduce_kernels::REDUCE_NOMINAL_4B_ITEMS }>(
                self.stream,
                &prepared,
                input,
                len,
                output,
            )
    }

    pub(crate) fn reduce_sum_blocks_i32_as_u32<Input, Partials>(
        &self,
        input: &Input,
        len: usize,
        partials: &mut Partials,
        blocks: u32,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Input: cuda_host::KernelSliceArg<Elem = i32>,
        Partials: cuda_host::KernelSliceArgMut<Elem = i32>,
    {
        if blocks == 0 {
            return Ok(());
        }
        let shared_bytes = (8 * core::mem::size_of::<u32>()) as u32;
        let launch =
            cuda_core::LaunchConfig1D::new(blocks, reduce_kernels::REDUCE_THREADS, shared_bytes);
        let prepared = self
            .reduce_module()
            .prepare_device_reduce_sum_blocks_i32_as_u32::<
                { reduce_kernels::REDUCE_NOMINAL_4B_ITEMS },
            >(launch)?;
        self.reduce_module()
            .device_reduce_sum_blocks_i32_as_u32::<{ reduce_kernels::REDUCE_NOMINAL_4B_ITEMS }>(
                self.stream,
                &prepared,
                input,
                len,
                partials,
            )
    }

    pub(crate) fn reduce_sum_single_tile_i32_as_u32<Input, Output>(
        &self,
        input: &Input,
        len: usize,
        output: &mut Output,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Input: cuda_host::KernelSliceArg<Elem = i32>,
        Output: cuda_host::KernelSliceArgMut<Elem = i32>,
    {
        let shared_bytes = (8 * core::mem::size_of::<u32>()) as u32;
        let launch =
            cuda_core::LaunchConfig1D::new(1, reduce_kernels::REDUCE_THREADS, shared_bytes);
        let prepared = self
            .reduce_module()
            .prepare_device_reduce_sum_single_tile_i32_as_u32::<
                { reduce_kernels::REDUCE_NOMINAL_4B_ITEMS },
            >(launch)?;
        self.reduce_module()
            .device_reduce_sum_single_tile_i32_as_u32::<
                { reduce_kernels::REDUCE_NOMINAL_4B_ITEMS },
            >(self.stream, &prepared, input, len, output)
    }
    /// Launch the dense four-byte fill policy used by pinned DeviceTransform
    /// on SM89: 256 threads and eight items per thread.
    #[doc(hidden)]
    pub fn range_fill_4b<Output>(
        &self,
        output: Output,
        value: Output::Item,
        len: u32,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Output: crate::range::DenseDeviceWriteAt,
    {
        if len == 0 {
            return Ok(());
        }
        let blocks = range_kernels::generation_blocks(len);
        let launch = cuda_core::LaunchConfig1D::new(blocks, range_kernels::GENERATION_THREADS, 0);
        let prepared = self
            .range_module()
            .prepare_range_fill_4b_for(&output, launch)?;
        self.range_module()
            .range_fill_4b(self.stream, &prepared, output, value, len)
    }

    /// Launch the dense four-byte index-generation policy used by pinned
    /// DeviceTransform on SM89: 256 threads and eight items per thread.
    #[doc(hidden)]
    pub fn range_generate_4b<Input, Output, F>(
        &self,
        input: Input,
        output: Output,
        op: F,
        len: u32,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Input: crate::DeviceReadAt,
        Output: crate::range::DenseDeviceWriteAt,
        F: Fn(Input::Item) -> Output::Item + Copy,
    {
        if len == 0 {
            return Ok(());
        }
        let blocks = range_kernels::generation_blocks(len);
        let launch = cuda_core::LaunchConfig1D::new(blocks, range_kernels::GENERATION_THREADS, 0);
        let prepared = self
            .range_module()
            .prepare_range_generate_4b_for(&input, &output, &op, launch)?;
        self.range_module()
            .range_generate_4b(self.stream, &prepared, input, output, op, len)
    }

    /// Launch allocation-free vectorized lower/upper bound search.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn range_bound<Haystack, Needles, Output, F>(
        &self,
        haystack: Haystack,
        needles: Needles,
        output: Output,
        compare: F,
        haystack_len: usize,
        needle_len: usize,
        upper: bool,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Haystack: crate::DeviceReadAt,
        Needles: crate::DeviceReadAt<Item = Haystack::Item>,
        Output: crate::DeviceWriteAt<Item = u64>,
        F: Fn(&Haystack::Item, &Haystack::Item) -> core::cmp::Ordering + Copy,
    {
        if needle_len == 0 {
            return Ok(());
        }
        let blocks = needle_len.div_ceil(512).min(65_535) as u32;
        let launch = cuda_core::LaunchConfig1D::new(blocks, 256, 0);
        let prepared = self
            .range_module()
            .prepare_range_bound_for(&haystack, &needles, &output, &compare, launch)?;
        self.range_module().range_bound(
            self.stream,
            &prepared,
            haystack,
            needles,
            output,
            compare,
            haystack_len,
            needle_len,
            upper,
        )
    }

    /// Launch both insertion bounds into separate affine outputs.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn range_equal_range<Haystack, Needles, Begins, Ends, F>(
        &self,
        haystack: Haystack,
        needles: Needles,
        begins: Begins,
        ends: Ends,
        compare: F,
        haystack_len: usize,
        needle_len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Haystack: crate::DeviceReadAt,
        Needles: crate::DeviceReadAt<Item = Haystack::Item>,
        Begins: crate::DeviceWriteAt<Item = u64>,
        Ends: crate::DeviceWriteAt<Item = u64>,
        F: Fn(&Haystack::Item, &Haystack::Item) -> core::cmp::Ordering + Copy,
    {
        if needle_len == 0 {
            return Ok(());
        }
        let blocks = needle_len.div_ceil(512).min(65_535) as u32;
        let launch = cuda_core::LaunchConfig1D::new(blocks, 256, 0);
        let prepared = self
            .range_module()
            .prepare_range_equal_range_for(&haystack, &needles, &begins, &ends, &compare, launch)?;
        self.range_module().range_equal_range(
            self.stream,
            &prepared,
            haystack,
            needles,
            begins,
            ends,
            compare,
            haystack_len,
            needle_len,
        )
    }

    /// Launch the generic zero-materialization range transform kernel.
    #[doc(hidden)]
    pub fn range_transform<Input, Output, F>(
        &self,
        input: Input,
        output: Output,
        op: F,
        len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Input: crate::DeviceReadAt,
        Output: crate::DeviceWriteAt,
        F: Fn(Input::Item) -> Output::Item + Copy,
    {
        if len == 0 {
            return Ok(());
        }
        // Pinned CCCL's SM89 transform policy launches two logical items per
        // 256-thread block lane and uses a 32-bit item count for common extents.
        // Keep the usize kernel as the safe fallback instead of narrowing a
        // larger range.
        let blocks = len.div_ceil(2).div_ceil(256).min(u32::MAX as usize) as u32;
        let launch = cuda_core::LaunchConfig1D::new(blocks, 256, 0);
        if let Ok(len_u32) = u32::try_from(len) {
            let prepared = self
                .range_module()
                .prepare_range_transform_u32_for(&input, &output, &op, launch)?;
            self.range_module().range_transform_u32(
                self.stream,
                &prepared,
                input,
                output,
                op,
                len_u32,
            )
        } else {
            let prepared = self
                .range_module()
                .prepare_range_transform_for(&input, &output, &op, launch)?;
            self.range_module()
                .range_transform(self.stream, &prepared, input, output, op, len)
        }
    }

    /// Launch the generic zero-materialization range for-each kernel.
    #[doc(hidden)]
    pub fn range_for_each<Input, F>(
        &self,
        input: Input,
        op: F,
        len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Input: crate::DeviceReadAt,
        F: Fn(Input::Item) + Copy,
    {
        if len == 0 {
            return Ok(());
        }
        let required_blocks = len.div_ceil(2).div_ceil(256);
        let blocks = required_blocks.min(u32::MAX as usize) as u32;
        let launch = cuda_core::LaunchConfig1D::new(blocks, 256, 0);
        if required_blocks <= u32::MAX as usize {
            let prepared = self
                .range_module()
                .prepare_range_for_each_for(&input, &op, launch)?;
            self.range_module()
                .range_for_each(self.stream, &prepared, input, op, len)
        } else {
            let prepared = self
                .range_module()
                .prepare_range_for_each_grid_for(&input, &op, launch)?;
            self.range_module()
                .range_for_each_grid(self.stream, &prepared, input, op, len)
        }
    }

    pub(crate) fn scan_lookback_u32<Input, Output, State>(
        &self,
        input: &Input,
        output: &mut Output,
        tile_state: &mut State,
        len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Input: cuda_host::KernelSliceArg<Elem = u32>,
        Output: cuda_host::KernelSliceArgMut<Elem = u32>,
        State: cuda_host::KernelSliceArgMut<Elem = u64>,
    {
        if len == 0 {
            return Ok(());
        }
        let tile_count = len.div_ceil(scan_kernels::LOOKBACK_SCAN_TILE_ITEMS);
        let init_items = tile_count.max(scan_kernels::LOOKBACK_SCAN_PADDING);
        let init_blocks = init_items
            .div_ceil(scan_kernels::LOOKBACK_SCAN_THREADS as usize)
            .min(u32::MAX as usize) as u32;
        let init_launch =
            cuda_core::LaunchConfig1D::new(init_blocks, scan_kernels::LOOKBACK_SCAN_THREADS, 0);
        let prepared = self
            .scan_module()
            .prepare_device_scan_lookback_init_u32(init_launch)?;
        self.scan_module().device_scan_lookback_init_u32(
            self.stream,
            &prepared,
            tile_state,
            tile_count,
        )?;

        let blocks = tile_count.min(u32::MAX as usize) as u32;
        let scan_launch = cuda_core::LaunchConfig1D::new(
            blocks,
            scan_kernels::LOOKBACK_SCAN_THREADS,
            scan_kernels::LOOKBACK_SCAN_SHARED_BYTES as u32,
        );
        let prepared = self
            .scan_module()
            .prepare_device_scan_lookback_u32(scan_launch)?;
        self.scan_module().device_scan_lookback_u32(
            self.stream,
            &prepared,
            input,
            output,
            tile_state,
            len,
        )
    }
    pub(crate) fn scan_tiles<T, Input, Output, Sums, const EXCLUSIVE: bool>(
        &self,
        input: &Input,
        output: &mut Output,
        block_sums: &mut Sums,
        len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        T: crate::block_scan::BlockScanValue,
        Input: cuda_host::KernelSliceArg<Elem = T>,
        Output: cuda_host::KernelSliceArgMut<Elem = T>,
        Sums: cuda_host::KernelSliceArgMut<Elem = T>,
    {
        if len == 0 {
            return Ok(());
        }
        let block_count = len.div_ceil(scan_kernels::SCAN_THREADS as usize);
        let blocks = block_count.min(u32::MAX as usize) as u32;
        let shared_bytes = (8 * core::mem::size_of::<T>()) as u32;
        let launch =
            cuda_core::LaunchConfig1D::new(blocks, scan_kernels::SCAN_THREADS, shared_bytes);
        let witness = T::zero();
        let prepared = self
            .scan_module()
            .prepare_device_scan_tiles_for::<T, EXCLUSIVE>(&witness, launch)?;
        self.scan_module().device_scan_tiles::<T, EXCLUSIVE>(
            self.stream,
            &prepared,
            input,
            output,
            block_sums,
            len,
        )
    }

    pub(crate) fn scan_add_block_offsets<T, Values, Offsets>(
        &self,
        values: &mut Values,
        block_offsets: &Offsets,
        len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        T: crate::block_scan::BlockScanValue,
        Values: cuda_host::KernelSliceArgMut<Elem = T>,
        Offsets: cuda_host::KernelSliceArg<Elem = T>,
    {
        if len == 0 {
            return Ok(());
        }
        let blocks = len
            .div_ceil(scan_kernels::SCAN_THREADS as usize)
            .min(u32::MAX as usize) as u32;
        let launch = cuda_core::LaunchConfig1D::new(blocks, scan_kernels::SCAN_THREADS, 0);
        let witness = T::zero();
        let prepared = self
            .scan_module()
            .prepare_device_scan_add_block_offsets_for::<T>(&witness, launch)?;
        self.scan_module().device_scan_add_block_offsets::<T>(
            self.stream,
            &prepared,
            values,
            block_offsets,
            len,
        )
    }

    pub(crate) fn compaction_mark_predicate<Input, I, Predicate, Flags>(
        &self,
        input: Input,
        flags: &mut Flags,
        predicate: Predicate,
        len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Input: crate::DeviceReadAt,
        I: crate::compaction::CompactionIndex,
        Predicate: Fn(Input::Item) -> bool + Copy,
        Flags: cuda_host::KernelSliceArgMut<Elem = I>,
    {
        if len == 0 {
            return Ok(());
        }
        let blocks = len.div_ceil(256).min(u32::MAX as usize) as u32;
        let config = cuda_core::LaunchConfig1D::new(blocks, 256, 0);
        let witness = I::zero();
        let prepared = self
            .compaction_module()
            .prepare_mark_predicate_for::<_, I, _>(&input, &witness, &predicate, config)?;
        self.compaction_module().mark_predicate::<_, I, _>(
            self.stream,
            &prepared,
            input,
            flags,
            predicate,
            len,
        )
    }

    pub(crate) fn compaction_mark_flagged<Selected, I, Flags>(
        &self,
        selected: Selected,
        flags: &mut Flags,
        len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Selected: crate::DeviceReadAt<Item = u32>,
        I: crate::compaction::CompactionIndex,
        Flags: cuda_host::KernelSliceArgMut<Elem = I>,
    {
        if len == 0 {
            return Ok(());
        }
        let blocks = len.div_ceil(256).min(u32::MAX as usize) as u32;
        let config = cuda_core::LaunchConfig1D::new(blocks, 256, 0);
        let witness = I::zero();
        let prepared = self
            .compaction_module()
            .prepare_mark_flagged_for::<_, I>(&selected, &witness, config)?;
        self.compaction_module()
            .mark_flagged::<_, I>(self.stream, &prepared, selected, flags, len)
    }

    pub(crate) fn compaction_mark_adjacent_unique<Input, I, Equivalent, Flags>(
        &self,
        input: Input,
        flags: &mut Flags,
        equivalent: Equivalent,
        len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Input: crate::DeviceReadAt,
        I: crate::compaction::CompactionIndex,
        Equivalent: Fn(Input::Item, Input::Item) -> bool + Copy,
        Flags: cuda_host::KernelSliceArgMut<Elem = I>,
    {
        if len == 0 {
            return Ok(());
        }
        let blocks = len.div_ceil(256).min(u32::MAX as usize) as u32;
        let config = cuda_core::LaunchConfig1D::new(blocks, 256, 0);
        let witness = I::zero();
        let prepared = self
            .compaction_module()
            .prepare_mark_adjacent_unique_for::<_, I, _>(&input, &witness, &equivalent, config)?;
        self.compaction_module().mark_adjacent_unique::<_, I, _>(
            self.stream,
            &prepared,
            input,
            flags,
            equivalent,
            len,
        )
    }

    pub(crate) fn compaction_scatter_selected<Input, Output, I, Flags, Positions>(
        &self,
        input: Input,
        output: Output,
        flags: &Flags,
        positions: &Positions,
        selected_count: &mut DeviceBuffer<u64>,
        len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Input: crate::DeviceReadAt,
        Output: crate::DeviceWriteAt<Item = Input::Item>,
        I: crate::compaction::CompactionIndex,
        Flags: cuda_host::KernelSliceArg<Elem = I>,
        Positions: cuda_host::KernelSliceArg<Elem = I>,
    {
        let blocks = len.div_ceil(256).max(1).min(u32::MAX as usize) as u32;
        let config = cuda_core::LaunchConfig1D::new(blocks, 256, 0);
        let witness = I::zero();
        let prepared = self
            .compaction_module()
            .prepare_scatter_selected_for::<_, _, I>(&input, &output, &witness, config)?;
        self.compaction_module().scatter_selected::<_, _, I>(
            self.stream,
            &prepared,
            input,
            output,
            flags,
            positions,
            selected_count,
            len,
        )
    }

    pub(crate) fn compaction_scatter_partition<Input, Output, I, Flags, Positions>(
        &self,
        input: Input,
        output: Output,
        flags: &Flags,
        positions: &Positions,
        selected_count: &mut DeviceBuffer<u64>,
        len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Input: crate::DeviceReadAt,
        Output: crate::DeviceWriteAt<Item = Input::Item>,
        I: crate::compaction::CompactionIndex,
        Flags: cuda_host::KernelSliceArg<Elem = I>,
        Positions: cuda_host::KernelSliceArg<Elem = I>,
    {
        let blocks = len.div_ceil(256).max(1).min(u32::MAX as usize) as u32;
        let config = cuda_core::LaunchConfig1D::new(blocks, 256, 0);
        let witness = I::zero();
        let prepared = self
            .compaction_module()
            .prepare_scatter_partition_for::<_, _, I>(&input, &output, &witness, config)?;
        self.compaction_module().scatter_partition::<_, _, I>(
            self.stream,
            &prepared,
            input,
            output,
            flags,
            positions,
            selected_count,
            len,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compaction_reduce_adjacent_groups<
        Keys,
        Values,
        OutputKeys,
        OutputValues,
        I,
        Operator,
        Heads,
        Positions,
    >(
        &self,
        keys: Keys,
        values: Values,
        output_keys: OutputKeys,
        output_values: OutputValues,
        heads: &Heads,
        positions: &Positions,
        group_count: &mut DeviceBuffer<u64>,
        operator: Operator,
        len: usize,
    ) -> Result<(), cuda_core::LaunchContractError>
    where
        Keys: crate::DeviceReadAt,
        Values: crate::DeviceReadAt,
        OutputKeys: crate::DeviceWriteAt<Item = Keys::Item>,
        OutputValues: crate::DeviceWriteAt<Item = Values::Item>,
        I: crate::compaction::CompactionIndex,
        Operator: crate::grouped_reduction::GroupReduceOperator<Values::Item>,
        Heads: cuda_host::KernelSliceArg<Elem = I>,
        Positions: cuda_host::KernelSliceArg<Elem = I>,
    {
        let blocks = len.div_ceil(256).max(1).min(u32::MAX as usize) as u32;
        let config = cuda_core::LaunchConfig1D::new(blocks, 256, 0);
        let witness = I::zero();
        let prepared = self
            .compaction_module()
            .prepare_reduce_adjacent_groups_for::<_, _, _, _, I, _>(
                &keys,
                &values,
                &output_keys,
                &output_values,
                &witness,
                &operator,
                config,
            )?;
        self.compaction_module()
            .reduce_adjacent_groups::<_, _, _, _, I, _>(
                self.stream,
                &prepared,
                keys,
                values,
                output_keys,
                output_values,
                heads,
                positions,
                group_count,
                operator,
                len,
            )
    }
}

/// Runtime owner for RCCL's fixed CUB kernel modules.
pub struct Program<'context> {
    context: &'context CudaContext,
    modules: KernelModules,
}

impl<'context> Program<'context> {
    /// Load the CUB kernels compiled for the cargo-oxide target.
    pub fn load(
        context: &'context Arc<CudaContext>,
    ) -> Result<Self, cuda_host::EmbeddedModuleError> {
        Ok(Self {
            context: context.as_ref(),
            modules: KernelModules::load(context)?,
        })
    }

    /// Bind this program to the stream used by a direct CUB call.
    pub fn on<'program, 'stream>(
        &'program self,
        stream: &'stream CudaStream,
    ) -> Result<Launch<'stream, 'program>, MemoryError> {
        if self.context != stream.context().as_ref() {
            return Err(MemoryError::ContextMismatch);
        }
        Ok(Launch {
            stream,
            modules: &self.modules,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::range_kernels::generation_blocks;

    #[test]
    fn generation_policy_uses_eight_items_per_256_thread_block() {
        assert_eq!(generation_blocks(0), 0);
        assert_eq!(generation_blocks(2_048), 1);
        assert_eq!(generation_blocks(2_049), 2);
        assert_eq!(generation_blocks(16_777_216), 8_192);
    }
}
