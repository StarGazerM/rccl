//! Fixed cuda-oxide ABI boundary for device-wide histograms.
//!
//! Architecture policy and sample-source selection are ordinary Rust. This
//! module contains only the two CUDA parameter layouts that cannot share one
//! entry signature: a device slice and an arithmetic-range descriptor.

use cuda_device::cuda_module;

#[cuda_module]
#[doc(hidden)]
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA entry points preserve the pinned CUB launch ABI as separate uniform parameters"
)]
mod device {
    use cuda_device::{DisjointSlice, Uniform, kernel, launch_bounds, launch_contract};

    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "private helper groups the fixed CUDA entry state"
    )]
    fn even_sweep<
        Samples,
        S,
        C,
        L,
        O,
        const CHANNELS: usize,
        const ACTIVE: usize,
        const THREADS: u32,
        const PIXELS: usize,
        const RLE: bool,
        const BYTE_SAMPLE: bool,
        const SHARED: bool,
        const SHARED_BYTES: usize,
    >(
        samples: Samples,
        output: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        privatized: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        transforms: [<S as crate::__private::HistogramSample<L>>::EvenTransform; ACTIVE],
        output_offsets: [usize; ACTIVE],
        privatized_lengths: [i32; ACTIVE],
        privatized_offsets: [usize; ACTIVE],
        image: (O, O, O, i32),
    ) where
        Samples: crate::__private::HistogramRead<S>,
        S: crate::__private::HistogramSample<L> + crate::__private::HistogramFourSampleLoad,
        C: crate::__private::HistogramCounter,
        L: Copy,
        O: crate::__private::HistogramOffset,
    {
        let transforms =
            crate::__private::EvenHistogramTransforms::<S, L, ACTIVE, BYTE_SAMPLE>::new(transforms);
        let shared_memory = crate::__private::static_block_shared_memory_from_size!(
            SHARED_BYTES;
            12, 16, 20, 24, 32, 40, 1036, 2064, 3092, 4120, 6176, 8232,
        );
        crate::__private::histogram_sweep_agent::<
            _,
            S,
            C,
            O,
            _,
            _,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            SHARED,
        >(
            samples,
            output,
            privatized,
            output_offsets,
            (privatized_lengths, privatized_offsets),
            image,
            transforms,
            transforms,
            shared_memory,
        );
    }

    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "private helper groups the fixed CUDA entry state"
    )]
    fn range_sweep<
        Samples,
        S,
        C,
        L,
        O,
        const CHANNELS: usize,
        const ACTIVE: usize,
        const THREADS: u32,
        const PIXELS: usize,
        const RLE: bool,
        const BYTE_SAMPLE: bool,
        const SHARED: bool,
        const SHARED_BYTES: usize,
    >(
        samples: Samples,
        output: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        privatized: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        levels: &[L],
        level_lengths: [usize; ACTIVE],
        level_offsets: [usize; ACTIVE],
        output_offsets: [usize; ACTIVE],
        privatized_lengths: [i32; ACTIVE],
        privatized_offsets: [usize; ACTIVE],
        image: (O, O, O, i32),
    ) where
        Samples: crate::__private::HistogramRead<S>,
        S: crate::__private::HistogramSample<L> + crate::__private::HistogramFourSampleLoad,
        C: crate::__private::HistogramCounter,
        L: Copy + PartialOrd,
        O: crate::__private::HistogramOffset,
    {
        let transforms =
            crate::__private::RangeHistogramTransforms::<S, L, ACTIVE, BYTE_SAMPLE>::new(
                levels,
                level_lengths,
                level_offsets,
            );
        let shared_memory = crate::__private::static_block_shared_memory_from_size!(
            SHARED_BYTES;
            12, 16, 20, 24, 32, 40, 1036, 2064, 3092, 4120, 6176, 8232,
        );
        crate::__private::histogram_sweep_agent::<
            _,
            S,
            C,
            O,
            _,
            _,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            SHARED,
        >(
            samples,
            output,
            privatized,
            output_offsets,
            (privatized_lengths, privatized_offsets),
            image,
            transforms,
            transforms,
            shared_memory,
        );
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn device_histogram_init<C, const ACTIVE: usize>(
        output: DisjointSlice<C>,
        output_lengths: Uniform<[i32; ACTIVE]>,
        output_offsets: Uniform<[usize; ACTIVE]>,
        queue: DisjointSlice<i32>,
    ) where
        C: crate::__private::HistogramCounter,
    {
        crate::__private::histogram_init_agent::<C, ACTIVE>(
            output,
            output_lengths.get(),
            output_offsets.get(),
            queue,
        );
    }

    #[kernel]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 2, coordinates = u32)]
    pub fn device_histogram_even_sweep<
        S,
        C,
        L,
        O,
        const CHANNELS: usize,
        const ACTIVE: usize,
        const THREADS: u32,
        const PIXELS: usize,
        const RLE: bool,
        const BYTE_SAMPLE: bool,
        const SHARED: bool,
        const SHARED_BYTES: usize,
    >(
        samples: &[S],
        output: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        privatized: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        transforms: Uniform<[<S as crate::__private::HistogramSample<L>>::EvenTransform; ACTIVE]>,
        output_offsets: Uniform<[usize; ACTIVE]>,
        privatized_lengths: Uniform<[i32; ACTIVE]>,
        privatized_offsets: Uniform<[usize; ACTIVE]>,
        num_row_pixels: Uniform<O>,
        num_rows: Uniform<O>,
        row_stride_samples: Uniform<O>,
        blocks_per_row: Uniform<i32>,
    ) where
        S: crate::__private::HistogramSample<L> + crate::__private::HistogramFourSampleLoad,
        C: crate::__private::HistogramCounter,
        L: Copy,
        O: crate::__private::HistogramOffset,
    {
        even_sweep::<
            _,
            S,
            C,
            L,
            O,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED,
            SHARED_BYTES,
        >(
            samples,
            output,
            privatized,
            transforms.get(),
            output_offsets.get(),
            privatized_lengths.get(),
            privatized_offsets.get(),
            (
                num_row_pixels.get(),
                num_rows.get(),
                row_stride_samples.get(),
                blocks_per_row.get(),
            ),
        );
    }

    #[kernel]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 2, coordinates = u32)]
    pub fn device_histogram_range_sweep<
        S,
        C,
        L,
        O,
        const CHANNELS: usize,
        const ACTIVE: usize,
        const THREADS: u32,
        const PIXELS: usize,
        const RLE: bool,
        const BYTE_SAMPLE: bool,
        const SHARED: bool,
        const SHARED_BYTES: usize,
    >(
        samples: &[S],
        output: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        privatized: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        levels: &[L],
        level_lengths: Uniform<[usize; ACTIVE]>,
        level_offsets: Uniform<[usize; ACTIVE]>,
        output_offsets: Uniform<[usize; ACTIVE]>,
        privatized_lengths: Uniform<[i32; ACTIVE]>,
        privatized_offsets: Uniform<[usize; ACTIVE]>,
        num_row_pixels: Uniform<O>,
        num_rows: Uniform<O>,
        row_stride_samples: Uniform<O>,
        blocks_per_row: Uniform<i32>,
    ) where
        S: crate::__private::HistogramSample<L> + crate::__private::HistogramFourSampleLoad,
        C: crate::__private::HistogramCounter,
        L: Copy + PartialOrd,
        O: crate::__private::HistogramOffset,
    {
        range_sweep::<
            _,
            S,
            C,
            L,
            O,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED,
            SHARED_BYTES,
        >(
            samples,
            output,
            privatized,
            levels,
            level_lengths.get(),
            level_offsets.get(),
            output_offsets.get(),
            privatized_lengths.get(),
            privatized_offsets.get(),
            (
                num_row_pixels.get(),
                num_rows.get(),
                row_stride_samples.get(),
                blocks_per_row.get(),
            ),
        );
    }

    #[kernel]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 2, coordinates = u32)]
    pub fn device_histogram_even_arithmetic_sweep<
        S,
        C,
        L,
        O,
        const CHANNELS: usize,
        const ACTIVE: usize,
        const THREADS: u32,
        const PIXELS: usize,
        const RLE: bool,
        const BYTE_SAMPLE: bool,
        const SHARED: bool,
        const SHARED_BYTES: usize,
    >(
        sample_start: Uniform<S>,
        sample_step: Uniform<S>,
        sample_len: Uniform<usize>,
        output: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        privatized: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        transforms: Uniform<[<S as crate::__private::HistogramSample<L>>::EvenTransform; ACTIVE]>,
        output_offsets: Uniform<[usize; ACTIVE]>,
        privatized_lengths: Uniform<[i32; ACTIVE]>,
        privatized_offsets: Uniform<[usize; ACTIVE]>,
        num_row_pixels: Uniform<O>,
        num_rows: Uniform<O>,
        row_stride_samples: Uniform<O>,
        blocks_per_row: Uniform<i32>,
    ) where
        S: crate::__private::HistogramSample<L>
            + crate::__private::HistogramFourSampleLoad
            + TryFrom<usize>
            + core::ops::Add<Output = S>
            + core::ops::Mul<Output = S>,
        C: crate::__private::HistogramCounter,
        L: Copy,
        O: crate::__private::HistogramOffset,
    {
        let samples = crate::__private::ArithmeticHistogramRange::new(
            sample_start.get(),
            sample_step.get(),
            sample_len.get(),
        );
        even_sweep::<
            _,
            S,
            C,
            L,
            O,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED,
            SHARED_BYTES,
        >(
            samples,
            output,
            privatized,
            transforms.get(),
            output_offsets.get(),
            privatized_lengths.get(),
            privatized_offsets.get(),
            (
                num_row_pixels.get(),
                num_rows.get(),
                row_stride_samples.get(),
                blocks_per_row.get(),
            ),
        );
    }

    #[kernel]
    #[launch_bounds(THREADS)]
    #[launch_contract(domain = 2, coordinates = u32)]
    pub fn device_histogram_range_arithmetic_sweep<
        S,
        C,
        L,
        O,
        const CHANNELS: usize,
        const ACTIVE: usize,
        const THREADS: u32,
        const PIXELS: usize,
        const RLE: bool,
        const BYTE_SAMPLE: bool,
        const SHARED: bool,
        const SHARED_BYTES: usize,
    >(
        sample_start: Uniform<S>,
        sample_step: Uniform<S>,
        sample_len: Uniform<usize>,
        output: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        privatized: DisjointSlice<C, cuda_device::thread::Index2D<1>>,
        levels: &[L],
        level_lengths: Uniform<[usize; ACTIVE]>,
        level_offsets: Uniform<[usize; ACTIVE]>,
        output_offsets: Uniform<[usize; ACTIVE]>,
        privatized_lengths: Uniform<[i32; ACTIVE]>,
        privatized_offsets: Uniform<[usize; ACTIVE]>,
        num_row_pixels: Uniform<O>,
        num_rows: Uniform<O>,
        row_stride_samples: Uniform<O>,
        blocks_per_row: Uniform<i32>,
    ) where
        S: crate::__private::HistogramSample<L>
            + crate::__private::HistogramFourSampleLoad
            + TryFrom<usize>
            + core::ops::Add<Output = S>
            + core::ops::Mul<Output = S>,
        C: crate::__private::HistogramCounter,
        L: Copy + PartialOrd,
        O: crate::__private::HistogramOffset,
    {
        let samples = crate::__private::ArithmeticHistogramRange::new(
            sample_start.get(),
            sample_step.get(),
            sample_len.get(),
        );
        range_sweep::<
            _,
            S,
            C,
            L,
            O,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED,
            SHARED_BYTES,
        >(
            samples,
            output,
            privatized,
            levels,
            level_lengths.get(),
            level_offsets.get(),
            output_offsets.get(),
            privatized_lengths.get(),
            privatized_offsets.get(),
            (
                num_row_pixels.get(),
                num_rows.get(),
                row_stride_samples.get(),
                blocks_per_row.get(),
            ),
        );
    }
}

#[doc(hidden)]
pub type HistogramModule = device::LoadedModule;

#[doc(hidden)]
pub fn load(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> Result<HistogramModule, cuda_host::EmbeddedModuleError> {
    // The module and loader are defined together above, so the loaded symbols
    // have exactly the ABI and launch contracts declared by this source.
    unsafe { device::load(context) }
}

// The host binding is ordinary Rust tied to this module's concrete loader.
// No caller-local item generation or method-name forwarding is involved.
impl<C, const ACTIVE: usize> crate::__private::HistogramInitKernel<C, ACTIVE> for HistogramModule
where
    C: crate::__private::HistogramScalar + crate::__private::HistogramCounter,
{
    fn launch_histogram_init(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        storage: crate::__private::HistogramLaunchStorage<'_, C, ACTIVE>,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError> {
        let max_bins = storage.output_lengths.iter().copied().max().unwrap_or(0);
        let grid = max_bins.div_ceil(256).max(1);
        let launch = crate::__private::cuda_core::LaunchConfig1D::new(
            grid.min(u32::MAX as usize) as u32,
            256,
            0,
        );
        let prepared = self
            .prepare_device_histogram_init::<C, ACTIVE>(launch)
            .map_err(crate::DeviceAlgorithmError::Launch)?;
        self.device_histogram_init::<C, ACTIVE>(
            stream,
            &prepared,
            &mut *storage.output,
            storage.output_lengths.map(|length| length as i32),
            storage.output_offsets,
            &mut *storage.queue,
        )
        .map_err(crate::DeviceAlgorithmError::Launch)
    }
}

impl<
    S,
    C,
    L,
    O,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED: bool,
    const SHARED_BYTES: usize,
>
    crate::__private::HistogramKernel<
        S,
        C,
        L,
        O,
        CHANNELS,
        ACTIVE,
        THREADS,
        PIXELS,
        RLE,
        BYTE_SAMPLE,
        SHARED,
        SHARED_BYTES,
    > for HistogramModule
where
    S: crate::__private::HistogramScalar
        + crate::__private::HistogramSample<L>
        + crate::__private::HistogramFourSampleLoad,
    C: crate::__private::HistogramScalar + crate::__private::HistogramCounter,
    L: crate::__private::HistogramScalar + ::core::cmp::PartialOrd,
    O: crate::__private::HistogramOffset,
{
    fn histogram_even_occupancy(&self) -> ::core::result::Result<u32, crate::DeviceAlgorithmError> {
        let launch = crate::__private::cuda_core::LaunchConfig2D::new((1, 1), (THREADS, 1), 0);
        let prepared = self
                                            .prepare_device_histogram_even_sweep::<
                                                S,
                                                C,
                                                L,
                                                O,
                                                CHANNELS,
                                                ACTIVE,
                                                THREADS,
                                                PIXELS,
                                                RLE,
                                                BYTE_SAMPLE,
                                                SHARED,
                                                SHARED_BYTES,
                                            >(launch)
                                            .map_err(crate::DeviceAlgorithmError::Launch)?;
        prepared
            .function()
            .max_active_blocks_per_multiprocessor(THREADS, 0)
            .map_err(crate::DeviceAlgorithmError::Driver)
    }

    fn histogram_range_occupancy(
        &self,
    ) -> ::core::result::Result<u32, crate::DeviceAlgorithmError> {
        let launch = crate::__private::cuda_core::LaunchConfig2D::new((1, 1), (THREADS, 1), 0);
        let prepared = self
                                            .prepare_device_histogram_range_sweep::<
                                                S,
                                                C,
                                                L,
                                                O,
                                                CHANNELS,
                                                ACTIVE,
                                                THREADS,
                                                PIXELS,
                                                RLE,
                                                BYTE_SAMPLE,
                                                SHARED,
                                                SHARED_BYTES,
                                            >(launch)
                                            .map_err(crate::DeviceAlgorithmError::Launch)?;
        prepared
            .function()
            .max_active_blocks_per_multiprocessor(THREADS, 0)
            .map_err(crate::DeviceAlgorithmError::Driver)
    }

    fn histogram_even_arithmetic_occupancy(
        &self,
    ) -> ::core::result::Result<u32, crate::DeviceAlgorithmError>
    where
        S: ::core::convert::TryFrom<usize>
            + ::core::ops::Add<Output = S>
            + ::core::ops::Mul<Output = S>,
    {
        let launch = crate::__private::cuda_core::LaunchConfig2D::new((1, 1), (THREADS, 1), 0);
        let prepared = self
                                            .prepare_device_histogram_even_arithmetic_sweep::<
                                                S,
                                                C,
                                                L,
                                                O,
                                                CHANNELS,
                                                ACTIVE,
                                                THREADS,
                                                PIXELS,
                                                RLE,
                                                BYTE_SAMPLE,
                                                SHARED,
                                                SHARED_BYTES,
                                            >(launch)
                                            .map_err(crate::DeviceAlgorithmError::Launch)?;
        prepared
            .function()
            .max_active_blocks_per_multiprocessor(THREADS, 0)
            .map_err(crate::DeviceAlgorithmError::Driver)
    }

    fn histogram_range_arithmetic_occupancy(
        &self,
    ) -> ::core::result::Result<u32, crate::DeviceAlgorithmError>
    where
        S: ::core::convert::TryFrom<usize>
            + ::core::ops::Add<Output = S>
            + ::core::ops::Mul<Output = S>,
    {
        let launch = crate::__private::cuda_core::LaunchConfig2D::new((1, 1), (THREADS, 1), 0);
        let prepared = self
                                            .prepare_device_histogram_range_arithmetic_sweep::<
                                                S,
                                                C,
                                                L,
                                                O,
                                                CHANNELS,
                                                ACTIVE,
                                                THREADS,
                                                PIXELS,
                                                RLE,
                                                BYTE_SAMPLE,
                                                SHARED,
                                                SHARED_BYTES,
                                            >(launch)
                                            .map_err(crate::DeviceAlgorithmError::Launch)?;
        prepared
            .function()
            .max_active_blocks_per_multiprocessor(THREADS, 0)
            .map_err(crate::DeviceAlgorithmError::Driver)
    }

    fn launch_histogram_even<Samples>(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        samples: &Samples,
        transforms: [<S as crate::__private::HistogramSample<L>>::EvenTransform; ACTIVE],
        storage: crate::__private::HistogramLaunchStorage<'_, C, ACTIVE>,
        geometry: crate::__private::HistogramLaunchGeometry<O>,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        Samples: crate::__private::cuda_host::KernelSliceArg<Elem = S>,
    {
        let launch = crate::__private::cuda_core::LaunchConfig2D::new(
            (
                geometry.blocks_per_row.min(u32::MAX as usize) as u32,
                geometry.blocks_per_col.min(u32::MAX as usize) as u32,
            ),
            (THREADS, 1),
            0,
        );
        let prepared = self
                                            .prepare_device_histogram_even_sweep::<
                                                S,
                                                C,
                                                L,
                                                O,
                                                CHANNELS,
                                                ACTIVE,
                                                THREADS,
                                                PIXELS,
                                                RLE,
                                                BYTE_SAMPLE,
                                                SHARED,
                                                SHARED_BYTES,
                                            >(launch)
                                            .map_err(crate::DeviceAlgorithmError::Launch)?;
        crate::__private::HistogramInitKernel::launch_histogram_init(
            self,
            stream,
            crate::__private::HistogramLaunchStorage {
                output: &mut *storage.output,
                privatized: &mut *storage.privatized,
                queue: &mut *storage.queue,
                output_lengths: storage.output_lengths,
                output_offsets: storage.output_offsets,
                privatized_lengths: storage.privatized_lengths,
                privatized_offsets: storage.privatized_offsets,
            },
        )?;
        self.device_histogram_even_sweep::<
                                            S,
                                            C,
                                            L,
                                            O,
                                            CHANNELS,
                                            ACTIVE,
                                            THREADS,
                                            PIXELS,
                                            RLE,
                                            BYTE_SAMPLE,
                                            SHARED,
                                            SHARED_BYTES,
                                        >(
                                            stream,
                                            &prepared,
                                            samples,
                                            &mut *storage.output,
                                            &mut *storage.privatized,
                                            transforms,
                                            storage.output_offsets,
                                            storage.privatized_lengths.map(|length| length as i32),
                                            storage.privatized_offsets,
                                            geometry.num_row_pixels,
                                            geometry.num_rows,
                                            geometry.row_stride_samples,
                                            geometry.tiles_per_row,
                                        )
                                        .map_err(crate::DeviceAlgorithmError::Launch)
    }

    fn launch_histogram_range<Samples>(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        samples: &Samples,
        levels: crate::__private::HistogramLevelStorage<'_, L, ACTIVE>,
        storage: crate::__private::HistogramLaunchStorage<'_, C, ACTIVE>,
        geometry: crate::__private::HistogramLaunchGeometry<O>,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        Samples: crate::__private::cuda_host::KernelSliceArg<Elem = S>,
        L: ::core::cmp::PartialOrd,
    {
        let launch = crate::__private::cuda_core::LaunchConfig2D::new(
            (
                geometry.blocks_per_row.min(u32::MAX as usize) as u32,
                geometry.blocks_per_col.min(u32::MAX as usize) as u32,
            ),
            (THREADS, 1),
            0,
        );
        let prepared = self
                                            .prepare_device_histogram_range_sweep::<
                                                S,
                                                C,
                                                L,
                                                O,
                                                CHANNELS,
                                                ACTIVE,
                                                THREADS,
                                                PIXELS,
                                                RLE,
                                                BYTE_SAMPLE,
                                                SHARED,
                                                SHARED_BYTES,
                                            >(launch)
                                            .map_err(crate::DeviceAlgorithmError::Launch)?;
        crate::__private::HistogramInitKernel::launch_histogram_init(
            self,
            stream,
            crate::__private::HistogramLaunchStorage {
                output: &mut *storage.output,
                privatized: &mut *storage.privatized,
                queue: &mut *storage.queue,
                output_lengths: storage.output_lengths,
                output_offsets: storage.output_offsets,
                privatized_lengths: storage.privatized_lengths,
                privatized_offsets: storage.privatized_offsets,
            },
        )?;
        self.device_histogram_range_sweep::<
                                            S,
                                            C,
                                            L,
                                            O,
                                            CHANNELS,
                                            ACTIVE,
                                            THREADS,
                                            PIXELS,
                                            RLE,
                                            BYTE_SAMPLE,
                                            SHARED,
                                            SHARED_BYTES,
                                        >(
                                            stream,
                                            &prepared,
                                            samples,
                                            &mut *storage.output,
                                            &mut *storage.privatized,
                                            levels.storage,
                                            levels.lengths,
                                            levels.offsets,
                                            storage.output_offsets,
                                            storage.privatized_lengths.map(|length| length as i32),
                                            storage.privatized_offsets,
                                            geometry.num_row_pixels,
                                            geometry.num_rows,
                                            geometry.row_stride_samples,
                                            geometry.tiles_per_row,
                                        )
                                        .map_err(crate::DeviceAlgorithmError::Launch)
    }

    fn launch_histogram_even_arithmetic(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        samples: &crate::ArithmeticRange<S>,
        transforms: [<S as crate::__private::HistogramSample<L>>::EvenTransform; ACTIVE],
        storage: crate::__private::HistogramLaunchStorage<'_, C, ACTIVE>,
        geometry: crate::__private::HistogramLaunchGeometry<O>,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        S: ::core::convert::TryFrom<usize>
            + ::core::ops::Add<Output = S>
            + ::core::ops::Mul<Output = S>,
    {
        let launch = crate::__private::cuda_core::LaunchConfig2D::new(
            (
                geometry.blocks_per_row.min(u32::MAX as usize) as u32,
                geometry.blocks_per_col.min(u32::MAX as usize) as u32,
            ),
            (THREADS, 1),
            0,
        );
        let prepared = self
                                            .prepare_device_histogram_even_arithmetic_sweep::<
                                                S,
                                                C,
                                                L,
                                                O,
                                                CHANNELS,
                                                ACTIVE,
                                                THREADS,
                                                PIXELS,
                                                RLE,
                                                BYTE_SAMPLE,
                                                SHARED,
                                                SHARED_BYTES,
                                            >(launch)
                                            .map_err(crate::DeviceAlgorithmError::Launch)?;
        crate::__private::HistogramInitKernel::launch_histogram_init(
            self,
            stream,
            crate::__private::HistogramLaunchStorage {
                output: &mut *storage.output,
                privatized: &mut *storage.privatized,
                queue: &mut *storage.queue,
                output_lengths: storage.output_lengths,
                output_offsets: storage.output_offsets,
                privatized_lengths: storage.privatized_lengths,
                privatized_offsets: storage.privatized_offsets,
            },
        )?;
        self.device_histogram_even_arithmetic_sweep::<
                                            S,
                                            C,
                                            L,
                                            O,
                                            CHANNELS,
                                            ACTIVE,
                                            THREADS,
                                            PIXELS,
                                            RLE,
                                            BYTE_SAMPLE,
                                            SHARED,
                                            SHARED_BYTES,
                                        >(
                                            stream,
                                            &prepared,
                                            *samples.start(),
                                            *samples.step(),
                                            samples.len(),
                                            &mut *storage.output,
                                            &mut *storage.privatized,
                                            transforms,
                                            storage.output_offsets,
                                            storage.privatized_lengths.map(|length| length as i32),
                                            storage.privatized_offsets,
                                            geometry.num_row_pixels,
                                            geometry.num_rows,
                                            geometry.row_stride_samples,
                                            geometry.tiles_per_row,
                                        )
                                        .map_err(crate::DeviceAlgorithmError::Launch)
    }

    fn launch_histogram_range_arithmetic(
        &self,
        stream: &crate::__private::cuda_core::CudaStream,
        samples: &crate::ArithmeticRange<S>,
        levels: crate::__private::HistogramLevelStorage<'_, L, ACTIVE>,
        storage: crate::__private::HistogramLaunchStorage<'_, C, ACTIVE>,
        geometry: crate::__private::HistogramLaunchGeometry<O>,
    ) -> ::core::result::Result<(), crate::DeviceAlgorithmError>
    where
        S: ::core::convert::TryFrom<usize>
            + ::core::ops::Add<Output = S>
            + ::core::ops::Mul<Output = S>,
        L: ::core::cmp::PartialOrd,
    {
        let launch = crate::__private::cuda_core::LaunchConfig2D::new(
            (
                geometry.blocks_per_row.min(u32::MAX as usize) as u32,
                geometry.blocks_per_col.min(u32::MAX as usize) as u32,
            ),
            (THREADS, 1),
            0,
        );
        let prepared = self
                                            .prepare_device_histogram_range_arithmetic_sweep::<
                                                S,
                                                C,
                                                L,
                                                O,
                                                CHANNELS,
                                                ACTIVE,
                                                THREADS,
                                                PIXELS,
                                                RLE,
                                                BYTE_SAMPLE,
                                                SHARED,
                                                SHARED_BYTES,
                                            >(launch)
                                            .map_err(crate::DeviceAlgorithmError::Launch)?;
        crate::__private::HistogramInitKernel::launch_histogram_init(
            self,
            stream,
            crate::__private::HistogramLaunchStorage {
                output: &mut *storage.output,
                privatized: &mut *storage.privatized,
                queue: &mut *storage.queue,
                output_lengths: storage.output_lengths,
                output_offsets: storage.output_offsets,
                privatized_lengths: storage.privatized_lengths,
                privatized_offsets: storage.privatized_offsets,
            },
        )?;
        self.device_histogram_range_arithmetic_sweep::<
                                            S,
                                            C,
                                            L,
                                            O,
                                            CHANNELS,
                                            ACTIVE,
                                            THREADS,
                                            PIXELS,
                                            RLE,
                                            BYTE_SAMPLE,
                                            SHARED,
                                            SHARED_BYTES,
                                        >(
                                            stream,
                                            &prepared,
                                            *samples.start(),
                                            *samples.step(),
                                            samples.len(),
                                            &mut *storage.output,
                                            &mut *storage.privatized,
                                            levels.storage,
                                            levels.lengths,
                                            levels.offsets,
                                            storage.output_offsets,
                                            storage.privatized_lengths.map(|length| length as i32),
                                            storage.privatized_offsets,
                                            geometry.num_row_pixels,
                                            geometry.num_rows,
                                            geometry.row_stride_samples,
                                            geometry.tiles_per_row,
                                        )
                                        .map_err(crate::DeviceAlgorithmError::Launch)
    }
}
