use core::ops::Range;

use cuda_core::{CudaStream, DeviceBuffer as CudaDeviceBuffer, DeviceCopy};
use cuda_host::{KernelSliceArg, KernelSliceArgMut};
use rccl_memory::DeviceAllocator;
type DeviceBuffer<T> = CudaDeviceBuffer<T>;

use crate::{
    ArithmeticRange, DeviceAlgorithmError, DeviceRange,
    histogram_backend::{HistogramFourSampleLoad, HistogramOffset, HistogramSample},
};

mod sealed {
    pub trait HistogramScalar {}
}

/// Primitive scalar layouts supported by the pinned CUB histogram graph.
///
/// This is a representation capability, not architecture policy.  The trait
/// is closed so aliases of supported primitive types work while unsupported
/// layouts cannot accidentally select a kernel by size alone.
#[doc(hidden)]
pub trait HistogramScalar: sealed::HistogramScalar + DeviceCopy {
    const SIZE: usize;
}

macro_rules! histogram_scalars {
    ($($scalar:ty),+ $(,)?) => {$ (
        impl sealed::HistogramScalar for $scalar {}

        impl HistogramScalar for $scalar {
            const SIZE: usize = core::mem::size_of::<Self>();
        }
    )+ };
}

histogram_scalars!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64);

/// Closed architecture policy selected by cfgs injected by `cargo oxide`.
/// It is intentionally absent from every public operation signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub struct HistogramPolicy {
    pub threads: u32,
    pub pixels: usize,
    pub rle: bool,
}

#[doc(hidden)]
pub const fn histogram_policy(
    sample_size: usize,
    counter_size: usize,
    active_channels: usize,
    channels: usize,
    even: bool,
) -> HistogramPolicy {
    // These inputs are used only by architecture-specific cfg branches. Keep
    // the architecture-neutral host build warning-free without changing the
    // selected device policy.
    let _ = (counter_size, channels, even);
    #[cfg(cuda_arch_ge_100)]
    if channels == 1 && active_channels == 1 && counter_size == 4 && sample_size == 1 {
        return if even {
            HistogramPolicy {
                threads: 928,
                pixels: 12,
                rle: false,
            }
        } else {
            HistogramPolicy {
                threads: 448,
                pixels: 12,
                rle: false,
            }
        };
    }

    #[cfg(cuda_arch_ge_90)]
    if channels == 1 && active_channels == 1 && counter_size == 4 {
        if sample_size == 1 {
            return HistogramPolicy {
                threads: 768,
                pixels: 12,
                rle: false,
            };
        }
        if sample_size == 2 {
            return HistogramPolicy {
                threads: 960,
                pixels: 10,
                rle: true,
            };
        }
    }

    let sample_words = sample_size.div_ceil(core::mem::size_of::<i32>());
    let pixels = 16 / active_channels / sample_words;
    HistogramPolicy {
        threads: 384,
        pixels: if pixels == 0 { 1 } else { pixels },
        rle: true,
    }
}

#[doc(hidden)]
pub const fn histogram_shared_bytes(
    counter_size: usize,
    active_channels: usize,
    shared: bool,
) -> usize {
    let counters = if shared { 257 } else { 1 };
    // CUB's TempStorage follows the privatized counters with `int tile_idx`
    // and the one-byte empty `BlockLoadDirect::TempStorage` union member.
    let raw = active_channels * counters * counter_size + core::mem::size_of::<i32>() + 1;
    raw.div_ceil(counter_size) * counter_size
}

/// Mutable launch storage shared by the init and sweep kernels.
#[doc(hidden)]
pub struct HistogramLaunchStorage<
    'a,
    C: DeviceCopy,
    const ACTIVE: usize,
    Output = CudaDeviceBuffer<C>,
    Privatized = CudaDeviceBuffer<C>,
    Queue = CudaDeviceBuffer<i32>,
> {
    pub output: &'a mut Output,
    pub privatized: &'a mut Privatized,
    pub queue: &'a mut Queue,
    pub output_lengths: [usize; ACTIVE],
    pub output_offsets: [usize; ACTIVE],
    pub privatized_lengths: [usize; ACTIVE],
    pub privatized_offsets: [usize; ACTIVE],
    pub __marker: core::marker::PhantomData<C>,
}

/// One immutable device allocation plus validated per-channel level ranges.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct HistogramLevelStorage<'a, L: DeviceCopy, const ACTIVE: usize> {
    pub storage: &'a DeviceBuffer<L>,
    pub lengths: [usize; ACTIVE],
    pub offsets: [usize; ACTIVE],
}

/// Resolved CUB sweep geometry after host validation.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct HistogramLaunchGeometry<O> {
    pub num_row_pixels: O,
    pub num_rows: O,
    pub row_stride_samples: O,
    pub tiles_per_row: i32,
    pub blocks_per_row: usize,
    pub blocks_per_col: usize,
}

/// Generated-module boundary for histogram entry points.
///
/// cuda-oxide creates the concrete `LoadedModule` in the caller's expansion,
/// so one declarative implementation bridge is unavoidable.  Policy,
/// validation, allocation, geometry, and shared/global routing live in the
/// ordinary functions below this trait.
#[doc(hidden)]
pub trait HistogramInitKernel<C, const ACTIVE: usize>
where
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
{
    fn launch_histogram_init<Output, Queue>(
        &self,
        stream: &CudaStream,
        storage: HistogramLaunchStorage<'_, C, ACTIVE, Output, Output, Queue>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Output: KernelSliceArgMut<Elem = C>,
        Queue: KernelSliceArgMut<Elem = i32>;
}

#[doc(hidden)]
pub trait HistogramKernel<
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
>: HistogramInitKernel<C, ACTIVE> where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    O: HistogramOffset,
{
    fn histogram_even_occupancy(&self) -> Result<u32, DeviceAlgorithmError>;

    fn histogram_range_occupancy(&self) -> Result<u32, DeviceAlgorithmError>;

    fn histogram_even_arithmetic_occupancy(&self) -> Result<u32, DeviceAlgorithmError>
    where
        S: TryFrom<usize> + core::ops::Add<Output = S> + core::ops::Mul<Output = S>;

    fn histogram_range_arithmetic_occupancy(&self) -> Result<u32, DeviceAlgorithmError>
    where
        S: TryFrom<usize> + core::ops::Add<Output = S> + core::ops::Mul<Output = S>;

    #[allow(clippy::too_many_arguments)]
    fn launch_histogram_even<Samples, Output, Queue>(
        &self,
        stream: &CudaStream,
        samples: &Samples,
        transforms: [<S as HistogramSample<L>>::EvenTransform; ACTIVE],
        storage: HistogramLaunchStorage<'_, C, ACTIVE, Output, Output, Queue>,
        geometry: HistogramLaunchGeometry<O>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Samples: KernelSliceArg<Elem = S>,
        Output: KernelSliceArgMut<Elem = C>,
        Queue: KernelSliceArgMut<Elem = i32>;

    #[allow(clippy::too_many_arguments)]
    fn launch_histogram_range<Samples, Output, Queue>(
        &self,
        stream: &CudaStream,
        samples: &Samples,
        levels: HistogramLevelStorage<'_, L, ACTIVE>,
        storage: HistogramLaunchStorage<'_, C, ACTIVE, Output, Output, Queue>,
        geometry: HistogramLaunchGeometry<O>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Samples: KernelSliceArg<Elem = S>,
        Output: KernelSliceArgMut<Elem = C>,
        Queue: KernelSliceArgMut<Elem = i32>,
        L: PartialOrd;

    #[allow(clippy::too_many_arguments)]
    fn launch_histogram_even_arithmetic<Output, Queue>(
        &self,
        stream: &CudaStream,
        samples: &ArithmeticRange<S>,
        transforms: [<S as HistogramSample<L>>::EvenTransform; ACTIVE],
        storage: HistogramLaunchStorage<'_, C, ACTIVE, Output, Output, Queue>,
        geometry: HistogramLaunchGeometry<O>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Output: KernelSliceArgMut<Elem = C>,
        Queue: KernelSliceArgMut<Elem = i32>,
        S: TryFrom<usize> + core::ops::Add<Output = S> + core::ops::Mul<Output = S>;

    #[allow(clippy::too_many_arguments)]
    fn launch_histogram_range_arithmetic<Output, Queue>(
        &self,
        stream: &CudaStream,
        samples: &ArithmeticRange<S>,
        levels: HistogramLevelStorage<'_, L, ACTIVE>,
        storage: HistogramLaunchStorage<'_, C, ACTIVE, Output, Output, Queue>,
        geometry: HistogramLaunchGeometry<O>,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Output: KernelSliceArgMut<Elem = C>,
        Queue: KernelSliceArgMut<Elem = i32>,
        S: TryFrom<usize> + core::ops::Add<Output = S> + core::ops::Mul<Output = S>,
        L: PartialOrd;
}

/// The four offset/storage specializations reachable from one histogram call.
#[doc(hidden)]
pub trait HistogramPolicyKernels<
    S,
    C,
    L,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>:
    HistogramKernel<
        S,
        C,
        L,
        i32,
        CHANNELS,
        ACTIVE,
        THREADS,
        PIXELS,
        RLE,
        BYTE_SAMPLE,
        true,
        SHARED_BYTES,
    > + HistogramKernel<
        S,
        C,
        L,
        usize,
        CHANNELS,
        ACTIVE,
        THREADS,
        PIXELS,
        RLE,
        BYTE_SAMPLE,
        true,
        SHARED_BYTES,
    > + HistogramKernel<
        S,
        C,
        L,
        i32,
        CHANNELS,
        ACTIVE,
        THREADS,
        PIXELS,
        RLE,
        BYTE_SAMPLE,
        false,
        GLOBAL_BYTES,
    > + HistogramKernel<
        S,
        C,
        L,
        usize,
        CHANNELS,
        ACTIVE,
        THREADS,
        PIXELS,
        RLE,
        BYTE_SAMPLE,
        false,
        GLOBAL_BYTES,
    > where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
{
}

impl<
    S,
    C,
    L,
    M,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>
    HistogramPolicyKernels<
        S,
        C,
        L,
        CHANNELS,
        ACTIVE,
        THREADS,
        PIXELS,
        RLE,
        BYTE_SAMPLE,
        SHARED_BYTES,
        GLOBAL_BYTES,
    > for M
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    M: HistogramKernel<
            S,
            C,
            L,
            i32,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            true,
            SHARED_BYTES,
        > + HistogramKernel<
            S,
            C,
            L,
            usize,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            true,
            SHARED_BYTES,
        > + HistogramKernel<
            S,
            C,
            L,
            i32,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        > + HistogramKernel<
            S,
            C,
            L,
            usize,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >,
{
}

#[doc(hidden)]
pub trait HistogramHostCopy<C: DeviceCopy> {
    fn to_host_vec(&self, stream: &CudaStream) -> Result<Vec<C>, DeviceAlgorithmError>;
}

impl<C: DeviceCopy> HistogramHostCopy<C> for CudaDeviceBuffer<C> {
    fn to_host_vec(&self, stream: &CudaStream) -> Result<Vec<C>, DeviceAlgorithmError> {
        CudaDeviceBuffer::to_host_vec(self, stream).map_err(DeviceAlgorithmError::Driver)
    }
}

impl<C, A> HistogramHostCopy<C> for rccl_memory::DeviceBuffer<'_, C, A>
where
    C: DeviceCopy,
    A: DeviceAllocator,
{
    fn to_host_vec(&self, _stream: &CudaStream) -> Result<Vec<C>, DeviceAlgorithmError> {
        rccl_memory::DeviceBuffer::to_host_vec(self).map_err(DeviceAlgorithmError::Memory)
    }
}

/// A device-wide histogram whose channel storage is one affine allocation.
/// Channel `i` occupies `lengths()[i]` consecutive counters.  Keeping the
/// channels together replaces CUB's host array of raw output pointers while
/// preserving its independently-sized channel histograms.
pub struct DeviceHistogram<C: DeviceCopy, const ACTIVE_CHANNELS: usize, B = CudaDeviceBuffer<C>> {
    counters: B,
    lengths: [usize; ACTIVE_CHANNELS],
    offsets: [usize; ACTIVE_CHANNELS],
    _marker: core::marker::PhantomData<C>,
}

impl<C: DeviceCopy, const ACTIVE_CHANNELS: usize> DeviceHistogram<C, ACTIVE_CHANNELS> {
    pub fn new(
        stream: &CudaStream,
        lengths: [usize; ACTIVE_CHANNELS],
    ) -> Result<Self, DeviceAlgorithmError> {
        let (offsets, total) = channel_layout(lengths)?;
        let counters =
            CudaDeviceBuffer::zeroed(stream, total).map_err(DeviceAlgorithmError::Driver)?;
        Ok(Self {
            counters,
            lengths,
            offsets,
            _marker: core::marker::PhantomData,
        })
    }
}

impl<'stream, C, A, const ACTIVE_CHANNELS: usize>
    DeviceHistogram<C, ACTIVE_CHANNELS, rccl_memory::DeviceBuffer<'stream, C, A>>
where
    C: DeviceCopy,
    A: DeviceAllocator,
{
    pub fn new_in(
        allocator: A,
        stream: &'stream CudaStream,
        lengths: [usize; ACTIVE_CHANNELS],
    ) -> Result<Self, DeviceAlgorithmError> {
        let (offsets, total) = channel_layout(lengths)?;
        let counters = rccl_memory::DeviceBuffer::zeroed_in(allocator, stream, total)
            .map_err(DeviceAlgorithmError::Memory)?;
        Ok(Self {
            counters,
            lengths,
            offsets,
            _marker: core::marker::PhantomData,
        })
    }
}

impl<C, B, const ACTIVE_CHANNELS: usize> DeviceHistogram<C, ACTIVE_CHANNELS, B>
where
    C: DeviceCopy,
    B: DeviceRange,
{
    pub fn lengths(&self) -> &[usize; ACTIVE_CHANNELS] {
        &self.lengths
    }

    pub fn len(&self) -> usize {
        self.counters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counters.is_empty()
    }

    pub fn to_host_channels(
        &self,
        stream: &CudaStream,
    ) -> Result<[Vec<C>; ACTIVE_CHANNELS], DeviceAlgorithmError>
    where
        C: Clone,
        B: HistogramHostCopy<C>,
    {
        let flat = self.counters.to_host_vec(stream)?;
        Ok(core::array::from_fn(|channel| {
            let begin = self.offsets[channel];
            let end = begin + self.lengths[channel];
            flat[begin..end].to_vec()
        }))
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> (&mut B, [usize; ACTIVE_CHANNELS], [usize; ACTIVE_CHANNELS]) {
        (&mut self.counters, self.lengths, self.offsets)
    }
}

/// Borrowed level ranges over an existing device allocation.
///
/// This is the Rust-native counterpart of CUB accepting level iterators that
/// alias its sample allocation: the shared borrow keeps the allocation alive,
/// and each channel range is bounds-checked once on the host.
pub struct DeviceHistogramLevelsView<'a, L: DeviceCopy, const ACTIVE_CHANNELS: usize> {
    levels: &'a DeviceBuffer<L>,
    lengths: [usize; ACTIVE_CHANNELS],
    offsets: [usize; ACTIVE_CHANNELS],
}

impl<'a, L: DeviceCopy, const ACTIVE_CHANNELS: usize>
    DeviceHistogramLevelsView<'a, L, ACTIVE_CHANNELS>
{
    pub fn from_ranges(
        levels: &'a DeviceBuffer<L>,
        ranges: [Range<usize>; ACTIVE_CHANNELS],
    ) -> Result<Self, DeviceAlgorithmError> {
        let lengths = core::array::from_fn(|channel| {
            ranges[channel].end.saturating_sub(ranges[channel].start)
        });
        let offsets = core::array::from_fn(|channel| ranges[channel].start);
        for range in ranges {
            if range.start > range.end || range.end > levels.len() {
                return Err(DeviceAlgorithmError::InvalidHistogramShape(
                    "histogram level range lies outside its device allocation",
                ));
            }
        }
        Ok(Self {
            levels,
            lengths,
            offsets,
        })
    }

    pub fn lengths(&self) -> &[usize; ACTIVE_CHANNELS] {
        &self.lengths
    }

    #[doc(hidden)]
    pub fn __parts(
        &self,
    ) -> (
        &DeviceBuffer<L>,
        [usize; ACTIVE_CHANNELS],
        [usize; ACTIVE_CHANNELS],
    ) {
        (self.levels, self.lengths, self.offsets)
    }
}

/// Ordered bin boundaries for `HistogramRange`, stored channel-major.
pub struct DeviceHistogramLevels<L: DeviceCopy, const ACTIVE_CHANNELS: usize> {
    levels: DeviceBuffer<L>,
    lengths: [usize; ACTIVE_CHANNELS],
    offsets: [usize; ACTIVE_CHANNELS],
}

impl<L: DeviceCopy, const ACTIVE_CHANNELS: usize> DeviceHistogramLevels<L, ACTIVE_CHANNELS> {
    pub fn from_channels(
        stream: &CudaStream,
        channels: [&[L]; ACTIVE_CHANNELS],
    ) -> Result<Self, DeviceAlgorithmError> {
        let lengths = channels.map(<[L]>::len);
        let (offsets, total) = channel_layout(lengths)?;
        let mut flat = Vec::with_capacity(total);
        for channel in channels {
            flat.extend_from_slice(channel);
        }
        let levels =
            DeviceBuffer::from_host(stream, &flat).map_err(DeviceAlgorithmError::Driver)?;
        Ok(Self {
            levels,
            lengths,
            offsets,
        })
    }

    pub fn lengths(&self) -> &[usize; ACTIVE_CHANNELS] {
        &self.lengths
    }

    pub fn as_view(&self) -> DeviceHistogramLevelsView<'_, L, ACTIVE_CHANNELS> {
        DeviceHistogramLevelsView {
            levels: &self.levels,
            lengths: self.lengths,
            offsets: self.offsets,
        }
    }

    #[doc(hidden)]
    pub fn __parts(
        &self,
    ) -> (
        &DeviceBuffer<L>,
        [usize; ACTIVE_CHANNELS],
        [usize; ACTIVE_CHANNELS],
    ) {
        (&self.levels, self.lengths, self.offsets)
    }
}

/// Shared host view over owned and borrowed level-range containers.
#[doc(hidden)]
pub trait HistogramLevels<L: DeviceCopy, const ACTIVE: usize> {
    fn histogram_level_parts(&self) -> (&DeviceBuffer<L>, [usize; ACTIVE], [usize; ACTIVE]);
}

impl<L: DeviceCopy, const ACTIVE: usize> HistogramLevels<L, ACTIVE>
    for DeviceHistogramLevels<L, ACTIVE>
{
    fn histogram_level_parts(&self) -> (&DeviceBuffer<L>, [usize; ACTIVE], [usize; ACTIVE]) {
        self.__parts()
    }
}

impl<L: DeviceCopy, const ACTIVE: usize> HistogramLevels<L, ACTIVE>
    for DeviceHistogramLevelsView<'_, L, ACTIVE>
{
    fn histogram_level_parts(&self) -> (&DeviceBuffer<L>, [usize; ACTIVE], [usize; ACTIVE]) {
        self.__parts()
    }
}

/// Region of a row-major, interleaved sample allocation to histogram.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistogramRegion {
    /// Consume the complete allocation as one tightly packed row.
    All,
    /// Consume `rows` pitched rows with `pixels_per_row` active pixels each.
    Rows {
        pixels_per_row: usize,
        rows: usize,
        row_stride_samples: usize,
    },
}

impl HistogramRegion {
    pub const fn rows(pixels_per_row: usize, rows: usize, row_stride_samples: usize) -> Self {
        Self::Rows {
            pixels_per_row,
            rows,
            row_stride_samples,
        }
    }

    #[doc(hidden)]
    pub fn __resolve<const CHANNELS: usize>(
        self,
        sample_count: usize,
    ) -> Result<(usize, usize, usize), DeviceAlgorithmError> {
        if CHANNELS == 0 || CHANNELS > 4 {
            return Err(DeviceAlgorithmError::InvalidHistogramShape(
                "histogram channel count must be in 1..=4",
            ));
        }
        match self {
            Self::All => {
                if !sample_count.is_multiple_of(CHANNELS) {
                    return Err(DeviceAlgorithmError::LengthMismatch {
                        role: "interleaved histogram samples",
                        expected: sample_count.div_ceil(CHANNELS) * CHANNELS,
                        actual: sample_count,
                    });
                }
                Ok((sample_count / CHANNELS, 1, sample_count))
            }
            Self::Rows {
                pixels_per_row,
                rows,
                row_stride_samples,
            } => {
                let active_samples = pixels_per_row
                    .checked_mul(CHANNELS)
                    .ok_or(DeviceAlgorithmError::SizeOverflow)?;
                if row_stride_samples < active_samples {
                    return Err(DeviceAlgorithmError::InvalidHistogramShape(
                        "histogram row stride is shorter than its active pixels",
                    ));
                }
                let required = if rows == 0 {
                    0
                } else {
                    (rows - 1)
                        .checked_mul(row_stride_samples)
                        .and_then(|prefix| prefix.checked_add(active_samples))
                        .ok_or(DeviceAlgorithmError::SizeOverflow)?
                };
                if required > sample_count {
                    return Err(DeviceAlgorithmError::LengthMismatch {
                        role: "pitched histogram samples",
                        expected: required,
                        actual: sample_count,
                    });
                }
                // CUB folds a tightly packed 2D image into one linear row
                // before occupancy and grid geometry are computed.  Preserve
                // that algorithmic choice instead of merely producing the
                // same counters with a different launch shape.
                if row_stride_samples == active_samples {
                    let linear_pixels = pixels_per_row
                        .checked_mul(rows)
                        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
                    let linear_samples = linear_pixels
                        .checked_mul(CHANNELS)
                        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
                    return Ok((linear_pixels, 1, linear_samples));
                }
                Ok((pixels_per_row, rows, row_stride_samples))
            }
        }
    }
}

/// Mirrors CUB's host-side `OffsetT -> int` narrowing guard for histogram
/// image traversal. Overflow keeps the pointer-sized path.
#[doc(hidden)]
pub fn histogram_uses_i32_offsets<S>(num_rows: usize, row_stride_samples: usize) -> bool {
    num_rows
        .checked_mul(row_stride_samples)
        .and_then(|samples| samples.checked_mul(core::mem::size_of::<S>()))
        .is_some_and(|bytes| bytes < i32::MAX as usize)
}

/// Opaque CUB histogram privatization and queue storage.
pub struct DeviceHistogramWorkspace<
    C: DeviceCopy,
    const ACTIVE_CHANNELS: usize,
    B = CudaDeviceBuffer<C>,
    Q = CudaDeviceBuffer<i32>,
> {
    privatized: B,
    queue: Q,
    privatized_lengths: [usize; ACTIVE_CHANNELS],
    privatized_offsets: [usize; ACTIVE_CHANNELS],
    blocks_per_row: usize,
    blocks_per_col: usize,
    region: (usize, usize, usize),
    narrow_offsets: bool,
    _marker: core::marker::PhantomData<C>,
}

/// Affine decomposition used only by the generated histogram launch.
#[doc(hidden)]
pub struct DeviceHistogramWorkspaceParts<
    'a,
    C: DeviceCopy,
    const ACTIVE_CHANNELS: usize,
    B = CudaDeviceBuffer<C>,
    Q = CudaDeviceBuffer<i32>,
> {
    pub privatized: &'a mut B,
    pub queue: &'a mut Q,
    pub privatized_lengths: [usize; ACTIVE_CHANNELS],
    pub privatized_offsets: [usize; ACTIVE_CHANNELS],
    pub blocks_per_row: usize,
    pub blocks_per_col: usize,
    pub region: (usize, usize, usize),
    pub narrow_offsets: bool,
    pub __marker: core::marker::PhantomData<C>,
}

impl<C: DeviceCopy, const ACTIVE_CHANNELS: usize> DeviceHistogramWorkspace<C, ACTIVE_CHANNELS> {
    #[doc(hidden)]
    pub fn __new(
        stream: &CudaStream,
        privatized_lengths: [usize; ACTIVE_CHANNELS],
        blocks_per_row: usize,
        blocks_per_col: usize,
        region: (usize, usize, usize),
        narrow_offsets: bool,
    ) -> Result<Self, DeviceAlgorithmError> {
        let (privatized_offsets, total) =
            workspace_channel_layout(privatized_lengths, blocks_per_row, blocks_per_col)?;
        let privatized =
            CudaDeviceBuffer::zeroed(stream, total).map_err(DeviceAlgorithmError::Driver)?;
        let queue = CudaDeviceBuffer::zeroed(stream, 1).map_err(DeviceAlgorithmError::Driver)?;
        Ok(Self {
            privatized,
            queue,
            privatized_lengths,
            privatized_offsets,
            blocks_per_row,
            blocks_per_col,
            region,
            narrow_offsets,
            _marker: core::marker::PhantomData,
        })
    }
}

impl<'stream, C, A, const ACTIVE_CHANNELS: usize>
    DeviceHistogramWorkspace<
        C,
        ACTIVE_CHANNELS,
        rccl_memory::DeviceBuffer<'stream, C, A>,
        rccl_memory::DeviceBuffer<'stream, i32, A>,
    >
where
    C: DeviceCopy,
    A: DeviceAllocator + Clone,
{
    #[doc(hidden)]
    pub fn __new_in(
        allocator: A,
        stream: &'stream CudaStream,
        privatized_lengths: [usize; ACTIVE_CHANNELS],
        blocks_per_row: usize,
        blocks_per_col: usize,
        region: (usize, usize, usize),
        narrow_offsets: bool,
    ) -> Result<Self, DeviceAlgorithmError> {
        let (privatized_offsets, total) =
            workspace_channel_layout(privatized_lengths, blocks_per_row, blocks_per_col)?;
        let privatized = rccl_memory::DeviceBuffer::zeroed_in(allocator.clone(), stream, total)
            .map_err(DeviceAlgorithmError::Memory)?;
        let queue = rccl_memory::DeviceBuffer::zeroed_in(allocator, stream, 1)
            .map_err(DeviceAlgorithmError::Memory)?;
        Ok(Self {
            privatized,
            queue,
            privatized_lengths,
            privatized_offsets,
            blocks_per_row,
            blocks_per_col,
            region,
            narrow_offsets,
            _marker: core::marker::PhantomData,
        })
    }
}

impl<C, B, Q, const ACTIVE_CHANNELS: usize> DeviceHistogramWorkspace<C, ACTIVE_CHANNELS, B, Q>
where
    C: DeviceCopy,
{
    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> DeviceHistogramWorkspaceParts<'_, C, ACTIVE_CHANNELS, B, Q> {
        DeviceHistogramWorkspaceParts::<C, ACTIVE_CHANNELS, B, Q> {
            privatized: &mut self.privatized,
            queue: &mut self.queue,
            privatized_lengths: self.privatized_lengths,
            privatized_offsets: self.privatized_offsets,
            blocks_per_row: self.blocks_per_row,
            blocks_per_col: self.blocks_per_col,
            region: self.region,
            narrow_offsets: self.narrow_offsets,
            __marker: core::marker::PhantomData,
        }
    }
}

struct HistogramWorkspaceLayout<const ACTIVE: usize> {
    privatized_lengths: [usize; ACTIVE],
    blocks_per_row: usize,
    blocks_per_col: usize,
    region: (usize, usize, usize),
    narrow_offsets: bool,
}

impl<const ACTIVE: usize> HistogramWorkspaceLayout<ACTIVE> {
    fn allocate<C: DeviceCopy>(
        self,
        stream: &CudaStream,
    ) -> Result<DeviceHistogramWorkspace<C, ACTIVE>, DeviceAlgorithmError> {
        DeviceHistogramWorkspace::__new(
            stream,
            self.privatized_lengths,
            self.blocks_per_row,
            self.blocks_per_col,
            self.region,
            self.narrow_offsets,
        )
    }

    fn allocate_in<'stream, C, A>(
        self,
        allocator: A,
        stream: &'stream CudaStream,
    ) -> Result<
        DeviceHistogramWorkspace<
            C,
            ACTIVE,
            rccl_memory::DeviceBuffer<'stream, C, A>,
            rccl_memory::DeviceBuffer<'stream, i32, A>,
        >,
        DeviceAlgorithmError,
    >
    where
        C: DeviceCopy,
        A: DeviceAllocator + Clone,
    {
        DeviceHistogramWorkspace::__new_in(
            allocator,
            stream,
            self.privatized_lengths,
            self.blocks_per_row,
            self.blocks_per_col,
            self.region,
            self.narrow_offsets,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn histogram_workspace_variant<
    S,
    C,
    L,
    M,
    O,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const EVEN: bool,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED: bool,
    const SHARED_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    privatized_lengths: [usize; ACTIVE],
    region: (usize, usize, usize),
    narrow_offsets: bool,
) -> Result<HistogramWorkspaceLayout<ACTIVE>, DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    M: HistogramKernel<
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
        >,
    O: HistogramOffset,
{
    let blocks_per_sm = if EVEN {
        module.histogram_even_occupancy()?
    } else {
        module.histogram_range_occupancy()?
    };
    let sms = stream
        .context()
        .multiprocessor_count()
        .map_err(DeviceAlgorithmError::Driver)?;
    let occupancy = (blocks_per_sm as usize)
        .checked_mul(sms as usize)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let tile_pixels = (THREADS as usize)
        .checked_mul(PIXELS)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let tiles_per_row = region.0.div_ceil(tile_pixels);
    let blocks_per_row = occupancy.min(tiles_per_row);
    let blocks_per_col = occupancy
        .checked_div(blocks_per_row)
        .unwrap_or(0)
        .min(region.1);
    Ok(HistogramWorkspaceLayout {
        privatized_lengths,
        blocks_per_row,
        blocks_per_col,
        region,
        narrow_offsets,
    })
}

/// Compute reusable histogram workspace geometry using the cfg-selected policy.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
fn device_histogram_workspace_layout<
    S,
    C,
    L,
    Samples,
    M,
    B,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const EVEN: bool,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &Samples,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE, B>,
) -> Result<HistogramWorkspaceLayout<ACTIVE>, DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    Samples: DeviceRange + ?Sized,
    B: DeviceRange,
    M: HistogramPolicyKernels<
            S,
            C,
            L,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED_BYTES,
            GLOBAL_BYTES,
        >,
{
    let resolved = region.__resolve::<CHANNELS>(samples.len())?;
    let output_lengths = *output.lengths();
    if output_lengths.contains(&0) {
        return Err(DeviceAlgorithmError::InvalidHistogramShape(
            "every histogram channel requires at least one bin",
        ));
    }
    if output_lengths
        .iter()
        .any(|&length| length >= i32::MAX as usize)
    {
        return Err(DeviceAlgorithmError::InvalidHistogramShape(
            "histogram channel exceeds CUB's signed level-count domain",
        ));
    }
    let max_bins = output_lengths.iter().copied().max().unwrap_or(0);
    let privatized_lengths = if BYTE_SAMPLE {
        [256; ACTIVE]
    } else {
        output_lengths
    };
    let narrow_offsets = histogram_uses_i32_offsets::<S>(resolved.1, resolved.2);
    let layout = if BYTE_SAMPLE || max_bins <= 256 {
        if narrow_offsets {
            histogram_workspace_variant::<
                S,
                C,
                L,
                M,
                i32,
                CHANNELS,
                ACTIVE,
                EVEN,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, privatized_lengths, resolved, true)
        } else {
            histogram_workspace_variant::<
                S,
                C,
                L,
                M,
                usize,
                CHANNELS,
                ACTIVE,
                EVEN,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, privatized_lengths, resolved, false)
        }
    } else if narrow_offsets {
        histogram_workspace_variant::<
            S,
            C,
            L,
            M,
            i32,
            CHANNELS,
            ACTIVE,
            EVEN,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, privatized_lengths, resolved, true)
    } else {
        histogram_workspace_variant::<
            S,
            C,
            L,
            M,
            usize,
            CHANNELS,
            ACTIVE,
            EVEN,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, privatized_lengths, resolved, false)
    }?;
    Ok(layout)
}

/// Allocate the reusable histogram workspace using the cfg-selected policy.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_histogram_workspace<
    S,
    C,
    L,
    Samples,
    M,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const EVEN: bool,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &Samples,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE>,
) -> Result<DeviceHistogramWorkspace<C, ACTIVE>, DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    Samples: DeviceRange + ?Sized,
    M: HistogramPolicyKernels<
            S,
            C,
            L,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED_BYTES,
            GLOBAL_BYTES,
        >,
{
    let layout = device_histogram_workspace_layout::<
        S,
        C,
        L,
        Samples,
        M,
        CudaDeviceBuffer<C>,
        CHANNELS,
        ACTIVE,
        EVEN,
        THREADS,
        PIXELS,
        RLE,
        BYTE_SAMPLE,
        SHARED_BYTES,
        GLOBAL_BYTES,
    >(stream, module, samples, region, output)?;
    layout.allocate(stream)
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_histogram_workspace_in<
    'stream,
    S,
    C,
    L,
    Samples,
    M,
    A,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const EVEN: bool,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>(
    allocator: A,
    stream: &'stream CudaStream,
    module: &M,
    samples: &Samples,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE, rccl_memory::DeviceBuffer<'stream, C, A>>,
) -> Result<
    DeviceHistogramWorkspace<
        C,
        ACTIVE,
        rccl_memory::DeviceBuffer<'stream, C, A>,
        rccl_memory::DeviceBuffer<'stream, i32, A>,
    >,
    DeviceAlgorithmError,
>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    Samples: DeviceRange + ?Sized,
    A: DeviceAllocator + Clone,
    M: HistogramPolicyKernels<
            S,
            C,
            L,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED_BYTES,
            GLOBAL_BYTES,
        >,
{
    let layout = device_histogram_workspace_layout::<
        S,
        C,
        L,
        Samples,
        M,
        rccl_memory::DeviceBuffer<'stream, C, A>,
        CHANNELS,
        ACTIVE,
        EVEN,
        THREADS,
        PIXELS,
        RLE,
        BYTE_SAMPLE,
        SHARED_BYTES,
        GLOBAL_BYTES,
    >(stream, module, samples, region, output)?;
    layout.allocate_in(allocator, stream)
}

#[allow(clippy::too_many_arguments)]
fn histogram_arithmetic_workspace_variant<
    S,
    C,
    L,
    M,
    O,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const EVEN: bool,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED: bool,
    const SHARED_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    privatized_lengths: [usize; ACTIVE],
    region: (usize, usize, usize),
    narrow_offsets: bool,
) -> Result<HistogramWorkspaceLayout<ACTIVE>, DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    M: HistogramKernel<
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
        >,
    O: HistogramOffset,
{
    let blocks_per_sm = if EVEN {
        module.histogram_even_arithmetic_occupancy()?
    } else {
        module.histogram_range_arithmetic_occupancy()?
    };
    let sms = stream
        .context()
        .multiprocessor_count()
        .map_err(DeviceAlgorithmError::Driver)?;
    let occupancy = (blocks_per_sm as usize)
        .checked_mul(sms as usize)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let tile_pixels = (THREADS as usize)
        .checked_mul(PIXELS)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let tiles_per_row = region.0.div_ceil(tile_pixels);
    let blocks_per_row = occupancy.min(tiles_per_row);
    let blocks_per_col = occupancy
        .checked_div(blocks_per_row)
        .unwrap_or(0)
        .min(region.1);
    Ok(HistogramWorkspaceLayout {
        privatized_lengths,
        blocks_per_row,
        blocks_per_col,
        region,
        narrow_offsets,
    })
}

/// Arithmetic-range workspace layout variant with its honest sample capability bound.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
fn device_histogram_arithmetic_workspace_layout<
    S,
    C,
    L,
    M,
    B,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const EVEN: bool,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE, B>,
) -> Result<HistogramWorkspaceLayout<ACTIVE>, DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    B: DeviceRange,
    M: HistogramPolicyKernels<
            S,
            C,
            L,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED_BYTES,
            GLOBAL_BYTES,
        >,
{
    let resolved = region.__resolve::<CHANNELS>(samples.len())?;
    let output_lengths = *output.lengths();
    if output_lengths.contains(&0) {
        return Err(DeviceAlgorithmError::InvalidHistogramShape(
            "every histogram channel requires at least one bin",
        ));
    }
    if output_lengths
        .iter()
        .any(|&length| length >= i32::MAX as usize)
    {
        return Err(DeviceAlgorithmError::InvalidHistogramShape(
            "histogram channel exceeds CUB's signed level-count domain",
        ));
    }
    let max_bins = output_lengths.iter().copied().max().unwrap_or(0);
    let privatized_lengths = if BYTE_SAMPLE {
        [256; ACTIVE]
    } else {
        output_lengths
    };
    let narrow_offsets = histogram_uses_i32_offsets::<S>(resolved.1, resolved.2);
    let layout = if BYTE_SAMPLE || max_bins <= 256 {
        if narrow_offsets {
            histogram_arithmetic_workspace_variant::<
                S,
                C,
                L,
                M,
                i32,
                CHANNELS,
                ACTIVE,
                EVEN,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, privatized_lengths, resolved, true)
        } else {
            histogram_arithmetic_workspace_variant::<
                S,
                C,
                L,
                M,
                usize,
                CHANNELS,
                ACTIVE,
                EVEN,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, privatized_lengths, resolved, false)
        }
    } else if narrow_offsets {
        histogram_arithmetic_workspace_variant::<
            S,
            C,
            L,
            M,
            i32,
            CHANNELS,
            ACTIVE,
            EVEN,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, privatized_lengths, resolved, true)
    } else {
        histogram_arithmetic_workspace_variant::<
            S,
            C,
            L,
            M,
            usize,
            CHANNELS,
            ACTIVE,
            EVEN,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, privatized_lengths, resolved, false)
    }?;
    Ok(layout)
}

/// Arithmetic-range workspace variant with its honest sample capability bound.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_histogram_arithmetic_workspace<
    S,
    C,
    L,
    M,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const EVEN: bool,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE>,
) -> Result<DeviceHistogramWorkspace<C, ACTIVE>, DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    M: HistogramPolicyKernels<
            S,
            C,
            L,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED_BYTES,
            GLOBAL_BYTES,
        >,
{
    let layout = device_histogram_arithmetic_workspace_layout::<
        S,
        C,
        L,
        M,
        CudaDeviceBuffer<C>,
        CHANNELS,
        ACTIVE,
        EVEN,
        THREADS,
        PIXELS,
        RLE,
        BYTE_SAMPLE,
        SHARED_BYTES,
        GLOBAL_BYTES,
    >(stream, module, samples, region, output)?;
    layout.allocate(stream)
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_histogram_arithmetic_workspace_in<
    'stream,
    S,
    C,
    L,
    M,
    A,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const EVEN: bool,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>(
    allocator: A,
    stream: &'stream CudaStream,
    module: &M,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &DeviceHistogram<C, ACTIVE, rccl_memory::DeviceBuffer<'stream, C, A>>,
) -> Result<
    DeviceHistogramWorkspace<
        C,
        ACTIVE,
        rccl_memory::DeviceBuffer<'stream, C, A>,
        rccl_memory::DeviceBuffer<'stream, i32, A>,
    >,
    DeviceAlgorithmError,
>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    A: DeviceAllocator + Clone,
    M: HistogramPolicyKernels<
            S,
            C,
            L,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED_BYTES,
            GLOBAL_BYTES,
        >,
{
    let layout = device_histogram_arithmetic_workspace_layout::<
        S,
        C,
        L,
        M,
        rccl_memory::DeviceBuffer<'stream, C, A>,
        CHANNELS,
        ACTIVE,
        EVEN,
        THREADS,
        PIXELS,
        RLE,
        BYTE_SAMPLE,
        SHARED_BYTES,
        GLOBAL_BYTES,
    >(stream, module, samples, region, output)?;
    layout.allocate_in(allocator, stream)
}

struct PreparedHistogramLaunch<'a, C: DeviceCopy, const ACTIVE: usize, Output, Queue> {
    storage: HistogramLaunchStorage<'a, C, ACTIVE, Output, Output, Queue>,
    blocks_per_row: usize,
    blocks_per_col: usize,
    narrow_offsets: bool,
}

fn prepare_histogram_launch<'a, C, Output, Queue, const ACTIVE: usize>(
    output: &'a mut DeviceHistogram<C, ACTIVE, Output>,
    workspace: &'a mut DeviceHistogramWorkspace<C, ACTIVE, Output, Queue>,
    resolved: (usize, usize, usize),
    byte_sample: bool,
) -> Result<PreparedHistogramLaunch<'a, C, ACTIVE, Output, Queue>, DeviceAlgorithmError>
where
    C: DeviceCopy,
    Output: DeviceRange + KernelSliceArgMut<Elem = C>,
    Queue: KernelSliceArgMut<Elem = i32>,
{
    let (output, output_lengths, output_offsets) = output.__parts_mut();
    let DeviceHistogramWorkspaceParts {
        privatized,
        queue,
        privatized_lengths,
        privatized_offsets,
        blocks_per_row,
        blocks_per_col,
        region,
        narrow_offsets,
        ..
    } = workspace.__parts_mut();
    if region != resolved {
        return Err(DeviceAlgorithmError::InvalidHistogramShape(
            "histogram workspace does not match this image region",
        ));
    }
    let expected_privatized_lengths = if byte_sample {
        [256; ACTIVE]
    } else {
        output_lengths
    };
    if privatized_lengths != expected_privatized_lengths {
        return Err(DeviceAlgorithmError::InvalidHistogramShape(
            "histogram workspace does not match the output bin layout",
        ));
    }
    Ok(PreparedHistogramLaunch {
        storage: HistogramLaunchStorage {
            output,
            privatized,
            queue,
            output_lengths,
            output_offsets,
            privatized_lengths,
            privatized_offsets,
            __marker: core::marker::PhantomData,
        },
        blocks_per_row,
        blocks_per_col,
        narrow_offsets,
    })
}

fn histogram_geometry<O, const THREADS: u32, const PIXELS: usize>(
    resolved: (usize, usize, usize),
    blocks_per_row: usize,
    blocks_per_col: usize,
) -> Result<HistogramLaunchGeometry<O>, DeviceAlgorithmError>
where
    O: HistogramOffset,
{
    let tile_pixels = (THREADS as usize)
        .checked_mul(PIXELS)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let tiles_per_row = resolved.0.div_ceil(tile_pixels);
    if tiles_per_row > i32::MAX as usize {
        return Err(DeviceAlgorithmError::InvalidHistogramShape(
            "histogram row exceeds CUB's signed tile-count domain",
        ));
    }
    Ok(HistogramLaunchGeometry {
        num_row_pixels: <O as HistogramOffset>::from_usize(resolved.0),
        num_rows: <O as HistogramOffset>::from_usize(resolved.1),
        row_stride_samples: <O as HistogramOffset>::from_usize(resolved.2),
        tiles_per_row: tiles_per_row as i32,
        blocks_per_row,
        blocks_per_col,
    })
}

#[allow(clippy::too_many_arguments)]
fn launch_histogram_even_variant<
    S,
    C,
    L,
    Samples,
    M,
    O,
    Output,
    Queue,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED: bool,
    const SHARED_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &Samples,
    transforms: [<S as HistogramSample<L>>::EvenTransform; ACTIVE],
    prepared: PreparedHistogramLaunch<'_, C, ACTIVE, Output, Queue>,
    resolved: (usize, usize, usize),
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    Samples: KernelSliceArg<Elem = S>,
    M: HistogramKernel<
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
        >,
    O: HistogramOffset,
    Output: KernelSliceArgMut<Elem = C>,
    Queue: KernelSliceArgMut<Elem = i32>,
{
    let geometry = histogram_geometry::<O, THREADS, PIXELS>(
        resolved,
        prepared.blocks_per_row,
        prepared.blocks_per_col,
    )?;
    let blocks = prepared
        .blocks_per_row
        .checked_mul(prepared.blocks_per_col)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if blocks == 0 {
        HistogramInitKernel::launch_histogram_init(module, stream, prepared.storage)
    } else {
        module.launch_histogram_even::<Samples, Output, Queue>(
            stream,
            samples,
            transforms,
            prepared.storage,
            geometry,
        )
    }
}

/// Launch an even histogram using ordinary Rust validation and routing.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_histogram_even<
    S,
    C,
    L,
    Samples,
    M,
    Output,
    Queue,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &Samples,
    region: HistogramRegion,
    output: &mut DeviceHistogram<C, ACTIVE, Output>,
    lower_levels: [L; ACTIVE],
    upper_levels: [L; ACTIVE],
    workspace: &mut DeviceHistogramWorkspace<C, ACTIVE, Output, Queue>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    Samples: DeviceRange + KernelSliceArg<Elem = S>,
    M: HistogramPolicyKernels<
            S,
            C,
            L,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED_BYTES,
            GLOBAL_BYTES,
        >,
    Output: DeviceRange + KernelSliceArgMut<Elem = C>,
    Queue: KernelSliceArgMut<Elem = i32>,
{
    let resolved = region.__resolve::<CHANNELS>(DeviceRange::len(samples))?;
    let narrow_offsets = histogram_uses_i32_offsets::<S>(resolved.1, resolved.2);
    let prepared = prepare_histogram_launch(output, workspace, resolved, BYTE_SAMPLE)?;
    if prepared.narrow_offsets != narrow_offsets {
        return Err(DeviceAlgorithmError::InvalidHistogramShape(
            "histogram workspace does not match the image offset domain",
        ));
    }
    for channel in 0..ACTIVE {
        if S::even_may_overflow(
            lower_levels[channel],
            upper_levels[channel],
            prepared.storage.output_lengths[channel],
        ) {
            return Err(DeviceAlgorithmError::InvalidHistogramShape(
                "even histogram bounds or bin arithmetic are invalid",
            ));
        }
    }
    let transforms = core::array::from_fn(|channel| {
        S::even_transform(
            lower_levels[channel],
            upper_levels[channel],
            prepared.storage.output_lengths[channel] as i32,
        )
    });
    let max_bins = prepared
        .storage
        .output_lengths
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if BYTE_SAMPLE || max_bins <= 256 {
        if narrow_offsets {
            launch_histogram_even_variant::<
                S,
                C,
                L,
                Samples,
                M,
                i32,
                Output,
                Queue,
                CHANNELS,
                ACTIVE,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, samples, transforms, prepared, resolved)
        } else {
            launch_histogram_even_variant::<
                S,
                C,
                L,
                Samples,
                M,
                usize,
                Output,
                Queue,
                CHANNELS,
                ACTIVE,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, samples, transforms, prepared, resolved)
        }
    } else if narrow_offsets {
        launch_histogram_even_variant::<
            S,
            C,
            L,
            Samples,
            M,
            i32,
            Output,
            Queue,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, samples, transforms, prepared, resolved)
    } else {
        launch_histogram_even_variant::<
            S,
            C,
            L,
            Samples,
            M,
            usize,
            Output,
            Queue,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, samples, transforms, prepared, resolved)
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_histogram_even_arithmetic_variant<
    S,
    C,
    L,
    M,
    O,
    Output,
    Queue,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED: bool,
    const SHARED_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &ArithmeticRange<S>,
    transforms: [<S as HistogramSample<L>>::EvenTransform; ACTIVE],
    prepared: PreparedHistogramLaunch<'_, C, ACTIVE, Output, Queue>,
    resolved: (usize, usize, usize),
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    M: HistogramKernel<
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
        >,
    O: HistogramOffset,
    Output: KernelSliceArgMut<Elem = C>,
    Queue: KernelSliceArgMut<Elem = i32>,
{
    let geometry = histogram_geometry::<O, THREADS, PIXELS>(
        resolved,
        prepared.blocks_per_row,
        prepared.blocks_per_col,
    )?;
    let blocks = prepared
        .blocks_per_row
        .checked_mul(prepared.blocks_per_col)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if blocks == 0 {
        HistogramInitKernel::launch_histogram_init(module, stream, prepared.storage)
    } else {
        module.launch_histogram_even_arithmetic(
            stream,
            samples,
            transforms,
            prepared.storage,
            geometry,
        )
    }
}

/// Arithmetic-range variant of [`device_histogram_even`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_histogram_even_arithmetic<
    S,
    C,
    L,
    M,
    Output,
    Queue,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &mut DeviceHistogram<C, ACTIVE, Output>,
    lower_levels: [L; ACTIVE],
    upper_levels: [L; ACTIVE],
    workspace: &mut DeviceHistogramWorkspace<C, ACTIVE, Output, Queue>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    M: HistogramPolicyKernels<
            S,
            C,
            L,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED_BYTES,
            GLOBAL_BYTES,
        >,
    Output: DeviceRange + KernelSliceArgMut<Elem = C>,
    Queue: KernelSliceArgMut<Elem = i32>,
{
    let resolved = region.__resolve::<CHANNELS>(samples.len())?;
    let narrow_offsets = histogram_uses_i32_offsets::<S>(resolved.1, resolved.2);
    let prepared = prepare_histogram_launch(output, workspace, resolved, BYTE_SAMPLE)?;
    if prepared.narrow_offsets != narrow_offsets {
        return Err(DeviceAlgorithmError::InvalidHistogramShape(
            "histogram workspace does not match the image offset domain",
        ));
    }
    for channel in 0..ACTIVE {
        if S::even_may_overflow(
            lower_levels[channel],
            upper_levels[channel],
            prepared.storage.output_lengths[channel],
        ) {
            return Err(DeviceAlgorithmError::InvalidHistogramShape(
                "even histogram bounds or bin arithmetic are invalid",
            ));
        }
    }
    let transforms = core::array::from_fn(|channel| {
        S::even_transform(
            lower_levels[channel],
            upper_levels[channel],
            prepared.storage.output_lengths[channel] as i32,
        )
    });
    let max_bins = prepared
        .storage
        .output_lengths
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if BYTE_SAMPLE || max_bins <= 256 {
        if narrow_offsets {
            launch_histogram_even_arithmetic_variant::<
                S,
                C,
                L,
                M,
                i32,
                Output,
                Queue,
                CHANNELS,
                ACTIVE,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, samples, transforms, prepared, resolved)
        } else {
            launch_histogram_even_arithmetic_variant::<
                S,
                C,
                L,
                M,
                usize,
                Output,
                Queue,
                CHANNELS,
                ACTIVE,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, samples, transforms, prepared, resolved)
        }
    } else if narrow_offsets {
        launch_histogram_even_arithmetic_variant::<
            S,
            C,
            L,
            M,
            i32,
            Output,
            Queue,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, samples, transforms, prepared, resolved)
    } else {
        launch_histogram_even_arithmetic_variant::<
            S,
            C,
            L,
            M,
            usize,
            Output,
            Queue,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, samples, transforms, prepared, resolved)
    }
}

fn validate_histogram_levels<'a, L, Levels, const ACTIVE: usize>(
    levels: &'a Levels,
    output_lengths: [usize; ACTIVE],
) -> Result<HistogramLevelStorage<'a, L, ACTIVE>, DeviceAlgorithmError>
where
    L: DeviceCopy + Copy,
    Levels: HistogramLevels<L, ACTIVE> + ?Sized,
{
    let (storage, level_lengths, level_offsets) = levels.histogram_level_parts();
    for channel in 0..ACTIVE {
        let expected = output_lengths[channel]
            .checked_add(1)
            .ok_or(DeviceAlgorithmError::SizeOverflow)?;
        if level_lengths[channel] != expected {
            return Err(DeviceAlgorithmError::LengthMismatch {
                role: "histogram range levels",
                expected,
                actual: level_lengths[channel],
            });
        }
        let Some(end) = level_offsets[channel].checked_add(level_lengths[channel]) else {
            return Err(DeviceAlgorithmError::SizeOverflow);
        };
        if level_lengths[channel] > i32::MAX as usize || end > storage.len() {
            return Err(DeviceAlgorithmError::InvalidHistogramShape(
                "histogram level ranges must be nonempty and lie in their device allocation",
            ));
        }
    }
    Ok(HistogramLevelStorage {
        storage,
        lengths: level_lengths,
        offsets: level_offsets,
    })
}

#[allow(clippy::too_many_arguments)]
fn launch_histogram_range_variant<
    S,
    C,
    L,
    Samples,
    M,
    O,
    Output,
    Queue,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED: bool,
    const SHARED_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &Samples,
    levels: HistogramLevelStorage<'_, L, ACTIVE>,
    prepared: PreparedHistogramLaunch<'_, C, ACTIVE, Output, Queue>,
    resolved: (usize, usize, usize),
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    Samples: KernelSliceArg<Elem = S>,
    M: HistogramKernel<
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
        >,
    O: HistogramOffset,
    Output: KernelSliceArgMut<Elem = C>,
    Queue: KernelSliceArgMut<Elem = i32>,
{
    let geometry = histogram_geometry::<O, THREADS, PIXELS>(
        resolved,
        prepared.blocks_per_row,
        prepared.blocks_per_col,
    )?;
    let blocks = prepared
        .blocks_per_row
        .checked_mul(prepared.blocks_per_col)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if blocks == 0 {
        HistogramInitKernel::launch_histogram_init(module, stream, prepared.storage)
    } else {
        module.launch_histogram_range::<Samples, Output, Queue>(
            stream,
            samples,
            levels,
            prepared.storage,
            geometry,
        )
    }
}

/// Launch a boundary-defined histogram using ordinary Rust routing.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_histogram_range<
    S,
    C,
    L,
    Samples,
    Levels,
    M,
    Output,
    Queue,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &Samples,
    region: HistogramRegion,
    output: &mut DeviceHistogram<C, ACTIVE, Output>,
    levels: &Levels,
    workspace: &mut DeviceHistogramWorkspace<C, ACTIVE, Output, Queue>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar + HistogramSample<L> + HistogramFourSampleLoad,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    Samples: DeviceRange + KernelSliceArg<Elem = S>,
    Levels: HistogramLevels<L, ACTIVE> + ?Sized,
    M: HistogramPolicyKernels<
            S,
            C,
            L,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED_BYTES,
            GLOBAL_BYTES,
        >,
    Output: DeviceRange + KernelSliceArgMut<Elem = C>,
    Queue: KernelSliceArgMut<Elem = i32>,
{
    let resolved = region.__resolve::<CHANNELS>(DeviceRange::len(samples))?;
    let output_lengths = *output.lengths();
    let levels = validate_histogram_levels(levels, output_lengths)?;
    let narrow_offsets = histogram_uses_i32_offsets::<S>(resolved.1, resolved.2);
    let prepared = prepare_histogram_launch(output, workspace, resolved, BYTE_SAMPLE)?;
    if prepared.narrow_offsets != narrow_offsets {
        return Err(DeviceAlgorithmError::InvalidHistogramShape(
            "histogram workspace does not match the image offset domain",
        ));
    }
    let max_bins = prepared
        .storage
        .output_lengths
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if BYTE_SAMPLE || max_bins <= 256 {
        if narrow_offsets {
            launch_histogram_range_variant::<
                S,
                C,
                L,
                Samples,
                M,
                i32,
                Output,
                Queue,
                CHANNELS,
                ACTIVE,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, samples, levels, prepared, resolved)
        } else {
            launch_histogram_range_variant::<
                S,
                C,
                L,
                Samples,
                M,
                usize,
                Output,
                Queue,
                CHANNELS,
                ACTIVE,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, samples, levels, prepared, resolved)
        }
    } else if narrow_offsets {
        launch_histogram_range_variant::<
            S,
            C,
            L,
            Samples,
            M,
            i32,
            Output,
            Queue,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, samples, levels, prepared, resolved)
    } else {
        launch_histogram_range_variant::<
            S,
            C,
            L,
            Samples,
            M,
            usize,
            Output,
            Queue,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, samples, levels, prepared, resolved)
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_histogram_range_arithmetic_variant<
    S,
    C,
    L,
    M,
    O,
    Output,
    Queue,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED: bool,
    const SHARED_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &ArithmeticRange<S>,
    levels: HistogramLevelStorage<'_, L, ACTIVE>,
    prepared: PreparedHistogramLaunch<'_, C, ACTIVE, Output, Queue>,
    resolved: (usize, usize, usize),
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    M: HistogramKernel<
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
        >,
    O: HistogramOffset,
    Output: KernelSliceArgMut<Elem = C>,
    Queue: KernelSliceArgMut<Elem = i32>,
{
    let geometry = histogram_geometry::<O, THREADS, PIXELS>(
        resolved,
        prepared.blocks_per_row,
        prepared.blocks_per_col,
    )?;
    let blocks = prepared
        .blocks_per_row
        .checked_mul(prepared.blocks_per_col)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    if blocks == 0 {
        HistogramInitKernel::launch_histogram_init(module, stream, prepared.storage)
    } else {
        module.launch_histogram_range_arithmetic(
            stream,
            samples,
            levels,
            prepared.storage,
            geometry,
        )
    }
}

/// Arithmetic-range variant of [`device_histogram_range`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_histogram_range_arithmetic<
    S,
    C,
    L,
    Levels,
    M,
    Output,
    Queue,
    const CHANNELS: usize,
    const ACTIVE: usize,
    const THREADS: u32,
    const PIXELS: usize,
    const RLE: bool,
    const BYTE_SAMPLE: bool,
    const SHARED_BYTES: usize,
    const GLOBAL_BYTES: usize,
>(
    stream: &CudaStream,
    module: &M,
    samples: &ArithmeticRange<S>,
    region: HistogramRegion,
    output: &mut DeviceHistogram<C, ACTIVE, Output>,
    levels: &Levels,
    workspace: &mut DeviceHistogramWorkspace<C, ACTIVE, Output, Queue>,
) -> Result<(), DeviceAlgorithmError>
where
    S: HistogramScalar
        + HistogramSample<L>
        + HistogramFourSampleLoad
        + TryFrom<usize>
        + core::ops::Add<Output = S>
        + core::ops::Mul<Output = S>,
    C: HistogramScalar + rccl_memory::device::HistogramCounter,
    L: HistogramScalar + PartialOrd,
    Levels: HistogramLevels<L, ACTIVE> + ?Sized,
    M: HistogramPolicyKernels<
            S,
            C,
            L,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            SHARED_BYTES,
            GLOBAL_BYTES,
        >,
    Output: DeviceRange + KernelSliceArgMut<Elem = C>,
    Queue: KernelSliceArgMut<Elem = i32>,
{
    let resolved = region.__resolve::<CHANNELS>(samples.len())?;
    let output_lengths = *output.lengths();
    let levels = validate_histogram_levels(levels, output_lengths)?;
    let narrow_offsets = histogram_uses_i32_offsets::<S>(resolved.1, resolved.2);
    let prepared = prepare_histogram_launch(output, workspace, resolved, BYTE_SAMPLE)?;
    if prepared.narrow_offsets != narrow_offsets {
        return Err(DeviceAlgorithmError::InvalidHistogramShape(
            "histogram workspace does not match the image offset domain",
        ));
    }
    let max_bins = prepared
        .storage
        .output_lengths
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if BYTE_SAMPLE || max_bins <= 256 {
        if narrow_offsets {
            launch_histogram_range_arithmetic_variant::<
                S,
                C,
                L,
                M,
                i32,
                Output,
                Queue,
                CHANNELS,
                ACTIVE,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, samples, levels, prepared, resolved)
        } else {
            launch_histogram_range_arithmetic_variant::<
                S,
                C,
                L,
                M,
                usize,
                Output,
                Queue,
                CHANNELS,
                ACTIVE,
                THREADS,
                PIXELS,
                RLE,
                BYTE_SAMPLE,
                true,
                SHARED_BYTES,
            >(stream, module, samples, levels, prepared, resolved)
        }
    } else if narrow_offsets {
        launch_histogram_range_arithmetic_variant::<
            S,
            C,
            L,
            M,
            i32,
            Output,
            Queue,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, samples, levels, prepared, resolved)
    } else {
        launch_histogram_range_arithmetic_variant::<
            S,
            C,
            L,
            M,
            usize,
            Output,
            Queue,
            CHANNELS,
            ACTIVE,
            THREADS,
            PIXELS,
            RLE,
            BYTE_SAMPLE,
            false,
            GLOBAL_BYTES,
        >(stream, module, samples, levels, prepared, resolved)
    }
}

fn channel_layout<const CHANNELS: usize>(
    lengths: [usize; CHANNELS],
) -> Result<([usize; CHANNELS], usize), DeviceAlgorithmError> {
    let mut offsets = [0usize; CHANNELS];
    let mut total = 0usize;
    for channel in 0..CHANNELS {
        offsets[channel] = total;
        total = total
            .checked_add(lengths[channel])
            .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    }
    Ok((offsets, total))
}

fn workspace_channel_layout<const CHANNELS: usize>(
    lengths: [usize; CHANNELS],
    blocks_per_row: usize,
    blocks_per_col: usize,
) -> Result<([usize; CHANNELS], usize), DeviceAlgorithmError> {
    let blocks = blocks_per_row
        .checked_mul(blocks_per_col)
        .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    let mut allocation_lengths = [0usize; CHANNELS];
    for channel in 0..CHANNELS {
        allocation_lengths[channel] = lengths[channel]
            .checked_mul(blocks)
            .ok_or(DeviceAlgorithmError::SizeOverflow)?;
    }
    channel_layout(allocation_lengths)
}

#[cfg(test)]
mod tests {
    use super::{
        HistogramPolicy, HistogramRegion, histogram_policy, histogram_shared_bytes,
        histogram_uses_i32_offsets,
    };
    use crate::DeviceAlgorithmError;

    #[test]
    fn complete_region_requires_whole_interleaved_pixels() {
        assert_eq!(HistogramRegion::All.__resolve::<4>(12).unwrap(), (3, 1, 12));
        assert!(matches!(
            HistogramRegion::All.__resolve::<4>(11),
            Err(DeviceAlgorithmError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn pitched_region_preserves_padding_but_checks_extent() {
        assert_eq!(
            HistogramRegion::rows(3, 2, 4).__resolve::<1>(8).unwrap(),
            (3, 2, 4)
        );
        assert!(matches!(
            HistogramRegion::rows(3, 2, 2).__resolve::<1>(8),
            Err(DeviceAlgorithmError::InvalidHistogramShape(_))
        ));
        assert!(matches!(
            HistogramRegion::rows(3, 3, 4).__resolve::<1>(10),
            Err(DeviceAlgorithmError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn tightly_packed_rows_collapse_to_cub_linear_geometry() {
        assert_eq!(
            HistogramRegion::rows(3, 4, 12).__resolve::<4>(48).unwrap(),
            (12, 1, 48)
        );
    }

    #[test]
    fn empty_images_are_valid() {
        assert_eq!(
            HistogramRegion::rows(17, 0, 68).__resolve::<4>(0).unwrap(),
            (0, 1, 0)
        );
    }

    #[test]
    fn offset_domain_matches_cub_byte_extent_guard() {
        assert!(histogram_uses_i32_offsets::<u8>(1, 1_000_000));
        assert!(histogram_uses_i32_offsets::<u32>(2, 1_000_000));
        assert!(!histogram_uses_i32_offsets::<u32>(1, i32::MAX as usize));
        assert!(!histogram_uses_i32_offsets::<u8>(usize::MAX, 2));
    }

    #[test]
    #[cfg(not(cuda_arch_ge_90))]
    fn fallback_policy_matches_pinned_policy500_arithmetic() {
        assert_eq!(
            histogram_policy(4, 4, 1, 1, true),
            HistogramPolicy {
                threads: 384,
                pixels: 16,
                rle: true,
            }
        );
        assert_eq!(histogram_policy(8, 8, 3, 4, false).pixels, 2);
        assert_eq!(histogram_shared_bytes(4, 3, true), 3_092);
        assert_eq!(histogram_shared_bytes(4, 3, false), 20);
    }
}
