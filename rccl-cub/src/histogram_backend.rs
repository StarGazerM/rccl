#![allow(
    non_upper_case_globals,
    reason = "unroll macros preserve the lowercase loop-index spelling of CUB source"
)]

use cuda_device::{
    DisjointSlice, read_only, thread,
    vector::{I32x4, as_vector_prefix},
};
use rccl_memory::device::{BlockSharedMemory, DeviceRegion, HistogramCounter, SharedRegion};

const PRIVATIZED_SMEM_BINS: usize = 256;
const PRIVATIZED_SMEM_STRIDE: usize = PRIVATIZED_SMEM_BINS + 1;

macro_rules! histogram_channels {
    ($limit:expr, $channel:ident, $body:block) => {{
        {
            const $channel: usize = 0;
            if $channel < $limit $body
        }
        {
            const $channel: usize = 1;
            if $channel < $limit $body
        }
        {
            const $channel: usize = 2;
            if $channel < $limit $body
        }
        {
            const $channel: usize = 3;
            if $channel < $limit $body
        }
    }};
}

macro_rules! histogram_pixels {
    ($limit:expr, $pixel:ident, $body:block) => {{
        {
            const $pixel: usize = 0;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 1;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 2;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 3;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 4;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 5;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 6;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 7;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 8;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 9;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 10;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 11;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 12;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 13;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 14;
            if $pixel < $limit $body
        }
        {
            const $pixel: usize = 15;
            if $pixel < $limit $body
        }
    }};
}

macro_rules! histogram_quads {
    ($limit:expr, $quad:ident, $body:block) => {{
        {
            const $quad: usize = 0;
            if $quad * 4 < $limit $body
        }
        {
            const $quad: usize = 1;
            if $quad * 4 < $limit $body
        }
        {
            const $quad: usize = 2;
            if $quad * 4 < $limit $body
        }
        {
            const $quad: usize = 3;
            if $quad * 4 < $limit $body
        }
    }};
}

/// Device-side random-access sample source used by the histogram agent.
///
/// The public range expression is lowered to one concrete implementation;
/// there is no runtime iterator tag or erased dispatch.
#[doc(hidden)]
pub trait HistogramRead<S>: Copy {
    type FourReader: HistogramFourReader<S>;
    type ThreadTile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool>:
        HistogramThreadTileRead<S>;

    fn sample_at(self, index: usize) -> Option<S>;

    fn thread_tile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool>(
        self,
        start: usize,
        valid_samples: i32,
    ) -> Option<Self::ThreadTile<PIXELS, CHANNELS, FULL>>;

    /// Form the native typed view used by CUB's aligned `PixelT`/`VecT` load
    /// specialization. The view is formed once before row consumption rather
    /// than reconstructed for each grouped load.
    fn four_reader(self) -> Option<Self::FourReader>;
}

/// Bounds-proved scalar loads for one thread's register tile.
#[doc(hidden)]
pub trait HistogramThreadTileRead<S>: Copy {
    type Pixel: HistogramThreadPixelRead<S>;

    fn pixel<const PIXEL: usize>(&self) -> Option<Self::Pixel>;
}

/// One active pixel from a bounds-proved thread tile.
#[doc(hidden)]
pub trait HistogramThreadPixelRead<S>: Copy {
    fn load<const CHANNEL: usize>(&self) -> S;
}

impl<S: Copy, const CHANNELS: usize> HistogramThreadPixelRead<S>
    for read_only::ThreadPixel<'_, S, CHANNELS>
{
    #[inline(always)]
    fn load<const CHANNEL: usize>(&self) -> S {
        self.load::<CHANNEL>()
    }
}

impl<'a, S: Copy, const PIXELS: usize, const CHANNELS: usize, const FULL: bool>
    HistogramThreadTileRead<S> for read_only::ThreadTile<'a, S, PIXELS, CHANNELS, FULL>
{
    type Pixel = read_only::ThreadPixel<'a, S, CHANNELS>;

    #[inline(always)]
    fn pixel<const PIXEL: usize>(&self) -> Option<Self::Pixel> {
        self.pixel::<PIXEL>()
    }
}

/// Typed reader for CUB's grouped `PixelT`/`VecT` loads.
#[doc(hidden)]
pub trait HistogramFourReader<S>: Copy {
    type Five: HistogramFiveReader<S>;

    fn four_at(self, vector_index: usize) -> Option<[S; 4]>;

    fn four_quads_at(self, vector_index: usize) -> Option<[[S; 4]; 4]>;

    fn five_at(self, vector_index: usize) -> Option<Self::Five>;
}

/// One bounds-proved five-pixel view used by CUB's full multi-channel tile
/// load. Each statically unrolled pixel then performs only its native vector
/// load; the slice extent is checked once for the complete register tile.
#[doc(hidden)]
pub trait HistogramFiveReader<S>: Copy {
    fn four_at<const PIXEL: usize>(self) -> [S; 4];
}

#[derive(Clone, Copy)]
pub struct NoHistogramFourReader<S>(core::marker::PhantomData<S>);

impl<S: Copy> HistogramFourReader<S> for NoHistogramFourReader<S> {
    type Five = Self;

    #[inline(always)]
    fn four_at(self, _vector_index: usize) -> Option<[S; 4]> {
        None
    }

    #[inline(always)]
    fn four_quads_at(self, _vector_index: usize) -> Option<[[S; 4]; 4]> {
        None
    }

    #[inline(always)]
    fn five_at(self, _vector_index: usize) -> Option<Self::Five> {
        None
    }
}

impl<S: Copy> HistogramFiveReader<S> for NoHistogramFourReader<S> {
    #[inline(always)]
    fn four_at<const PIXEL: usize>(self) -> [S; 4] {
        let _ = PIXEL;
        cuda_device::debug::trap()
    }
}

#[derive(Clone, Copy)]
pub struct I32HistogramFourReader<'a> {
    vectors: &'a [I32x4],
}

#[derive(Clone, Copy)]
pub struct I32HistogramFiveReader<'a> {
    vectors: &'a [I32x4; 5],
}

impl HistogramFiveReader<i32> for I32HistogramFiveReader<'_> {
    #[inline(always)]
    fn four_at<const PIXEL: usize>(self) -> [i32; 4] {
        if PIXEL >= 5 {
            cuda_device::debug::trap();
        }
        read_only::load(&self.vectors[PIXEL]).to_array()
    }
}

impl<'a> HistogramFourReader<i32> for I32HistogramFourReader<'a> {
    type Five = I32HistogramFiveReader<'a>;

    #[inline(always)]
    fn four_at(self, vector_index: usize) -> Option<[i32; 4]> {
        Some(read_only::load(self.vectors.get(vector_index)?).to_array())
    }

    #[inline(always)]
    fn four_quads_at(self, vector_index: usize) -> Option<[[i32; 4]; 4]> {
        let end = vector_index.checked_add(4)?;
        let vectors: &[I32x4; 4] = self.vectors.get(vector_index..end)?.try_into().ok()?;
        Some([
            read_only::load(&vectors[0]).to_array(),
            read_only::load(&vectors[1]).to_array(),
            read_only::load(&vectors[2]).to_array(),
            read_only::load(&vectors[3]).to_array(),
        ])
    }

    #[inline(always)]
    fn five_at(self, vector_index: usize) -> Option<Self::Five> {
        let end = vector_index.checked_add(5)?;
        let vectors = self.vectors.get(vector_index..end)?.try_into().ok()?;
        Some(I32HistogramFiveReader { vectors })
    }
}

#[derive(Clone, Copy)]
struct ScalarHistogramRead<Samples>(Samples);

#[derive(Clone, Copy)]
struct GroupedHistogramRead<Samples, Reader> {
    samples: Samples,
    reader: Reader,
}

trait HistogramTileRead<S>: Copy {
    type Five: HistogramFiveReader<S>;
    type ThreadTile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool>:
        HistogramThreadTileRead<S>;

    const GROUPED: bool;

    fn thread_tile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool>(
        self,
        start: usize,
        valid_samples: i32,
    ) -> Option<Self::ThreadTile<PIXELS, CHANNELS, FULL>>;
    fn four_at(self, vector_index: usize) -> Option<[S; 4]>;
    fn four_quads_at(self, vector_index: usize) -> Option<[[S; 4]; 4]>;
    fn five_at(self, vector_index: usize) -> Option<Self::Five>;
}

impl<Samples, S> HistogramTileRead<S> for ScalarHistogramRead<Samples>
where
    Samples: HistogramRead<S>,
    S: Copy,
{
    type Five = NoHistogramFourReader<S>;
    type ThreadTile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool> =
        Samples::ThreadTile<PIXELS, CHANNELS, FULL>;

    const GROUPED: bool = false;

    #[inline(always)]
    fn thread_tile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool>(
        self,
        start: usize,
        valid_samples: i32,
    ) -> Option<Self::ThreadTile<PIXELS, CHANNELS, FULL>> {
        self.0
            .thread_tile::<PIXELS, CHANNELS, FULL>(start, valid_samples)
    }

    #[inline(always)]
    fn four_at(self, _vector_index: usize) -> Option<[S; 4]> {
        None
    }

    #[inline(always)]
    fn four_quads_at(self, _vector_index: usize) -> Option<[[S; 4]; 4]> {
        None
    }

    #[inline(always)]
    fn five_at(self, _vector_index: usize) -> Option<Self::Five> {
        None
    }
}

impl<Samples, Reader, S> HistogramTileRead<S> for GroupedHistogramRead<Samples, Reader>
where
    Samples: HistogramRead<S>,
    Reader: HistogramFourReader<S>,
{
    type Five = Reader::Five;
    type ThreadTile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool> =
        Samples::ThreadTile<PIXELS, CHANNELS, FULL>;

    const GROUPED: bool = true;

    #[inline(always)]
    fn thread_tile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool>(
        self,
        start: usize,
        valid_samples: i32,
    ) -> Option<Self::ThreadTile<PIXELS, CHANNELS, FULL>> {
        self.samples
            .thread_tile::<PIXELS, CHANNELS, FULL>(start, valid_samples)
    }

    #[inline(always)]
    fn four_at(self, vector_index: usize) -> Option<[S; 4]> {
        self.reader.four_at(vector_index)
    }

    #[inline(always)]
    fn four_quads_at(self, vector_index: usize) -> Option<[[S; 4]; 4]> {
        self.reader.four_quads_at(vector_index)
    }

    #[inline(always)]
    fn five_at(self, vector_index: usize) -> Option<Self::Five> {
        self.reader.five_at(vector_index)
    }
}

/// Primitive-specific realization of CUB's `CubVector<SampleT, 4>` load.
///
/// This is a Rust type-class boundary, not runtime type dispatch. Each sample
/// monomorphization selects its own grouped representation, while non-native
/// sources never enter the aligned path.
#[doc(hidden)]
pub trait HistogramFourSampleLoad: Copy {
    type Reader<'a>: HistogramFourReader<Self>
    where
        Self: 'a;

    fn four_reader(slice: &[Self]) -> Option<Self::Reader<'_>>;
}

macro_rules! scalar_four_sample_load {
    ($($sample:ty),+ $(,)?) => {
        $(
            impl HistogramFourSampleLoad for $sample {
                type Reader<'a> = NoHistogramFourReader<Self>;

                #[inline(always)]
                fn four_reader(_slice: &[Self]) -> Option<Self::Reader<'_>> {
                    None
                }
            }
        )+
    };
}

scalar_four_sample_load!(i8, u8, i16, u16, u32, i64, u64, f32, f64);

impl HistogramFourSampleLoad for i32 {
    type Reader<'a> = I32HistogramFourReader<'a>;

    #[inline(always)]
    fn four_reader(slice: &[Self]) -> Option<Self::Reader<'_>> {
        let (vectors, _) = as_vector_prefix::<I32x4>(slice)?;
        Some(I32HistogramFourReader { vectors })
    }
}

#[inline(always)]
fn rows_use_grouped_loads(
    channels: usize,
    pixels_per_thread: usize,
    has_native_reader: bool,
    num_rows: usize,
    row_stride_samples: usize,
) -> bool {
    has_native_reader
        && if channels == 1 {
            pixels_per_thread.is_multiple_of(4)
                && (num_rows == 1 || row_stride_samples.is_multiple_of(4))
        } else {
            channels == 4 && row_stride_samples.is_multiple_of(4)
        }
}

/// Signed or pointer-sized image-offset arithmetic selected by the histogram
/// host dispatch.
///
/// The pinned CUB entry point instantiates its agent with `int` whenever the
/// complete pitched image fits that domain, and retains the caller's wide
/// offset type otherwise.  This trait is the Rust generic equivalent of that
/// `OffsetT`; it is monomorphized into distinct kernels rather than erased or
/// selected inside the device code.
#[doc(hidden)]
pub trait HistogramOffset: Copy + PartialOrd {
    fn from_i32(value: i32) -> Self;
    fn from_usize(value: usize) -> Self;
    fn as_usize(self) -> usize;
    fn as_i32(self) -> i32;
    fn add(self, right: Self) -> Self;
    fn sub(self, right: Self) -> Self;
    fn mul(self, right: Self) -> Self;
}

impl HistogramOffset for i32 {
    #[inline(always)]
    fn from_i32(value: i32) -> Self {
        value
    }

    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value as Self
    }

    #[inline(always)]
    fn as_usize(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn as_i32(self) -> i32 {
        self
    }

    #[inline(always)]
    fn add(self, right: Self) -> Self {
        self.wrapping_add(right)
    }

    #[inline(always)]
    fn sub(self, right: Self) -> Self {
        self.wrapping_sub(right)
    }

    #[inline(always)]
    fn mul(self, right: Self) -> Self {
        self.wrapping_mul(right)
    }
}

impl HistogramOffset for usize {
    #[inline(always)]
    fn from_i32(value: i32) -> Self {
        value as Self
    }

    #[inline(always)]
    fn from_usize(value: usize) -> Self {
        value
    }

    #[inline(always)]
    fn as_usize(self) -> usize {
        self
    }

    #[inline(always)]
    fn as_i32(self) -> i32 {
        self as i32
    }

    #[inline(always)]
    fn add(self, right: Self) -> Self {
        self.wrapping_add(right)
    }

    #[inline(always)]
    fn sub(self, right: Self) -> Self {
        self.wrapping_sub(right)
    }

    #[inline(always)]
    fn mul(self, right: Self) -> Self {
        self.wrapping_mul(right)
    }
}

impl<'a, S: HistogramFourSampleLoad + 'a> HistogramRead<S> for &'a [S] {
    type FourReader = S::Reader<'a>;
    type ThreadTile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool> =
        read_only::ThreadTile<'a, S, PIXELS, CHANNELS, FULL>;

    #[inline(always)]
    fn sample_at(self, index: usize) -> Option<S> {
        let sample = self.get(index)?;
        Some(read_only::load(sample))
    }

    #[inline(always)]
    fn thread_tile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool>(
        self,
        start: usize,
        valid_samples: i32,
    ) -> Option<Self::ThreadTile<PIXELS, CHANNELS, FULL>> {
        read_only::ThreadTile::new(self, start, valid_samples)
    }

    #[inline(always)]
    fn four_reader(self) -> Option<Self::FourReader> {
        S::four_reader(self)
    }
}

/// Device lowering of [`crate::ArithmeticRange`].
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct ArithmeticHistogramRange<S> {
    start: S,
    step: S,
    len: usize,
}

#[derive(Clone, Copy)]
pub struct ArithmeticHistogramThreadTile<
    S,
    const PIXELS: usize,
    const CHANNELS: usize,
    const FULL: bool,
> {
    samples: ArithmeticHistogramRange<S>,
    start: usize,
    valid_samples: usize,
}

#[derive(Clone, Copy)]
pub struct ArithmeticHistogramThreadPixel<S, const CHANNELS: usize> {
    samples: ArithmeticHistogramRange<S>,
    start: usize,
}

impl<S, const CHANNELS: usize> HistogramThreadPixelRead<S>
    for ArithmeticHistogramThreadPixel<S, CHANNELS>
where
    S: Copy + TryFrom<usize> + core::ops::Add<Output = S> + core::ops::Mul<Output = S>,
{
    #[inline(always)]
    fn load<const CHANNEL: usize>(&self) -> S {
        if CHANNEL >= CHANNELS {
            cuda_device::debug::trap();
        }
        let Some(index) = self.start.checked_add(CHANNEL) else {
            cuda_device::debug::trap();
        };
        let Ok(offset) = S::try_from(index) else {
            cuda_device::debug::trap();
        };
        self.samples.start + self.samples.step * offset
    }
}

impl<S, const PIXELS: usize, const CHANNELS: usize, const FULL: bool> HistogramThreadTileRead<S>
    for ArithmeticHistogramThreadTile<S, PIXELS, CHANNELS, FULL>
where
    S: Copy + TryFrom<usize> + core::ops::Add<Output = S> + core::ops::Mul<Output = S>,
{
    type Pixel = ArithmeticHistogramThreadPixel<S, CHANNELS>;

    #[inline(always)]
    fn pixel<const PIXEL: usize>(&self) -> Option<Self::Pixel> {
        if PIXEL >= PIXELS {
            return None;
        }
        let local = PIXEL * CHANNELS;
        if !FULL && local + CHANNELS > self.valid_samples {
            return None;
        }
        Some(ArithmeticHistogramThreadPixel {
            samples: self.samples,
            start: self.start + local,
        })
    }
}

impl<S> ArithmeticHistogramRange<S> {
    #[inline(always)]
    pub fn new(start: S, step: S, len: usize) -> Self {
        Self { start, step, len }
    }
}

impl<S> HistogramRead<S> for ArithmeticHistogramRange<S>
where
    S: Copy + TryFrom<usize> + core::ops::Add<Output = S> + core::ops::Mul<Output = S>,
{
    type FourReader = NoHistogramFourReader<S>;
    type ThreadTile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool> =
        ArithmeticHistogramThreadTile<S, PIXELS, CHANNELS, FULL>;

    #[inline(always)]
    fn sample_at(self, index: usize) -> Option<S> {
        if index >= self.len {
            return None;
        }
        let offset = S::try_from(index).ok()?;
        Some(self.start + self.step * offset)
    }

    #[inline(always)]
    fn thread_tile<const PIXELS: usize, const CHANNELS: usize, const FULL: bool>(
        self,
        start: usize,
        valid_samples: i32,
    ) -> Option<Self::ThreadTile<PIXELS, CHANNELS, FULL>> {
        let capacity = PIXELS.checked_mul(CHANNELS)?;
        if capacity == 0 || capacity > i32::MAX as usize || valid_samples < 0 {
            return None;
        }
        let valid_samples = valid_samples as usize;
        if valid_samples > capacity
            || !valid_samples.is_multiple_of(CHANNELS)
            || (FULL && valid_samples != capacity)
        {
            return None;
        }
        if valid_samples == 0 {
            return Some(ArithmeticHistogramThreadTile {
                samples: self,
                start: 0,
                valid_samples,
            });
        }
        if start >= self.len || valid_samples > self.len - start {
            return None;
        }
        Some(ArithmeticHistogramThreadTile {
            samples: self,
            start,
            valid_samples,
        })
    }

    #[inline(always)]
    fn four_reader(self) -> Option<Self::FourReader> {
        None
    }
}

/// Primitive sample arithmetic used by CUB's histogram transforms.
///
/// This is an ordinary Rust type-class boundary: the associated implementation
/// resolves from `S` and `L`, while the operation macro is reserved for target
/// policy and launch-schedule selection.
#[doc(hidden)]
pub trait HistogramSample<L>: Copy + Default {
    type EvenTransform: HistogramDecode<Self>;

    fn even_transform(lower: L, upper: L, bins: i32) -> Self::EvenTransform;

    #[inline(always)]
    fn even_bin(self, lower: L, upper: L, bins: i32) -> i32 {
        Self::even_transform(lower, upper, bins).bin(self)
    }

    fn even_may_overflow(lower: L, upper: L, bins: usize) -> bool;
    fn byte_bin(self) -> i32;
    fn from_byte_bin(bin: usize) -> Self;
    fn to_level(self) -> L;
}

/// One initialized CUB histogram decode transform.
#[doc(hidden)]
pub trait HistogramDecode<S>: Copy {
    fn bin(self, sample: S) -> i32;
}

/// Compile-time channel selection for CUB's privatized decode-op array.
#[doc(hidden)]
pub trait HistogramSampleDecode<S>: Copy {
    fn bin<const CHANNEL: usize>(self, sample: S) -> i32;
}

/// Compile-time channel selection for CUB's output decode-op array.
#[doc(hidden)]
pub trait HistogramOutputDecode: Copy {
    fn bin<const CHANNEL: usize>(self, privatized_bin: usize) -> i32;
}

#[doc(hidden)]
pub trait HistogramWide: Copy {
    type NonZero: Copy;

    fn nonzero(value: Self) -> Option<Self::NonZero>;
    fn divide(numerator: Self, divisor: Self::NonZero) -> Self;
}

#[doc(hidden)]
pub trait HistogramScaleCommon<Wide: HistogramWide>: Copy {
    type NonZeroRange: Copy;

    fn nonzero_range(max: Self, min: Self) -> Option<Self::NonZeroRange>;
    fn widen_range(range: Self::NonZeroRange) -> Wide::NonZero;
}

macro_rules! histogram_scale_common {
    ($common:ty, $unsigned:ty, $nonzero:ty, $wide:ty, $wide_nonzero:ty) => {
        impl HistogramScaleCommon<$wide> for $common {
            type NonZeroRange = $nonzero;

            #[inline(always)]
            fn nonzero_range(max: Self, min: Self) -> Option<Self::NonZeroRange> {
                <$nonzero>::new(max.wrapping_sub(min) as $unsigned)
            }

            #[inline(always)]
            fn widen_range(range: Self::NonZeroRange) -> <$wide as HistogramWide>::NonZero {
                <$wide_nonzero>::from(range)
            }
        }
    };
}

histogram_scale_common!(i8, u8, core::num::NonZeroU8, u32, core::num::NonZeroU32);
histogram_scale_common!(u8, u8, core::num::NonZeroU8, u32, core::num::NonZeroU32);
histogram_scale_common!(i16, u16, core::num::NonZeroU16, u32, core::num::NonZeroU32);
histogram_scale_common!(u16, u16, core::num::NonZeroU16, u32, core::num::NonZeroU32);
histogram_scale_common!(i32, u32, core::num::NonZeroU32, u64, core::num::NonZeroU64);
histogram_scale_common!(u32, u32, core::num::NonZeroU32, u64, core::num::NonZeroU64);
histogram_scale_common!(i64, u64, core::num::NonZeroU64, u64, core::num::NonZeroU64);
histogram_scale_common!(u64, u64, core::num::NonZeroU64, u64, core::num::NonZeroU64);

impl HistogramWide for u32 {
    type NonZero = core::num::NonZeroU32;

    #[inline(always)]
    fn nonzero(value: Self) -> Option<Self::NonZero> {
        core::num::NonZeroU32::new(value)
    }

    #[inline(always)]
    fn divide(numerator: Self, divisor: Self::NonZero) -> Self {
        numerator / divisor
    }
}

impl HistogramWide for u64 {
    type NonZero = core::num::NonZeroU64;

    #[inline(always)]
    fn nonzero(value: Self) -> Option<Self::NonZero> {
        core::num::NonZeroU64::new(value)
    }

    #[inline(always)]
    fn divide(numerator: Self, divisor: Self::NonZero) -> Self {
        numerator / divisor
    }
}

/// Integral realization of CUB's `ScaleTransform`.
///
/// `range` carries Rust's non-zero proof, so the sample hot path has the same
/// unchecked division semantics as CUB after public argument validation,
/// while retaining a safe generated-kernel interface.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct IntegralHistogramScale<S, L, Common, Wide>
where
    Wide: HistogramWide,
    Common: HistogramScaleCommon<Wide>,
{
    max: Common,
    min: Common,
    bins: Common,
    range: Common::NonZeroRange,
    marker: core::marker::PhantomData<(S, L)>,
}

/// Floating-point realization of CUB's reciprocal `ScaleTransform`.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct FloatingHistogramScale<S, L, Common> {
    max: Common,
    min: Common,
    reciprocal: Common,
    marker: core::marker::PhantomData<(S, L)>,
}

/// One channel's initialized CUB `SearchTransform`.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct RangeHistogramTransform<'a, L> {
    levels: read_only::ReadOnlyRange<'a, L>,
}

impl<'a, L> RangeHistogramTransform<'a, L> {
    #[inline(always)]
    pub fn new(levels: read_only::ReadOnlyRange<'a, L>) -> Self {
        Self { levels }
    }
}

impl<L: Copy + PartialOrd> HistogramDecode<L> for RangeHistogramTransform<'_, L> {
    #[inline(always)]
    fn bin(self, sample: L) -> i32 {
        let bin = self.levels.upper_bound(sample).wrapping_sub(1);
        if bin >= self.levels.len().wrapping_sub(1) {
            -1
        } else {
            bin
        }
    }
}

/// Host-initialized CUB `ScaleTransform` array used by both decode stages.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct EvenHistogramTransforms<
    S: HistogramSample<L>,
    L,
    const ACTIVE: usize,
    const BYTE_SAMPLE: bool,
> {
    transforms: [S::EvenTransform; ACTIVE],
    marker: core::marker::PhantomData<L>,
}

impl<S: HistogramSample<L>, L, const ACTIVE: usize, const BYTE_SAMPLE: bool>
    EvenHistogramTransforms<S, L, ACTIVE, BYTE_SAMPLE>
{
    #[inline(always)]
    pub fn new(transforms: [S::EvenTransform; ACTIVE]) -> Self {
        Self {
            transforms,
            marker: core::marker::PhantomData,
        }
    }
}

impl<S, L, const ACTIVE: usize, const BYTE_SAMPLE: bool> HistogramSampleDecode<S>
    for EvenHistogramTransforms<S, L, ACTIVE, BYTE_SAMPLE>
where
    S: HistogramSample<L>,
    L: Copy,
{
    #[inline(always)]
    fn bin<const CHANNEL: usize>(self, sample: S) -> i32 {
        if BYTE_SAMPLE {
            sample.byte_bin()
        } else {
            let Some(transform) = self.transforms.get(CHANNEL) else {
                cuda_device::debug::trap();
            };
            transform.bin(sample)
        }
    }
}

impl<S, L, const ACTIVE: usize, const BYTE_SAMPLE: bool> HistogramOutputDecode
    for EvenHistogramTransforms<S, L, ACTIVE, BYTE_SAMPLE>
where
    S: HistogramSample<L>,
    L: Copy,
{
    #[inline(always)]
    fn bin<const CHANNEL: usize>(self, privatized_bin: usize) -> i32 {
        if BYTE_SAMPLE {
            let Some(transform) = self.transforms.get(CHANNEL) else {
                cuda_device::debug::trap();
            };
            transform.bin(S::from_byte_bin(privatized_bin))
        } else {
            privatized_bin as i32
        }
    }
}

/// Host-initialized CUB `SearchTransform` array used by both decode stages.
#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct RangeHistogramTransforms<S, L, const ACTIVE: usize, const BYTE_SAMPLE: bool> {
    levels: read_only::ReadOnlySlices<L, ACTIVE>,
    marker: core::marker::PhantomData<S>,
}

impl<S, L, const ACTIVE: usize, const BYTE_SAMPLE: bool>
    RangeHistogramTransforms<S, L, ACTIVE, BYTE_SAMPLE>
{
    #[inline(always)]
    pub fn new(levels: read_only::ReadOnlySlices<L, ACTIVE>) -> Self {
        Self {
            levels,
            marker: core::marker::PhantomData,
        }
    }
}

impl<S, L, const ACTIVE: usize, const BYTE_SAMPLE: bool> HistogramSampleDecode<S>
    for RangeHistogramTransforms<S, L, ACTIVE, BYTE_SAMPLE>
where
    S: HistogramSample<L>,
    L: Copy + PartialOrd,
{
    #[inline(always)]
    fn bin<const CHANNEL: usize>(self, sample: S) -> i32 {
        if BYTE_SAMPLE {
            sample.byte_bin()
        } else {
            let levels = self.levels.get_const::<CHANNEL>();
            RangeHistogramTransform::new(levels).bin(sample.to_level())
        }
    }
}

impl<S, L, const ACTIVE: usize, const BYTE_SAMPLE: bool> HistogramOutputDecode
    for RangeHistogramTransforms<S, L, ACTIVE, BYTE_SAMPLE>
where
    S: HistogramSample<L>,
    L: Copy + PartialOrd,
{
    #[inline(always)]
    fn bin<const CHANNEL: usize>(self, privatized_bin: usize) -> i32 {
        if BYTE_SAMPLE {
            let levels = self.levels.get_const::<CHANNEL>();
            RangeHistogramTransform::new(levels).bin(S::from_byte_bin(privatized_bin).to_level())
        } else {
            privatized_bin as i32
        }
    }
}

macro_rules! integral_histogram_sample {
    ($sample:ty, $level:ty, $common:ty, $wide:ty, signed) => {
        impl HistogramDecode<$sample> for IntegralHistogramScale<$sample, $level, $common, $wide> {
            #[inline(always)]
            fn bin(self, sample: $sample) -> i32 {
                let sample = sample as $common;
                if sample >= self.min && sample < self.max {
                    <$wide as HistogramWide>::divide(
                        (sample.wrapping_sub(self.min) as $wide).wrapping_mul(self.bins as $wide),
                        <$common as HistogramScaleCommon<$wide>>::widen_range(self.range),
                    ) as i32
                } else {
                    -1
                }
            }
        }

        impl HistogramSample<$level> for $sample {
            type EvenTransform = IntegralHistogramScale<Self, $level, $common, $wide>;

            #[inline(always)]
            fn even_transform(lower: $level, upper: $level, bins: i32) -> Self::EvenTransform {
                let min = lower as $common;
                let max = upper as $common;
                let Some(range) = <$common as HistogramScaleCommon<$wide>>::nonzero_range(max, min)
                else {
                    cuda_device::debug::trap();
                };
                IntegralHistogramScale {
                    max,
                    min,
                    bins: bins as $common,
                    range,
                    marker: core::marker::PhantomData,
                }
            }

            #[inline(always)]
            fn even_may_overflow(lower: $level, upper: $level, bins: usize) -> bool {
                let lower = lower as $common as i128;
                let upper = upper as $common as i128;
                lower >= upper
                    || bins == 0
                    || bins as u128 > <$common>::MAX as u128
                    || (upper - lower) as u128 > <$wide>::MAX as u128 / bins as u128
            }

            #[inline(always)]
            fn byte_bin(self) -> i32 {
                self as i32
            }

            #[inline(always)]
            fn from_byte_bin(bin: usize) -> Self {
                bin as Self
            }

            #[inline(always)]
            fn to_level(self) -> $level {
                self as $level
            }
        }
    };
    ($sample:ty, $level:ty, $common:ty, $wide:ty, unsigned) => {
        impl HistogramDecode<$sample> for IntegralHistogramScale<$sample, $level, $common, $wide> {
            #[inline(always)]
            fn bin(self, sample: $sample) -> i32 {
                let sample = sample as $common;
                if sample >= self.min && sample < self.max {
                    <$wide as HistogramWide>::divide(
                        (sample.wrapping_sub(self.min) as $wide).wrapping_mul(self.bins as $wide),
                        <$common as HistogramScaleCommon<$wide>>::widen_range(self.range),
                    ) as i32
                } else {
                    -1
                }
            }
        }

        impl HistogramSample<$level> for $sample {
            type EvenTransform = IntegralHistogramScale<Self, $level, $common, $wide>;

            #[inline(always)]
            fn even_transform(lower: $level, upper: $level, bins: i32) -> Self::EvenTransform {
                let min = lower as $common;
                let max = upper as $common;
                let Some(range) = <$common as HistogramScaleCommon<$wide>>::nonzero_range(max, min)
                else {
                    cuda_device::debug::trap();
                };
                IntegralHistogramScale {
                    max,
                    min,
                    bins: bins as $common,
                    range,
                    marker: core::marker::PhantomData,
                }
            }

            #[inline(always)]
            fn even_may_overflow(lower: $level, upper: $level, bins: usize) -> bool {
                let lower = lower as $common as u128;
                let upper = upper as $common as u128;
                lower >= upper
                    || bins == 0
                    || bins as u128 > <$common>::MAX as u128
                    || upper - lower > <$wide>::MAX as u128 / bins as u128
            }

            #[inline(always)]
            fn byte_bin(self) -> i32 {
                self as i32
            }

            #[inline(always)]
            fn from_byte_bin(bin: usize) -> Self {
                bin as Self
            }

            #[inline(always)]
            fn to_level(self) -> $level {
                self as $level
            }
        }
    };
}

macro_rules! floating_histogram_sample {
    ($sample:ty, $level:ty, $common:ty) => {
        impl HistogramDecode<$sample> for FloatingHistogramScale<$sample, $level, $common> {
            #[inline(always)]
            fn bin(self, sample: $sample) -> i32 {
                let sample = sample as $common;
                if sample >= self.min && sample < self.max {
                    ((sample - self.min) * self.reciprocal) as i32
                } else {
                    -1
                }
            }
        }

        impl HistogramSample<$level> for $sample {
            type EvenTransform = FloatingHistogramScale<Self, $level, $common>;

            #[inline(always)]
            fn even_transform(lower: $level, upper: $level, bins: i32) -> Self::EvenTransform {
                let min = lower as $common;
                let max = upper as $common;
                FloatingHistogramScale {
                    max,
                    min,
                    reciprocal: bins as $common / (max - min),
                    marker: core::marker::PhantomData,
                }
            }

            #[inline(always)]
            fn even_may_overflow(_lower: $level, _upper: $level, _bins: usize) -> bool {
                false
            }

            #[inline(always)]
            fn byte_bin(self) -> i32 {
                self as i32
            }

            #[inline(always)]
            fn from_byte_bin(bin: usize) -> Self {
                bin as Self
            }

            #[inline(always)]
            fn to_level(self) -> $level {
                self as $level
            }
        }
    };
}

integral_histogram_sample!(i8, i8, i8, u32, signed);
integral_histogram_sample!(i8, u8, i32, u64, signed);
integral_histogram_sample!(i8, i16, i32, u64, signed);
integral_histogram_sample!(i8, u16, i32, u64, signed);
integral_histogram_sample!(i8, i32, i32, u64, signed);
integral_histogram_sample!(i8, u32, u32, u64, unsigned);
integral_histogram_sample!(i8, i64, i64, u64, signed);
integral_histogram_sample!(i8, u64, u64, u64, unsigned);

integral_histogram_sample!(u8, i8, i32, u64, signed);
integral_histogram_sample!(u8, u8, u8, u32, unsigned);
integral_histogram_sample!(u8, i16, i32, u64, signed);
integral_histogram_sample!(u8, u16, i32, u64, signed);
integral_histogram_sample!(u8, i32, i32, u64, signed);
integral_histogram_sample!(u8, u32, u32, u64, unsigned);
integral_histogram_sample!(u8, i64, i64, u64, signed);
integral_histogram_sample!(u8, u64, u64, u64, unsigned);

integral_histogram_sample!(i16, i8, i32, u64, signed);
integral_histogram_sample!(i16, u8, i32, u64, signed);
integral_histogram_sample!(i16, i16, i16, u32, signed);
integral_histogram_sample!(i16, u16, i32, u64, signed);
integral_histogram_sample!(i16, i32, i32, u64, signed);
integral_histogram_sample!(i16, u32, u32, u64, unsigned);
integral_histogram_sample!(i16, i64, i64, u64, signed);
integral_histogram_sample!(i16, u64, u64, u64, unsigned);

integral_histogram_sample!(u16, i8, i32, u64, signed);
integral_histogram_sample!(u16, u8, i32, u64, signed);
integral_histogram_sample!(u16, i16, i32, u64, signed);
integral_histogram_sample!(u16, u16, u16, u32, unsigned);
integral_histogram_sample!(u16, i32, i32, u64, signed);
integral_histogram_sample!(u16, u32, u32, u64, unsigned);
integral_histogram_sample!(u16, i64, i64, u64, signed);
integral_histogram_sample!(u16, u64, u64, u64, unsigned);

integral_histogram_sample!(i32, i8, i32, u64, signed);
integral_histogram_sample!(i32, u8, i32, u64, signed);
integral_histogram_sample!(i32, i16, i32, u64, signed);
integral_histogram_sample!(i32, u16, i32, u64, signed);
integral_histogram_sample!(i32, i32, i32, u64, signed);
integral_histogram_sample!(i32, u32, u32, u64, unsigned);
integral_histogram_sample!(i32, i64, i64, u64, signed);
integral_histogram_sample!(i32, u64, u64, u64, unsigned);

integral_histogram_sample!(u32, i8, u32, u64, unsigned);
integral_histogram_sample!(u32, u8, u32, u64, unsigned);
integral_histogram_sample!(u32, i16, u32, u64, unsigned);
integral_histogram_sample!(u32, u16, u32, u64, unsigned);
integral_histogram_sample!(u32, i32, u32, u64, unsigned);
integral_histogram_sample!(u32, u32, u32, u64, unsigned);
integral_histogram_sample!(u32, i64, i64, u64, signed);
integral_histogram_sample!(u32, u64, u64, u64, unsigned);

integral_histogram_sample!(i64, i8, i64, u64, signed);
integral_histogram_sample!(i64, u8, i64, u64, signed);
integral_histogram_sample!(i64, i16, i64, u64, signed);
integral_histogram_sample!(i64, u16, i64, u64, signed);
integral_histogram_sample!(i64, i32, i64, u64, signed);
integral_histogram_sample!(i64, u32, i64, u64, signed);
integral_histogram_sample!(i64, i64, i64, u64, signed);
integral_histogram_sample!(i64, u64, u64, u64, unsigned);

integral_histogram_sample!(u64, i8, u64, u64, unsigned);
integral_histogram_sample!(u64, u8, u64, u64, unsigned);
integral_histogram_sample!(u64, i16, u64, u64, unsigned);
integral_histogram_sample!(u64, u16, u64, u64, unsigned);
integral_histogram_sample!(u64, i32, u64, u64, unsigned);
integral_histogram_sample!(u64, u32, u64, u64, unsigned);
integral_histogram_sample!(u64, i64, u64, u64, unsigned);
integral_histogram_sample!(u64, u64, u64, u64, unsigned);

floating_histogram_sample!(i8, f32, f32);
floating_histogram_sample!(u8, f32, f32);
floating_histogram_sample!(i16, f32, f32);
floating_histogram_sample!(u16, f32, f32);
floating_histogram_sample!(i32, f32, f32);
floating_histogram_sample!(u32, f32, f32);
floating_histogram_sample!(i64, f32, f32);
floating_histogram_sample!(u64, f32, f32);
floating_histogram_sample!(i8, f64, f64);
floating_histogram_sample!(u8, f64, f64);
floating_histogram_sample!(i16, f64, f64);
floating_histogram_sample!(u16, f64, f64);
floating_histogram_sample!(i32, f64, f64);
floating_histogram_sample!(u32, f64, f64);
floating_histogram_sample!(i64, f64, f64);
floating_histogram_sample!(u64, f64, f64);

floating_histogram_sample!(f32, i8, f32);
floating_histogram_sample!(f32, u8, f32);
floating_histogram_sample!(f32, i16, f32);
floating_histogram_sample!(f32, u16, f32);
floating_histogram_sample!(f32, i32, f32);
floating_histogram_sample!(f32, u32, f32);
floating_histogram_sample!(f32, i64, f32);
floating_histogram_sample!(f32, u64, f32);
floating_histogram_sample!(f32, f32, f32);
floating_histogram_sample!(f32, f64, f64);

floating_histogram_sample!(f64, i8, f64);
floating_histogram_sample!(f64, u8, f64);
floating_histogram_sample!(f64, i16, f64);
floating_histogram_sample!(f64, u16, f64);
floating_histogram_sample!(f64, i32, f64);
floating_histogram_sample!(f64, u32, f64);
floating_histogram_sample!(f64, i64, f64);
floating_histogram_sample!(f64, u64, f64);
floating_histogram_sample!(f64, f32, f64);
floating_histogram_sample!(f64, f64, f64);

/// The privatized counter storage selected for one histogram agent.
///
/// This is the Rust counterpart of CUB's `TwoDimSubscriptableCounterT`:
/// shared and global storage are distinct monomorphizations, so neither a
/// nullable view nor a storage-kind branch survives in the accumulation path.
trait PrivatizedHistogram<C: HistogramCounter> {
    fn zero(&mut self, channel: usize, bin: usize);
    fn read(&mut self, channel: usize, bin: usize) -> C;
    fn add(&mut self, channel: usize, bin: usize, value: C);
}

struct SharedPrivatizedHistogram<C> {
    counters: SharedRegion<C>,
}

impl<C: HistogramCounter> PrivatizedHistogram<C> for SharedPrivatizedHistogram<C> {
    #[inline(always)]
    fn zero(&mut self, channel: usize, bin: usize) {
        self.counters
            .write(channel * PRIVATIZED_SMEM_STRIDE + bin, C::zero());
    }

    #[inline(always)]
    fn read(&mut self, channel: usize, bin: usize) -> C {
        self.counters.read(channel * PRIVATIZED_SMEM_STRIDE + bin)
    }

    #[inline(always)]
    fn add(&mut self, channel: usize, bin: usize, value: C) {
        C::fetch_add_block(
            &mut self.counters,
            channel * PRIVATIZED_SMEM_STRIDE + bin,
            value,
        );
    }
}

struct GlobalPrivatizedHistogram<'region, 'storage, C, const ACTIVE_CHANNELS: usize> {
    counters: &'region mut DeviceRegion<'storage, C, thread::Index2D<1>>,
    block_index: usize,
    lengths: [i32; ACTIVE_CHANNELS],
    offsets: [usize; ACTIVE_CHANNELS],
}

impl<C: HistogramCounter, const ACTIVE_CHANNELS: usize> PrivatizedHistogram<C>
    for GlobalPrivatizedHistogram<'_, '_, C, ACTIVE_CHANNELS>
{
    #[inline(always)]
    fn zero(&mut self, channel: usize, bin: usize) {
        self.counters.write(
            self.offsets[channel] + self.block_index * self.lengths[channel] as usize + bin,
            C::zero(),
        );
    }

    #[inline(always)]
    fn read(&mut self, channel: usize, bin: usize) -> C {
        self.counters
            .read(self.offsets[channel] + self.block_index * self.lengths[channel] as usize + bin)
    }

    #[inline(always)]
    fn add(&mut self, channel: usize, bin: usize, value: C) {
        C::fetch_add_device(
            self.counters,
            self.offsets[channel] + self.block_index * self.lengths[channel] as usize + bin,
            value,
        );
    }
}

/// Exact initialization phase of CUB's `DeviceHistogramInitKernel`.
#[inline(always)]
pub fn histogram_init_agent<C, const ACTIVE_CHANNELS: usize>(
    output: DisjointSlice<'_, C>,
    output_lengths: [i32; ACTIVE_CHANNELS],
    output_offsets: [usize; ACTIVE_CHANNELS],
    queue: DisjointSlice<'_, i32>,
) where
    C: HistogramCounter,
{
    let thread_index = thread::threadIdx_x() as usize;
    let output_bin = thread::blockIdx_x() as usize * 256 + thread_index;
    let mut output = DeviceRegion::new(output);
    let mut queue = DeviceRegion::new(queue);

    if thread_index == 0 && thread::blockIdx_x() == 0 {
        queue.write(0, 0);
    }
    histogram_channels!(ACTIVE_CHANNELS, channel, {
        if (output_bin as i32) < output_lengths[channel] {
            let channel_offset = if ACTIVE_CHANNELS == 1 {
                0
            } else {
                output_offsets[channel]
            };
            output.write(channel_offset + output_bin, C::zero());
        }
    });
}

/// Exact SM50-family `AgentHistogram` sweep used by the pinned policy on
/// SM52--SM89, with the same blocked loading, per-thread RLE, privatization,
/// even-share row traversal, and final atomic composition.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the tuple groups are the semantic CUB agent state, not a public calling convention"
)]
pub fn histogram_sweep_agent<
    Samples,
    S,
    C,
    Offset,
    Decode,
    OutputDecode,
    const CHANNELS: usize,
    const ACTIVE_CHANNELS: usize,
    const THREADS: usize,
    const PIXELS_PER_THREAD: usize,
    const RLE_COMPRESS: bool,
    const SHARED_PRIVATIZATION: bool,
>(
    samples: Samples,
    output_storage: DisjointSlice<'_, C, thread::Index2D<1>>,
    privatized_storage: DisjointSlice<'_, C, thread::Index2D<1>>,
    output_offsets: [usize; ACTIVE_CHANNELS],
    privatized_layout: ([i32; ACTIVE_CHANNELS], [usize; ACTIVE_CHANNELS]),
    image: (Offset, Offset, Offset, i32),
    decode: Decode,
    output_decode: OutputDecode,
    shared_memory: BlockSharedMemory,
) where
    Samples: HistogramRead<S>,
    S: Copy + Default,
    C: HistogramCounter,
    Offset: HistogramOffset,
    Decode: HistogramSampleDecode<S>,
    OutputDecode: HistogramOutputDecode,
{
    let (privatized_lengths, privatized_offsets) = privatized_layout;
    let logical_block_x = thread::blockIdx_x() as i32;
    let logical_block_y = thread::blockIdx_y() as i32;
    let blocks_per_row = thread::gridDim_x() as i32;
    let block_storage_index =
        logical_block_y as usize * blocks_per_row as usize + logical_block_x as usize;
    let mut output = DeviceRegion::new(output_storage);
    let mut privatized = DeviceRegion::new(privatized_storage);
    if SHARED_PRIVATIZATION {
        let mut private = SharedPrivatizedHistogram {
            counters: shared_memory.region::<C>(0, ACTIVE_CHANNELS * PRIVATIZED_SMEM_STRIDE),
        };
        histogram_sweep_with_private::<
            Samples,
            S,
            C,
            Offset,
            Decode,
            OutputDecode,
            _,
            CHANNELS,
            ACTIVE_CHANNELS,
            THREADS,
            PIXELS_PER_THREAD,
            RLE_COMPRESS,
        >(
            samples,
            &mut output,
            output_offsets,
            privatized_lengths,
            image,
            decode,
            output_decode,
            &mut private,
        );
    } else {
        let mut private = GlobalPrivatizedHistogram {
            counters: &mut privatized,
            block_index: block_storage_index,
            lengths: privatized_lengths,
            offsets: privatized_offsets,
        };
        histogram_sweep_with_private::<
            Samples,
            S,
            C,
            Offset,
            Decode,
            OutputDecode,
            _,
            CHANNELS,
            ACTIVE_CHANNELS,
            THREADS,
            PIXELS_PER_THREAD,
            RLE_COMPRESS,
        >(
            samples,
            &mut output,
            output_offsets,
            privatized_lengths,
            image,
            decode,
            output_decode,
            &mut private,
        );
    }
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "private helper keeps CUB's typed privatized-counter specialization explicit"
)]
fn histogram_sweep_with_private<
    Samples,
    S,
    C,
    Offset,
    Decode,
    OutputDecode,
    Private,
    const CHANNELS: usize,
    const ACTIVE_CHANNELS: usize,
    const THREADS: usize,
    const PIXELS_PER_THREAD: usize,
    const RLE_COMPRESS: bool,
>(
    samples: Samples,
    output: &mut DeviceRegion<'_, C, thread::Index2D<1>>,
    output_offsets: [usize; ACTIVE_CHANNELS],
    privatized_lengths: [i32; ACTIVE_CHANNELS],
    image: (Offset, Offset, Offset, i32),
    decode: Decode,
    output_decode: OutputDecode,
    private: &mut Private,
) where
    Samples: HistogramRead<S>,
    S: Copy + Default,
    C: HistogramCounter,
    Offset: HistogramOffset,
    Decode: HistogramSampleDecode<S>,
    OutputDecode: HistogramOutputDecode,
    Private: PrivatizedHistogram<C>,
{
    let (num_row_pixels, num_rows, row_stride_samples, _tiles_per_row) = image;
    let thread_index = thread::threadIdx_x() as usize;
    let logical_block_x = thread::blockIdx_x() as i32;
    let logical_block_y = thread::blockIdx_y() as i32;
    let blocks_per_row = thread::gridDim_x() as i32;
    let blocks_per_col = thread::gridDim_y() as i32;

    histogram_channels!(ACTIVE_CHANNELS, channel, {
        let count = privatized_lengths[channel] as usize;
        let mut bin = thread_index;
        while bin < count {
            private.zero(channel, bin);
            bin += THREADS;
        }
    });
    thread::sync_threads();

    let grouped_load_shape = CHANNELS == 1 || CHANNELS == 4;
    let four_reader = if grouped_load_shape {
        samples.four_reader()
    } else {
        None
    };
    let aligned_rows = rows_use_grouped_loads(
        CHANNELS,
        PIXELS_PER_THREAD,
        four_reader.is_some(),
        num_rows.as_usize(),
        row_stride_samples.as_usize(),
    );
    if let Some(reader) = four_reader
        && aligned_rows
    {
        consume_rows::<
            _,
            S,
            C,
            Offset,
            Decode,
            Private,
            CHANNELS,
            ACTIVE_CHANNELS,
            THREADS,
            PIXELS_PER_THREAD,
            RLE_COMPRESS,
        >(
            GroupedHistogramRead { samples, reader },
            (num_row_pixels, num_rows, row_stride_samples),
            (
                logical_block_x,
                logical_block_y,
                blocks_per_row,
                blocks_per_col,
            ),
            decode,
            private,
        );
    } else {
        consume_rows::<
            _,
            S,
            C,
            Offset,
            Decode,
            Private,
            CHANNELS,
            ACTIVE_CHANNELS,
            THREADS,
            PIXELS_PER_THREAD,
            RLE_COMPRESS,
        >(
            ScalarHistogramRead(samples),
            (num_row_pixels, num_rows, row_stride_samples),
            (
                logical_block_x,
                logical_block_y,
                blocks_per_row,
                blocks_per_col,
            ),
            decode,
            private,
        );
    }

    thread::sync_threads();
    histogram_channels!(ACTIVE_CHANNELS, channel, {
        let private_count = privatized_lengths[channel] as usize;
        let mut bin = thread_index;
        while bin < private_count {
            let count = private.read(channel, bin);
            if count.is_positive() {
                let output_bin = output_decode.bin::<channel>(bin);
                // Like CUB's output `BinSelect`, every nonnegative result is
                // constructed from this channel's validated level/bin domain.
                if output_bin >= 0 {
                    let channel_offset = if ACTIVE_CHANNELS == 1 {
                        0
                    } else {
                        output_offsets[channel]
                    };
                    C::fetch_add_device(output, channel_offset + output_bin as usize, count);
                }
            }
            bin += THREADS;
        }
    });
}

#[inline(always)]
fn consume_rows<
    Samples,
    S,
    C,
    Offset,
    Decode,
    Private,
    const CHANNELS: usize,
    const ACTIVE_CHANNELS: usize,
    const THREADS: usize,
    const PIXELS_PER_THREAD: usize,
    const RLE_COMPRESS: bool,
>(
    samples: Samples,
    image: (Offset, Offset, Offset),
    grid: (i32, i32, i32, i32),
    decode: Decode,
    private: &mut Private,
) where
    Samples: HistogramTileRead<S>,
    S: Copy + Default,
    C: HistogramCounter,
    Offset: HistogramOffset,
    Decode: HistogramSampleDecode<S>,
    Private: PrivatizedHistogram<C>,
{
    let (num_row_pixels, num_rows, row_stride_samples) = image;
    let (logical_block_x, logical_block_y, blocks_per_row, blocks_per_col) = grid;
    let tile_samples = Offset::from_usize(THREADS * PIXELS_PER_THREAD * CHANNELS);
    let mut row = logical_block_y;
    while Offset::from_i32(row) < num_rows {
        let row_begin = Offset::from_i32(row).mul(row_stride_samples);
        let row_end = row_begin.add(num_row_pixels.mul(Offset::from_usize(CHANNELS)));
        let mut tile_offset = row_begin.add(Offset::from_i32(logical_block_x).mul(tile_samples));
        while tile_offset < row_end {
            let remaining = row_end.sub(tile_offset);
            if CHANNELS == 1 {
                if remaining < tile_samples {
                    consume_single_channel_tile::<
                        Samples,
                        S,
                        C,
                        Offset,
                        Decode,
                        Private,
                        ACTIVE_CHANNELS,
                        PIXELS_PER_THREAD,
                        RLE_COMPRESS,
                        false,
                    >(samples, tile_offset, remaining.as_i32(), decode, private);
                    break;
                }
                consume_single_channel_tile::<
                    Samples,
                    S,
                    C,
                    Offset,
                    Decode,
                    Private,
                    ACTIVE_CHANNELS,
                    PIXELS_PER_THREAD,
                    RLE_COMPRESS,
                    true,
                >(samples, tile_offset, tile_samples.as_i32(), decode, private);
                tile_offset = tile_offset.add(Offset::from_i32(blocks_per_row).mul(tile_samples));
                continue;
            }
            if remaining < tile_samples {
                consume_tile::<
                    Samples,
                    S,
                    C,
                    Offset,
                    Decode,
                    Private,
                    CHANNELS,
                    ACTIVE_CHANNELS,
                    PIXELS_PER_THREAD,
                    RLE_COMPRESS,
                    false,
                >(samples, tile_offset, remaining.as_i32(), decode, private);
                break;
            }
            consume_tile::<
                Samples,
                S,
                C,
                Offset,
                Decode,
                Private,
                CHANNELS,
                ACTIVE_CHANNELS,
                PIXELS_PER_THREAD,
                RLE_COMPRESS,
                true,
            >(samples, tile_offset, tile_samples.as_i32(), decode, private);
            tile_offset = tile_offset.add(Offset::from_i32(blocks_per_row).mul(tile_samples));
        }
        row = row.wrapping_add(blocks_per_col);
    }
}

#[inline(always)]
fn consume_tile<
    Samples,
    S,
    C,
    Offset,
    Decode,
    Private,
    const CHANNELS: usize,
    const ACTIVE_CHANNELS: usize,
    const PIXELS_PER_THREAD: usize,
    const RLE_COMPRESS: bool,
    const FULL_TILE: bool,
>(
    samples: Samples,
    tile_offset: Offset,
    valid_samples: i32,
    decode: Decode,
    private: &mut Private,
) where
    Samples: HistogramTileRead<S>,
    S: Copy + Default,
    C: HistogramCounter,
    Offset: HistogramOffset,
    Decode: HistogramSampleDecode<S>,
    Private: PrivatizedHistogram<C>,
{
    let thread_pixel = thread::threadIdx_x() as i32 * PIXELS_PER_THREAD as i32;
    let thread_base = tile_offset.as_usize() + thread_pixel as usize * CHANNELS;
    let mut thread_samples = [[S::default(); CHANNELS]; PIXELS_PER_THREAD];
    if Samples::GROUPED && CHANNELS == 4 && PIXELS_PER_THREAD == 5 && FULL_TILE {
        let Some(tile) = samples.five_at(thread_base / 4) else {
            cuda_device::debug::trap();
        };
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            let pixel_samples = tile.four_at::<pixel>();
            histogram_channels!(CHANNELS, channel, {
                thread_samples[pixel][channel] = pixel_samples[channel];
            });
        });
    } else if Samples::GROUPED {
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            let valid =
                FULL_TILE || ((thread_pixel + pixel as i32) * CHANNELS as i32) < valid_samples;
            if valid {
                let Some(pixel_samples) = samples.four_at(thread_base / 4 + pixel) else {
                    cuda_device::debug::trap();
                };
                histogram_channels!(CHANNELS, channel, {
                    thread_samples[pixel][channel] = pixel_samples[channel];
                });
            }
        });
    } else {
        let thread_sample = thread_pixel * CHANNELS as i32;
        let capacity = (PIXELS_PER_THREAD * CHANNELS) as i32;
        let thread_valid_samples = if FULL_TILE {
            capacity
        } else {
            valid_samples.wrapping_sub(thread_sample).clamp(0, capacity)
        };
        let Some(tile) = samples.thread_tile::<PIXELS_PER_THREAD, CHANNELS, FULL_TILE>(
            thread_base,
            thread_valid_samples,
        ) else {
            cuda_device::debug::trap();
        };
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if let Some(pixel_samples) = tile.pixel::<pixel>() {
                histogram_channels!(CHANNELS, channel, {
                    thread_samples[pixel][channel] = pixel_samples.load::<channel>();
                });
            }
        });
    }

    if RLE_COMPRESS {
        accumulate_rle::<
            S,
            C,
            Decode,
            Private,
            CHANNELS,
            ACTIVE_CHANNELS,
            PIXELS_PER_THREAD,
            FULL_TILE,
        >(
            &thread_samples,
            thread_pixel,
            valid_samples,
            decode,
            private,
        );
    } else {
        accumulate_direct::<
            S,
            C,
            Decode,
            Private,
            CHANNELS,
            ACTIVE_CHANNELS,
            PIXELS_PER_THREAD,
            FULL_TILE,
        >(
            &thread_samples,
            thread_pixel,
            valid_samples,
            decode,
            private,
        );
    }
}

#[inline(always)]
fn consume_single_channel_tile<
    Samples,
    S,
    C,
    Offset,
    Decode,
    Private,
    const ACTIVE_CHANNELS: usize,
    const PIXELS_PER_THREAD: usize,
    const RLE_COMPRESS: bool,
    const FULL_TILE: bool,
>(
    samples: Samples,
    tile_offset: Offset,
    valid_samples: i32,
    decode: Decode,
    private: &mut Private,
) where
    Samples: HistogramTileRead<S>,
    S: Copy + Default,
    C: HistogramCounter,
    Offset: HistogramOffset,
    Decode: HistogramSampleDecode<S>,
    Private: PrivatizedHistogram<C>,
{
    let thread_pixel = thread::threadIdx_x() as i32 * PIXELS_PER_THREAD as i32;
    let thread_base = tile_offset.as_usize() + thread_pixel as usize;
    let mut thread_samples = [S::default(); PIXELS_PER_THREAD];
    if Samples::GROUPED && FULL_TILE && PIXELS_PER_THREAD == 16 {
        let vector_base = thread_base / 4;
        let Some(groups) = samples.four_quads_at(vector_base) else {
            cuda_device::debug::trap();
        };
        histogram_quads!(PIXELS_PER_THREAD, quad, {
            let pixel = quad * 4;
            let group = groups[quad];
            thread_samples[pixel] = group[0];
            thread_samples[pixel + 1] = group[1];
            thread_samples[pixel + 2] = group[2];
            thread_samples[pixel + 3] = group[3];
        });
    } else if Samples::GROUPED && FULL_TILE {
        let vector_base = thread_base / 4;
        histogram_quads!(PIXELS_PER_THREAD, quad, {
            let pixel = quad * 4;
            let Some(group) = samples.four_at(vector_base + quad) else {
                cuda_device::debug::trap();
            };
            thread_samples[pixel] = group[0];
            thread_samples[pixel + 1] = group[1];
            thread_samples[pixel + 2] = group[2];
            thread_samples[pixel + 3] = group[3];
        });
    } else {
        let thread_valid_samples = if FULL_TILE {
            PIXELS_PER_THREAD as i32
        } else {
            valid_samples
                .wrapping_sub(thread_pixel)
                .clamp(0, PIXELS_PER_THREAD as i32)
        };
        let Some(tile) = samples
            .thread_tile::<PIXELS_PER_THREAD, 1, FULL_TILE>(thread_base, thread_valid_samples)
        else {
            cuda_device::debug::trap();
        };
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if let Some(pixel_sample) = tile.pixel::<pixel>() {
                thread_samples[pixel] = pixel_sample.load::<0>();
            }
        });
    }

    if RLE_COMPRESS {
        let mut bins = [-1i32; PIXELS_PER_THREAD];
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if FULL_TILE || (thread_pixel + pixel as i32) < valid_samples {
                bins[pixel] = decode.bin::<0>(thread_samples[pixel]);
            }
        });
        let mut accumulator = 1u32;
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if pixel + 1 < PIXELS_PER_THREAD {
                if bins[pixel] != bins[pixel + 1] {
                    add_private::<C, Private>(bins[pixel], accumulator, 0, private);
                    accumulator = 0;
                }
                accumulator += 1;
            }
        });
        add_private::<C, Private>(bins[PIXELS_PER_THREAD - 1], accumulator, 0, private);
    } else {
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if FULL_TILE || (thread_pixel + pixel as i32) < valid_samples {
                add_private::<C, Private>(decode.bin::<0>(thread_samples[pixel]), 1, 0, private);
            }
        });
    }
}

#[inline(always)]
fn accumulate_rle<
    S,
    C,
    Decode,
    Private,
    const CHANNELS: usize,
    const ACTIVE_CHANNELS: usize,
    const PIXELS_PER_THREAD: usize,
    const FULL_TILE: bool,
>(
    samples: &[[S; CHANNELS]; PIXELS_PER_THREAD],
    thread_pixel: i32,
    valid_samples: i32,
    decode: Decode,
    private: &mut Private,
) where
    S: Copy,
    C: HistogramCounter,
    Decode: HistogramSampleDecode<S>,
    Private: PrivatizedHistogram<C>,
{
    histogram_channels!(ACTIVE_CHANNELS, channel, {
        let mut bins = [-1i32; PIXELS_PER_THREAD];
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if FULL_TILE || ((thread_pixel + pixel as i32) * CHANNELS as i32) < valid_samples {
                bins[pixel] = decode.bin::<channel>(samples[pixel][channel]);
            }
        });

        let mut accumulator = 1u32;
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if pixel + 1 < PIXELS_PER_THREAD {
                if bins[pixel] != bins[pixel + 1] {
                    add_private::<C, Private>(bins[pixel], accumulator, channel, private);
                    accumulator = 0;
                }
                accumulator += 1;
            }
        });
        add_private::<C, Private>(bins[PIXELS_PER_THREAD - 1], accumulator, channel, private);
    });
}

#[inline(always)]
fn accumulate_direct<
    S,
    C,
    Decode,
    Private,
    const CHANNELS: usize,
    const ACTIVE_CHANNELS: usize,
    const PIXELS_PER_THREAD: usize,
    const FULL_TILE: bool,
>(
    samples: &[[S; CHANNELS]; PIXELS_PER_THREAD],
    thread_pixel: i32,
    valid_samples: i32,
    decode: Decode,
    private: &mut Private,
) where
    S: Copy,
    C: HistogramCounter,
    Decode: HistogramSampleDecode<S>,
    Private: PrivatizedHistogram<C>,
{
    histogram_pixels!(PIXELS_PER_THREAD, pixel, {
        if FULL_TILE || ((thread_pixel + pixel as i32) * CHANNELS as i32) < valid_samples {
            histogram_channels!(ACTIVE_CHANNELS, channel, {
                add_private::<C, Private>(
                    decode.bin::<channel>(samples[pixel][channel]),
                    1,
                    channel,
                    private,
                );
            });
        }
    });
}

#[inline(always)]
fn add_private<C, Private>(bin: i32, run_length: u32, channel: usize, private: &mut Private)
where
    C: HistogramCounter,
    Private: PrivatizedHistogram<C>,
{
    // `Decode` is the Rust realization of CUB's `BinSelect`: it returns -1
    // for rejected samples and otherwise a bin in the validated privatized
    // domain. Rechecking the upper bound here changes CUB's hot atomic path.
    if bin < 0 {
        return;
    }
    let value = C::from_run_length(run_length);
    private.add(channel, bin as usize, value);
}

/// CUB's `UpperBound(levels, sample) - 1` range transform.
#[inline(always)]
pub fn histogram_range_bin<L: Copy + PartialOrd>(levels: &[L], sample: L) -> i32 {
    // This is CUB's SearchTransform exactly: UpperBound over all output
    // levels, minus one, followed only by the upper output-bin rejection.
    // A sample below the first level naturally produces -1.
    let bin = read_only::upper_bound(levels, sample).wrapping_sub(1);
    if bin >= (levels.len() as i32).wrapping_sub(1) {
        -1
    } else {
        bin
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EvenHistogramTransforms, HistogramFourReader, HistogramFourSampleLoad,
        HistogramOutputDecode, HistogramSample, HistogramSampleDecode, histogram_range_bin,
        rows_use_grouped_loads,
    };

    #[test]
    fn i32_grouped_reader_forms_one_checked_view_and_rejects_invalid_offsets() {
        #[repr(align(16))]
        struct Aligned([i32; 8]);

        let values = Aligned([0_i32, 1, 2, 3, 4, 5, 6, 7]);
        let reader = <i32 as HistogramFourSampleLoad>::four_reader(&values.0)
            .expect("test input has i32x4 alignment");
        assert_eq!(reader.vectors.len(), 2);
        assert_eq!(reader.vectors[0].to_array(), [0, 1, 2, 3]);
        assert_eq!(reader.vectors[1].to_array(), [4, 5, 6, 7]);
        assert_eq!(reader.four_at(2), None);
        assert_eq!(reader.four_at(usize::MAX), None);
    }

    #[test]
    fn i32_five_pixel_reader_proves_the_complete_cub_register_tile_once() {
        #[repr(align(16))]
        struct Aligned([i32; 20]);

        let values = Aligned(core::array::from_fn(|index| index as i32));
        let reader = <i32 as HistogramFourSampleLoad>::four_reader(&values.0)
            .expect("test input has i32x4 alignment");
        let tile = reader
            .five_at(0)
            .expect("five complete RGBA pixels are available");
        assert_eq!(tile.vectors[0].to_array(), [0, 1, 2, 3]);
        assert_eq!(tile.vectors[4].to_array(), [16, 17, 18, 19]);
        assert!(reader.five_at(1).is_none());
        assert!(reader.five_at(usize::MAX).is_none());
    }

    #[test]
    fn grouped_row_alignment_matches_cub_single_and_multi_rules() {
        assert!(rows_use_grouped_loads(1, 16, true, 1, 5));
        assert!(!rows_use_grouped_loads(1, 16, true, 2, 5));
        assert!(rows_use_grouped_loads(1, 16, true, 2, 8));
        assert!(!rows_use_grouped_loads(1, 5, true, 1, 8));
        assert!(!rows_use_grouped_loads(4, 5, true, 1, 5));
        assert!(rows_use_grouped_loads(4, 5, true, 1, 8));
        assert!(!rows_use_grouped_loads(4, 5, false, 1, 8));
    }

    #[test]
    fn even_integer_bin_regression_matches_cub_489() {
        let samples = [2_i32, 6, 7, 2, 3, 0, 2, 2, 6, 999];
        let mut histogram = [0_i32; 7];
        for sample in samples {
            let bin = sample.even_bin(0_i32, 12_i32, histogram.len() as i32);
            if bin >= 0 {
                histogram[bin as usize] += 1;
            }
        }
        assert_eq!(histogram, [1, 5, 0, 2, 1, 0, 0]);
    }

    #[test]
    fn i32_scale_transform_has_cub_value_layout() {
        assert_eq!(
            core::mem::size_of::<<i32 as HistogramSample<i32>>::EvenTransform>(),
            4 * core::mem::size_of::<i32>(),
        );
    }

    #[test]
    fn even_transform_array_selects_the_unrolled_channel_at_type_level() {
        let transforms = [
            <i32 as HistogramSample<i32>>::even_transform(0, 10, 5),
            <i32 as HistogramSample<i32>>::even_transform(10, 20, 5),
        ];
        let decode = EvenHistogramTransforms::<i32, i32, 2, false>::new(transforms);
        assert_eq!(HistogramSampleDecode::bin::<0>(decode, 4), 2);
        assert_eq!(HistogramSampleDecode::bin::<1>(decode, 14), 2);
        assert_eq!(HistogramOutputDecode::bin::<0>(decode, 3), 3);
    }

    #[test]
    fn integral_common_type_uses_cpp_promotions() {
        assert_eq!(<i8 as HistogramSample<u8>>::even_bin(5, 0, 10, 5), 2);
        assert_eq!(<u16 as HistogramSample<i16>>::even_bin(9, 0, 12, 6), 4);
    }

    #[test]
    fn even_overflow_contract_matches_cub() {
        assert!(!<u32 as HistogramSample<u32>>::even_may_overflow(
            0,
            u32::MAX,
            2,
        ));
        assert!(<u64 as HistogramSample<u64>>::even_may_overflow(
            0,
            u64::MAX,
            2,
        ));
        assert!(<i8 as HistogramSample<i8>>::even_may_overflow(0, 100, 128,));
    }

    #[test]
    fn range_uses_upper_bound_and_excludes_last_level() {
        let levels = [0.0_f32, 2.0, 4.0, 8.0];
        assert_eq!(histogram_range_bin(&levels, -1.0), -1);
        assert_eq!(histogram_range_bin(&levels, 0.0), 0);
        assert_eq!(histogram_range_bin(&levels, 2.0), 1);
        assert_eq!(histogram_range_bin(&levels, 7.9), 2);
        assert_eq!(histogram_range_bin(&levels, 8.0), -1);

        let duplicates = [0_i32, 2, 2, 8];
        assert_eq!(histogram_range_bin(&duplicates, 2), 2);
    }
}
