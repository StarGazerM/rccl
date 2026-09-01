use cuda_device::{DisjointSlice, thread};
use rccl_memory::device::{BlockSharedMemory, DeviceRegion, HistogramCounter};

const PRIVATIZED_SMEM_BINS: usize = 256;
const PRIVATIZED_SMEM_STRIDE: usize = PRIVATIZED_SMEM_BINS + 1;

macro_rules! histogram_channels {
    ($limit:expr, $channel:ident, $body:block) => {{
        {
            let $channel = 0usize;
            if $channel < $limit $body
        }
        {
            let $channel = 1usize;
            if $channel < $limit $body
        }
        {
            let $channel = 2usize;
            if $channel < $limit $body
        }
        {
            let $channel = 3usize;
            if $channel < $limit $body
        }
    }};
}

macro_rules! histogram_pixels {
    ($limit:expr, $pixel:ident, $body:block) => {{
        {
            let $pixel = 0usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 1usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 2usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 3usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 4usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 5usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 6usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 7usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 8usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 9usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 10usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 11usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 12usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 13usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 14usize;
            if $pixel < $limit $body
        }
        {
            let $pixel = 15usize;
            if $pixel < $limit $body
        }
    }};
}

/// Device-side random-access sample source used by the histogram agent.
///
/// The public range expression is lowered to one concrete implementation;
/// there is no runtime iterator tag or erased dispatch.
#[doc(hidden)]
pub trait HistogramRead<S>: Copy {
    fn sample_at(self, index: usize) -> Option<S>;
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
pub trait HistogramOffset:
    Copy
    + PartialOrd
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
{
    fn from_i32(value: i32) -> Self;
    fn from_usize(value: usize) -> Self;
    fn as_usize(self) -> usize;
    fn as_i32(self) -> i32;
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
}

impl<S: Copy> HistogramRead<S> for &[S] {
    #[inline(always)]
    fn sample_at(self, index: usize) -> Option<S> {
        self.get(index).copied()
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
    #[inline(always)]
    fn sample_at(self, index: usize) -> Option<S> {
        if index >= self.len {
            return None;
        }
        let offset = S::try_from(index).ok()?;
        Some(self.start + self.step * offset)
    }
}

/// Primitive sample arithmetic used by CUB's histogram transforms.
///
/// This is an ordinary Rust type-class boundary: the associated implementation
/// resolves from `S` and `L`, while the operation macro is reserved for target
/// policy and launch-schedule selection.
#[doc(hidden)]
pub trait HistogramSample<L>: Copy + Default {
    fn even_bin(self, lower: L, upper: L, bins: i32) -> i32;
    fn even_may_overflow(lower: L, upper: L, bins: usize) -> bool;
    fn byte_bin(self) -> i32;
    fn from_byte_bin(bin: usize) -> Self;
    fn to_level(self) -> L;
}

macro_rules! integral_histogram_sample {
    ($sample:ty, $level:ty, $common:ty, $wide:ty, signed) => {
        impl HistogramSample<$level> for $sample {
            #[inline(always)]
            fn even_bin(self, lower: $level, upper: $level, bins: i32) -> i32 {
                let sample = self as $common;
                let lower = lower as $common;
                let upper = upper as $common;
                if sample >= lower && sample < upper {
                    (((sample.wrapping_sub(lower) as $wide) * bins as $wide)
                        / (upper.wrapping_sub(lower) as $wide)) as i32
                } else {
                    -1
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
        impl HistogramSample<$level> for $sample {
            #[inline(always)]
            fn even_bin(self, lower: $level, upper: $level, bins: i32) -> i32 {
                let sample = self as $common;
                let lower = lower as $common;
                let upper = upper as $common;
                if sample >= lower && sample < upper {
                    (((sample.wrapping_sub(lower) as $wide) * bins as $wide)
                        / (upper.wrapping_sub(lower) as $wide)) as i32
                } else {
                    -1
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
        impl HistogramSample<$level> for $sample {
            #[inline(always)]
            fn even_bin(self, lower: $level, upper: $level, bins: i32) -> i32 {
                let sample = self as $common;
                let lower = lower as $common;
                let upper = upper as $common;
                if sample >= lower && sample < upper {
                    ((sample - lower) * (bins as $common / (upper - lower))) as i32
                } else {
                    -1
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
            output.write(output_offsets[channel] + output_bin, C::zero());
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
    output_storage: DisjointSlice<'_, C>,
    privatized_storage: DisjointSlice<'_, C>,
    output_layout: ([i32; ACTIVE_CHANNELS], [usize; ACTIVE_CHANNELS]),
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
    Decode: Fn(S, usize) -> i32 + Copy,
    OutputDecode: Fn(usize, usize) -> i32 + Copy,
{
    let (output_lengths, output_offsets) = output_layout;
    let (privatized_lengths, privatized_offsets) = privatized_layout;
    let (num_row_pixels, num_rows, row_stride_samples, blocks_per_row) = image;
    let thread_index = thread::threadIdx_x() as usize;
    let block_index = thread::blockIdx_x() as i32;
    let block_storage_index = block_index as usize;
    let logical_block_x = block_index % blocks_per_row;
    let logical_block_y = block_index / blocks_per_row;
    let blocks_per_col = thread::gridDim_x() as i32 / blocks_per_row;
    let tile_samples = Offset::from_usize(THREADS * PIXELS_PER_THREAD * CHANNELS);
    let mut output = DeviceRegion::new(output_storage);
    let mut privatized = DeviceRegion::new(privatized_storage);
    if SHARED_PRIVATIZATION {
        let mut bins = shared_memory.region::<C>(0, ACTIVE_CHANNELS * PRIVATIZED_SMEM_STRIDE);
        histogram_channels!(ACTIVE_CHANNELS, channel, {
            let count = privatized_lengths[channel] as usize;
            let channel_base = channel * PRIVATIZED_SMEM_STRIDE;
            let mut bin = thread_index;
            while bin < count {
                bins.write(channel_base + bin, C::zero());
                bin += THREADS;
            }
        });
    } else {
        histogram_channels!(ACTIVE_CHANNELS, channel, {
            let count = privatized_lengths[channel] as usize;
            let channel_base = privatized_offsets[channel] + block_storage_index * count;
            let mut bin = thread_index;
            while bin < count {
                privatized.write(channel_base + bin, C::zero());
                bin += THREADS;
            }
        });
    }
    thread::sync_threads();

    let mut row = logical_block_y;
    while Offset::from_i32(row) < num_rows {
        let row_begin = Offset::from_i32(row) * row_stride_samples;
        let row_end = row_begin + num_row_pixels * Offset::from_usize(CHANNELS);
        let mut tile_offset = row_begin + Offset::from_i32(logical_block_x) * tile_samples;
        while tile_offset < row_end {
            let remaining = row_end - tile_offset;
            if CHANNELS == 1 {
                if remaining < tile_samples {
                    consume_single_channel_tile::<
                        Samples,
                        S,
                        C,
                        Offset,
                        Decode,
                        ACTIVE_CHANNELS,
                        PIXELS_PER_THREAD,
                        RLE_COMPRESS,
                        SHARED_PRIVATIZATION,
                        false,
                    >(
                        samples,
                        tile_offset,
                        remaining.as_i32(),
                        block_storage_index,
                        privatized_lengths,
                        privatized_offsets,
                        decode,
                        &shared_memory,
                        &mut privatized,
                    );
                    break;
                }
                consume_single_channel_tile::<
                    Samples,
                    S,
                    C,
                    Offset,
                    Decode,
                    ACTIVE_CHANNELS,
                    PIXELS_PER_THREAD,
                    RLE_COMPRESS,
                    SHARED_PRIVATIZATION,
                    true,
                >(
                    samples,
                    tile_offset,
                    tile_samples.as_i32(),
                    block_storage_index,
                    privatized_lengths,
                    privatized_offsets,
                    decode,
                    &shared_memory,
                    &mut privatized,
                );
                tile_offset = tile_offset + Offset::from_i32(blocks_per_row) * tile_samples;
                continue;
            }
            if remaining < tile_samples {
                consume_tile::<
                    Samples,
                    S,
                    C,
                    Offset,
                    Decode,
                    CHANNELS,
                    ACTIVE_CHANNELS,
                    PIXELS_PER_THREAD,
                    RLE_COMPRESS,
                    SHARED_PRIVATIZATION,
                    false,
                >(
                    samples,
                    tile_offset,
                    remaining.as_i32(),
                    block_storage_index,
                    privatized_lengths,
                    privatized_offsets,
                    decode,
                    &shared_memory,
                    &mut privatized,
                );
                break;
            }
            consume_tile::<
                Samples,
                S,
                C,
                Offset,
                Decode,
                CHANNELS,
                ACTIVE_CHANNELS,
                PIXELS_PER_THREAD,
                RLE_COMPRESS,
                SHARED_PRIVATIZATION,
                true,
            >(
                samples,
                tile_offset,
                tile_samples.as_i32(),
                block_storage_index,
                privatized_lengths,
                privatized_offsets,
                decode,
                &shared_memory,
                &mut privatized,
            );
            tile_offset = tile_offset + Offset::from_i32(blocks_per_row) * tile_samples;
        }
        row += blocks_per_col;
    }

    thread::sync_threads();
    histogram_channels!(ACTIVE_CHANNELS, channel, {
        let private_count = privatized_lengths[channel] as usize;
        let private_base = privatized_offsets[channel] + block_storage_index * private_count;
        let shared_base = channel * PRIVATIZED_SMEM_STRIDE;
        let mut bin = thread_index;
        while bin < private_count {
            let count = if SHARED_PRIVATIZATION {
                shared_memory
                    .region::<C>(0, ACTIVE_CHANNELS * PRIVATIZED_SMEM_STRIDE)
                    .read(shared_base + bin)
            } else {
                privatized.read(private_base + bin)
            };
            if count.is_positive() {
                let output_bin = output_decode(bin, channel);
                if output_bin >= 0 && output_bin < output_lengths[channel] {
                    C::fetch_add_device(
                        &mut output,
                        output_offsets[channel] + output_bin as usize,
                        count,
                    );
                }
            }
            bin += THREADS;
        }
    });
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "private helper mirrors CUB's complete/partial ConsumeTile specializations"
)]
fn consume_tile<
    Samples,
    S,
    C,
    Offset,
    Decode,
    const CHANNELS: usize,
    const ACTIVE_CHANNELS: usize,
    const PIXELS_PER_THREAD: usize,
    const RLE_COMPRESS: bool,
    const SHARED_PRIVATIZATION: bool,
    const FULL_TILE: bool,
>(
    samples: Samples,
    tile_offset: Offset,
    valid_samples: i32,
    block_index: usize,
    privatized_lengths: [i32; ACTIVE_CHANNELS],
    privatized_offsets: [usize; ACTIVE_CHANNELS],
    decode: Decode,
    shared_memory: &BlockSharedMemory,
    privatized: &mut DeviceRegion<'_, C>,
) where
    Samples: HistogramRead<S>,
    S: Copy + Default,
    C: HistogramCounter,
    Offset: HistogramOffset,
    Decode: Fn(S, usize) -> i32 + Copy,
{
    let thread_pixel = thread::threadIdx_x() as i32 * PIXELS_PER_THREAD as i32;
    let thread_base = tile_offset.as_usize() + thread_pixel as usize * CHANNELS;
    let mut thread_samples = [[S::default(); CHANNELS]; PIXELS_PER_THREAD];
    let mut is_valid = [false; PIXELS_PER_THREAD];
    histogram_pixels!(PIXELS_PER_THREAD, pixel, {
        let valid = FULL_TILE || ((thread_pixel + pixel as i32) * CHANNELS as i32) < valid_samples;
        is_valid[pixel] = valid;
        if valid {
            histogram_channels!(CHANNELS, channel, {
                let index = thread_base + pixel * CHANNELS + channel;
                if let Some(sample) = samples.sample_at(index) {
                    thread_samples[pixel][channel] = sample;
                } else {
                    is_valid[pixel] = false;
                }
            });
        }
    });

    if RLE_COMPRESS {
        accumulate_rle::<
            S,
            C,
            Decode,
            CHANNELS,
            ACTIVE_CHANNELS,
            PIXELS_PER_THREAD,
            SHARED_PRIVATIZATION,
        >(
            &thread_samples,
            &is_valid,
            block_index,
            privatized_lengths,
            privatized_offsets,
            decode,
            shared_memory,
            privatized,
        );
    } else {
        accumulate_direct::<
            S,
            C,
            Decode,
            CHANNELS,
            ACTIVE_CHANNELS,
            PIXELS_PER_THREAD,
            SHARED_PRIVATIZATION,
        >(
            &thread_samples,
            &is_valid,
            block_index,
            privatized_lengths,
            privatized_offsets,
            decode,
            shared_memory,
            privatized,
        );
    }
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "private helper preserves CUB's single-channel full/partial tile specialization"
)]
fn consume_single_channel_tile<
    Samples,
    S,
    C,
    Offset,
    Decode,
    const ACTIVE_CHANNELS: usize,
    const PIXELS_PER_THREAD: usize,
    const RLE_COMPRESS: bool,
    const SHARED_PRIVATIZATION: bool,
    const FULL_TILE: bool,
>(
    samples: Samples,
    tile_offset: Offset,
    valid_samples: i32,
    block_index: usize,
    privatized_lengths: [i32; ACTIVE_CHANNELS],
    privatized_offsets: [usize; ACTIVE_CHANNELS],
    decode: Decode,
    shared_memory: &BlockSharedMemory,
    privatized: &mut DeviceRegion<'_, C>,
) where
    Samples: HistogramRead<S>,
    S: Copy + Default,
    C: HistogramCounter,
    Offset: HistogramOffset,
    Decode: Fn(S, usize) -> i32 + Copy,
{
    let thread_pixel = thread::threadIdx_x() as i32 * PIXELS_PER_THREAD as i32;
    let thread_base = tile_offset.as_usize() + thread_pixel as usize;
    let mut thread_samples = [S::default(); PIXELS_PER_THREAD];
    let mut is_valid = [false; PIXELS_PER_THREAD];
    histogram_pixels!(PIXELS_PER_THREAD, pixel, {
        let valid = FULL_TILE || (thread_pixel + pixel as i32) < valid_samples;
        is_valid[pixel] = valid;
        if valid {
            if let Some(sample) = samples.sample_at(thread_base + pixel) {
                thread_samples[pixel] = sample;
            } else {
                is_valid[pixel] = false;
            }
        }
    });
    if RLE_COMPRESS {
        let mut bins = [-1i32; PIXELS_PER_THREAD];
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if is_valid[pixel] {
                bins[pixel] = decode(thread_samples[pixel], 0);
            }
        });
        let mut accumulator = 1u32;
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if pixel + 1 < PIXELS_PER_THREAD {
                if bins[pixel] != bins[pixel + 1] {
                    add_private::<C, ACTIVE_CHANNELS, SHARED_PRIVATIZATION>(
                        bins[pixel],
                        accumulator,
                        0,
                        block_index,
                        privatized_lengths,
                        privatized_offsets,
                        shared_memory,
                        privatized,
                    );
                    accumulator = 0;
                }
                accumulator += 1;
            }
        });
        add_private::<C, ACTIVE_CHANNELS, SHARED_PRIVATIZATION>(
            bins[PIXELS_PER_THREAD - 1],
            accumulator,
            0,
            block_index,
            privatized_lengths,
            privatized_offsets,
            shared_memory,
            privatized,
        );
    } else {
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if is_valid[pixel] {
                add_private::<C, ACTIVE_CHANNELS, SHARED_PRIVATIZATION>(
                    decode(thread_samples[pixel], 0),
                    1,
                    0,
                    block_index,
                    privatized_lengths,
                    privatized_offsets,
                    shared_memory,
                    privatized,
                );
            }
        });
    }
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "private helper mirrors one CUB per-channel accumulation phase"
)]
fn accumulate_rle<
    S,
    C,
    Decode,
    const CHANNELS: usize,
    const ACTIVE_CHANNELS: usize,
    const PIXELS_PER_THREAD: usize,
    const SHARED_PRIVATIZATION: bool,
>(
    samples: &[[S; CHANNELS]; PIXELS_PER_THREAD],
    is_valid: &[bool; PIXELS_PER_THREAD],
    block_index: usize,
    privatized_lengths: [i32; ACTIVE_CHANNELS],
    privatized_offsets: [usize; ACTIVE_CHANNELS],
    decode: Decode,
    shared_memory: &BlockSharedMemory,
    privatized: &mut DeviceRegion<'_, C>,
) where
    S: Copy,
    C: HistogramCounter,
    Decode: Fn(S, usize) -> i32 + Copy,
{
    histogram_channels!(ACTIVE_CHANNELS, channel, {
        let mut bins = [-1i32; PIXELS_PER_THREAD];
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if is_valid[pixel] {
                bins[pixel] = decode(samples[pixel][channel], channel);
            }
        });

        let mut accumulator = 1u32;
        histogram_pixels!(PIXELS_PER_THREAD, pixel, {
            if pixel + 1 < PIXELS_PER_THREAD {
                if bins[pixel] != bins[pixel + 1] {
                    add_private::<C, ACTIVE_CHANNELS, SHARED_PRIVATIZATION>(
                        bins[pixel],
                        accumulator,
                        channel,
                        block_index,
                        privatized_lengths,
                        privatized_offsets,
                        shared_memory,
                        privatized,
                    );
                    accumulator = 0;
                }
                accumulator += 1;
            }
        });
        add_private::<C, ACTIVE_CHANNELS, SHARED_PRIVATIZATION>(
            bins[PIXELS_PER_THREAD - 1],
            accumulator,
            channel,
            block_index,
            privatized_lengths,
            privatized_offsets,
            shared_memory,
            privatized,
        );
    });
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "private helper mirrors one CUB per-channel accumulation phase"
)]
fn accumulate_direct<
    S,
    C,
    Decode,
    const CHANNELS: usize,
    const ACTIVE_CHANNELS: usize,
    const PIXELS_PER_THREAD: usize,
    const SHARED_PRIVATIZATION: bool,
>(
    samples: &[[S; CHANNELS]; PIXELS_PER_THREAD],
    is_valid: &[bool; PIXELS_PER_THREAD],
    block_index: usize,
    privatized_lengths: [i32; ACTIVE_CHANNELS],
    privatized_offsets: [usize; ACTIVE_CHANNELS],
    decode: Decode,
    shared_memory: &BlockSharedMemory,
    privatized: &mut DeviceRegion<'_, C>,
) where
    S: Copy,
    C: HistogramCounter,
    Decode: Fn(S, usize) -> i32 + Copy,
{
    histogram_pixels!(PIXELS_PER_THREAD, pixel, {
        if is_valid[pixel] {
            histogram_channels!(ACTIVE_CHANNELS, channel, {
                add_private::<C, ACTIVE_CHANNELS, SHARED_PRIVATIZATION>(
                    decode(samples[pixel][channel], channel),
                    1,
                    channel,
                    block_index,
                    privatized_lengths,
                    privatized_offsets,
                    shared_memory,
                    privatized,
                );
            });
        }
    });
}

#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "private helper keeps the shared/global privatization branch explicit"
)]
fn add_private<C, const ACTIVE_CHANNELS: usize, const SHARED_PRIVATIZATION: bool>(
    bin: i32,
    run_length: u32,
    channel: usize,
    block_index: usize,
    privatized_lengths: [i32; ACTIVE_CHANNELS],
    privatized_offsets: [usize; ACTIVE_CHANNELS],
    shared_memory: &BlockSharedMemory,
    privatized: &mut DeviceRegion<'_, C>,
) where
    C: HistogramCounter,
{
    if bin < 0 || bin >= privatized_lengths[channel] {
        return;
    }
    let value = C::from_run_length(run_length);
    if SHARED_PRIVATIZATION {
        let mut shared = shared_memory.region::<C>(0, ACTIVE_CHANNELS * PRIVATIZED_SMEM_STRIDE);
        C::fetch_add_block(
            &mut shared,
            channel * PRIVATIZED_SMEM_STRIDE + bin as usize,
            value,
        );
    } else {
        C::fetch_add_device(
            privatized,
            privatized_offsets[channel]
                + block_index * privatized_lengths[channel] as usize
                + bin as usize,
            value,
        );
    }
}

/// CUB's `UpperBound(levels, sample) - 1` range transform.
#[inline(always)]
pub fn histogram_range_bin<L: Copy + PartialOrd>(levels: &[L], sample: L) -> i32 {
    // The public call has already established CUB's signed level-count
    // domain. Keep the device search in that same domain instead of widening
    // every UpperBound control operation to Rust's host-sized `usize`.
    let mut begin = 0i32;
    let mut end = levels.len() as i32;
    while begin < end {
        let middle = begin + (end - begin) / 2;
        if sample < levels[middle as usize] {
            end = middle;
        } else {
            begin = middle + 1;
        }
    }
    let bin = begin - 1;
    if bin >= 0 && bin + 1 < levels.len() as i32 {
        bin
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::{HistogramSample, histogram_range_bin};

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
    }
}
