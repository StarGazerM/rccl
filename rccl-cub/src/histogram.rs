use core::ops::Range;

use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};

use crate::DeviceAlgorithmError;

/// A device-wide histogram whose channel storage is one affine allocation.
///
/// Channel `i` occupies `lengths()[i]` consecutive counters.  Keeping the
/// channels together replaces CUB's host array of raw output pointers while
/// preserving its independently-sized channel histograms.
pub struct DeviceHistogram<C: DeviceCopy, const ACTIVE_CHANNELS: usize> {
    counters: DeviceBuffer<C>,
    lengths: [usize; ACTIVE_CHANNELS],
    offsets: [usize; ACTIVE_CHANNELS],
}

impl<C: DeviceCopy, const ACTIVE_CHANNELS: usize> DeviceHistogram<C, ACTIVE_CHANNELS> {
    pub fn new(
        stream: &CudaStream,
        lengths: [usize; ACTIVE_CHANNELS],
    ) -> Result<Self, DeviceAlgorithmError> {
        let (offsets, total) = channel_layout(lengths)?;
        let counters = DeviceBuffer::zeroed(stream, total).map_err(DeviceAlgorithmError::Driver)?;
        Ok(Self {
            counters,
            lengths,
            offsets,
        })
    }

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
    ) -> Result<[Vec<C>; ACTIVE_CHANNELS], DeviceAlgorithmError> {
        let flat = self
            .counters
            .to_host_vec(stream)
            .map_err(DeviceAlgorithmError::Driver)?;
        Ok(core::array::from_fn(|channel| {
            let begin = self.offsets[channel];
            let end = begin + self.lengths[channel];
            flat[begin..end].to_vec()
        }))
    }

    #[doc(hidden)]
    pub fn __parts_mut(
        &mut self,
    ) -> (
        &mut DeviceBuffer<C>,
        [usize; ACTIVE_CHANNELS],
        [usize; ACTIVE_CHANNELS],
    ) {
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
pub struct DeviceHistogramWorkspace<C: DeviceCopy, const ACTIVE_CHANNELS: usize> {
    privatized: DeviceBuffer<C>,
    queue: DeviceBuffer<i32>,
    privatized_lengths: [usize; ACTIVE_CHANNELS],
    privatized_offsets: [usize; ACTIVE_CHANNELS],
    blocks_per_row: usize,
    blocks_per_col: usize,
    region: (usize, usize, usize),
    narrow_offsets: bool,
}

/// Affine decomposition used only by the generated histogram launch.
#[doc(hidden)]
pub struct DeviceHistogramWorkspaceParts<'a, C: DeviceCopy, const ACTIVE_CHANNELS: usize> {
    pub privatized: &'a mut DeviceBuffer<C>,
    pub queue: &'a mut DeviceBuffer<i32>,
    pub privatized_lengths: [usize; ACTIVE_CHANNELS],
    pub privatized_offsets: [usize; ACTIVE_CHANNELS],
    pub blocks_per_row: usize,
    pub blocks_per_col: usize,
    pub region: (usize, usize, usize),
    pub narrow_offsets: bool,
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
        let blocks = blocks_per_row
            .checked_mul(blocks_per_col)
            .ok_or(DeviceAlgorithmError::SizeOverflow)?;
        let mut allocation_lengths = [0usize; ACTIVE_CHANNELS];
        for channel in 0..ACTIVE_CHANNELS {
            allocation_lengths[channel] = privatized_lengths[channel]
                .checked_mul(blocks)
                .ok_or(DeviceAlgorithmError::SizeOverflow)?;
        }
        let (privatized_offsets, total) = channel_layout(allocation_lengths)?;
        let privatized =
            DeviceBuffer::zeroed(stream, total).map_err(DeviceAlgorithmError::Driver)?;
        let queue = DeviceBuffer::zeroed(stream, 1).map_err(DeviceAlgorithmError::Driver)?;
        Ok(Self {
            privatized,
            queue,
            privatized_lengths,
            privatized_offsets,
            blocks_per_row,
            blocks_per_col,
            region,
            narrow_offsets,
        })
    }

    #[doc(hidden)]
    pub fn __parts_mut(&mut self) -> DeviceHistogramWorkspaceParts<'_, C, ACTIVE_CHANNELS> {
        DeviceHistogramWorkspaceParts {
            privatized: &mut self.privatized,
            queue: &mut self.queue,
            privatized_lengths: self.privatized_lengths,
            privatized_offsets: self.privatized_offsets,
            blocks_per_row: self.blocks_per_row,
            blocks_per_col: self.blocks_per_col,
            region: self.region,
            narrow_offsets: self.narrow_offsets,
        }
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

/// Materialize CUB's host-initialized array of `SearchTransform` ranges while
/// retaining one shared borrow of their owning device allocation.
#[doc(hidden)]
pub fn histogram_read_only_slices<'a, L: DeviceCopy + Copy, const CHANNELS: usize>(
    storage: &'a DeviceBuffer<L>,
    lengths: [usize; CHANNELS],
    offsets: [usize; CHANNELS],
) -> Result<cuda_host::ReadOnlySlices<'a, L, CHANNELS>, DeviceAlgorithmError> {
    let mut overflowed = false;
    let ranges = core::array::from_fn(|channel| {
        let begin = offsets[channel];
        let end = begin.checked_add(lengths[channel]).unwrap_or_else(|| {
            overflowed = true;
            begin
        });
        begin..end
    });
    if overflowed {
        return Err(DeviceAlgorithmError::SizeOverflow);
    }
    cuda_host::ReadOnlySlices::try_from_ranges(storage, ranges).ok_or(
        DeviceAlgorithmError::InvalidHistogramShape(
            "histogram level ranges must be nonempty and lie in their device allocation",
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::{HistogramRegion, histogram_uses_i32_offsets};
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
}
