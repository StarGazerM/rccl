//! Word-interleaved frame-of-reference columns. Each lane owns a bit stream;
//! adjacent lane words are adjacent u32s. Tails pad only the last lane word.
use crate::DeviceReadAt;
use cuda_core::DeviceCopy;

/// A column frame. Offsets count u32 words from the payload origin.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, DeviceCopy)]
pub struct ForFrame {
    pub base: u64,
    pub offset: u64,
    pub bits: u32,
    pub rows: u32,
}

/// Storage required for a column, including its partial lane-word tail.
/// Widths 33..64 use an explicit wide, row-major representation.
/// `lanes` must be a nonzero power of two.
#[inline(always)]
pub const fn for_words(rows: u32, lanes: u32, bits: u32) -> u64 {
    if bits > 32 { (rows as u64 * bits as u64).div_ceil(32) }
    else { lanes as u64 * ((((rows as u64 + lanes as u64 - 1) >> lanes.trailing_zeros()) * bits as u64).div_ceil(32)) }
}

/// Borrowed column decoder; the owner retains its allocation through execution.
#[derive(Clone, Copy)]
pub struct ForColumn {
    pub payload: *const u32,
    pub base: u64,
    pub bits: u32,
    pub rows: u32,
    pub lanes: u32,
}
impl ForColumn {
    /// # Safety
    /// `row < rows <= lanes * 32`, width is at most 64, lanes is a nonzero
    /// power of two, and payload contains `for_words(rows, lanes, bits)`
    /// initialized words and remains immutable.
    #[inline(always)]
    pub unsafe fn read(self, row: u32) -> u64 {
        if self.bits == 0 { return self.base; }
        if self.bits > 32 {
            let bit = row as u64 * self.bits as u64;
            let word = bit / 32;
            let shift = (bit % 32) as u32;
            let mut value = unsafe { *self.payload.add(word as usize) as u64 } >> shift;
            if self.bits > 32 - shift {
                value |= unsafe { *self.payload.add(word as usize + 1) as u64 } << (32 - shift);
            }
            if shift != 0 && self.bits > 64 - shift {
                value |= unsafe { *self.payload.add(word as usize + 2) as u64 } << (64 - shift);
            }
            return self.base + (value & (u64::MAX >> (64 - self.bits)));
        }
        let bit = (row >> self.lanes.trailing_zeros()) * self.bits;
        let word = (bit / 32) * self.lanes + (row & (self.lanes - 1));
        let shift = bit % 32;
        let mut value = unsafe { *self.payload.add(word as usize) } >> shift;
        if shift != 0 && self.bits > 32 - shift {
            value |= unsafe { *self.payload.add((word + self.lanes) as usize) } << (32 - shift);
        }
        self.base + (value & (u32::MAX >> (32 - self.bits))) as u64
    }

    /// Compact step-uniform extraction for a warp-owned interleaved tile.
    /// # Safety
    /// `step * lanes + lane` obeys `read`'s contract.
    #[inline(always)]
    pub unsafe fn read_step(self, step: u32, lane: u32) -> u64 {
        if self.bits == 0 { return self.base; }
        if self.bits > 32 { return unsafe { self.read(step * self.lanes + lane) }; }
        let bit = step * self.bits;
        let word = bit >> 5;
        let shift = bit & 31;
        let lane_words = ((((self.rows + self.lanes - 1) >> self.lanes.trailing_zeros()) * self.bits) + 31) >> 5;
        let lo = unsafe { *self.payload.add((word * self.lanes + lane) as usize) };
        let hi = unsafe { *self.payload.add(((word + 1).min(lane_words - 1) * self.lanes + lane) as usize) };
        let residual = ((((hi as u64) << 32 | lo as u64) >> shift) as u32) & (u32::MAX >> (32 - self.bits));
        self.base + residual as u64
    }

    /// Visit one independent lane, consuming each value immediately. The
    /// compact loop keeps width, base and mask invariant; bit positions and
    /// shifts are step-uniform across the consecutive lanes of a subgroup.
    /// # Safety
    /// Same allocation contract as `read`; `lane < lanes`, `stride > 0`,
    /// `begin <= end <= rows <= lanes * 32`.
    #[inline(always)]
    pub unsafe fn visit_lane<F: FnMut(u32, u64)>(self, lane: u32, begin: u32, end: u32, stride: u32, mut consume: F) {
        if begin == end { return; }
        let lane_shift = self.lanes.trailing_zeros();
        let mut step = begin >> lane_shift;
        let stop = (end + self.lanes - 1) >> lane_shift;
        if self.bits == 0 {
            while step < stop {
                let row = step * self.lanes + lane;
                if row >= begin && row < end && (row - begin) % stride == 0 { consume(row, self.base); }
                step += 1;
            }
            return;
        }
        if self.bits > 32 {
            while step < stop {
                let row = step * self.lanes + lane;
                if row >= begin && row < end && (row - begin) % stride == 0 { consume(row, unsafe { self.read(row) }); }
                step += 1;
            }
            return;
        }
        while step < stop {
            let row = step * self.lanes + lane;
            if row >= begin && row < end && (row - begin) % stride == 0 {
                consume(row, unsafe { self.read_step(step, lane) });
            }
            step += 1;
        }
    }
}

/// Pack one independent output word, so concurrent word writers never race.
/// Input is any lowered read range (including a projection or merged tile).
/// # Safety
/// All `rows` input values fit `base + bits` without overflow; dimensions match
/// `for_words`, and `word` is in that output range. No output allocation occurs.
#[inline(always)]
pub unsafe fn pack_for_word<R: DeviceReadAt<Item = u64>>(input: R, rows: u32, lanes: u32, base: u64, bits: u32, word: u32) -> u32 {
    if bits == 0 { return 0; }
    let (lane, first_bit, row_stride) = if bits <= 32 {
        (word & (lanes - 1), (word >> lanes.trailing_zeros()) as u64 * 32, lanes)
    } else { (0, word as u64 * 32, 1) };
    let mut step = first_bit / bits as u64;
    let mut result = 0u32;
    while step * (bits as u64) < first_bit + 32 {
        let row = step * row_stride as u64 + lane as u64;
        if row >= rows as u64 { break; }
        let residual = unsafe { input.read_at(row as usize) } - base;
        let value_bit = step * bits as u64;
        if value_bit < first_bit { result |= (residual >> (first_bit - value_bit)) as u32; }
        else { result |= (residual as u32) << (value_bit - first_bit); }
        step += 1;
    }
    result
}

/// Device lowering for a column spanning independently framed blocks.
#[derive(Clone, Copy)]
pub struct DeviceInterleavedForRange<P, F> {
    pub payload: P,
    pub frames: F,
    pub block_rows: u32,
    pub lanes: u32,
}
impl<P: DeviceReadAt<Item = u32>, F: DeviceReadAt<Item = ForFrame>> DeviceReadAt for DeviceInterleavedForRange<P, F> {
    type Item = u64;
    #[inline(always)]
    unsafe fn read_at(self, index: usize) -> u64 {
        let frame = unsafe { self.frames.read_at(index / self.block_rows as usize) };
        let row = (index % self.block_rows as usize) as u32;
        if frame.bits == 0 { return frame.base; }
        let (bit, lane, step) = if frame.bits <= 32 {
            ((row >> self.lanes.trailing_zeros()) as u64 * frame.bits as u64, row & (self.lanes - 1), self.lanes)
        } else { (row as u64 * frame.bits as u64, 0, 1) };
        let word = frame.offset + (bit / 32) * step as u64 + lane as u64;
        let shift = (bit % 32) as u32;
        let mut value = unsafe { self.payload.read_at(word as usize) as u64 } >> shift;
        if frame.bits > 32 - shift { value |= unsafe { self.payload.read_at((word + step as u64) as usize) as u64 } << (32 - shift); }
        if shift != 0 && frame.bits > 64 - shift { value |= unsafe { self.payload.read_at((word + 2 * step as u64) as usize) as u64 } << (64 - shift); }
        frame.base + (value & (u64::MAX >> (64 - frame.bits)))
    }
}
