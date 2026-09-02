/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Read-only global-memory loads.
//!
//! [`load`] is the safe Rust equivalent of CUDA's `__ldg`: its shared borrow
//! proves that the operation cannot write through the supplied reference.  The
//! cuda-oxide compiler preserves the load as one cache-qualified instruction,
//! including for the over-aligned vector values in [`crate::vector`].

use core::{marker::PhantomData, num::NonZeroU32};

/// A bounds-proved contiguous register tile for one CUDA thread.
///
/// CUDA block algorithms commonly prove a complete global-memory tile once,
/// then issue several statically unrolled register loads from that tile. A
/// Rust slice access at every unrolled coordinate would repeat the same bounds
/// failure edge and retain it in PTX. This capability keeps the shared borrow,
/// validates the complete per-thread extent in [`Self::new`], and only exposes
/// coordinates whose pixel and channel are const parameters.
///
/// `FULL` preserves the same specialization as CUB's complete and partial
/// tile overloads. In the partial form, [`Self::pixel`] returns `None` exactly
/// for pixels beyond `valid_samples`; the backing slice extent was still
/// checked once by the constructor.
#[derive(Clone, Copy)]
pub struct ThreadTile<'a, T, const PIXELS: usize, const CHANNELS: usize, const FULL: bool> {
    samples: &'a [T],
    valid_samples: usize,
}

/// One active pixel borrowed from a [`ThreadTile`].
#[derive(Clone, Copy)]
pub struct ThreadPixel<'a, T, const CHANNELS: usize> {
    samples: &'a [T],
    base: usize,
}

impl<'a, T: Copy, const CHANNELS: usize> ThreadPixel<'a, T, CHANNELS> {
    /// Load one compile-time channel from this active pixel.
    #[inline(always)]
    pub fn load<const CHANNEL: usize>(&self) -> T {
        if CHANNEL >= CHANNELS {
            crate::debug::trap();
        }
        __load_at(self.samples, self.base + CHANNEL)
    }
}

impl<'a, T: Copy, const PIXELS: usize, const CHANNELS: usize, const FULL: bool>
    ThreadTile<'a, T, PIXELS, CHANNELS, FULL>
{
    /// Prove the complete active extent beginning at `start`.
    ///
    /// `valid_samples` is the number of active scalar samples owned by this
    /// thread, not the remaining size of the block tile. A full tile requires
    /// the complete `PIXELS * CHANNELS` extent; a partial tile accepts any
    /// prefix of it, including an empty prefix for inactive threads.
    #[inline(always)]
    pub fn new(input: &'a [T], start: usize, valid_samples: i32) -> Option<Self> {
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
            return Some(Self {
                samples: &input[..0],
                valid_samples,
            });
        }
        let end = start.checked_add(valid_samples)?;
        let samples = input.get(start..end)?;
        Some(Self {
            samples,
            valid_samples,
        })
    }

    /// Borrow one statically selected pixel when its complete channel group is
    /// active. `None` is only the normal partial-tile inactive state.
    #[inline(always)]
    pub fn pixel<const PIXEL: usize>(&self) -> Option<ThreadPixel<'a, T, CHANNELS>> {
        if PIXEL >= PIXELS {
            return None;
        }
        let local = PIXEL * CHANNELS;
        if !FULL && local + CHANNELS > self.valid_samples {
            return None;
        }
        Some(ThreadPixel {
            samples: self.samples,
            base: local,
        })
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct SliceDescriptor<T> {
    address: u64,
    len: NonZeroU32,
    padding: u32,
    marker: PhantomData<T>,
}

/// A fixed set of read-only device slices passed to a kernel by value.
///
/// The host launcher accepts only `cuda_host::ReadOnlySlices`, whose checked
/// constructor borrows one device allocation and validates every range. The
/// device representation is intentionally not constructible in safe Rust;
/// kernels can only recover shared slices whose lifetime is bounded by this
/// launch parameter.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ReadOnlySlices<T, const N: usize> {
    slices: [SliceDescriptor<T>; N],
}

impl<T, const N: usize> ReadOnlySlices<T, N> {
    /// Borrow one checked channel range.
    #[inline(always)]
    pub fn get(&self, index: usize) -> Option<ReadOnlyRange<'_, T>> {
        let descriptor = self.slices.get(index)?;
        Some(ReadOnlyRange {
            address: descriptor.address,
            len: descriptor.len,
            marker: PhantomData,
        })
    }

    /// Borrow a channel selected by a monomorphized histogram agent.
    #[inline(always)]
    pub fn get_const<const INDEX: usize>(&self) -> ReadOnlyRange<'_, T> {
        if INDEX >= N {
            crate::debug::trap();
        }
        let descriptor = &self.slices[INDEX];
        ReadOnlyRange {
            address: descriptor.address,
            len: descriptor.len,
            marker: PhantomData,
        }
    }
}

/// One host-validated CUB `SearchTransform` range.
///
/// The address and signed level count match CUB's transform object. The
/// lifetime is borrowed from the owning [`ReadOnlySlices`] kernel argument,
/// and the fields cannot be forged by device code.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ReadOnlyRange<'a, T> {
    address: u64,
    len: NonZeroU32,
    marker: PhantomData<&'a T>,
}

impl<T> ReadOnlyRange<'_, T> {
    /// Number of validated elements in this range.
    #[inline(always)]
    #[must_use]
    pub fn len(self) -> i32 {
        self.len.get() as i32
    }

    /// Ranges passed by [`ReadOnlySlices`] are always nonempty.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(self) -> bool {
        false
    }

    /// Return the first index whose value compares greater than `sample`.
    #[inline(always)]
    #[must_use]
    pub fn upper_bound(self, sample: T) -> i32
    where
        T: Copy + PartialOrd,
    {
        let mut retval = 0u32;
        let mut num_items = self.len.get();
        loop {
            let half = num_items >> 1;
            let probe = retval.wrapping_add(half);
            if sample < __load_address_at(self.address, probe as usize) {
                num_items = half;
            } else {
                let consumed = half.wrapping_add(1);
                retval = retval.wrapping_add(consumed);
                num_items = num_items.wrapping_sub(consumed);
            }
            if num_items == 0 {
                break;
            }
        }
        retval as i32
    }
}

/// Compiler boundary for a host-validated [`ReadOnlyRange`] probe.
///
/// The containing range is the only caller and maintains CUB's exact
/// `retval + half < len` interval invariant. Keeping the device address and
/// signed count separate avoids reconstructing a Rust fat slice in every
/// unrolled channel transform.
#[inline(never)]
fn __load_address_at<T: Copy>(_address: u64, _index: usize) -> T {
    unreachable!("read-only address load executed outside cuda-oxide device compilation")
}

/// Copy a value through CUDA's read-only global-memory path.
///
/// Device compilation recognizes this compiler stub as a memory intrinsic so
/// a value whose type carries 8- or 16-byte alignment is not scalarized when
/// its lanes are subsequently consumed.
#[must_use]
#[inline(never)]
pub fn load<T: Copy>(_value: &T) -> T {
    unreachable!("read_only::load executed outside cuda-oxide device compilation")
}

/// Return the first index whose value compares greater than `sample`.
///
/// This is CUDA's read-only-cache realization of the standard upper-bound
/// search. The loop maintains `retval <= input.len()` and
/// `retval + num_items <= input.len()`, so every private indexed load is
/// proved in bounds. cuda-oxide recognizes that private load boundary and
/// emits one cache-qualified transaction without duplicating a bounds trap at
/// every search iteration.
#[must_use]
#[inline(always)]
pub fn upper_bound<T: Copy + PartialOrd>(input: &[T], sample: T) -> i32 {
    let mut retval = 0i32;
    let mut num_items = input.len() as i32;
    while num_items > 0 {
        let half = num_items >> 1;
        let probe = retval.wrapping_add(half);
        if sample < __load_at(input, probe as usize) {
            num_items = half;
        } else {
            let consumed = half.wrapping_add(1);
            retval = retval.wrapping_add(consumed);
            num_items = num_items.wrapping_sub(consumed);
        }
    }
    retval
}

/// Compiler boundary for the one indexed read in [`upper_bound`].
///
/// This function is private so arbitrary device code cannot present an
/// unproved index. Its call sites are the interval-preserving search above and
/// the [`ThreadPixel`] token, whose private fields can only be produced after
/// [`ThreadTile`] has validated the complete active extent.
#[inline(never)]
fn __load_at<T: Copy>(input: &[T], index: usize) -> T {
    input[index]
}

#[cfg(test)]
mod tests {
    use super::{ReadOnlyRange, ReadOnlySlices, SliceDescriptor, ThreadTile};

    #[test]
    fn read_only_slice_set_matches_cub_search_transform_per_channel() {
        assert_eq!(
            core::mem::size_of::<ReadOnlySlices<u32, 3>>(),
            6 * core::mem::size_of::<u64>()
        );
        assert_eq!(
            core::mem::align_of::<ReadOnlySlices<u32, 3>>(),
            core::mem::align_of::<u64>()
        );
        assert_eq!(
            core::mem::size_of::<ReadOnlyRange<'static, u32>>(),
            2 * core::mem::size_of::<u64>()
        );
        assert_eq!(core::mem::offset_of!(SliceDescriptor<u32>, address), 0);
        assert_eq!(core::mem::offset_of!(SliceDescriptor<u32>, len), 8);
        assert_eq!(core::mem::offset_of!(SliceDescriptor<u32>, padding), 12);
    }

    #[test]
    fn thread_tile_proves_one_extent_and_keeps_partial_pixels_explicit() {
        let samples = [0i32; 12];
        let full =
            ThreadTile::<_, 2, 2, true>::new(&samples, 4, 4).expect("complete two-pixel extent");
        assert!(full.pixel::<0>().is_some());
        assert!(full.pixel::<1>().is_some());

        let partial =
            ThreadTile::<_, 2, 2, false>::new(&samples, 8, 2).expect("one complete active pixel");
        assert!(partial.pixel::<0>().is_some());
        assert!(partial.pixel::<1>().is_none());

        assert!(ThreadTile::<_, 2, 2, true>::new(&samples, 9, 4).is_none());
        assert!(ThreadTile::<_, 2, 2, false>::new(&samples, 0, 3).is_none());
        let inactive = ThreadTile::<_, 2, 2, false>::new(&samples, usize::MAX, 0)
            .expect("inactive CUDA threads never form their nominal address");
        assert!(inactive.pixel::<0>().is_none());
    }
}
