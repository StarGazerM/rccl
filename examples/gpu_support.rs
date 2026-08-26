//! Target-local GPU fixtures shared by examples and GPU test binaries.
//!
//! This file is included with `mod gpu_support`; it is not an RCCL library
//! module. Inputs are constructed and outputs are validated in CUDA memory.

#![expect(
    dead_code,
    reason = "each including GPU target uses a different fixture subset"
)]

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig};
use rccl::memory::{DeviceSlice, DeviceSliceMut};
use std::{cmp::Ordering, error::Error, sync::Arc};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I32Pair {
    pub first: i32,
    pub second: i32,
}

unsafe impl DeviceCopy for I32Pair {}

#[inline(always)]
pub fn order_i32_pair(left: &I32Pair, right: &I32Pair) -> Ordering {
    (left.first, left.second).cmp(&(right.first, right.second))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I32Triple {
    pub first: i32,
    pub second: i32,
    pub third: i32,
}

unsafe impl DeviceCopy for I32Triple {}

/// Private by-value CUDA ABI range used only by these fixture kernels.
#[repr(C)]
struct FixtureRange<T> {
    pointer: *const T,
    len: u64,
}

impl<T> Copy for FixtureRange<T> {}
impl<T> Clone for FixtureRange<T> {
    fn clone(&self) -> Self {
        *self
    }
}
unsafe impl<T> DeviceCopy for FixtureRange<T> {}

impl<T> FixtureRange<T> {
    #[inline(always)]
    unsafe fn read(self, index: u64) -> T {
        debug_assert!(index < self.len);
        unsafe { self.pointer.add(index as usize).read() }
    }
}

/// Writable counterpart to [`FixtureRange`]. Its address never escapes the
/// immediate launch helper that constructs it from a borrowed device slice.
#[repr(C)]
struct FixtureRangeMut<T> {
    pointer: *mut T,
    len: u64,
}

impl<T> Copy for FixtureRangeMut<T> {}
impl<T> Clone for FixtureRangeMut<T> {
    fn clone(&self) -> Self {
        *self
    }
}
unsafe impl<T> DeviceCopy for FixtureRangeMut<T> {}

impl<T> FixtureRangeMut<T> {
    #[inline(always)]
    unsafe fn write(self, index: u64, value: T) {
        debug_assert!(index < self.len);
        unsafe { self.pointer.add(index as usize).write(value) };
    }

    const fn as_mut_ptr(self) -> *mut T {
        self.pointer
    }
}

#[inline(always)]
pub fn order_i32_triple(left: &I32Triple, right: &I32Triple) -> Ordering {
    (left.first, left.second, left.third).cmp(&(right.first, right.second, right.third))
}

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum U32Pattern {
    Random = 0,
    Uniform = 1,
    Ascending = 2,
    Descending = 3,
    LowEntropy = 4,
    Alternating = 5,
}

#[expect(
    dead_code,
    reason = "cuda-oxide consumes these kernels during device compilation"
)]
#[cuda_host::cuda_module]
mod kernels {
    use super::{FixtureRange, FixtureRangeMut, I32Pair, I32Triple};
    use core::cmp::Ordering;
    use cuda_core::DeviceCopy;
    use cuda_device::{
        atomic::{AtomicOrdering, DeviceAtomicU32},
        kernel, thread,
    };

    #[inline(always)]
    fn splitmix64(index: u64) -> u64 {
        let mut value = index.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    #[kernel]
    pub unsafe fn fill_u32(output: FixtureRangeMut<u32>, count: u32, seed: u64, pattern: u32) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            let random = splitmix64(index as u64 ^ seed) as u32;
            let value = match pattern {
                1 => seed as u32,
                2 => index,
                3 => count - 1 - index,
                4 => random & 15,
                5 => {
                    if index & 1 == 0 {
                        0
                    } else {
                        u32::MAX
                    }
                }
                _ => random,
            };
            unsafe { output.write(index as u64, value) };
        }
    }

    #[kernel]
    pub unsafe fn fill_i64(output: FixtureRangeMut<i64>, count: u32, seed: u64) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            unsafe { output.write(index as u64, splitmix64(index as u64 ^ seed) as i64) };
        }
    }

    #[kernel]
    pub unsafe fn fill_iota_u32(output: FixtureRangeMut<u32>, count: u32) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            unsafe { output.write(index as u64, index) };
        }
    }

    #[kernel]
    pub unsafe fn fill_i32_pairs(output: FixtureRangeMut<I32Pair>, count: u32, seed: u64) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            let first = (splitmix64(index as u64 ^ seed) & 0x3ff) as i32 - 512;
            let second = splitmix64(index as u64 ^ seed.rotate_left(17)) as i32;
            unsafe { output.write(index as u64, I32Pair { first, second }) };
        }
    }

    #[kernel]
    pub unsafe fn fill_i32_triples(output: FixtureRangeMut<I32Triple>, count: u32, seed: u64) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            let first = (splitmix64(index as u64 ^ seed) & 0xff) as i32 - 128;
            let second = splitmix64(index as u64 ^ seed.rotate_left(17)) as i32;
            let third = splitmix64(index as u64 ^ seed.rotate_left(31)) as i32;
            unsafe {
                output.write(
                    index as u64,
                    I32Triple {
                        first,
                        second,
                        third,
                    },
                )
            };
        }
    }

    #[kernel]
    pub unsafe fn validate_order<C, K>(
        order: C,
        keys: FixtureRange<K>,
        count: u32,
        errors: FixtureRangeMut<u32>,
    ) where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy,
    {
        let index = thread::index_1d().get() as u32;
        if index + 1 < count {
            let left = unsafe { keys.read(index as u64) };
            let right = unsafe { keys.read(index as u64 + 1) };
            if order(&right, &left).is_lt() {
                unsafe { DeviceAtomicU32::from_ptr(errors.as_mut_ptr()) }
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }

    /// Independent oracle for a captured `i64` mask. Keeping the mask as an
    /// ordinary scalar kernel argument prevents closure lowering in the sort
    /// and closure lowering in its validator from masking the same defect.
    #[kernel]
    pub unsafe fn validate_masked_i64_order(
        keys: FixtureRange<i64>,
        count: u32,
        mask: i64,
        errors: FixtureRangeMut<u32>,
    ) {
        let index = thread::index_1d().get() as u32;
        if index + 1 < count {
            let left = unsafe { keys.read(index as u64) } & mask;
            let right = unsafe { keys.read(index as u64 + 1) } & mask;
            if right < left {
                unsafe { DeviceAtomicU32::from_ptr(errors.as_mut_ptr()) }
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }

    #[kernel]
    pub unsafe fn validate_masked_u32_order(
        keys: FixtureRange<u32>,
        count: u32,
        mask: u32,
        errors: FixtureRangeMut<u32>,
    ) {
        let index = thread::index_1d().get() as u32;
        if index + 1 < count {
            let left = unsafe { keys.read(index as u64) } & mask;
            let right = unsafe { keys.read(index as u64 + 1) } & mask;
            if right < left {
                unsafe { DeviceAtomicU32::from_ptr(errors.as_mut_ptr()) }
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }

    #[kernel]
    pub unsafe fn validate_iota_u32(
        values: FixtureRange<u32>,
        count: u32,
        errors: FixtureRangeMut<u32>,
    ) {
        let index = thread::index_1d().get() as u32;
        if index < count && unsafe { values.read(index as u64) } != index {
            unsafe { DeviceAtomicU32::from_ptr(errors.as_mut_ptr()) }
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    pub unsafe fn validate_zero_outside_u32(
        values: FixtureRange<u32>,
        count: u32,
        begin: u32,
        end: u32,
        errors: FixtureRangeMut<u32>,
    ) {
        let index = thread::index_1d().get() as u32;
        if index < count
            && (index < begin || index >= end)
            && unsafe { values.read(index as u64) } != 0
        {
            unsafe { DeviceAtomicU32::from_ptr(errors.as_mut_ptr()) }
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    pub unsafe fn validate_stable_pairs<C, K>(
        order: C,
        keys: FixtureRange<K>,
        values: FixtureRange<u32>,
        count: u32,
        errors: FixtureRangeMut<u32>,
    ) where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy,
    {
        let index = thread::index_1d().get() as u32;
        if index + 1 < count {
            let left_key = unsafe { keys.read(index as u64) };
            let right_key = unsafe { keys.read(index as u64 + 1) };
            let left_value = unsafe { values.read(index as u64) };
            let right_value = unsafe { values.read(index as u64 + 1) };
            let inversion = order(&right_key, &left_key).is_lt();
            let equivalent =
                order(&left_key, &right_key).is_eq() && order(&right_key, &left_key).is_eq();
            if inversion || (equivalent && left_value >= right_value) {
                unsafe { DeviceAtomicU32::from_ptr(errors.as_mut_ptr()) }
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }

    /// Independent scalar-argument oracle for the captured-mask stable pair
    /// sort exercised by the public test matrix.
    #[kernel]
    pub unsafe fn validate_masked_u32_stable_pairs(
        keys: FixtureRange<u32>,
        values: FixtureRange<u32>,
        count: u32,
        mask: u32,
        errors: FixtureRangeMut<u32>,
    ) {
        let index = thread::index_1d().get() as u32;
        if index + 1 < count {
            let left_key = unsafe { keys.read(index as u64) } & mask;
            let right_key = unsafe { keys.read(index as u64 + 1) } & mask;
            let left_value = unsafe { values.read(index as u64) };
            let right_value = unsafe { values.read(index as u64 + 1) };
            if right_key < left_key || (right_key == left_key && left_value >= right_value) {
                unsafe { DeviceAtomicU32::from_ptr(errors.as_mut_ptr()) }
                    .fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }
}

pub struct GpuSupport {
    module: kernels::LoadedModule,
}

impl GpuSupport {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self, cuda_host::EmbeddedModuleError> {
        Ok(Self {
            module: kernels::load(context)?,
        })
    }

    fn count(len: usize) -> Result<u32, Box<dyn Error>> {
        Ok(u32::try_from(len)?)
    }

    fn input_range<T, B: DeviceSlice<T> + ?Sized>(input: &B) -> FixtureRange<T> {
        FixtureRange {
            pointer: input.cu_deviceptr() as *const T,
            len: input.len() as u64,
        }
    }

    fn output_range<T, B: DeviceSliceMut<T> + ?Sized>(output: &mut B) -> FixtureRangeMut<T> {
        FixtureRangeMut {
            pointer: output.cu_deviceptr() as *mut T,
            len: output.len() as u64,
        }
    }

    pub fn fill_u32<B: DeviceSliceMut<u32> + ?Sized>(
        &self,
        stream: &CudaStream,
        output: &mut B,
        seed: u64,
    ) -> Result<(), Box<dyn Error>> {
        self.fill_pattern_u32(stream, output, seed, U32Pattern::Random)
    }

    pub fn fill_low_entropy_u32<B: DeviceSliceMut<u32> + ?Sized>(
        &self,
        stream: &CudaStream,
        output: &mut B,
        seed: u64,
    ) -> Result<(), Box<dyn Error>> {
        self.fill_pattern_u32(stream, output, seed, U32Pattern::LowEntropy)
    }

    pub fn fill_pattern_u32<B: DeviceSliceMut<u32> + ?Sized>(
        &self,
        stream: &CudaStream,
        output: &mut B,
        seed: u64,
        pattern: U32Pattern,
    ) -> Result<(), Box<dyn Error>> {
        output.ensure_usable_on(stream)?;
        let count = Self::count(output.len())?;
        if count != 0 {
            unsafe {
                self.module.fill_u32(
                    stream,
                    LaunchConfig::for_num_elems(count),
                    Self::output_range(output),
                    count,
                    seed,
                    pattern as u32,
                )?;
            }
        }
        Ok(())
    }

    pub fn fill_i64<B: DeviceSliceMut<i64> + ?Sized>(
        &self,
        stream: &CudaStream,
        output: &mut B,
        seed: u64,
    ) -> Result<(), Box<dyn Error>> {
        output.ensure_usable_on(stream)?;
        let count = Self::count(output.len())?;
        if count != 0 {
            unsafe {
                self.module.fill_i64(
                    stream,
                    LaunchConfig::for_num_elems(count),
                    Self::output_range(output),
                    count,
                    seed,
                )?;
            }
        }
        Ok(())
    }

    pub fn fill_iota_u32<B: DeviceSliceMut<u32> + ?Sized>(
        &self,
        stream: &CudaStream,
        output: &mut B,
    ) -> Result<(), Box<dyn Error>> {
        output.ensure_usable_on(stream)?;
        let count = Self::count(output.len())?;
        if count != 0 {
            unsafe {
                self.module.fill_iota_u32(
                    stream,
                    LaunchConfig::for_num_elems(count),
                    Self::output_range(output),
                    count,
                )?;
            }
        }
        Ok(())
    }

    pub fn fill_i32_pairs<B: DeviceSliceMut<I32Pair> + ?Sized>(
        &self,
        stream: &CudaStream,
        output: &mut B,
        seed: u64,
    ) -> Result<(), Box<dyn Error>> {
        output.ensure_usable_on(stream)?;
        let count = Self::count(output.len())?;
        if count != 0 {
            unsafe {
                self.module.fill_i32_pairs(
                    stream,
                    LaunchConfig::for_num_elems(count),
                    Self::output_range(output),
                    count,
                    seed,
                )?;
            }
        }
        Ok(())
    }

    pub fn fill_i32_triples<B: DeviceSliceMut<I32Triple> + ?Sized>(
        &self,
        stream: &CudaStream,
        output: &mut B,
        seed: u64,
    ) -> Result<(), Box<dyn Error>> {
        output.ensure_usable_on(stream)?;
        let count = Self::count(output.len())?;
        if count != 0 {
            unsafe {
                self.module.fill_i32_triples(
                    stream,
                    LaunchConfig::for_num_elems(count),
                    Self::output_range(output),
                    count,
                    seed,
                )?;
            }
        }
        Ok(())
    }

    pub fn validate_order<C, K, B>(
        &self,
        stream: &CudaStream,
        keys: &B,
        order: C,
    ) -> Result<u32, Box<dyn Error>>
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy,
        B: DeviceSlice<K> + ?Sized,
    {
        keys.ensure_usable_on(stream)?;
        let count = Self::count(keys.len())?;
        if count < 2 {
            return Ok(0);
        }
        let mut errors = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            self.module.validate_order::<C, K>(
                stream,
                LaunchConfig::for_num_elems(count - 1),
                order,
                Self::input_range(keys),
                count,
                Self::output_range(&mut errors),
            )?;
        }
        Ok(errors.to_host_vec(stream)?[0])
    }

    pub fn validate_iota_u32<B: DeviceSlice<u32> + ?Sized>(
        &self,
        stream: &CudaStream,
        values: &B,
    ) -> Result<u32, Box<dyn Error>> {
        values.ensure_usable_on(stream)?;
        let count = Self::count(values.len())?;
        if count == 0 {
            return Ok(0);
        }
        let mut errors = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            self.module.validate_iota_u32(
                stream,
                LaunchConfig::for_num_elems(count),
                Self::input_range(values),
                count,
                Self::output_range(&mut errors),
            )?;
        }
        Ok(errors.to_host_vec(stream)?[0])
    }

    pub fn validate_masked_i64_order<B: DeviceSlice<i64> + ?Sized>(
        &self,
        stream: &CudaStream,
        keys: &B,
        mask: i64,
    ) -> Result<u32, Box<dyn Error>> {
        keys.ensure_usable_on(stream)?;
        let count = Self::count(keys.len())?;
        if count < 2 {
            return Ok(0);
        }
        let mut errors = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            self.module.validate_masked_i64_order(
                stream,
                LaunchConfig::for_num_elems(count - 1),
                Self::input_range(keys),
                count,
                mask,
                Self::output_range(&mut errors),
            )?;
        }
        Ok(errors.to_host_vec(stream)?[0])
    }

    pub fn validate_masked_u32_order<B: DeviceSlice<u32> + ?Sized>(
        &self,
        stream: &CudaStream,
        keys: &B,
        mask: u32,
    ) -> Result<u32, Box<dyn Error>> {
        keys.ensure_usable_on(stream)?;
        let count = Self::count(keys.len())?;
        if count < 2 {
            return Ok(0);
        }
        let mut errors = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            self.module.validate_masked_u32_order(
                stream,
                LaunchConfig::for_num_elems(count - 1),
                Self::input_range(keys),
                count,
                mask,
                Self::output_range(&mut errors),
            )?;
        }
        Ok(errors.to_host_vec(stream)?[0])
    }

    pub fn validate_zero_outside_u32<B: DeviceSlice<u32> + ?Sized>(
        &self,
        stream: &CudaStream,
        values: &B,
        begin: usize,
        end: usize,
    ) -> Result<u32, Box<dyn Error>> {
        values.ensure_usable_on(stream)?;
        if begin > end || end > values.len() {
            return Err(format!(
                "validation range [{begin}..{end}) is outside {} elements",
                values.len()
            )
            .into());
        }
        let count = Self::count(values.len())?;
        if count == 0 {
            return Ok(0);
        }
        let begin = u32::try_from(begin)?;
        let end = u32::try_from(end)?;
        let mut errors = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            self.module.validate_zero_outside_u32(
                stream,
                LaunchConfig::for_num_elems(count),
                Self::input_range(values),
                count,
                begin,
                end,
                Self::output_range(&mut errors),
            )?;
        }
        Ok(errors.to_host_vec(stream)?[0])
    }

    pub fn validate_stable_u32_pairs<C, KB, VB>(
        &self,
        stream: &CudaStream,
        keys: &KB,
        values: &VB,
        order: C,
    ) -> Result<u32, Box<dyn Error>>
    where
        C: Fn(&u32, &u32) -> Ordering + Copy + 'static,
        KB: DeviceSlice<u32> + ?Sized,
        VB: DeviceSlice<u32> + ?Sized,
    {
        self.validate_stable_pairs(stream, keys, values, order)
    }

    pub fn validate_stable_pairs<C, K, KB, VB>(
        &self,
        stream: &CudaStream,
        keys: &KB,
        values: &VB,
        order: C,
    ) -> Result<u32, Box<dyn Error>>
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy,
        KB: DeviceSlice<K> + ?Sized,
        VB: DeviceSlice<u32> + ?Sized,
    {
        keys.ensure_usable_on(stream)?;
        values.ensure_usable_on(stream)?;
        if keys.len() != values.len() {
            return Err("key/value length mismatch".into());
        }
        let count = Self::count(keys.len())?;
        if count < 2 {
            return Ok(0);
        }
        let mut errors = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            self.module.validate_stable_pairs::<C, K>(
                stream,
                LaunchConfig::for_num_elems(count - 1),
                order,
                Self::input_range(keys),
                Self::input_range(values),
                count,
                Self::output_range(&mut errors),
            )?;
        }
        Ok(errors.to_host_vec(stream)?[0])
    }

    pub fn validate_masked_u32_stable_pairs<KB, VB>(
        &self,
        stream: &CudaStream,
        keys: &KB,
        values: &VB,
        mask: u32,
    ) -> Result<u32, Box<dyn Error>>
    where
        KB: DeviceSlice<u32> + ?Sized,
        VB: DeviceSlice<u32> + ?Sized,
    {
        keys.ensure_usable_on(stream)?;
        values.ensure_usable_on(stream)?;
        if keys.len() != values.len() {
            return Err(format!(
                "key length {} does not match value length {}",
                keys.len(),
                values.len()
            )
            .into());
        }
        let count = Self::count(keys.len())?;
        if count < 2 {
            return Ok(0);
        }
        let mut errors = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            self.module.validate_masked_u32_stable_pairs(
                stream,
                LaunchConfig::for_num_elems(count - 1),
                Self::input_range(keys),
                Self::input_range(values),
                count,
                mask,
                Self::output_range(&mut errors),
            )?;
        }
        Ok(errors.to_host_vec(stream)?[0])
    }
}
