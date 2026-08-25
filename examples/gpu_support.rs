//! Target-local GPU fixtures shared by examples and GPU test binaries.
//!
//! This file is included with `mod gpu_support`; it is not an RCCL library
//! module. Inputs are constructed and outputs are validated in CUDA memory.

#![expect(
    dead_code,
    reason = "each including GPU target uses a different fixture subset"
)]

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig};
use rccl::{
    DeviceComparator,
    memory::{DeviceSlice, DeviceSliceMut},
};
use std::{error::Error, sync::Arc};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I32Pair {
    pub first: i32,
    pub second: i32,
}

unsafe impl DeviceCopy for I32Pair {}

#[derive(Clone, Copy, Debug, Default)]
pub struct I32PairLess;

unsafe impl DeviceComparator<I32Pair> for I32PairLess {
    unsafe fn less(&self, left: &I32Pair, right: &I32Pair) -> bool {
        (left.first, left.second) < (right.first, right.second)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I32Triple {
    pub first: i32,
    pub second: i32,
    pub third: i32,
}

unsafe impl DeviceCopy for I32Triple {}

#[derive(Clone, Copy, Debug, Default)]
pub struct I32TripleLess;

unsafe impl DeviceComparator<I32Triple> for I32TripleLess {
    unsafe fn less(&self, left: &I32Triple, right: &I32Triple) -> bool {
        (left.first, left.second, left.third) < (right.first, right.second, right.third)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MaskedI64Less {
    pub mask: i64,
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

unsafe impl DeviceComparator<i64> for MaskedI64Less {
    unsafe fn less(&self, left: &i64, right: &i64) -> bool {
        (left & self.mask) < (right & self.mask)
    }
}

#[expect(
    dead_code,
    reason = "cuda-oxide consumes these kernels during device compilation"
)]
#[cuda_host::cuda_module]
mod kernels {
    use super::{I32Pair, I32Triple};
    use cuda_core::DeviceCopy;
    use cuda_device::{
        atomic::{AtomicOrdering, DeviceAtomicU32},
        kernel, thread,
    };
    use rccl::DeviceComparator;

    #[inline(always)]
    fn splitmix64(index: u64) -> u64 {
        let mut value = index.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    #[kernel]
    pub unsafe fn fill_u32(output: *mut u32, count: u32, seed: u64, pattern: u32) {
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
            unsafe { output.add(index as usize).write(value) };
        }
    }

    #[kernel]
    pub unsafe fn fill_i64(output: *mut i64, count: u32, seed: u64) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            unsafe {
                output
                    .add(index as usize)
                    .write(splitmix64(index as u64 ^ seed) as i64)
            };
        }
    }

    #[kernel]
    pub unsafe fn fill_iota_u32(output: *mut u32, count: u32) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            unsafe { output.add(index as usize).write(index) };
        }
    }

    #[kernel]
    pub unsafe fn fill_i32_pairs(output: *mut I32Pair, count: u32, seed: u64) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            let first = (splitmix64(index as u64 ^ seed) & 0x3ff) as i32 - 512;
            let second = splitmix64(index as u64 ^ seed.rotate_left(17)) as i32;
            unsafe { output.add(index as usize).write(I32Pair { first, second }) };
        }
    }

    #[kernel]
    pub unsafe fn fill_i32_triples(output: *mut I32Triple, count: u32, seed: u64) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            let first = (splitmix64(index as u64 ^ seed) & 0xff) as i32 - 128;
            let second = splitmix64(index as u64 ^ seed.rotate_left(17)) as i32;
            let third = splitmix64(index as u64 ^ seed.rotate_left(31)) as i32;
            unsafe {
                output.add(index as usize).write(I32Triple {
                    first,
                    second,
                    third,
                })
            };
        }
    }

    #[kernel]
    pub unsafe fn validate_order<C, K>(comparator: C, keys: *const K, count: u32, errors: *mut u32)
    where
        C: DeviceComparator<K>,
        K: DeviceCopy,
    {
        let index = thread::index_1d().get() as u32;
        if index + 1 < count {
            let left = unsafe { &*keys.add(index as usize) };
            let right = unsafe { &*keys.add(index as usize + 1) };
            if unsafe { comparator.less(right, left) } {
                unsafe { DeviceAtomicU32::from_ptr(errors) }.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }

    #[kernel]
    pub unsafe fn validate_stable_pairs<C, K>(
        comparator: C,
        keys: *const K,
        values: *const u32,
        count: u32,
        errors: *mut u32,
    ) where
        C: DeviceComparator<K>,
        K: DeviceCopy,
    {
        let index = thread::index_1d().get() as u32;
        if index + 1 < count {
            let left_key = unsafe { &*keys.add(index as usize) };
            let right_key = unsafe { &*keys.add(index as usize + 1) };
            let left_value = unsafe { values.add(index as usize).read() };
            let right_value = unsafe { values.add(index as usize + 1).read() };
            let inversion = unsafe { comparator.less(right_key, left_key) };
            let equivalent = !unsafe { comparator.less(left_key, right_key) }
                && !unsafe { comparator.less(right_key, left_key) };
            if inversion || (equivalent && left_value >= right_value) {
                unsafe { DeviceAtomicU32::from_ptr(errors) }.fetch_add(1, AtomicOrdering::Relaxed);
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

    pub fn fill_u32<B: DeviceSliceMut<u32> + ?Sized>(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut B,
        seed: u64,
    ) -> Result<(), Box<dyn Error>> {
        self.fill_pattern_u32(stream, output, seed, U32Pattern::Random)
    }

    pub fn fill_low_entropy_u32<B: DeviceSliceMut<u32> + ?Sized>(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut B,
        seed: u64,
    ) -> Result<(), Box<dyn Error>> {
        self.fill_pattern_u32(stream, output, seed, U32Pattern::LowEntropy)
    }

    pub fn fill_pattern_u32<B: DeviceSliceMut<u32> + ?Sized>(
        &self,
        stream: &Arc<CudaStream>,
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
                    output.cu_deviceptr() as *mut u32,
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
        stream: &Arc<CudaStream>,
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
                    output.cu_deviceptr() as *mut i64,
                    count,
                    seed,
                )?;
            }
        }
        Ok(())
    }

    pub fn fill_iota_u32<B: DeviceSliceMut<u32> + ?Sized>(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut B,
    ) -> Result<(), Box<dyn Error>> {
        output.ensure_usable_on(stream)?;
        let count = Self::count(output.len())?;
        if count != 0 {
            unsafe {
                self.module.fill_iota_u32(
                    stream,
                    LaunchConfig::for_num_elems(count),
                    output.cu_deviceptr() as *mut u32,
                    count,
                )?;
            }
        }
        Ok(())
    }

    pub fn fill_i32_pairs<B: DeviceSliceMut<I32Pair> + ?Sized>(
        &self,
        stream: &Arc<CudaStream>,
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
                    output.cu_deviceptr() as *mut I32Pair,
                    count,
                    seed,
                )?;
            }
        }
        Ok(())
    }

    pub fn fill_i32_triples<B: DeviceSliceMut<I32Triple> + ?Sized>(
        &self,
        stream: &Arc<CudaStream>,
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
                    output.cu_deviceptr() as *mut I32Triple,
                    count,
                    seed,
                )?;
            }
        }
        Ok(())
    }

    pub fn validate_order<C, K, B>(
        &self,
        stream: &Arc<CudaStream>,
        keys: &B,
        comparator: C,
    ) -> Result<u32, Box<dyn Error>>
    where
        C: DeviceComparator<K>,
        K: DeviceCopy,
        B: DeviceSlice<K> + ?Sized,
    {
        keys.ensure_usable_on(stream)?;
        let count = Self::count(keys.len())?;
        if count < 2 {
            return Ok(0);
        }
        let errors = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            self.module.validate_order::<C, K>(
                stream,
                LaunchConfig::for_num_elems(count - 1),
                comparator,
                keys.cu_deviceptr() as *const K,
                count,
                errors.cu_deviceptr() as *mut u32,
            )?;
        }
        Ok(errors.to_host_vec(stream)?[0])
    }

    pub fn validate_stable_u32_pairs<C, KB, VB>(
        &self,
        stream: &Arc<CudaStream>,
        keys: &KB,
        values: &VB,
        comparator: C,
    ) -> Result<u32, Box<dyn Error>>
    where
        C: DeviceComparator<u32>,
        KB: DeviceSlice<u32> + ?Sized,
        VB: DeviceSlice<u32> + ?Sized,
    {
        self.validate_stable_pairs(stream, keys, values, comparator)
    }

    pub fn validate_stable_pairs<C, K, KB, VB>(
        &self,
        stream: &Arc<CudaStream>,
        keys: &KB,
        values: &VB,
        comparator: C,
    ) -> Result<u32, Box<dyn Error>>
    where
        C: DeviceComparator<K>,
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
        let errors = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            self.module.validate_stable_pairs::<C, K>(
                stream,
                LaunchConfig::for_num_elems(count - 1),
                comparator,
                keys.cu_deviceptr() as *const K,
                values.cu_deviceptr() as *const u32,
                count,
                errors.cu_deviceptr() as *mut u32,
            )?;
        }
        Ok(errors.to_host_vec(stream)?[0])
    }
}
