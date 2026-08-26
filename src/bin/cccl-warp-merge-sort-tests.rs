#![feature(generic_const_exprs)]
#![expect(
    incomplete_features,
    reason = "WarpMergeSort matrix kernels currently require generic_const_exprs"
)]

//! GPU-only Rust port of CCCL's `catch2_test_warp_merge_sort.cu` matrix.
//!
//! Key/value fixtures, per-warp valid counts, and reference validation all
//! remain on the device. The host only receives one scalar failure counter per
//! case, so this test never stages record arrays through CPU memory.

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig};
use rccl::cub::{WarpMergeSort, WarpMergeSortTempStorage, warp_merge_sort_temp_storage_bytes};
use std::{cmp::Ordering, error::Error};

const TOTAL_WARPS: usize = 2;
const KEY_SEED: u64 = 0x4343_434c;
const VALUE_SEED: u64 = 0x5354_4142;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CubCustom {
    first: i32,
    second: i32,
}

unsafe impl DeviceCopy for CubCustom {}

trait MatrixKey: DeviceCopy + Copy + Default + 'static {
    const LABEL: &'static str;
    const OOB: Self;

    fn generate(index: usize, seed: u64) -> Self;

    /// # Safety
    ///
    /// Both references must be readable in the active CUDA execution context.
    unsafe fn device_less(left: &Self, right: &Self) -> bool;
}

trait MatrixValue: DeviceCopy + Copy + Default + 'static {
    const LABEL: &'static str;
    const ZERO: Self;

    fn generate(index: usize, seed: u64) -> Self;

    /// # Safety
    ///
    /// Both references must be readable in the active CUDA execution context.
    unsafe fn device_equal(left: &Self, right: &Self) -> bool;
}

#[inline(always)]
fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

macro_rules! primitive_key {
    ($ty:ty, $label:literal, $oob:expr) => {
        impl MatrixKey for $ty {
            const LABEL: &'static str = $label;
            const OOB: Self = $oob;

            #[inline(always)]
            fn generate(index: usize, seed: u64) -> Self {
                // Duplicate keys make pair validation exercise stability.
                (mix(index as u64 ^ seed) % 23) as Self
            }

            #[inline(always)]
            unsafe fn device_less(left: &Self, right: &Self) -> bool {
                left < right
            }
        }
    };
}

primitive_key!(u8, "u8", u8::MAX);
primitive_key!(i32, "i32", i32::MAX);
primitive_key!(i64, "i64", i64::MAX);

impl MatrixKey for CubCustom {
    const LABEL: &'static str = "CubCustom";
    const OOB: Self = Self {
        first: i32::MAX,
        second: i32::MAX,
    };

    #[inline(always)]
    fn generate(index: usize, seed: u64) -> Self {
        let random = mix(index as u64 ^ seed);
        Self {
            first: (random % 7) as i32,
            second: ((random >> 11) % 13) as i32,
        }
    }

    #[inline(always)]
    unsafe fn device_less(left: &Self, right: &Self) -> bool {
        left.first < right.first || (left.first == right.first && left.second < right.second)
    }
}

impl MatrixValue for i32 {
    const LABEL: &'static str = "i32";
    const ZERO: Self = 0;

    #[inline(always)]
    fn generate(index: usize, seed: u64) -> Self {
        mix(index as u64 ^ seed) as i32
    }

    #[inline(always)]
    unsafe fn device_equal(left: &Self, right: &Self) -> bool {
        left == right
    }
}

impl MatrixValue for CubCustom {
    const LABEL: &'static str = "CubCustom";
    const ZERO: Self = Self {
        first: 0,
        second: 0,
    };

    #[inline(always)]
    fn generate(index: usize, seed: u64) -> Self {
        let random = mix(index as u64 ^ seed);
        Self {
            first: random as i32,
            second: (random >> 32) as i32,
        }
    }

    #[inline(always)]
    unsafe fn device_equal(left: &Self, right: &Self) -> bool {
        left.first == right.first && left.second == right.second
    }
}

#[inline(always)]
fn cub_order<K: MatrixKey>(left: &K, right: &K) -> Ordering {
    if unsafe { K::device_less(left, right) } {
        Ordering::Less
    } else if unsafe { K::device_less(right, left) } {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

#[expect(
    dead_code,
    reason = "cuda-oxide consumes these kernels during device compilation"
)]
#[cuda_host::cuda_module]
mod kernels {
    use super::*;
    use cuda_device::{
        DisjointSlice, DynamicSharedArray,
        atomic::{AtomicOrdering, DeviceAtomicU32},
        kernel, thread,
    };

    #[inline(always)]
    fn expected_valid<const ITEMS: usize, const LOGICAL: usize>(warp: usize, action: u32) -> u32 {
        let tile_items = ITEMS * LOGICAL;
        match (action, warp) {
            (0 | 1, _) => tile_items as u32,
            (2 | 4, 0) => 0,
            (2 | 4, _) => (tile_items - 1) as u32,
            (3 | 5, 0) => ITEMS.saturating_sub(1).max(1) as u32,
            (3, _) => (tile_items / 2 + 1).min(tile_items) as u32,
            (_, _) => tile_items as u32,
        }
    }

    #[inline(always)]
    unsafe fn key_equal<K: MatrixKey>(left: &K, right: &K) -> bool {
        !unsafe { K::device_less(left, right) } && !unsafe { K::device_less(right, left) }
    }

    #[inline(always)]
    unsafe fn report_if(failed: bool, errors: &mut DisjointSlice<u32>) {
        if failed {
            unsafe { DeviceAtomicU32::from_ptr(errors.as_mut_ptr()) }
                .fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[kernel]
    pub unsafe fn fill_default_logical_warp(mut input: DisjointSlice<i32>, count: u32) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            *unsafe { input.get_unchecked_mut(index as usize) } = (count - 1 - index) as i32;
        }
    }

    /// Mirrors `catch2_test_nvrtc.cu`: omitting the logical-warp argument must
    /// instantiate the CUB-compatible 32-lane default.
    #[kernel]
    pub unsafe fn default_logical_warp(input: &[i32], mut output: DisjointSlice<i32>) {
        const ITEMS: usize = 4;
        let linear_tid = thread::threadIdx_x() as usize;
        let storage = unsafe {
            WarpMergeSortTempStorage::<i32, ITEMS>::from_raw_warp(
                DynamicSharedArray::<u8, 16>::get(),
                0,
            )
        };
        let warp_sort = WarpMergeSort::<i32, ITEMS>::new(storage);
        let offset = linear_tid * ITEMS;
        let mut thread_keys = [input[offset]; ITEMS];
        let mut item = 1usize;
        while item < ITEMS {
            thread_keys[item] = input[offset + item];
            item += 1;
        }
        unsafe { warp_sort.sort(&mut thread_keys, i32::cmp) };
        let mut item = 0usize;
        while item < ITEMS {
            *unsafe { output.get_unchecked_mut(offset + item) } = thread_keys[item];
            item += 1;
        }
    }

    #[kernel]
    pub unsafe fn validate_default_logical_warp(
        output: &[i32],
        count: u32,
        mut errors: DisjointSlice<u32>,
    ) {
        let index = thread::index_1d().get() as u32;
        if index < count {
            let failed = output[index as usize] != index as i32;
            unsafe { report_if(failed, &mut errors) };
        }
    }

    #[kernel]
    pub unsafe fn fill_keys<K, const ITEMS: usize, const LOGICAL: usize>(
        mut input: DisjointSlice<K>,
        mut segment_sizes: DisjointSlice<u32>,
        count: u32,
        action: u32,
    ) where
        K: MatrixKey,
    {
        let index = thread::index_1d().get() as usize;
        if index < count as usize {
            *unsafe { input.get_unchecked_mut(index) } = K::generate(index, KEY_SEED);
        }
        if index < TOTAL_WARPS {
            *unsafe { segment_sizes.get_unchecked_mut(index) } =
                expected_valid::<ITEMS, LOGICAL>(index, action);
        }
    }

    #[kernel]
    pub unsafe fn fill_pairs<K, V, const ITEMS: usize, const LOGICAL: usize>(
        mut keys: DisjointSlice<K>,
        mut values: DisjointSlice<V>,
        mut segment_sizes: DisjointSlice<u32>,
        count: u32,
        action: u32,
    ) where
        K: MatrixKey,
        V: MatrixValue,
    {
        let index = thread::index_1d().get() as usize;
        if index < count as usize {
            *unsafe { keys.get_unchecked_mut(index) } = K::generate(index, KEY_SEED);
            *unsafe { values.get_unchecked_mut(index) } = V::generate(index, VALUE_SEED);
        }
        if index < TOTAL_WARPS {
            *unsafe { segment_sizes.get_unchecked_mut(index) } =
                expected_valid::<ITEMS, LOGICAL>(index, action);
        }
    }

    #[kernel]
    pub unsafe fn keys<K, const ITEMS: usize, const LOGICAL: usize>(
        input: &[K],
        mut output: DisjointSlice<K>,
        segment_sizes: &[u32],
        action: u32,
    ) where
        K: MatrixKey,
    {
        let tid = thread::threadIdx_x() as usize;
        let warp_id = tid / LOGICAL;
        if warp_id >= TOTAL_WARPS {
            return;
        }

        let storage = unsafe {
            WarpMergeSortTempStorage::<K, ITEMS, LOGICAL>::from_raw_warp(
                DynamicSharedArray::<u8, 16>::get(),
                warp_id,
            )
        };
        let warp_sort = WarpMergeSort::<K, ITEMS, LOGICAL>::new(storage);
        let warp_items = ITEMS * LOGICAL;
        let warp_offset = warp_id * warp_items;
        let thread_offset = warp_offset + warp_sort.get_linear_tid() as usize * ITEMS;
        let valid_items = segment_sizes[warp_id];
        let mut thread_keys = [input[thread_offset]; ITEMS];
        let mut item = 1usize;
        while item < ITEMS {
            thread_keys[item] = input[thread_offset + item];
            item += 1;
        }

        match action {
            0 => unsafe { warp_sort.sort(&mut thread_keys, cub_order::<K>) },
            1 => unsafe { warp_sort.stable_sort(&mut thread_keys, cub_order::<K>) },
            2 => unsafe {
                warp_sort.sort_partial(&mut thread_keys, cub_order::<K>, valid_items, K::OOB)
            },
            3 => unsafe {
                warp_sort.stable_sort_partial(&mut thread_keys, cub_order::<K>, valid_items, K::OOB)
            },
            4 => unsafe {
                warp_sort.sort_partial_without_sentinel(
                    &mut thread_keys,
                    cub_order::<K>,
                    valid_items,
                )
            },
            _ => unsafe {
                warp_sort.stable_sort_partial_without_sentinel(
                    &mut thread_keys,
                    cub_order::<K>,
                    valid_items,
                )
            },
        }

        let mut item = 0usize;
        while item < ITEMS {
            let tile_item = warp_sort.get_linear_tid() as usize * ITEMS + item;
            *unsafe { output.get_unchecked_mut(thread_offset + item) } =
                if tile_item < valid_items as usize {
                    thread_keys[item]
                } else {
                    K::OOB
                };
            item += 1;
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "kernel macro emits a flat CUDA key-value ABI companion"
    )]
    #[kernel]
    pub unsafe fn pairs<K, V, const ITEMS: usize, const LOGICAL: usize>(
        keys_input: &[K],
        mut keys_output: DisjointSlice<K>,
        values_input: &[V],
        mut values_output: DisjointSlice<V>,
        segment_sizes: &[u32],
        action: u32,
    ) where
        K: MatrixKey,
        V: MatrixValue,
    {
        let tid = thread::threadIdx_x() as usize;
        let warp_id = tid / LOGICAL;
        if warp_id >= TOTAL_WARPS {
            return;
        }

        let storage = unsafe {
            WarpMergeSortTempStorage::<K, ITEMS, LOGICAL, V>::from_raw_warp(
                DynamicSharedArray::<u8, 16>::get(),
                warp_id,
            )
        };
        let warp_sort = WarpMergeSort::<K, ITEMS, LOGICAL, V>::new(storage);
        let warp_items = ITEMS * LOGICAL;
        let warp_offset = warp_id * warp_items;
        let thread_offset = warp_offset + warp_sort.get_linear_tid() as usize * ITEMS;
        let valid_items = segment_sizes[warp_id];
        let mut thread_keys = [keys_input[thread_offset]; ITEMS];
        let mut thread_values = [values_input[thread_offset]; ITEMS];
        let mut item = 1usize;
        while item < ITEMS {
            thread_keys[item] = keys_input[thread_offset + item];
            thread_values[item] = values_input[thread_offset + item];
            item += 1;
        }

        match action {
            0 => unsafe {
                warp_sort.sort_pairs(&mut thread_keys, &mut thread_values, cub_order::<K>)
            },
            1 => unsafe {
                warp_sort.stable_sort_pairs(&mut thread_keys, &mut thread_values, cub_order::<K>)
            },
            2 => unsafe {
                warp_sort.sort_pairs_partial(
                    &mut thread_keys,
                    &mut thread_values,
                    cub_order::<K>,
                    valid_items,
                    K::OOB,
                )
            },
            3 => unsafe {
                warp_sort.stable_sort_pairs_partial(
                    &mut thread_keys,
                    &mut thread_values,
                    cub_order::<K>,
                    valid_items,
                    K::OOB,
                )
            },
            4 => unsafe {
                warp_sort.sort_pairs_partial_without_sentinel(
                    &mut thread_keys,
                    &mut thread_values,
                    cub_order::<K>,
                    valid_items,
                )
            },
            _ => unsafe {
                warp_sort.stable_sort_pairs_partial_without_sentinel(
                    &mut thread_keys,
                    &mut thread_values,
                    cub_order::<K>,
                    valid_items,
                )
            },
        }

        let mut item = 0usize;
        while item < ITEMS {
            let tile_item = warp_sort.get_linear_tid() as usize * ITEMS + item;
            if tile_item < valid_items as usize {
                *unsafe { keys_output.get_unchecked_mut(thread_offset + item) } = thread_keys[item];
                *unsafe { values_output.get_unchecked_mut(thread_offset + item) } =
                    thread_values[item];
            } else {
                *unsafe { keys_output.get_unchecked_mut(thread_offset + item) } = K::OOB;
                *unsafe { values_output.get_unchecked_mut(thread_offset + item) } = V::ZERO;
            }
            item += 1;
        }
    }

    #[kernel]
    pub unsafe fn validate_keys<K, const ITEMS: usize, const LOGICAL: usize>(
        output: &[K],
        segment_sizes: &[u32],
        count: u32,
        action: u32,
        mut errors: DisjointSlice<u32>,
    ) where
        K: MatrixKey,
    {
        let index = thread::index_1d().get() as usize;
        if index >= count as usize {
            return;
        }

        let warp_items = ITEMS * LOGICAL;
        let warp = index / warp_items;
        let local = index - warp * warp_items;
        let warp_offset = warp * warp_items;
        let valid = segment_sizes[warp] as usize;
        let current = output[index];
        let mut failed = valid != expected_valid::<ITEMS, LOGICAL>(warp, action) as usize;

        if local >= valid {
            failed |= !unsafe { key_equal(&current, &K::OOB) };
        } else {
            if local > 0 {
                let previous = output[index - 1];
                failed |= unsafe { K::device_less(&current, &previous) };
            }

            // A value belongs exactly in [lower, lower + equal). Rebuilding
            // these counts from the deterministic device fixture proves both
            // sorted order and preservation of the complete input multiset.
            let mut lower = 0usize;
            let mut equal = 0usize;
            let mut candidate = 0usize;
            while candidate < valid {
                let generated = K::generate(warp_offset + candidate, KEY_SEED);
                if unsafe { K::device_less(&generated, &current) } {
                    lower += 1;
                } else if unsafe { key_equal(&generated, &current) } {
                    equal += 1;
                }
                candidate += 1;
            }
            failed |= local < lower || local >= lower + equal;
        }

        unsafe { report_if(failed, &mut errors) };
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "kernel macro emits a flat CUDA validation ABI companion"
    )]
    #[kernel]
    pub unsafe fn validate_pairs<K, V, const ITEMS: usize, const LOGICAL: usize>(
        keys: &[K],
        values: &[V],
        segment_sizes: &[u32],
        count: u32,
        action: u32,
        mut errors: DisjointSlice<u32>,
    ) where
        K: MatrixKey,
        V: MatrixValue,
    {
        let index = thread::index_1d().get() as usize;
        if index >= count as usize {
            return;
        }

        let warp_items = ITEMS * LOGICAL;
        let warp = index / warp_items;
        let local = index - warp * warp_items;
        let warp_offset = warp * warp_items;
        let valid = segment_sizes[warp] as usize;
        let current_key = keys[index];
        let current_value = values[index];
        let mut failed = valid != expected_valid::<ITEMS, LOGICAL>(warp, action) as usize;

        if local >= valid {
            failed |= !unsafe { key_equal(&current_key, &K::OOB) };
            failed |= !unsafe { V::device_equal(&current_value, &V::ZERO) };
        } else {
            if local > 0 {
                let previous = keys[index - 1];
                failed |= unsafe { K::device_less(&current_key, &previous) };
            }

            let mut lower = 0usize;
            let mut equal = 0usize;
            let mut candidate = 0usize;
            while candidate < valid {
                let generated = K::generate(warp_offset + candidate, KEY_SEED);
                if unsafe { K::device_less(&generated, &current_key) } {
                    lower += 1;
                } else if unsafe { key_equal(&generated, &current_key) } {
                    equal += 1;
                }
                candidate += 1;
            }

            if local < lower || local >= lower + equal {
                failed = true;
            } else {
                // CCCL's host reference is a stable sort even for the Sort
                // entry point. Select the corresponding equal-key source in
                // original order and check its value without a host reference.
                let target_equal = local - lower;
                let mut seen_equal = 0usize;
                let mut found = false;
                let mut source = 0usize;
                while source < valid {
                    let source_key = K::generate(warp_offset + source, KEY_SEED);
                    if unsafe { key_equal(&source_key, &current_key) } {
                        if seen_equal == target_equal {
                            let expected_value = V::generate(warp_offset + source, VALUE_SEED);
                            failed |= !unsafe { V::device_equal(&current_value, &expected_value) };
                            found = true;
                            break;
                        }
                        seen_equal += 1;
                    }
                    source += 1;
                }
                failed |= !found;
            }
        }

        unsafe { report_if(failed, &mut errors) };
    }
}

fn launch_config<K, V, const ITEMS: usize, const LOGICAL: usize>() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: ((TOTAL_WARPS * LOGICAL) as u32, 1, 1),
        shared_mem_bytes: (TOTAL_WARPS
            * warp_merge_sort_temp_storage_bytes::<K, V, ITEMS, LOGICAL>())
            as u32,
    }
}

fn read_failure_count(
    failures: &DeviceBuffer<u32>,
    stream: &CudaStream,
) -> Result<u32, Box<dyn Error>> {
    let mut host_failure = 0u32;
    failures.copy_to_host(stream, std::slice::from_mut(&mut host_failure))?;
    Ok(host_failure)
}

fn run_default_logical_warp_smoke(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
) -> Result<(), Box<dyn Error>> {
    const ITEMS: usize = 4;
    const LOGICAL: usize = 32;
    const COUNT: usize = ITEMS * LOGICAL;
    let mut input = DeviceBuffer::<i32>::zeroed(stream, COUNT)?;
    let mut output = DeviceBuffer::<i32>::zeroed(stream, COUNT)?;
    let mut failures = DeviceBuffer::<u32>::zeroed(stream, 1)?;

    unsafe {
        module.fill_default_logical_warp(
            stream,
            LaunchConfig::for_num_elems(COUNT as u32),
            &mut input,
            COUNT as u32,
        )?;
        module.default_logical_warp(
            stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (LOGICAL as u32, 1, 1),
                shared_mem_bytes: warp_merge_sort_temp_storage_bytes::<i32, (), ITEMS, LOGICAL>()
                    as u32,
            },
            &input,
            &mut output,
        )?;
        module.validate_default_logical_warp(
            stream,
            LaunchConfig::for_num_elems(COUNT as u32),
            &output,
            COUNT as u32,
            &mut failures,
        )?;
    }

    let failure_count = read_failure_count(&failures, stream)?;
    if failure_count != 0 {
        return Err(format!("default logical-warp smoke found {failure_count} errors").into());
    }
    println!("PASS CUB WarpMergeSort default logical warp (NVRTC instantiation)");
    Ok(())
}

fn run_keys<K: MatrixKey, const ITEMS: usize, const LOGICAL: usize>(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
) -> Result<(usize, usize), Box<dyn Error>> {
    let count = TOTAL_WARPS * ITEMS * LOGICAL;
    let mut input = DeviceBuffer::<K>::zeroed(stream, count)?;
    let mut output = DeviceBuffer::<K>::zeroed(stream, count)?;
    let mut valid = DeviceBuffer::<u32>::zeroed(stream, TOTAL_WARPS)?;

    for action in 0..6u32 {
        let mut failures = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            module.fill_keys::<K, ITEMS, LOGICAL>(
                stream,
                LaunchConfig::for_num_elems(count as u32),
                &mut input,
                &mut valid,
                count as u32,
                action,
            )?;
            module.keys::<K, ITEMS, LOGICAL>(
                stream,
                launch_config::<K, (), ITEMS, LOGICAL>(),
                &input,
                &mut output,
                &valid,
                action,
            )?;
            module.validate_keys::<K, ITEMS, LOGICAL>(
                stream,
                LaunchConfig::for_num_elems(count as u32),
                &output,
                &valid,
                count as u32,
                action,
                &mut failures,
            )?;
        }

        let failure_count = read_failure_count(&failures, stream)?;
        if failure_count != 0 {
            return Err(format!(
                "CUB warp keys {}-i{ITEMS}-w{LOGICAL} action {action} found {failure_count} errors",
                K::LABEL
            )
            .into());
        }
    }

    println!("PASS CUB warp keys {}-i{ITEMS}-w{LOGICAL}", K::LABEL);
    Ok((4, 2))
}

fn run_pairs<K: MatrixKey, V: MatrixValue, const ITEMS: usize, const LOGICAL: usize>(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
) -> Result<(usize, usize), Box<dyn Error>> {
    let count = TOTAL_WARPS * ITEMS * LOGICAL;
    let mut keys_input = DeviceBuffer::<K>::zeroed(stream, count)?;
    let mut keys_output = DeviceBuffer::<K>::zeroed(stream, count)?;
    let mut values_input = DeviceBuffer::<V>::zeroed(stream, count)?;
    let mut values_output = DeviceBuffer::<V>::zeroed(stream, count)?;
    let mut valid = DeviceBuffer::<u32>::zeroed(stream, TOTAL_WARPS)?;

    for action in 0..6u32 {
        let mut failures = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            module.fill_pairs::<K, V, ITEMS, LOGICAL>(
                stream,
                LaunchConfig::for_num_elems(count as u32),
                &mut keys_input,
                &mut values_input,
                &mut valid,
                count as u32,
                action,
            )?;
            module.pairs::<K, V, ITEMS, LOGICAL>(
                stream,
                launch_config::<K, V, ITEMS, LOGICAL>(),
                &keys_input,
                &mut keys_output,
                &values_input,
                &mut values_output,
                &valid,
                action,
            )?;
            module.validate_pairs::<K, V, ITEMS, LOGICAL>(
                stream,
                LaunchConfig::for_num_elems(count as u32),
                &keys_output,
                &values_output,
                &valid,
                count as u32,
                action,
                &mut failures,
            )?;
        }

        let failure_count = read_failure_count(&failures, stream)?;
        if failure_count != 0 {
            return Err(format!(
                "CUB warp pairs {}/{}-i{ITEMS}-w{LOGICAL} action {action} found {failure_count} errors",
                K::LABEL,
                V::LABEL
            )
            .into());
        }
    }

    println!(
        "PASS CUB warp pairs {}/{}-i{ITEMS}-w{LOGICAL}",
        K::LABEL,
        V::LABEL
    );
    Ok((4, 2))
}

macro_rules! run_geometries {
    ($run:ident, $module:expr, $stream:expr, $($types:ty),+ $(,)?) => {{
        let mut upstream = 0usize;
        let mut extensions = 0usize;
        $(
            for result in [
                $run::<$types, 1, 32>($module, $stream)?,
                $run::<$types, 1, 4>($module, $stream)?,
                $run::<$types, 4, 32>($module, $stream)?,
                $run::<$types, 4, 4>($module, $stream)?,
                $run::<$types, 7, 32>($module, $stream)?,
                $run::<$types, 7, 4>($module, $stream)?,
            ] {
                upstream += result.0;
                extensions += result.1;
            }
        )+
        (upstream, extensions)
    }};
}

macro_rules! run_pair_geometries {
    ($module:expr, $stream:expr, $($key:ty => [$($value:ty),+]),+ $(,)?) => {{
        let mut upstream = 0usize;
        let mut extensions = 0usize;
        $(
            $(
                for result in [
                    run_pairs::<$key, $value, 1, 32>($module, $stream)?,
                    run_pairs::<$key, $value, 1, 4>($module, $stream)?,
                    run_pairs::<$key, $value, 4, 32>($module, $stream)?,
                    run_pairs::<$key, $value, 4, 4>($module, $stream)?,
                    run_pairs::<$key, $value, 7, 32>($module, $stream)?,
                    run_pairs::<$key, $value, 7, 4>($module, $stream)?,
                ] {
                    upstream += result.0;
                    extensions += result.1;
                }
            )+
        )+
        (upstream, extensions)
    }};
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;

    run_default_logical_warp_smoke(&module, &stream)?;

    let keys = run_geometries!(run_keys, &module, &stream, u8, i32, i64, CubCustom);
    let pairs = run_pair_geometries!(
        &module,
        &stream,
        u8 => [i32, CubCustom],
        i32 => [i32, CubCustom],
        i64 => [i32, CubCustom],
        CubCustom => [i32, CubCustom],
    );
    let upstream = keys.0 + pairs.0;
    let extensions = keys.1 + pairs.1;
    assert_eq!(upstream, 288);
    assert_eq!(extensions, 144);
    println!(
        "CUB WarpMergeSort compatibility matrix: PASS ({upstream} upstream + {extensions} sentinel-free cases)"
    );
    Ok(())
}
