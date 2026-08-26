//! CUB-style device-wide merge backend for the Thrust-shaped facade.
//!
//! The implementation follows CCCL's DeviceMerge structure: a Merge Path
//! partition kernel computes one intersection per tile, then one block owns
//! each independent tile and performs a register-blocked serial merge.  The
//! host dispatch allocates the small partition array for one call. Compile
//! target policy is passed as literal const arguments; no runtime policy record
//! is retained.

use crate::device::{DeviceRange, DeviceRangeMut};
use core::cmp::Ordering;
use cuda_core::{CudaStream, DeviceCopy, DriverError, LaunchConfig};
use rccl_memory::{
    CudaAsyncMemoryResource, DeviceAllocator, DeviceBuffer as ResourceBuffer, DeviceSlice,
    DeviceSliceMut, MemoryError,
};
use std::{error::Error, fmt, marker::PhantomData, mem::size_of, sync::Arc};

const MERGE_PORTABLE_SHARED_BYTES: usize = 48 * 1024;

#[inline(always)]
fn natural_order<K: PartialOrd>(left: &K, right: &K) -> Ordering {
    if left < right {
        Ordering::Less
    } else if right < left {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// CUB's `nominal_4B_items_to_items`: floor-scale a tuning point by the
/// combined record size and clamp it to `1..=nominal`.
pub const fn nominal_4b_items_to_items(nominal: u32, combined_bytes: usize) -> u32 {
    if combined_bytes == 0 {
        return nominal;
    }
    let scaled = (nominal as usize * 4) / combined_bytes;
    if scaled == 0 {
        1
    } else if scaled > nominal as usize {
        nominal
    } else {
        scaled as u32
    }
}

const fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment != 0 && alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

pub const fn merge_shared_bytes(
    block_threads: u32,
    items_per_thread: u32,
    key_bytes: usize,
    value_bytes: usize,
) -> usize {
    let tile = block_threads as usize * items_per_thread as usize;
    let keys = tile.saturating_mul(key_bytes);
    if value_bytes == 0 {
        keys
    } else {
        // Device allocations and the dynamic shared base are 256-byte aligned;
        // using that boundary also covers every DeviceCopy layout accepted by
        // rccl-memory.
        align_up(keys, 256).saturating_add(tile.saturating_mul(value_bytes))
    }
}

#[inline(always)]
const fn pair_items_per_thread<const NOMINAL_4B_ITEMS: u32, K, V>() -> u32 {
    nominal_4b_items_to_items(
        NOMINAL_4B_ITEMS,
        size_of::<K>().saturating_add(size_of::<V>()),
    )
}

#[inline(always)]
const fn key_items_per_thread<const NOMINAL_4B_ITEMS: u32, K>() -> u32 {
    nominal_4b_items_to_items(NOMINAL_4B_ITEMS, size_of::<K>())
}

#[inline(always)]
const fn key_shared_bytes<const BLOCK_THREADS: u32, const NOMINAL_4B_ITEMS: u32, K>() -> usize {
    merge_shared_bytes(
        BLOCK_THREADS,
        key_items_per_thread::<NOMINAL_4B_ITEMS, K>(),
        size_of::<K>(),
        0,
    )
}

#[inline(always)]
const fn pair_shared_bytes<const BLOCK_THREADS: u32, const NOMINAL_4B_ITEMS: u32, K, V>() -> usize {
    merge_shared_bytes(
        BLOCK_THREADS,
        pair_items_per_thread::<NOMINAL_4B_ITEMS, K, V>(),
        size_of::<K>(),
        size_of::<V>(),
    )
}

#[cfg(test)]
#[inline(always)]
const fn virtual_shared_bytes(shared_bytes: usize) -> usize {
    if shared_bytes > MERGE_PORTABLE_SHARED_BYTES {
        align_up(shared_bytes, 128)
    } else {
        0
    }
}

/// Compile-time CUB agent selection for a key-only merge. The target schedule
/// provides the literals; the key shape becomes concrete at monomorphization.
#[inline(always)]
const fn merge_key_uses_fallback<
    const NORMAL_BLOCK_THREADS: u32,
    const NORMAL_ITEMS: u32,
    const FALLBACK_BLOCK_THREADS: u32,
    K,
>() -> bool {
    key_shared_bytes::<NORMAL_BLOCK_THREADS, NORMAL_ITEMS, K>() > MERGE_PORTABLE_SHARED_BYTES
        && key_shared_bytes::<FALLBACK_BLOCK_THREADS, 1, K>() <= MERGE_PORTABLE_SHARED_BYTES
}

/// Compile-time CUB agent selection for a key/value merge.
#[inline(always)]
const fn merge_pair_uses_fallback<
    const NORMAL_BLOCK_THREADS: u32,
    const NORMAL_ITEMS: u32,
    const FALLBACK_BLOCK_THREADS: u32,
    K,
    V,
>() -> bool {
    pair_shared_bytes::<NORMAL_BLOCK_THREADS, NORMAL_ITEMS, K, V>() > MERGE_PORTABLE_SHARED_BYTES
        && pair_shared_bytes::<FALLBACK_BLOCK_THREADS, 1, K, V>() <= MERGE_PORTABLE_SHARED_BYTES
}

/// Canonical order values shared with the sorting facade.
/// Device-callable random-access input.  The concrete iterator type is a
/// kernel parameter, so contiguous pointers, counting iterators, and mapped
/// iterators remain separate compile-time specializations just as in Thrust.
///
/// # Safety
///
/// Every index submitted by the merge engine must be readable for the complete
/// kernel launch, and `read` must be valid device code.
pub(crate) unsafe trait MergeDeviceInput<T: DeviceCopy>:
    DeviceCopy + 'static
{
    const DIRECT_REFERENCE: bool = false;

    unsafe fn read(&self, index: u64) -> T;

    unsafe fn pointer(&self, _index: u64) -> *const T {
        core::ptr::null()
    }

    unsafe fn copy_to(&self, index: u64, destination: *mut T) {
        unsafe { destination.write(self.read(index)) };
    }
}

/// Device-callable random-access output.
///
/// # Safety
///
/// Every logical output index must identify a distinct, live device element.
pub(crate) unsafe trait MergeDeviceOutput<T: DeviceCopy>:
    DeviceCopy + 'static
{
    unsafe fn write(&self, index: u64, value: T);

    unsafe fn write_from(&self, index: u64, source: *const T) {
        unsafe { self.write(index, source.read()) };
    }
}

unsafe impl<T: DeviceCopy + 'static> MergeDeviceInput<T> for DeviceRange<T> {
    const DIRECT_REFERENCE: bool = true;

    #[inline(always)]
    unsafe fn read(&self, index: u64) -> T {
        unsafe { (*self).read(index) }
    }

    #[inline(always)]
    unsafe fn pointer(&self, index: u64) -> *const T {
        unsafe { self.as_ptr().add(index as usize) }
    }

    #[inline(always)]
    unsafe fn copy_to(&self, index: u64, destination: *mut T) {
        unsafe { self.pointer(index).copy_to_nonoverlapping(destination, 1) };
    }
}

unsafe impl<T: DeviceCopy + 'static> MergeDeviceOutput<T> for DeviceRangeMut<T> {
    #[inline(always)]
    unsafe fn write(&self, index: u64, value: T) {
        unsafe { (*self).write(index, value) };
    }

    #[inline(always)]
    unsafe fn write_from(&self, index: u64, source: *const T) {
        unsafe { source.copy_to_nonoverlapping(self.as_mut_ptr().add(index as usize), 1) };
    }
}
#[derive(Debug)]
#[non_exhaustive]
pub enum MergeError {
    Driver(DriverError),
    Memory(MemoryError),
    Module(cuda_host::EmbeddedModuleError),
    LengthOverflow,
    OutputTooSmall { required: usize, actual: usize },
    WorkspaceTooSmall { required: usize, actual: usize },
    LengthMismatch { keys: usize, values: usize },
    InputOutputOverlap,
    ZeroSizedKey,
    ZeroSizedValue,
}

impl fmt::Display for MergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => write!(f, "CUDA driver error: {error}"),
            Self::Memory(error) => write!(f, "CUDA memory error: {error}"),
            Self::Module(error) => write!(f, "embedded CUDA module error: {error}"),
            Self::LengthOverflow => write!(f, "the combined merge length overflows usize/u64"),
            Self::OutputTooSmall { required, actual } => write!(
                f,
                "merge output has {actual} elements but {required} are required"
            ),
            Self::WorkspaceTooSmall { required, actual } => write!(
                f,
                "merge workspace has {actual} partition slots but {required} are required"
            ),
            Self::LengthMismatch { keys, values } => {
                write!(
                    f,
                    "merge key/value lengths differ: {keys} keys, {values} values"
                )
            }
            Self::InputOutputOverlap => {
                write!(f, "merge output must not overlap either input range")
            }
            Self::ZeroSizedKey => write!(f, "device merge does not accept zero-sized keys"),
            Self::ZeroSizedValue => write!(f, "device merge does not accept zero-sized values"),
        }
    }
}

impl Error for MergeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Memory(error) => Some(error),
            Self::Module(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DriverError> for MergeError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

impl From<MemoryError> for MergeError {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

impl From<cuda_host::EmbeddedModuleError> for MergeError {
    fn from(value: cuda_host::EmbeddedModuleError) -> Self {
        Self::Module(value)
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "kernel macro emits flat CUDA ABI companion functions inside this module"
)]
#[cuda_host::cuda_module]
mod kernels {
    use super::*;
    use core::mem::MaybeUninit;
    use cuda_device::{DynamicSharedArray, kernel, thread};

    // The exact absolute marker call is intentional: `#[kernel]` forwards it
    // to each generic entry without the u32-to-usize evaluatability witness
    // introduced by cuda-oxide's `#[launch_bounds]` convenience attribute.

    #[inline(always)]
    unsafe fn merge_path_inputs<C, K, I1, I2>(
        first: &I1,
        second: &I2,
        first_len: u64,
        second_len: u64,
        diagonal: u64,
        order: &C,
    ) -> u64
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        I1: MergeDeviceInput<K>,
        I2: MergeDeviceInput<K>,
    {
        let mut begin = diagonal.saturating_sub(second_len);
        let mut end = diagonal.min(first_len);
        while begin < end {
            let middle = begin + (end - begin) / 2;
            let second_before_first = if I1::DIRECT_REFERENCE && I2::DIRECT_REFERENCE {
                order(unsafe { &*second.pointer(diagonal - 1 - middle) }, unsafe {
                    &*first.pointer(middle)
                })
                .is_lt()
            } else {
                let first_key = unsafe { first.read(middle) };
                let second_key = unsafe { second.read(diagonal - 1 - middle) };
                order(&second_key, &first_key).is_lt()
            };
            if second_before_first {
                end = middle;
            } else {
                begin = middle + 1;
            }
        }
        begin
    }

    #[kernel]
    pub unsafe fn partition_keys<C, K, I1, I2>(
        first: I1,
        first_len: u64,
        second: I2,
        second_len: u64,
        tile_items: u64,
        partition_count: u64,
        partitions: DeviceRangeMut<u64>,
        order: C,
    ) where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        I1: MergeDeviceInput<K>,
        I2: MergeDeviceInput<K>,
    {
        let partition = thread::index_1d().get() as u64;
        if partition < partition_count {
            let diagonal = (partition * tile_items).min(first_len + second_len);
            let offset = unsafe {
                merge_path_inputs(&first, &second, first_len, second_len, diagonal, &order)
            };
            unsafe { partitions.write(partition, offset) };
        }
    }

    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "device helper mirrors the flat merge kernel ABI"
    )]
    unsafe fn merge_tile_keys<
        const BLOCK_THREADS: u32,
        const NOMINAL_4B_ITEMS: u32,
        const UNROLL: u32,
        C,
        K,
        I1,
        I2,
        O,
    >(
        first: I1,
        first_len: u64,
        second: I2,
        second_len: u64,
        output: O,
        partitions: DeviceRange<u64>,
        virtual_keys: DeviceRangeMut<K>,
        order: &C,
    ) where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        I1: MergeDeviceInput<K>,
        I2: MergeDeviceInput<K>,
        O: MergeDeviceOutput<K>,
    {
        let tid = thread::threadIdx_x();
        let tile_index = thread::blockIdx_x() as u64;
        let items_per_thread = key_items_per_thread::<NOMINAL_4B_ITEMS, K>();
        let tile_items = BLOCK_THREADS * items_per_thread;
        let tile_begin = tile_index * tile_items as u64;
        let total = first_len + second_len;
        let tile_end = (tile_begin + tile_items as u64).min(total);
        let tile_count = (tile_end - tile_begin) as u32;

        let first_begin = unsafe { partitions.read(tile_index) };
        let first_end = unsafe { partitions.read(tile_index + 1) };
        let second_begin = tile_begin - first_begin;
        let second_end = tile_end - first_end;
        let first_count = (first_end - first_begin) as u32;
        let second_count = (second_end - second_begin) as u32;

        let required_shared =
            merge_shared_bytes(BLOCK_THREADS, items_per_thread, size_of::<K>(), 0);
        let shared = if required_shared > MERGE_PORTABLE_SHARED_BYTES {
            unsafe {
                virtual_keys
                    .as_mut_ptr()
                    .add(tile_index as usize * tile_items as usize)
            }
        } else {
            DynamicSharedArray::<K, 256>::get()
        };
        let mut slot = tid;
        while slot < first_count {
            unsafe { first.copy_to(first_begin + slot as u64, shared.add(slot as usize)) };
            slot += BLOCK_THREADS;
        }
        slot = tid;
        while slot < second_count {
            unsafe {
                second.copy_to(
                    second_begin + slot as u64,
                    shared.add((first_count + slot) as usize),
                )
            };
            slot += BLOCK_THREADS;
        }
        thread::sync_threads();

        let diagonal = (tid * items_per_thread).min(tile_count);
        let mut first_local = unsafe {
            merge_path_inputs(
                &DeviceRange::from_raw_parts(shared, first_count as usize),
                &DeviceRange::from_raw_parts(
                    shared.add(first_count as usize),
                    second_count as usize,
                ),
                first_count as u64,
                second_count as u64,
                diagonal as u64,
                order,
            ) as u32
        };
        let mut second_local = diagonal - first_local;
        let valid = items_per_thread.min(tile_count - diagonal);

        if items_per_thread == 1 {
            let take_second = second_local < second_count
                && (first_local >= first_count
                    || order(
                        unsafe { &*shared.add((first_count + second_local) as usize) },
                        unsafe { &*shared.add(first_local as usize) },
                    )
                    .is_lt());
            let source = if take_second {
                first_count + second_local
            } else {
                first_local
            };
            thread::sync_threads();
            if tid < tile_count {
                unsafe { output.write_from(tile_begin + tid as u64, shared.add(source as usize)) };
            }
            return;
        }

        // Spell the bounded serial merge as scalar slots. The record-size
        // scaling expression becomes constant when K is monomorphized, so
        // LLVM removes every slot above `items_per_thread`. Unlike a 1..=15
        // host dispatch, this produces one GPU specialization for the normal
        // geometry instead of collecting fifteen kernel entry points.
        macro_rules! merge_scalar_slots {
            (
                slots [$(($local:ident, $index:literal)),+ $(,)?],
                last ($last_local:ident, $last_index:literal) $(,)?
            ) => {{
                $(let mut $local = MaybeUninit::<K>::uninit();)+
                let mut $last_local = MaybeUninit::<K>::uninit();
                let mut item = 0u32;
                while item < valid {
                    cuda_device::thread::__unroll_config::<UNROLL>();
                    let take_second = second_local < second_count
                        && (first_local >= first_count
                            || order(
                                unsafe { &*shared.add((first_count + second_local) as usize) },
                                unsafe { &*shared.add(first_local as usize) },
                            )
                            .is_lt());
                    let key = if take_second {
                        let value = unsafe {
                            shared.add((first_count + second_local) as usize).read()
                        };
                        second_local += 1;
                        value
                    } else {
                        let value = unsafe { shared.add(first_local as usize).read() };
                        first_local += 1;
                        value
                    };
                    match item {
                        $($index => { $local.write(key); },)+
                        $last_index => { $last_local.write(key); },
                        _ => { $last_local.write(key); },
                    }
                    item += 1;
                }
                thread::sync_threads();
                item = 0;
                while item < valid {
                    cuda_device::thread::__unroll_config::<UNROLL>();
                    let key = match item {
                        $($index => unsafe { $local.assume_init() },)+
                        $last_index => unsafe { $last_local.assume_init() },
                        _ => unsafe { $last_local.assume_init() },
                    };
                    unsafe { shared.add((diagonal + item) as usize).write(key) };
                    item += 1;
                }
                thread::sync_threads();
            }};
        }
        merge_scalar_slots!(
            slots [
                (local_0, 0u32),
                (local_1, 1u32),
                (local_2, 2u32),
                (local_3, 3u32),
                (local_4, 4u32),
                (local_5, 5u32),
                (local_6, 6u32),
                (local_7, 7u32),
                (local_8, 8u32),
                (local_9, 9u32),
                (local_10, 10u32),
                (local_11, 11u32),
                (local_12, 12u32),
                (local_13, 13u32),
            ],
            last (local_14, 14u32),
        );

        slot = tid;
        while slot < tile_count {
            unsafe { output.write_from(tile_begin + slot as u64, shared.add(slot as usize)) };
            slot += BLOCK_THREADS;
        }
    }

    #[kernel]
    pub unsafe fn merge_keys<
        const BLOCK_THREADS: u32,
        const NOMINAL_4B_ITEMS: u32,
        const UNROLL: u32,
        C,
        K,
        I1,
        I2,
        O,
    >(
        first: I1,
        first_len: u64,
        second: I2,
        second_len: u64,
        output: O,
        partitions: DeviceRange<u64>,
        virtual_keys: DeviceRangeMut<K>,
        order: C,
    ) where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        I1: MergeDeviceInput<K>,
        I2: MergeDeviceInput<K>,
        O: MergeDeviceOutput<K>,
    {
        ::cuda_device::thread::__launch_bounds_config::<BLOCK_THREADS, 0>();
        unsafe {
            merge_tile_keys::<BLOCK_THREADS, NOMINAL_4B_ITEMS, UNROLL, C, K, I1, I2, O>(
                first,
                first_len,
                second,
                second_len,
                output,
                partitions,
                virtual_keys,
                &order,
            )
        };
    }

    #[kernel]
    pub unsafe fn merge_pairs<
        const BLOCK_THREADS: u32,
        const NOMINAL_4B_ITEMS: u32,
        const UNROLL: u32,
        C,
        K,
        V,
        K1,
        V1,
        K2,
        V2,
        KO,
        VO,
    >(
        first_keys: K1,
        first_values: V1,
        first_len: u64,
        second_keys: K2,
        second_values: V2,
        second_len: u64,
        output_keys: KO,
        output_values: VO,
        partitions: DeviceRange<u64>,
        virtual_keys: DeviceRangeMut<K>,
        virtual_values: DeviceRangeMut<V>,
        order: C,
    ) where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        V: DeviceCopy + 'static,
        K1: MergeDeviceInput<K>,
        V1: MergeDeviceInput<V>,
        K2: MergeDeviceInput<K>,
        V2: MergeDeviceInput<V>,
        KO: MergeDeviceOutput<K>,
        VO: MergeDeviceOutput<V>,
    {
        ::cuda_device::thread::__launch_bounds_config::<BLOCK_THREADS, 0>();
        let tid = thread::threadIdx_x();
        let tile_index = thread::blockIdx_x() as u64;
        let items_per_thread = pair_items_per_thread::<NOMINAL_4B_ITEMS, K, V>();
        let tile_items = BLOCK_THREADS * items_per_thread;
        let tile_begin = tile_index * tile_items as u64;
        let total = first_len + second_len;
        let tile_end = (tile_begin + tile_items as u64).min(total);
        let tile_count = (tile_end - tile_begin) as u32;

        let first_begin = unsafe { partitions.read(tile_index) };
        let first_end = unsafe { partitions.read(tile_index + 1) };
        let second_begin = tile_begin - first_begin;
        let second_end = tile_end - first_end;
        let first_count = (first_end - first_begin) as u32;
        let second_count = (second_end - second_begin) as u32;

        let values_offset = align_up(tile_items as usize * size_of::<K>(), 256);
        let required_shared = values_offset + tile_items as usize * size_of::<V>();
        let keys_shared = if required_shared > MERGE_PORTABLE_SHARED_BYTES {
            unsafe {
                virtual_keys
                    .as_mut_ptr()
                    .add(tile_index as usize * tile_items as usize)
            }
        } else {
            DynamicSharedArray::<K, 256>::get()
        };
        let values_shared = if required_shared > MERGE_PORTABLE_SHARED_BYTES {
            unsafe {
                virtual_values
                    .as_mut_ptr()
                    .add(tile_index as usize * tile_items as usize)
            }
        } else {
            DynamicSharedArray::<V, 256>::offset(values_offset)
        };
        let mut slot = tid;
        while slot < first_count {
            unsafe {
                first_keys.copy_to(first_begin + slot as u64, keys_shared.add(slot as usize));
                first_values.copy_to(first_begin + slot as u64, values_shared.add(slot as usize));
            }
            slot += BLOCK_THREADS;
        }
        slot = tid;
        while slot < second_count {
            unsafe {
                second_keys.copy_to(
                    second_begin + slot as u64,
                    keys_shared.add((first_count + slot) as usize),
                );
                second_values.copy_to(
                    second_begin + slot as u64,
                    values_shared.add((first_count + slot) as usize),
                );
            }
            slot += BLOCK_THREADS;
        }
        thread::sync_threads();

        let diagonal = (tid * items_per_thread).min(tile_count);
        let mut first_local = unsafe {
            merge_path_inputs(
                &DeviceRange::from_raw_parts(keys_shared, first_count as usize),
                &DeviceRange::from_raw_parts(
                    keys_shared.add(first_count as usize),
                    second_count as usize,
                ),
                first_count as u64,
                second_count as u64,
                diagonal as u64,
                &order,
            ) as u32
        };
        let mut second_local = diagonal - first_local;
        let valid = items_per_thread.min(tile_count - diagonal);

        if items_per_thread == 1 {
            let take_second = second_local < second_count
                && (first_local >= first_count
                    || order(
                        unsafe { &*keys_shared.add((first_count + second_local) as usize) },
                        unsafe { &*keys_shared.add(first_local as usize) },
                    )
                    .is_lt());
            let source = if take_second {
                first_count + second_local
            } else {
                first_local
            };
            thread::sync_threads();
            if tid < tile_count {
                unsafe {
                    output_keys
                        .write_from(tile_begin + tid as u64, keys_shared.add(source as usize));
                    output_values
                        .write_from(tile_begin + tid as u64, values_shared.add(source as usize));
                }
            }
            return;
        }

        macro_rules! merge_scalar_pair_slots {
            (
                slots [$(($local_key:ident, $local_value:ident, $index:literal)),+ $(,)?],
                last ($last_key:ident, $last_value:ident, $last_index:literal) $(,)?
            ) => {{
                $(
                    let mut $local_key = MaybeUninit::<K>::uninit();
                    let mut $local_value = MaybeUninit::<V>::uninit();
                )+
                let mut $last_key = MaybeUninit::<K>::uninit();
                let mut $last_value = MaybeUninit::<V>::uninit();
                let mut item = 0u32;
                while item < valid {
                    cuda_device::thread::__unroll_config::<UNROLL>();
                    let take_second = second_local < second_count
                        && (first_local >= first_count
                            || order(
                                unsafe {
                                    &*keys_shared.add((first_count + second_local) as usize)
                                },
                                unsafe { &*keys_shared.add(first_local as usize) },
                            )
                            .is_lt());
                    let index = if take_second {
                        let index = first_count + second_local;
                        second_local += 1;
                        index
                    } else {
                        let index = first_local;
                        first_local += 1;
                        index
                    };
                    let key = unsafe { keys_shared.add(index as usize).read() };
                    let value = unsafe { values_shared.add(index as usize).read() };
                    match item {
                        $(
                            $index => {
                                $local_key.write(key);
                                $local_value.write(value);
                            },
                        )+
                        $last_index => {
                            $last_key.write(key);
                            $last_value.write(value);
                        },
                        _ => {
                            $last_key.write(key);
                            $last_value.write(value);
                        },
                    }
                    item += 1;
                }
                thread::sync_threads();
                item = 0;
                while item < valid {
                    cuda_device::thread::__unroll_config::<UNROLL>();
                    let key = match item {
                        $($index => unsafe { $local_key.assume_init() },)+
                        $last_index => unsafe { $last_key.assume_init() },
                        _ => unsafe { $last_key.assume_init() },
                    };
                    unsafe { keys_shared.add((diagonal + item) as usize).write(key) };
                    item += 1;
                }
                thread::sync_threads();
                slot = tid;
                while slot < tile_count {
                    unsafe {
                        output_keys.write_from(
                            tile_begin + slot as u64,
                            keys_shared.add(slot as usize),
                        )
                    };
                    slot += BLOCK_THREADS;
                }
                thread::sync_threads();
                item = 0;
                while item < valid {
                    cuda_device::thread::__unroll_config::<UNROLL>();
                    let value = match item {
                        $($index => unsafe { $local_value.assume_init() },)+
                        $last_index => unsafe { $last_value.assume_init() },
                        _ => unsafe { $last_value.assume_init() },
                    };
                    unsafe {
                        values_shared
                            .add((diagonal + item) as usize)
                            .write(value)
                    };
                    item += 1;
                }
                thread::sync_threads();
                slot = tid;
                while slot < tile_count {
                    unsafe {
                        output_values.write_from(
                            tile_begin + slot as u64,
                            values_shared.add(slot as usize),
                        )
                    };
                    slot += BLOCK_THREADS;
                }
            }};
        }
        merge_scalar_pair_slots!(
            slots [
                (local_key_0, local_value_0, 0u32),
                (local_key_1, local_value_1, 1u32),
                (local_key_2, local_value_2, 2u32),
                (local_key_3, local_value_3, 3u32),
                (local_key_4, local_value_4, 4u32),
                (local_key_5, local_value_5, 5u32),
                (local_key_6, local_value_6, 6u32),
                (local_key_7, local_value_7, 7u32),
                (local_key_8, local_value_8, 8u32),
                (local_key_9, local_value_9, 9u32),
                (local_key_10, local_value_10, 10u32),
                (local_key_11, local_value_11, 11u32),
                (local_key_12, local_value_12, 12u32),
                (local_key_13, local_value_13, 13u32),
            ],
            last (local_key_14, local_value_14, 14u32),
        );
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "host launch helper must forward the complete CUDA merge ABI"
)]
unsafe fn launch_merge_keys<
    const BLOCK_THREADS: u32,
    const NOMINAL_4B_ITEMS: u32,
    const UNROLL: u32,
    C,
    K,
    I1,
    I2,
    O,
>(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    first: I1,
    first_len: u64,
    second: I2,
    second_len: u64,
    output: O,
    partitions: DeviceRangeMut<u64>,
    virtual_keys: DeviceRangeMut<K>,
    order: C,
) -> Result<(), MergeError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    I1: MergeDeviceInput<K>,
    I2: MergeDeviceInput<K>,
    O: MergeDeviceOutput<K>,
{
    let items_per_thread = key_items_per_thread::<NOMINAL_4B_ITEMS, K>();
    let tile = BLOCK_THREADS as u64 * items_per_thread as u64;
    let total = first_len + second_len;
    let tiles = total.div_ceil(tile);
    let partition_count = tiles + 1;
    let partition_launch = LaunchConfig::for_num_elems(partition_count as u32);
    let required_shared = merge_shared_bytes(BLOCK_THREADS, items_per_thread, size_of::<K>(), 0);
    let merge_launch = LaunchConfig {
        grid_dim: (tiles as u32, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: if required_shared <= MERGE_PORTABLE_SHARED_BYTES {
            required_shared as u32
        } else {
            0
        },
    };
    unsafe {
        module.partition_keys::<C, K, I1, I2>(
            stream,
            partition_launch,
            first,
            first_len,
            second,
            second_len,
            tile,
            partition_count,
            partitions,
            order,
        )?;
        module.merge_keys::<BLOCK_THREADS, NOMINAL_4B_ITEMS, UNROLL, C, K, I1, I2, O>(
            stream,
            merge_launch,
            first,
            first_len,
            second,
            second_len,
            output,
            partitions.as_read_only(),
            virtual_keys,
            order,
        )?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "host launch helper must forward separate key and value ranges"
)]
unsafe fn launch_merge_pairs<
    const BLOCK_THREADS: u32,
    const NOMINAL_4B_ITEMS: u32,
    const UNROLL: u32,
    C,
    K,
    V,
    K1,
    V1,
    K2,
    V2,
    KO,
    VO,
>(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    first_keys: K1,
    first_values: V1,
    first_len: u64,
    second_keys: K2,
    second_values: V2,
    second_len: u64,
    output_keys: KO,
    output_values: VO,
    partitions: DeviceRangeMut<u64>,
    virtual_keys: DeviceRangeMut<K>,
    virtual_values: DeviceRangeMut<V>,
    order: C,
) -> Result<(), MergeError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    V: DeviceCopy + 'static,
    K1: MergeDeviceInput<K>,
    V1: MergeDeviceInput<V>,
    K2: MergeDeviceInput<K>,
    V2: MergeDeviceInput<V>,
    KO: MergeDeviceOutput<K>,
    VO: MergeDeviceOutput<V>,
{
    let items_per_thread = pair_items_per_thread::<NOMINAL_4B_ITEMS, K, V>();
    let tile = BLOCK_THREADS as u64 * items_per_thread as u64;
    let total = first_len + second_len;
    let tiles = total.div_ceil(tile);
    let partition_count = tiles + 1;
    let partition_launch = LaunchConfig::for_num_elems(partition_count as u32);
    let shared = merge_shared_bytes(
        BLOCK_THREADS,
        items_per_thread,
        size_of::<K>(),
        size_of::<V>(),
    );
    let merge_launch = LaunchConfig {
        grid_dim: (tiles as u32, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: if shared <= MERGE_PORTABLE_SHARED_BYTES {
            shared as u32
        } else {
            0
        },
    };
    unsafe {
        module.partition_keys::<C, K, K1, K2>(
            stream,
            partition_launch,
            first_keys,
            first_len,
            second_keys,
            second_len,
            tile,
            partition_count,
            partitions,
            order,
        )?;
        module.merge_pairs::<
            BLOCK_THREADS,
            NOMINAL_4B_ITEMS,
            UNROLL,
            C,
            K,
            V,
            K1,
            V1,
            K2,
            V2,
            KO,
            VO,
        >(
            stream,
            merge_launch,
            first_keys,
            first_values,
            first_len,
            second_keys,
            second_values,
            second_len,
            output_keys,
            output_values,
            partitions.as_read_only(),
            virtual_keys,
            virtual_values,
            order,
        )?;
    }
    Ok(())
}

pub(crate) struct MergeWorkspace<
    'stream,
    K: DeviceCopy,
    V: DeviceCopy,
    A: DeviceAllocator + Clone = CudaAsyncMemoryResource,
> {
    partitions: ResourceBuffer<'stream, u64, A>,
    virtual_keys: ResourceBuffer<'stream, K, A>,
    virtual_values: Option<ResourceBuffer<'stream, V, A>>,
    capacity: usize,
}

struct MergeWorkspaceView<'workspace, K, V> {
    partitions: DeviceRangeMut<u64>,
    virtual_keys: DeviceRangeMut<K>,
    virtual_values: Option<DeviceRangeMut<V>>,
    _borrow: PhantomData<&'workspace mut (K, V)>,
}

impl<K: DeviceCopy, V: DeviceCopy, A: DeviceAllocator + Clone> MergeWorkspace<'_, K, V, A> {
    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
        self.partitions.ensure_usable_on(stream)?;
        self.virtual_keys.ensure_usable_on(stream)?;
        if let Some(values) = &self.virtual_values {
            values.ensure_usable_on(stream)?;
        }
        Ok(())
    }

    fn launch_view(&mut self) -> MergeWorkspaceView<'_, K, V> {
        // SAFETY: every range is bounded by its owning typed allocation. The
        // non-Copy view retains this exclusive workspace borrow for dispatch.
        let partitions = unsafe {
            DeviceRangeMut::from_raw_parts(
                self.partitions.cu_deviceptr() as *mut u64,
                self.partitions.len(),
            )
        };
        let virtual_keys = unsafe {
            DeviceRangeMut::from_raw_parts(
                self.virtual_keys.cu_deviceptr() as *mut K,
                self.virtual_keys.len(),
            )
        };
        let virtual_values = self.virtual_values.as_mut().map(|values| unsafe {
            DeviceRangeMut::from_raw_parts(values.cu_deviceptr() as *mut V, values.len())
        });
        MergeWorkspaceView {
            partitions,
            virtual_keys,
            virtual_values,
            _borrow: PhantomData,
        }
    }
}

/// CUB device-merge dispatch state for one allocator and CUDA context.
#[doc(hidden)]
pub(crate) struct DeviceMerge<A: DeviceAllocator + Clone> {
    module: Arc<kernels::LoadedModule>,
    allocator: A,
}

impl<A: DeviceAllocator + Clone> DeviceMerge<A> {
    pub(crate) fn new(context: &crate::KernelContext, allocator: A) -> Result<Self, MergeError> {
        Ok(Self {
            module: context.loaded_module(|module| kernels::from_module(module.clone()))?,
            allocator,
        })
    }

    pub(crate) fn allocate_keys_workspace<
        'stream,
        const BLOCK_THREADS: u32,
        const NOMINAL_4B_ITEMS: u32,
        K,
    >(
        &self,
        stream: &'stream CudaStream,
        capacity: usize,
    ) -> Result<MergeWorkspace<'stream, K, (), A>, MergeError>
    where
        K: DeviceCopy,
    {
        if size_of::<K>() == 0 {
            return Err(MergeError::ZeroSizedKey);
        }
        let items_per_thread = key_items_per_thread::<NOMINAL_4B_ITEMS, K>();
        let tile_items = BLOCK_THREADS * items_per_thread;
        let required_shared =
            merge_shared_bytes(BLOCK_THREADS, items_per_thread, size_of::<K>(), 0);
        let (tiles, partitions) = workspace_shape(capacity, tile_items)?;
        let virtual_items = if required_shared > MERGE_PORTABLE_SHARED_BYTES {
            tiles
                .checked_mul(tile_items as usize)
                .ok_or(MergeError::LengthOverflow)?
        } else {
            0
        };
        Ok(MergeWorkspace {
            partitions: ResourceBuffer::zeroed_in(self.allocator.clone(), stream, partitions)?,
            virtual_keys: unsafe {
                ResourceBuffer::uninitialized_in(self.allocator.clone(), stream, virtual_items)?
            },
            virtual_values: None,
            capacity,
        })
    }

    pub(crate) fn allocate_pairs_workspace<
        'stream,
        const BLOCK_THREADS: u32,
        const NOMINAL_4B_ITEMS: u32,
        K,
        V,
    >(
        &self,
        stream: &'stream CudaStream,
        capacity: usize,
    ) -> Result<MergeWorkspace<'stream, K, V, A>, MergeError>
    where
        K: DeviceCopy,
        V: DeviceCopy,
    {
        if size_of::<K>() == 0 {
            return Err(MergeError::ZeroSizedKey);
        }
        if size_of::<V>() == 0 {
            return Err(MergeError::ZeroSizedValue);
        }
        let items_per_thread = pair_items_per_thread::<NOMINAL_4B_ITEMS, K, V>();
        let tile_items = BLOCK_THREADS * items_per_thread;
        let required_shared = merge_shared_bytes(
            BLOCK_THREADS,
            items_per_thread,
            size_of::<K>(),
            size_of::<V>(),
        );
        let (tiles, partitions) = workspace_shape(capacity, tile_items)?;
        let virtual_items = if required_shared > MERGE_PORTABLE_SHARED_BYTES {
            tiles
                .checked_mul(tile_items as usize)
                .ok_or(MergeError::LengthOverflow)?
        } else {
            0
        };
        Ok(MergeWorkspace {
            partitions: ResourceBuffer::zeroed_in(self.allocator.clone(), stream, partitions)?,
            virtual_keys: unsafe {
                ResourceBuffer::uninitialized_in(self.allocator.clone(), stream, virtual_items)?
            },
            virtual_values: Some(unsafe {
                ResourceBuffer::uninitialized_in(self.allocator.clone(), stream, virtual_items)?
            }),
            capacity,
        })
    }

    /// Enqueue a key-only merge using a caller-owned workspace.
    ///
    /// # Safety
    ///
    /// Every input, output, and workspace borrow must remain alive and
    /// inaccessible until all submitted stream work completes, including when
    /// a later launch in this method returns an error.
    pub(crate) unsafe fn merge_keys<
        const BLOCK_THREADS: u32,
        const NOMINAL_4B_ITEMS: u32,
        const UNROLL: u32,
        C,
        K,
        A1,
        A2,
        O,
    >(
        &self,
        stream: &CudaStream,
        first: &A1,
        second: &A2,
        output: &mut O,
        order: C,
        workspace: &mut MergeWorkspace<'_, K, (), A>,
    ) -> Result<usize, MergeError>
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        A1: DeviceSlice<K> + ?Sized,
        A2: DeviceSlice<K> + ?Sized,
        O: DeviceSliceMut<K> + ?Sized,
    {
        first.ensure_usable_on(stream)?;
        second.ensure_usable_on(stream)?;
        output.ensure_usable_on(stream)?;
        let total = checked_total(first.len(), second.len())?;
        check_output(output.len(), total)?;
        check_no_overlap::<K, _, _>(first, output, total)?;
        check_no_overlap::<K, _, _>(second, output, total)?;
        if total == 0 {
            return Ok(0);
        }
        let tile_items = BLOCK_THREADS * key_items_per_thread::<NOMINAL_4B_ITEMS, K>();
        self.check_workspace(workspace, total, tile_items)?;
        workspace.ensure_usable_on(stream)?;
        let workspace = workspace.launch_view();
        unsafe {
            launch_merge_keys::<BLOCK_THREADS, NOMINAL_4B_ITEMS, UNROLL, C, K, _, _, _>(
                &self.module,
                stream,
                DeviceRange::from_raw_parts(first.cu_deviceptr() as *const K, first.len()),
                first.len() as u64,
                DeviceRange::from_raw_parts(second.cu_deviceptr() as *const K, second.len()),
                second.len() as u64,
                DeviceRangeMut::from_raw_parts(output.cu_deviceptr() as *mut K, output.len()),
                workspace.partitions,
                workspace.virtual_keys,
                order,
            )?;
        }
        Ok(total)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "internal key-value merge boundary preserves separately borrowed ranges"
    )]
    /// Enqueue a key/value merge using a caller-owned workspace.
    ///
    /// # Safety
    ///
    /// Every input, output, and workspace borrow must remain alive and
    /// inaccessible until all submitted stream work completes, including when
    /// a later launch in this method returns an error.
    pub(crate) unsafe fn merge_pairs<
        const BLOCK_THREADS: u32,
        const NOMINAL_4B_ITEMS: u32,
        const UNROLL: u32,
        C,
        K,
        V,
        K1,
        V1,
        K2,
        V2,
        KO,
        VO,
    >(
        &self,
        stream: &CudaStream,
        first_keys: &K1,
        first_values: &V1,
        second_keys: &K2,
        second_values: &V2,
        output_keys: &mut KO,
        output_values: &mut VO,
        order: C,
        workspace: &mut MergeWorkspace<'_, K, V, A>,
    ) -> Result<(usize, usize), MergeError>
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        V: DeviceCopy + 'static,
        K1: DeviceSlice<K> + ?Sized,
        V1: DeviceSlice<V> + ?Sized,
        K2: DeviceSlice<K> + ?Sized,
        V2: DeviceSlice<V> + ?Sized,
        KO: DeviceSliceMut<K> + ?Sized,
        VO: DeviceSliceMut<V> + ?Sized,
    {
        check_pair_lengths(first_keys.len(), first_values.len())?;
        check_pair_lengths(second_keys.len(), second_values.len())?;
        first_keys.ensure_usable_on(stream)?;
        first_values.ensure_usable_on(stream)?;
        second_keys.ensure_usable_on(stream)?;
        second_values.ensure_usable_on(stream)?;
        output_keys.ensure_usable_on(stream)?;
        output_values.ensure_usable_on(stream)?;
        let total = checked_total(first_keys.len(), second_keys.len())?;
        check_output(output_keys.len(), total)?;
        check_output(output_values.len(), total)?;
        check_no_overlap::<K, _, _>(first_keys, output_keys, total)?;
        check_no_overlap::<K, _, _>(second_keys, output_keys, total)?;
        check_no_overlap::<V, _, _>(first_values, output_values, total)?;
        check_no_overlap::<V, _, _>(second_values, output_values, total)?;
        if total == 0 {
            return Ok((0, 0));
        }
        let tile_items = BLOCK_THREADS * pair_items_per_thread::<NOMINAL_4B_ITEMS, K, V>();
        self.check_workspace(workspace, total, tile_items)?;
        workspace.ensure_usable_on(stream)?;
        let workspace = workspace.launch_view();
        let virtual_values = workspace
            .virtual_values
            .expect("pair workspace always owns typed value spill storage");
        unsafe {
            launch_merge_pairs::<BLOCK_THREADS, NOMINAL_4B_ITEMS, UNROLL, C, K, V, _, _, _, _, _, _>(
                &self.module,
                stream,
                DeviceRange::from_raw_parts(
                    first_keys.cu_deviceptr() as *const K,
                    first_keys.len(),
                ),
                DeviceRange::from_raw_parts(
                    first_values.cu_deviceptr() as *const V,
                    first_values.len(),
                ),
                first_keys.len() as u64,
                DeviceRange::from_raw_parts(
                    second_keys.cu_deviceptr() as *const K,
                    second_keys.len(),
                ),
                DeviceRange::from_raw_parts(
                    second_values.cu_deviceptr() as *const V,
                    second_values.len(),
                ),
                second_keys.len() as u64,
                DeviceRangeMut::from_raw_parts(
                    output_keys.cu_deviceptr() as *mut K,
                    output_keys.len(),
                ),
                DeviceRangeMut::from_raw_parts(
                    output_values.cu_deviceptr() as *mut V,
                    output_values.len(),
                ),
                workspace.partitions,
                workspace.virtual_keys,
                virtual_values,
                order,
            )?;
        }
        Ok((total, total))
    }

    fn check_workspace<K: DeviceCopy, V: DeviceCopy>(
        &self,
        workspace: &MergeWorkspace<'_, K, V, A>,
        total: usize,
        tile_items: u32,
    ) -> Result<(), MergeError> {
        if total > workspace.capacity {
            let required = total.div_ceil(tile_items as usize) + 1;
            return Err(MergeError::WorkspaceTooSmall {
                required,
                actual: workspace.partitions.len(),
            });
        }
        Ok(())
    }
}

fn workspace_shape(capacity: usize, tile_items: u32) -> Result<(usize, usize), MergeError> {
    let tiles = capacity.div_ceil(tile_items as usize);
    if tiles > i32::MAX as usize {
        return Err(MergeError::LengthOverflow);
    }
    let partitions = tiles.checked_add(1).ok_or(MergeError::LengthOverflow)?;
    Ok((tiles, partitions))
}

fn checked_total(first: usize, second: usize) -> Result<usize, MergeError> {
    let total = first
        .checked_add(second)
        .ok_or(MergeError::LengthOverflow)?;
    u64::try_from(total).map_err(|_| MergeError::LengthOverflow)?;
    Ok(total)
}

fn check_pair_lengths(keys: usize, values: usize) -> Result<(), MergeError> {
    if keys == values {
        Ok(())
    } else {
        Err(MergeError::LengthMismatch { keys, values })
    }
}

fn check_output(actual: usize, required: usize) -> Result<(), MergeError> {
    if actual >= required {
        Ok(())
    } else {
        Err(MergeError::OutputTooSmall { required, actual })
    }
}

fn check_no_overlap<T, I, O>(input: &I, output: &O, output_len: usize) -> Result<(), MergeError>
where
    I: DeviceSlice<T> + ?Sized,
    O: DeviceSlice<T> + ?Sized,
{
    let bytes = size_of::<T>();
    let input_begin = input.cu_deviceptr() as usize;
    let input_end = input_begin.saturating_add(input.len().saturating_mul(bytes));
    let output_begin = output.cu_deviceptr() as usize;
    let output_end = output_begin.saturating_add(output_len.saturating_mul(bytes));
    if input_begin < output_end && output_begin < input_end {
        Err(MergeError::InputOutputOverlap)
    } else {
        Ok(())
    }
}

/// Enqueue a stable ascending merge using the selected CUB geometry.
///
/// # Safety
///
/// The caller must keep every borrowed range alive and inaccessible until
/// `stream` has completed all enqueued work, including when this function
/// returns an error after an earlier launch was queued.
pub unsafe fn merge<
    const NORMAL_BLOCK_THREADS: u32,
    const NORMAL_ITEMS: u32,
    const NORMAL_UNROLL: u32,
    const FALLBACK_BLOCK_THREADS: u32,
    K,
    A1,
    A2,
    O,
    A,
>(
    context: &crate::KernelContext,
    stream: &CudaStream,
    allocator: &A,
    first: &A1,
    second: &A2,
    output: &mut O,
) -> Result<usize, MergeError>
where
    K: DeviceCopy + PartialOrd + 'static,
    A1: DeviceSlice<K> + ?Sized,
    A2: DeviceSlice<K> + ?Sized,
    O: DeviceSliceMut<K> + ?Sized,
    A: DeviceAllocator + Clone,
{
    unsafe {
        merge_by::<
            NORMAL_BLOCK_THREADS,
            NORMAL_ITEMS,
            NORMAL_UNROLL,
            FALLBACK_BLOCK_THREADS,
            _,
            K,
            A1,
            A2,
            O,
            A,
        >(
            context,
            stream,
            allocator,
            first,
            second,
            output,
            natural_order::<K>,
        )
    }
}

/// Enqueue a stable merge with a state-carrying device order.
///
/// # Safety
///
/// The caller must keep every borrowed range alive and inaccessible until
/// `stream` has completed all enqueued work, including when this function
/// returns an error after an earlier launch was queued.
pub unsafe fn merge_by<
    const NORMAL_BLOCK_THREADS: u32,
    const NORMAL_ITEMS: u32,
    const NORMAL_UNROLL: u32,
    const FALLBACK_BLOCK_THREADS: u32,
    C,
    K,
    A1,
    A2,
    O,
    A,
>(
    context: &crate::KernelContext,
    stream: &CudaStream,
    allocator: &A,
    first: &A1,
    second: &A2,
    output: &mut O,
    order: C,
) -> Result<usize, MergeError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    A1: DeviceSlice<K> + ?Sized,
    A2: DeviceSlice<K> + ?Sized,
    O: DeviceSliceMut<K> + ?Sized,
    A: DeviceAllocator + Clone,
{
    if merge_key_uses_fallback::<NORMAL_BLOCK_THREADS, NORMAL_ITEMS, FALLBACK_BLOCK_THREADS, K>() {
        return unsafe {
            merge_by_geometry::<FALLBACK_BLOCK_THREADS, 1, 2, C, K, A1, A2, O, A>(
                context, stream, allocator, first, second, output, order,
            )
        };
    }
    unsafe {
        merge_by_geometry::<NORMAL_BLOCK_THREADS, NORMAL_ITEMS, NORMAL_UNROLL, C, K, A1, A2, O, A>(
            context, stream, allocator, first, second, output, order,
        )
    }
}

unsafe fn merge_by_geometry<
    const BLOCK_THREADS: u32,
    const NOMINAL_4B_ITEMS: u32,
    const UNROLL: u32,
    C,
    K,
    A1,
    A2,
    O,
    A,
>(
    context: &crate::KernelContext,
    stream: &CudaStream,
    allocator: &A,
    first: &A1,
    second: &A2,
    output: &mut O,
    order: C,
) -> Result<usize, MergeError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    A1: DeviceSlice<K> + ?Sized,
    A2: DeviceSlice<K> + ?Sized,
    O: DeviceSliceMut<K> + ?Sized,
    A: DeviceAllocator + Clone,
{
    let total = checked_total(first.len(), second.len())?;
    if total == 0 {
        return Ok(0);
    }
    let engine = DeviceMerge::new(context, allocator.clone())?;
    let mut workspace =
        engine.allocate_keys_workspace::<BLOCK_THREADS, NOMINAL_4B_ITEMS, K>(stream, total)?;
    unsafe {
        engine.merge_keys::<BLOCK_THREADS, NOMINAL_4B_ITEMS, UNROLL, C, K, _, _, _>(
            stream,
            first,
            second,
            output,
            order,
            &mut workspace,
        )
    }
}

/// Enqueue a stable ascending key/value merge. Both returned logical ends equal
/// the combined key count.
///
/// # Safety
///
/// The caller must keep every borrowed range alive and inaccessible until
/// `stream` has completed all enqueued work, including when this function
/// returns an error after an earlier launch was queued.
#[expect(
    clippy::too_many_arguments,
    reason = "key-value merge API keeps four independently borrowed ranges explicit"
)]
pub unsafe fn merge_by_key<
    const NORMAL_BLOCK_THREADS: u32,
    const NORMAL_ITEMS: u32,
    const NORMAL_UNROLL: u32,
    const FALLBACK_BLOCK_THREADS: u32,
    K,
    V,
    K1,
    V1,
    K2,
    V2,
    KO,
    VO,
    A,
>(
    context: &crate::KernelContext,
    stream: &CudaStream,
    allocator: &A,
    first_keys: &K1,
    first_values: &V1,
    second_keys: &K2,
    second_values: &V2,
    output_keys: &mut KO,
    output_values: &mut VO,
) -> Result<(usize, usize), MergeError>
where
    K: DeviceCopy + PartialOrd + 'static,
    V: DeviceCopy + 'static,
    K1: DeviceSlice<K> + ?Sized,
    V1: DeviceSlice<V> + ?Sized,
    K2: DeviceSlice<K> + ?Sized,
    V2: DeviceSlice<V> + ?Sized,
    KO: DeviceSliceMut<K> + ?Sized,
    VO: DeviceSliceMut<V> + ?Sized,
    A: DeviceAllocator + Clone,
{
    unsafe {
        merge_by_key_with::<
            NORMAL_BLOCK_THREADS,
            NORMAL_ITEMS,
            NORMAL_UNROLL,
            FALLBACK_BLOCK_THREADS,
            _,
            K,
            V,
            K1,
            V1,
            K2,
            V2,
            KO,
            VO,
            A,
        >(
            context,
            stream,
            allocator,
            first_keys,
            first_values,
            second_keys,
            second_values,
            output_keys,
            output_values,
            natural_order::<K>,
        )
    }
}

/// Enqueue a stable key/value merge with a state-carrying device order.
///
/// # Safety
///
/// The caller must keep every borrowed range alive and inaccessible until
/// `stream` has completed all enqueued work, including when this function
/// returns an error after an earlier launch was queued.
#[expect(
    clippy::too_many_arguments,
    reason = "key-value merge API keeps four independently borrowed ranges explicit"
)]
pub unsafe fn merge_by_key_with<
    const NORMAL_BLOCK_THREADS: u32,
    const NORMAL_ITEMS: u32,
    const NORMAL_UNROLL: u32,
    const FALLBACK_BLOCK_THREADS: u32,
    C,
    K,
    V,
    K1,
    V1,
    K2,
    V2,
    KO,
    VO,
    A,
>(
    context: &crate::KernelContext,
    stream: &CudaStream,
    allocator: &A,
    first_keys: &K1,
    first_values: &V1,
    second_keys: &K2,
    second_values: &V2,
    output_keys: &mut KO,
    output_values: &mut VO,
    order: C,
) -> Result<(usize, usize), MergeError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    V: DeviceCopy + 'static,
    K1: DeviceSlice<K> + ?Sized,
    V1: DeviceSlice<V> + ?Sized,
    K2: DeviceSlice<K> + ?Sized,
    V2: DeviceSlice<V> + ?Sized,
    KO: DeviceSliceMut<K> + ?Sized,
    VO: DeviceSliceMut<V> + ?Sized,
    A: DeviceAllocator + Clone,
{
    if merge_pair_uses_fallback::<NORMAL_BLOCK_THREADS, NORMAL_ITEMS, FALLBACK_BLOCK_THREADS, K, V>(
    ) {
        return unsafe {
            merge_by_key_with_geometry::<
                FALLBACK_BLOCK_THREADS,
                1,
                2,
                C,
                K,
                V,
                K1,
                V1,
                K2,
                V2,
                KO,
                VO,
                A,
            >(
                context,
                stream,
                allocator,
                first_keys,
                first_values,
                second_keys,
                second_values,
                output_keys,
                output_values,
                order,
            )
        };
    }
    unsafe {
        merge_by_key_with_geometry::<
            NORMAL_BLOCK_THREADS,
            NORMAL_ITEMS,
            NORMAL_UNROLL,
            C,
            K,
            V,
            K1,
            V1,
            K2,
            V2,
            KO,
            VO,
            A,
        >(
            context,
            stream,
            allocator,
            first_keys,
            first_values,
            second_keys,
            second_values,
            output_keys,
            output_values,
            order,
        )
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "geometry-specialized helper forwards four independently borrowed ranges"
)]
unsafe fn merge_by_key_with_geometry<
    const BLOCK_THREADS: u32,
    const NOMINAL_4B_ITEMS: u32,
    const UNROLL: u32,
    C,
    K,
    V,
    K1,
    V1,
    K2,
    V2,
    KO,
    VO,
    A,
>(
    context: &crate::KernelContext,
    stream: &CudaStream,
    allocator: &A,
    first_keys: &K1,
    first_values: &V1,
    second_keys: &K2,
    second_values: &V2,
    output_keys: &mut KO,
    output_values: &mut VO,
    order: C,
) -> Result<(usize, usize), MergeError>
where
    C: Fn(&K, &K) -> Ordering + Copy + 'static,
    K: DeviceCopy + 'static,
    V: DeviceCopy + 'static,
    K1: DeviceSlice<K> + ?Sized,
    V1: DeviceSlice<V> + ?Sized,
    K2: DeviceSlice<K> + ?Sized,
    V2: DeviceSlice<V> + ?Sized,
    KO: DeviceSliceMut<K> + ?Sized,
    VO: DeviceSliceMut<V> + ?Sized,
    A: DeviceAllocator + Clone,
{
    check_pair_lengths(first_keys.len(), first_values.len())?;
    check_pair_lengths(second_keys.len(), second_values.len())?;
    let total = checked_total(first_keys.len(), second_keys.len())?;
    if total == 0 {
        return Ok((0, 0));
    }
    let engine = DeviceMerge::new(context, allocator.clone())?;
    let mut workspace =
        engine.allocate_pairs_workspace::<BLOCK_THREADS, NOMINAL_4B_ITEMS, K, V>(stream, total)?;
    unsafe {
        engine.merge_pairs::<BLOCK_THREADS, NOMINAL_4B_ITEMS, UNROLL, C, K, V, _, _, _, _, _, _>(
            stream,
            first_keys,
            first_values,
            second_keys,
            second_values,
            output_keys,
            output_values,
            order,
            &mut workspace,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cccl_items_per_thread_scaling_matches_the_formula() {
        assert_eq!(nominal_4b_items_to_items(15, 1), 15);
        assert_eq!(nominal_4b_items_to_items(15, 4), 15);
        assert_eq!(nominal_4b_items_to_items(15, 8), 7);
        assert_eq!(nominal_4b_items_to_items(15, 16), 3);
        assert_eq!(nominal_4b_items_to_items(15, 64), 1);
    }

    #[test]
    fn literal_schedule_determines_tile_and_virtual_shared_geometry() {
        assert_eq!(512 * key_items_per_thread::<15, u32>(), 512 * 15);
        assert_eq!(512 * pair_items_per_thread::<15, u32, u32>(), 512 * 7);

        assert!(merge_key_uses_fallback::<512, 15, 64, [u8; 512]>());
        assert_eq!(64 * key_items_per_thread::<1, [u8; 512]>(), 64);
        assert_eq!(
            virtual_shared_bytes(key_shared_bytes::<64, 1, [u8; 512]>()),
            0
        );

        assert!(!merge_key_uses_fallback::<512, 15, 64, [u8; 1024]>());
        assert_eq!(512 * key_items_per_thread::<15, [u8; 1024]>(), 512);
        assert_eq!(
            virtual_shared_bytes(key_shared_bytes::<512, 15, [u8; 1024]>()),
            512 * 1024
        );
    }

    #[test]
    fn data_shape_selects_geometry_without_a_strategy_type() {
        assert!(!merge_key_uses_fallback::<512, 15, 64, u32>());
        assert!(merge_key_uses_fallback::<512, 15, 64, [u8; 512]>());
        assert!(!merge_key_uses_fallback::<512, 15, 64, [u8; 1024]>());
        assert!(!merge_pair_uses_fallback::<512, 15, 64, u32, u32>());
    }
}
