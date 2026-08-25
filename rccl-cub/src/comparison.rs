//! Typed stable comparison-sort backend for the Thrust-shaped facade.
//!
//! Raw addresses exist only at the private CUDA kernel ABI. The safe host
//! boundary accepts borrowed `DeviceSliceMut` ranges and a concrete `Copy`
//! comparator value.

use crate::DeviceComparator;
use cuda_core::{CudaStream, DeviceCopy, DriverError, LaunchConfig};
use rccl_memory::{DeviceAllocator, DeviceBuffer as ResourceBuffer, DeviceSliceMut, MemoryError};
use std::{error::Error, fmt, marker::PhantomData, mem::size_of, sync::Arc};

/// Private erased contract at the byte-record kernel boundary.
///
/// # Safety
///
/// Implementations must interpret both pointers as live, properly aligned
/// values of the same key type and must preserve the typed comparator's
/// strict-weak-order contract.
unsafe trait RawDeviceComparator: Copy + 'static {
    unsafe fn less(&self, left: *const u8, right: *const u8) -> bool;
}

#[derive(Debug)]
struct TypedComparator<C, K> {
    comparator: C,
    _key: PhantomData<fn() -> K>,
}

impl<C: Copy, K> Copy for TypedComparator<C, K> {}

impl<C: Copy, K> Clone for TypedComparator<C, K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C, K> TypedComparator<C, K> {
    const fn new(comparator: C) -> Self {
        Self {
            comparator,
            _key: PhantomData,
        }
    }
}

unsafe impl<C, K> RawDeviceComparator for TypedComparator<C, K>
where
    C: DeviceComparator<K>,
    K: DeviceCopy + 'static,
{
    #[inline(always)]
    unsafe fn less(&self, left: *const u8, right: *const u8) -> bool {
        let left = unsafe { &*(left as *const K) };
        let right = unsafe { &*(right as *const K) };
        unsafe { self.comparator.less(left, right) }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "kernel macro emits flat CUDA ABI companion functions inside this module"
)]
#[cuda_host::cuda_module]
mod kernels {
    use super::RawDeviceComparator;
    use cuda_device::{kernel, thread};

    #[inline(always)]
    unsafe fn copy_record(source: *const u8, destination: *mut u8, bytes: u32) {
        let mut byte = 0;
        while byte < bytes {
            unsafe {
                destination
                    .add(byte as usize)
                    .write(source.add(byte as usize).read())
            };
            byte += 1;
        }
    }

    #[inline(always)]
    unsafe fn lower_bound<C: RawDeviceComparator>(
        comparator: &C,
        records: *const u8,
        begin: u64,
        end: u64,
        key: *const u8,
        key_bytes: u32,
    ) -> u64 {
        let mut first = begin;
        let mut count = end - begin;
        while count != 0 {
            let step = count / 2;
            let middle = first + step;
            let candidate = unsafe { records.add((middle * key_bytes as u64) as usize) };
            if unsafe { comparator.less(candidate, key) } {
                first = middle + 1;
                count -= step + 1;
            } else {
                count = step;
            }
        }
        first - begin
    }

    #[inline(always)]
    unsafe fn upper_bound<C: RawDeviceComparator>(
        comparator: &C,
        records: *const u8,
        begin: u64,
        end: u64,
        key: *const u8,
        key_bytes: u32,
    ) -> u64 {
        let mut first = begin;
        let mut count = end - begin;
        while count != 0 {
            let step = count / 2;
            let middle = first + step;
            let candidate = unsafe { records.add((middle * key_bytes as u64) as usize) };
            if !unsafe { comparator.less(key, candidate) } {
                first = middle + 1;
                count -= step + 1;
            } else {
                count = step;
            }
        }
        first - begin
    }

    /// Stable merge by input rank. Left-run records use lower_bound in the
    /// right run and right-run records use upper_bound in the left run.
    #[kernel]
    pub unsafe fn merge_records<C: RawDeviceComparator>(
        comparator: C,
        source_keys: *const u8,
        source_values: *const u8,
        destination_keys: *mut u8,
        destination_values: *mut u8,
        count: u64,
        width: u64,
        key_bytes: u32,
        value_bytes: u32,
    ) {
        let mut index = thread::index_1d().get() as u64;
        let stride = thread::gridDim_x() as u64 * thread::blockDim_x() as u64;
        while index < count {
            let pair_width = width.saturating_mul(2);
            let pair_begin = (index / pair_width) * pair_width;
            let middle = (pair_begin + width).min(count);
            let pair_end = (pair_begin + pair_width).min(count);
            let key = unsafe { source_keys.add((index * key_bytes as u64) as usize) };
            let destination = if index < middle {
                let rank = unsafe {
                    lower_bound(&comparator, source_keys, middle, pair_end, key, key_bytes)
                };
                pair_begin + (index - pair_begin) + rank
            } else {
                let rank = unsafe {
                    upper_bound(&comparator, source_keys, pair_begin, middle, key, key_bytes)
                };
                pair_begin + (index - middle) + rank
            };
            unsafe {
                copy_record(
                    key,
                    destination_keys.add((destination * key_bytes as u64) as usize),
                    key_bytes,
                )
            };
            if value_bytes != 0 {
                unsafe {
                    copy_record(
                        source_values.add((index * value_bytes as u64) as usize),
                        destination_values.add((destination * value_bytes as u64) as usize),
                        value_bytes,
                    )
                };
            }
            index += stride;
        }
    }

    #[kernel]
    pub unsafe fn copy_records<C: RawDeviceComparator>(
        source: *const u8,
        destination: *mut u8,
        count: u64,
        record_bytes: u32,
    ) {
        let mut index = thread::index_1d().get() as u64;
        let stride = thread::gridDim_x() as u64 * thread::blockDim_x() as u64;
        while index < count {
            unsafe {
                copy_record(
                    source.add((index * record_bytes as u64) as usize),
                    destination.add((index * record_bytes as u64) as usize),
                    record_bytes,
                )
            };
            index += stride;
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum ComparisonSortError {
    Driver(DriverError),
    Memory(MemoryError),
    Module(cuda_host::EmbeddedModuleError),
    LengthMismatch { keys: usize, values: usize },
    CapacityExceeded { requested: usize, capacity: usize },
    WorkspaceLayoutMismatch,
    SizeOverflow,
    ZeroSizedKey,
}

impl fmt::Display for ComparisonSortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(error) => write!(f, "CUDA driver error: {error}"),
            Self::Memory(error) => write!(f, "CUDA memory error: {error}"),
            Self::Module(error) => write!(f, "embedded CUDA module error: {error}"),
            Self::LengthMismatch { keys, values } => {
                write!(f, "key length {keys} does not match value length {values}")
            }
            Self::CapacityExceeded {
                requested,
                capacity,
            } => write!(f, "workspace capacity is {capacity}, requested {requested}"),
            Self::WorkspaceLayoutMismatch => {
                write!(f, "workspace element layout does not match this sort")
            }
            Self::SizeOverflow => write!(f, "comparison workspace size overflow"),
            Self::ZeroSizedKey => write!(f, "comparison sort does not accept zero-sized keys"),
        }
    }
}

impl Error for ComparisonSortError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Driver(error) => Some(error),
            Self::Memory(error) => Some(error),
            Self::Module(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DriverError> for ComparisonSortError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

impl From<MemoryError> for ComparisonSortError {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

impl From<cuda_host::EmbeddedModuleError> for ComparisonSortError {
    fn from(value: cuda_host::EmbeddedModuleError) -> Self {
        Self::Module(value)
    }
}

#[doc(hidden)]
pub struct ComparisonWorkspace<A: DeviceAllocator + Clone> {
    key_scratch: ResourceBuffer<u8, A>,
    value_scratch: ResourceBuffer<u8, A>,
    capacity: usize,
    key_bytes: usize,
    value_bytes: usize,
}

#[doc(hidden)]
pub struct DeviceMergeSort<A: DeviceAllocator + Clone> {
    module: Arc<kernels::LoadedModule>,
    allocator: A,
    multiprocessors: u32,
}

impl<A: DeviceAllocator + Clone> DeviceMergeSort<A> {
    pub fn new(context: &crate::KernelContext, allocator: A) -> Result<Self, ComparisonSortError> {
        Ok(Self {
            module: context.loaded_module(|module| kernels::from_module(module.clone()))?,
            allocator,
            multiprocessors: context.multiprocessors(),
        })
    }

    pub fn allocate_workspace<K: DeviceCopy, V: DeviceCopy>(
        &self,
        stream: &Arc<CudaStream>,
        capacity: usize,
    ) -> Result<ComparisonWorkspace<A>, ComparisonSortError> {
        let key_bytes = size_of::<K>();
        let value_bytes = size_of::<V>();
        let key_storage = capacity
            .checked_mul(key_bytes)
            .ok_or(ComparisonSortError::SizeOverflow)?;
        let value_storage = capacity
            .checked_mul(value_bytes)
            .ok_or(ComparisonSortError::SizeOverflow)?;
        Ok(ComparisonWorkspace {
            key_scratch: ResourceBuffer::zeroed_in(
                self.allocator.clone(),
                stream.clone(),
                key_storage,
            )?,
            value_scratch: ResourceBuffer::zeroed_in(
                self.allocator.clone(),
                stream.clone(),
                value_storage,
            )?,
            capacity,
            key_bytes,
            value_bytes,
        })
    }

    pub fn allocate_key_workspace<K: DeviceCopy>(
        &self,
        stream: &Arc<CudaStream>,
        capacity: usize,
    ) -> Result<ComparisonWorkspace<A>, ComparisonSortError> {
        let key_bytes = size_of::<K>();
        let key_storage = capacity
            .checked_mul(key_bytes)
            .ok_or(ComparisonSortError::SizeOverflow)?;
        Ok(ComparisonWorkspace {
            key_scratch: ResourceBuffer::zeroed_in(
                self.allocator.clone(),
                stream.clone(),
                key_storage,
            )?,
            value_scratch: ResourceBuffer::zeroed_in(self.allocator.clone(), stream.clone(), 0)?,
            capacity,
            key_bytes,
            value_bytes: 0,
        })
    }

    pub fn stable_sort_keys_by<const MERGE_BLOCK_THREADS: u32, C, K, B>(
        &self,
        stream: &Arc<CudaStream>,
        keys: &mut B,
        comparator: C,
        workspace: &mut ComparisonWorkspace<A>,
    ) -> Result<(), ComparisonSortError>
    where
        C: DeviceComparator<K>,
        K: DeviceCopy + 'static,
        B: DeviceSliceMut<K> + ?Sized,
    {
        keys.ensure_usable_on(stream)?;
        self.check_workspace::<K>(keys.len(), workspace, 0)?;
        if keys.len() < 2 {
            return Ok(());
        }
        self.merge_passes::<MERGE_BLOCK_THREADS, _>(
            stream,
            TypedComparator::<C, K>::new(comparator),
            keys.cu_deviceptr() as *mut u8,
            std::ptr::null_mut(),
            keys.len(),
            workspace,
            false,
        )
    }

    pub fn stable_sort_pairs_by<const MERGE_BLOCK_THREADS: u32, C, K, V, KB, VB>(
        &self,
        stream: &Arc<CudaStream>,
        keys: &mut KB,
        values: &mut VB,
        comparator: C,
        workspace: &mut ComparisonWorkspace<A>,
    ) -> Result<(), ComparisonSortError>
    where
        C: DeviceComparator<K>,
        K: DeviceCopy + 'static,
        V: DeviceCopy,
        KB: DeviceSliceMut<K> + ?Sized,
        VB: DeviceSliceMut<V> + ?Sized,
    {
        keys.ensure_usable_on(stream)?;
        values.ensure_usable_on(stream)?;
        if keys.len() != values.len() {
            return Err(ComparisonSortError::LengthMismatch {
                keys: keys.len(),
                values: values.len(),
            });
        }
        self.check_workspace::<K>(keys.len(), workspace, size_of::<V>())?;
        if keys.len() < 2 {
            return Ok(());
        }
        self.merge_passes::<MERGE_BLOCK_THREADS, _>(
            stream,
            TypedComparator::<C, K>::new(comparator),
            keys.cu_deviceptr() as *mut u8,
            values.cu_deviceptr() as *mut u8,
            keys.len(),
            workspace,
            true,
        )
    }

    fn launch_config<const MERGE_BLOCK_THREADS: u32>(&self, count: usize) -> LaunchConfig {
        let blocks = count
            .div_ceil(MERGE_BLOCK_THREADS as usize)
            .min((self.multiprocessors * 8) as usize)
            .max(1) as u32;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (MERGE_BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn check_workspace<K: DeviceCopy>(
        &self,
        count: usize,
        workspace: &ComparisonWorkspace<A>,
        value_bytes: usize,
    ) -> Result<(), ComparisonSortError> {
        if count != 0 && size_of::<K>() == 0 {
            return Err(ComparisonSortError::ZeroSizedKey);
        }
        if count > workspace.capacity {
            return Err(ComparisonSortError::CapacityExceeded {
                requested: count,
                capacity: workspace.capacity,
            });
        }
        if workspace.key_bytes != size_of::<K>() || workspace.value_bytes != value_bytes {
            return Err(ComparisonSortError::WorkspaceLayoutMismatch);
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "host launch helper must forward the complete CUDA kernel ABI"
    )]
    fn merge_passes<const MERGE_BLOCK_THREADS: u32, C: RawDeviceComparator>(
        &self,
        stream: &Arc<CudaStream>,
        comparator: C,
        original_keys: *mut u8,
        original_values: *mut u8,
        count: usize,
        workspace: &mut ComparisonWorkspace<A>,
        pairs: bool,
    ) -> Result<(), ComparisonSortError> {
        let cfg = self.launch_config::<MERGE_BLOCK_THREADS>(count);
        let scratch_keys = workspace.key_scratch.cu_deviceptr() as *mut u8;
        let scratch_values = workspace.value_scratch.cu_deviceptr() as *mut u8;
        let mut source_keys = original_keys as *const u8;
        let mut source_values = original_values as *const u8;
        let mut destination_keys = scratch_keys;
        let mut destination_values = scratch_values;
        let mut width = 1u64;
        while width < count as u64 {
            unsafe {
                self.module.merge_records::<C>(
                    stream,
                    cfg,
                    comparator,
                    source_keys,
                    source_values,
                    destination_keys,
                    destination_values,
                    count as u64,
                    width,
                    workspace.key_bytes as u32,
                    if pairs {
                        workspace.value_bytes as u32
                    } else {
                        0
                    },
                )?;
            }
            source_keys = destination_keys as *const u8;
            source_values = destination_values as *const u8;
            destination_keys = if destination_keys == scratch_keys {
                original_keys
            } else {
                scratch_keys
            };
            destination_values = if destination_values == scratch_values {
                original_values
            } else {
                scratch_values
            };
            width = width.saturating_mul(2);
        }
        if !std::ptr::eq(source_keys, original_keys as *const u8) {
            unsafe {
                self.module.copy_records::<C>(
                    stream,
                    cfg,
                    source_keys,
                    original_keys,
                    count as u64,
                    workspace.key_bytes as u32,
                )?;
                if pairs {
                    self.module.copy_records::<C>(
                        stream,
                        cfg,
                        source_values,
                        original_values,
                        count as u64,
                        workspace.value_bytes as u32,
                    )?;
                }
            }
        }
        Ok(())
    }
}
