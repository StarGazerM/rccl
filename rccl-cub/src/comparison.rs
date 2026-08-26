//! Typed stable comparison-sort backend for the Thrust-shaped facade.
//!
//! Raw addresses exist only at the private CUDA kernel ABI. The safe host
//! boundary accepts borrowed `DeviceSliceMut` ranges and a concrete `Copy`
//! ordering closure.

use crate::device::{DeviceRange, DeviceRangeMut};
use core::cmp::Ordering;
use cuda_core::{CudaStream, DeviceCopy, DriverError, LaunchConfig};
use rccl_memory::{DeviceAllocator, DeviceBuffer as ResourceBuffer, DeviceSliceMut, MemoryError};
use std::{error::Error, fmt, marker::PhantomData, mem::size_of, sync::Arc};

#[allow(
    clippy::too_many_arguments,
    reason = "kernel macro emits flat CUDA ABI companion functions inside this module"
)]
#[cuda_host::cuda_module]
mod kernels {
    use crate::device::{DeviceRange, DeviceRangeMut};
    use core::cmp::Ordering;
    use cuda_core::DeviceCopy;
    use cuda_device::{kernel, thread};

    #[inline(always)]
    fn comes_before<C, K>(order: &C, left: &K, right: &K) -> bool
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
    {
        order(left, right).is_lt()
    }

    #[inline(always)]
    unsafe fn lower_bound<C, K>(
        order: &C,
        records: DeviceRange<K>,
        begin: u64,
        end: u64,
        key: &K,
    ) -> u64
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
    {
        let mut first = begin;
        let mut count = end - begin;
        while count != 0 {
            let step = count / 2;
            let middle = first + step;
            let candidate = unsafe { records.read(middle) };
            if comes_before::<C, K>(order, &candidate, key) {
                first = middle + 1;
                count -= step + 1;
            } else {
                count = step;
            }
        }
        first - begin
    }

    #[inline(always)]
    unsafe fn upper_bound<C, K>(
        order: &C,
        records: DeviceRange<K>,
        begin: u64,
        end: u64,
        key: &K,
    ) -> u64
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
    {
        let mut first = begin;
        let mut count = end - begin;
        while count != 0 {
            let step = count / 2;
            let middle = first + step;
            let candidate = unsafe { records.read(middle) };
            if !comes_before::<C, K>(order, key, &candidate) {
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
    pub unsafe fn merge_records<C, K, V>(
        order: C,
        source_keys: DeviceRange<K>,
        source_values: DeviceRange<V>,
        destination_keys: DeviceRangeMut<K>,
        destination_values: DeviceRangeMut<V>,
        count: u64,
        width: u64,
        copy_values: u32,
    ) where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        V: DeviceCopy + 'static,
    {
        let mut index = thread::index_1d().get() as u64;
        let stride = thread::gridDim_x() as u64 * thread::blockDim_x() as u64;
        while index < count {
            let pair_width = width.saturating_mul(2);
            let pair_begin = (index / pair_width) * pair_width;
            let middle = (pair_begin + width).min(count);
            let pair_end = (pair_begin + pair_width).min(count);
            let key = unsafe { source_keys.read(index) };
            let destination = if index < middle {
                let rank =
                    unsafe { lower_bound::<C, K>(&order, source_keys, middle, pair_end, &key) };
                pair_begin + (index - pair_begin) + rank
            } else {
                let rank =
                    unsafe { upper_bound::<C, K>(&order, source_keys, pair_begin, middle, &key) };
                pair_begin + (index - middle) + rank
            };
            unsafe { destination_keys.write(destination, key) };
            if copy_values != 0 {
                let value = unsafe { source_values.read(index) };
                unsafe { destination_values.write(destination, value) };
            }
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
    LengthMismatch {
        keys: usize,
        values: usize,
    },
    CapacityExceeded {
        requested: usize,
        capacity: usize,
    },
    MissingKernel {
        name: &'static str,
        source: DriverError,
    },
    KernelLaunch {
        name: &'static str,
        source: DriverError,
    },
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
            Self::MissingKernel { name, source } => {
                write!(
                    f,
                    "comparison kernel `{name}` is absent from the loaded module: {source}"
                )
            }
            Self::KernelLaunch { name, source } => {
                write!(f, "comparison kernel `{name}` launch failed: {source}")
            }
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
            Self::MissingKernel { source, .. } | Self::KernelLaunch { source, .. } => Some(source),
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
pub struct ComparisonWorkspace<'stream, K: DeviceCopy, V: DeviceCopy, A: DeviceAllocator + Clone> {
    key_scratch: ResourceBuffer<'stream, K, A>,
    value_scratch: Option<ResourceBuffer<'stream, V, A>>,
    capacity: usize,
}

struct ComparisonWorkspaceView<'workspace, K, V> {
    key_scratch: DeviceRangeMut<K>,
    value_scratch: Option<DeviceRangeMut<V>>,
    _borrow: PhantomData<&'workspace mut (K, V)>,
}

impl<K: DeviceCopy, V: DeviceCopy, A: DeviceAllocator + Clone> ComparisonWorkspace<'_, K, V, A> {
    fn launch_view(&mut self) -> ComparisonWorkspaceView<'_, K, V> {
        // SAFETY: these ranges are bounded by their typed allocations. The
        // non-Copy view retains this exclusive workspace borrow for dispatch.
        let key_scratch = unsafe {
            DeviceRangeMut::from_raw_parts(
                self.key_scratch.cu_deviceptr() as *mut K,
                self.key_scratch.len(),
            )
        };
        let value_scratch = self.value_scratch.as_mut().map(|values| unsafe {
            DeviceRangeMut::from_raw_parts(values.cu_deviceptr() as *mut V, values.len())
        });
        ComparisonWorkspaceView {
            key_scratch,
            value_scratch,
            _borrow: PhantomData,
        }
    }
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

    pub fn allocate_workspace<'stream, K: DeviceCopy, V: DeviceCopy>(
        &self,
        stream: &'stream CudaStream,
        capacity: usize,
    ) -> Result<ComparisonWorkspace<'stream, K, V, A>, ComparisonSortError> {
        if capacity != 0 && size_of::<K>() == 0 {
            return Err(ComparisonSortError::ZeroSizedKey);
        }
        Ok(ComparisonWorkspace {
            key_scratch: ResourceBuffer::zeroed_in(self.allocator.clone(), stream, capacity)?,
            value_scratch: if size_of::<V>() == 0 {
                None
            } else {
                Some(ResourceBuffer::zeroed_in(
                    self.allocator.clone(),
                    stream,
                    capacity,
                )?)
            },
            capacity,
        })
    }

    pub fn allocate_key_workspace<'stream, K: DeviceCopy>(
        &self,
        stream: &'stream CudaStream,
        capacity: usize,
    ) -> Result<ComparisonWorkspace<'stream, K, (), A>, ComparisonSortError> {
        if capacity != 0 && size_of::<K>() == 0 {
            return Err(ComparisonSortError::ZeroSizedKey);
        }
        Ok(ComparisonWorkspace {
            key_scratch: ResourceBuffer::zeroed_in(self.allocator.clone(), stream, capacity)?,
            value_scratch: None,
            capacity,
        })
    }

    /// Enqueue a stable key sort on `stream`.
    ///
    /// # Safety
    ///
    /// `keys` and `workspace` must remain alive and inaccessible until every
    /// operation submitted to `stream` by this call has completed, including
    /// when this function returns an error after an earlier launch succeeded.
    pub unsafe fn stable_sort_keys_by<const MERGE_BLOCK_THREADS: u32, C, K, B>(
        &self,
        stream: &CudaStream,
        keys: &mut B,
        order: C,
        workspace: &mut ComparisonWorkspace<'_, K, (), A>,
    ) -> Result<(), ComparisonSortError>
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        B: DeviceSliceMut<K> + ?Sized,
    {
        keys.ensure_usable_on(stream)?;
        self.check_workspace(keys.len(), workspace.capacity)?;
        if keys.len() < 2 {
            return Ok(());
        }
        // SAFETY: `keys` owns the typed allocation and this method's unsafe
        // enqueue contract retains its exclusive borrow through completion.
        let keys =
            unsafe { DeviceRangeMut::from_raw_parts(keys.cu_deviceptr() as *mut K, keys.len()) };
        unsafe {
            self.merge_passes::<MERGE_BLOCK_THREADS, C, K, ()>(stream, order, keys, None, workspace)
        }
    }

    /// Enqueue a stable key/value sort on `stream`.
    ///
    /// # Safety
    ///
    /// `keys`, `values`, and `workspace` must remain alive and inaccessible
    /// until every operation submitted to `stream` by this call has completed,
    /// including when this function reports a later launch failure.
    pub unsafe fn stable_sort_pairs_by<const MERGE_BLOCK_THREADS: u32, C, K, V, KB, VB>(
        &self,
        stream: &CudaStream,
        keys: &mut KB,
        values: &mut VB,
        order: C,
        workspace: &mut ComparisonWorkspace<'_, K, V, A>,
    ) -> Result<(), ComparisonSortError>
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        V: DeviceCopy + 'static,
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
        self.check_workspace(keys.len(), workspace.capacity)?;
        if keys.len() < 2 {
            return Ok(());
        }
        let count = keys.len();
        // SAFETY: both owners cover initialized, equally sized typed ranges,
        // and the unsafe enqueue contract retains their exclusive borrows.
        let keys = unsafe { DeviceRangeMut::from_raw_parts(keys.cu_deviceptr() as *mut K, count) };
        let values =
            unsafe { DeviceRangeMut::from_raw_parts(values.cu_deviceptr() as *mut V, count) };
        unsafe {
            self.merge_passes::<MERGE_BLOCK_THREADS, C, K, V>(
                stream,
                order,
                keys,
                Some(values),
                workspace,
            )
        }
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

    fn check_workspace(&self, count: usize, capacity: usize) -> Result<(), ComparisonSortError> {
        if count > capacity {
            return Err(ComparisonSortError::CapacityExceeded {
                requested: count,
                capacity,
            });
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "private helper marshals the complete comparison kernel ABI"
    )]
    unsafe fn launch_merge_records<C, K, V>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig,
        order: C,
        source_keys: DeviceRange<K>,
        source_values: DeviceRange<V>,
        destination_keys: DeviceRangeMut<K>,
        destination_values: DeviceRangeMut<V>,
        count: u64,
        width: u64,
        copy_values: u32,
    ) -> Result<(), ComparisonSortError>
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        V: DeviceCopy + 'static,
    {
        // The pinned cuda-oxide typed module path currently computes a
        // different PTX type-id suffix for a closure-generic kernel declared
        // in another crate. Its generated instantiation helper returns the
        // collector's actual symbol, so use that symbol and marshal the same
        // by-value ABI explicitly. A failed lookup reports the exact name.
        let kernel_name = kernels::cuda_oxide_instantiate_246e25db_merge_records::<C, K, V>(&order);
        let function = self
            .module
            .as_cuda_module()
            .load_function(kernel_name)
            .map_err(|source| ComparisonSortError::MissingKernel {
                name: kernel_name,
                source,
            })?;

        let mut order_arg = order;
        let mut source_keys_arg = source_keys;
        let mut source_values_arg = source_values;
        let mut destination_keys_arg = destination_keys;
        let mut destination_values_arg = destination_values;
        let mut count_arg = count;
        let mut width_arg = width;
        let mut copy_values_arg = copy_values;
        let mut arguments = Vec::<*mut std::ffi::c_void>::with_capacity(8);
        cuda_host::push_kernel_scalar(&mut arguments, &mut order_arg);
        cuda_host::push_kernel_scalar(&mut arguments, &mut source_keys_arg);
        cuda_host::push_kernel_scalar(&mut arguments, &mut source_values_arg);
        cuda_host::push_kernel_scalar(&mut arguments, &mut destination_keys_arg);
        cuda_host::push_kernel_scalar(&mut arguments, &mut destination_values_arg);
        cuda_host::push_kernel_scalar(&mut arguments, &mut count_arg);
        cuda_host::push_kernel_scalar(&mut arguments, &mut width_arg);
        cuda_host::push_kernel_scalar(&mut arguments, &mut copy_values_arg);

        unsafe {
            cuda_core::launch_kernel_on_stream(
                &function,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                stream,
                &mut arguments,
            )
        }
        .map_err(|source| ComparisonSortError::KernelLaunch {
            name: kernel_name,
            source,
        })?;
        Ok(())
    }

    unsafe fn merge_passes<const MERGE_BLOCK_THREADS: u32, C, K, V>(
        &self,
        stream: &CudaStream,
        order: C,
        original_keys: DeviceRangeMut<K>,
        original_values: Option<DeviceRangeMut<V>>,
        workspace: &mut ComparisonWorkspace<'_, K, V, A>,
    ) -> Result<(), ComparisonSortError>
    where
        C: Fn(&K, &K) -> Ordering + Copy + 'static,
        K: DeviceCopy + 'static,
        V: DeviceCopy + 'static,
    {
        let count = original_keys.len() as usize;
        let cfg = self.launch_config::<MERGE_BLOCK_THREADS>(count);
        let workspace = workspace.launch_view();
        let scratch_keys = workspace.key_scratch;
        let scratch_values = workspace.value_scratch;
        let copy_values = u32::from(scratch_values.is_some());
        let original_values = original_values.unwrap_or_else(|| {
            debug_assert_eq!(copy_values, 0);
            // No value access occurs when `copy_values == 0`.
            DeviceRangeMut::empty()
        });
        let scratch_values = scratch_values.unwrap_or_else(|| {
            debug_assert_eq!(copy_values, 0);
            DeviceRangeMut::empty()
        });
        let mut source_keys = original_keys.as_read_only();
        let mut source_values = original_values.as_read_only();
        let mut destination_keys = scratch_keys;
        let mut destination_values = scratch_values;
        let mut width = 1u64;
        while width < count as u64 {
            unsafe {
                self.launch_merge_records::<C, K, V>(
                    stream,
                    cfg,
                    order,
                    source_keys,
                    source_values,
                    destination_keys,
                    destination_values,
                    count as u64,
                    width,
                    copy_values,
                )?;
            }
            source_keys = destination_keys.as_read_only();
            source_values = destination_values.as_read_only();
            destination_keys = if destination_keys.same_start(scratch_keys) {
                original_keys
            } else {
                scratch_keys
            };
            destination_values = if destination_values.same_start(scratch_values) {
                original_values
            } else {
                scratch_values
            };
            width = width.saturating_mul(2);
        }
        if source_keys.as_ptr() != original_keys.as_read_only().as_ptr() {
            unsafe {
                // `width == count` gives every record an empty right run, so
                // its stable merge rank is its input index. Reuse the already
                // instantiated closure kernel for the final ping-pong copy;
                // this also copies values in the same launch.
                self.launch_merge_records::<C, K, V>(
                    stream,
                    cfg,
                    order,
                    source_keys,
                    source_values,
                    original_keys,
                    original_values,
                    count as u64,
                    count as u64,
                    copy_values,
                )?;
            }
        }
        Ok(())
    }
}
