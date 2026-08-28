use super::*;

/// Common read-only device allocation surface used by RCCL algorithms.
///
/// # Safety
///
/// The pointer must remain valid for `len()` contiguous `T` values for the
/// lifetime of `self`, and it must belong to `context()`.
pub unsafe trait ContiguousDeviceRange<T> {
    fn len(&self) -> usize;
    fn cu_deviceptr(&self) -> sys::CUdeviceptr;
    fn context(&self) -> &Arc<CudaContext>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn num_bytes(&self) -> usize {
        self.len().saturating_mul(std::mem::size_of::<T>())
    }

    /// Check that this allocation may be used by work submitted to `stream`.
    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
        validate_stream(self.context(), stream)
    }
}

/// Writable extension of [`ContiguousDeviceRange`].
///
/// # Safety
///
/// Mutable access to the implementor must grant exclusive write access to all
/// `len()` device elements for the submitted operation.
pub unsafe trait ContiguousDeviceRangeMut<T>: ContiguousDeviceRange<T> {}

unsafe impl<T> ContiguousDeviceRange<T> for cuda_core::DeviceBuffer<T> {
    fn len(&self) -> usize {
        self.len()
    }

    fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.cu_deviceptr()
    }

    fn context(&self) -> &Arc<CudaContext> {
        self.context()
    }
}

unsafe impl<T> ContiguousDeviceRangeMut<T> for cuda_core::DeviceBuffer<T> {}

/// Enqueue a same-length device-to-device copy between safe contiguous ranges.
///
/// Pointer extraction and CUDA submission stay inside this memory boundary;
/// algorithm generators retain ordinary Rust borrows and never emit pointer
/// arithmetic or `unsafe` blocks.
pub fn copy_device_range<T, S, D>(
    source: &S,
    destination: &mut D,
    stream: &CudaStream,
) -> Result<(), MemoryError>
where
    T: DeviceCopy,
    S: ContiguousDeviceRange<T> + ?Sized,
    D: ContiguousDeviceRangeMut<T> + ?Sized,
{
    source.ensure_usable_on(stream)?;
    destination.ensure_usable_on(stream)?;
    if source.len() != destination.len() {
        return Err(MemoryError::LengthMismatch {
            source: source.len(),
            destination: destination.len(),
        });
    }
    let bytes = source
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(MemoryError::SizeOverflow)?;
    if bytes != 0 {
        unsafe {
            memcpy_dtod_async(
                destination.cu_deviceptr(),
                source.cu_deviceptr(),
                bytes,
                stream.cu_stream(),
            )?;
        }
    }
    Ok(())
}
