use super::*;

mod temporary_value {
    pub trait Sealed {}

    macro_rules! primitive {
        ($($ty:ty),+ $(,)?) => {$(
            impl Sealed for $ty {}
        )+};
    }

    primitive!(
        u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64
    );
}

/// A device value for which every bit pattern is a valid Rust value.
///
/// CUB temporary storage is intentionally uninitialized. Sealing this trait to
/// numeric primitives makes those bytes valid typed storage without adding a
/// whole-workspace memset that the CCCL algorithm does not perform.
#[doc(hidden)]
pub trait TemporaryValue: DeviceCopy + temporary_value::Sealed {}

impl<T> TemporaryValue for T where T: DeviceCopy + temporary_value::Sealed {}

/// One non-owning, typed region inside a [`TemporaryWorkspace1`] through
/// [`TemporaryWorkspace5`].
///
/// Values cannot be constructed or detached by users. The owning workspace
/// creates non-overlapping 256-byte-compatible regions and exposes them only
/// through mutable borrows.
#[doc(hidden)]
pub struct TemporaryRegion<'stream, T: TemporaryValue> {
    ptr: sys::CUdeviceptr,
    len: usize,
    stream: &'stream CudaStream,
    marker: PhantomData<&'stream mut [T]>,
}

impl<T: TemporaryValue> TemporaryRegion<'_, T> {
    fn new(ptr: sys::CUdeviceptr, len: usize, stream: &CudaStream) -> TemporaryRegion<'_, T> {
        TemporaryRegion {
            ptr,
            len,
            stream,
            marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn zero_async(&mut self) -> Result<(), MemoryError> {
        let bytes = self
            .len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(MemoryError::SizeOverflow)?;
        if bytes != 0 {
            unsafe { memset_d8_async(self.ptr, 0, bytes, self.stream.cu_stream())? };
        }
        Ok(())
    }

    /// Copy the first temporary value to the host at a synchronous algorithm
    /// completion point.
    ///
    /// Set operations use this for CCCL's device output-count word.  The
    /// stream synchronization and the following synchronous DtoH copy match
    /// `synchronize(policy)` followed by `get_value(policy, ptr)`.
    #[doc(hidden)]
    pub fn first_to_host(&self) -> Result<T, MemoryError> {
        if self.is_empty() {
            return Err(MemoryError::SizeOverflow);
        }
        self.stream.synchronize().map_err(MemoryError::Driver)?;
        let mut host = MaybeUninit::<T>::uninit();
        // SAFETY: a nonempty TemporaryRegion owns at least one initialized
        // element-sized device slot. The preceding stream synchronization
        // completes the kernel write, and the synchronous driver copy returns
        // only after filling `host`.
        unsafe {
            sys::cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), self.ptr, std::mem::size_of::<T>())
                .result()
                .map_err(MemoryError::Driver)?;
            Ok(host.assume_init())
        }
    }
}

impl<T: TemporaryValue> fmt::Debug for TemporaryRegion<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TemporaryRegion")
            .field("len", &self.len)
            .field("device", &self.stream.context().ordinal())
            .finish()
    }
}

unsafe impl<T: TemporaryValue> ContiguousDeviceRange<T> for TemporaryRegion<'_, T> {
    fn len(&self) -> usize {
        self.len
    }

    fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }

    fn context(&self) -> &Arc<CudaContext> {
        self.stream.context()
    }

    fn ensure_usable_on(&self, stream: &CudaStream) -> Result<(), MemoryError> {
        validate_stream(self.stream.context(), stream)?;
        if self.stream.cu_stream() != stream.cu_stream() {
            return Err(MemoryError::StreamMismatch);
        }
        Ok(())
    }
}

unsafe impl<T: TemporaryValue> ContiguousDeviceRangeMut<T> for TemporaryRegion<'_, T> {}

// SAFETY: TemporaryRegion is created only by a workspace that owns the live
// allocation, proves the region's aligned byte extent, and hands it out under
// an exclusive borrow. TemporaryValue guarantees arbitrary initial bytes are
// valid values of T.
unsafe impl<T: TemporaryValue> cuda_host::KernelSliceArg for TemporaryRegion<'_, T> {
    type Elem = T;

    fn cu_deviceptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: a mutable borrow of a workspace region uniquely covers that region;
// sibling regions were constructed from disjoint rounded byte extents.
unsafe impl<T: TemporaryValue> cuda_host::KernelSliceArgMut for TemporaryRegion<'_, T> {}

struct TemporaryAllocation<'stream, A: DeviceAllocator> {
    block: Option<DeviceBlock>,
    allocator: A,
    stream: &'stream CudaStream,
}

impl<'stream, A: DeviceAllocator> TemporaryAllocation<'stream, A> {
    fn new(allocator: A, stream: &'stream CudaStream, size: usize) -> Result<Self, MemoryError> {
        validate_stream(allocator.context(), stream)?;
        let layout = Layout::from_size_align(size, DEVICE_ALLOCATION_ALIGNMENT)?;
        let block = allocator.allocate(layout, stream)?;
        Ok(Self {
            block: Some(block),
            allocator,
            stream,
        })
    }

    fn ptr(&self) -> sys::CUdeviceptr {
        self.block.as_ref().map_or(0, DeviceBlock::cu_deviceptr)
    }
}

impl<A: DeviceAllocator> Drop for TemporaryAllocation<'_, A> {
    fn drop(&mut self) {
        let Some(block) = self.block.take() else {
            return;
        };
        match unsafe { self.allocator.deallocate(block, self.stream) } {
            Ok(()) => {}
            Err(MemoryError::Driver(error)) => {
                self.allocator.context().record_err::<()>(Err(error))
            }
            Err(_) => {}
        }
    }
}

fn temporary_region_bytes<T: TemporaryValue>(len: usize) -> Result<usize, MemoryError> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or(MemoryError::SizeOverflow)
}

fn align_up(value: usize, alignment: usize) -> Result<usize, MemoryError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(MemoryError::InvalidAlignment {
            requested: alignment,
        });
    }
    if alignment > DEVICE_ALLOCATION_ALIGNMENT {
        return Err(MemoryError::UnsupportedAlignment {
            requested: alignment,
            supported: DEVICE_ALLOCATION_ALIGNMENT,
        });
    }
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or(MemoryError::SizeOverflow)
}

pub(crate) fn temporary_layout<const N: usize>(
    sizes: [usize; N],
    alignments: [usize; N],
) -> Result<([usize; N], usize), MemoryError> {
    let mut offsets = [0usize; N];
    let mut cursor = 0usize;
    let mut index = 0usize;
    while index < N {
        cursor = align_up(cursor, alignments[index])?;
        offsets[index] = cursor;
        cursor = cursor
            .checked_add(align_up(sizes[index], alignments[index])?)
            .ok_or(MemoryError::SizeOverflow)?;
        index += 1;
    }
    // CUB alias_temporaries reserves this tail so its input pointer may be
    // advanced to a 256-byte boundary even when the caller's byte range is not.
    let total = cursor
        .checked_add(DEVICE_ALLOCATION_ALIGNMENT - 1)
        .ok_or(MemoryError::SizeOverflow)?;
    Ok((offsets, total))
}

macro_rules! define_temporary_workspace {
    ($name:ident, $count:literal, $(($field:ident, $ty:ident, $index:tt)),+ $(,)?) => {
        #[doc = "One owning CUB-style byte allocation split into disjoint typed regions."]
        #[doc(hidden)]
        pub struct $name<'stream, A: DeviceAllocator, $($ty: TemporaryValue),+> {
            $($field: TemporaryRegion<'stream, $ty>,)+
            allocation: TemporaryAllocation<'stream, A>,
        }

        impl<'stream, A: DeviceAllocator, $($ty: TemporaryValue),+>
            $name<'stream, A, $($ty),+>
        {
            pub fn new_in(
                allocator: A,
                stream: &'stream CudaStream,
                counts: [usize; $count],
                alignments: [usize; $count],
            ) -> Result<Self, MemoryError> {
                Self::new_in_with_extra_tail(allocator, stream, counts, alignments, 0)
            }

            /// Preserve an additional outer aliasing level without changing
            /// any typed-region offset. CCCL set operations nest their inner
            /// three-part alias inside the Thrust output-count alias, adding
            /// one more 256-byte rounded tail to the single allocation.
            #[doc(hidden)]
            pub fn new_in_with_extra_tail(
                allocator: A,
                stream: &'stream CudaStream,
                counts: [usize; $count],
                alignments: [usize; $count],
                extra_tail: usize,
            ) -> Result<Self, MemoryError> {
                let sizes = [$(temporary_region_bytes::<$ty>(counts[$index])?),+];
                let (offsets, base_total) = temporary_layout(sizes, alignments)?;
                let total = base_total
                    .checked_add(extra_tail)
                    .ok_or(MemoryError::SizeOverflow)?;
                let allocation = TemporaryAllocation::new(allocator, stream, total)?;
                let base = allocation.ptr();
                Ok(Self {
                    $($field: TemporaryRegion::new(
                        base + offsets[$index] as sys::CUdeviceptr,
                        counts[$index],
                        stream,
                    ),)+
                    allocation,
                })
            }

            pub fn parts_mut(&mut self) -> ($(&mut TemporaryRegion<'stream, $ty>,)+) {
                ($(&mut self.$field,)+)
            }

            pub fn allocation_bytes(&self) -> usize {
                self.allocation.block.as_ref().map_or(0, DeviceBlock::size)
            }
        }
    };
}

define_temporary_workspace!(TemporaryWorkspace1, 1, (first, T1, 0));
define_temporary_workspace!(TemporaryWorkspace2, 2, (first, T1, 0), (second, T2, 1),);
define_temporary_workspace!(
    TemporaryWorkspace3,
    3,
    (first, T1, 0),
    (second, T2, 1),
    (third, T3, 2),
);
define_temporary_workspace!(
    TemporaryWorkspace4,
    4,
    (first, T1, 0),
    (second, T2, 1),
    (third, T3, 2),
    (fourth, T4, 3),
);
define_temporary_workspace!(
    TemporaryWorkspace5,
    5,
    (first, T1, 0),
    (second, T2, 1),
    (third, T3, 2),
    (fourth, T4, 3),
    (fifth, T5, 4),
);
