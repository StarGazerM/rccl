//! Device-callable scalar vocabulary owned by the CUB layer.

use cuda_core::DeviceCopy;

/// Private typed CUDA ABI view for a contiguous read-only device range.
///
/// This is intentionally a device-side scalar rather than a public host
/// capability. Host code may construct it only inside an unsafe enqueue
/// boundary whose caller retains the real allocation borrow until completion.
#[repr(C)]
pub(crate) struct DeviceRange<T> {
    pointer: *const T,
    len: u64,
}

impl<T> Copy for DeviceRange<T> {}

impl<T> Clone for DeviceRange<T> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<T> DeviceCopy for DeviceRange<T> {}

impl<T> DeviceRange<T> {
    /// # Safety
    ///
    /// `pointer` must cover `len` initialized device values for the complete
    /// kernel operation that receives this private ABI value.
    pub(crate) const unsafe fn from_raw_parts(pointer: *const T, len: usize) -> Self {
        Self {
            pointer,
            len: len as u64,
        }
    }

    pub(crate) const fn as_ptr(self) -> *const T {
        self.pointer
    }

    /// # Safety
    ///
    /// `index` must be less than `len()`.
    #[inline(always)]
    pub(crate) unsafe fn read(self, index: u64) -> T {
        unsafe { self.pointer.add(index as usize).read() }
    }
}

/// Private typed CUDA ABI view for a contiguous writable device range.
#[repr(C)]
pub(crate) struct DeviceRangeMut<T> {
    pointer: *mut T,
    len: u64,
}

impl<T> Copy for DeviceRangeMut<T> {}

impl<T> Clone for DeviceRangeMut<T> {
    fn clone(&self) -> Self {
        *self
    }
}

unsafe impl<T> DeviceCopy for DeviceRangeMut<T> {}

impl<T> DeviceRangeMut<T> {
    pub(crate) const fn empty() -> Self {
        Self {
            pointer: core::ptr::null_mut(),
            len: 0,
        }
    }

    /// # Safety
    ///
    /// `pointer` must cover `len` device values with exclusive write authority
    /// for the complete kernel operation that receives this ABI value.
    pub(crate) const unsafe fn from_raw_parts(pointer: *mut T, len: usize) -> Self {
        Self {
            pointer,
            len: len as u64,
        }
    }

    pub(crate) const fn len(self) -> u64 {
        self.len
    }

    pub(crate) const fn as_mut_ptr(self) -> *mut T {
        self.pointer
    }

    pub(crate) const fn as_read_only(self) -> DeviceRange<T> {
        DeviceRange {
            pointer: self.pointer,
            len: self.len,
        }
    }

    pub(crate) fn same_start(self, other: Self) -> bool {
        self.pointer == other.pointer
    }

    /// # Safety
    ///
    /// `index` must be less than `len()` and no other thread may access that
    /// element incompatibly during this operation.
    #[inline(always)]
    pub(crate) unsafe fn write(self, index: u64, value: T) {
        unsafe { self.pointer.add(index as usize).write(value) };
    }
}

/// Storage-compatible CUDA half-precision key.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default)]
pub struct F16(pub u16);

unsafe impl DeviceCopy for F16 {}

/// Storage-compatible CUDA bfloat16 key.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BFloat16(pub u16);

unsafe impl DeviceCopy for BFloat16 {}

// This defines the default total order used by RCCL's storage-only half types.
// It is CUB's bit-transform order: signed zeros compare equal and every NaN
// payload has a deterministic place.
#[inline(always)]
const fn float16_order_code(raw: u16) -> u16 {
    if raw & 0x7fff == 0 {
        0x8000
    } else if raw & 0x8000 != 0 {
        !raw
    } else {
        raw ^ 0x8000
    }
}

macro_rules! ordered_float16_key {
    ($key:ty) => {
        impl PartialEq for $key {
            fn eq(&self, other: &Self) -> bool {
                float16_order_code(self.0) == float16_order_code(other.0)
            }
        }

        impl Eq for $key {}

        impl Ord for $key {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                float16_order_code(self.0).cmp(&float16_order_code(other.0))
            }
        }

        impl PartialOrd for $key {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
    };
}

ordered_float16_key!(F16);
ordered_float16_key!(BFloat16);

/// CUDA-compatible one-byte Boolean key.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceBool(u8);

unsafe impl DeviceCopy for DeviceBool {}

impl DeviceBool {
    pub const FALSE: Self = Self(0);
    pub const TRUE: Self = Self(1);

    pub const fn new(value: bool) -> Self {
        Self(value as u8)
    }

    pub const fn get(self) -> bool {
        self.0 != 0
    }
}

impl From<bool> for DeviceBool {
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}

impl PartialEq for DeviceBool {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl Eq for DeviceBool {}

impl Ord for DeviceBool {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.get().cmp(&other.get())
    }
}

impl PartialOrd for DeviceBool {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
