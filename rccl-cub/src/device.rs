//! Device-callable scalar and comparator vocabulary owned by the CUB layer.

use cuda_core::DeviceCopy;

/// A typed, state-carrying comparator compiled into RCCL kernels.
///
/// # Safety
///
/// `less` must be GPU-callable and define a strict weak ordering. The value
/// must have the same bit-valid representation on host and device and must not
/// contain host references. Captured device addresses must remain live until
/// the operation completes.
pub unsafe trait DeviceComparator<K: DeviceCopy>: Copy + 'static {
    /// Return whether `left` precedes `right`.
    ///
    /// # Safety
    ///
    /// This method executes in device code. Implementations must uphold the
    /// trait-level representation, lifetime, and strict-weak-order contract.
    unsafe fn less(&self, left: &K, right: &K) -> bool;
}

/// Canonical ascending comparator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Less;

unsafe impl<K> DeviceComparator<K> for Less
where
    K: DeviceCopy + PartialOrd + 'static,
{
    #[inline(always)]
    unsafe fn less(&self, left: &K, right: &K) -> bool {
        left < right
    }
}

/// Canonical descending comparator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Greater;

unsafe impl<K> DeviceComparator<K> for Greater
where
    K: DeviceCopy + PartialOrd + 'static,
{
    #[inline(always)]
    unsafe fn less(&self, left: &K, right: &K) -> bool {
        left > right
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
