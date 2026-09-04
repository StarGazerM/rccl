use cuda_core::DeviceCopy;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64};

mod sealed {
    pub trait Key {}
    pub trait Value {}

    impl Key for u32 {}
    impl Key for i32 {}
    impl Value for u32 {}
    impl Value for i32 {}
    impl Value for f32 {}
    impl Value for u64 {}
}

/// A four-byte, bitwise-stable key accepted by the static map.
///
/// This is sealed to integer keys because IEEE floating-point equality does
/// not agree with bitwise sentinel identity for NaNs and signed zero. The
/// reachable vflog map keys are `uint32_t`, represented by `u32`.
pub trait MapKey: sealed::Key + Copy + Eq + DeviceCopy {
    fn to_bits(self) -> u32;
    fn from_bits(bits: u32) -> Self;
}

impl MapKey for u32 {
    #[inline(always)]
    fn to_bits(self) -> u32 {
        self
    }

    #[inline(always)]
    fn from_bits(bits: u32) -> Self {
        bits
    }
}

impl MapKey for i32 {
    #[inline(always)]
    fn to_bits(self) -> u32 {
        self as u32
    }

    #[inline(always)]
    fn from_bits(bits: u32) -> Self {
        bits as i32
    }
}

/// Atomic output word used by [`MapValue`] kernels.
pub trait MapValueAtomic<V: MapValue>: Sync {
    fn store_value(&self, value: V);
}

impl MapValueAtomic<u32> for DeviceAtomicU32 {
    #[inline(always)]
    fn store_value(&self, value: u32) {
        self.store(value, AtomicOrdering::Relaxed);
    }
}

impl MapValueAtomic<i32> for DeviceAtomicU32 {
    #[inline(always)]
    fn store_value(&self, value: i32) {
        self.store(value as u32, AtomicOrdering::Relaxed);
    }
}

impl MapValueAtomic<f32> for DeviceAtomicU32 {
    #[inline(always)]
    fn store_value(&self, value: f32) {
        self.store(value.to_bits(), AtomicOrdering::Relaxed);
    }
}

impl MapValueAtomic<u64> for DeviceAtomicU64 {
    #[inline(always)]
    fn store_value(&self, value: u64) {
        self.store(value, AtomicOrdering::Relaxed);
    }
}

/// A bitwise-stable mapped value accepted by the static map.
///
/// Four-byte values use the original packed slot: `key:u32 | value:u32` in one
/// `u64` CAS word. `u64` values use a two-word slot with a release-published
/// key/control word and a full 64-bit value word, so readers never infer status
/// from a torn value lane.
pub trait MapValue: sealed::Value + Copy + DeviceCopy {
    type Atomic: MapValueAtomic<Self>;

    const SLOT_WORDS: usize;

    fn to_bits(self) -> u64;
    fn from_bits(bits: u64) -> Self;

    #[inline(always)]
    fn store_atomic(output: &[Self::Atomic], index: usize, value: Self) {
        output[index].store_value(value);
    }
}

impl MapValue for u32 {
    type Atomic = DeviceAtomicU32;

    const SLOT_WORDS: usize = 1;

    #[inline(always)]
    fn to_bits(self) -> u64 {
        self as u64
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        bits as u32
    }
}

impl MapValue for i32 {
    type Atomic = DeviceAtomicU32;

    const SLOT_WORDS: usize = 1;

    #[inline(always)]
    fn to_bits(self) -> u64 {
        self as u32 as u64
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        bits as u32 as i32
    }
}

impl MapValue for f32 {
    type Atomic = DeviceAtomicU32;

    const SLOT_WORDS: usize = 1;

    #[inline(always)]
    fn to_bits(self) -> u64 {
        f32::to_bits(self) as u64
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        f32::from_bits(bits as u32)
    }
}

impl MapValue for u64 {
    type Atomic = DeviceAtomicU64;

    const SLOT_WORDS: usize = 2;

    #[inline(always)]
    fn to_bits(self) -> u64 {
        self
    }

    #[inline(always)]
    fn from_bits(bits: u64) -> Self {
        bits
    }
}
