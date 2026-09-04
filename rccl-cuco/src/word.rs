use cuda_core::DeviceCopy;

mod sealed {
    pub trait Key {}
    pub trait Value {}

    impl Key for u32 {}
    impl Key for i32 {}
    impl Value for u32 {}
    impl Value for i32 {}
    impl Value for f32 {}
}

/// A four-byte, bitwise-stable key accepted by the packed static map.
///
/// This is sealed to integer keys because IEEE floating-point equality does
/// not agree with bitwise sentinel identity for NaNs and signed zero.
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

/// A four-byte, bitwise-stable mapped value accepted by the packed map.
pub trait MapValue: sealed::Value + Copy + DeviceCopy {
    fn to_bits(self) -> u32;
    fn from_bits(bits: u32) -> Self;
}

impl MapValue for u32 {
    #[inline(always)]
    fn to_bits(self) -> u32 {
        self
    }

    #[inline(always)]
    fn from_bits(bits: u32) -> Self {
        bits
    }
}

impl MapValue for i32 {
    #[inline(always)]
    fn to_bits(self) -> u32 {
        self as u32
    }

    #[inline(always)]
    fn from_bits(bits: u32) -> Self {
        bits as i32
    }
}

impl MapValue for f32 {
    #[inline(always)]
    fn to_bits(self) -> u32 {
        f32::to_bits(self)
    }

    #[inline(always)]
    fn from_bits(bits: u32) -> Self {
        f32::from_bits(bits)
    }
}
