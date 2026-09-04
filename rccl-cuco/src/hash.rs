//! Bit-compatible xxHash64 for stable byte sequences.

const PRIME1: u64 = 11_400_714_785_074_694_791;
const PRIME2: u64 = 14_029_467_366_897_019_727;
const PRIME3: u64 = 1_609_587_929_392_839_161;
const PRIME4: u64 = 9_650_029_242_287_828_579;
const PRIME5: u64 = 2_870_177_450_012_600_261;

/// Marks a value whose complete object representation is stable hash input.
///
/// # Safety
///
/// Every byte in `Self` must be initialized and must have a stable meaning for
/// equality. Implementations for records therefore require a stable layout
/// such as `#[repr(C)]` and must not expose uninitialized padding. Integer
/// fields must use their little-endian object representation. Pointers,
/// references, compiler-selected enum layouts, and records with implicit or
/// uninitialized padding do not satisfy this contract.
///
/// Prefer [`XxHash64::hash_bytes`] with an explicit encoding when a record
/// cannot meet all of these requirements. There is deliberately no blanket
/// implementation for `Copy` or `DeviceCopy` values.
pub unsafe trait StableHashBytes: Sized {}

#[cfg(target_endian = "little")]
macro_rules! stable_integer {
    ($($ty:ty),+ $(,)?) => {
        $(
            // SAFETY: integers have no padding and their native object
            // representation is the required little-endian encoding.
            unsafe impl StableHashBytes for $ty {}
        )+
    };
}

#[cfg(target_endian = "little")]
stable_integer!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128);

// SAFETY: Rust arrays contain their elements contiguously without inter-element
// or trailing padding. The element contract initializes every element byte.
unsafe impl<T: StableHashBytes, const N: usize> StableHashBytes for [T; N] {}

/// A seed-bearing xxHash64 value compatible with pinned cuCollections XXH64.
///
/// The oracle is `cuco::xxhash_64`, aliasing `cuco::detail::XXHash_64`, at
/// cuCollections commit `1d538b6befc286ab450d9e27d7ffbc4f1415c641`.
/// The audited SRDatalog HISA workload uses that exact implementation for
/// padding-free `TBlob { u32 data[N] }` records and for the resulting `u64`
/// values passed to its Bloom filter.
///
/// Input bytes are consumed in increasing address order and multi-byte chunks
/// are decoded little-endian. Any alignment is accepted, including a byte
/// slice starting at an odd address, and lengths need not be multiples of 4,
/// 8, or 32. The seed is an ordinary 64-bit wrapping xxHash seed and defaults
/// to zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct XxHash64 {
    seed: u64,
}

impl XxHash64 {
    /// Constructs a hasher with `seed`.
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Returns this hasher's seed.
    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Hashes an explicitly encoded byte sequence.
    ///
    /// This is the preferred API for layouts containing padding: construct a
    /// padding-free encoding and hash that encoding rather than the object
    /// representation.
    #[inline(always)]
    pub fn hash_bytes(self, bytes: &[u8]) -> u64 {
        // SAFETY: the slice guarantees that `bytes.as_ptr()` addresses all
        // `bytes.len()` initialized bytes. A byte slice promises no alignment.
        unsafe { self.hash_pointer(bytes.as_ptr(), bytes.len(), 1) }
    }

    #[inline(always)]
    unsafe fn hash_pointer(self, pointer: *const u8, len: usize, alignment: usize) -> u64 {
        let mut offset = 0usize;
        let mut hash;

        if len >= 32 {
            let limit = len - 32;
            let mut lane1 = self.seed.wrapping_add(PRIME1).wrapping_add(PRIME2);
            let mut lane2 = self.seed.wrapping_add(PRIME2);
            let mut lane3 = self.seed;
            let mut lane4 = self.seed.wrapping_sub(PRIME1);

            loop {
                lane1 = round(lane1, read_u64_le(pointer, offset, alignment));
                lane2 = round(lane2, read_u64_le(pointer, offset + 8, alignment));
                lane3 = round(lane3, read_u64_le(pointer, offset + 16, alignment));
                lane4 = round(lane4, read_u64_le(pointer, offset + 24, alignment));
                offset += 32;
                if offset > limit {
                    break;
                }
            }

            hash = lane1
                .rotate_left(1)
                .wrapping_add(lane2.rotate_left(7))
                .wrapping_add(lane3.rotate_left(12))
                .wrapping_add(lane4.rotate_left(18));
            hash = merge_round(hash, lane1);
            hash = merge_round(hash, lane2);
            hash = merge_round(hash, lane3);
            hash = merge_round(hash, lane4);
        } else {
            hash = self.seed.wrapping_add(PRIME5);
        }

        hash = hash.wrapping_add(len as u64);

        while len - offset >= 8 {
            let mixed = round(0, read_u64_le(pointer, offset, alignment));
            hash ^= mixed;
            hash = hash
                .rotate_left(27)
                .wrapping_mul(PRIME1)
                .wrapping_add(PRIME4);
            offset += 8;
        }

        if len - offset >= 4 {
            hash ^= (read_u32_le(pointer, offset, alignment) as u64).wrapping_mul(PRIME1);
            hash = hash
                .rotate_left(23)
                .wrapping_mul(PRIME2)
                .wrapping_add(PRIME3);
            offset += 4;
        }

        while offset < len {
            // SAFETY: the caller provides `len` readable bytes and the loop
            // condition proves that `offset` is in bounds.
            let byte = unsafe { *pointer.add(offset) };
            hash ^= (byte as u64).wrapping_mul(PRIME5);
            hash = hash.rotate_left(11).wrapping_mul(PRIME1);
            offset += 1;
        }

        avalanche(hash)
    }

    /// Hashes the complete representation of an explicitly stable value.
    ///
    /// This performs no allocation or copy. Primitive integer and integer-array
    /// implementations are provided on little-endian targets. A `#[repr(C)]`
    /// workload record must opt in with an `unsafe impl StableHashBytes` after
    /// proving that it contains no uninitialized padding.
    #[inline(always)]
    pub fn hash_value<T: StableHashBytes>(self, value: &T) -> u64 {
        let pointer = core::ptr::from_ref(value).cast::<u8>();
        // SAFETY: `StableHashBytes` guarantees that all `size_of::<T>()` bytes
        // beginning at `value` are initialized stable hash input. Passing the
        // type's alignment preserves wide loads for aligned primitive values
        // without weakening the unaligned `hash_bytes` contract.
        unsafe {
            self.hash_pointer(
                pointer,
                core::mem::size_of::<T>(),
                core::mem::align_of::<T>(),
            )
        }
    }
}

#[inline(always)]
fn read_u64_le(pointer: *const u8, offset: usize, alignment: usize) -> u64 {
    if alignment >= 8 {
        // SAFETY: the stable value has at least eight-byte alignment and all
        // u64 read offsets are multiples of eight.
        u64::from_le(unsafe { core::ptr::read(pointer.add(offset).cast::<u64>()) })
    } else if alignment >= 4 {
        // SAFETY: both words are four-byte aligned and within the eight bytes
        // whose availability the caller established.
        let low =
            u32::from_le(unsafe { core::ptr::read(pointer.add(offset).cast::<u32>()) }) as u64;
        let high =
            u32::from_le(unsafe { core::ptr::read(pointer.add(offset + 4).cast::<u32>()) }) as u64;
        low | (high << 32)
    } else {
        // SAFETY: the caller established eight readable bytes. `read_unaligned`
        // handles byte slices and odd-address inputs.
        u64::from_le(unsafe { core::ptr::read_unaligned(pointer.add(offset).cast::<u64>()) })
    }
}

#[inline(always)]
fn read_u32_le(pointer: *const u8, offset: usize, alignment: usize) -> u32 {
    let value = if alignment >= 4 {
        // SAFETY: the stable value has at least four-byte alignment and every
        // u32 read offset is a multiple of four.
        unsafe { core::ptr::read(pointer.add(offset).cast::<u32>()) }
    } else {
        // SAFETY: the caller established four readable bytes. `read_unaligned`
        // handles byte slices and odd-address inputs.
        unsafe { core::ptr::read_unaligned(pointer.add(offset).cast::<u32>()) }
    };
    u32::from_le(value)
}

#[inline(always)]
fn round(accumulator: u64, lane: u64) -> u64 {
    accumulator
        .wrapping_add(lane.wrapping_mul(PRIME2))
        .rotate_left(31)
        .wrapping_mul(PRIME1)
}

#[inline(always)]
fn merge_round(hash: u64, lane: u64) -> u64 {
    (hash ^ round(0, lane))
        .wrapping_mul(PRIME1)
        .wrapping_add(PRIME4)
}

#[inline(always)]
fn avalanche(mut hash: u64) -> u64 {
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(PRIME2);
    hash ^= hash >> 29;
    hash = hash.wrapping_mul(PRIME3);
    hash ^ (hash >> 32)
}
