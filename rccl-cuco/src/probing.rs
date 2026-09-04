use crate::MapKey;

const PRIME2: u32 = 0x85eb_ca77;
const PRIME3: u32 = 0xc2b2_ae3d;
const PRIME4: u32 = 0x27d4_eb2f;
const PRIME5: u32 = 0x1656_67b1;

/// cuCollections' default 32-bit xxHash for four-byte keys.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct XxHash32 {
    seed: u32,
}

impl XxHash32 {
    pub const fn new(seed: u32) -> Self {
        Self { seed }
    }

    #[inline(always)]
    pub fn hash<K: MapKey>(&self, key: K) -> u32 {
        let mut hash = self.seed.wrapping_add(PRIME5).wrapping_add(4);
        hash = hash
            .wrapping_add(key.to_bits().wrapping_mul(PRIME3))
            .rotate_left(17)
            .wrapping_mul(PRIME4);
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(PRIME2);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(PRIME3);
        hash ^ (hash >> 16)
    }
}

/// Device-safe probe construction. Indices and steps are expressed in
/// cooperative-group units; `StaticMapRef` adds the calling lane.
pub trait ProbingScheme<K: MapKey>: Copy {
    const DOUBLE_HASHING: bool;

    fn initial_group(&self, key: K, groups: usize) -> usize;
    fn step_groups(&self, key: K, groups: usize) -> usize;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinearProbing {
    hash: XxHash32,
}

impl LinearProbing {
    pub const fn new(hash: XxHash32) -> Self {
        Self { hash }
    }
}

impl<K: MapKey> ProbingScheme<K> for LinearProbing {
    const DOUBLE_HASHING: bool = false;

    #[inline(always)]
    fn initial_group(&self, key: K, groups: usize) -> usize {
        self.hash.hash(key) as usize % groups
    }

    #[inline(always)]
    fn step_groups(&self, _key: K, _groups: usize) -> usize {
        1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DoubleHashing {
    first: XxHash32,
    second: XxHash32,
}

impl Default for DoubleHashing {
    fn default() -> Self {
        Self::new(XxHash32::new(0), XxHash32::new(1))
    }
}

impl DoubleHashing {
    pub const fn new(first: XxHash32, second: XxHash32) -> Self {
        Self { first, second }
    }
}

impl<K: MapKey> ProbingScheme<K> for DoubleHashing {
    const DOUBLE_HASHING: bool = true;

    #[inline(always)]
    fn initial_group(&self, key: K, groups: usize) -> usize {
        self.first.hash(key) as usize % groups
    }

    #[inline(always)]
    fn step_groups(&self, key: K, groups: usize) -> usize {
        self.second.hash(key) as usize % (groups - 1) + 1
    }
}

pub(crate) fn valid_capacity<K, P>(requested: usize, cg: usize) -> Option<usize>
where
    K: MapKey,
    P: ProbingScheme<K>,
{
    let groups = requested.max(1).div_ceil(cg);
    let groups = if P::DOUBLE_HASHING {
        next_prime(groups)?
    } else if requested == 0 {
        // cuCollections deliberately allocates one additional group for the
        // zero-extent linear-probing case.
        groups.checked_add(1)?
    } else {
        groups
    };
    groups.checked_mul(cg)
}

fn next_prime(mut value: usize) -> Option<usize> {
    if value <= 2 {
        return Some(2);
    }
    value |= 1;
    loop {
        if is_prime(value) {
            return Some(value);
        }
        value = value.checked_add(2)?;
    }
}

fn is_prime(value: usize) -> bool {
    if value < 2 {
        return false;
    }
    for divisor in [2usize, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if value.is_multiple_of(divisor) {
            return value == divisor;
        }
    }
    let mut divisor = 41usize;
    while divisor <= value / divisor {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxhash_matches_pinned_four_byte_vectors() {
        let hash = XxHash32::new(0);
        assert_eq!(hash.hash(0_u32), 148_298_089);
        assert_eq!(hash.hash(1_u32), 4_089_149_075);
        assert_eq!(hash.hash(0x1234_5678_u32), 4_035_584_688);
    }

    #[test]
    fn capacity_matches_cuco_group_rounding() {
        assert_eq!(valid_capacity::<u32, LinearProbing>(0, 4), Some(8));
        assert_eq!(valid_capacity::<u32, DoubleHashing>(0, 4), Some(8));
        assert_eq!(valid_capacity::<u32, LinearProbing>(1, 4), Some(4));
        assert_eq!(valid_capacity::<u32, LinearProbing>(17, 4), Some(20));
        assert_eq!(valid_capacity::<u32, DoubleHashing>(17, 4), Some(20));
        assert_eq!(valid_capacity::<u32, DoubleHashing>(20, 4), Some(20));
        assert_eq!(valid_capacity::<u32, DoubleHashing>(21, 4), Some(28));
    }
}
