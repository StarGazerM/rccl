use core::{fmt, marker::PhantomData};

use cuda_core::{CudaStream, DriverError, LaunchContractError};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use rccl_memory::{
    BorrowedCudaAsyncMemoryResource, ContiguousDeviceRange, ContiguousDeviceRangeMut,
    CudaCompletion, DeviceAllocator, DeviceBuffer, MemoryError,
};

use crate::{Launch, StableHashBytes, XxHash64, bloom_backend::BloomBackend};

/// Number of 32-bit words in the audited HISA/cuCollections filter block.
pub const BLOOM_WORDS_PER_BLOCK: usize = 8;
/// Number of fingerprint bits published for each key (one in every word).
pub const BLOOM_HASH_COUNT: u32 = 8;
/// Exact block count used by the audited HISA `membership_filter`.
pub const HISA_BLOOM_BLOCKS: usize = 10_000_000;

#[inline(always)]
fn bloom_word_salt(word: usize) -> u32 {
    match word {
        0 => 0x47b6_137b,
        1 => 0x4497_4d91,
        2 => 0x8824_ad5b,
        3 => 0xa2b7_289d,
        4 => 0x7054_95c7,
        5 => 0x2df1_424b,
        6 => 0x9efc_4947,
        7 => 0x5c6_fb31,
        _ => 0,
    }
}

#[inline(always)]
pub(crate) fn bloom_block_index(hash: u64, num_blocks: usize) -> usize {
    let upper = (hash >> 32) as u32;
    (((upper as u64) * num_blocks as u64) >> 32) as usize
}

#[inline(always)]
pub(crate) fn bloom_word_pattern(hash: u64, word: usize) -> u32 {
    let lower = hash as u32;
    let bit = bloom_word_salt(word).wrapping_mul(lower) >> 27;
    1u32 << bit
}

fn blocked_false_positive_rate(keys_per_block: f64) -> f64 {
    // M is Poisson(keys_per_block). Given M occupants, one word matches with
    // probability 1-(31/32)^M. Expanding the eighth power and taking E[] gives
    // this exact Poisson-mixture expression for a blocked filter.
    const CHOOSE_8: [f64; 9] = [1.0, 8.0, 28.0, 56.0, 70.0, 56.0, 28.0, 8.0, 1.0];
    let mut result = 0.0;
    let mut q_power = 1.0;
    let mut index = 0usize;
    while index <= BLOOM_HASH_COUNT as usize {
        let term = (keys_per_block * (q_power - 1.0)).exp() * CHOOSE_8[index];
        result += if index.is_multiple_of(2) { term } else { -term };
        q_power *= 31.0 / 32.0;
        index += 1;
    }
    result.clamp(0.0, 1.0)
}

/// Failure reported while configuring, allocating, or launching a Bloom operation.
#[derive(Debug)]
pub enum BloomError {
    Memory(MemoryError),
    Launch(DriverError),
    Contract(LaunchContractError),
    ZeroBlocks,
    InvalidFalsePositiveRate(f64),
    CapacityOverflow,
    OutputTooSmall { needed: usize, available: usize },
    UnalignedStorage { address: u64 },
}

impl fmt::Display for BloomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory(error) => write!(f, "Bloom device memory error: {error}"),
            Self::Launch(error) => write!(f, "Bloom CUDA launch error: {error}"),
            Self::Contract(error) => write!(f, "Bloom kernel contract error: {error}"),
            Self::ZeroBlocks => write!(f, "a Bloom filter must contain at least one block"),
            Self::InvalidFalsePositiveRate(rate) => {
                write!(
                    f,
                    "false-positive rate must be finite and in (0, 1), got {rate}"
                )
            }
            Self::CapacityOverflow => write!(f, "Bloom filter capacity overflows the target"),
            Self::OutputTooSmall { needed, available } => write!(
                f,
                "Bloom output has {available} bytes but {needed} query results are required"
            ),
            Self::UnalignedStorage { address } => write!(
                f,
                "Bloom storage address 0x{address:x} does not satisfy the required 8-byte alignment"
            ),
        }
    }
}

impl std::error::Error for BloomError {}

impl From<MemoryError> for BloomError {
    fn from(value: MemoryError) -> Self {
        Self::Memory(value)
    }
}

impl From<DriverError> for BloomError {
    fn from(value: DriverError) -> Self {
        Self::Launch(value)
    }
}

impl From<LaunchContractError> for BloomError {
    fn from(value: LaunchContractError) -> Self {
        Self::Contract(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum BloomCapacity {
    Blocks(usize),
    FalsePositive {
        expected_insertions: usize,
        maximum_rate: f64,
    },
}

/// Configuration for the HISA-compatible blocked Bloom policy.
/// The policy is fixed deliberately to the pinned sectorized cuCollections
/// oracle: eight `u32` words per block, one salted multiplicative fingerprint
/// bit in each word, upper-hash multiply-shift block selection, device-scope
/// relaxed atomic OR publication, and `XxHash64`. `with_num_blocks` is
/// bit-for-bit capacity compatible with cuCollections; `for_capacity` finds
/// the smallest block count whose blocked-filter Poisson mixture
/// `E[(1-(31/32)^M)^8]`, `M ~ Poisson(n/B)`, meets the requested rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomFilterConfig {
    capacity: BloomCapacity,
    seed: u64,
}

impl BloomFilterConfig {
    /// Select an exact cuCollections block extent. No rounding is performed.
    pub const fn with_num_blocks(num_blocks: usize) -> Self {
        Self {
            capacity: BloomCapacity::Blocks(num_blocks),
            seed: 0,
        }
    }

    /// Select the exact audited HISA workload policy: 10,000,000 blocks and seed zero.
    pub const fn hisa() -> Self {
        Self::with_num_blocks(HISA_BLOOM_BLOCKS)
    }

    /// Size from an expected distinct insertion count and a measured FPR bound.
    pub const fn for_capacity(expected_insertions: usize, maximum_rate: f64) -> Self {
        Self {
            capacity: BloomCapacity::FalsePositive {
                expected_insertions,
                maximum_rate,
            },
            seed: 0,
        }
    }

    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub const fn seed(self) -> u64 {
        self.seed
    }

    pub fn num_blocks(self) -> Result<usize, BloomError> {
        let blocks = match self.capacity {
            BloomCapacity::Blocks(blocks) => blocks,
            BloomCapacity::FalsePositive {
                expected_insertions,
                maximum_rate,
            } => {
                if !maximum_rate.is_finite() || maximum_rate <= 0.0 || maximum_rate >= 1.0 {
                    return Err(BloomError::InvalidFalsePositiveRate(maximum_rate));
                }
                if expected_insertions == 0 {
                    1
                } else {
                    let mut low = 1usize;
                    let mut high = 1usize;
                    while blocked_false_positive_rate(expected_insertions as f64 / high as f64)
                        > maximum_rate
                    {
                        low = high.saturating_add(1);
                        high = high.checked_mul(2).ok_or(BloomError::CapacityOverflow)?;
                        if high > u32::MAX as usize {
                            high = u32::MAX as usize;
                            if blocked_false_positive_rate(expected_insertions as f64 / high as f64)
                                > maximum_rate
                            {
                                return Err(BloomError::CapacityOverflow);
                            }
                            break;
                        }
                    }
                    while low < high {
                        let middle = low + (high - low) / 2;
                        if blocked_false_positive_rate(expected_insertions as f64 / middle as f64)
                            <= maximum_rate
                        {
                            high = middle;
                        } else {
                            low = middle + 1;
                        }
                    }
                    low
                }
            }
        };
        if blocks == 0 {
            return Err(BloomError::ZeroBlocks);
        }
        if blocks > u32::MAX as usize {
            return Err(BloomError::CapacityOverflow);
        }
        blocks
            .checked_mul(BLOOM_WORDS_PER_BLOCK)
            .ok_or(BloomError::CapacityOverflow)?;
        Ok(blocks)
    }

    pub fn expected_false_positive_rate(self) -> Result<Option<f64>, BloomError> {
        let BloomCapacity::FalsePositive {
            expected_insertions,
            ..
        } = self.capacity
        else {
            return Ok(None);
        };
        let blocks = self.num_blocks()?;
        Ok(Some(blocked_false_positive_rate(
            expected_insertions as f64 / blocks as f64,
        )))
    }

    pub fn build<'stream, K, M>(
        self,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<BloomFilter<'stream, K, BorrowedCudaAsyncMemoryResource<'stream>>, BloomError>
    where
        K: StableHashBytes,
        M: BloomBackend<K>,
    {
        self.build_in(
            BorrowedCudaAsyncMemoryResource::new(launch.stream().context()),
            launch,
        )
    }

    pub fn build_in<'stream, K, A, M>(
        self,
        allocator: A,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<BloomFilter<'stream, K, A>, BloomError>
    where
        K: StableHashBytes,
        A: DeviceAllocator,
        M: BloomBackend<K>,
    {
        BloomFilter::with_config_in(self, allocator, launch)
    }
}

/// Owning stream-bound blocked Bloom filter.
pub struct BloomFilter<'stream, K, A>
where
    K: StableHashBytes,
    A: DeviceAllocator,
{
    words: DeviceBuffer<'stream, u32, A>,
    num_blocks: usize,
    seed: u64,
    marker: PhantomData<fn(&K)>,
}

impl<'stream, K> BloomFilter<'stream, K, BorrowedCudaAsyncMemoryResource<'stream>>
where
    K: StableHashBytes,
{
    pub fn with_config<M>(
        config: BloomFilterConfig,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<Self, BloomError>
    where
        M: BloomBackend<K>,
    {
        config.build(launch)
    }
}

impl<'stream, K, A> BloomFilter<'stream, K, A>
where
    K: StableHashBytes,
    A: DeviceAllocator,
{
    pub fn with_config_in<M>(
        config: BloomFilterConfig,
        allocator: A,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<Self, BloomError>
    where
        M: BloomBackend<K>,
    {
        let num_blocks = config.num_blocks()?;
        let words_len = num_blocks
            .checked_mul(BLOOM_WORDS_PER_BLOCK)
            .ok_or(BloomError::CapacityOverflow)?;
        let words = DeviceBuffer::zeroed_in(allocator, launch.stream(), words_len)?;
        if !words
            .cu_deviceptr()
            .is_multiple_of(core::mem::align_of::<u64>() as u64)
        {
            return Err(BloomError::UnalignedStorage {
                address: words.cu_deviceptr(),
            });
        }
        Ok(Self {
            words,
            num_blocks,
            seed: config.seed(),
            marker: PhantomData,
        })
    }

    pub const fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub const fn word_len(&self) -> usize {
        self.num_blocks * BLOOM_WORDS_PER_BLOCK
    }

    pub const fn bit_capacity(&self) -> usize {
        self.word_len() * u32::BITS as usize
    }

    pub const fn seed(&self) -> u64 {
        self.seed
    }

    pub fn stream(&self) -> &'stream CudaStream {
        self.words.stream()
    }

    pub fn clear<M>(&mut self, launch: &Launch<'stream, '_, M>) -> Result<(), BloomError>
    where
        M: BloomBackend<K>,
    {
        self.clear_async(launch)?.wait()
    }

    pub fn clear_async<'borrow, 'program, M>(
        &'borrow mut self,
        launch: &'borrow Launch<'stream, 'program, M>,
    ) -> Result<PendingBloomOperation<'borrow>, BloomError>
    where
        M: BloomBackend<K>,
    {
        self.ensure_launch(launch)?;
        launch
            .module()
            .bloom_clear(launch.stream(), &mut self.words)?;
        Ok(PendingBloomOperation::submitted(launch.stream()))
    }

    /// Add a device-resident range of stable records.
    pub fn add<M, R>(&mut self, keys: &R, launch: &Launch<'stream, '_, M>) -> Result<(), BloomError>
    where
        M: BloomBackend<K>,
        R: ContiguousDeviceRange<K> + ?Sized,
    {
        self.add_async(keys, launch)?.wait()
    }

    /// Rust-style alias for [`Self::add`].
    pub fn insert<M, R>(
        &mut self,
        keys: &R,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), BloomError>
    where
        M: BloomBackend<K>,
        R: ContiguousDeviceRange<K> + ?Sized,
    {
        self.add(keys, launch)
    }

    pub fn add_async<'borrow, 'program, M, R>(
        &'borrow mut self,
        keys: &'borrow R,
        launch: &'borrow Launch<'stream, 'program, M>,
    ) -> Result<PendingBloomOperation<'borrow>, BloomError>
    where
        M: BloomBackend<K>,
        R: ContiguousDeviceRange<K> + ?Sized,
    {
        self.ensure_launch(launch)?;
        keys.ensure_usable_on(launch.stream())?;
        if keys.is_empty() {
            return Ok(PendingBloomOperation::ready(launch.stream()));
        }
        launch.module().bloom_add(
            launch.stream(),
            &mut self.words,
            keys,
            self.num_blocks,
            self.seed,
        )?;
        Ok(PendingBloomOperation::submitted(launch.stream()))
    }

    /// Write one byte (`0` or `1`) per query into device memory.
    pub fn contains<M, R, O>(
        &self,
        keys: &R,
        output: &mut O,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), BloomError>
    where
        M: BloomBackend<K>,
        R: ContiguousDeviceRange<K> + ?Sized,
        O: ContiguousDeviceRangeMut<u8> + ?Sized,
    {
        self.contains_async(keys, output, launch)?.wait()
    }

    pub fn contains_async<'borrow, 'program, M, R, O>(
        &'borrow self,
        keys: &'borrow R,
        output: &'borrow mut O,
        launch: &'borrow Launch<'stream, 'program, M>,
    ) -> Result<PendingBloomOperation<'borrow>, BloomError>
    where
        M: BloomBackend<K>,
        R: ContiguousDeviceRange<K> + ?Sized,
        O: ContiguousDeviceRangeMut<u8> + ?Sized,
    {
        self.ensure_launch(launch)?;
        keys.ensure_usable_on(launch.stream())?;
        output.ensure_usable_on(launch.stream())?;
        if output.len() < keys.len() {
            return Err(BloomError::OutputTooSmall {
                needed: keys.len(),
                available: output.len(),
            });
        }
        if keys.is_empty() {
            return Ok(PendingBloomOperation::ready(launch.stream()));
        }
        launch.module().bloom_contains(
            launch.stream(),
            &self.words,
            keys,
            output,
            self.num_blocks,
            self.seed,
        )?;
        Ok(PendingBloomOperation::submitted(launch.stream()))
    }

    #[doc(hidden)]
    pub fn add_omitting_word_for_test<M, R>(
        &mut self,
        keys: &R,
        omitted_word: u32,
        launch: &Launch<'stream, '_, M>,
    ) -> Result<(), BloomError>
    where
        M: BloomBackend<K>,
        R: ContiguousDeviceRange<K> + ?Sized,
    {
        self.ensure_launch(launch)?;
        keys.ensure_usable_on(launch.stream())?;
        if omitted_word >= BLOOM_WORDS_PER_BLOCK as u32 {
            return Err(BloomError::CapacityOverflow);
        }
        if !keys.is_empty() {
            launch.module().bloom_add_fault(
                launch.stream(),
                &mut self.words,
                keys,
                self.num_blocks,
                self.seed,
                omitted_word,
            )?;
            launch.stream().synchronize()?;
        }
        Ok(())
    }

    fn ensure_launch<M>(&self, launch: &Launch<'stream, '_, M>) -> Result<(), BloomError> {
        self.words.ensure_usable_on(launch.stream())?;
        Ok(())
    }
}

struct PendingBloomResources<'borrow>(PhantomData<&'borrow mut ()>);

/// Completion guard retaining every Bloom operation borrow until synchronization.
pub struct PendingBloomOperation<'borrow>(
    CudaCompletion<'borrow, (), BloomError, PendingBloomResources<'borrow>>,
);

impl<'borrow> PendingBloomOperation<'borrow> {
    fn submitted(stream: &'borrow CudaStream) -> Self {
        // SAFETY: the caller has submitted work only to `stream`; the returned
        // lifetime retains its filter/input/output/program borrows until wait or drop.
        Self(unsafe {
            CudaCompletion::from_submission(stream, PendingBloomResources(PhantomData), |_| Ok(()))
        })
    }

    fn ready(stream: &'borrow CudaStream) -> Self {
        // SAFETY: empty operations submit no GPU work.
        Self(unsafe { CudaCompletion::ready(stream, PendingBloomResources(PhantomData), Ok(())) })
    }

    pub fn wait(self) -> Result<(), BloomError> {
        self.0.wait()
    }

    pub fn is_complete(&self) -> Result<bool, DriverError> {
        self.0.is_complete()
    }
}

/// Copyable device-side mutation reference using device-scope relaxed atomics.
#[derive(Clone, Copy)]
pub struct BloomFilterRef<'a, K: StableHashBytes> {
    words: &'a [DeviceAtomicU32],
    num_blocks: usize,
    seed: u64,
    marker: PhantomData<fn(&K)>,
}

impl<'a, K: StableHashBytes> BloomFilterRef<'a, K> {
    /// # Safety
    ///
    /// `words` must contain exactly `num_blocks * 8` elements, `num_blocks`
    /// must be non-zero, and the allocation may only be mutated atomically.
    pub const unsafe fn new(words: &'a [DeviceAtomicU32], num_blocks: usize, seed: u64) -> Self {
        Self {
            words,
            num_blocks,
            seed,
            marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn add(&self, key: &K) {
        let hash = XxHash64::new(self.seed).hash_value(key);
        let base = bloom_block_index(hash, self.num_blocks) * BLOOM_WORDS_PER_BLOCK;
        self.publish(base, hash, None);
    }

    /// Add one explicitly encoded key without allocating or copying it.
    #[inline(always)]
    pub fn add_encoded(&self, bytes: &[u8]) {
        let hash = XxHash64::new(self.seed).hash_bytes(bytes);
        let base = bloom_block_index(hash, self.num_blocks) * BLOOM_WORDS_PER_BLOCK;
        self.publish(base, hash, None);
    }

    #[inline(always)]
    pub(crate) fn add_omitting(&self, key: &K, omitted_word: u32) {
        let hash = XxHash64::new(self.seed).hash_value(key);
        let base = bloom_block_index(hash, self.num_blocks) * BLOOM_WORDS_PER_BLOCK;
        self.publish(base, hash, Some(omitted_word));
    }

    #[inline(always)]
    fn publish(&self, base: usize, hash: u64, omitted_word: Option<u32>) {
        let mut word = 0usize;
        while word < BLOOM_WORDS_PER_BLOCK {
            if omitted_word != Some(word as u32) {
                // SAFETY: construction proves an exact non-zero block extent,
                // and multiply-shift returns a block in that extent.
                unsafe { self.words.get_unchecked(base + word) }
                    .fetch_or(bloom_word_pattern(hash, word), AtomicOrdering::Relaxed);
            }
            word += 1;
        }
    }
}

/// Copyable lookup-only device reference.
#[derive(Clone, Copy)]
pub struct BloomFilterReadRef<'a, K: StableHashBytes> {
    words: &'a [u32],
    num_blocks: usize,
    seed: u64,
    marker: PhantomData<fn(&K)>,
}

impl<'a, K: StableHashBytes> BloomFilterReadRef<'a, K> {
    pub const fn new(words: &'a [u32], num_blocks: usize, seed: u64) -> Self {
        Self {
            words,
            num_blocks,
            seed,
            marker: PhantomData,
        }
    }

    #[inline(always)]
    pub fn contains(&self, key: &K) -> bool {
        self.contains_hash(XxHash64::new(self.seed).hash_value(key))
    }

    /// Query one explicitly encoded key without allocating or copying it.
    #[inline(always)]
    pub fn contains_encoded(&self, bytes: &[u8]) -> bool {
        self.contains_hash(XxHash64::new(self.seed).hash_bytes(bytes))
    }
    /// Query through the 32-byte-aligned block storage owned by `BloomFilter`.
    ///
    /// # Safety
    ///
    /// `self.words` must begin at an address aligned to at least 16 bytes. Each
    /// filter block starts 32 bytes after the previous block, preserving that
    /// alignment.
    #[inline(always)]
    pub(crate) unsafe fn contains_aligned(&self, key: &K) -> bool {
        // SAFETY: inherited from this method's caller.
        unsafe { self.contains_hash_aligned(XxHash64::new(self.seed).hash_value(key)) }
    }

    #[inline(always)]
    unsafe fn contains_hash_aligned(&self, hash: u64) -> bool {
        let base = bloom_block_index(hash, self.num_blocks) * BLOOM_WORDS_PER_BLOCK;
        // Keep adjacent word pairs whole through their comparisons so the
        // compiler emits four aligned 64-bit loads instead of eight scalar
        // loads. This is the same sector layout and evaluation order as the
        // pinned cuCollections vector load without widening the arithmetic.
        // SAFETY: upheld by the caller; multiply-shift bounds `base` and the
        // exact filter extent supplies all eight words.
        let stored = unsafe { self.words.as_ptr().add(base).cast::<[u64; 4]>().read() };
        let expected0 =
            bloom_word_pattern(hash, 0) as u64 | (bloom_word_pattern(hash, 1) as u64) << 32;
        let expected1 =
            bloom_word_pattern(hash, 2) as u64 | (bloom_word_pattern(hash, 3) as u64) << 32;
        let expected2 =
            bloom_word_pattern(hash, 4) as u64 | (bloom_word_pattern(hash, 5) as u64) << 32;
        let expected3 =
            bloom_word_pattern(hash, 6) as u64 | (bloom_word_pattern(hash, 7) as u64) << 32;
        ((stored[0] & expected0) ^ expected0)
            | ((stored[1] & expected1) ^ expected1)
            | ((stored[2] & expected2) ^ expected2)
            | ((stored[3] & expected3) ^ expected3)
            == 0
    }

    #[inline(always)]
    fn contains_hash(&self, hash: u64) -> bool {
        let base = bloom_block_index(hash, self.num_blocks) * BLOOM_WORDS_PER_BLOCK;
        // SAFETY: multiply-shift bounds `base`, construction guarantees eight
        // words per block, and `[u32; 8]` needs no stronger alignment than u32.
        let stored = unsafe { self.words.as_ptr().add(base).cast::<[u32; 8]>().read() };
        let mut missing = 0u32;
        missing |= (stored[0] & bloom_word_pattern(hash, 0)) ^ bloom_word_pattern(hash, 0);
        missing |= (stored[1] & bloom_word_pattern(hash, 1)) ^ bloom_word_pattern(hash, 1);
        missing |= (stored[2] & bloom_word_pattern(hash, 2)) ^ bloom_word_pattern(hash, 2);
        missing |= (stored[3] & bloom_word_pattern(hash, 3)) ^ bloom_word_pattern(hash, 3);
        missing |= (stored[4] & bloom_word_pattern(hash, 4)) ^ bloom_word_pattern(hash, 4);
        missing |= (stored[5] & bloom_word_pattern(hash, 5)) ^ bloom_word_pattern(hash, 5);
        missing |= (stored[6] & bloom_word_pattern(hash, 6)) ^ bloom_word_pattern(hash, 6);
        missing |= (stored[7] & bloom_word_pattern(hash, 7)) ^ bloom_word_pattern(hash, 7);
        missing == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_hisa_policy_is_not_rounded() {
        let policy = BloomFilterConfig::hisa();
        assert_eq!(policy.num_blocks().unwrap(), 10_000_000);
        assert_eq!(policy.seed(), 0);
        assert_eq!(BLOOM_WORDS_PER_BLOCK, 8);
        assert_eq!(BLOOM_HASH_COUNT, 8);
    }

    #[test]
    fn capacity_policy_respects_requested_estimate() {
        for (items, bound) in [(0, 0.01), (1, 0.1), (10_000, 0.01), (1_000_000, 0.001)] {
            let policy = BloomFilterConfig::for_capacity(items, bound);
            assert!(policy.expected_false_positive_rate().unwrap().unwrap() <= bound);
        }
    }

    #[test]
    fn invalid_policies_are_rejected() {
        assert!(matches!(
            BloomFilterConfig::with_num_blocks(0).num_blocks(),
            Err(BloomError::ZeroBlocks)
        ));
        for rate in [0.0, 1.0, -0.5, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                BloomFilterConfig::for_capacity(1, rate).num_blocks(),
                Err(BloomError::InvalidFalsePositiveRate(_))
            ));
        }
    }

    #[test]
    fn deterministic_fault_oracle_omits_a_required_bit() {
        let key = 0x0123_4567_89ab_cdefu64;
        let hash = XxHash64::new(17).hash_value(&key);
        let mut words = [0u32; BLOOM_WORDS_PER_BLOCK];
        for word in 0..BLOOM_WORDS_PER_BLOCK {
            if word != 3 {
                words[word] |= bloom_word_pattern(hash, word);
            }
        }
        let reference = BloomFilterReadRef::<u64>::new(&words, 1, 17);
        assert!(
            !reference.contains(&key),
            "the no-false-negative gate must catch omitted publication"
        );
        words[3] |= bloom_word_pattern(hash, 3);
        assert!(BloomFilterReadRef::<u64>::new(&words, 1, 17).contains(&key));
    }
}
