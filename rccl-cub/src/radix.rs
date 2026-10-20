use core::ops::DerefMut;

use cuda_core::{CudaStream, DeviceCopy};
use cuda_host::{KernelSliceArg, KernelSliceArgMut};

use crate::{
    DeviceAlgorithmError, DeviceRadixSortBufferWorkspace, DeviceRadixSortKeysWorkspace,
    DeviceRadixSortPairsWorkspace, DoubleBuffer,
    workspace::{WorkspaceBuffer, radix_keys_policy, radix_pairs_policy, radix_reg_bound},
};

/// Reject targets outside the pinned Policy800 implementation without adding
/// an architecture value or a second configuration channel.
pub(crate) fn ensure_radix_target() -> Result<(), DeviceAlgorithmError> {
    #[cfg(any(not(cuda_arch_ge_80), cuda_arch_ge_90))]
    return Err(DeviceAlgorithmError::UnsupportedTarget(
        "radix sort supports sm_80 through sm_89; select the target with cargo oxide --arch",
    ));

    #[cfg(all(cuda_arch_ge_80, not(cuda_arch_ge_90)))]
    Ok(())
}

mod sealed {
    pub trait RadixKey {}
    pub trait RadixValue {}
}

/// A primitive key supported by the pinned 32/64-bit Policy800 radix graph.
///
/// The trait is closed because supporting another representation requires an
/// ordering-preserving bit transform, not merely a Rust memory layout. Type
/// aliases work without exposing that policy through the public API.
pub trait RadixKey: sealed::RadixKey + DeviceCopy {
    type Bits: DeviceCopy;

    const BITS: u32;

    fn zero() -> Self;
    fn default_ordered() -> Self::Bits;
    fn minimum_ordered() -> Self::Bits;
    fn to_ordered_ascending(key: Self) -> Self::Bits;
    fn from_ordered_ascending(bits: Self::Bits) -> Self;
    fn to_ordered_descending(key: Self) -> Self::Bits;
    fn from_ordered_descending(bits: Self::Bits) -> Self;
    fn digit(bits: Self::Bits, bit: u32, width: u32) -> u32;
    fn rank_digit(bits: Self::Bits, bit: u32, width: u32) -> u32;
}

macro_rules! radix_key {
    (
        $key:ty, $bits:ty,
        $to_asc:path, $from_asc:path,
        $to_desc:path, $from_desc:path,
        $digit:path, $rank:path
    ) => {
        impl sealed::RadixKey for $key {}

        impl RadixKey for $key {
            type Bits = $bits;

            const BITS: u32 = <$bits>::BITS;

            #[inline(always)]
            fn zero() -> Self {
                0 as $key
            }

            #[inline(always)]
            fn default_ordered() -> Self::Bits {
                <$bits>::MAX
            }

            #[inline(always)]
            fn minimum_ordered() -> Self::Bits {
                <$bits>::MIN
            }

            #[inline(always)]
            fn to_ordered_ascending(key: Self) -> Self::Bits {
                $to_asc(key)
            }

            #[inline(always)]
            fn from_ordered_ascending(bits: Self::Bits) -> Self {
                $from_asc(bits)
            }

            #[inline(always)]
            fn to_ordered_descending(key: Self) -> Self::Bits {
                $to_desc(key)
            }

            #[inline(always)]
            fn from_ordered_descending(bits: Self::Bits) -> Self {
                $from_desc(bits)
            }

            #[inline(always)]
            fn digit(bits: Self::Bits, bit: u32, width: u32) -> u32 {
                $digit(bits, bit, width)
            }

            #[inline(always)]
            fn rank_digit(bits: Self::Bits, bit: u32, width: u32) -> u32 {
                $rank(bits, bit, width)
            }
        }
    };
}

use crate::ordering_backend as backend;

radix_key!(
    u32,
    u32,
    backend::radix_u32_to_ordered_ascending,
    backend::radix_u32_from_ordered_ascending,
    backend::radix_u32_to_ordered_descending,
    backend::radix_u32_from_ordered_descending,
    backend::radix_digit_masked_u32,
    backend::radix_rank_digit_u32
);
radix_key!(
    i32,
    u32,
    backend::radix_i32_to_ordered_ascending,
    backend::radix_i32_from_ordered_ascending,
    backend::radix_i32_to_ordered_descending,
    backend::radix_i32_from_ordered_descending,
    backend::radix_digit_masked_u32,
    backend::radix_rank_digit_u32
);
radix_key!(
    f32,
    u32,
    backend::radix_f32_to_ordered_ascending,
    backend::radix_f32_from_ordered_ascending,
    backend::radix_f32_to_ordered_descending,
    backend::radix_f32_from_ordered_descending,
    backend::radix_float_digit_masked_u32,
    backend::radix_float_rank_digit_u32
);
radix_key!(
    u64,
    u64,
    backend::radix_u64_to_ordered_ascending,
    backend::radix_u64_from_ordered_ascending,
    backend::radix_u64_to_ordered_descending,
    backend::radix_u64_from_ordered_descending,
    backend::radix_digit_masked_u64,
    backend::radix_rank_digit_u64
);
radix_key!(
    i64,
    u64,
    backend::radix_i64_to_ordered_ascending,
    backend::radix_i64_from_ordered_ascending,
    backend::radix_i64_to_ordered_descending,
    backend::radix_i64_from_ordered_descending,
    backend::radix_digit_masked_u64,
    backend::radix_rank_digit_u64
);
radix_key!(
    f64,
    u64,
    backend::radix_f64_to_ordered_ascending,
    backend::radix_f64_from_ordered_ascending,
    backend::radix_f64_to_ordered_descending,
    backend::radix_f64_from_ordered_descending,
    backend::radix_float_digit_masked_u64,
    backend::radix_float_rank_digit_u64
);

/// Value layouts for which the generated pair kernel has a pinned static
/// shared-memory specialization. This is layout capability, not strategy.
pub trait RadixValue: sealed::RadixValue + DeviceCopy + Default {}

macro_rules! radix_values {
    ($($value:ty),+ $(,)?) => {$(
        impl sealed::RadixValue for $value {}
        impl RadixValue for $value {}
    )+};
}

radix_values!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64,
);

#[doc(hidden)]
pub const fn radix_items(nominal: usize, item_size: usize) -> usize {
    radix_reg_bound(384, nominal, item_size).1
}

#[doc(hidden)]
pub const fn radix_single_items(item_size: usize) -> usize {
    radix_reg_bound(256, 19, item_size).1
}

#[doc(hidden)]
pub const fn radix_shared_bytes(items: usize, item_size: usize) -> usize {
    384 * items * item_size + 256 * core::mem::size_of::<u64>()
}

/// Common generated-kernel stages. Architecture-independent validation and
/// pass control live in the ordinary functions below.
#[doc(hidden)]
pub trait RadixKernel<K: RadixKey, const DESC: bool> {
    fn launch_radix_init<C, B>(
        &self,
        stream: &CudaStream,
        counters: &mut C,
        bins: &mut B,
        passes: usize,
    ) -> Result<(), DeviceAlgorithmError>
    where
        C: KernelSliceArgMut<Elem = i32>,
        B: KernelSliceArgMut<Elem = u64>;

    fn launch_radix_histogram_4<Keys, Bins, const ITEMS: usize>(
        &self,
        stream: &CudaStream,
        keys: &Keys,
        bins: &mut Bins,
        begin_bit: u32,
        end_bit: u32,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Keys: KernelSliceArg<Elem = K>,
        Bins: KernelSliceArgMut<Elem = u64>;

    fn launch_radix_histogram_8<Keys, Bins, const ITEMS: usize>(
        &self,
        stream: &CudaStream,
        keys: &Keys,
        bins: &mut Bins,
        begin_bit: u32,
        end_bit: u32,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Keys: KernelSliceArg<Elem = K>,
        Bins: KernelSliceArgMut<Elem = u64>;

    fn launch_radix_exclusive_sum<Bins, const PASSES: usize>(
        &self,
        stream: &CudaStream,
        bins: &mut Bins,
        passes: usize,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Bins: KernelSliceArgMut<Elem = u64>;
}

#[doc(hidden)]
pub trait RadixKeysKernel<K: RadixKey, const DESC: bool>: RadixKernel<K, DESC> {
    #[allow(clippy::too_many_arguments)]
    fn launch_radix_single_keys<Input, Output, const ITEMS: usize>(
        &self,
        stream: &CudaStream,
        input: &Input,
        output: &mut Output,
        begin_bit: u32,
        end_bit: u32,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Input: KernelSliceArg<Elem = K>,
        Output: KernelSliceArgMut<Elem = K>;

    #[allow(clippy::too_many_arguments)]
    fn launch_radix_onesweep_keys<
        Lookback,
        Counters,
        Bins,
        Input,
        Output,
        const ITEMS: usize,
        const UNION_BYTES: usize,
        const SHARED_BYTES: usize,
    >(
        &self,
        stream: &CudaStream,
        lookback: &mut Lookback,
        counters: &mut Counters,
        bins: &Bins,
        input: &Input,
        output: &mut Output,
        bin_offset: usize,
        pass: usize,
        current_bit: u32,
        num_bits: u32,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Lookback: KernelSliceArgMut<Elem = i32>,
        Counters: KernelSliceArgMut<Elem = i32>,
        Bins: KernelSliceArg<Elem = u64>,
        Input: KernelSliceArg<Elem = K>,
        Output: KernelSliceArgMut<Elem = K>;
}

#[doc(hidden)]
pub trait RadixPairsKernel<K: RadixKey, V: RadixValue, const DESC: bool>:
    RadixKernel<K, DESC>
{
    #[allow(clippy::too_many_arguments)]
    fn launch_radix_single_pairs<
        InputKeys,
        InputValues,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
    >(
        &self,
        stream: &CudaStream,
        input_keys: &InputKeys,
        input_values: &InputValues,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
        begin_bit: u32,
        end_bit: u32,
    ) -> Result<(), DeviceAlgorithmError>
    where
        InputKeys: KernelSliceArg<Elem = K>,
        InputValues: KernelSliceArg<Elem = V>,
        OutputKeys: KernelSliceArgMut<Elem = K>,
        OutputValues: KernelSliceArgMut<Elem = V>;

    #[allow(clippy::too_many_arguments)]
    fn launch_radix_onesweep_pairs<
        Lookback,
        Counters,
        Bins,
        InputKeys,
        InputValues,
        OutputKeys,
        OutputValues,
        const ITEMS: usize,
        const UNION_BYTES: usize,
        const SHARED_BYTES: usize,
    >(
        &self,
        stream: &CudaStream,
        lookback: &mut Lookback,
        counters: &mut Counters,
        bins: &Bins,
        input_keys: &InputKeys,
        input_values: &InputValues,
        output_keys: &mut OutputKeys,
        output_values: &mut OutputValues,
        bin_offset: usize,
        pass: usize,
        current_bit: u32,
        num_bits: u32,
    ) -> Result<(), DeviceAlgorithmError>
    where
        Lookback: KernelSliceArgMut<Elem = i32>,
        Counters: KernelSliceArgMut<Elem = i32>,
        Bins: KernelSliceArg<Elem = u64>,
        InputKeys: KernelSliceArg<Elem = K>,
        InputValues: KernelSliceArg<Elem = V>,
        OutputKeys: KernelSliceArgMut<Elem = K>,
        OutputValues: KernelSliceArgMut<Elem = V>;
}

fn validate_radix(
    len: usize,
    bit_range: &core::ops::Range<u32>,
    key_bits: u32,
    tile_items: usize,
) -> Result<usize, DeviceAlgorithmError> {
    if bit_range.start > bit_range.end || bit_range.end > key_bits {
        return Err(DeviceAlgorithmError::InvalidBitRange {
            begin: bit_range.start,
            end: bit_range.end,
            key_bits,
        });
    }
    let portion_size = ((1usize << 28) - 1) / tile_items * tile_items;
    if len > portion_size {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "Policy800 one-portion maximum",
            expected: portion_size,
            actual: len,
        });
    }
    Ok(((bit_range.end - bit_range.start) as usize).div_ceil(8))
}

fn validate_controls<B, L, C>(
    bins: &B,
    lookback: &L,
    counters: &C,
    passes: usize,
    blocks: usize,
) -> Result<(), DeviceAlgorithmError>
where
    B: KernelSliceArg<Elem = u64>,
    L: KernelSliceArg<Elem = i32>,
    C: KernelSliceArg<Elem = i32>,
{
    let required_bins = passes * 256;
    if bins.len() < required_bins {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "radix bins",
            required: required_bins,
            actual: bins.len(),
        });
    }
    // Lookback slots are addressed `block * 256 + digit` for the blocks of this
    // sort and the whole buffer is zeroed before every pass, so a workspace
    // sized for a longer input serves any shorter one.
    if lookback.len() < blocks * 256 {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "radix lookback",
            required: blocks * 256,
            actual: lookback.len(),
        });
    }
    if counters.len() < passes {
        return Err(DeviceAlgorithmError::WorkspaceTooSmall {
            role: "radix pass counters",
            required: passes,
            actual: counters.len(),
        });
    }
    Ok(())
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_radix_sort_keys_buffer<
    K,
    M,
    First,
    Second,
    B,
    L,
    C,
    const SINGLE_ITEMS: usize,
    const ITEMS: usize,
    const UNION_BYTES: usize,
    const SHARED_BYTES: usize,
    const PASSES: usize,
    const DESC: bool,
>(
    module: &M,
    stream: &CudaStream,
    buffers: &mut DoubleBuffer<First, Second>,
    bit_range: core::ops::Range<u32>,
    workspace: &mut DeviceRadixSortBufferWorkspace<B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    M: RadixKeysKernel<K, DESC>,
    First: DerefMut,
    Second: DerefMut,
    First::Target: KernelSliceArgMut<Elem = K> + Sized,
    Second::Target: KernelSliceArgMut<Elem = K> + Sized,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: KernelSliceArgMut<Elem = u64>,
    L::Range: KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: KernelSliceArgMut<Elem = i32>,
{
    let len = buffers.len();
    let policy = radix_keys_policy::<K>();
    debug_assert_eq!(policy.single_tile_items, 256 * SINGLE_ITEMS);
    debug_assert_eq!(policy.tile_items, 384 * ITEMS);
    let passes = validate_radix(len, &bit_range, K::BITS, policy.tile_items)?;
    if len == 0 || passes == 0 {
        return Ok(());
    }
    if len <= policy.single_tile_items {
        let current_is_first = buffers.current_is_first();
        let (first, second) = buffers.first_and_second_mut();
        if current_is_first {
            module.launch_radix_single_keys::<_, _, SINGLE_ITEMS>(
                stream,
                &**first,
                &mut **second,
                bit_range.start,
                bit_range.end,
            )?;
        } else {
            module.launch_radix_single_keys::<_, _, SINGLE_ITEMS>(
                stream,
                &**second,
                &mut **first,
                bit_range.start,
                bit_range.end,
            )?;
        }
        buffers.swap();
        return Ok(());
    }

    let blocks = len.div_ceil(policy.tile_items);
    let (bins, lookback, counters) = workspace.__parts_mut();
    validate_controls(&*bins, &*lookback, &*counters, passes, blocks)?;
    module.launch_radix_init(stream, counters, bins, passes)?;
    debug_assert_eq!(policy.passes, PASSES);
    if PASSES == 4 {
        module.launch_radix_histogram_4::<_, _, 16>(
            stream,
            &**buffers.first(),
            bins,
            bit_range.start,
            bit_range.end,
        )?;
    } else {
        module.launch_radix_histogram_8::<_, _, 16>(
            stream,
            &**buffers.first(),
            bins,
            bit_range.start,
            bit_range.end,
        )?;
    }
    module.launch_radix_exclusive_sum::<_, PASSES>(stream, bins, passes)?;

    for pass in 0..passes {
        crate::__private::ZeroRadixLookback::zero_radix_lookback(&mut *lookback, stream)?;
        let current_bit = bit_range.start + (pass * 8) as u32;
        let num_bits = (bit_range.end - current_bit).min(8);
        let current_is_first = buffers.current_is_first();
        let (first, second) = buffers.first_and_second_mut();
        if current_is_first {
            module.launch_radix_onesweep_keys::<_, _, _, _, _, ITEMS, UNION_BYTES, SHARED_BYTES>(
                stream,
                lookback,
                counters,
                &*bins,
                &**first,
                &mut **second,
                pass * 256,
                pass,
                current_bit,
                num_bits,
            )?;
        } else {
            module.launch_radix_onesweep_keys::<_, _, _, _, _, ITEMS, UNION_BYTES, SHARED_BYTES>(
                stream,
                lookback,
                counters,
                &*bins,
                &**second,
                &mut **first,
                pass * 256,
                pass,
                current_bit,
                num_bits,
            )?;
        }
        buffers.swap();
    }
    Ok(())
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_radix_sort_keys<
    K,
    M,
    S,
    B,
    L,
    C,
    Keys,
    const SINGLE_ITEMS: usize,
    const ITEMS: usize,
    const UNION_BYTES: usize,
    const SHARED_BYTES: usize,
    const PASSES: usize,
    const DESC: bool,
>(
    module: &M,
    stream: &CudaStream,
    keys: &mut Keys,
    workspace: &mut DeviceRadixSortKeysWorkspace<K, S, B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    M: RadixKeysKernel<K, DESC>,
    Keys: KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRangeMut<K>,
    S: WorkspaceBuffer<K>,
    S::Range: KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRange<K>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: KernelSliceArgMut<Elem = u64>,
    L::Range: KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: KernelSliceArgMut<Elem = i32>,
{
    let len = KernelSliceArg::len(keys);
    let (scratch, controls) = workspace.__parts_mut();
    let scratch_len = scratch.len();
    if scratch_len != len {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "radix scratch keys",
            expected: len,
            actual: scratch_len,
        });
    }
    let mut buffers = DoubleBuffer::__from_equal_parts(&mut *keys, &mut *scratch, len);
    device_radix_sort_keys_buffer::<
        K,
        M,
        _,
        _,
        B,
        L,
        C,
        SINGLE_ITEMS,
        ITEMS,
        UNION_BYTES,
        SHARED_BYTES,
        PASSES,
        DESC,
    >(module, stream, &mut buffers, 0..K::BITS, controls)?;
    let current_is_first = buffers.current_is_first();
    if !current_is_first {
        crate::__private::copy_device_range::<K, _, _>(&*scratch, &mut *keys, stream)
            .map_err(DeviceAlgorithmError::Memory)?;
    }
    Ok(())
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_radix_sort_pairs_buffer<
    K,
    V,
    M,
    FirstKeys,
    FirstValues,
    SecondKeys,
    SecondValues,
    B,
    L,
    C,
    const SINGLE_ITEMS: usize,
    const ITEMS: usize,
    const UNION_BYTES: usize,
    const SHARED_BYTES: usize,
    const PASSES: usize,
    const DESC: bool,
>(
    module: &M,
    stream: &CudaStream,
    buffers: &mut DoubleBuffer<(FirstKeys, FirstValues), (SecondKeys, SecondValues)>,
    bit_range: core::ops::Range<u32>,
    workspace: &mut DeviceRadixSortBufferWorkspace<B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    V: RadixValue,
    M: RadixPairsKernel<K, V, DESC>,
    FirstKeys: DerefMut,
    FirstValues: DerefMut,
    SecondKeys: DerefMut,
    SecondValues: DerefMut,
    FirstKeys::Target: KernelSliceArgMut<Elem = K> + Sized,
    FirstValues::Target: KernelSliceArgMut<Elem = V> + Sized,
    SecondKeys::Target: KernelSliceArgMut<Elem = K> + Sized,
    SecondValues::Target: KernelSliceArgMut<Elem = V> + Sized,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: KernelSliceArgMut<Elem = u64>,
    L::Range: KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: KernelSliceArgMut<Elem = i32>,
{
    let len = buffers.len();
    let policy = radix_pairs_policy::<K, V>();
    debug_assert_eq!(policy.single_tile_items, 256 * SINGLE_ITEMS);
    debug_assert_eq!(policy.tile_items, 384 * ITEMS);
    let passes = validate_radix(len, &bit_range, K::BITS, policy.tile_items)?;
    if len == 0 || passes == 0 {
        return Ok(());
    }
    if len <= policy.single_tile_items {
        let current_is_first = buffers.current_is_first();
        let (first, second) = buffers.first_and_second_mut();
        let (first_keys, first_values) = first;
        let (second_keys, second_values) = second;
        if current_is_first {
            module.launch_radix_single_pairs::<_, _, _, _, SINGLE_ITEMS>(
                stream,
                &**first_keys,
                &**first_values,
                &mut **second_keys,
                &mut **second_values,
                bit_range.start,
                bit_range.end,
            )?;
        } else {
            module.launch_radix_single_pairs::<_, _, _, _, SINGLE_ITEMS>(
                stream,
                &**second_keys,
                &**second_values,
                &mut **first_keys,
                &mut **first_values,
                bit_range.start,
                bit_range.end,
            )?;
        }
        buffers.swap();
        return Ok(());
    }

    let blocks = len.div_ceil(policy.tile_items);
    let (bins, lookback, counters) = workspace.__parts_mut();
    validate_controls(&*bins, &*lookback, &*counters, passes, blocks)?;
    module.launch_radix_init(stream, counters, bins, passes)?;
    debug_assert_eq!(policy.passes, PASSES);
    if PASSES == 4 {
        module.launch_radix_histogram_4::<_, _, 16>(
            stream,
            &*buffers.first().0,
            bins,
            bit_range.start,
            bit_range.end,
        )?;
    } else {
        module.launch_radix_histogram_8::<_, _, 16>(
            stream,
            &*buffers.first().0,
            bins,
            bit_range.start,
            bit_range.end,
        )?;
    }
    module.launch_radix_exclusive_sum::<_, PASSES>(stream, bins, passes)?;

    for pass in 0..passes {
        crate::__private::ZeroRadixLookback::zero_radix_lookback(&mut *lookback, stream)?;
        let current_bit = bit_range.start + (pass * 8) as u32;
        let num_bits = (bit_range.end - current_bit).min(8);
        let current_is_first = buffers.current_is_first();
        let (first, second) = buffers.first_and_second_mut();
        let (first_keys, first_values) = first;
        let (second_keys, second_values) = second;
        if current_is_first {
            module.launch_radix_onesweep_pairs::<
                _, _, _, _, _, _, _, ITEMS, UNION_BYTES, SHARED_BYTES,
            >(
                stream,
                lookback,
                counters,
                &*bins,
                &**first_keys,
                &**first_values,
                &mut **second_keys,
                &mut **second_values,
                pass * 256,
                pass,
                current_bit,
                num_bits,
            )?;
        } else {
            module.launch_radix_onesweep_pairs::<
                _, _, _, _, _, _, _, ITEMS, UNION_BYTES, SHARED_BYTES,
            >(
                stream,
                lookback,
                counters,
                &*bins,
                &**second_keys,
                &**second_values,
                &mut **first_keys,
                &mut **first_values,
                pass * 256,
                pass,
                current_bit,
                num_bits,
            )?;
        }
        buffers.swap();
    }
    Ok(())
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_radix_sort_pairs<
    K,
    V,
    M,
    S,
    U,
    B,
    L,
    C,
    Keys,
    Values,
    const SINGLE_ITEMS: usize,
    const ITEMS: usize,
    const UNION_BYTES: usize,
    const SHARED_BYTES: usize,
    const PASSES: usize,
    const DESC: bool,
>(
    module: &M,
    stream: &CudaStream,
    keys: &mut Keys,
    values: &mut Values,
    workspace: &mut DeviceRadixSortPairsWorkspace<K, V, S, U, B, L, C>,
) -> Result<(), DeviceAlgorithmError>
where
    K: RadixKey,
    V: RadixValue,
    M: RadixPairsKernel<K, V, DESC>,
    Keys: KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRangeMut<K>,
    Values: KernelSliceArgMut<Elem = V> + rccl_memory::ContiguousDeviceRangeMut<V>,
    S: WorkspaceBuffer<K>,
    U: WorkspaceBuffer<V>,
    S::Range: KernelSliceArgMut<Elem = K> + rccl_memory::ContiguousDeviceRange<K>,
    U::Range: KernelSliceArgMut<Elem = V> + rccl_memory::ContiguousDeviceRange<V>,
    B: WorkspaceBuffer<u64>,
    L: WorkspaceBuffer<i32>,
    C: WorkspaceBuffer<i32>,
    B::Range: KernelSliceArgMut<Elem = u64>,
    L::Range: KernelSliceArgMut<Elem = i32> + crate::__private::ZeroRadixLookback,
    C::Range: KernelSliceArgMut<Elem = i32>,
{
    let len = KernelSliceArg::len(keys);
    let values_len = KernelSliceArg::len(values);
    if values_len != len {
        return Err(DeviceAlgorithmError::LengthMismatch {
            role: "radix values",
            expected: len,
            actual: values_len,
        });
    }
    let (scratch_keys, scratch_values, controls) = workspace.__parts_mut();
    for (role, actual) in [
        ("radix scratch keys", scratch_keys.len()),
        ("radix scratch values", scratch_values.len()),
    ] {
        if actual != len {
            return Err(DeviceAlgorithmError::LengthMismatch {
                role,
                expected: len,
                actual,
            });
        }
    }
    let mut buffers = DoubleBuffer::__from_equal_parts(
        (&mut *keys, &mut *values),
        (&mut *scratch_keys, &mut *scratch_values),
        len,
    );
    device_radix_sort_pairs_buffer::<
        K,
        V,
        M,
        _,
        _,
        _,
        _,
        B,
        L,
        C,
        SINGLE_ITEMS,
        ITEMS,
        UNION_BYTES,
        SHARED_BYTES,
        PASSES,
        DESC,
    >(module, stream, &mut buffers, 0..K::BITS, controls)?;
    let current_is_first = buffers.current_is_first();
    if !current_is_first {
        crate::__private::copy_device_range::<K, _, _>(&*scratch_keys, &mut *keys, stream)
            .map_err(DeviceAlgorithmError::Memory)?;
        crate::__private::copy_device_range::<V, _, _>(&*scratch_values, &mut *values, stream)
            .map_err(DeviceAlgorithmError::Memory)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    type Alias = u32;

    #[test]
    fn aliases_use_rust_trait_semantics_without_macro_type_parsing() {
        assert_eq!(Alias::BITS, 32);
        assert_eq!(Alias::to_ordered_ascending(7), 7);
        assert_eq!(i32::to_ordered_ascending(-1), 0x7fff_ffff);
        assert_eq!(
            f32::from_ordered_ascending(f32::to_ordered_ascending(-3.5)),
            -3.5
        );
    }

    #[test]
    fn policy_constants_match_the_removed_proc_macro() {
        assert_eq!(radix_single_items(core::mem::size_of::<u32>()), 19);
        let items = radix_items(21, core::mem::size_of::<u32>());
        assert_eq!(items, 21);
        assert_eq!(radix_shared_bytes(items, 4), 34_304);

        let pair_items = radix_items(17, 8);
        assert_eq!(pair_items, 8);
        assert_eq!(radix_shared_bytes(pair_items, 8), 26_624);
    }
}
