use cuda_core::{CudaStream, LaunchConfig1D, LaunchContractError};
use cuda_device::atomic::DeviceAtomicU32;
use cuda_host::{KernelSliceArg, KernelSliceArgMut};
use rccl_memory::{ContiguousDeviceRange, ContiguousDeviceRangeMut, DeviceAllocator, DeviceBuffer};

use crate::{BloomError, StableHashBytes};

const BLOCK_THREADS: u32 = 256;

fn launch_config(items: usize) -> Result<LaunchConfig1D, BloomError> {
    let items = u32::try_from(items).map_err(|_| BloomError::CapacityOverflow)?;
    Ok(LaunchConfig1D::new(
        items.div_ceil(BLOCK_THREADS),
        BLOCK_THREADS,
        0,
    ))
}

#[doc(hidden)]
pub trait BloomBackend<K: StableHashBytes>: Sized {
    fn bloom_clear<A: DeviceAllocator>(
        &self,
        stream: &CudaStream,
        words: &mut DeviceBuffer<'_, u32, A>,
    ) -> Result<(), BloomError>;

    fn bloom_add<A, R>(
        &self,
        stream: &CudaStream,
        words: &mut DeviceBuffer<'_, u32, A>,
        keys: &R,
        num_blocks: usize,
        seed: u64,
    ) -> Result<(), BloomError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<K> + ?Sized;

    fn bloom_contains<A, R, O>(
        &self,
        stream: &CudaStream,
        words: &DeviceBuffer<'_, u32, A>,
        keys: &R,
        output: &mut O,
        num_blocks: usize,
        seed: u64,
    ) -> Result<(), BloomError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<K> + ?Sized,
        O: ContiguousDeviceRangeMut<u8> + ?Sized;

    #[doc(hidden)]
    fn bloom_add_fault<A, R>(
        &self,
        stream: &CudaStream,
        words: &mut DeviceBuffer<'_, u32, A>,
        keys: &R,
        num_blocks: usize,
        seed: u64,
        omitted_word: u32,
    ) -> Result<(), BloomError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<K> + ?Sized;
}

#[doc(hidden)]
pub trait BloomKernelModule {
    fn launch_bloom_clear<S>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        words: &mut S,
    ) -> Result<(), LaunchContractError>
    where
        S: KernelSliceArgMut<Elem = u32>;

    fn launch_bloom_add<K, S, R>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        words: &S,
        keys: &R,
        num_blocks: usize,
        seed: u64,
    ) -> Result<(), LaunchContractError>
    where
        K: StableHashBytes,
        S: KernelSliceArg<Elem = DeviceAtomicU32>,
        R: KernelSliceArg<Elem = K>;

    fn launch_bloom_contains<K, S, R, O>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        words: &S,
        keys: &R,
        output: &mut O,
        num_blocks: usize,
        seed: u64,
    ) -> Result<(), LaunchContractError>
    where
        K: StableHashBytes,
        S: KernelSliceArg<Elem = u32>,
        R: KernelSliceArg<Elem = K>,
        O: KernelSliceArgMut<Elem = u8>;

    fn launch_bloom_add_fault<K, S, R>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        words: &S,
        keys: &R,
        num_blocks: usize,
        seed: u64,
        omitted_word: u32,
    ) -> Result<(), LaunchContractError>
    where
        K: StableHashBytes,
        S: KernelSliceArg<Elem = DeviceAtomicU32>,
        R: KernelSliceArg<Elem = K>;
}

impl<M, K> BloomBackend<K> for M
where
    M: BloomKernelModule,
    K: StableHashBytes,
{
    fn bloom_clear<A: DeviceAllocator>(
        &self,
        stream: &CudaStream,
        words: &mut DeviceBuffer<'_, u32, A>,
    ) -> Result<(), BloomError> {
        words.ensure_usable_on(stream)?;
        let config = launch_config(words.len())?;
        let len = words.len();
        let mut words = crate::kernel_args::RangeMutArg::subrange(words, 0, len);
        self.launch_bloom_clear(stream, config, &mut words)
            .map_err(BloomError::Contract)
    }

    fn bloom_add<A, R>(
        &self,
        stream: &CudaStream,
        words: &mut DeviceBuffer<'_, u32, A>,
        keys: &R,
        num_blocks: usize,
        seed: u64,
    ) -> Result<(), BloomError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<K> + ?Sized,
    {
        let config = launch_config(keys.len())?;
        let words = crate::kernel_args::AtomicU32RangeMutArg::new(words);
        let keys = crate::kernel_args::RangeArg::new(keys);
        self.launch_bloom_add::<K, _, _>(stream, config, &words, &keys, num_blocks, seed)
            .map_err(BloomError::Contract)
    }

    fn bloom_contains<A, R, O>(
        &self,
        stream: &CudaStream,
        words: &DeviceBuffer<'_, u32, A>,
        keys: &R,
        output: &mut O,
        num_blocks: usize,
        seed: u64,
    ) -> Result<(), BloomError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<K> + ?Sized,
        O: ContiguousDeviceRangeMut<u8> + ?Sized,
    {
        let config = launch_config(keys.len())?;
        let words = crate::kernel_args::RangeArg::new(words);
        let keys = crate::kernel_args::RangeArg::new(keys);
        let output_len = keys.len();
        let mut output = crate::kernel_args::RangeMutArg::subrange(output, 0, output_len);
        self.launch_bloom_contains::<K, _, _, _>(
            stream,
            config,
            &words,
            &keys,
            &mut output,
            num_blocks,
            seed,
        )
        .map_err(BloomError::Contract)
    }

    fn bloom_add_fault<A, R>(
        &self,
        stream: &CudaStream,
        words: &mut DeviceBuffer<'_, u32, A>,
        keys: &R,
        num_blocks: usize,
        seed: u64,
        omitted_word: u32,
    ) -> Result<(), BloomError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<K> + ?Sized,
    {
        let config = launch_config(keys.len())?;
        let words = crate::kernel_args::AtomicU32RangeMutArg::new(words);
        let keys = crate::kernel_args::RangeArg::new(keys);
        self.launch_bloom_add_fault::<K, _, _>(
            stream,
            config,
            &words,
            &keys,
            num_blocks,
            seed,
            omitted_word,
        )
        .map_err(BloomError::Contract)
    }
}

/// Bind the generated cuda-oxide Bloom entry points to the ordinary backend.
#[macro_export]
#[doc(hidden)]
macro_rules! __cuco_bloom_backend {
    ($module:ident) => {
        impl $crate::__private::BloomKernelModule for $module::LoadedModule {
            fn launch_bloom_clear<S>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                words: &mut S,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                S: $crate::__private::cuda_host::KernelSliceArgMut<Elem = u32>,
            {
                let prepared = self.prepare_bloom_clear(config)?;
                self.bloom_clear(stream, &prepared, words)
            }

            fn launch_bloom_add<K, S, R>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                words: &S,
                keys: &R,
                num_blocks: usize,
                seed: u64,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::StableHashBytes,
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
                R: $crate::__private::cuda_host::KernelSliceArg<Elem = K>,
            {
                let prepared = self.prepare_bloom_add::<K>(config)?;
                self.bloom_add::<K>(stream, &prepared, words, keys, num_blocks, seed)
            }

            fn launch_bloom_contains<K, S, R, O>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                words: &S,
                keys: &R,
                output: &mut O,
                num_blocks: usize,
                seed: u64,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::StableHashBytes,
                S: $crate::__private::cuda_host::KernelSliceArg<Elem = u32>,
                R: $crate::__private::cuda_host::KernelSliceArg<Elem = K>,
                O: $crate::__private::cuda_host::KernelSliceArgMut<Elem = u8>,
            {
                let prepared = self.prepare_bloom_contains::<K>(config)?;
                self.bloom_contains::<K>(stream, &prepared, words, keys, output, num_blocks, seed)
            }

            fn launch_bloom_add_fault<K, S, R>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                words: &S,
                keys: &R,
                num_blocks: usize,
                seed: u64,
                omitted_word: u32,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                K: $crate::StableHashBytes,
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
                R: $crate::__private::cuda_host::KernelSliceArg<Elem = K>,
            {
                let prepared = self.prepare_bloom_add_fault::<K>(config)?;
                self.bloom_add_fault::<K>(
                    stream,
                    &prepared,
                    words,
                    keys,
                    num_blocks,
                    seed,
                    omitted_word,
                )
            }
        }
    };
}
