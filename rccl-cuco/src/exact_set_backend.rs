use cuda_core::{CudaStream, LaunchConfig1D, LaunchContractError};
use cuda_device::atomic::{DeviceAtomicU32, DeviceAtomicU64};
use cuda_host::{KernelSliceArg, KernelSliceArgMut};
use rccl_memory::{ContiguousDeviceRange, ContiguousDeviceRangeMut, DeviceAllocator, DeviceBuffer};

use crate::{ExactSetError, Row};

const BLOCK_THREADS: u32 = 128;
const COOPERATIVE_GROUP: usize = 4;

fn row_launch_config(rows: usize) -> Result<LaunchConfig1D, ExactSetError> {
    let threads = rows
        .checked_mul(COOPERATIVE_GROUP)
        .ok_or(ExactSetError::CapacityOverflow)?;
    let threads = u32::try_from(threads).map_err(|_| ExactSetError::CapacityOverflow)?;
    Ok(LaunchConfig1D::new(
        threads.div_ceil(BLOCK_THREADS),
        BLOCK_THREADS,
        0,
    ))
}

fn clear_launch_config(slots: usize) -> Result<LaunchConfig1D, ExactSetError> {
    let slots = u32::try_from(slots).map_err(|_| ExactSetError::CapacityOverflow)?;
    Ok(LaunchConfig1D::new(
        slots.div_ceil(BLOCK_THREADS),
        BLOCK_THREADS,
        0,
    ))
}

fn row2_bits<const N: usize>(row: Row<N>) -> u64 {
    debug_assert_eq!(N, 2);
    let mut bits = 0u64;
    let mut column = 0usize;
    while column < N && column < 2 {
        bits |= (row.data[column] as u64) << (column * 32);
        column += 1;
    }
    bits
}

#[doc(hidden)]
pub trait ExactSetBackend<const N: usize>: Sized {
    fn exact_set_clear<A: DeviceAllocator>(
        &self,
        stream: &CudaStream,
        slots: &mut DeviceBuffer<'_, u64, A>,
        states: &mut DeviceBuffer<'_, u32, A>,
        empty_row: Row<N>,
    ) -> Result<(), ExactSetError>;

    #[allow(clippy::too_many_arguments)]
    fn exact_set_insert<A, R>(
        &self,
        stream: &CudaStream,
        slots: &mut DeviceBuffer<'_, u64, A>,
        states: &mut DeviceBuffer<'_, u32, A>,
        hashes: &mut DeviceBuffer<'_, u64, A>,
        words: &mut DeviceBuffer<'_, u32, A>,
        metadata: &mut DeviceBuffer<'_, u64, A>,
        rows: &R,
        capacity: usize,
        empty_row: Row<N>,
        seed: u64,
        forced_hash: Option<u64>,
    ) -> Result<(), ExactSetError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<Row<N>> + ?Sized;

    #[allow(clippy::too_many_arguments)]
    fn exact_set_insert_prevalidated<A, R>(
        &self,
        stream: &CudaStream,
        slots: &mut DeviceBuffer<'_, u64, A>,
        states: &mut DeviceBuffer<'_, u32, A>,
        hashes: &mut DeviceBuffer<'_, u64, A>,
        words: &mut DeviceBuffer<'_, u32, A>,
        rows: &R,
        capacity: usize,
        empty_row: Row<N>,
        seed: u64,
    ) -> Result<(), ExactSetError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<Row<N>> + ?Sized;

    #[allow(clippy::too_many_arguments)]
    fn exact_set_contains<A, R, O>(
        &self,
        stream: &CudaStream,
        slots: &DeviceBuffer<'_, u64, A>,
        states: &DeviceBuffer<'_, u32, A>,
        hashes: &DeviceBuffer<'_, u64, A>,
        words: &DeviceBuffer<'_, u32, A>,
        rows: &R,
        output: &mut O,
        capacity: usize,
        empty_row: Row<N>,
        seed: u64,
        forced_hash: Option<u64>,
    ) -> Result<(), ExactSetError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
        O: ContiguousDeviceRangeMut<u32> + ?Sized;
}

#[doc(hidden)]
pub trait ExactSetKernelModule {
    fn launch_exact_set_clear<S>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        states: &mut S,
    ) -> Result<(), LaunchContractError>
    where
        S: KernelSliceArgMut<Elem = u32>;
    fn launch_exact_set_clear_direct<S>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &mut S,
        empty_slot: u64,
    ) -> Result<(), LaunchContractError>
    where
        S: KernelSliceArgMut<Elem = u64>;

    #[allow(clippy::too_many_arguments)]
    fn launch_exact_set_insert<const N: usize, S, H, W, M, R>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        states: &S,
        hashes: &H,
        words: &W,
        metadata: &M,
        rows: &R,
        empty_row: Row<N>,
        seed: u64,
        force_hash: u32,
        forced_hash: u64,
    ) -> Result<(), LaunchContractError>
    where
        S: KernelSliceArg<Elem = DeviceAtomicU32>,
        H: KernelSliceArg<Elem = DeviceAtomicU64>,
        W: KernelSliceArg<Elem = DeviceAtomicU32>,
        M: KernelSliceArg<Elem = DeviceAtomicU64>,
        R: KernelSliceArg<Elem = Row<N>>;
    #[allow(clippy::too_many_arguments)]
    fn launch_exact_set_insert_direct<const N: usize, S, M, R>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        metadata: &M,
        rows: &R,
        empty_slot: u64,
        seed: u64,
        force_hash: u32,
        forced_hash: u64,
    ) -> Result<(), LaunchContractError>
    where
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        M: KernelSliceArg<Elem = DeviceAtomicU64>,
        R: KernelSliceArg<Elem = Row<N>>;

    #[allow(clippy::too_many_arguments)]
    fn launch_exact_set_insert_prevalidated<const N: usize, S, H, W, R>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        states: &S,
        hashes: &H,
        words: &W,
        rows: &R,
        seed: u64,
    ) -> Result<(), LaunchContractError>
    where
        S: KernelSliceArg<Elem = DeviceAtomicU32>,
        H: KernelSliceArg<Elem = DeviceAtomicU64>,
        W: KernelSliceArg<Elem = DeviceAtomicU32>,
        R: KernelSliceArg<Elem = Row<N>>;
    fn launch_exact_set_insert_prevalidated_direct<const N: usize, S, R>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        rows: &R,
        empty_slot: u64,
        seed: u64,
    ) -> Result<(), LaunchContractError>
    where
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        R: KernelSliceArg<Elem = Row<N>>;

    #[allow(clippy::too_many_arguments)]
    fn launch_exact_set_contains<const N: usize, S, H, W, R, O>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        states: &S,
        hashes: &H,
        words: &W,
        rows: &R,
        output: &O,
        seed: u64,
        force_hash: u32,
        forced_hash: u64,
    ) -> Result<(), LaunchContractError>
    where
        S: KernelSliceArg<Elem = DeviceAtomicU32>,
        H: KernelSliceArg<Elem = DeviceAtomicU64>,
        W: KernelSliceArg<Elem = DeviceAtomicU32>,
        R: KernelSliceArg<Elem = Row<N>>,
        O: KernelSliceArg<Elem = DeviceAtomicU32>;
    #[allow(clippy::too_many_arguments)]
    fn launch_exact_set_contains_direct<const N: usize, S, R, O>(
        &self,
        stream: &CudaStream,
        config: LaunchConfig1D,
        slots: &S,
        rows: &R,
        output: &O,
        empty_slot: u64,
        seed: u64,
        force_hash: u32,
        forced_hash: u64,
    ) -> Result<(), LaunchContractError>
    where
        S: KernelSliceArg<Elem = DeviceAtomicU64>,
        R: KernelSliceArg<Elem = Row<N>>,
        O: KernelSliceArg<Elem = DeviceAtomicU32>;
}

impl<M, const N: usize> ExactSetBackend<N> for M
where
    M: ExactSetKernelModule,
{
    fn exact_set_clear<A: DeviceAllocator>(
        &self,
        stream: &CudaStream,
        slots: &mut DeviceBuffer<'_, u64, A>,
        states: &mut DeviceBuffer<'_, u32, A>,
        empty_row: Row<N>,
    ) -> Result<(), ExactSetError> {
        let config = clear_launch_config(slots.len())?;
        if N == 2 {
            let len = slots.len();
            let mut slots = crate::kernel_args::RangeMutArg::subrange(slots, 0, len);
            self.launch_exact_set_clear_direct(stream, config, &mut slots, row2_bits(empty_row))?;
        } else {
            let len = states.len();
            let mut states = crate::kernel_args::RangeMutArg::subrange(states, 0, len);
            self.launch_exact_set_clear(stream, config, &mut states)?;
        }
        Ok(())
    }

    fn exact_set_insert<A, R>(
        &self,
        stream: &CudaStream,
        slots: &mut DeviceBuffer<'_, u64, A>,
        states: &mut DeviceBuffer<'_, u32, A>,
        hashes: &mut DeviceBuffer<'_, u64, A>,
        words: &mut DeviceBuffer<'_, u32, A>,
        metadata: &mut DeviceBuffer<'_, u64, A>,
        rows: &R,
        capacity: usize,
        empty_row: Row<N>,
        seed: u64,
        forced_hash: Option<u64>,
    ) -> Result<(), ExactSetError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
    {
        let config = row_launch_config(rows.len())?;
        let metadata_owner = crate::kernel_args::AtomicU64StorageArg::new(metadata);
        let metadata = metadata_owner.subrange(0, 4);
        let rows = crate::kernel_args::RangeArg::new(rows);
        if N == 2 {
            let slots_owner = crate::kernel_args::AtomicU64StorageArg::new(slots);
            let slots = slots_owner.subrange(0, capacity);
            self.launch_exact_set_insert_direct::<N, _, _, _>(
                stream,
                config,
                &slots,
                &metadata,
                &rows,
                row2_bits(empty_row),
                seed,
                forced_hash.is_some() as u32,
                forced_hash.unwrap_or(0),
            )?;
        } else {
            let states = crate::kernel_args::AtomicU32RangeMutArg::new(states);
            let hashes_owner = crate::kernel_args::AtomicU64StorageArg::new(hashes);
            let hashes = hashes_owner.subrange(0, capacity);
            let words = crate::kernel_args::AtomicU32RangeMutArg::new(words);
            self.launch_exact_set_insert::<N, _, _, _, _, _>(
                stream,
                config,
                &states,
                &hashes,
                &words,
                &metadata,
                &rows,
                empty_row,
                seed,
                forced_hash.is_some() as u32,
                forced_hash.unwrap_or(0),
            )?;
        }
        Ok(())
    }

    fn exact_set_insert_prevalidated<A, R>(
        &self,
        stream: &CudaStream,
        slots: &mut DeviceBuffer<'_, u64, A>,
        states: &mut DeviceBuffer<'_, u32, A>,
        hashes: &mut DeviceBuffer<'_, u64, A>,
        words: &mut DeviceBuffer<'_, u32, A>,
        rows: &R,
        capacity: usize,
        empty_row: Row<N>,
        seed: u64,
    ) -> Result<(), ExactSetError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
    {
        let config = row_launch_config(rows.len())?;
        let rows = crate::kernel_args::RangeArg::new(rows);
        if N == 2 {
            let slots_owner = crate::kernel_args::AtomicU64StorageArg::new(slots);
            let slots = slots_owner.subrange(0, capacity);
            self.launch_exact_set_insert_prevalidated_direct::<N, _, _>(
                stream,
                config,
                &slots,
                &rows,
                row2_bits(empty_row),
                seed,
            )?;
        } else {
            let states = crate::kernel_args::AtomicU32RangeMutArg::new(states);
            let hashes_owner = crate::kernel_args::AtomicU64StorageArg::new(hashes);
            let hashes = hashes_owner.subrange(0, capacity);
            let words = crate::kernel_args::AtomicU32RangeMutArg::new(words);
            self.launch_exact_set_insert_prevalidated::<N, _, _, _, _>(
                stream, config, &states, &hashes, &words, &rows, seed,
            )?;
        }
        Ok(())
    }

    fn exact_set_contains<A, R, O>(
        &self,
        stream: &CudaStream,
        slots: &DeviceBuffer<'_, u64, A>,
        states: &DeviceBuffer<'_, u32, A>,
        hashes: &DeviceBuffer<'_, u64, A>,
        words: &DeviceBuffer<'_, u32, A>,
        rows: &R,
        output: &mut O,
        capacity: usize,
        empty_row: Row<N>,
        seed: u64,
        forced_hash: Option<u64>,
    ) -> Result<(), ExactSetError>
    where
        A: DeviceAllocator,
        R: ContiguousDeviceRange<Row<N>> + ?Sized,
        O: ContiguousDeviceRangeMut<u32> + ?Sized,
    {
        let config = row_launch_config(rows.len())?;
        let rows = crate::kernel_args::RangeArg::new(rows);
        let output = crate::kernel_args::AtomicU32RangeMutArg::new(output);
        if N == 2 {
            let slots = crate::kernel_args::AtomicU64RangeArg::new(slots);
            self.launch_exact_set_contains_direct::<N, _, _, _>(
                stream,
                config,
                &slots,
                &rows,
                &output,
                row2_bits(empty_row),
                seed,
                forced_hash.is_some() as u32,
                forced_hash.unwrap_or(0),
            )?;
        } else {
            debug_assert_eq!(states.len(), capacity);
            debug_assert_eq!(hashes.len(), capacity);
            debug_assert_eq!(words.len(), capacity * N);
            let states = crate::kernel_args::AtomicU32RangeArg::new(states);
            let hashes = crate::kernel_args::AtomicU64RangeArg::new(hashes);
            let words = crate::kernel_args::AtomicU32RangeArg::new(words);
            self.launch_exact_set_contains::<N, _, _, _, _, _>(
                stream,
                config,
                &states,
                &hashes,
                &words,
                &rows,
                &output,
                seed,
                forced_hash.is_some() as u32,
                forced_hash.unwrap_or(0),
            )?;
        }
        Ok(())
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! __cuco_exact_set_backend {
    ($module:ident) => {
        impl $crate::__private::ExactSetKernelModule for $module::LoadedModule {
            fn launch_exact_set_clear<S>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                states: &mut S,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                S: $crate::__private::cuda_host::KernelSliceArgMut<Elem = u32>,
            {
                let prepared = self.prepare_exact_set_clear(config)?;
                self.exact_set_clear(stream, &prepared, states)
            }
            fn launch_exact_set_clear_direct<S>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &mut S,
                empty_slot: u64,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                S: $crate::__private::cuda_host::KernelSliceArgMut<Elem = u64>,
            {
                let prepared = self.prepare_exact_set_clear_direct(config)?;
                self.exact_set_clear_direct(stream, &prepared, slots, empty_slot)
            }
            fn launch_exact_set_insert<const N: usize, S, H, W, C, R>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                states: &S,
                hashes: &H,
                words: &W,
                metadata: &C,
                rows: &R,
                empty_row: $crate::Row<N>,
                seed: u64,
                force_hash: u32,
                forced_hash: u64,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
                H: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                W: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                R: $crate::__private::cuda_host::KernelSliceArg<Elem = $crate::Row<N>>,
            {
                let prepared = self.prepare_exact_set_insert::<N>(config)?;
                self.exact_set_insert::<N>(
                    stream,
                    &prepared,
                    states,
                    hashes,
                    words,
                    metadata,
                    rows,
                    empty_row,
                    seed,
                    force_hash,
                    forced_hash,
                )
            }
            fn launch_exact_set_insert_direct<const N: usize, S, C, R>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                metadata: &C,
                rows: &R,
                empty_slot: u64,
                seed: u64,
                force_hash: u32,
                forced_hash: u64,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                C: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                R: $crate::__private::cuda_host::KernelSliceArg<Elem = $crate::Row<N>>,
            {
                let prepared = self.prepare_exact_set_insert_direct::<N>(config)?;
                self.exact_set_insert_direct::<N>(
                    stream,
                    &prepared,
                    slots,
                    metadata,
                    rows,
                    empty_slot,
                    seed,
                    force_hash,
                    forced_hash,
                )
            }
            fn launch_exact_set_insert_prevalidated<const N: usize, S, H, W, R>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                states: &S,
                hashes: &H,
                words: &W,
                rows: &R,
                seed: u64,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
                H: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                W: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
                R: $crate::__private::cuda_host::KernelSliceArg<Elem = $crate::Row<N>>,
            {
                let prepared = self.prepare_exact_set_insert_prevalidated::<N>(config)?;
                self.exact_set_insert_prevalidated::<N>(
                    stream, &prepared, states, hashes, words, rows, seed,
                )
            }
            fn launch_exact_set_insert_prevalidated_direct<const N: usize, S, R>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                rows: &R,
                empty_slot: u64,
                seed: u64,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                R: $crate::__private::cuda_host::KernelSliceArg<Elem = $crate::Row<N>>,
            {
                let prepared = self.prepare_exact_set_insert_prevalidated_direct::<N>(config)?;
                self.exact_set_insert_prevalidated_direct::<N>(
                    stream, &prepared, slots, rows, empty_slot, seed,
                )
            }
            fn launch_exact_set_contains<const N: usize, S, H, W, R, O>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                states: &S,
                hashes: &H,
                words: &W,
                rows: &R,
                output: &O,
                seed: u64,
                force_hash: u32,
                forced_hash: u64,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
                H: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                W: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
                R: $crate::__private::cuda_host::KernelSliceArg<Elem = $crate::Row<N>>,
                O: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
            {
                let prepared = self.prepare_exact_set_contains::<N>(config)?;
                self.exact_set_contains::<N>(
                    stream,
                    &prepared,
                    states,
                    hashes,
                    words,
                    rows,
                    output,
                    seed,
                    force_hash,
                    forced_hash,
                )
            }
            fn launch_exact_set_contains_direct<const N: usize, S, R, O>(
                &self,
                stream: &$crate::__private::cuda_core::CudaStream,
                config: $crate::__private::cuda_core::LaunchConfig1D,
                slots: &S,
                rows: &R,
                output: &O,
                empty_slot: u64,
                seed: u64,
                force_hash: u32,
                forced_hash: u64,
            ) -> Result<(), $crate::__private::cuda_core::LaunchContractError>
            where
                S: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU64,
                    >,
                R: $crate::__private::cuda_host::KernelSliceArg<Elem = $crate::Row<N>>,
                O: $crate::__private::cuda_host::KernelSliceArg<
                        Elem = $crate::__private::cuda_device::atomic::DeviceAtomicU32,
                    >,
            {
                let prepared = self.prepare_exact_set_contains_direct::<N>(config)?;
                self.exact_set_contains_direct::<N>(
                    stream,
                    &prepared,
                    slots,
                    rows,
                    output,
                    empty_slot,
                    seed,
                    force_hash,
                    forced_hash,
                )
            }
        }
    };
}
