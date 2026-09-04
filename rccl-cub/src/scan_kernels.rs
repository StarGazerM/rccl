//! Fixed kernels using the private 256-thread block scan.

use cuda_device::cuda_module;

pub const SCAN_THREADS: u32 = 256;

pub const LOOKBACK_SCAN_THREADS: u32 = 128;
pub const LOOKBACK_SCAN_ITEMS: usize = 15;
pub const LOOKBACK_SCAN_TILE_ITEMS: usize =
    LOOKBACK_SCAN_THREADS as usize * LOOKBACK_SCAN_ITEMS;
pub const LOOKBACK_SCAN_PADDING: usize = 32;
const LOOKBACK_SCAN_SHARED_BYTES: usize = 8_192;

macro_rules! for_each_lookback_item {
    ($item:ident, $body:block) => {{
        macro_rules! one {
            ($index:expr) => {{
                let $item = $index;
                $body
            }};
        }
        one!(0usize);
        one!(1usize);
        one!(2usize);
        one!(3usize);
        one!(4usize);
        one!(5usize);
        one!(6usize);
        one!(7usize);
        one!(8usize);
        one!(9usize);
        one!(10usize);
        one!(11usize);
        one!(12usize);
        one!(13usize);
        one!(14usize);
    }};
}

#[cuda_module]
#[doc(hidden)]
mod device {
    use cuda_device::{
        DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp,
    };

    const TILE_INVALID: u32 = 0;
    const TILE_PARTIAL: u32 = 1;
    const TILE_INCLUSIVE: u32 = 2;

    #[inline(always)]
    fn descriptor(status: u32, value: u32) -> u64 {
        ((status as u64) << 32) | value as u64
    }

    #[inline(always)]
    fn descriptor_status(value: u64) -> u32 {
        (value >> 32) as u32
    }

    #[inline(always)]
    fn descriptor_value(value: u64) -> u32 {
        value as u32
    }

    #[inline(always)]
    fn wait_for_state(
        state: &mut crate::__private::DeviceRegion<'_, u64>,
        predecessor: u32,
    ) -> (u32, u32) {
        let physical = predecessor
            .wrapping_add(super::LOOKBACK_SCAN_PADDING as u32) as usize;
        let mut value = state.read_volatile(physical);
        while warp::any_sync(u32::MAX, descriptor_status(value) == TILE_INVALID) {
            crate::__private::set_lookback_retry_delay();
            value = state.read_volatile(physical);
        }
        (descriptor_status(value), descriptor_value(value))
    }

    #[inline(always)]
    fn tail_segmented_sum(value: u32, inclusive_status: bool, lane: u32) -> u32 {
        let mut flags = warp::ballot_sync(u32::MAX, inclusive_status);
        flags &= warp::lanemask_ge();
        flags |= 1u32 << 31;
        let last_lane = flags.trailing_zeros();
        let mut output = value;
        macro_rules! step {
            ($offset:expr) => {{
                let other = warp::shuffle_down_sync(u32::MAX, output, $offset);
                if lane + $offset <= last_lane {
                    output = output.wrapping_add(other);
                }
            }};
        }
        step!(1u32);
        step!(2u32);
        step!(4u32);
        step!(8u32);
        step!(16u32);
        output
    }

    #[inline(always)]
    fn tile_lookback(
        state: &mut crate::__private::DeviceRegion<'_, u64>,
        tile_index: u32,
        block_aggregate: u32,
    ) -> u32 {
        let lane = warp::lane_id();
        if lane == 0 {
            state.write_volatile(
                super::LOOKBACK_SCAN_PADDING + tile_index as usize,
                descriptor(TILE_PARTIAL, block_aggregate),
            );
        }
        let mut predecessor = tile_index.wrapping_sub(lane).wrapping_sub(1);
        crate::__private::set_lookback_initial_delay();
        let (mut status, value) = wait_for_state(state, predecessor);
        let mut exclusive = tail_segmented_sum(value, status == TILE_INCLUSIVE, lane);
        while warp::all_sync(u32::MAX, status != TILE_INCLUSIVE) {
            predecessor = predecessor.wrapping_sub(32);
            let (next_status, next_value) = wait_for_state(state, predecessor);
            status = next_status;
            exclusive = tail_segmented_sum(next_value, status == TILE_INCLUSIVE, lane)
                .wrapping_add(exclusive);
        }
        if lane == 0 {
            state.write_volatile(
                super::LOOKBACK_SCAN_PADDING + tile_index as usize,
                descriptor(
                    TILE_INCLUSIVE,
                    exclusive.wrapping_add(block_aggregate),
                ),
            );
        }
        exclusive
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn device_scan_lookback_init_u32(
        tile_state: DisjointSlice<u64>,
        tile_count: usize,
    ) {
        let _ = launch_context;
        let mut tile_state = crate::__private::DeviceRegion::new(tile_state);
        let index = thread::blockIdx_x() as usize * super::LOOKBACK_SCAN_THREADS as usize
            + thread::threadIdx_x() as usize;
        if index < super::LOOKBACK_SCAN_PADDING {
            tile_state.write(index, descriptor(TILE_INCLUSIVE, 0));
        }
        if index < tile_count {
            tile_state.write(
                super::LOOKBACK_SCAN_PADDING + index,
                descriptor(TILE_INVALID, 0),
            );
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(128)]
    #[launch_contract(domain = 1, block = (128, 1, 1))]
    pub fn device_scan_lookback_u32(
        values: DisjointSlice<u32>,
        tile_state: DisjointSlice<u64>,
        len: usize,
    ) {
        let _ = launch_context;
        let shared = crate::__private::static_block_shared_memory!(
            __RCCL_SCAN_LOOKBACK_U32,
            super::LOOKBACK_SCAN_SHARED_BYTES,
            16usize
        );
        let thread_index = thread::threadIdx_x() as usize;
        let tile_index = thread::blockIdx_x();
        let tile_base = tile_index as usize * super::LOOKBACK_SCAN_TILE_ITEMS;
        let mut values = crate::__private::DeviceRegion::new(values);
        let mut tile_state = crate::__private::DeviceRegion::new(tile_state);
        let mut exchange = shared.region::<u32>(0, super::LOOKBACK_SCAN_TILE_ITEMS);
        let mut local = [0u32; super::LOOKBACK_SCAN_ITEMS];

        for_each_lookback_item!(item, {
            let index = tile_base
                + item * super::LOOKBACK_SCAN_THREADS as usize
                + thread_index;
            exchange.write(item * super::LOOKBACK_SCAN_THREADS as usize + thread_index, if index < len {
                values.read(index)
            } else {
                0
            });
        });
        thread::sync_threads();
        for_each_lookback_item!(item, {
            local[item] = exchange.read(thread_index * super::LOOKBACK_SCAN_ITEMS + item);
        });
        thread::sync_threads();

        let mut aggregate = 0u32;
        for_each_lookback_item!(item, {
            aggregate = aggregate.wrapping_add(local[item]);
        });
        let scanned = crate::__private::block_scan_sum_128(
            aggregate,
            super::LOOKBACK_SCAN_THREADS,
            &shared,
        );

        let mut tile_prefix = 0u32;
        if tile_index == 0 {
            if thread_index == 0 {
                tile_state.write_volatile(
                    super::LOOKBACK_SCAN_PADDING,
                    descriptor(TILE_INCLUSIVE, scanned.aggregate),
                );
            }
        } else {
            let prefix_offset = super::LOOKBACK_SCAN_TILE_ITEMS * core::mem::size_of::<u32>();
            let mut prefix_storage = shared.region::<u32>(prefix_offset, 1);
            if thread_index < 32 {
                let prefix = tile_lookback(&mut tile_state, tile_index, scanned.aggregate);
                if thread_index == 0 {
                    prefix_storage.write(0, prefix);
                }
            }
            thread::sync_threads();
            tile_prefix = prefix_storage.read(0);
        }

        // The exchange aliases the block-scan scratch. Ensure every warp has
        // consumed the aggregate before any thread begins the store transpose.
        thread::sync_threads();

        let mut running = tile_prefix.wrapping_add(scanned.exclusive);
        for_each_lookback_item!(item, {
            let value = local[item];
            local[item] = running;
            running = running.wrapping_add(value);
        });
        let _ = running;
        for_each_lookback_item!(item, {
            exchange.write(thread_index * super::LOOKBACK_SCAN_ITEMS + item, local[item]);
        });
        thread::sync_threads();
        for_each_lookback_item!(item, {
            let index = tile_base
                + item * super::LOOKBACK_SCAN_THREADS as usize
                + thread_index;
            if index < len {
                values.write(
                    index,
                    exchange.read(item * super::LOOKBACK_SCAN_THREADS as usize + thread_index),
                );
            }
        });
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared_range = (32, 64), dynamic_shared_alignment = 8)]
    pub fn device_scan_tiles<T, const EXCLUSIVE: bool>(
        input: &[T],
        output: DisjointSlice<T>,
        block_sums: DisjointSlice<T>,
        len: usize,
    ) where
        T: crate::__private::BlockScanValue,
    {
        let _ = launch_context;
        let shared = crate::__private::BlockSharedMemory::new(8 * core::mem::size_of::<T>());
        let thread_index = thread::threadIdx_x() as usize;
        let mut tile_index = thread::blockIdx_x() as usize;
        let tile_stride = thread::gridDim_x() as usize;
        let tile_count = len.div_ceil(super::SCAN_THREADS as usize);
        let mut output = crate::__private::DeviceRegion::new(output);
        let mut block_sums = crate::__private::DeviceRegion::new(block_sums);

        while tile_index < tile_count {
            let tile_base = tile_index * super::SCAN_THREADS as usize;
            let valid_items = (len - tile_base).min(super::SCAN_THREADS as usize) as u32;
            let index = tile_base + thread_index;
            let value = input.get(index).copied().unwrap_or_else(T::zero);
            let scanned = crate::__private::block_scan_sum_256(value, valid_items, &shared);
            if index < len {
                output.write(
                    index,
                    if EXCLUSIVE {
                        scanned.exclusive
                    } else {
                        scanned.inclusive
                    },
                );
            }
            if thread_index == 0 && tile_index < block_sums.len() {
                block_sums.write(tile_index, scanned.aggregate);
            }
            if tile_index + tile_stride < tile_count {
                // The block primitive follows CUB's temp-storage reuse
                // contract; synchronize only when this physical block loops
                // onto another logical tile.
                thread::sync_threads();
            }
            tile_index += tile_stride;
        }
    }

    #[kernel(launch_context = launch_context)]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
    pub fn device_scan_add_block_offsets<T>(
        values: DisjointSlice<T>,
        block_offsets: &[T],
        len: usize,
    ) where
        T: crate::__private::BlockScanValue,
    {
        let mut values = crate::__private::DeviceRegion::new(values);
        let mut index = thread::index_1d_u32(launch_context).get() as usize;
        let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;
        while index < len {
            let block = index / super::SCAN_THREADS as usize;
            let offset = block_offsets.get(block).copied().unwrap_or_else(|| {
                cuda_device::debug::trap();
            });
            let value = values.read(index);
            values.write(index, T::add(offset, value));
            index += stride;
        }
    }
}

#[doc(hidden)]
pub type ScanModule = device::LoadedModule;

#[doc(hidden)]
pub fn load(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> Result<ScanModule, cuda_host::EmbeddedModuleError> {
    // SAFETY: the loader is generated from the entries in this module.
    unsafe { device::load(context) }
}
