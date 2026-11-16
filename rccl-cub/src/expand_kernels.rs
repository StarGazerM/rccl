//! Library-owned actual-length range scheduling; no application metadata.
#[cuda_device::cuda_module]
mod device {
    use crate::{DeviceReadAt, DeviceWriteAt};
    use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared_range = (64, 64))]
    pub fn expand_bounds<R, B, F>(
        ranges: R,
        bounds: B,
        record: F,
        len: usize,
        entries: DisjointSlice<u64>,
        totals: DisjointSlice<u64>,
    ) where
        R: DeviceReadAt,
        B: Fn(R::Item) -> (u64, u64) + Copy,
        F: Fn(usize, R::Item) + Copy,
    {
        let tid = thread::threadIdx_x() as usize;
        let mut tile = thread::blockIdx_x() as usize;
        let tiles = len.div_ceil(256);
        let shared = crate::__private::BlockSharedMemory::new(64);
        let mut entries = crate::__private::DeviceRegion::new(entries);
        let mut totals = crate::__private::DeviceRegion::new(totals);
        while tile < tiles {
            let index = tile * 256 + tid;
            let (begin, end) = if index < len {
                let value = unsafe { ranges.read_at(index) };
                let (begin, end) = bounds(value);
                if begin > end {
                    cuda_device::debug::trap();
                }
                record(index, value);
                (begin, end)
            } else {
                (0, 0)
            };
            let length = end - begin;
            let prefix = crate::__private::block_scan_sum_256(
                length,
                (len - tile * 256).min(256) as u32,
                &shared,
            );
            // At the first overflowing prefix, the wrapped sum is less than
            // this lane's nonnegative addend. Every logical prefix is checked.
            if prefix.inclusive < length {
                cuda_device::debug::trap();
            }
            if index < len {
                entries.write(index * 3, begin);
                entries.write(index * 3 + 1, end);
                entries.write(index * 3 + 2, prefix.exclusive);
            }
            if tid == 0 {
                totals.write(tile + 1, prefix.aggregate);
            }
            thread::sync_threads();
            tile += thread::gridDim_x() as usize;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared_range = (64, 64))]
    pub fn expand_offsets(totals: DisjointSlice<u64>, tiles: usize, capacity: u64) {
        let tid = thread::threadIdx_x() as usize;
        let shared = crate::__private::BlockSharedMemory::new(64);
        let mut totals = crate::__private::DeviceRegion::new(totals);
        let mut base = 0;
        let mut carry = 0u64;
        while base < tiles {
            let index = base + tid;
            let value = if index < tiles {
                totals.read(index + 1)
            } else {
                0
            };
            let prefix = crate::__private::block_scan_sum_256(
                value,
                (tiles - base).min(256) as u32,
                &shared,
            );
            if prefix.inclusive < value || carry.checked_add(prefix.aggregate).is_none() {
                cuda_device::debug::trap();
            }
            if index < tiles {
                totals.write(index + 1, carry + prefix.exclusive);
            }
            carry += prefix.aggregate;
            thread::sync_threads();
            base += 256;
        }
        if carry > capacity {
            cuda_device::debug::trap();
        }
        if tid == 0 {
            totals.write(0, carry);
        }
    }

    #[inline(always)]
    fn locate(entries: &[u64], totals: &[u64], len: usize, position: u64) -> usize {
        let mut lo = 0;
        let mut hi = len.div_ceil(256);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if totals[mid + 1] <= position {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let tile = lo - 1;
        let local = position - totals[tile + 1];
        lo = tile * 256;
        hi = (lo + 256).min(len);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if entries[mid * 3 + 2] <= local {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo - 1
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn expand_pairs<P, R>(entries: &[u64], totals: &[u64], len: usize, probes: P, rows: R)
    where
        P: DeviceWriteAt<Item = u64>,
        R: DeviceWriteAt<Item = u64>,
    {
        let total = totals[0];
        let mut index = thread::index_1d().get() as u64;
        let stride = thread::gridDim_x() as u64 * 256;
        while index < total {
            let owner = locate(entries, totals, len, index);
            let base = totals[owner / 256 + 1] + entries[owner * 3 + 2];
            let row = entries[owner * 3] + (index - base);
            unsafe {
                probes.write_at(index as usize, owner as u64);
                rows.write_at(index as usize, row);
            }
            if total - index <= stride {
                break;
            }
            index += stride;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn expand_reduce<S, F>(entries: &[u64], totals: &[u64], len: usize, sources: S, sink: F)
    where
        S: DeviceReadAt,
        S::Item: DeviceReadAt<Item = u64>,
        F: Fn(usize, u64) + Copy,
    {
        let lane = thread::threadIdx_x() & 31;
        let total = totals[0];
        let mut task = thread::index_1d().get() as u64 >> 5;
        while task < total.div_ceil(1024) {
            let mut position = task * 1024;
            let stop = position.saturating_add(1024).min(total);
            let mut owner = if lane == 0 {
                locate(entries, totals, len, position) as u64
            } else {
                0
            };
            owner = cuda_device::warp::shuffle_u64(owner, 0);
            while position < stop {
                let i = owner as usize;
                let begin = entries[i * 3];
                let end = entries[i * 3 + 1];
                let base = totals[i / 256 + 1] + entries[i * 3 + 2];
                let finish = (base + (end - begin)).min(stop);
                if finish - position >= 32 {
                    let source = unsafe { sources.read_at(i) };
                    let first = begin + (position - base);
                    let last = begin + (finish - base);
                    let sum = if last - first > lane as u64 {
                        unsafe {
                            source.sum_strided((first + lane as u64) as usize, last as usize, 32)
                        }
                    } else {
                        0
                    };
                    let sum = crate::warp_reduce::<u64, 32>(sum, 32, |a, b| a.wrapping_add(b));
                    if lane == 0 {
                        sink(i, sum);
                    }
                    position = finish;
                    owner += 1;
                } else {
                    let atom = position.saturating_add(lane as u64);
                    if atom < stop {
                        let i = locate(entries, totals, len, atom);
                        let base = totals[i / 256 + 1] + entries[i * 3 + 2];
                        let row = entries[i * 3] + (atom - base);
                        let source = unsafe { sources.read_at(i) };
                        sink(i, unsafe { source.read_at(row as usize) });
                    }
                    position = position.saturating_add(32).min(stop);
                    if position < stop {
                        owner = if lane == 0 {
                            locate(entries, totals, len, position) as u64
                        } else {
                            0
                        };
                        owner = cuda_device::warp::shuffle_u64(owner, 0);
                    }
                }
            }
            task += thread::gridDim_x() as u64 * 8;
        }
    }
}
pub(crate) type ExpandModule = device::LoadedModule;
pub(crate) fn load(
    context: &std::sync::Arc<cuda_core::CudaContext>,
) -> Result<ExpandModule, cuda_host::EmbeddedModuleError> {
    unsafe { device::load(context) }
}
