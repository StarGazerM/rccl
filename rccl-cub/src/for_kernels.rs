//! Framed-column analysis, prefix preparation and race-free word packing.
#[cuda_device::cuda_module]
mod device {
    use crate::{DeviceReadAt, DeviceWriteAt, ForFrame};
    use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared_range = (128, 128))]
    pub fn for_analyze<R: DeviceReadAt<Item = u64>>(input: R, len: usize, block_rows: u32, lanes: u32, entries: DisjointSlice<u64>) {
        let tid = thread::threadIdx_x() as usize;
        let shared = cuda_device::DynamicSharedArray::<u64>::get();
        let mut entries = crate::__private::DeviceRegion::new(entries);
        let mut block = thread::blockIdx_x() as usize;
        while block < len.div_ceil(block_rows as usize) {
            let start = block * block_rows as usize;
            let rows = (len - start).min(block_rows as usize);
            let mut min = u64::MAX;
            let mut max = 0u64;
            let mut row = tid;
            while row < rows {
                let value = unsafe { input.read_at(start + row) };
                min = min.min(value); max = max.max(value); row += 256;
            }
            min = crate::warp_reduce::<u64, 32>(min, 32, |a, b| a.min(b));
            max = crate::warp_reduce::<u64, 32>(max, 32, |a, b| a.max(b));
            if tid % 32 == 0 { unsafe { *shared.add(tid / 32) = min; *shared.add(8 + tid / 32) = max; } }
            thread::sync_threads();
            if tid == 0 {
                min = u64::MAX; max = 0;
                let mut warp = 0;
                while warp < 8 { unsafe { min = min.min(*shared.add(warp)); max = max.max(*shared.add(8 + warp)); } warp += 1; }
                let bits = 64 - (max - min).leading_zeros();
                entries.write(block * 3, min);
                entries.write(block * 3 + 1, bits as u64);
                entries.write(block * 3 + 2, crate::for_words(rows as u32, lanes, bits));
            }
            thread::sync_threads();
            block += thread::gridDim_x() as usize;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared_range = (64, 64))]
    pub fn for_offsets(entries: DisjointSlice<u64>, total: DisjointSlice<u64>, blocks: usize, capacity: u64) {
        let tid = thread::threadIdx_x() as usize;
        let shared = crate::__private::BlockSharedMemory::new(64);
        let mut entries = crate::__private::DeviceRegion::new(entries);
        let mut total = crate::__private::DeviceRegion::new(total);
        let mut carry = 0u64;
        let mut start = 0;
        while start < blocks {
            let index = start + tid;
            let words = if index < blocks { entries.read(index * 3 + 2) } else { 0 };
            let prefix = crate::__private::block_scan_sum_256(words, (blocks - start).min(256) as u32, &shared);
            if prefix.inclusive < words || carry.checked_add(prefix.aggregate).is_none() { cuda_device::debug::trap(); }
            if index < blocks { entries.write(index * 3 + 2, carry + prefix.exclusive); }
            carry += prefix.aggregate;
            thread::sync_threads();
            start += 256;
        }
        if carry > capacity { cuda_device::debug::trap(); }
        if tid == 0 { total.write(0, carry); }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn for_pack<R, P, F>(input: R, payload: P, frames: F, len: usize, block_rows: u32, lanes: u32, entries: &[u64])
    where R: DeviceReadAt<Item = u64>, P: DeviceWriteAt<Item = u32>, F: DeviceWriteAt<Item = ForFrame> {
        let tid = thread::threadIdx_x() as u32;
        let mut block = thread::blockIdx_x() as usize;
        while block < len.div_ceil(block_rows as usize) {
            let start = block * block_rows as usize;
            let rows = (len - start).min(block_rows as usize) as u32;
            let base = entries[block * 3];
            let bits = entries[block * 3 + 1] as u32;
            let offset = entries[block * 3 + 2];
            if tid == 0 { unsafe { frames.write_at(block, ForFrame { base, bits, offset, rows }); } }
            let mut word = tid;
            while (word as u64) < crate::for_words(rows, lanes, bits) {
                let value = unsafe { crate::pack_for_word(crate::DeviceOffset::new(input, start), rows, lanes, base, bits, word) };
                unsafe { payload.write_at((offset + word as u64) as usize, value); }
                word += 256;
            }
            block += thread::gridDim_x() as usize;
        }
    }
}
pub(crate) type ForModule = device::LoadedModule;
pub(crate) fn load(context: &std::sync::Arc<cuda_core::CudaContext>) -> Result<ForModule, cuda_host::EmbeddedModuleError> {
    unsafe { device::load(context) }
}
