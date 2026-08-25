#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Record {
    pub major: u32,
    pub minor: i32,
}

unsafe impl gpu_primitives::DeviceCopy for Record {}

#[derive(Clone, Copy)]
pub struct RecordLess;

unsafe impl gpu_primitives::DeviceComparator<Record> for RecordLess {
    unsafe fn less(&self, left: &Record, right: &Record) -> bool {
        left.major < right.major || (left.major == right.major && left.minor < right.minor)
    }
}

pub fn renamed_sort_facade_compiles<'stream, A, B>(
    exec: &gpu_primitives::cuda::Execution<'stream, A>,
    records: &mut B,
) where
    A: gpu_primitives::memory::DeviceAllocator + Clone,
    B: gpu_primitives::memory::DeviceSliceMut<Record> + ?Sized,
{
    let _ = gpu_primitives::sort_by(exec, records, RecordLess);
    let _ = gpu_primitives::stable_sort_by(exec, records, RecordLess);
}

pub fn renamed_merge_facade_compiles<'stream, A, I1, I2, O>(
    exec: &gpu_primitives::cuda::Execution<'stream, A>,
    first: &I1,
    second: &I2,
    output: &mut O,
) where
    A: gpu_primitives::memory::DeviceAllocator + Clone,
    I1: gpu_primitives::memory::DeviceSlice<u32> + ?Sized,
    I2: gpu_primitives::memory::DeviceSlice<u32> + ?Sized,
    O: gpu_primitives::memory::DeviceSliceMut<u32> + ?Sized,
{
    let _ = gpu_primitives::merge(exec, first, second, output);
    let _ = gpu_primitives::merge_by(exec, first, second, output, gpu_primitives::Greater);
}

pub fn compile_time_api_check() {
    fn assert_device_copy<T: gpu_primitives::DeviceCopy>() {}
    fn assert_comparator<C: gpu_primitives::DeviceComparator<u32>>() {}

    assert_device_copy::<Record>();
    assert_comparator::<gpu_primitives::Less>();
    assert!(core::mem::size_of::<gpu_primitives::cuda::Execution<'static>>() > 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    type Allocator = gpu_primitives::memory::CudaAsyncMemoryResource;
    type RecordBuffer = gpu_primitives::memory::DeviceBuffer<Record, Allocator>;
    type U32Buffer = gpu_primitives::memory::DeviceBuffer<u32, Allocator>;

    #[test]
    fn renamed_dependency_instantiates_sort_and_merge_facades() {
        compile_time_api_check();
        let _ = renamed_sort_facade_compiles::<Allocator, RecordBuffer>;
        let _ = renamed_merge_facade_compiles::<Allocator, U32Buffer, U32Buffer, U32Buffer>;
    }
}
