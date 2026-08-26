#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Record {
    pub major: u32,
    pub minor: i32,
}

unsafe impl gpu_primitives::DeviceCopy for Record {}

fn order_record(left: &Record, right: &Record) -> core::cmp::Ordering {
    (left.major, left.minor).cmp(&(right.major, right.minor))
}

pub fn renamed_sort_facade_compiles<'stream, A, B>(
    exec: &gpu_primitives::cuda::Execution<'stream, A>,
    records: &mut B,
) where
    A: gpu_primitives::memory::DeviceAllocator + Clone,
    B: gpu_primitives::memory::DeviceSliceMut<Record> + ?Sized,
{
    let _ = gpu_primitives::sort_by(exec, records, order_record);
    let _ = gpu_primitives::stable_sort_by(exec, records, order_record);
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
    let _ = gpu_primitives::merge_by(exec, first, second, output, |left: &u32, right: &u32| {
        right.cmp(left)
    });
}

pub fn compile_time_api_check() {
    fn assert_device_copy<T: gpu_primitives::DeviceCopy>() {}
    assert_device_copy::<Record>();
    assert!(core::mem::size_of::<gpu_primitives::cuda::Execution<'static>>() > 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    type Allocator = gpu_primitives::memory::CudaAsyncMemoryResource;
    type RecordBuffer<'stream> = gpu_primitives::memory::DeviceBuffer<'stream, Record, Allocator>;
    type U32Buffer<'stream> = gpu_primitives::memory::DeviceBuffer<'stream, u32, Allocator>;

    #[test]
    fn renamed_dependency_instantiates_sort_and_merge_facades() {
        compile_time_api_check();
        fn instantiate<'stream>(_: core::marker::PhantomData<&'stream ()>) {
            let _ = renamed_sort_facade_compiles::<Allocator, RecordBuffer<'stream>>;
            let _ = renamed_merge_facade_compiles::<
                Allocator,
                U32Buffer<'stream>,
                U32Buffer<'stream>,
                U32Buffer<'stream>,
            >;
        }
        let _ = instantiate;
    }
}
