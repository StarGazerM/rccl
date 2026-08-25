#[derive(Clone, Copy)]
struct Reverse;

unsafe impl rccl::DeviceComparator<u32> for Reverse {
    unsafe fn less(&self, left: &u32, right: &u32) -> bool {
        left > right
    }
}

fn merge_surface_compiles<'stream, A, I1, I2, O>(
    exec: &rccl::cuda::Execution<'stream, A>,
    first: &I1,
    second: &I2,
    output: &mut O,
) where
    A: rccl::memory::DeviceAllocator + Clone,
    I1: rccl::memory::DeviceSlice<u32> + ?Sized,
    I2: rccl::memory::DeviceSlice<u32> + ?Sized,
    O: rccl::memory::DeviceSliceMut<u32> + ?Sized,
{
    let _ = rccl::merge(exec, first, second, output);
    let _ = rccl::merge_by(exec, first, second, output, Reverse);
}

#[test]
fn ordinary_merge_api_uses_thrust_execution() {
    fn assert_comparator<C: rccl::DeviceComparator<u32>>() {}
    assert_comparator::<rccl::Less>();
    assert!(core::mem::size_of::<rccl::cuda::Execution<'static>>() > 0);
    let _ = merge_surface_compiles::<
        rccl::memory::CudaAsyncMemoryResource,
        rccl::memory::DeviceBuffer<u32, rccl::memory::CudaAsyncMemoryResource>,
        rccl::memory::DeviceBuffer<u32, rccl::memory::CudaAsyncMemoryResource>,
        rccl::memory::DeviceBuffer<u32, rccl::memory::CudaAsyncMemoryResource>,
    >;
}
