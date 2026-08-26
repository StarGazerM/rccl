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
    let _ = rccl::merge_by(exec, first, second, output, |left: &u32, right: &u32| {
        right.cmp(left)
    });
}

#[test]
fn ordinary_merge_api_uses_thrust_execution() {
    type Buffer<'stream> =
        rccl::memory::DeviceBuffer<'stream, u32, rccl::memory::CudaAsyncMemoryResource>;
    fn instantiate<'stream>(_: core::marker::PhantomData<&'stream ()>) {
        let _ = merge_surface_compiles::<
            rccl::memory::CudaAsyncMemoryResource,
            Buffer<'stream>,
            Buffer<'stream>,
            Buffer<'stream>,
        >;
    }

    assert!(core::mem::size_of::<rccl::cuda::Execution<'static>>() > 0);
    let _ = instantiate;
}
