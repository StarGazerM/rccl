fn one_execution_compiles_for_sort_and_merge<'stream, A, B1, B2, O>(
    exec: &rccl::cuda::Execution<'stream, A>,
    sortable: &mut B1,
    first: &B1,
    second: &B2,
    output: &mut O,
) where
    A: rccl::memory::DeviceAllocator + Clone,
    B1: rccl::memory::DeviceSliceMut<u32> + ?Sized,
    B2: rccl::memory::DeviceSlice<u32> + ?Sized,
    O: rccl::memory::DeviceSliceMut<u32> + ?Sized,
{
    let _ = rccl::sort(exec, sortable);
    let _ = rccl::merge(exec, first, second, output);
}

#[test]
fn sort_and_merge_share_the_thrust_execution_type() {
    fn accepts<'stream, A>(_: &rccl::cuda::Execution<'stream, A>)
    where
        A: rccl::memory::DeviceAllocator + Clone,
    {
    }

    type Buffer<'stream> =
        rccl::memory::DeviceBuffer<'stream, u32, rccl::memory::CudaAsyncMemoryResource>;
    fn instantiate<'stream>(_: core::marker::PhantomData<&'stream ()>) {
        let _ = one_execution_compiles_for_sort_and_merge::<
            rccl::memory::CudaAsyncMemoryResource,
            Buffer<'stream>,
            Buffer<'stream>,
            Buffer<'stream>,
        >;
    }

    let _ = accepts::<rccl::memory::CudaAsyncMemoryResource>;
    let _ = instantiate;
}
