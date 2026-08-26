pub fn direct_thrust_surface_compiles<'stream, A, B1, B2, O>(
    exec: &host_algorithms::cuda::Execution<'stream, A>,
    sortable: &mut B1,
    first: &B1,
    second: &B2,
    output: &mut O,
) where
    A: host_algorithms::memory::DeviceAllocator + Clone,
    B1: host_algorithms::memory::DeviceSliceMut<u32> + ?Sized,
    B2: host_algorithms::memory::DeviceSlice<u32> + ?Sized,
    O: host_algorithms::memory::DeviceSliceMut<u32> + ?Sized,
{
    let _ = host_algorithms::sort(exec, sortable);
    let _ = host_algorithms::merge(exec, first, second, output);
}

#[cfg(test)]
mod tests {
    use super::*;

    type Allocator = host_algorithms::memory::CudaAsyncMemoryResource;
    type Buffer<'stream> = host_algorithms::memory::DeviceBuffer<'stream, u32, Allocator>;

    #[test]
    fn direct_thrust_dependency_instantiates_sort_and_merge_facades() {
        fn instantiate<'stream>(_: core::marker::PhantomData<&'stream ()>) {
            let _ = direct_thrust_surface_compiles::<
                Allocator,
                Buffer<'stream>,
                Buffer<'stream>,
                Buffer<'stream>,
            >;
        }
        let _ = instantiate;
    }
}
