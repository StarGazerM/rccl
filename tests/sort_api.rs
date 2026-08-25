use rccl::{BFloat16, DeviceBool, DeviceComparator, DeviceCopy, F16};

#[derive(Clone, Copy)]
struct Reverse;

unsafe impl DeviceComparator<u32> for Reverse {
    unsafe fn less(&self, left: &u32, right: &u32) -> bool {
        left > right
    }
}

fn sort_surface_compiles<'stream, A, B>(exec: &rccl::cuda::Execution<'stream, A>, keys: &mut B)
where
    A: rccl::memory::DeviceAllocator + Clone,
    B: rccl::memory::DeviceSliceMut<u32> + ?Sized,
{
    let _ = rccl::sort(exec, keys);
    let _ = rccl::stable_sort(exec, keys);
    let _ = rccl::sort_by(exec, keys, Reverse);
    let _ = rccl::stable_sort_by(exec, keys, Reverse);
}

fn pair_sort_surface_compiles<'stream, A, KB, VB>(
    exec: &rccl::cuda::Execution<'stream, A>,
    keys: &mut KB,
    values: &mut VB,
) where
    A: rccl::memory::DeviceAllocator + Clone,
    KB: rccl::memory::DeviceSliceMut<u32> + ?Sized,
    VB: rccl::memory::DeviceSliceMut<u32> + ?Sized,
{
    let _ = rccl::sort_by_key(exec, keys, values);
    let _ = rccl::stable_sort_by_key(exec, keys, values);
    let _ = rccl::sort_by_key_with(exec, keys, values, Reverse);
    let _ = rccl::stable_sort_by_key_with(exec, keys, values, Reverse);
}

#[test]
fn ordinary_sort_api_uses_thrust_execution() {
    fn assert_device_copy<T: DeviceCopy>() {}
    assert_device_copy::<u32>();
    assert!(core::mem::size_of::<Option<rccl::cuda::Execution<'static>>>() > 0);
    let _ = sort_surface_compiles::<
        rccl::memory::CudaAsyncMemoryResource,
        rccl::memory::DeviceBuffer<u32, rccl::memory::CudaAsyncMemoryResource>,
    >;
    let _ = pair_sort_surface_compiles::<
        rccl::memory::CudaAsyncMemoryResource,
        rccl::memory::DeviceBuffer<u32, rccl::memory::CudaAsyncMemoryResource>,
        rccl::memory::DeviceBuffer<u32, rccl::memory::CudaAsyncMemoryResource>,
    >;
}

#[test]
fn public_half_and_bool_keys_have_the_documented_total_order() {
    assert!(F16(0xbc00) < F16(0x3c00));
    assert_eq!(F16(0x8000), F16(0x0000));
    assert!(BFloat16(0xbf80) < BFloat16(0x3f80));
    assert_eq!(BFloat16(0x8000), BFloat16(0x0000));

    assert_eq!(DeviceBool::new(false), DeviceBool::FALSE);
    assert_eq!(DeviceBool::from(true), DeviceBool::TRUE);
    assert!(!DeviceBool::FALSE.get());
    assert!(DeviceBool::TRUE.get());
    assert!(DeviceBool::FALSE < DeviceBool::TRUE);
}
