use rccl::{BFloat16, DeviceBool, DeviceCopy, F16};

fn sort_surface_compiles<'stream, A, B>(exec: &rccl::cuda::Execution<'stream, A>, keys: &mut B)
where
    A: rccl::memory::DeviceAllocator + Clone,
    B: rccl::memory::DeviceSliceMut<u32> + ?Sized,
{
    let _ = rccl::sort(exec, keys);
    let _ = rccl::stable_sort(exec, keys);
    let _ = rccl::sort_by(exec, keys, |left: &u32, right: &u32| right.cmp(left));
    let _ = rccl::stable_sort_by(exec, keys, |left: &u32, right: &u32| right.cmp(left));
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
    let _ = rccl::sort_by_key_with(exec, keys, values, |left: &u32, right: &u32| {
        right.cmp(left)
    });
    let _ = rccl::stable_sort_by_key_with(exec, keys, values, |left: &u32, right: &u32| {
        right.cmp(left)
    });
}

#[test]
fn ordinary_sort_api_uses_thrust_execution() {
    fn assert_device_copy<T: DeviceCopy>() {}
    type Buffer<'stream> =
        rccl::memory::DeviceBuffer<'stream, u32, rccl::memory::CudaAsyncMemoryResource>;
    fn instantiate<'stream>(_: core::marker::PhantomData<&'stream ()>) {
        let _ = sort_surface_compiles::<rccl::memory::CudaAsyncMemoryResource, Buffer<'stream>>;
        let _ = pair_sort_surface_compiles::<
            rccl::memory::CudaAsyncMemoryResource,
            Buffer<'stream>,
            Buffer<'stream>,
        >;
    }

    assert_device_copy::<u32>();
    assert!(core::mem::size_of::<Option<rccl::cuda::Execution<'static>>>() > 0);
    let _ = instantiate;
}

#[test]
fn ordering_arguments_are_ordinary_copy_closures() {
    fn accepts_order<K, C>(order: C) -> C
    where
        C: Fn(&K, &K) -> core::cmp::Ordering + Copy + 'static,
    {
        order
    }

    let descending = accepts_order::<u32, _>(|left: &u32, right: &u32| right.cmp(left));
    assert!(descending(&2, &1).is_lt());

    let mask = 0xff_u32;
    let masked =
        accepts_order::<u32, _>(move |left: &u32, right: &u32| (left & mask).cmp(&(right & mask)));
    assert!(masked(&0x101, &0x002).is_lt());
    assert!(masked(&0x101, &0x201).is_eq());
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
