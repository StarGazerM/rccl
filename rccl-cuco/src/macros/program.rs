/// Build the cuda-oxide program containing the selected cuCollections kernels.
#[macro_export]
macro_rules! cuco_program {
    ($context:expr $(,)?) => {{
        $crate::__cuco_device_kernels!(__rccl_cuco_kernels);
        $crate::__cuco_backend!(__rccl_cuco_kernels);
        $crate::__cuco_bloom_backend!(__rccl_cuco_kernels);
        $crate::__cuco_exact_set_backend!(__rccl_cuco_kernels);
        let __rccl_context = $context;
        unsafe { __rccl_cuco_kernels::load(__rccl_context) }.map(|__rccl_module| {
            $crate::Program::__from_module(__rccl_context.as_ref(), __rccl_module)
        })
    }};
}
