#[macro_export]
macro_rules! thrust_program {
    ($context:expr $(,)?) => {{
        $crate::__private::rccl_cub::__private::__device_kernels!(__rccl_thrust_kernels);
        let __rccl_context = $context;
        unsafe { __rccl_thrust_kernels::load(__rccl_context) }.map(|__rccl_module| {
            $crate::cuda::Program::__from_module(__rccl_context.as_ref(), __rccl_module)
        })
    }};
}
