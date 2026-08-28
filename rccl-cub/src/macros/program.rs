/// Build the cuda-oxide program that owns RCCL's device-wide CUB kernels.
#[macro_export]
macro_rules! cub_program {
    ($context:expr $(,)?) => {{
        $crate::__private::__device_kernels!(__rccl_cub_kernels);
        let __rccl_context = $context;
        unsafe { __rccl_cub_kernels::load(__rccl_context) }.map(|__rccl_module| {
            $crate::cuda::Program::__from_module(__rccl_context.as_ref(), __rccl_module)
        })
    }};
}
