#[macro_export]
#[doc(hidden)]
macro_rules! __cub_load_program {
    ($context:expr, $module:ident) => {{
        let __rccl_context = $context;
        unsafe { $module::load(__rccl_context) }.map(|__rccl_module| {
            $crate::cuda::Program::__from_module(__rccl_context.as_ref(), __rccl_module)
        })
    }};
}

/// Build the cuda-oxide program that owns RCCL's selected device-wide CUB kernels.
#[macro_export]
macro_rules! cub_program {
    ($context:expr; histogram $(,)?) => {{
        $crate::__private::__device_kernels!(__rccl_cub_kernels; histogram);
        $crate::__cub_load_program!($context, __rccl_cub_kernels)
    }};
    ($context:expr $(,)?) => {{
        $crate::__private::__device_kernels!(__rccl_cub_kernels);
        $crate::__cub_load_program!($context, __rccl_cub_kernels)
    }};
}
