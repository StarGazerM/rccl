#[macro_export]
#[doc(hidden)]
macro_rules! __thrust_load_program {
    ($context:expr, $module:ident) => {{
        let __rccl_context = $context;
        unsafe { $module::load(__rccl_context) }.map(|__rccl_module| {
            $crate::cuda::Program::__from_module(__rccl_context.as_ref(), __rccl_module)
        })
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! __thrust_selected_backends {
    ($module:ident;) => {};
    ($module:ident; merge $(, $rest:ident)*) => {
        $crate::__thrust_merge_backends!($module);
        $crate::__thrust_selected_backends!($module; $($rest),*);
    };
    ($module:ident; set_operations $(, $rest:ident)*) => {
        $crate::__thrust_set_backends!($module);
        $crate::__thrust_selected_backends!($module; $($rest),*);
    };
}

#[macro_export]
macro_rules! thrust_program {
    ($context:expr; set_operations $(,)?) => {{
        $crate::__private::rccl_cub::__private::__device_kernels!(
            __rccl_thrust_kernels; set_operations
        );
        $crate::__thrust_set_backends!(__rccl_thrust_kernels);
        $crate::__thrust_load_program!($context, __rccl_thrust_kernels)
    }};
    ($context:expr; $($capability:ident),+ $(,)?) => {{
        $crate::__private::rccl_cub::__private::__device_kernels!(__rccl_thrust_kernels);
        $crate::__thrust_selected_backends!(__rccl_thrust_kernels; $($capability),+);
        $crate::__thrust_load_program!($context, __rccl_thrust_kernels)
    }};
    ($context:expr $(,)?) => {{
        $crate::__private::rccl_cub::__private::__device_kernels!(__rccl_thrust_kernels);
        $crate::__thrust_backends!(__rccl_thrust_kernels);
        $crate::__thrust_load_program!($context, __rccl_thrust_kernels)
    }};
}
