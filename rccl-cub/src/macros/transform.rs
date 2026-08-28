#[doc(hidden)]
#[macro_export]
macro_rules! __device_transform_repeat_in_place {
    ($module:expr, $stream:expr, $output:expr, $repeated:expr, $op:expr $(,)?) => {{
        let __rccl_module = $module;
        let __rccl_stream = $stream;
        let __rccl_output = $output;
        let __rccl_repeated = $repeated;
        let __rccl_op = $op;
        let __rccl_output_len = __rccl_output.len();
        if __rccl_output_len == 0 {
            ::core::result::Result::Ok(())
        } else {
            let __rccl_launch = $crate::__private::cuda_core::LaunchConfig1D::new(
                __rccl_output_len.div_ceil(8).div_ceil(256) as u32,
                256,
                0,
            );
            match __rccl_module.prepare_transform_repeat_in_place_for(
                &__rccl_repeated,
                &__rccl_op,
                __rccl_launch,
            ) {
                ::core::result::Result::Ok(__rccl_prepared) => __rccl_module
                    .transform_repeat_in_place(
                        __rccl_stream,
                        &__rccl_prepared,
                        __rccl_repeated,
                        __rccl_op,
                        __rccl_output,
                    ),
                ::core::result::Result::Err(__rccl_error) => {
                    ::core::result::Result::Err(__rccl_error)
                }
            }
        }
    }};
}
