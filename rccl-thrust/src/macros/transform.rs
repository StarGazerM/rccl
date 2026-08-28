#[macro_export]
macro_rules! transform {
    ($execution:expr, $output:expr, $repeated:expr, $op:expr $(,)?) => {{
        let __rccl_execution = $execution;
        let __rccl_output = $output;
        let __rccl_repeated = $repeated;
        let __rccl_op = $op;
        let __rccl_output_len = $crate::DeviceRange::len(&*__rccl_output);
        if $crate::DeviceRange::len(&__rccl_repeated) != __rccl_output_len {
            ::core::result::Result::Err($crate::TransformError::LengthMismatch {
                input: $crate::DeviceRange::len(&__rccl_repeated),
                output: __rccl_output_len,
            })
        } else if __rccl_output_len == 0 {
            ::core::result::Result::Ok(())
        } else {
            $crate::__private::rccl_cub::__device_transform_repeat_in_place!(
                __rccl_execution.module(),
                __rccl_execution.stream(),
                __rccl_output,
                *__rccl_repeated.value(),
                __rccl_op,
            )
            .map_err($crate::TransformError::Launch)
        }
    }};
}
