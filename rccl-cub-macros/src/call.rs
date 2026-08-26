//! `syn` ASTs for shallow CUB operation macros.
//!
//! The operand shapes come from the pinned CCCL examples and API tests. They
//! retain ordinary Rust expressions; the proc macro does not attempt to see
//! rustc-inferred types.
#![allow(dead_code)]

use syn::{
    Expr, Result, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

macro_rules! operation_call {
    ($name:ident, $operation:literal, $($field:ident),+ $(,)?) => {
        #[derive(Debug)]
        pub(crate) struct $name {
            $(pub(crate) $field: Expr,)+
        }

        impl Parse for $name {
            fn parse(input: ParseStream<'_>) -> Result<Self> {
                let expressions = Punctuated::<Expr, Token![,]>::parse_terminated(input)?;
                let expected = [$(stringify!($field)),+].len();
                if expressions.len() != expected {
                    return Err(input.error(format!(
                        "{} expects {} operands ({}) but received {}",
                        $operation,
                        expected,
                        [$(stringify!($field)),+].join(", "),
                        expressions.len(),
                    )));
                }
                let mut expressions = expressions.into_iter();
                Ok(Self {
                    $($field: expressions.next().expect("validated operand count"),)+
                })
            }
        }
    };
}

// Pinned CCCL: DeviceReduce::Reduce(input, output, count, op, init, env).
// RCCL infers count from the input range and puts the execution environment
// first, followed by the remaining CCCL operands in their upstream order.
operation_call!(
    DeviceReduceCall,
    "device_reduce!",
    execution,
    input,
    output,
    op,
    init,
);

operation_call!(
    DeviceReduceSumCall,
    "device_reduce_sum!",
    execution,
    input,
    output,
);
operation_call!(
    DeviceReduceMinCall,
    "device_reduce_min!",
    execution,
    input,
    output,
);
operation_call!(
    DeviceReduceMaxCall,
    "device_reduce_max!",
    execution,
    input,
    output,
);

// Pinned CCCL env API uses separate value and signed 64-bit index outputs.
operation_call!(
    DeviceReduceArgMinCall,
    "device_reduce_arg_min!",
    execution,
    input,
    value_output,
    index_output,
);
operation_call!(
    DeviceReduceArgMaxCall,
    "device_reduce_arg_max!",
    execution,
    input,
    value_output,
    index_output,
);
operation_call!(
    DeviceReduceArgMinMaxCall,
    "device_reduce_arg_min_max!",
    execution,
    input,
    min_output,
    min_index_output,
    max_output,
    max_index_output,
);

operation_call!(
    DeviceTransformReduceCall,
    "device_transform_reduce!",
    execution,
    input,
    output,
    op,
    transform,
    init,
);

operation_call!(
    DeviceReduceByKeyCall,
    "device_reduce_by_key!",
    execution,
    keys_input,
    unique_keys_output,
    values_input,
    aggregates_output,
    num_runs_output,
    op,
);

#[cfg(test)]
mod tests {
    use super::*;
    use quote::ToTokens;

    fn tokens(expression: &Expr) -> String {
        expression.to_token_stream().to_string()
    }

    #[test]
    fn reduce_matches_the_pinned_reduce_env_operand_shape() {
        let call: DeviceReduceCall = syn::parse_quote! {
            &exec, &values, &mut result, |left, right| left + right, 0_i64
        };
        assert_eq!(tokens(&call.execution), "& exec");
        assert_eq!(tokens(&call.input), "& values");
        assert!(matches!(call.output, Expr::Reference(_)));
        assert!(matches!(call.op, Expr::Closure(_)));
        assert_eq!(tokens(&call.init), "0_i64");
    }

    #[test]
    fn sum_min_and_max_are_distinct_three_operand_operations() {
        let sum: DeviceReduceSumCall = syn::parse_quote! { &exec, &values, &mut sum_output };
        let min: DeviceReduceMinCall = syn::parse_quote! { &exec, &values, &mut min_output };
        let max: DeviceReduceMaxCall = syn::parse_quote! { &exec, &values, &mut max_output };
        assert_eq!(tokens(&sum.output), "& mut sum_output");
        assert_eq!(tokens(&min.output), "& mut min_output");
        assert_eq!(tokens(&max.output), "& mut max_output");
    }

    #[test]
    fn arg_operations_preserve_separate_value_and_index_outputs() {
        let arg_min: DeviceReduceArgMinCall =
            syn::parse_quote! { &exec, &values, &mut min_value, &mut min_index };
        let arg_max: DeviceReduceArgMaxCall =
            syn::parse_quote! { &exec, &values, &mut max_value, &mut max_index };
        let both: DeviceReduceArgMinMaxCall = syn::parse_quote! {
            &exec, &values,
            &mut min_value, &mut min_index,
            &mut max_value, &mut max_index
        };
        assert_eq!(tokens(&arg_min.index_output), "& mut min_index");
        assert_eq!(tokens(&arg_max.index_output), "& mut max_index");
        assert_eq!(tokens(&both.min_index_output), "& mut min_index");
        assert_eq!(tokens(&both.max_index_output), "& mut max_index");
    }

    #[test]
    fn transform_reduce_keeps_both_device_closures_as_rust_expressions() {
        let call: DeviceTransformReduceCall = syn::parse_quote! {
            &exec,
            &values,
            &mut output,
            |left, right| left + right,
            |value| -value,
            0_i32
        };
        assert!(matches!(call.op, Expr::Closure(_)));
        assert!(matches!(call.transform, Expr::Closure(_)));
    }

    #[test]
    fn reduce_by_key_keeps_all_pinned_output_roles() {
        let call: DeviceReduceByKeyCall = syn::parse_quote! {
            &exec,
            &keys,
            &mut unique_keys,
            &values,
            &mut aggregates,
            &mut num_runs,
            |left, right| left.min(right)
        };
        assert_eq!(tokens(&call.unique_keys_output), "& mut unique_keys");
        assert_eq!(tokens(&call.aggregates_output), "& mut aggregates");
        assert_eq!(tokens(&call.num_runs_output), "& mut num_runs");
        assert!(matches!(call.op, Expr::Closure(_)));
    }

    #[test]
    fn wrong_arity_and_the_old_named_temp_syntax_are_rejected() {
        let missing = syn::parse_str::<DeviceReduceSumCall>("&exec, &values")
            .expect_err("a missing output must not be masked");
        assert!(missing.to_string().contains("expects 3 operands"));

        let extra = syn::parse_str::<DeviceReduceArgMinCall>(
            "&exec, &values, &mut value, &mut index, &mut accidental_extra",
        )
        .expect_err("an extra output must not be ignored");
        assert!(extra.to_string().contains("expects 4 operands"));

        let obsolete = syn::parse_str::<DeviceReduceCall>(
            "on: &exec, input: &values, output: &mut out, init: 0, op: |a,b| a+b, temp: auto",
        )
        .expect_err("the invented named-field/temp:auto API must stay deleted");
        assert!(!obsolete.to_string().is_empty());
    }
}
