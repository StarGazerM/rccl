//! `syn` ASTs for shallow CUB operation macros.
//!
//! The operand shapes come from the pinned CCCL examples and API tests. They
//! retain ordinary Rust expressions; the proc macro does not attempt to see
//! rustc-inferred types.
#![allow(dead_code)]

use syn::{
    BinOp, Expr, ExprClosure, ExprLit, Lit, LitInt, Pat, Result, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

pub(crate) fn is_addition_closure(expression: &Expr) -> bool {
    let Expr::Closure(closure) = ungroup(expression) else {
        return false;
    };
    let Some((left, right)) = closure_parameters(closure) else {
        return false;
    };
    match ungroup(&closure.body) {
        Expr::Binary(binary) if matches!(binary.op, BinOp::Add(_)) => {
            is_identifier(&binary.left, left) && is_identifier(&binary.right, right)
        }
        Expr::MethodCall(call) if call.method == "wrapping_add" && call.args.len() == 1 => {
            is_identifier(&call.receiver, left)
                && call
                    .args
                    .first()
                    .is_some_and(|argument| is_identifier(argument, right))
        }
        _ => false,
    }
}

fn closure_parameters(closure: &ExprClosure) -> Option<(&syn::Ident, &syn::Ident)> {
    let mut inputs = closure.inputs.iter();
    let left = pattern_identifier(inputs.next()?)?;
    let right = pattern_identifier(inputs.next()?)?;
    inputs.next().is_none().then_some((left, right))
}

fn pattern_identifier(pattern: &Pat) -> Option<&syn::Ident> {
    match pattern {
        Pat::Ident(identifier) => Some(&identifier.ident),
        Pat::Type(typed) => pattern_identifier(&typed.pat),
        Pat::Paren(paren) => pattern_identifier(&paren.pat),
        _ => None,
    }
}

fn is_identifier(expression: &Expr, expected: &syn::Ident) -> bool {
    let Expr::Path(path) = ungroup(expression) else {
        return false;
    };
    path.qself.is_none() && path.path.is_ident(expected)
}

fn ungroup(expression: &Expr) -> &Expr {
    match expression {
        Expr::Group(group) => ungroup(&group.expr),
        Expr::Paren(paren) => ungroup(&paren.expr),
        other => other,
    }
}

#[derive(Debug, syn_derive::Parse)]
struct ExpressionList {
    #[parse(Punctuated::parse_terminated)]
    expressions: Punctuated<Expr, Token![,]>,
}

#[derive(Debug)]
pub(crate) struct WarpReduceCall {
    pub(crate) logical_warp_threads: LitInt,
    pub(crate) input: Expr,
    pub(crate) valid_items: Option<Expr>,
    pub(crate) op: Expr,
}

impl Parse for WarpReduceCall {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let ExpressionList { expressions } = input.parse()?;
        if !matches!(expressions.len(), 3 | 4) {
            return Err(input.error(
                "warp_reduce! expects (logical_warp_threads, value, op) or (logical_warp_threads, value, valid_items, op)",
            ));
        }
        let mut expressions = expressions.into_iter();
        let width_expression = expressions.next().expect("validated operand count");
        let logical_warp_threads = parse_logical_warp_threads(width_expression)?;
        let input = expressions.next().expect("validated operand count");
        let remaining = expressions.collect::<Vec<_>>();
        let (valid_items, op) = match remaining.as_slice() {
            [op] => (None, op.clone()),
            [valid_items, op] => (Some(valid_items.clone()), op.clone()),
            _ => unreachable!("validated operand count"),
        };
        Ok(Self {
            logical_warp_threads,
            input,
            valid_items,
            op,
        })
    }
}

#[derive(Debug)]
pub(crate) struct WarpReduceItemsCall {
    pub(crate) logical_warp_threads: LitInt,
    pub(crate) items: Expr,
    pub(crate) op: Expr,
}

impl Parse for WarpReduceItemsCall {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let ExpressionList { expressions } = input.parse()?;
        if expressions.len() != 3 {
            return Err(input.error("warp_reduce_items! expects (logical_warp_threads, items, op)"));
        }
        let mut expressions = expressions.into_iter();
        let width_expression = expressions.next().expect("validated operand count");
        let logical_warp_threads = parse_logical_warp_threads(width_expression)?;
        Ok(Self {
            logical_warp_threads,
            items: expressions.next().expect("validated operand count"),
            op: expressions.next().expect("validated operand count"),
        })
    }
}

fn parse_logical_warp_threads(expression: Expr) -> Result<LitInt> {
    match expression {
        Expr::Lit(ExprLit {
            lit: Lit::Int(width),
            ..
        }) => {
            validate_logical_warp_threads(&width)?;
            Ok(width)
        }
        Expr::Group(group) => parse_logical_warp_threads(*group.expr),
        Expr::Paren(paren) => parse_logical_warp_threads(*paren.expr),
        other => Err(syn::Error::new_spanned(
            other,
            "logical warp width must be an integer literal owned by the operation macro",
        )),
    }
}

fn validate_logical_warp_threads(width: &LitInt) -> Result<()> {
    let width_value = width.base10_parse::<u32>()?;
    if !matches!(width_value, 1 | 2 | 4 | 8 | 16 | 32) {
        return Err(syn::Error::new_spanned(
            width,
            "cuda-oxide's safe warp tiles currently support widths 1, 2, 4, 8, 16, and 32",
        ));
    }
    Ok(())
}

macro_rules! operation_call {
    ($name:ident, $operation:literal, $($field:ident),+ $(,)?) => {
        #[derive(Debug)]
        pub(crate) struct $name {
            $(pub(crate) $field: Expr,)+
        }

        impl Parse for $name {
            fn parse(input: ParseStream<'_>) -> Result<Self> {
                let ExpressionList { expressions } = input.parse()?;
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

// Internal CUB lowering target for the currently implemented Thrust
// intersection. Thrust owns RepeatN validation; CUB owns the concrete kernel,
// launch geometry, and typed cuda-oxide call.
operation_call!(
    DeviceTransformRepeatInPlaceCall,
    "__device_transform_repeat_in_place!",
    module,
    stream,
    output,
    repeated,
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
    fn warp_reduce_preserves_rust_values_closures_and_partial_extent() {
        let full: WarpReduceCall = syn::parse_quote! { 32, value, |left, right| left + right };
        assert_eq!(full.logical_warp_threads.base10_parse::<u32>().unwrap(), 32);
        assert!(full.valid_items.is_none());
        assert!(matches!(full.op, Expr::Closure(_)));

        let partial: WarpReduceCall =
            syn::parse_quote! { 16, value, valid_items, |left, right| left.max(right) };
        assert_eq!(tokens(partial.valid_items.as_ref().unwrap()), "valid_items");
        assert!(matches!(partial.op, Expr::Closure(_)));

        let items: WarpReduceItemsCall =
            syn::parse_quote! { 8, thread_items, |left, right| left + right };
        assert_eq!(tokens(&items.items), "thread_items");
    }

    #[test]
    fn warp_reduce_rejects_runtime_and_unsupported_tile_widths() {
        let runtime =
            syn::parse_str::<WarpReduceCall>("logical_width, value, |left, right| left + right")
                .expect_err("tile width belongs to the macro's compile-time input");
        assert!(runtime.to_string().contains("integer literal"));

        let non_power = syn::parse_str::<WarpReduceCall>("9, value, |left, right| left + right")
            .expect_err("the safe backend must not mask unsupported shared-memory tiles");
        assert!(non_power.to_string().contains("1, 2, 4, 8, 16, and 32"));
    }

    #[test]
    fn warp_reduce_accepts_a_literal_forwarded_by_a_rust_macro() {
        let width = proc_macro2::Group::new(proc_macro2::Delimiter::None, quote::quote!(16));
        let call = syn::parse2::<WarpReduceCall>(
            quote::quote!(#width, value, |left, right| left + right),
        )
        .expect("an invisible macro-expansion group around a literal is still compile-time syntax");
        assert_eq!(call.logical_warp_threads.base10_parse::<u32>().unwrap(), 16);

        let parenthesized: WarpReduceItemsCall =
            syn::parse_quote! { (8), items, |left, right| left + right };
        assert_eq!(
            parenthesized
                .logical_warp_threads
                .base10_parse::<u32>()
                .unwrap(),
            8,
        );
    }

    #[test]
    fn only_an_exact_addition_closure_selects_the_builtin_sum_path() {
        let add: Expr = syn::parse_quote!(|left, right| left + right);
        let typed: Expr = syn::parse_quote!(|left: i32, right: i32| (left + right));
        let wrapping: Expr = syn::parse_quote!(|left, right| left.wrapping_add(right));
        let reversed: Expr = syn::parse_quote!(|left, right| right + left);
        let adjusted: Expr = syn::parse_quote!(|left, right| left + right + 1);
        assert!(is_addition_closure(&add));
        assert!(is_addition_closure(&typed));
        assert!(is_addition_closure(&wrapping));
        assert!(!is_addition_closure(&reversed));
        assert!(!is_addition_closure(&adjusted));
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
