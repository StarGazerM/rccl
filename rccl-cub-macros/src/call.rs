//! `syn` ASTs for shallow CUB operation macros.
//!
//! The operand shapes come from the pinned CCCL examples and API tests. They
//! retain ordinary Rust expressions. Types that affect TMP lowering are
//! explicit syntax because a proc macro runs before rustc type inference.
#![allow(dead_code)]

use syn::{
    BinOp, Expr, ExprClosure, LitInt, Pat, Result, Token, Type, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

#[derive(Debug)]
struct TypedExpressionList {
    types: Punctuated<Type, Token![,]>,
    expressions: Punctuated<Expr, Token![,]>,
}

fn parse_histogram_tmp(
    input: ParseStream<'_>,
    operation: &str,
) -> Result<(Type, Type, Type, LitInt, LitInt)> {
    input.parse::<Token![::]>().map_err(|_| {
        input.error(format!(
            "{operation} needs macro-visible sample/counter/level and channel arguments"
        ))
    })?;
    input.parse::<Token![<]>()?;
    let sample_type = input.parse::<Type>()?;
    input.parse::<Token![,]>()?;
    let counter_type = input.parse::<Type>()?;
    input.parse::<Token![,]>()?;
    let level_type = input.parse::<Type>()?;
    input.parse::<Token![,]>()?;
    let channels = input.parse::<LitInt>()?;
    input.parse::<Token![,]>()?;
    let active_channels = input.parse::<LitInt>()?;
    if input.peek(Token![,]) {
        input.parse::<Token![,]>()?;
    }
    input.parse::<Token![>]>()?;
    let channels_value = channels.base10_parse::<usize>()?;
    let active_value = active_channels.base10_parse::<usize>()?;
    if !(1..=4).contains(&channels_value) {
        return Err(syn::Error::new_spanned(
            channels,
            "DeviceHistogram supports 1 through 4 interleaved channels",
        ));
    }
    if active_value == 0 || active_value > channels_value {
        return Err(syn::Error::new_spanned(
            active_channels,
            "active histogram channels must be in 1..=CHANNELS",
        ));
    }
    Ok((
        sample_type,
        counter_type,
        level_type,
        channels,
        active_channels,
    ))
}

fn parse_histogram_expressions(
    input: ParseStream<'_>,
    operation: &str,
    expected: usize,
) -> Result<Vec<Expr>> {
    let content;
    parenthesized!(content in input);
    let expressions = Punctuated::<Expr, Token![,]>::parse_terminated(&content)?;
    if !input.is_empty() {
        return Err(input.error(format!("unexpected tokens after {operation} operands")));
    }
    if expressions.len() != expected {
        return Err(input.error(format!(
            "{operation} expects {expected} runtime operands but received {}",
            expressions.len()
        )));
    }
    Ok(expressions.into_iter().collect())
}

macro_rules! histogram_call {
    ($name:ident, $operation:literal, $($field:ident),+ $(,)?) => {
        #[derive(Debug)]
        pub(crate) struct $name {
            pub(crate) sample_type: Type,
            pub(crate) counter_type: Type,
            pub(crate) level_type: Type,
            pub(crate) channels: LitInt,
            pub(crate) active_channels: LitInt,
            $(pub(crate) $field: Expr,)+
        }

        impl Parse for $name {
            fn parse(input: ParseStream<'_>) -> Result<Self> {
                let (sample_type, counter_type, level_type, channels, active_channels) =
                    parse_histogram_tmp(input, $operation)?;
                let expected = [$(stringify!($field)),+].len();
                let mut expressions =
                    parse_histogram_expressions(input, $operation, expected)?.into_iter();
                Ok(Self {
                    sample_type,
                    counter_type,
                    level_type,
                    channels,
                    active_channels,
                    $($field: expressions.next().expect("validated histogram operand count"),)+
                })
            }
        }
    };
}

histogram_call!(
    DeviceHistogramWorkspaceCall,
    "device_histogram_workspace!",
    execution,
    samples,
    region,
    output,
);

histogram_call!(
    DeviceHistogramEvenCall,
    "device_histogram_even!",
    execution,
    samples,
    region,
    output,
    lower_levels,
    upper_levels,
    workspace,
);

histogram_call!(
    DeviceHistogramRangeCall,
    "device_histogram_range!",
    execution,
    samples,
    region,
    output,
    levels,
    workspace,
);

impl Parse for TypedExpressionList {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<Token![::]>().map_err(|_| {
            input.error("this CUB operation needs macro-visible TMP arguments; use `::<Type>(...)`")
        })?;
        input.parse::<Token![<]>()?;
        let types = Punctuated::<Type, Token![,]>::parse_separated_nonempty(input)?;
        input.parse::<Token![>]>()?;
        let content;
        parenthesized!(content in input);
        let expressions = Punctuated::<Expr, Token![,]>::parse_terminated(&content)?;
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after the CUB operation operands"));
        }
        Ok(Self { types, expressions })
    }
}

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
    pub(crate) value_type: Type,
    pub(crate) logical_warp_threads: LitInt,
    pub(crate) input: Expr,
    pub(crate) valid_items: Option<Expr>,
    pub(crate) op: Expr,
}

impl Parse for WarpReduceCall {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let (value_type, logical_warp_threads, _) = parse_warp_tmp(input, false)?;
        let content;
        parenthesized!(content in input);
        let expressions = Punctuated::<Expr, Token![,]>::parse_terminated(&content)?;
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after warp_reduce! operands"));
        }
        if !matches!(expressions.len(), 2 | 3) {
            return Err(input.error(
                "warp_reduce! expects ::<Type, WIDTH>(value, op) or ::<Type, WIDTH>(value, valid_items, op)",
            ));
        }
        let mut expressions = expressions.into_iter();
        let input = expressions.next().expect("validated operand count");
        let remaining = expressions.collect::<Vec<_>>();
        let (valid_items, op) = match remaining.as_slice() {
            [op] => (None, op.clone()),
            [valid_items, op] => (Some(valid_items.clone()), op.clone()),
            _ => unreachable!("validated operand count"),
        };
        Ok(Self {
            value_type,
            logical_warp_threads,
            input,
            valid_items,
            op,
        })
    }
}

#[derive(Debug)]
pub(crate) struct WarpReduceItemsCall {
    pub(crate) value_type: Type,
    pub(crate) logical_warp_threads: LitInt,
    pub(crate) items_per_thread: LitInt,
    pub(crate) items: Expr,
    pub(crate) op: Expr,
}

impl Parse for WarpReduceItemsCall {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let (value_type, logical_warp_threads, items_per_thread) = parse_warp_tmp(input, true)?;
        let content;
        parenthesized!(content in input);
        let expressions = Punctuated::<Expr, Token![,]>::parse_terminated(&content)?;
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after warp_reduce_items! operands"));
        }
        if expressions.len() != 2 {
            return Err(input.error("warp_reduce_items! expects ::<Type, WIDTH, ITEMS>(items, op)"));
        }
        let mut expressions = expressions.into_iter();
        Ok(Self {
            value_type,
            logical_warp_threads,
            items_per_thread: items_per_thread.expect("required by parse_warp_tmp"),
            items: expressions.next().expect("validated operand count"),
            op: expressions.next().expect("validated operand count"),
        })
    }
}

fn parse_warp_tmp(
    input: ParseStream<'_>,
    needs_items: bool,
) -> Result<(Type, LitInt, Option<LitInt>)> {
    input.parse::<Token![::]>().map_err(|_| {
        input.error("WarpReduce needs macro-visible TMP arguments; use `::<Type, WIDTH>(...)`")
    })?;
    input.parse::<Token![<]>()?;
    let value_type = input.parse::<Type>()?;
    input.parse::<Token![,]>()?;
    let logical_warp_threads = input.parse::<LitInt>()?;
    validate_logical_warp_threads(&logical_warp_threads)?;
    let items_per_thread = if needs_items {
        input.parse::<Token![,]>()?;
        let items = input.parse::<LitInt>()?;
        if items.base10_parse::<usize>()? == 0 {
            return Err(syn::Error::new_spanned(
                items,
                "warp_reduce_items! requires ITEMS greater than zero",
            ));
        }
        Some(items)
    } else {
        None
    };
    input.parse::<Token![>]>()?;
    Ok((value_type, logical_warp_threads, items_per_thread))
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

macro_rules! typed_operation_call {
    ($name:ident, $operation:literal, [$($type_field:ident),+], $($field:ident),+ $(,)?) => {
        #[derive(Debug)]
        pub(crate) struct $name {
            $(pub(crate) $type_field: Type,)+
            $(pub(crate) $field: Expr,)+
        }

        impl Parse for $name {
            fn parse(input: ParseStream<'_>) -> Result<Self> {
                let TypedExpressionList { types, expressions } = input.parse()?;
                let expected_types = [$(stringify!($type_field)),+].len();
                if types.len() != expected_types {
                    return Err(input.error(format!(
                        "{} expects {} TMP type argument(s) ({}) but received {}",
                        $operation,
                        expected_types,
                        [$(stringify!($type_field)),+].join(", "),
                        types.len(),
                    )));
                }
                let expected_expressions = [$(stringify!($field)),+].len();
                if expressions.len() != expected_expressions {
                    return Err(input.error(format!(
                        "{} expects {} operands ({}) but received {}",
                        $operation,
                        expected_expressions,
                        [$(stringify!($field)),+].join(", "),
                        expressions.len(),
                    )));
                }
                let mut types = types.into_iter();
                let mut expressions = expressions.into_iter();
                Ok(Self {
                    $($type_field: types.next().expect("validated TMP type count"),)+
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

// Rust-native forms of pinned CUB DeviceMerge. Extents come from the borrowed
// buffers, so the C++ count operands are deliberately absent. The ordering is
// an ordinary Rust closure and the output borrow carries write exclusivity.
typed_operation_call!(
    DeviceMergeKeysCall,
    "device_merge_keys!",
    [key_type],
    execution,
    first,
    second,
    workspace,
    output,
    order,
);

typed_operation_call!(
    DeviceMergeKeysWorkspaceCall,
    "device_merge_keys_workspace!",
    [key_type],
    execution,
    first_len,
    second_len,
);

typed_operation_call!(
    DeviceMergeKeysPartitionCountCall,
    "__device_merge_keys_partition_count!",
    [key_type],
    first_len,
    second_len,
);

typed_operation_call!(
    DeviceMergePairsPartitionCountCall,
    "__device_merge_pairs_partition_count!",
    [key_type, value_type],
    first_len,
    second_len,
);

typed_operation_call!(
    DeviceMergePairsCall,
    "device_merge_pairs!",
    [key_type, value_type],
    execution,
    first_keys,
    first_values,
    second_keys,
    second_values,
    workspace,
    output_keys,
    output_values,
    order,
);

typed_operation_call!(
    DeviceMergePairsWorkspaceCall,
    "device_merge_pairs_workspace!",
    [key_type, value_type],
    execution,
    first_len,
    second_len,
);

// Hidden set-operation lowering. The operation is a Rust zero-sized trait
// type (the analogue of CCCL's serial functor type); key/value widths and the
// target affect the pinned policy and therefore remain macro-visible.
typed_operation_call!(
    DeviceSetWorkspaceCountsCall,
    "__device_set_workspace_counts!",
    [key_type, value_type],
    execution,
    first_len,
    second_len,
);

typed_operation_call!(
    DeviceSetKeysCall,
    "__device_set_keys!",
    [key_type, operation_type],
    execution,
    first,
    second,
    workspace,
    output,
    order,
);

typed_operation_call!(
    DeviceSetDiscardCall,
    "__device_set_discard!",
    [key_type, operation_type],
    execution,
    first,
    second,
    workspace,
    order,
);

typed_operation_call!(
    DeviceSetArithmeticCall,
    "__device_set_arithmetic!",
    [key_type, operation_type],
    execution,
    first,
    second,
    workspace,
    output,
    order,
);

typed_operation_call!(
    DeviceSetPairsCall,
    "__device_set_pairs!",
    [key_type, value_type, operation_type],
    execution,
    first_keys,
    first_values,
    second_keys,
    second_values,
    workspace,
    output_keys,
    output_values,
    order,
);

// DeviceMergeSort's in-place key and key/value forms. Copy and stable forms
// have distinct parsers below even where they share a stable implementation:
// their ownership and observable input-preservation contracts differ.
typed_operation_call!(
    DeviceMergeSortKeysCall,
    "device_merge_sort_keys!",
    [key_type],
    execution,
    keys,
    workspace,
    order,
);

typed_operation_call!(
    DeviceMergeSortKeysCopyCall,
    "device_merge_sort_keys_copy!",
    [key_type],
    execution,
    input,
    workspace,
    output,
    order,
);

typed_operation_call!(
    DeviceMergeSortKeysWorkspaceCall,
    "device_merge_sort_keys_workspace!",
    [key_type],
    execution,
    len,
);

typed_operation_call!(
    DeviceMergeSortPartitionCountCall,
    "__device_merge_sort_partition_count!",
    [key_type],
    len,
);

typed_operation_call!(
    DeviceMergeSortPairsCall,
    "device_merge_sort_pairs!",
    [key_type, value_type],
    execution,
    keys,
    values,
    workspace,
    order,
);

typed_operation_call!(
    DeviceMergeSortPairsCopyCall,
    "device_merge_sort_pairs_copy!",
    [key_type, value_type],
    execution,
    input_keys,
    input_values,
    workspace,
    output_keys,
    output_values,
    order,
);

typed_operation_call!(
    DeviceMergeSortPairsWorkspaceCall,
    "device_merge_sort_pairs_workspace!",
    [key_type, value_type],
    execution,
    len,
);

typed_operation_call!(
    DeviceRadixSortKeysCall,
    "device_radix_sort_keys!",
    [key_type],
    execution,
    keys,
    workspace,
);

typed_operation_call!(
    DeviceRadixSortKeysBufferCall,
    "device_radix_sort_keys_buffer!",
    [key_type],
    execution,
    buffers,
    bit_range,
    workspace,
);

typed_operation_call!(
    DeviceRadixSortWorkspaceCall,
    "device_radix_sort_workspace!",
    [key_type],
    execution,
    len,
);

typed_operation_call!(
    DeviceRadixSortWorkspaceCountsCall,
    "__device_radix_sort_workspace_counts!",
    [key_type],
    len,
);

typed_operation_call!(
    DeviceRadixSortKeysBufferWorkspaceCall,
    "device_radix_sort_keys_buffer_workspace!",
    [key_type],
    execution,
    len,
);

typed_operation_call!(
    DeviceRadixSortPairsWorkspaceCall,
    "device_radix_sort_pairs_workspace!",
    [key_type, value_type],
    execution,
    len,
);

typed_operation_call!(
    DeviceRadixSortPairsWorkspaceCountsCall,
    "__device_radix_sort_pairs_workspace_counts!",
    [key_type, value_type],
    len,
);

typed_operation_call!(
    DeviceRadixSortPairsBufferWorkspaceCall,
    "device_radix_sort_pairs_buffer_workspace!",
    [key_type, value_type],
    execution,
    len,
);

typed_operation_call!(
    DeviceRadixSortPairsCall,
    "device_radix_sort_pairs!",
    [key_type, value_type],
    execution,
    keys,
    values,
    workspace,
);

typed_operation_call!(
    DeviceRadixSortPairsBufferCall,
    "device_radix_sort_pairs_buffer!",
    [key_type, value_type],
    execution,
    buffers,
    bit_range,
    workspace,
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
        let full: WarpReduceCall =
            syn::parse_quote! { ::<i32, 32>(value, |left, right| left + right) };
        assert_eq!(full.value_type.to_token_stream().to_string(), "i32");
        assert_eq!(full.logical_warp_threads.base10_parse::<u32>().unwrap(), 32);
        assert!(full.valid_items.is_none());
        assert!(matches!(full.op, Expr::Closure(_)));

        let partial: WarpReduceCall = syn::parse_quote! {
            ::<u64, 16>(value, valid_items, |left, right| left.max(right))
        };
        assert_eq!(tokens(partial.valid_items.as_ref().unwrap()), "valid_items");
        assert!(matches!(partial.op, Expr::Closure(_)));

        let items: WarpReduceItemsCall = syn::parse_quote! {
            ::<i16, 8, 4>(thread_items, |left, right| left + right)
        };
        assert_eq!(items.items_per_thread.base10_parse::<usize>().unwrap(), 4);
        assert_eq!(tokens(&items.items), "thread_items");
    }

    #[test]
    fn warp_reduce_rejects_runtime_and_unsupported_tile_widths() {
        let runtime = syn::parse_str::<WarpReduceCall>(
            "::<i32, logical_width>(value, |left, right| left + right)",
        )
        .expect_err("tile width belongs to the macro's compile-time input");
        assert!(!runtime.to_string().is_empty());

        let non_power =
            syn::parse_str::<WarpReduceCall>("::<i32, 9>(value, |left, right| left + right)")
                .expect_err("the safe backend must not mask unsupported shared-memory tiles");
        assert!(non_power.to_string().contains("1, 2, 4, 8, 16, and 32"));

        let empty_items = syn::parse_str::<WarpReduceItemsCall>(
            "::<i32, 32, 0>(items, |left, right| left + right)",
        )
        .expect_err("a zero-item TMP must fail during macro parsing");
        assert!(empty_items.to_string().contains("greater than zero"));
    }

    #[test]
    fn warp_reduce_accepts_a_literal_forwarded_by_a_rust_macro() {
        let width = proc_macro2::Group::new(proc_macro2::Delimiter::None, quote::quote!(16));
        let call = syn::parse2::<WarpReduceCall>(
            quote::quote!(::<i32, #width>(value, |left, right| left + right)),
        )
        .expect("an invisible macro-expansion group around a literal is still compile-time syntax");
        assert_eq!(call.logical_warp_threads.base10_parse::<u32>().unwrap(), 16);

        let parenthesized: WarpReduceItemsCall =
            syn::parse_quote! { ::<i32, 8, 4>(items, |left, right| left + right) };
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

    #[test]
    fn ordering_tmp_types_are_macro_visible_and_untyped_calls_fail_closed() {
        let keys: DeviceMergeSortKeysCall = syn::parse_quote! {
            ::<u32>(
                &exec,
                &mut keys,
                &mut workspace,
                |left: &u32, right: &u32| left.cmp(right)
            )
        };
        assert_eq!(keys.key_type.to_token_stream().to_string(), "u32");

        let leaked_target = syn::parse_str::<DeviceMergeSortKeysCall>(
            "::<u32, sm_89>(&exec, &mut keys, &mut workspace, |left: &u32, right: &u32| left.cmp(right))",
        )
        .expect_err("CUDA architecture must be selected by the proc macro");
        assert!(leaked_target.to_string().contains("TMP type"));

        let merge_pairs: DeviceMergeSortPairsCall = syn::parse_quote! {
            ::<u32, u32>(
                &exec,
                &mut keys,
                &mut values,
                &mut workspace,
                |left: &u32, right: &u32| left.cmp(right)
            )
        };
        assert_eq!(merge_pairs.value_type.to_token_stream().to_string(), "u32");
        assert_eq!(tokens(&merge_pairs.workspace), "& mut workspace");

        let pairs: DeviceRadixSortPairsCall = syn::parse_quote! {
            ::<u64, i32>(
                &exec,
                &mut keys,
                &mut values,
                &mut workspace
            )
        };
        assert_eq!(pairs.key_type.to_token_stream().to_string(), "u64");
        assert_eq!(pairs.value_type.to_token_stream().to_string(), "i32");

        let error = syn::parse_str::<DeviceRadixSortKeysCall>("&exec, &mut keys")
            .expect_err("runtime-inferred key types must not select a CUB policy");
        assert!(error.to_string().contains("macro-visible TMP"));

        let leaked_radix_parts = syn::parse_str::<DeviceRadixSortKeysCall>(
            "::<u32>(&exec, &mut keys, &mut scratch, &mut bins, &mut lookback, &mut counters)",
        )
        .expect_err("direct CUB must expose one opaque workspace, not its agent buffers");
        assert!(
            leaked_radix_parts
                .to_string()
                .contains("expects 3 operands")
        );

        let leaked_merge_sort_parts = syn::parse_str::<DeviceMergeSortKeysCall>(
            "::<u32>(&exec, &mut keys, &mut scratch, &mut partitions, |a: &u32, b: &u32| a.cmp(b))",
        )
        .expect_err("merge-sort scratch and partitions must remain inside one workspace");
        assert!(
            leaked_merge_sort_parts
                .to_string()
                .contains("expects 4 operands")
        );

        let missing_workspace_execution =
            syn::parse_str::<DeviceRadixSortWorkspaceCall>("::<u32>(len)")
                .expect_err("workspace construction must be tied to a launch stream");
        assert!(
            missing_workspace_execution
                .to_string()
                .contains("expects 2 operands")
        );
    }
}
