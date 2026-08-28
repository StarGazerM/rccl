//! `syn` ASTs for shallow Thrust algorithm invocations.
//!
//! Calls stay close to ordinary Rust function syntax: explicit TMP types plus
//! positional Rust expressions. Rustc owns checking, paths, and borrows, but
//! any type fact needed to select a lowering must be visible to this parser.
#![allow(dead_code)]

use syn::{
    Expr, Result, Token, Type, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

#[derive(Debug)]
struct TypedExpressionList {
    types: Punctuated<Type, Token![,]>,
    expressions: Punctuated<Expr, Token![,]>,
}

impl Parse for TypedExpressionList {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<Token![::]>().map_err(|_| {
            input.error(
                "this Thrust algorithm needs macro-visible TMP arguments; use `::<Type>(...)`",
            )
        })?;
        input.parse::<Token![<]>()?;
        let types = Punctuated::<Type, Token![,]>::parse_separated_nonempty(input)?;
        input.parse::<Token![>]>()?;
        let content;
        parenthesized!(content in input);
        let expressions = Punctuated::<Expr, Token![,]>::parse_terminated(&content)?;
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after the Thrust algorithm operands"));
        }
        Ok(Self { types, expressions })
    }
}

#[derive(Debug, syn_derive::Parse)]
struct ExpressionList {
    #[parse(Punctuated::parse_terminated)]
    expressions: Punctuated<Expr, Token![,]>,
}

macro_rules! algorithm_call {
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

macro_rules! typed_algorithm_call {
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

typed_algorithm_call!(SortCall, "sort!", [key_type], execution, range);
typed_algorithm_call!(
    SortByCall,
    "sort_by!",
    [key_type],
    execution,
    range,
    compare
);
typed_algorithm_call!(
    SortByKeyCall,
    "sort_by_key!",
    [key_type, value_type],
    execution,
    keys,
    values
);
typed_algorithm_call!(
    SortByKeyWithCall,
    "sort_by_key_with!",
    [key_type, value_type],
    execution,
    keys,
    values,
    compare,
);
typed_algorithm_call!(
    MergeCall,
    "merge!",
    [value_type],
    execution,
    first,
    second,
    output
);
typed_algorithm_call!(
    MergeByCall,
    "merge_by!",
    [value_type],
    execution,
    first,
    second,
    output,
    compare,
);
typed_algorithm_call!(
    MergeByKeyCall,
    "merge_by_key!",
    [key_type, value_type],
    execution,
    first_keys,
    first_values,
    second_keys,
    second_values,
    output_keys,
    output_values,
);
typed_algorithm_call!(
    MergeByKeyWithCall,
    "merge_by_key_with!",
    [key_type, value_type],
    execution,
    first_keys,
    first_values,
    second_keys,
    second_values,
    output_keys,
    output_values,
    compare,
);
algorithm_call!(
    TransformRepeatInPlaceCall,
    "transform!",
    execution,
    output,
    repeated,
    op,
);
typed_algorithm_call!(
    StableSortByKeyCall,
    "stable_sort_by_key!",
    [key_type, value_type],
    execution,
    keys,
    values,
    compare,
);

#[cfg(test)]
mod tests {
    use super::*;
    use quote::ToTokens;

    #[test]
    fn sort_by_is_positional_rust_syntax_with_an_ordering_closure() {
        let call: SortByCall = syn::parse_quote! {
            ::<i64>(
                &exec,
                &mut keys,
                |left: &i64, right: &i64| right.cmp(left)
            )
        };
        assert!(matches!(call.compare, Expr::Closure(_)));
        assert!(matches!(call.range, Expr::Reference(_)));
        assert_eq!(call.execution.into_token_stream().to_string(), "& exec");
    }

    #[test]
    fn by_key_with_comparator_exposes_only_thrust_operands() {
        let call: StableSortByKeyCall = syn::parse_quote! {
            ::<i64, i32>(
                &exec, &mut keys, &mut values,
                |left: &i64, right: &i64| left.cmp(right)
            )
        };
        assert!(matches!(call.keys, Expr::Reference(_)));
        assert!(matches!(call.values, Expr::Reference(_)));
        assert!(matches!(call.compare, Expr::Closure(_)));
    }

    #[test]
    fn transform_repeat_in_place_keeps_the_closure_and_borrows_as_rust() {
        let call: TransformRepeatInPlaceCall = syn::parse_quote! {
            &exec, &mut values, RepeatN::new(10_i32, values.len()),
            |left, right| left + right
        };
        assert!(matches!(call.output, Expr::Reference(_)));
        assert!(matches!(call.repeated, Expr::Call(_)));
        assert!(matches!(call.op, Expr::Closure(_)));
    }

    #[test]
    fn parser_does_not_mask_missing_extra_or_old_named_operands() {
        let missing = syn::parse_str::<StableSortByKeyCall>(
            "::<i64, i32>(&exec, &mut keys, |left: &i64, right: &i64| left.cmp(right))",
        )
        .expect_err("the values range is required by the by-key shape");
        assert!(missing.to_string().contains("expects 4 operands"));

        let extra = syn::parse_str::<SortByCall>(
            "::<i64>(&exec, &mut keys, |left: &i64, right: &i64| left.cmp(right), accidental)",
        )
        .expect_err("an extra operand must not be ignored");
        assert!(extra.to_string().contains("expects 3 operands"));

        let obsolete = syn::parse_str::<SortByCall>(
            "on: &exec, range: &mut keys, compare: |left: &i64, right: &i64| left.cmp(right)",
        )
        .expect_err("the invented named-field API must stay deleted");
        assert!(!obsolete.to_string().is_empty());

        let leaked_key_workspace = syn::parse_str::<SortCall>(
            "::<i64>(&exec, &mut keys, &mut scratch_keys, &mut bins, &mut lookback, &mut counters)",
        )
        .expect_err("Thrust sort must not expose CUB radix workspace");
        assert!(
            leaked_key_workspace
                .to_string()
                .contains("expects 2 operands")
        );

        let leaked_pair_workspace = syn::parse_str::<SortByKeyCall>(
            "::<i64, i32>(&exec, &mut keys, &mut values, &mut scratch_keys, &mut scratch_values, &mut bins, &mut lookback, &mut counters)",
        )
        .expect_err("Thrust sort_by_key must not expose CUB radix workspace");
        assert!(
            leaked_pair_workspace
                .to_string()
                .contains("expects 3 operands")
        );
    }
}
