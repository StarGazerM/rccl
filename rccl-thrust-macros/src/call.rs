//! `syn` ASTs for shallow Thrust algorithm invocations.
//!
//! Calls stay close to ordinary Rust function syntax: positional Rust
//! expressions inside a macro invocation. Rustc still owns type inference,
//! closure checking, paths, and borrows.
#![allow(dead_code)]

use syn::{
    Expr, Result, Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

macro_rules! algorithm_call {
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

algorithm_call!(SortByCall, "sort_by!", execution, range, compare);
algorithm_call!(
    StableSortByKeyCall,
    "stable_sort_by_key!",
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
            &exec, &mut keys, |left: &i64, right: &i64| right.cmp(left)
        };
        assert!(matches!(call.compare, Expr::Closure(_)));
        assert!(matches!(call.range, Expr::Reference(_)));
        assert_eq!(call.execution.into_token_stream().to_string(), "& exec");
    }

    #[test]
    fn by_key_is_a_distinct_four_operand_shape() {
        let call: StableSortByKeyCall = syn::parse_quote! {
            &exec, &mut keys, &mut values, |left: &i64, right: &i64| left.cmp(right)
        };
        assert!(matches!(call.keys, Expr::Reference(_)));
        assert!(matches!(call.values, Expr::Reference(_)));
        assert!(matches!(call.compare, Expr::Closure(_)));
    }

    #[test]
    fn parser_does_not_mask_missing_extra_or_old_named_operands() {
        let missing = syn::parse_str::<StableSortByKeyCall>(
            "&exec, &mut keys, |left: &i64, right: &i64| left.cmp(right)",
        )
        .expect_err("the values range is required by the by-key shape");
        assert!(missing.to_string().contains("expects 4 operands"));

        let extra = syn::parse_str::<SortByCall>(
            "&exec, &mut keys, |left: &i64, right: &i64| left.cmp(right), accidental",
        )
        .expect_err("an extra operand must not be ignored");
        assert!(extra.to_string().contains("expects 3 operands"));

        let obsolete = syn::parse_str::<SortByCall>(
            "on: &exec, range: &mut keys, compare: |left: &i64, right: &i64| left.cmp(right)",
        )
        .expect_err("the invented named-field API must stay deleted");
        assert!(!obsolete.to_string().is_empty());
    }
}
