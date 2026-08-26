//! `syn` ASTs for shallow, user-facing Thrust algorithm invocations.
//!
//! The macro changes algorithm semantics without replacing Rust syntax. Every
//! value below remains a `syn::Expr`, so closures, methods, paths, borrows, and
//! inference continue through rustc normally.
#![allow(dead_code)]

use syn::{
    Expr, Ident, Result, Token,
    parse::{Parse, ParseStream},
};

#[derive(Debug)]
pub(crate) struct SortByCall {
    pub(crate) on: Expr,
    pub(crate) range: Expr,
    pub(crate) compare: Expr,
}

impl Parse for SortByCall {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut fields = Fields::parse(input, &["on", "range", "compare"])?;
        Ok(Self {
            on: fields.take("on", input)?,
            range: fields.take("range", input)?,
            compare: fields.take("compare", input)?,
        })
    }
}

#[derive(Debug)]
pub(crate) struct StableSortByKeyCall {
    pub(crate) on: Expr,
    pub(crate) keys: Expr,
    pub(crate) values: Expr,
    pub(crate) compare: Expr,
}

impl Parse for StableSortByKeyCall {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut fields = Fields::parse(input, &["on", "keys", "values", "compare"])?;
        Ok(Self {
            on: fields.take("on", input)?,
            keys: fields.take("keys", input)?,
            values: fields.take("values", input)?,
            compare: fields.take("compare", input)?,
        })
    }
}

struct Fields {
    values: std::collections::BTreeMap<String, Expr>,
}

impl Fields {
    fn parse(input: ParseStream<'_>, allowed: &[&str]) -> Result<Self> {
        let mut values = std::collections::BTreeMap::new();
        while !input.is_empty() {
            let field: Ident = input.parse()?;
            let name = field.to_string();
            if !allowed.contains(&name.as_str()) {
                return Err(syn::Error::new(
                    field.span(),
                    "unknown algorithm macro field",
                ));
            }
            input.parse::<Token![:]>()?;
            let value: Expr = input.parse()?;
            if values.insert(name.clone(), value).is_some() {
                return Err(syn::Error::new(
                    field.span(),
                    format!("duplicate `{name}` field"),
                ));
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        Ok(Self { values })
    }

    fn take(&mut self, name: &str, input: ParseStream<'_>) -> Result<Expr> {
        self.values
            .remove(name)
            .ok_or_else(|| input.error(format!("missing `{name}` field")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::ToTokens;

    #[test]
    fn sort_by_keeps_a_rust_closure_and_mutable_borrow() {
        let call: SortByCall = syn::parse_quote! {
            on: &exec,
            range: &mut keys,
            compare: |left: &i64, right: &i64| left > right,
        };
        assert!(matches!(call.compare, Expr::Closure(_)));
        assert!(matches!(call.range, Expr::Reference(_)));
        assert_eq!(call.on.into_token_stream().to_string(), "& exec");
    }

    #[test]
    fn by_key_is_a_distinct_semantic_shape_not_a_name_accident() {
        let call: StableSortByKeyCall = syn::parse_quote! {
            on: execution::<Sm90>(),
            keys: &mut keys,
            values: &mut values,
            compare: ordering::descending::<i64>(),
        };
        assert!(call.on.into_token_stream().to_string().contains("Sm90"));
        assert!(matches!(call.keys, Expr::Reference(_)));
        assert!(matches!(call.values, Expr::Reference(_)));
        assert!(
            call.compare
                .into_token_stream()
                .to_string()
                .contains("descending")
        );
    }

    #[test]
    fn parser_does_not_mask_bad_calls() {
        let duplicate = syn::parse_str::<SortByCall>(
            "on: &exec, range: &mut first, range: &mut second, compare: |a,b| a < b",
        )
        .expect_err("duplicate range must fail");
        assert!(duplicate.to_string().contains("duplicate `range`"));

        let missing = syn::parse_str::<StableSortByKeyCall>(
            "on: &exec, keys: &mut keys, compare: |a,b| a < b",
        )
        .expect_err("values are required by the by-key semantic shape");
        assert!(missing.to_string().contains("missing `values`"));
    }
}
