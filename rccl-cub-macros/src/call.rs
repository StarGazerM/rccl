//! `syn` ASTs for the shallow, user-facing CUB operation macros.
//!
//! These nodes deliberately retain Rust expressions. The compiler assigns
//! CUB semantics to the fields, while rustc still owns parsing and type
//! checking of paths, closures, borrows, generics, and inferred types.
#![allow(dead_code)]

use syn::{
    Expr, Ident, Result, Token,
    parse::{Parse, ParseStream},
};

mod keyword {
    syn::custom_keyword!(auto);
}

#[derive(Debug)]
pub(crate) enum TempStorageArg {
    Auto,
    Borrowed(Box<Expr>),
}

#[derive(Debug)]
pub(crate) struct DeviceReduceCall {
    pub(crate) on: Expr,
    pub(crate) input: Expr,
    pub(crate) output: Expr,
    pub(crate) init: Expr,
    pub(crate) op: Expr,
    pub(crate) temp: TempStorageArg,
}

impl Parse for DeviceReduceCall {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut on = None;
        let mut range = None;
        let mut output = None;
        let mut init = None;
        let mut op = None;
        let mut temp = None;

        while !input.is_empty() {
            let field: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            match field.to_string().as_str() {
                "on" => set_once(&mut on, input.parse()?, &field)?,
                "input" => set_once(&mut range, input.parse()?, &field)?,
                "output" => set_once(&mut output, input.parse()?, &field)?,
                "init" => set_once(&mut init, input.parse()?, &field)?,
                "op" => set_once(&mut op, input.parse()?, &field)?,
                "temp" => {
                    let value = if input.peek(keyword::auto) {
                        input.parse::<keyword::auto>()?;
                        TempStorageArg::Auto
                    } else {
                        TempStorageArg::Borrowed(Box::new(input.parse()?))
                    };
                    set_once(&mut temp, value, &field)?;
                }
                _ => {
                    return Err(syn::Error::new(
                        field.span(),
                        "unknown device_reduce! field",
                    ));
                }
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }

        Ok(Self {
            on: required(on, "on", input)?,
            input: required(range, "input", input)?,
            output: required(output, "output", input)?,
            init: required(init, "init", input)?,
            op: required(op, "op", input)?,
            temp: required(temp, "temp", input)?,
        })
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, field: &Ident) -> Result<()> {
    if slot.replace(value).is_some() {
        return Err(syn::Error::new(
            field.span(),
            format!("duplicate `{field}` field"),
        ));
    }
    Ok(())
}

fn required<T>(value: Option<T>, field: &str, input: ParseStream<'_>) -> Result<T> {
    value.ok_or_else(|| input.error(format!("missing `{field}` field")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::ToTokens;

    #[test]
    fn rust_closure_paths_borrows_and_inference_remain_rust_syntax() {
        let call: DeviceReduceCall = syn::parse_quote! {
            on: &exec,
            input: values.view::<i64>(),
            output: &mut result,
            init: 0_i64,
            op: |left, right| left + right,
            temp: auto,
        };
        assert_eq!(call.on.into_token_stream().to_string(), "& exec");
        assert!(call.input.into_token_stream().to_string().contains("view"));
        assert!(matches!(call.temp, TempStorageArg::Auto));
        assert!(matches!(call.op, Expr::Closure(_)));
    }

    #[test]
    fn borrowed_storage_is_an_ordinary_expression() {
        let call: DeviceReduceCall = syn::parse_quote! {
            on: execution(), input: &values, output: &mut result,
            init: Default::default(), op: ops::sum::<i64>(), temp: &mut scratch,
        };
        assert!(matches!(
            call.temp,
            TempStorageArg::Borrowed(expression) if matches!(*expression, Expr::Reference(_))
        ));
    }

    #[test]
    fn missing_and_duplicate_fields_are_rejected() {
        let missing = syn::parse_str::<DeviceReduceCall>(
            "on: &exec, input: &values, output: &mut out, init: 0, op: |a,b| a+b",
        )
        .expect_err("temp is semantic and required");
        assert!(missing.to_string().contains("missing `temp`"));

        let duplicate = syn::parse_str::<DeviceReduceCall>(
            "on: &exec, on: &other, input: &values, output: &mut out, init: 0, op: |a,b| a+b, temp: auto",
        )
        .expect_err("duplicates must not be silently masked");
        assert!(duplicate.to_string().contains("duplicate `on`"));
    }
}
