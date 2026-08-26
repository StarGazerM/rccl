extern crate proc_macro;

mod ast;
mod audit;
mod expand;

use proc_macro::TokenStream;

const CUB_DSL: &str = include_str!("cub.dsl");

fn parse_program() -> syn::Result<ast::CubProgram> {
    let program = syn::parse_str::<ast::CubProgram>(CUB_DSL)?;
    program.validate()?;
    Ok(program)
}

/// Expand the complete macro-owned CUB program into the `rccl-cub` crate.
///
/// The invocation deliberately accepts no tokens: runtime crates cannot
/// restate schedules, policies, kernel inventories, or specialization facts.
#[proc_macro]
pub fn expand_cub(input: TokenStream) -> TokenStream {
    if !input.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "expand_cub! accepts no runtime-crate configuration",
        )
        .into_compile_error()
        .into();
    }
    match parse_program() {
        Ok(program) => match expand::program(&program) {
            Ok(tokens) => tokens.into(),
            Err(error) => error.into_compile_error().into(),
        },
        Err(error) => error.into_compile_error().into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_catalog_parses_and_validates() {
        let program = parse_program().expect("the embedded CUB DSL must be valid");
        assert_eq!(program.targets.len(), 4);
        assert!(program.supports.len() >= 15);
        assert_eq!(program.primitives.len(), 59);
        assert!(
            program
                .primitives
                .iter()
                .any(|primitive| primitive.name == "warp_merge_sort")
        );
        assert!(
            program
                .primitives
                .iter()
                .any(|primitive| primitive.name == "device_radix_sort")
        );
        let generated = expand::program(&program).expect("generated CUB Rust must pass its audit");
        syn::parse2::<syn::File>(generated).expect("generated CUB output must remain valid Rust");
    }

    #[test]
    fn dependency_cycle_is_rejected() {
        let source = r#"
            targets [sm80];
            schedule basic { target sm80 { threads: 32; } }
            primitive first {
                status: planned; scope: device; requires: [second];
                schedule: basic; kernels: [];
            }
            primitive second {
                status: planned; scope: device; requires: [first];
                schedule: basic; kernels: [];
            }
        "#;
        let program = syn::parse_str::<ast::CubProgram>(source).expect("syntax must parse");
        let error = program.validate().expect_err("cycle must be rejected");
        assert!(
            error.to_string().contains("first -> second -> first")
                || error.to_string().contains("second -> first -> second")
        );
    }
}
