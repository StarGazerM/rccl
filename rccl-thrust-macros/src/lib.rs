extern crate proc_macro;

mod ast;
mod audit;
mod expand;

use proc_macro::TokenStream;

const THRUST_DSL: &str = include_str!("thrust.dsl");

fn parse_program() -> syn::Result<ast::ThrustProgram> {
    let program = syn::parse_str::<ast::ThrustProgram>(THRUST_DSL)?;
    program.validate()?;
    Ok(program)
}

/// Expand the complete macro-owned Thrust program into `rccl-thrust`.
///
/// Runtime crates cannot pass schedules, policies, or specialization facts.
#[proc_macro]
pub fn expand_thrust(input: TokenStream) -> TokenStream {
    if !input.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "expand_thrust! accepts no runtime-crate configuration",
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
        let program = parse_program().expect("the embedded Thrust DSL must be valid");
        assert_eq!(program.cub.len(), 12);
        assert!(program.supports.len() >= 30);
        assert_eq!(program.adaptors.len(), 19);
        assert_eq!(program.algorithms.len(), 34);
        assert!(
            program
                .algorithms
                .iter()
                .any(|algorithm| algorithm.name == "sort")
        );
        assert!(program.algorithms.iter().all(|algorithm| {
            algorithm.status != "implemented" && algorithm.status != "native_rust"
        }));
        let generated =
            expand::program(&program).expect("generated Thrust Rust must pass its audit");
        syn::parse2::<syn::File>(generated)
            .expect("generated Thrust output must remain valid Rust");
    }

    #[test]
    fn runtime_selection_target_is_rejected() {
        let source = r#"
            cub [device_radix_sort, device_merge_sort];
            algorithm sort {
                status: planned;
                form: in_place;
                requires: [device_radix_sort];
                select { otherwise => device_merge_sort; }
            }
        "#;
        let program = syn::parse_str::<ast::ThrustProgram>(source).expect("syntax must parse");
        let error = program
            .validate()
            .expect_err("an undeclared backend must be rejected");
        assert!(error.to_string().contains("not listed in `requires`"));
    }
}
