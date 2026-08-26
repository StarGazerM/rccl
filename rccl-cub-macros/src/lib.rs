extern crate proc_macro;

mod audit;
mod call;
mod catalog;
mod expand;

use proc_macro::TokenStream;

/// Emit the Rust-native CUB support vocabulary into `rccl-cub`.
///
/// This is an internal crate-construction macro, not the user-facing CUB
/// operation DSL. User operations are shallow call-site macros and are only
/// exported after their cuda-oxide lowering exists.
#[proc_macro]
pub fn emit_support(input: TokenStream) -> TokenStream {
    if !input.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "emit_support! accepts no runtime-crate configuration",
        )
        .into_compile_error()
        .into();
    }
    if let Err(message) = catalog::validate() {
        return syn::Error::new(proc_macro2::Span::call_site(), message)
            .into_compile_error()
            .into();
    }
    match expand::support() {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_catalog_is_complete_and_valid() {
        catalog::validate().expect("the typed CUB compiler catalog must be valid");
        assert_eq!(catalog::TARGETS.len(), 4);
        assert!(catalog::SUPPORT.len() >= 15);
        assert_eq!(catalog::PRIMITIVES.len(), 59);
        assert!(
            catalog::PRIMITIVES
                .iter()
                .any(|primitive| primitive.name == "warp_merge_sort")
        );
        assert!(
            catalog::PRIMITIVES
                .iter()
                .any(|primitive| primitive.name == "device_radix_sort")
        );
    }

    #[test]
    fn generated_support_is_valid_audited_rust() {
        let generated = expand::support().expect("generated CUB Rust must pass its audit");
        syn::parse2::<syn::File>(generated).expect("generated CUB output must remain valid Rust");
    }
}
