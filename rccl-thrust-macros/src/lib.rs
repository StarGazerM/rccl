extern crate proc_macro;

mod audit;
mod call;
mod catalog;
mod expand;

use proc_macro::TokenStream;

/// Emit the Rust-native Thrust support vocabulary into `rccl-thrust`.
///
/// User-facing algorithms are separate shallow call-site macros. They are
/// exported one by one only when their CUB/cuda-oxide lowering is real.
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
        catalog::validate().expect("the typed Thrust compiler catalog must be valid");
        assert_eq!(catalog::ADAPTORS.len(), 19);
        assert_eq!(catalog::ALGORITHMS.len(), 34);
        assert!(catalog::SUPPORT.len() >= 50);
        assert!(
            catalog::ALGORITHMS
                .iter()
                .any(|algorithm| algorithm.name == "sort")
        );
    }

    #[test]
    fn generated_support_is_valid_audited_rust() {
        let generated = expand::support().expect("generated Thrust Rust must pass its audit");
        syn::parse2::<syn::File>(generated)
            .expect("generated Thrust output must remain valid Rust");
    }
}
