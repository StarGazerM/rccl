#[cfg(test)]
mod audit;
#[cfg(test)]
mod call;
#[cfg(test)]
mod catalog;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_catalog_is_complete_and_valid() {
        catalog::validate().expect("the typed Thrust compiler catalog must be valid");
        assert_eq!(catalog::ADAPTORS.len(), 19);
        assert_eq!(catalog::ALGORITHMS.len(), 34);
        assert!(
            catalog::ALGORITHMS
                .iter()
                .any(|algorithm| algorithm.name == "sort")
        );
    }

    #[test]
    fn generated_algorithm_audit_accepts_safe_rust() {
        let generated = audit::generated_rust(quote::quote! {
            fn generated_helper(value: usize) -> usize { value + 1 }
        })
        .expect("safe generated Rust must pass its audit");
        syn::parse2::<syn::File>(generated).expect("audited output must remain valid Rust");
    }
}
