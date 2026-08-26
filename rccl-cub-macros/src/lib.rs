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
        catalog::validate().expect("the typed CUB compiler catalog must be valid");
        assert_eq!(catalog::TARGETS.len(), 4);
        assert_eq!(catalog::PRIMITIVES.len(), 59);
        assert_eq!(catalog::EXERCISED_OPERATIONS.len(), 105);
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
    fn generated_algorithm_audit_accepts_safe_rust() {
        let generated = audit::generated_rust(quote::quote! {
            fn generated_helper(value: usize) -> usize { value + 1 }
        })
        .expect("safe generated Rust must pass its audit");
        syn::parse2::<syn::File>(generated).expect("audited output must remain valid Rust");
    }
}
