mod audit;
mod call;
#[cfg(test)]
mod catalog;

use proc_macro::TokenStream;
use quote::quote;

#[proc_macro]
pub fn warp_reduce(input: TokenStream) -> TokenStream {
    let call = syn::parse_macro_input!(input as call::WarpReduceCall);
    expand_warp_reduce(call)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro]
pub fn warp_reduce_items(input: TokenStream) -> TokenStream {
    let call = syn::parse_macro_input!(input as call::WarpReduceItemsCall);
    expand_warp_reduce_items(call)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_warp_reduce(call: call::WarpReduceCall) -> syn::Result<proc_macro2::TokenStream> {
    let width = call.logical_warp_threads;
    let value = call.input;
    let op = call.op;
    let valid_items = call
        .valid_items
        .map_or_else(|| quote!(#width), |valid_items| quote!(#valid_items));
    audit::generated_expression(quote! {{
        ::rccl_cub::__private::warp_reduce::<#width, _, _>(
            #value,
            #valid_items,
            #op,
        )
    }})
}

fn expand_warp_reduce_items(
    call: call::WarpReduceItemsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let width = call.logical_warp_threads;
    let items = call.items;
    let op = call.op;
    audit::generated_expression(quote! {{
        ::rccl_cub::__private::warp_reduce_items::<#width, _, _, _>(
            #items,
            #op,
        )
    }})
}

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

    #[test]
    fn warp_reduce_expansion_is_a_safe_rust_expression() {
        let call: call::WarpReduceCall =
            syn::parse_quote! { 32, value, |left, right| left + right };
        let output = expand_warp_reduce(call).expect("the operation must lower");
        let text = output.to_string();
        assert!(text.contains("__private :: warp_reduce"));
        assert!(text.contains("| left , right | left + right"));
        assert!(!text.contains("unsafe"));
        assert!(!text.contains("* mut"));
    }
}
