use quote::quote;

pub(super) fn expand_user_program(
    mut module: syn::ItemMod,
) -> syn::Result<proc_macro2::TokenStream> {
    let ident = module.ident.clone();
    let Some((_brace, items)) = module.content.as_mut() else {
        return Err(syn::Error::new_spanned(
            &module.ident,
            "#[rccl_cub::program] requires an inline module",
        ));
    };
    if module.attrs.iter().any(|attribute| {
        attribute
            .path()
            .segments
            .last()
            .is_some_and(|part| part.ident == "cuda_module")
    }) {
        return Err(syn::Error::new_spanned(
            &module.ident,
            "#[rccl_cub::program] already supplies #[cuda_module]; remove the duplicate attribute",
        ));
    }
    if items.iter().any(|item| {
        matches!(
            item,
            syn::Item::Macro(item_macro)
                if item_macro.ident.as_ref().is_some_and(|name| {
                    name == "program" || name == "__rccl_cub_program"
                })
        )
    }) {
        return Err(syn::Error::new_spanned(
            &module.ident,
            "#[rccl_cub::program] reserves the `program!` macro name",
        ));
    }

    let binding: syn::Item = syn::parse_quote! {
        #[cfg(not(target_arch = "nvptx64"))]
        macro_rules! __rccl_cub_program {
            ($context:expr) => {{
                let __rccl_context = $context;
                unsafe { $crate::#ident::load(__rccl_context) }
            }};
        }
    };
    let export: syn::Item = syn::parse_quote! {
        #[cfg(not(target_arch = "nvptx64"))]
        pub(crate) use __rccl_cub_program as program;
    };
    items.push(binding);
    items.push(export);

    Ok(quote! {
        #[::cuda_device::cuda_module]
        #module
    })
}
