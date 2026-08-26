mod audit;
mod call;
#[cfg(test)]
mod catalog;

use proc_macro::TokenStream;
use quote::quote;

#[proc_macro]
pub fn thrust_kernels(input: TokenStream) -> TokenStream {
    if !input.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "thrust_kernels! takes no input",
        )
        .into_compile_error()
        .into();
    }
    expand_thrust_kernels()
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[proc_macro]
pub fn transform(input: TokenStream) -> TokenStream {
    let call = syn::parse_macro_input!(input as call::TransformRepeatInPlaceCall);
    expand_transform(call)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_thrust_kernels() -> syn::Result<proc_macro2::TokenStream> {
    audit::generated_rust(quote! {
        ::rccl_cub::__private::__device_transform_kernels!(__rccl_thrust_kernels);
    })
}

fn expand_transform(
    call: call::TransformRepeatInPlaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let execution = call.execution;
    let output = call.output;
    let repeated = call.repeated;
    let op = call.op;
    audit::generated_expression(quote! {{
        let __rccl_execution = #execution;
        let __rccl_output = #output;
        let __rccl_repeated = #repeated;
        let __rccl_op = #op;
        let __rccl_output_len = ::rccl_thrust::DeviceRange::len(&*__rccl_output);
        if ::rccl_thrust::DeviceRange::len(&__rccl_repeated) != __rccl_output_len {
            ::core::result::Result::Err(::rccl_thrust::TransformError::LengthMismatch {
                input: ::rccl_thrust::DeviceRange::len(&__rccl_repeated),
                output: __rccl_output_len,
            })
        } else if __rccl_output_len == 0 {
            ::core::result::Result::Ok(())
        } else {
            ::rccl_cub::__private::__device_transform_repeat_in_place!(
                __rccl_execution.module(),
                __rccl_execution.stream(),
                __rccl_output,
                *__rccl_repeated.value(),
                __rccl_op,
            )
            .map_err(::rccl_thrust::TransformError::Launch)
        }
    }})
}

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

    #[test]
    fn transform_expansion_is_safe_and_keeps_rust_borrows() {
        let call: call::TransformRepeatInPlaceCall = syn::parse_quote! {
            &exec, &mut values, repeated, |left, right| left + right
        };
        let output = expand_transform(call).expect("transform must lower");
        let text = output.to_string();
        assert!(text.contains("__device_transform_repeat_in_place"));
        assert!(text.contains("& mut values"));
        assert!(!text.contains("unsafe"));
        assert!(!text.contains("* mut"));
    }
}
