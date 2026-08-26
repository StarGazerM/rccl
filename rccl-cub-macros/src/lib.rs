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

/// Internal item lowering used by the Thrust macro layer.
#[doc(hidden)]
#[proc_macro]
pub fn __device_transform_kernels(input: TokenStream) -> TokenStream {
    let module = syn::parse_macro_input!(input as syn::Ident);
    expand_device_transform_kernels(module)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Internal operation lowering used by the Thrust macro layer.
#[doc(hidden)]
#[proc_macro]
pub fn __device_transform_repeat_in_place(input: TokenStream) -> TokenStream {
    let call = syn::parse_macro_input!(input as call::DeviceTransformRepeatInPlaceCall);
    expand_device_transform_repeat_in_place(call)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_device_transform_kernels(module: syn::Ident) -> syn::Result<proc_macro2::TokenStream> {
    audit::generated_rust(quote! {
        #[doc(hidden)]
        use ::rccl_cub::__private::cuda_core;
        #[doc(hidden)]
        use ::rccl_cub::__private::cuda_host;

        #[::rccl_cub::__private::cuda_device::cuda_module]
        #[doc(hidden)]
        mod #module {
            use ::rccl_cub::__private::cuda_device::{
                DisjointSlice, LinearTiles, kernel, launch_bounds, launch_contract, thread,
            };

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn transform_repeat_in_place<T, F>(
                repeated: T,
                op: F,
                mut output: DisjointSlice<T, LinearTiles<1>>,
            )
            where
                T: Copy,
                F: Fn(T, T) -> T + Copy,
            {
                let index = thread::index_1d_u32(launch_context);
                let mut runs = output.grid_stride_runs32(index);
                while let Some(mut run) = runs.next_run() {
                    if let Some(mut slot) = run.at(0) {
                        let value = slot.read();
                        slot.write(op(value, repeated));
                    }
                }
            }
        }
    })
}

fn expand_device_transform_repeat_in_place(
    call: call::DeviceTransformRepeatInPlaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let module = call.module;
    let stream = call.stream;
    let output = call.output;
    let repeated = call.repeated;
    let op = call.op;
    audit::generated_expression(quote! {{
        let __rccl_module = #module;
        let __rccl_stream = #stream;
        let __rccl_output = #output;
        let __rccl_repeated = #repeated;
        let __rccl_op = #op;
        let __rccl_output_len = __rccl_output.len();
        if __rccl_output_len == 0 {
            ::core::result::Result::Ok(())
        } else {
            let __rccl_launch = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                __rccl_output_len.div_ceil(8).div_ceil(256) as u32,
                256,
                0,
            );
            match __rccl_module.prepare_transform_repeat_in_place_for(
                &__rccl_repeated,
                &__rccl_op,
                __rccl_launch,
            ) {
                ::core::result::Result::Ok(__rccl_prepared) => __rccl_module
                    .transform_repeat_in_place(
                        __rccl_stream,
                        &__rccl_prepared,
                        __rccl_repeated,
                        __rccl_op,
                        __rccl_output,
                    ),
                ::core::result::Result::Err(__rccl_error) => {
                    ::core::result::Result::Err(__rccl_error)
                }
            }
        }
    }})
}

fn expand_warp_reduce(call: call::WarpReduceCall) -> syn::Result<proc_macro2::TokenStream> {
    let width = call.logical_warp_threads;
    let value = call.input;
    let op = call.op;
    let expansion = match call.valid_items {
        Some(valid_items) => quote! {{
            ::rccl_cub::__private::warp_reduce::<#width, _, _>(
                #value,
                #valid_items,
                #op,
            )
        }},
        None if call::is_addition_closure(&op) => quote! {{
            ::rccl_cub::__private::warp_reduce_sum_full::<#width, _>(#value)
        }},
        None => quote! {{
            ::rccl_cub::__private::warp_reduce_full::<#width, _, _>(#value, #op)
        }},
    };
    audit::generated_expression(expansion)
}

fn expand_warp_reduce_items(
    call: call::WarpReduceItemsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let width = call.logical_warp_threads;
    let items = call.items;
    let op = call.op;
    let function = if call::is_addition_closure(&op) {
        quote!(::rccl_cub::__private::warp_reduce_items_sum)
    } else {
        quote!(::rccl_cub::__private::warp_reduce_items)
    };
    audit::generated_expression(quote! {{
        #function::<#width, _, _, _>(#items, #op)
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
        assert!(text.contains("__private :: warp_reduce_sum_full"));
        assert!(!text.contains("shuffle_down"));
        assert!(!text.contains("unsafe"));
        assert!(!text.contains("* mut"));
    }

    #[test]
    fn device_transform_lowering_owns_kernel_and_launch_rust() {
        let kernels = expand_device_transform_kernels(syn::parse_quote!(__test_kernels))
            .expect("CUB must generate the transform kernel");
        let kernel_text = kernels.to_string();
        assert!(kernel_text.contains("transform_repeat_in_place"));
        assert!(kernel_text.contains("grid_stride_runs32"));

        let call: call::DeviceTransformRepeatInPlaceCall = syn::parse_quote! {
            module, stream, &mut output, repeated, |left, right| left + right
        };
        let operation = expand_device_transform_repeat_in_place(call)
            .expect("CUB must lower the transform launch");
        let operation_text = operation.to_string();
        assert!(operation_text.contains("prepare_transform_repeat_in_place_for"));
        assert!(!operation_text.contains("unsafe"));
        assert!(!operation_text.contains("* mut"));
    }
}
