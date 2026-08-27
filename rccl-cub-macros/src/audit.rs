use proc_macro2::TokenStream;
use syn::File;
use syn::{
    Attribute, Error, Expr, ExprCast, ExprMethodCall, ExprRawAddr, ExprUnsafe, ItemForeignMod,
    ItemImpl, ItemStatic, ItemTrait, ItemUnion, Macro, Path, Signature, StaticMutability,
    TypeFnPtr, TypePtr,
    visit::{self, Visit},
};

#[allow(dead_code)]
pub(crate) fn generated_rust(tokens: TokenStream) -> syn::Result<TokenStream> {
    let file = syn::parse2::<File>(tokens.clone())?;
    let mut audit = SafetyAudit::default();
    audit.visit_file(&file);
    match audit.error {
        Some(error) => Err(error),
        None => Ok(tokens),
    }
}

pub(crate) fn generated_expression(tokens: TokenStream) -> syn::Result<TokenStream> {
    let expression = syn::parse2::<Expr>(tokens.clone())?;
    let mut audit = SafetyAudit::default();
    audit.visit_expr(&expression);
    match audit.error {
        Some(error) => Err(error),
        None => Ok(tokens),
    }
}

#[derive(Default)]
struct SafetyAudit {
    error: Option<Error>,
}

impl SafetyAudit {
    fn reject(&mut self, tokens: impl quote::ToTokens, message: &str) {
        let error = Error::new_spanned(tokens, message);
        if let Some(existing) = &mut self.error {
            existing.combine(error);
        } else {
            self.error = Some(error);
        }
    }
}

fn is_trusted_cub_lowering(path: &Path) -> bool {
    let segments = path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>();
    matches!(
        segments.as_slice(),
        [crate_name, private, operation]
            if crate_name == "rccl_cub"
                && private == "__private"
                && matches!(
                    operation.as_str(),
                    "__device_kernels" | "__device_transform_repeat_in_place"
                )
    )
}

impl<'ast> Visit<'ast> for SafetyAudit {
    fn visit_path(&mut self, path: &'ast Path) {
        if path.segments.iter().any(|segment| segment.ident == "Arc") {
            self.reject(
                path,
                "generated RCCL code may not use Arc; preserve ownership with borrows and affine runtime values",
            );
        }
        if path.segments.iter().any(|segment| {
            matches!(
                segment.ident.to_string().as_str(),
                "NonNull"
                    | "UnsafeCell"
                    | "addr_of"
                    | "addr_of_mut"
                    | "ptr"
                    | "transmute"
                    | "transmute_copy"
            )
        }) {
            self.reject(
                path,
                "generated RCCL code may not use raw-pointer or unsafe-interior plumbing",
            );
        }
        visit::visit_path(self, path);
    }

    fn visit_type_ptr(&mut self, pointer: &'ast TypePtr) {
        self.reject(
            pointer,
            "generated RCCL code may not use raw pointer types; use slices or checked device views",
        );
        visit::visit_type_ptr(self, pointer);
    }

    fn visit_expr_unsafe(&mut self, expression: &'ast ExprUnsafe) {
        self.reject(
            expression,
            "generated RCCL algorithm code may not contain unsafe blocks",
        );
        visit::visit_expr_unsafe(self, expression);
    }

    fn visit_signature(&mut self, signature: &'ast Signature) {
        if matches!(&signature.safety, syn::Safety::Unsafe(_)) {
            self.reject(
                signature,
                "generated RCCL functions and methods may not be unsafe",
            );
        }
        visit::visit_signature(self, signature);
    }

    fn visit_item_trait(&mut self, item: &'ast ItemTrait) {
        if item.unsafety.is_some() {
            self.reject(item, "generated RCCL traits may not be unsafe");
        }
        visit::visit_item_trait(self, item);
    }

    fn visit_item_impl(&mut self, item: &'ast ItemImpl) {
        if item.unsafety.is_some() {
            self.reject(item, "generated RCCL impls may not be unsafe");
        }
        visit::visit_item_impl(self, item);
    }

    fn visit_type_fn_ptr(&mut self, function: &'ast TypeFnPtr) {
        if function.unsafety.is_some() {
            self.reject(function, "generated RCCL function types may not be unsafe");
        }
        visit::visit_type_fn_ptr(self, function);
    }

    fn visit_expr_cast(&mut self, expression: &'ast ExprCast) {
        if matches!(expression.ty.as_ref(), syn::Type::Ptr(_)) {
            self.reject(
                expression,
                "generated RCCL code may not manufacture raw pointers with casts",
            );
        }
        visit::visit_expr_cast(self, expression);
    }

    fn visit_expr_raw_addr(&mut self, expression: &'ast ExprRawAddr) {
        self.reject(
            expression,
            "generated RCCL code may not create raw addresses with `&raw`",
        );
        visit::visit_expr_raw_addr(self, expression);
    }

    fn visit_expr_method_call(&mut self, expression: &'ast ExprMethodCall) {
        if matches!(
            expression.method.to_string().as_str(),
            "as_ptr" | "as_mut_ptr" | "as_ptr_range" | "cast_mut" | "cast_const"
        ) {
            self.reject(
                expression,
                "generated RCCL code may not extract raw pointers from safe values",
            );
        }
        visit::visit_expr_method_call(self, expression);
    }

    fn visit_item_foreign_mod(&mut self, item: &'ast ItemForeignMod) {
        self.reject(
            item,
            "generated RCCL algorithm code may not declare an FFI block",
        );
        visit::visit_item_foreign_mod(self, item);
    }

    fn visit_item_static(&mut self, item: &'ast ItemStatic) {
        if matches!(item.mutability, StaticMutability::Mut(_)) {
            self.reject(item, "generated RCCL code may not declare `static mut`");
        }
        visit::visit_item_static(self, item);
    }

    fn visit_item_union(&mut self, item: &'ast ItemUnion) {
        self.reject(item, "generated RCCL code may not declare unions");
        visit::visit_item_union(self, item);
    }

    fn visit_macro(&mut self, invocation: &'ast Macro) {
        if !is_trusted_cub_lowering(&invocation.path) {
            self.reject(
                invocation,
                "generated RCCL output may not contain opaque nested macro invocations",
            );
        }
        visit::visit_macro(self, invocation);
    }

    fn visit_attribute(&mut self, attribute: &'ast Attribute) {
        let name = attribute
            .path()
            .segments
            .last()
            .map(|segment| segment.ident.to_string());
        let allowed = name.is_some_and(|name| {
            matches!(
                name.as_str(),
                "cfg"
                    | "cuda_module"
                    | "derive"
                    | "doc"
                    | "inline"
                    | "kernel"
                    | "launch_bounds"
                    | "launch_contract"
                    | "repr"
                    | "test"
            )
        });
        if !allowed {
            self.reject(
                attribute,
                "generated RCCL output may use only audited built-in attributes",
            );
        }
        visit::visit_attribute(self, attribute);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    #[test]
    fn safe_borrowed_output_passes() {
        generated_rust(quote! {
            fn fill<T: Copy>(output: &mut [T], value: T) {
                output.fill(value);
            }
        })
        .expect("borrowed slice code must pass");
    }

    #[test]
    fn safe_operation_expression_passes() {
        generated_expression(quote! {{
            let value = 7_i32;
            crate::helper(value, |left, right| left + right)
        }})
        .expect("a safe operation expression must pass");
    }

    #[test]
    fn trusted_cub_lowering_macro_passes() {
        generated_rust(quote! {
            ::rccl_cub::__private::__device_kernels!(__test_kernels);
        })
        .expect("the audited Thrust-to-CUB macro edge must pass");
    }

    #[test]
    fn arc_pointer_unsafe_and_opaque_macro_are_all_rejected() {
        let error = generated_rust(quote! {
            fn bad(value: std::sync::Arc<u32>, pointer: *mut u32, values: &[u32]) {
                let inferred = values.as_ptr();
                unsafe { pointer.write(*value); }
                println!("{inferred:p}");
            }
        })
        .expect_err("unsafe generated ownership must be rejected");
        let message = error.into_compile_error().to_string();
        assert!(message.contains("Arc"));
        assert!(message.contains("raw pointer"));
        assert!(message.contains("unsafe blocks"));
        assert!(message.contains("extract raw pointers"));
        assert!(message.contains("opaque nested macro"));
    }
}
