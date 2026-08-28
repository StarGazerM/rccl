use proc_macro2::TokenStream;
use syn::File;
use syn::{
    Attribute, Error, Expr, ExprCast, ExprLit, ExprMethodCall, ExprRawAddr, ExprUnsafe,
    ItemForeignMod, ItemImpl, ItemStatic, ItemTrait, ItemUnion, Lit, Macro, Path, Signature,
    StaticMutability, TypeFnPtr, TypePtr,
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

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Algorithm {
    Merge,
    MergeSort,
    RadixSort,
}

/// Audit an operation lowering after ordinary Rust safety checks.
///
/// These checks protect semantic boundaries that Rust's type checker cannot
/// see. In particular, an implementation with the right result type is not a
/// CUB port when it allocates inside the call, selects a schedule at runtime,
/// or substitutes a different algorithm.
#[allow(dead_code)]
pub(crate) fn generated_algorithm_expression(
    algorithm: Algorithm,
    tokens: TokenStream,
) -> syn::Result<TokenStream> {
    let expression = syn::parse2::<Expr>(tokens.clone())?;

    let mut safety = SafetyAudit::default();
    safety.visit_expr(&expression);
    if let Some(error) = safety.error {
        return Err(error);
    }

    let mut implementation = ImplementationAudit::new(algorithm);
    implementation.visit_expr(&expression);
    implementation.require_topology(&expression);
    match implementation.error {
        Some(error) => Err(error),
        None => Ok(tokens),
    }
}

#[derive(Default)]
struct SafetyAudit {
    error: Option<Error>,
}

struct ImplementationAudit {
    algorithm: Algorithm,
    error: Option<Error>,
    saw_merge_partition: bool,
    saw_merge_agent: bool,
    saw_merge_sort_block_sort: bool,
    saw_merge_sort_partition: bool,
    saw_merge_sort_merge: bool,
    saw_radix_init: bool,
    saw_radix_histogram: bool,
    saw_radix_exclusive_sum: bool,
    saw_radix_onesweep: bool,
}

impl ImplementationAudit {
    fn new(algorithm: Algorithm) -> Self {
        Self {
            algorithm,
            error: None,
            saw_merge_partition: false,
            saw_merge_agent: false,
            saw_merge_sort_block_sort: false,
            saw_merge_sort_partition: false,
            saw_merge_sort_merge: false,
            saw_radix_init: false,
            saw_radix_histogram: false,
            saw_radix_exclusive_sum: false,
            saw_radix_onesweep: false,
        }
    }

    fn require_topology(&mut self, expression: &Expr) {
        match self.algorithm {
            Algorithm::Merge => {
                if !self.saw_merge_partition {
                    self.reject(
                        expression,
                        "DeviceMerge lowering must launch CUB's merge-path partition kernel",
                    );
                }
                if !self.saw_merge_agent {
                    self.reject(
                        expression,
                        "DeviceMerge lowering must launch CUB's block merge agent",
                    );
                }
            }
            Algorithm::MergeSort => {
                if !self.saw_merge_sort_block_sort {
                    self.reject(
                        expression,
                        "DeviceMergeSort lowering must launch CUB's AgentBlockSort kernel",
                    );
                }
                if !self.saw_merge_sort_partition {
                    self.reject(
                        expression,
                        "DeviceMergeSort lowering must launch CUB's AgentPartition kernel",
                    );
                }
                if !self.saw_merge_sort_merge {
                    self.reject(
                        expression,
                        "DeviceMergeSort lowering must launch CUB's AgentMerge kernel",
                    );
                }
            }
            Algorithm::RadixSort => {
                if !self.saw_radix_init {
                    self.reject(
                        expression,
                        "DeviceRadixSort lowering must initialize CUB's bins and pass counters",
                    );
                }
                if !self.saw_radix_histogram {
                    self.reject(
                        expression,
                        "DeviceRadixSort lowering must launch CUB's multi-pass histogram agent",
                    );
                }
                if !self.saw_radix_exclusive_sum {
                    self.reject(
                        expression,
                        "DeviceRadixSort lowering must launch CUB's bin exclusive-sum kernel",
                    );
                }
                if !self.saw_radix_onesweep {
                    self.reject(
                        expression,
                        "DeviceRadixSort lowering must launch CUB's decoupled-lookback one-sweep agent",
                    );
                }
            }
        }
    }

    fn observe_merge_sort_name(&mut self, name: &str) {
        self.saw_merge_sort_block_sort |= name.starts_with("device_merge_sort_block_keys_")
            || name.starts_with("device_merge_sort_block_pairs_");
        self.saw_merge_sort_partition |= name == "device_merge_sort_partition";
        self.saw_merge_sort_merge |= name.starts_with("device_merge_sort_merge_keys_")
            || name.starts_with("device_merge_sort_merge_pairs_");
    }

    fn observe_radix_name(&mut self, name: &str) {
        self.saw_radix_init |= name == "device_radix_sort_init";
        self.saw_radix_histogram |=
            name == "device_radix_sort_histogram_4" || name == "device_radix_sort_histogram_8";
        self.saw_radix_exclusive_sum |= name == "device_radix_sort_exclusive_sum";
        self.saw_radix_onesweep |= name.starts_with("device_radix_sort_onesweep_keys_")
            || name.starts_with("device_radix_sort_onesweep_pairs_");
    }

    fn reject(&mut self, tokens: impl quote::ToTokens, message: &str) {
        let error = Error::new_spanned(tokens, message);
        if let Some(existing) = &mut self.error {
            existing.combine(error);
        } else {
            self.error = Some(error);
        }
    }
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
                    "__device_kernels"
                        | "__device_transform_repeat_in_place"
                        | "static_block_shared_memory"
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
        if path.segments.iter().any(|segment| {
            matches!(
                segment.ident.to_string().as_str(),
                "nvrtc"
                    | "compile_ptx"
                    | "load_ptx"
                    | "load_cubin"
                    | "materialized_cubin"
                    | "ptxas"
                    | "cuModuleLoadData"
                    | "cuModuleLoadDataEx"
            )
        }) {
            self.reject(
                path,
                "generated RCCL code may not compile, embed, or load PTX/cubin artifacts; emit pure Rust for cuda-oxide",
            );
        }
        visit::visit_path(self, path);
    }

    fn visit_expr_lit(&mut self, expression: &'ast ExprLit) {
        if let Lit::Str(literal) = &expression.lit {
            let value = literal.value();
            if value.contains(".ptx")
                || value.contains(".cubin")
                || value.contains(".version ")
                || value.contains(".target sm_")
            {
                self.reject(
                    expression,
                    "generated RCCL code may not contain CUDA assembly or materialized PTX/cubin paths; emit pure Rust for cuda-oxide",
                );
            }
        }
        visit::visit_expr_lit(self, expression);
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
        let method = expression.method.to_string();
        if matches!(
            method.as_str(),
            "as_ptr" | "as_mut_ptr" | "as_ptr_range" | "cast_mut" | "cast_const"
        ) {
            self.reject(
                expression,
                "generated RCCL code may not extract raw pointers from safe values",
            );
        }
        if matches!(
            method.as_str(),
            "compile_ptx" | "load_ptx" | "load_cubin" | "materialized_cubin" | "ptxas"
        ) {
            self.reject(
                expression,
                "generated RCCL code may not compile, embed, or load PTX/cubin artifacts; emit pure Rust for cuda-oxide",
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
        let exact_device_abi_lint = attribute.path().is_ident("allow")
            && attribute.parse_args::<Path>().is_ok_and(|path| {
                path.segments.len() == 2
                    && path.segments[0].ident == "clippy"
                    && path.segments[1].ident == "too_many_arguments"
            });
        let allowed = exact_device_abi_lint
            || name.is_some_and(|name| {
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

impl<'ast> Visit<'ast> for ImplementationAudit {
    fn visit_path(&mut self, path: &'ast Path) {
        let names = path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect::<Vec<_>>();

        if names
            .windows(2)
            .any(|pair| pair == ["DeviceBuffer", "zeroed"])
        {
            self.reject(
                path,
                "generated CUB operations may not allocate or clear temporary storage; borrow operation workspace prepared outside the launch path",
            );
        }

        match self.algorithm {
            Algorithm::Merge => {
                self.saw_merge_partition |= names
                    .iter()
                    .any(|name| name == "device_partition_merge_path");
                self.saw_merge_agent |= names.iter().any(|name| {
                    matches!(
                        name.as_str(),
                        "device_merge_keys_agent" | "device_merge_pairs_agent"
                    )
                });
                if names.iter().any(|name| name == "stable_left_partition") {
                    self.reject(
                        path,
                        "DeviceMerge may not substitute one global MergePath search per output thread; port the partition kernel and block merge agent",
                    );
                }
            }
            Algorithm::MergeSort => {
                for name in &names {
                    self.observe_merge_sort_name(name);
                }
                if names.iter().any(|name| {
                    matches!(
                        name.as_str(),
                        "merge_sort_source"
                            | "device_merge_sort_keys_pass"
                            | "device_merge_sort_pairs_pass"
                    )
                }) {
                    self.reject(
                        path,
                        "DeviceMergeSort may not substitute one-output-per-thread pairwise passes; port AgentBlockSort, AgentPartition, and AgentMerge",
                    );
                }
            }
            Algorithm::RadixSort => {
                for name in &names {
                    self.observe_radix_name(name);
                }
                if names.iter().any(|name| {
                    name.starts_with("device_merge_sort") || name == "merge_sort_source"
                }) {
                    self.reject(
                        path,
                        "DeviceRadixSort may not lower a radix digit to comparison merge sort; port the selected CUB radix agent graph",
                    );
                }
            }
        }

        visit::visit_path(self, path);
    }

    fn visit_expr_method_call(&mut self, expression: &'ast ExprMethodCall) {
        if expression.method == "compute_capability" {
            self.reject(
                expression,
                "generated CUB operations may not perform runtime TMP or architecture selection; the proc macro must emit the selected policy",
            );
        }
        let method = expression.method.to_string();
        if self.algorithm == Algorithm::Merge {
            self.saw_merge_partition |= method == "device_partition_merge_path";
            self.saw_merge_agent |= matches!(
                method.as_str(),
                "device_merge_keys_agent" | "device_merge_pairs_agent"
            );
            if matches!(method.as_str(), "device_merge_keys" | "device_merge_pairs") {
                self.reject(
                    expression,
                    "DeviceMerge may not use the deleted global per-thread merge kernel",
                );
            }
        }
        match self.algorithm {
            Algorithm::Merge if method == "stable_left_partition" => self.reject(
                expression,
                "DeviceMerge may not substitute one global MergePath search per output thread; port the partition kernel and block merge agent",
            ),
            Algorithm::MergeSort => {
                self.observe_merge_sort_name(&method);
                if matches!(
                    method.as_str(),
                    "merge_sort_source"
                        | "device_merge_sort_keys_pass"
                        | "device_merge_sort_pairs_pass"
                ) {
                    self.reject(
                        expression,
                        "DeviceMergeSort may not substitute one-output-per-thread pairwise passes; port AgentBlockSort, AgentPartition, and AgentMerge",
                    );
                }
            }
            Algorithm::RadixSort
                if method.starts_with("device_merge_sort") || method == "merge_sort_source" =>
            {
                self.reject(
                    expression,
                    "DeviceRadixSort may not lower a radix digit to comparison merge sort; port the selected CUB radix agent graph",
                );
            }
            Algorithm::RadixSort => self.observe_radix_name(&method),
            _ => {}
        }
        visit::visit_expr_method_call(self, expression);
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
    fn only_the_device_abi_arity_lint_may_be_allowed() {
        generated_rust(quote! {
            #[allow(clippy::too_many_arguments)]
            fn device_entry(a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32) {
                let _ = (a, b, c, d, e, f, g, h);
            }
        })
        .expect("the explicit CUDA entry ABI exception must pass");

        let error = generated_rust(quote! {
            #[allow(clippy::all)]
            fn hidden_warning() {}
        })
        .expect_err("broad lint suppression must remain rejected");
        assert!(
            error
                .into_compile_error()
                .to_string()
                .contains("audited built-in attributes")
        );
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

    #[test]
    fn runtime_policy_and_operation_owned_workspace_are_rejected() {
        let error = generated_algorithm_expression(
            Algorithm::Merge,
            quote! {{
                let target = execution.stream().context().compute_capability()?;
                let scratch = cuda_core::DeviceBuffer::zeroed(execution.stream(), count)?;
                (target, scratch)
            }},
        )
        .expect_err("runtime TMP and hidden workspace allocation must fail closed");
        let message = error.into_compile_error().to_string();
        assert!(message.contains("runtime TMP"));
        assert!(message.contains("temporary storage"));
    }

    #[test]
    fn generated_cuda_artifact_and_jit_paths_are_rejected() {
        for generated in [
            quote! {{ nvrtc::compile_ptx(source) }},
            quote! {{ cuda_bindings::cuModuleLoadData(image) }},
            quote! {{ loader.load_cubin("kernel.cubin") }},
            quote! {{ invoke("ptxas", "kernel.ptx") }},
            quote! {{ compile(".version 8.8\n.target sm_89") }},
        ] {
            let error = generated_expression(generated).expect_err(
                "generated RCCL code must not own CUDA artifact compilation or loading",
            );
            let message = error.into_compile_error().to_string();
            assert!(
                message.contains("pure Rust for cuda-oxide"),
                "unexpected rejection: {message}"
            );
        }
    }

    #[test]
    fn comparison_sort_cannot_mask_a_radix_agent_gap() {
        let error = generated_algorithm_expression(
            Algorithm::RadixSort,
            quote! {{
                module.device_merge_sort_keys_pass(stream, input, output)
            }},
        )
        .expect_err("a comparison-sort radix substitution must fail closed");
        assert!(
            error
                .into_compile_error()
                .to_string()
                .contains("radix digit")
        );
    }

    #[test]
    fn global_per_thread_merge_partition_is_rejected() {
        let error = generated_algorithm_expression(
            Algorithm::Merge,
            quote! {{
                stable_left_partition(first, second, output_index)
            }},
        )
        .expect_err("the pre-agent merge substitution must fail closed");
        assert!(
            error
                .into_compile_error()
                .to_string()
                .contains("partition kernel")
        );
    }
}
