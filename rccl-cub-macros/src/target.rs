use crate::syntax::ungroup_type;

pub(super) fn tmp_primitive_size(value_type: &syn::Type) -> syn::Result<usize> {
    let semantic_type = ungroup_type(value_type);
    let syn::Type::Path(path) = semantic_type else {
        return Err(syn::Error::new_spanned(
            value_type,
            "this CUB schedule needs macro-visible type layout metadata",
        ));
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return Err(syn::Error::new_spanned(
            value_type,
            "aliases and custom values need explicit macro-visible size/alignment metadata",
        ));
    }
    match path.path.segments[0].ident.to_string().as_str() {
        "bool" | "u8" | "i8" => Ok(1),
        "u16" | "i16" => Ok(2),
        "u32" | "i32" | "f32" => Ok(4),
        "u64" | "i64" | "usize" | "isize" | "f64" => Ok(8),
        "u128" | "i128" => Ok(16),
        _ => Err(syn::Error::new_spanned(
            value_type,
            "aliases and custom values need explicit macro-visible size/alignment metadata",
        )),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CudaTarget {
    pub(super) capability: u16,
}

/// Parse one CUDA architecture used internally by CUB policy selection.
pub(super) fn tmp_cuda_target(target: &syn::Type) -> syn::Result<CudaTarget> {
    let syn::Type::Path(path) = ungroup_type(target) else {
        return Err(syn::Error::new_spanned(
            target,
            "the CUB target must be a macro-only architecture such as `sm_89`",
        ));
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return Err(syn::Error::new_spanned(
            target,
            "the CUB target must be a single architecture token such as `sm_89`",
        ));
    }
    let spelling = path.path.segments[0].ident.to_string();
    let Some(numeric_with_suffix) = spelling.strip_prefix("sm_") else {
        return Err(syn::Error::new_spanned(
            target,
            "the CUB target must use CUDA's `sm_XX` spelling",
        ));
    };
    let numeric = numeric_with_suffix.trim_end_matches(['a', 'f']);
    if numeric.len() < 2
        || numeric.len() > 3
        || numeric_with_suffix[numeric.len()..]
            .chars()
            .any(|suffix| !matches!(suffix, 'a' | 'f'))
    {
        return Err(syn::Error::new_spanned(
            target,
            "invalid CUDA architecture; expected `sm_XX`, `sm_XXX`, or an `a`/`f` suffix",
        ));
    }
    let capability = numeric
        .parse::<u16>()
        .map_err(|_| syn::Error::new_spanned(target, "CUDA architecture digits must be decimal"))?;
    if capability < 52 {
        return Err(syn::Error::new_spanned(
            target,
            "the pinned CCCL ordering policies require sm_52 or newer",
        ));
    }
    Ok(CudaTarget { capability })
}

/// Read the same target selected by cargo-oxide. Public CUB and Thrust calls
/// never carry architecture tokens; they are compiler inputs, not semantic
/// algorithm operands. Plain host-only Cargo checks use SM89 as the test
/// baseline because cargo-oxide is not driving that build.
pub(super) fn configured_cuda_target() -> syn::Result<CudaTarget> {
    #[cfg(not(test))]
    let spelling =
        proc_macro::tracked::env_var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| "sm_89".to_owned());
    #[cfg(test)]
    let spelling = std::env::var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| "sm_89".to_owned());
    let spelling = if spelling.starts_with("sm_") {
        spelling
    } else if let Some(numeric) = spelling.strip_prefix("compute_") {
        format!("sm_{numeric}")
    } else {
        format!("sm_{spelling}")
    };
    let target = syn::parse_str::<syn::Type>(&spelling).map_err(|_| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!("invalid CUDA_OXIDE_TARGET `{spelling}`"),
        )
    })?;
    tmp_cuda_target(&target)
}
