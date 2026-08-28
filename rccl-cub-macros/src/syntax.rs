pub(super) fn ungroup_type(value_type: &syn::Type) -> &syn::Type {
    match value_type {
        syn::Type::Group(group) => ungroup_type(&group.elem),
        syn::Type::Paren(paren) => ungroup_type(&paren.elem),
        other => other,
    }
}
