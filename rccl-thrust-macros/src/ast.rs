use proc_macro2::Span;
use std::collections::{BTreeMap, BTreeSet};
use syn::{
    Ident, Result, Token, braced, bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

mod keyword {
    syn::custom_keyword!(cub);
    syn::custom_keyword!(support);
    syn::custom_keyword!(adaptor);
    syn::custom_keyword!(algorithm);
    syn::custom_keyword!(select);
}

#[derive(Debug)]
pub(crate) struct ThrustProgram {
    pub(crate) cub: Vec<Ident>,
    pub(crate) supports: Vec<Support>,
    pub(crate) adaptors: Vec<Adaptor>,
    pub(crate) algorithms: Vec<Algorithm>,
}

#[derive(Debug)]
pub(crate) struct Support {
    pub(crate) name: Ident,
    pub(crate) rust: Ident,
    pub(crate) status: Ident,
}

#[derive(Debug)]
pub(crate) struct Adaptor {
    pub(crate) name: Ident,
    pub(crate) status: Ident,
    pub(crate) capabilities: Vec<Ident>,
}

#[derive(Debug)]
pub(crate) struct Algorithm {
    pub(crate) name: Ident,
    pub(crate) status: Ident,
    pub(crate) form: Ident,
    pub(crate) requires: Vec<Ident>,
    pub(crate) selections: Vec<Selection>,
}

#[derive(Debug)]
pub(crate) struct Selection {
    pub(crate) predicate: Ident,
    pub(crate) primitive: Ident,
}

impl Parse for ThrustProgram {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<keyword::cub>()?;
        let cub = parse_ident_list(input)?;
        input.parse::<Token![;]>()?;
        let mut supports = Vec::new();
        let mut adaptors = Vec::new();
        let mut algorithms = Vec::new();
        while !input.is_empty() {
            if input.peek(keyword::support) {
                supports.push(input.parse()?);
            } else if input.peek(keyword::adaptor) {
                adaptors.push(input.parse()?);
            } else if input.peek(keyword::algorithm) {
                algorithms.push(input.parse()?);
            } else {
                return Err(input.error("expected `support`, `adaptor`, or `algorithm`"));
            }
        }
        Ok(Self {
            cub,
            supports,
            adaptors,
            algorithms,
        })
    }
}

impl Parse for Support {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<keyword::support>()?;
        let name = input.parse()?;
        let fields = parse_ident_fields(input)?;
        Ok(Self {
            name,
            rust: take_field(&fields, "rust", input.span())?,
            status: take_field(&fields, "status", input.span())?,
        })
    }
}

impl Parse for Adaptor {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<keyword::adaptor>()?;
        let name = input.parse()?;
        let content;
        braced!(content in input);
        let mut status = None;
        let mut capabilities = None;
        while !content.is_empty() {
            let field: Ident = content.parse()?;
            content.parse::<Token![:]>()?;
            match field.to_string().as_str() {
                "status" => set_once(&mut status, content.parse::<Ident>()?, &field)?,
                "capabilities" => set_once(&mut capabilities, parse_ident_list(&content)?, &field)?,
                _ => return Err(syn::Error::new(field.span(), "unknown adaptor field")),
            }
            content.parse::<Token![;]>()?;
        }
        Ok(Self {
            name,
            status: required(status, "status", input.span())?,
            capabilities: required(capabilities, "capabilities", input.span())?,
        })
    }
}

impl Parse for Algorithm {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<keyword::algorithm>()?;
        let name = input.parse()?;
        let content;
        braced!(content in input);
        let mut status = None;
        let mut form = None;
        let mut requires = None;
        let mut selections = None;
        while !content.is_empty() {
            if content.peek(keyword::select) {
                content.parse::<keyword::select>()?;
                let selection_content;
                braced!(selection_content in content);
                let mut values = Vec::new();
                while !selection_content.is_empty() {
                    values.push(selection_content.parse()?);
                }
                if selections.replace(values).is_some() {
                    return Err(content.error("duplicate `select` block"));
                }
                continue;
            }
            let field: Ident = content.parse()?;
            content.parse::<Token![:]>()?;
            match field.to_string().as_str() {
                "status" => set_once(&mut status, content.parse::<Ident>()?, &field)?,
                "form" => set_once(&mut form, content.parse::<Ident>()?, &field)?,
                "requires" => set_once(&mut requires, parse_ident_list(&content)?, &field)?,
                _ => return Err(syn::Error::new(field.span(), "unknown algorithm field")),
            }
            content.parse::<Token![;]>()?;
        }
        Ok(Self {
            name,
            status: required(status, "status", input.span())?,
            form: required(form, "form", input.span())?,
            requires: required(requires, "requires", input.span())?,
            selections: required(selections, "select", input.span())?,
        })
    }
}

impl Parse for Selection {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let predicate = input.parse()?;
        input.parse::<Token![=>]>()?;
        let primitive = input.parse()?;
        input.parse::<Token![;]>()?;
        Ok(Self {
            predicate,
            primitive,
        })
    }
}

fn parse_ident_fields(input: ParseStream<'_>) -> Result<BTreeMap<String, Ident>> {
    let content;
    braced!(content in input);
    let mut fields = BTreeMap::new();
    while !content.is_empty() {
        let field: Ident = content.parse()?;
        content.parse::<Token![:]>()?;
        let value: Ident = content.parse()?;
        content.parse::<Token![;]>()?;
        let name = field.to_string();
        if fields.insert(name.clone(), value).is_some() {
            return Err(syn::Error::new(
                field.span(),
                format!("duplicate field `{name}`"),
            ));
        }
    }
    Ok(fields)
}

fn take_field(fields: &BTreeMap<String, Ident>, name: &str, span: Span) -> Result<Ident> {
    fields
        .get(name)
        .cloned()
        .ok_or_else(|| syn::Error::new(span, format!("missing `{name}` field")))
}

fn parse_ident_list(input: ParseStream<'_>) -> Result<Vec<Ident>> {
    let content;
    bracketed!(content in input);
    Ok(Punctuated::<Ident, Token![,]>::parse_terminated(&content)?
        .into_iter()
        .collect())
}

fn set_once<T>(slot: &mut Option<T>, value: T, field: &Ident) -> Result<()> {
    if slot.replace(value).is_some() {
        return Err(syn::Error::new(
            field.span(),
            format!("duplicate `{field}` field"),
        ));
    }
    Ok(())
}

fn required<T>(value: Option<T>, field: &str, span: Span) -> Result<T> {
    value.ok_or_else(|| syn::Error::new(span, format!("missing `{field}` field")))
}

impl ThrustProgram {
    pub(crate) fn validate(&self) -> Result<()> {
        let cub = unique_names("imported CUB primitive", self.cub.iter())?;
        if cub.is_empty() {
            return Err(syn::Error::new(
                Span::call_site(),
                "Thrust must import at least one generated CUB primitive",
            ));
        }
        unique_names("support", self.supports.iter().map(|item| &item.name))?;
        unique_names("adaptor", self.adaptors.iter().map(|item| &item.name))?;
        unique_names("algorithm", self.algorithms.iter().map(|item| &item.name))?;
        let allowed_status = ["planned", "implemented", "native_rust", "backend_blocked"];
        let allowed_form = [
            "input",
            "output",
            "in_place",
            "input_output",
            "reduction",
            "query",
            "allocation",
        ];
        let allowed_capability = [
            "input",
            "output",
            "random_access",
            "contiguous",
            "product",
            "permutation",
            "strided",
        ];

        for support in &self.supports {
            if !allowed_status.contains(&support.status.to_string().as_str()) {
                return Err(syn::Error::new(
                    support.status.span(),
                    "unknown support status",
                ));
            }
            if support.rust == "compatibility" {
                return Err(syn::Error::new(
                    support.rust.span(),
                    "support mappings must name a Rust semantic type, not a compatibility layer",
                ));
            }
        }
        for adaptor in &self.adaptors {
            if !allowed_status.contains(&adaptor.status.to_string().as_str()) {
                return Err(syn::Error::new(
                    adaptor.status.span(),
                    "unknown adaptor status",
                ));
            }
            let capabilities = unique_names("capability", adaptor.capabilities.iter())?;
            if capabilities.is_empty() {
                return Err(syn::Error::new(
                    adaptor.name.span(),
                    "adaptor must declare at least one capability",
                ));
            }
            for capability in capabilities {
                if !allowed_capability.contains(&capability.as_str()) {
                    return Err(syn::Error::new(
                        adaptor.name.span(),
                        format!("unknown adaptor capability `{capability}`"),
                    ));
                }
            }
        }
        for algorithm in &self.algorithms {
            if !allowed_status.contains(&algorithm.status.to_string().as_str()) {
                return Err(syn::Error::new(
                    algorithm.status.span(),
                    "unknown algorithm status",
                ));
            }
            if !allowed_form.contains(&algorithm.form.to_string().as_str()) {
                return Err(syn::Error::new(
                    algorithm.form.span(),
                    "unknown algorithm form",
                ));
            }
            let dependencies = unique_names("CUB dependency", algorithm.requires.iter())?;
            if dependencies.is_empty() {
                return Err(syn::Error::new(
                    algorithm.name.span(),
                    "algorithm must lower through at least one CUB primitive",
                ));
            }
            for dependency in &dependencies {
                if !cub.contains(dependency) {
                    return Err(syn::Error::new(
                        algorithm.name.span(),
                        format!(
                            "CUB dependency `{dependency}` is not imported by this Thrust program"
                        ),
                    ));
                }
            }
            if algorithm.selections.is_empty() {
                return Err(syn::Error::new(
                    algorithm.name.span(),
                    "algorithm must declare compile-time selection",
                ));
            }
            let predicates = unique_names(
                "selection predicate",
                algorithm
                    .selections
                    .iter()
                    .map(|selection| &selection.predicate),
            )?;
            if !predicates.contains("otherwise") {
                return Err(syn::Error::new(
                    algorithm.name.span(),
                    "selection must contain an `otherwise` rule",
                ));
            }
            for selection in &algorithm.selections {
                if !dependencies.contains(&selection.primitive.to_string()) {
                    return Err(syn::Error::new(
                        selection.primitive.span(),
                        format!(
                            "selection target `{}` is not listed in `requires`",
                            selection.primitive
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn unique_names<'a>(
    kind: &str,
    names: impl IntoIterator<Item = &'a Ident>,
) -> Result<BTreeSet<String>> {
    let mut unique = BTreeSet::new();
    for name in names {
        if !unique.insert(name.to_string()) {
            return Err(syn::Error::new(
                name.span(),
                format!("duplicate {kind} `{name}`"),
            ));
        }
    }
    Ok(unique)
}
