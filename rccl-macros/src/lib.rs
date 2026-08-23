use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Ident, LitInt, Path, Result, Token, Visibility, braced,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

struct PolicyEntry {
    name: Ident,
    min_sm: LitInt,
    threads: LitInt,
    items: LitInt,
    radix_bits: LitInt,
    min_blocks: LitInt,
    unroll: LitInt,
}

impl Parse for PolicyEntry {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![=>]>()?;
        let content;
        braced!(content in input);

        let mut min_sm = None;
        let mut threads = None;
        let mut items = None;
        let mut radix_bits = None;
        let mut min_blocks = None;
        let mut unroll = None;

        while !content.is_empty() {
            let field: Ident = content.parse()?;
            content.parse::<Token![:]>()?;
            let value: LitInt = content.parse()?;
            match field.to_string().as_str() {
                "min_sm" => min_sm = Some(value),
                "threads" => threads = Some(value),
                "items" => items = Some(value),
                "radix_bits" => radix_bits = Some(value),
                "min_blocks" => min_blocks = Some(value),
                "unroll" => unroll = Some(value),
                _ => return Err(syn::Error::new(field.span(), "unknown radix policy field")),
            }
            if !content.is_empty() {
                content.parse::<Token![,]>()?;
            }
        }

        let name_span = name.span();
        let missing = |field| syn::Error::new(name_span, format!("missing `{field}`"));
        Ok(Self {
            name,
            min_sm: min_sm.ok_or_else(|| missing("min_sm"))?,
            threads: threads.ok_or_else(|| missing("threads"))?,
            items: items.ok_or_else(|| missing("items"))?,
            radix_bits: radix_bits.ok_or_else(|| missing("radix_bits"))?,
            min_blocks: min_blocks.ok_or_else(|| missing("min_blocks"))?,
            unroll: unroll.ok_or_else(|| missing("unroll"))?,
        })
    }
}

struct PolicyChain {
    vis: Visibility,
    selector: Ident,
    policy_trait: Path,
    namespace: LitInt,
    entries: Vec<PolicyEntry>,
}

impl Parse for PolicyChain {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let vis: Visibility = input.parse()?;
        input.parse::<Token![enum]>()?;
        let selector: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let policy_trait: Path = input.parse()?;
        let content;
        braced!(content in input);

        let namespace_name: Ident = content.parse()?;
        if namespace_name != "namespace" {
            return Err(syn::Error::new(
                namespace_name.span(),
                "expected `namespace`",
            ));
        }
        content.parse::<Token![:]>()?;
        let namespace: LitInt = content.parse()?;
        content.parse::<Token![;]>()?;

        let mut entries: Vec<PolicyEntry> = Vec::new();
        while !content.is_empty() {
            entries.push(content.parse()?);
            if !content.is_empty() {
                content.parse::<Token![,]>()?;
            }
        }
        if entries.is_empty() {
            return Err(syn::Error::new(
                selector.span(),
                "at least one policy is required",
            ));
        }

        let mut previous = None;
        for entry in &entries {
            let sm = entry.min_sm.base10_parse::<u32>()?;
            if let Some(prev) = previous {
                if sm <= prev {
                    return Err(syn::Error::new(
                        entry.min_sm.span(),
                        "policies must be ordered by strictly increasing min_sm",
                    ));
                }
            }
            let threads = entry.threads.base10_parse::<u32>()?;
            if threads < 128 || threads > 1024 || threads % 32 != 0 {
                return Err(syn::Error::new(
                    entry.threads.span(),
                    "threads must be a warp multiple in 128..=1024",
                ));
            }
            let bits = entry.radix_bits.base10_parse::<u32>()?;
            if !(4..=8).contains(&bits) {
                return Err(syn::Error::new(
                    entry.radix_bits.span(),
                    "radix_bits must be in 4..=8",
                ));
            }
            let items = entry.items.base10_parse::<u32>()?;
            if items == 0 || items > 32 {
                return Err(syn::Error::new(
                    entry.items.span(),
                    "items must be in 1..=32",
                ));
            }
            previous = Some(sm);
        }

        Ok(Self {
            vis,
            selector,
            policy_trait,
            namespace,
            entries,
        })
    }
}

/// Generate cuda-oxide policy types plus an ordered SM dispatch chain.
///
/// The policy types are zero-sized compile-time parameters. Passing one to a
/// generic `#[kernel]` makes cuda-oxide emit a distinct PTX specialization;
/// none of these fields becomes a runtime kernel argument.
#[proc_macro]
pub fn radix_policy_chain(input: TokenStream) -> TokenStream {
    let chain = parse_macro_input!(input as PolicyChain);
    let PolicyChain {
        vis,
        selector,
        policy_trait,
        namespace,
        entries,
    } = chain;

    let names: Vec<_> = entries.iter().map(|entry| &entry.name).collect();
    let min_sms: Vec<_> = entries.iter().map(|entry| &entry.min_sm).collect();
    let threads: Vec<_> = entries.iter().map(|entry| &entry.threads).collect();
    let items: Vec<_> = entries.iter().map(|entry| &entry.items).collect();
    let radix_bits: Vec<_> = entries.iter().map(|entry| &entry.radix_bits).collect();
    let min_blocks: Vec<_> = entries.iter().map(|entry| &entry.min_blocks).collect();
    let unroll: Vec<_> = entries.iter().map(|entry| &entry.unroll).collect();
    let ids: Vec<u64> = (1..=entries.len() as u64).collect();

    let reversed: Vec<_> = names.iter().zip(min_sms.iter()).rev().collect();
    let select_arms = reversed.iter().map(|(name, min_sm)| {
        quote! {
            if sm >= #min_sm {
                return Some(Self::#name);
            }
        }
    });

    let expanded = quote! {
        #(
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
            #vis struct #names;

            impl cuda_device::config::Policy for #names {
                const ID: cuda_device::config::PolicyId =
                    cuda_device::config::PolicyId::new(#namespace, #ids);
            }

            impl #policy_trait for #names {
                const MIN_SM: u32 = #min_sms;
                const BLOCK_THREADS: u32 = #threads;
                const ITEMS_PER_THREAD: u32 = #items;
                const RADIX_BITS: u32 = #radix_bits;
                const MIN_BLOCKS_PER_SM: u32 = #min_blocks;
                const UNROLL: u32 = #unroll;
            }
        )*

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #vis enum #selector {
            #(#names),*
        }

        impl #selector {
            #vis const fn select(sm: u32) -> Option<Self> {
                #(#select_arms)*
                None
            }

            #vis const fn name(self) -> &'static str {
                match self {
                    #(Self::#names => stringify!(#names)),*
                }
            }

            #vis const fn config(self) -> crate::PolicyConfig {
                match self {
                    #(
                        Self::#names => crate::PolicyConfig {
                            min_sm: #min_sms,
                            block_threads: #threads,
                            items_per_thread: #items,
                            radix_bits: #radix_bits,
                            min_blocks_per_sm: #min_blocks,
                            unroll: #unroll,
                        }
                    ),*
                }
            }
        }
    };
    expanded.into()
}
