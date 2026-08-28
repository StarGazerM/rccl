#![cfg_attr(not(test), feature(proc_macro_tracked_env))]

mod audit;
mod call;
#[cfg(test)]
mod catalog;

use proc_macro::TokenStream;
use quote::{format_ident, quote};

/// Declare a crate-root cuda-oxide program containing user kernels that call
/// CUB block/warp primitives.
///
/// The attribute owns the exact artifact-provenance bind, so the caller uses
/// `module_name::program!(context)` without spelling an unsafe module load.
#[proc_macro_attribute]
pub fn program(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[rccl_cub::program] does not take arguments",
        )
        .into_compile_error()
        .into();
    }
    let module = syn::parse_macro_input!(item as syn::ItemMod);
    expand_user_program(module)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_user_program(mut module: syn::ItemMod) -> syn::Result<proc_macro2::TokenStream> {
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

    // Audit the semantic unsafe boundary before emitting it inside the scoped
    // macro below. The actual `$crate` spelling makes the path immune to a
    // caller-side module shadow while keeping every type inferred.
    audit::generated_program_expression(
        quote! {{ unsafe { crate::#ident::load(__rccl_context) } }},
        &ident,
    )?;

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

/// Construct a typed owner for the CUB device-wide kernel family.
///
/// The generated owner keeps cuda-oxide artifact loading out of the public CUB
/// call shape. RCCL emits Rust only; cuda-oxide still owns compilation,
/// embedding, typed binding, and launch validation.
#[proc_macro]
pub fn cub_program(input: TokenStream) -> TokenStream {
    let context = syn::parse_macro_input!(input as syn::Expr);
    expand_cub_program(context)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_cub_program(context: syn::Expr) -> syn::Result<proc_macro2::TokenStream> {
    let module = syn::parse_quote!(__rccl_cub_kernels);
    audit::generated_program_expression(
        quote! {{
            ::rccl_cub::__private::__device_kernels!(__rccl_cub_kernels);
            let __rccl_context = #context;
            unsafe { __rccl_cub_kernels::load(__rccl_context) }.map(|__rccl_module| {
                ::rccl_cub::cuda::Program::__from_module(
                    __rccl_context.as_ref(),
                    __rccl_module,
                )
            })
        }},
        &module,
    )
}

/// Internal item lowering used by the Thrust macro layer.
#[doc(hidden)]
#[proc_macro]
pub fn __device_kernels(input: TokenStream) -> TokenStream {
    let module = syn::parse_macro_input!(input as syn::Ident);
    expand_device_kernels(module)
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

macro_rules! operation_macro {
    ($macro_name:ident, $call:ty, $expand:ident) => {
        #[proc_macro]
        pub fn $macro_name(input: TokenStream) -> TokenStream {
            let call = syn::parse_macro_input!(input as $call);
            $expand(call)
                .unwrap_or_else(syn::Error::into_compile_error)
                .into()
        }
    };
}

operation_macro!(
    device_merge_keys,
    call::DeviceMergeKeysCall,
    expand_device_merge_keys
);
operation_macro!(
    __device_merge_keys_partition_count,
    call::DeviceMergeKeysPartitionCountCall,
    expand_device_merge_keys_partition_count
);
operation_macro!(
    device_merge_keys_workspace,
    call::DeviceMergeKeysWorkspaceCall,
    expand_device_merge_keys_workspace
);
operation_macro!(
    __device_merge_pairs_partition_count,
    call::DeviceMergePairsPartitionCountCall,
    expand_device_merge_pairs_partition_count
);
operation_macro!(
    device_merge_pairs_workspace,
    call::DeviceMergePairsWorkspaceCall,
    expand_device_merge_pairs_workspace
);
operation_macro!(
    device_merge_pairs,
    call::DeviceMergePairsCall,
    expand_device_merge_pairs
);
operation_macro!(
    device_merge_sort_keys,
    call::DeviceMergeSortKeysCall,
    expand_device_merge_sort_keys
);
operation_macro!(
    __device_merge_sort_partition_count,
    call::DeviceMergeSortPartitionCountCall,
    expand_device_merge_sort_partition_count
);
operation_macro!(
    device_merge_sort_keys_workspace,
    call::DeviceMergeSortKeysWorkspaceCall,
    expand_device_merge_sort_keys_workspace
);
operation_macro!(
    device_merge_sort_pairs_workspace,
    call::DeviceMergeSortPairsWorkspaceCall,
    expand_device_merge_sort_pairs_workspace
);
operation_macro!(
    device_stable_merge_sort_keys,
    call::DeviceMergeSortKeysCall,
    expand_device_merge_sort_keys
);
operation_macro!(
    device_merge_sort_keys_copy,
    call::DeviceMergeSortKeysCopyCall,
    expand_device_merge_sort_keys_copy
);
operation_macro!(
    device_stable_merge_sort_keys_copy,
    call::DeviceMergeSortKeysCopyCall,
    expand_device_merge_sort_keys_copy
);
operation_macro!(
    device_merge_sort_pairs,
    call::DeviceMergeSortPairsCall,
    expand_device_merge_sort_pairs
);
operation_macro!(
    device_stable_merge_sort_pairs,
    call::DeviceMergeSortPairsCall,
    expand_device_merge_sort_pairs
);
operation_macro!(
    device_merge_sort_pairs_copy,
    call::DeviceMergeSortPairsCopyCall,
    expand_device_merge_sort_pairs_copy
);
operation_macro!(
    device_radix_sort_keys,
    call::DeviceRadixSortKeysCall,
    expand_device_radix_sort_keys
);
operation_macro!(
    device_radix_sort_keys_buffer,
    call::DeviceRadixSortKeysBufferCall,
    expand_device_radix_sort_keys_buffer
);
operation_macro!(
    device_radix_sort_keys_buffer_descending,
    call::DeviceRadixSortKeysBufferCall,
    expand_device_radix_sort_keys_buffer_descending
);
operation_macro!(
    device_radix_sort_workspace,
    call::DeviceRadixSortWorkspaceCall,
    expand_device_radix_sort_workspace
);
operation_macro!(
    __device_radix_sort_workspace_counts,
    call::DeviceRadixSortWorkspaceCountsCall,
    expand_device_radix_sort_workspace_counts
);
operation_macro!(
    device_radix_sort_keys_buffer_workspace,
    call::DeviceRadixSortKeysBufferWorkspaceCall,
    expand_device_radix_sort_keys_buffer_workspace
);
operation_macro!(
    device_radix_sort_pairs_workspace,
    call::DeviceRadixSortPairsWorkspaceCall,
    expand_device_radix_sort_pairs_workspace
);
operation_macro!(
    __device_radix_sort_pairs_workspace_counts,
    call::DeviceRadixSortPairsWorkspaceCountsCall,
    expand_device_radix_sort_pairs_workspace_counts
);
operation_macro!(
    device_radix_sort_pairs_buffer_workspace,
    call::DeviceRadixSortPairsBufferWorkspaceCall,
    expand_device_radix_sort_pairs_buffer_workspace
);
operation_macro!(
    device_radix_sort_keys_descending,
    call::DeviceRadixSortKeysCall,
    expand_device_radix_sort_keys_descending
);
operation_macro!(
    device_radix_sort_pairs,
    call::DeviceRadixSortPairsCall,
    expand_device_radix_sort_pairs
);
operation_macro!(
    device_radix_sort_pairs_buffer,
    call::DeviceRadixSortPairsBufferCall,
    expand_device_radix_sort_pairs_buffer
);
operation_macro!(
    device_radix_sort_pairs_buffer_descending,
    call::DeviceRadixSortPairsBufferCall,
    expand_device_radix_sort_pairs_buffer_descending
);
operation_macro!(
    device_radix_sort_pairs_descending,
    call::DeviceRadixSortPairsCall,
    expand_device_radix_sort_pairs_descending
);

#[derive(Clone, Copy)]
enum PrimitiveRadixKind {
    Unsigned,
    Signed,
    Floating,
}

#[derive(Clone, Copy)]
struct PrimitiveRadix {
    bits: u32,
    kind: PrimitiveRadixKind,
}

struct PrimitiveRadixLowering {
    bit_type: syn::Type,
    sentinel: proc_macro2::TokenStream,
    default_key: proc_macro2::TokenStream,
    to_ordered: proc_macro2::TokenStream,
    from_ordered: proc_macro2::TokenStream,
    digit: proc_macro2::TokenStream,
    rank_digit: proc_macro2::TokenStream,
}

struct PrimitiveSingleTileLowering {
    default_ordered: proc_macro2::TokenStream,
    to_ordered: proc_macro2::TokenStream,
    from_ordered: proc_macro2::TokenStream,
}

/// Pinned CUB `detail::RegBoundScaling` evaluated by the proc macro.
///
/// The result is TMP policy state in C++; it must not be reconstructed by the
/// running Rust program or selected from a runtime device property.
fn reg_bound_policy(
    nominal_threads: usize,
    nominal_items: usize,
    type_size: usize,
) -> (usize, usize) {
    let items = (nominal_items * 4 / type_size.max(4)).max(1);
    let static_shared_words = (48 * 1024) / (type_size * items);
    let shared_limited_threads = static_shared_words.div_ceil(32) * 32;
    (nominal_threads.min(shared_limited_threads), items)
}

fn primitive_radix_lowering(
    key_type: &syn::Type,
    primitive: PrimitiveRadix,
    descending: bool,
) -> syn::Result<PrimitiveRadixLowering> {
    match (primitive.bits, primitive.kind) {
        (32, PrimitiveRadixKind::Unsigned) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_u32_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_u32_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_u32_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_u32_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u32),
                sentinel: quote!(u32::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u32),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_digit_masked_u32),
                rank_digit: quote!(::rccl_cub::__private::radix_rank_digit_u32),
            })
        }
        (32, PrimitiveRadixKind::Signed) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_i32_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_i32_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_i32_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_i32_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u32),
                sentinel: quote!(u32::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u32),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_digit_masked_u32),
                rank_digit: quote!(::rccl_cub::__private::radix_rank_digit_u32),
            })
        }
        (32, PrimitiveRadixKind::Floating) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_f32_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_f32_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_f32_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_f32_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u32),
                sentinel: quote!(u32::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u32),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_float_digit_masked_u32),
                rank_digit: quote!(::rccl_cub::__private::radix_float_rank_digit_u32),
            })
        }
        (64, PrimitiveRadixKind::Unsigned) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_u64_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_u64_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_u64_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_u64_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u64),
                sentinel: quote!(u64::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u64),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_digit_masked_u64),
                rank_digit: quote!(::rccl_cub::__private::radix_rank_digit_u64),
            })
        }
        (64, PrimitiveRadixKind::Signed) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_i64_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_i64_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_i64_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_i64_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u64),
                sentinel: quote!(u64::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u64),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_digit_masked_u64),
                rank_digit: quote!(::rccl_cub::__private::radix_rank_digit_u64),
            })
        }
        (64, PrimitiveRadixKind::Floating) => {
            let (to_ordered, from_ordered) = if descending {
                (
                    quote!(::rccl_cub::__private::radix_f64_to_ordered_descending),
                    quote!(::rccl_cub::__private::radix_f64_from_ordered_descending),
                )
            } else {
                (
                    quote!(::rccl_cub::__private::radix_f64_to_ordered_ascending),
                    quote!(::rccl_cub::__private::radix_f64_from_ordered_ascending),
                )
            };
            Ok(PrimitiveRadixLowering {
                bit_type: syn::parse_quote!(u64),
                sentinel: quote!(u64::MAX),
                default_key: quote!(::rccl_cub::__private::radix_default_u64),
                to_ordered,
                from_ordered,
                digit: quote!(::rccl_cub::__private::radix_float_digit_masked_u64),
                rank_digit: quote!(::rccl_cub::__private::radix_float_rank_digit_u64),
            })
        }
        _ => Err(syn::Error::new_spanned(
            key_type,
            "the exact pinned SM80 one-sweep path requires a 32-bit or 64-bit primitive key; smaller keys use CUB's different multi-pass policy",
        )),
    }
}

/// CUB's single-tile kernel uses ascending bit-order conversion in both
/// directions and reverses the stable rank bins for descending order. Its
/// guarded-load key is the primitive trait's raw minimum/maximum value.
fn primitive_single_tile_lowering(
    primitive: PrimitiveRadix,
    descending: bool,
) -> syn::Result<PrimitiveSingleTileLowering> {
    let lowering = match (primitive.bits, primitive.kind, descending) {
        (32, PrimitiveRadixKind::Unsigned, false) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_default_u32),
            to_ordered: quote!(::rccl_cub::__private::radix_u32_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_u32_from_ordered_ascending),
        },
        (32, PrimitiveRadixKind::Unsigned, true) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_min_ordered_u32),
            to_ordered: quote!(::rccl_cub::__private::radix_u32_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_u32_from_ordered_ascending),
        },
        (32, PrimitiveRadixKind::Signed, false) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_default_u32),
            to_ordered: quote!(::rccl_cub::__private::radix_i32_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_i32_from_ordered_ascending),
        },
        (32, PrimitiveRadixKind::Signed, true) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_min_ordered_u32),
            to_ordered: quote!(::rccl_cub::__private::radix_i32_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_i32_from_ordered_ascending),
        },
        (32, PrimitiveRadixKind::Floating, false) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_default_u32),
            to_ordered: quote!(::rccl_cub::__private::radix_f32_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_f32_from_ordered_ascending),
        },
        (32, PrimitiveRadixKind::Floating, true) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_min_ordered_u32),
            to_ordered: quote!(::rccl_cub::__private::radix_f32_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_f32_from_ordered_ascending),
        },
        (64, PrimitiveRadixKind::Unsigned, false) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_default_u64),
            to_ordered: quote!(::rccl_cub::__private::radix_u64_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_u64_from_ordered_ascending),
        },
        (64, PrimitiveRadixKind::Unsigned, true) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_min_ordered_u64),
            to_ordered: quote!(::rccl_cub::__private::radix_u64_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_u64_from_ordered_ascending),
        },
        (64, PrimitiveRadixKind::Signed, false) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_default_u64),
            to_ordered: quote!(::rccl_cub::__private::radix_i64_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_i64_from_ordered_ascending),
        },
        (64, PrimitiveRadixKind::Signed, true) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_min_ordered_u64),
            to_ordered: quote!(::rccl_cub::__private::radix_i64_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_i64_from_ordered_ascending),
        },
        (64, PrimitiveRadixKind::Floating, false) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_default_u64),
            to_ordered: quote!(::rccl_cub::__private::radix_f64_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_f64_from_ordered_ascending),
        },
        (64, PrimitiveRadixKind::Floating, true) => PrimitiveSingleTileLowering {
            default_ordered: quote!(::rccl_cub::__private::radix_min_ordered_u64),
            to_ordered: quote!(::rccl_cub::__private::radix_f64_to_ordered_ascending),
            from_ordered: quote!(::rccl_cub::__private::radix_f64_from_ordered_ascending),
        },
        _ => {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "unsupported primitive single-tile radix lowering",
            ));
        }
    };
    Ok(lowering)
}

/// Re-express CUB's second digit extraction as a distinct Rust operation.
///
fn primitive_radix(key_type: &syn::Type) -> syn::Result<PrimitiveRadix> {
    let syn::Type::Path(path) = key_type else {
        return Err(syn::Error::new_spanned(
            key_type,
            "DeviceRadixSort requires a macro-visible primitive numeric type; aliases need explicit radix metadata",
        ));
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return Err(syn::Error::new_spanned(
            key_type,
            "DeviceRadixSort cannot resolve a qualified or aliased key type before rustc type inference",
        ));
    }
    let name = path.path.segments[0].ident.to_string();
    let (bits, kind) = match name.as_str() {
        "u8" => (8, PrimitiveRadixKind::Unsigned),
        "u16" => (16, PrimitiveRadixKind::Unsigned),
        "u32" => (32, PrimitiveRadixKind::Unsigned),
        "u64" => (64, PrimitiveRadixKind::Unsigned),
        "i8" => (8, PrimitiveRadixKind::Signed),
        "i16" => (16, PrimitiveRadixKind::Signed),
        "i32" => (32, PrimitiveRadixKind::Signed),
        "i64" => (64, PrimitiveRadixKind::Signed),
        "f32" => (32, PrimitiveRadixKind::Floating),
        "f64" => (64, PrimitiveRadixKind::Floating),
        _ => {
            return Err(syn::Error::new_spanned(
                key_type,
                "custom and aliased radix keys need macro-visible decomposer metadata",
            ));
        }
    };
    Ok(PrimitiveRadix { bits, kind })
}

fn radix_onesweep_keys_kernel(shared_bytes: usize) -> proc_macro2::TokenStream {
    let kernel = format_ident!("device_radix_sort_onesweep_keys_{shared_bytes}");
    let storage = format_ident!("__RCCL_RADIX_ONESWEEP_KEYS_{shared_bytes}");
    let union_bytes = shared_bytes - 256 * core::mem::size_of::<u64>();
    quote! {
        #[kernel]
        #[launch_bounds(384)]
        #[launch_contract(domain = 1, coordinates = u32, block = (384, 1, 1))]
        pub fn #kernel<
            K,
            B,
            DefaultKey,
            ToOrdered,
            FromOrdered,
            Digit,
            const ITEMS: usize,
        >(
            lookback: DisjointSlice<i32>,
            counter: DisjointSlice<i32>,
            bins: &[u64],
            input: &[K],
            output: DisjointSlice<K>,
            bin_offset: Uniform<usize>,
            counter_offset: Uniform<usize>,
            current_bit: Uniform<u32>,
            num_bits: Uniform<u32>,
            default_key: DefaultKey,
            to_ordered: ToOrdered,
            from_ordered: FromOrdered,
            digit: Digit,
        ) where
            K: Copy,
            B: Copy,
            DefaultKey: Fn() -> B + Copy,
            ToOrdered: Fn(K) -> B + Copy,
            FromOrdered: Fn(B) -> K + Copy,
            Digit: Fn(B, u32, u32) -> u32 + Copy,
        {
            let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                #storage,
                #shared_bytes,
                16usize
            );
            ::rccl_cub::__private::radix_onesweep_keys_agent::<
                K,
                B,
                DefaultKey,
                ToOrdered,
                FromOrdered,
                Digit,
                384,
                ITEMS,
                #union_bytes,
            >(
                (lookback, counter, bins),
                (input, output),
                (
                    bin_offset.get(),
                    counter_offset.get(),
                    current_bit.get(),
                    num_bits.get(),
                ),
                (default_key, to_ordered, from_ordered, digit),
                shared_memory,
            );
        }
    }
}

fn radix_onesweep_pairs_kernel(shared_bytes: usize) -> proc_macro2::TokenStream {
    let kernel = format_ident!("device_radix_sort_onesweep_pairs_{shared_bytes}");
    let storage = format_ident!("__RCCL_RADIX_ONESWEEP_PAIRS_{shared_bytes}");
    let union_bytes = shared_bytes - 256 * core::mem::size_of::<u64>();
    quote! {
        #[kernel]
        #[launch_bounds(384)]
        #[launch_contract(domain = 1, coordinates = u32, block = (384, 1, 1))]
        pub fn #kernel<
            K,
            V,
            B,
            DefaultKey,
            ToOrdered,
            FromOrdered,
            Digit,
            const ITEMS: usize,
        >(
            lookback: DisjointSlice<i32>,
            counter: DisjointSlice<i32>,
            bins: &[u64],
            input_keys: &[K],
            input_values: &[V],
            output_keys: DisjointSlice<K>,
            output_values: DisjointSlice<V>,
            bin_offset: Uniform<usize>,
            counter_offset: Uniform<usize>,
            current_bit: Uniform<u32>,
            num_bits: Uniform<u32>,
            default_key: DefaultKey,
            to_ordered: ToOrdered,
            from_ordered: FromOrdered,
            digit: Digit,
        ) where
            K: Copy,
            V: Copy,
            B: Copy,
            DefaultKey: Fn() -> B + Copy,
            ToOrdered: Fn(K) -> B + Copy,
            FromOrdered: Fn(B) -> K + Copy,
            Digit: Fn(B, u32, u32) -> u32 + Copy,
        {
            let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                #storage,
                #shared_bytes,
                16usize
            );
            ::rccl_cub::__private::radix_onesweep_pairs_agent::<
                K,
                V,
                B,
                DefaultKey,
                ToOrdered,
                FromOrdered,
                Digit,
                384,
                ITEMS,
                #union_bytes,
            >(
                (lookback, counter, bins),
                (input_keys, input_values, output_keys, output_values),
                (
                    bin_offset.get(),
                    counter_offset.get(),
                    current_bit.get(),
                    num_bits.get(),
                ),
                (default_key, to_ordered, from_ordered, digit),
                shared_memory,
            );
        }
    }
}

fn expand_device_kernels(module: syn::Ident) -> syn::Result<proc_macro2::TokenStream> {
    let radix_onesweep_keys = [34_304usize, 32_768usize]
        .into_iter()
        .map(radix_onesweep_keys_kernel);
    let radix_onesweep_pairs = [28_160usize, 26_624usize, 32_768usize]
        .into_iter()
        .map(radix_onesweep_pairs_kernel);
    audit::generated_rust(quote! {
        #[doc(hidden)]
        use ::rccl_cub::__private::cuda_core;
        #[doc(hidden)]
        use ::rccl_cub::__private::cuda_host;

        #[::rccl_cub::__private::cuda_device::cuda_module]
        #[doc(hidden)]
        // Device entry points preserve the pinned CUB agent ABI. Grouping
        // unrelated launch operands into a host-style options struct would
        // change that ABI and obscure which values are uniform kernel inputs.
        #[allow(clippy::too_many_arguments)]
        mod #module {
            use ::rccl_cub::__private::cuda_device::{
                DisjointSlice, LinearTiles, Uniform, kernel, launch_bounds, launch_contract, thread,
            };

            #[inline(always)]
            fn ordered_before<K, F>(order: &F, left: &K, right: &K) -> bool
            where
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                order(left, right) == ::core::cmp::Ordering::Less
            }

            /// Return the stable co-rank of `diagonal` in two sorted runs.
            ///
            /// Equality is assigned to the left run first. This is the exact
            /// semantic distinction between CUB's stable merge and a merely
            /// sorted permutation; no source index or pointer escapes.
            #[inline(always)]
            fn merge_path<K, F>(
                first: &[K],
                first_begin: usize,
                first_count: usize,
                second: &[K],
                second_begin: usize,
                second_count: usize,
                diagonal: usize,
                order: &F,
            ) -> usize
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let mut low = if diagonal > second_count {
                    diagonal - second_count
                } else {
                    0
                };
                let mut high = if diagonal < first_count {
                    diagonal
                } else {
                    first_count
                };
                while low < high {
                    let middle = low + (high - low) / 2;
                    let second_taken = diagonal - middle;
                    let partition_is_high_enough = middle == first_count
                        || second_taken == 0
                        || ordered_before(
                            order,
                            &second[second_begin + second_taken - 1],
                            &first[first_begin + middle],
                        );
                    if partition_is_high_enough {
                        high = middle;
                    } else {
                        low = middle + 1;
                    }
                }
                low
            }

            #[inline(always)]
            fn merged_source<K, F>(
                first: &[K],
                first_begin: usize,
                first_count: usize,
                second: &[K],
                second_begin: usize,
                second_count: usize,
                diagonal: usize,
                order: &F,
            ) -> (bool, usize)
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let first_taken = merge_path(
                    first,
                    first_begin,
                    first_count,
                    second,
                    second_begin,
                    second_count,
                    diagonal,
                    order,
                );
                let second_taken = diagonal - first_taken;
                let take_first = first_taken < first_count
                    && (second_taken == second_count
                        || !ordered_before(
                            order,
                            &second[second_begin + second_taken],
                            &first[first_begin + first_taken],
                        ));
                if take_first {
                    (true, first_begin + first_taken)
                } else {
                    (false, second_begin + second_taken)
                }
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, block = (256, 1, 1))]
            pub fn device_partition_merge_path<K, F, const ITEMS: usize>(
                first: &[K],
                second: &[K],
                order: F,
                mut partitions: DisjointSlice<usize>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let diagonal_index = thread::index_1d(launch_context);
                let raw_index = diagonal_index.get();
                if let ::core::option::Option::Some(partition) =
                    partitions.get_mut(diagonal_index)
                {
                    let tile_items = 512usize * ITEMS;
                    let total = first.len() + second.len();
                    let diagonal = (raw_index * tile_items).min(total);
                    *partition = ::rccl_cub::__private::global_merge_path(
                        first, second, diagonal, &order,
                    );
                }
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_keys_agent<
                K,
                F,
                const ITEMS: usize,
                const K_SIZE: usize,
                const USE_LOAD_TO_SHARED: bool,
                const SHARED_BYTES: usize,
            >(
                first: &[K],
                second: &[K],
                partitions: &[usize],
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_keys_agent::<
                    K,
                    F,
                    512,
                    ITEMS,
                    K_SIZE,
                    USE_LOAD_TO_SHARED,
                    SHARED_BYTES,
                >(
                    first,
                    second,
                    partitions,
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_pairs_agent<
                K,
                V,
                F,
                const ITEMS: usize,
                const K_SIZE: usize,
                const V_SIZE: usize,
                const USE_LOAD_TO_SHARED_KEYS: bool,
                const USE_LOAD_TO_SHARED_VALUES: bool,
                const SHARED_BYTES: usize,
            >(
                first_keys: &[K],
                first_values: &[V],
                second_keys: &[K],
                second_values: &[V],
                partitions: &[usize],
                order: F,
                output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
                output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_pairs_agent::<
                    K,
                    V,
                    F,
                    512,
                    ITEMS,
                    K_SIZE,
                    V_SIZE,
                    USE_LOAD_TO_SHARED_KEYS,
                    USE_LOAD_TO_SHARED_VALUES,
                    SHARED_BYTES,
                >(
                    (first_keys, first_values),
                    (second_keys, second_values),
                    partitions,
                    order,
                    (output_keys, output_values),
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_keys_256<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                input: &[K],
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_keys_agent::<K, F, 256, ITEMS, SHARED_BYTES>(
                    input,
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_keys_in_place_256<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_keys_in_place_agent::<K, F, 256, ITEMS, SHARED_BYTES>(
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_keys_512<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                input: &[K],
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_keys_agent::<K, F, 512, ITEMS, SHARED_BYTES>(
                    input,
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_keys_in_place_512<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_keys_in_place_agent::<K, F, 512, ITEMS, SHARED_BYTES>(
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_pairs_256<
                K,
                V,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                input_keys: &[K],
                input_values: &[V],
                order: F,
                output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
                output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_pairs_agent::<K, V, F, 256, ITEMS, SHARED_BYTES>(
                    input_keys,
                    input_values,
                    order,
                    output_keys,
                    output_values,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_pairs_in_place_256<
                K,
                V,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                order: F,
                output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
                output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_pairs_in_place_agent::<
                    K,
                    V,
                    F,
                    256,
                    ITEMS,
                    SHARED_BYTES,
                >(order, output_keys, output_values, launch_context);
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_pairs_512<
                K,
                V,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                input_keys: &[K],
                input_values: &[V],
                order: F,
                output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
                output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_pairs_agent::<K, V, F, 512, ITEMS, SHARED_BYTES>(
                    input_keys,
                    input_values,
                    order,
                    output_keys,
                    output_values,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_block_pairs_in_place_512<
                K,
                V,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                order: F,
                output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
                output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_block_pairs_in_place_agent::<
                    K,
                    V,
                    F,
                    512,
                    ITEMS,
                    SHARED_BYTES,
                >(order, output_keys, output_values, launch_context);
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, block = (256, 1, 1))]
            pub fn device_merge_sort_partition<K, F, const THREADS: usize, const ITEMS: usize>(
                source: &[K],
                target_merged_tiles: usize,
                order: F,
                mut partitions: DisjointSlice<usize>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                let index = thread::index_1d(launch_context);
                let partition_index = index.get();
                let partition_count = partitions.len();
                if let ::core::option::Option::Some(partition) = partitions.get_mut(index) {
                    *partition = ::rccl_cub::__private::merge_sort_partition_point::<
                        K,
                        F,
                        THREADS,
                        ITEMS,
                    >(
                        source,
                        partition_index,
                        partition_count,
                        target_merged_tiles,
                        &order,
                    );
                }
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_merge_keys_256<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                source: &[K],
                partitions: &[usize],
                target_merged_tiles: usize,
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_merge_keys_agent::<K, F, 256, ITEMS, SHARED_BYTES>(
                    source,
                    partitions,
                    target_merged_tiles,
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_merge_keys_512<
                K,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                source: &[K],
                partitions: &[usize],
                target_merged_tiles: usize,
                order: F,
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_merge_keys_agent::<K, F, 512, ITEMS, SHARED_BYTES>(
                    source,
                    partitions,
                    target_merged_tiles,
                    order,
                    output,
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (256, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_merge_pairs_256<
                K,
                V,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                source_keys: &[K],
                source_values: &[V],
                partitions: &[usize],
                target_merged_tiles: usize,
                order: F,
                output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
                output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_merge_pairs_agent::<K, V, F, 256, ITEMS, SHARED_BYTES>(
                    (source_keys, source_values),
                    partitions,
                    target_merged_tiles,
                    order,
                    (output_keys, output_values),
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(512)]
            #[launch_contract(
                domain = 1,
                coordinates = u32,
                block = (512, 1, 1),
                dynamic_shared_range = (0, 49152),
                dynamic_shared_alignment = 16,
            )]
            pub fn device_merge_sort_merge_pairs_512<
                K,
                V,
                F,
                const ITEMS: usize,
                const SHARED_BYTES: usize,
            >(
                source_keys: &[K],
                source_values: &[V],
                partitions: &[usize],
                target_merged_tiles: usize,
                order: F,
                output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
                output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
            )
            where
                K: Copy,
                V: Copy,
                F: Fn(&K, &K) -> ::core::cmp::Ordering + Copy,
            {
                ::rccl_cub::__private::merge_sort_merge_pairs_agent::<K, V, F, 512, ITEMS, SHARED_BYTES>(
                    (source_keys, source_values),
                    partitions,
                    target_merged_tiles,
                    order,
                    (output_keys, output_values),
                    launch_context,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn device_radix_sort_init(
                counters: DisjointSlice<i32, LinearTiles<1>>,
                bins: DisjointSlice<u64, LinearTiles<1>>,
            ) {
                let counter_thread = thread::index_1d_u32(launch_context);
                let bin_thread = thread::index_1d_u32(launch_context);
                ::rccl_cub::__private::radix_init_bins_and_counters_agent(
                    counters,
                    bins,
                    counter_thread,
                    bin_thread,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256, 1)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn device_radix_sort_single_tile_keys<
                K,
                B,
                DefaultOrdered,
                ToOrdered,
                FromOrdered,
                RankDigit,
                const ITEMS: usize,
                const DESCENDING: bool,
            >(
                input: &[K],
                output: DisjointSlice<K, LinearTiles<ITEMS>>,
                begin_bit: Uniform<u32>,
                end_bit: Uniform<u32>,
                default_ordered: DefaultOrdered,
                to_ordered: ToOrdered,
                from_ordered: FromOrdered,
                rank_digit: RankDigit,
            ) where
                K: Copy,
                B: Copy,
                DefaultOrdered: Fn() -> B + Copy,
                ToOrdered: Fn(K) -> B + Copy,
                FromOrdered: Fn(B) -> K + Copy,
                RankDigit: Fn(B, u32, u32) -> u32 + Copy,
            {
                let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                    __RCCL_RADIX_SINGLE_TILE_KEYS,
                    33_856usize,
                    32usize
                );
                ::rccl_cub::__private::radix_single_tile_keys_agent::<
                    K,
                    B,
                    DefaultOrdered,
                    ToOrdered,
                    FromOrdered,
                    RankDigit,
                    256,
                    ITEMS,
                    DESCENDING,
                >(
                    input,
                    output,
                    begin_bit.get()..end_bit.get(),
                    (default_ordered, to_ordered, from_ordered, rank_digit),
                    launch_context,
                    shared_memory,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256, 1)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn device_radix_sort_single_tile_pairs<
                K,
                V,
                B,
                DefaultOrdered,
                ToOrdered,
                FromOrdered,
                RankDigit,
                const ITEMS: usize,
                const DESCENDING: bool,
            >(
                input_keys: &[K],
                input_values: &[V],
                output_keys: DisjointSlice<K, LinearTiles<ITEMS>>,
                output_values: DisjointSlice<V, LinearTiles<ITEMS>>,
                begin_bit: Uniform<u32>,
                end_bit: Uniform<u32>,
                default_ordered: DefaultOrdered,
                to_ordered: ToOrdered,
                from_ordered: FromOrdered,
                rank_digit: RankDigit,
            ) where
                K: Copy,
                V: Copy,
                B: Copy,
                DefaultOrdered: Fn() -> B + Copy,
                ToOrdered: Fn(K) -> B + Copy,
                FromOrdered: Fn(B) -> K + Copy,
                RankDigit: Fn(B, u32, u32) -> u32 + Copy,
            {
                let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                    __RCCL_RADIX_SINGLE_TILE_PAIRS,
                    33_856usize,
                    32usize
                );
                ::rccl_cub::__private::radix_single_tile_pairs_agent::<
                    K,
                    V,
                    B,
                    DefaultOrdered,
                    ToOrdered,
                    FromOrdered,
                    RankDigit,
                    256,
                    ITEMS,
                    DESCENDING,
                >(
                    (input_keys, input_values),
                    (output_keys, output_values),
                    begin_bit.get()..end_bit.get(),
                    (default_ordered, to_ordered, from_ordered, rank_digit),
                    launch_context,
                    shared_memory,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn device_radix_sort_histogram_4<
                K,
                B,
                DefaultKey,
                ToOrdered,
                Digit,
                const ITEMS: usize,
            >(
                keys: &[K],
                begin_bit: u32,
                end_bit: u32,
                default_key: DefaultKey,
                to_ordered: ToOrdered,
                digit: Digit,
                bins: DisjointSlice<u64>,
            ) where
                K: Copy,
                B: Copy,
                DefaultKey: Fn() -> B + Copy,
                ToOrdered: Fn(K) -> B + Copy,
                Digit: Fn(B, u32, u32) -> u32 + Copy,
            {
                let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                    __RCCL_RADIX_HISTOGRAM_4,
                    4_096usize,
                    16usize
                );
                ::rccl_cub::__private::radix_histogram_agent::<
                    K,
                    B,
                    DefaultKey,
                    ToOrdered,
                    Digit,
                    128,
                    ITEMS,
                    4usize,
                >(
                    keys,
                    begin_bit..end_bit,
                    default_key,
                    to_ordered,
                    digit,
                    bins,
                    shared_memory,
                );
            }

            #[kernel]
            #[launch_bounds(128)]
            #[launch_contract(domain = 1, coordinates = u32, block = (128, 1, 1))]
            pub fn device_radix_sort_histogram_8<
                K,
                B,
                DefaultKey,
                ToOrdered,
                Digit,
                const ITEMS: usize,
            >(
                keys: &[K],
                begin_bit: u32,
                end_bit: u32,
                default_key: DefaultKey,
                to_ordered: ToOrdered,
                digit: Digit,
                bins: DisjointSlice<u64>,
            ) where
                K: Copy,
                B: Copy,
                DefaultKey: Fn() -> B + Copy,
                ToOrdered: Fn(K) -> B + Copy,
                Digit: Fn(B, u32, u32) -> u32 + Copy,
            {
                let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                    __RCCL_RADIX_HISTOGRAM_8,
                    8_192usize,
                    16usize
                );
                ::rccl_cub::__private::radix_histogram_agent::<
                    K,
                    B,
                    DefaultKey,
                    ToOrdered,
                    Digit,
                    128,
                    ITEMS,
                    8usize,
                >(
                    keys,
                    begin_bit..end_bit,
                    default_key,
                    to_ordered,
                    digit,
                    bins,
                    shared_memory,
                );
            }

            #[kernel(launch_context = launch_context)]
            #[launch_bounds(256)]
            #[launch_contract(domain = 1, coordinates = u32, block = (256, 1, 1))]
            pub fn device_radix_sort_exclusive_sum<const PASSES: usize>(
                bins: DisjointSlice<u64>,
            ) {
                let shared_memory = ::rccl_cub::__private::static_block_shared_memory!(
                    __RCCL_RADIX_EXCLUSIVE_SUM,
                    2_336usize,
                    16usize
                );
                let bin_thread = thread::index_1d_u32(launch_context);
                ::rccl_cub::__private::radix_exclusive_sum_agent::<PASSES>(
                    bins,
                    bin_thread,
                    shared_memory,
                );
            }

            #(#radix_onesweep_keys)*
            #(#radix_onesweep_pairs)*

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

fn tmp_primitive_size(value_type: &syn::Type) -> syn::Result<usize> {
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
struct CudaTarget {
    capability: u16,
}

/// Parse one CUDA architecture used internally by CUB policy selection.
fn tmp_cuda_target(target: &syn::Type) -> syn::Result<CudaTarget> {
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
/// algorithm operands. Plain host-only Cargo checks use SM89 as the audit
/// baseline because cargo-oxide is not driving that build.
fn configured_cuda_target() -> syn::Result<CudaTarget> {
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

fn nominal_4b_items(nominal: usize, item_size: usize) -> usize {
    (nominal * 4 / item_size).clamp(1, nominal)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MergeSortPolicy {
    threads: usize,
    items: usize,
}

fn merge_sort_policy(target: CudaTarget, key_size: usize) -> MergeSortPolicy {
    if target.capability >= 60 {
        MergeSortPolicy {
            threads: 256,
            items: nominal_4b_items(17, key_size),
        }
    } else {
        MergeSortPolicy {
            threads: 512,
            items: nominal_4b_items(15, key_size),
        }
    }
}

fn align_16(bytes: usize) -> Option<usize> {
    bytes.checked_add(15).map(|value| value / 16 * 16)
}

/// Size of the exact CUB `AgentBlockSort`/`AgentMerge` shared-memory union.
///
/// `BlockExchange` inserts one item after each 32 logical items when the
/// per-thread item count is a power of two greater than four. `BlockStore`
/// derives from that 16-byte-aligned exchange storage and appends one volatile
/// `int` for the partial-tile guard. The block-merge member remains a dense
/// `T[ITEMS_PER_TILE + 1]`; the enclosing union takes the larger member.
fn merge_sort_shared_bytes(
    threads: usize,
    items: usize,
    key_size: usize,
    value_size: Option<usize>,
) -> Option<usize> {
    let tile_items = threads.checked_mul(items)?;
    let exchange_items = if items > 4 && items.is_power_of_two() {
        tile_items.checked_add(tile_items / 32)?
    } else {
        tile_items
    };
    let representation = |item_size: usize| -> Option<usize> {
        let merge = tile_items.checked_add(1)?.checked_mul(item_size)?;
        let exchange = align_16(exchange_items.checked_mul(item_size)?)?;
        let store = align_16(exchange.checked_add(core::mem::size_of::<i32>())?)?;
        Some(merge.max(store))
    };
    let mut bytes = representation(key_size)?;
    if let Some(value_size) = value_size {
        bytes = bytes.max(representation(value_size)?);
    }
    align_16(bytes)
}

fn merge_load_to_shared_policy(
    target: CudaTarget,
    key_size: usize,
    value_size: Option<usize>,
) -> (bool, bool) {
    if target.capability >= 100 {
        return (true, value_size.is_some());
    }
    if target.capability >= 90 {
        let selected = match value_size {
            None => key_size != 8,
            Some(value_size) => match key_size {
                8 => value_size != 8,
                16 => matches!(value_size, 1 | 8),
                _ => true,
            },
        };
        return (selected, value_size.is_some() && selected);
    }
    if target.capability >= 80 {
        let selected = match value_size {
            None => key_size < 4,
            Some(value_size) => {
                key_size == 1
                    || (key_size == 2 && value_size < 4)
                    || (key_size == 4 && value_size == 1)
            }
        };
        return (selected, value_size.is_some() && selected);
    }
    (false, false)
}

fn load_to_shared_buffer_bytes(items: usize, item_size: usize) -> usize {
    let bytes = items * item_size;
    if item_size >= 16 || bytes == 0 {
        bytes
    } else {
        bytes.div_ceil(16) * 16 + 16
    }
}

fn merge_shared_bytes(
    threads: usize,
    items: usize,
    key_size: usize,
    value_size: Option<usize>,
    use_load_to_shared_keys: bool,
    use_load_to_shared_values: bool,
) -> usize {
    let tile_items = threads * items;
    let exchange_items = if items > 4 && items.is_power_of_two() {
        tile_items + tile_items / 32
    } else {
        tile_items
    };
    let representation = |item_size: usize, use_load_to_shared: bool| {
        if use_load_to_shared {
            load_to_shared_buffer_bytes(tile_items + 1, item_size)
                + usize::from(item_size < 16) * 32
        } else {
            (tile_items + 1) * item_size
        }
    };
    let block_store = |item_size: usize| {
        align_16(align_16(exchange_items * item_size).unwrap() + core::mem::size_of::<i32>())
            .unwrap()
    };
    let mut union_bytes =
        representation(key_size, use_load_to_shared_keys).max(block_store(key_size));
    if let Some(value_size) = value_size {
        union_bytes = union_bytes
            .max(representation(value_size, use_load_to_shared_values))
            .max(block_store(value_size));
    }
    let union_bytes = align_16(union_bytes).unwrap();
    if use_load_to_shared_keys || use_load_to_shared_values {
        // `temp_storages_with_bl2sh` derives from the 16-byte-aligned union,
        // appends BlockLoadToShared's u64 mbarrier cell, then rounds the whole
        // derived object back to its inherited 16-byte alignment.
        align_16(union_bytes + core::mem::size_of::<u64>()).unwrap()
    } else {
        union_bytes
    }
}

fn merge_keys_launch(
    key_type: &syn::Type,
    key_size: usize,
    use_load_to_shared: bool,
    items: usize,
) -> proc_macro2::TokenStream {
    let tile_items = 512usize * items;
    let shared_bytes = merge_shared_bytes(512, items, key_size, None, use_load_to_shared, false);
    quote! {{
        let __rccl_num_tiles = __rccl_output_len.div_ceil(#tile_items);
        let __rccl_num_diagonals = __rccl_num_tiles + 1;
        let __rccl_partition_launch = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
            __rccl_num_diagonals.div_ceil(256).min(u32::MAX as usize) as u32,
            256,
            0,
        );
        match __rccl_execution
            .module()
            .prepare_device_partition_merge_path::<#key_type, _, #items>(__rccl_partition_launch)
        {
            ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
            ),
            ::core::result::Result::Ok(__rccl_partition_prepared) => match __rccl_execution
                .module()
                .device_partition_merge_path::<#key_type, _, #items>(
                    __rccl_execution.stream(),
                    &__rccl_partition_prepared,
                    __rccl_first,
                    __rccl_second,
                    __rccl_order,
                    &mut *__rccl_partitions,
                )
            {
                ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                ),
                ::core::result::Result::Ok(()) => {
                    let __rccl_agent_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_tiles.min(u32::MAX as usize) as u32,
                            512,
                            #shared_bytes as u32,
                        );
                    match __rccl_execution
                        .module()
                        .prepare_device_merge_keys_agent::<#key_type, _, #items, #key_size, #use_load_to_shared, #shared_bytes>(
                            __rccl_agent_launch,
                        )
                    {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(__rccl_agent_prepared) => __rccl_execution
                            .module()
                            .device_merge_keys_agent::<#key_type, _, #items, #key_size, #use_load_to_shared, #shared_bytes>(
                                __rccl_execution.stream(),
                                &__rccl_agent_prepared,
                                __rccl_first,
                                __rccl_second,
                                &*__rccl_partitions,
                                __rccl_order,
                                __rccl_output,
                            )
                            .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                    }
                }
            },
        }
    }}
}

fn merge_pairs_launch(
    key_type: &syn::Type,
    value_type: &syn::Type,
    key_size: usize,
    value_size: usize,
    use_load_to_shared_keys: bool,
    use_load_to_shared_values: bool,
    items: usize,
) -> proc_macro2::TokenStream {
    let tile_items = 512usize * items;
    let shared_bytes = merge_shared_bytes(
        512,
        items,
        key_size,
        Some(value_size),
        use_load_to_shared_keys,
        use_load_to_shared_values,
    );
    quote! {{
        let __rccl_num_tiles = __rccl_output_len.div_ceil(#tile_items);
        let __rccl_num_diagonals = __rccl_num_tiles + 1;
        let __rccl_partition_launch = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
            __rccl_num_diagonals.div_ceil(256).min(u32::MAX as usize) as u32,
            256,
            0,
        );
        match __rccl_execution
            .module()
            .prepare_device_partition_merge_path::<#key_type, _, #items>(__rccl_partition_launch)
        {
            ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
            ),
            ::core::result::Result::Ok(__rccl_partition_prepared) => match __rccl_execution
                .module()
                .device_partition_merge_path::<#key_type, _, #items>(
                    __rccl_execution.stream(),
                    &__rccl_partition_prepared,
                    __rccl_first_keys,
                    __rccl_second_keys,
                    __rccl_order,
                    &mut *__rccl_partitions,
                )
            {
                ::core::result::Result::Err(__rccl_error) => ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                ),
                ::core::result::Result::Ok(()) => {
                    let __rccl_agent_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_tiles.min(u32::MAX as usize) as u32,
                            512,
                            #shared_bytes as u32,
                        );
                    match __rccl_execution
                        .module()
                        .prepare_device_merge_pairs_agent::<#key_type, #value_type, _, #items, #key_size, #value_size, #use_load_to_shared_keys, #use_load_to_shared_values, #shared_bytes>(
                            __rccl_agent_launch,
                        )
                    {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(__rccl_agent_prepared) => __rccl_execution
                            .module()
                            .device_merge_pairs_agent::<#key_type, #value_type, _, #items, #key_size, #value_size, #use_load_to_shared_keys, #use_load_to_shared_values, #shared_bytes>(
                                __rccl_execution.stream(),
                                &__rccl_agent_prepared,
                                __rccl_first_keys,
                                __rccl_first_values,
                                __rccl_second_keys,
                                __rccl_second_values,
                                &*__rccl_partitions,
                                __rccl_order,
                                __rccl_output_keys,
                                __rccl_output_values,
                            )
                            .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                    }
                }
            },
        }
    }}
}

fn expand_device_merge_keys(
    call: call::DeviceMergeKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = configured_cuda_target()?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let items = nominal_4b_items(nominal_items, key_size);
    let (use_load_to_shared, _) = merge_load_to_shared_policy(target, key_size, None);
    let launch = merge_keys_launch(&key_type, key_size, use_load_to_shared, items);
    let execution = call.execution;
    let first = call.first;
    let second = call.second;
    let workspace = call.workspace;
    let output = call.output;
    let order = call.order;
    audit::generated_algorithm_expression(
        audit::Algorithm::Merge,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_first = #first;
            let __rccl_second = #second;
            let __rccl_workspace = #workspace;
            let __rccl_partitions = __rccl_workspace.__partitions_mut();
            let __rccl_output = #output;
            let __rccl_order = #order;
            match __rccl_first.len().checked_add(__rccl_second.len()) {
                ::core::option::Option::None => ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::SizeOverflow,
                ),
                ::core::option::Option::Some(__rccl_output_len)
                    if __rccl_output.len() != __rccl_output_len =>
                {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "output",
                        expected: __rccl_output_len,
                        actual: __rccl_output.len(),
                    })
                }
                ::core::option::Option::Some(0) => ::core::result::Result::Ok(()),
                ::core::option::Option::Some(__rccl_output_len) => {
                    let __rccl_required_partitions = __rccl_output_len.div_ceil(#items * 512) + 1;
                    if __rccl_partitions.len() != __rccl_required_partitions {
                        ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                role: "merge partitions",
                                expected: __rccl_required_partitions,
                                actual: __rccl_partitions.len(),
                            },
                        )
                    } else {
                        #launch
                    }
                }
            }
        }},
    )
}

fn expand_device_merge_keys_workspace(
    call: call::DeviceMergeKeysWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let target = configured_cuda_target()?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let tile_items = 512usize * nominal_4b_items(nominal_items, key_size);
    let execution = call.execution;
    let first_len = call.first_len;
    let second_len = call.second_len;
    audit::generated_expression(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceMergeWorkspace,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_total = (#first_len)
                .checked_add(#second_len)
                .ok_or(::rccl_cub::DeviceAlgorithmError::SizeOverflow)?;
            let __rccl_count = if __rccl_total == 0 {
                0usize
            } else {
                __rccl_total.div_ceil(#tile_items) + 1
            };
            let __rccl_partitions =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<usize>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_count,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            ::core::result::Result::Ok(
                ::rccl_cub::DeviceMergeWorkspace::__from_partitions(__rccl_partitions),
            )
        })()
    }})
}

fn expand_device_merge_keys_partition_count(
    call: call::DeviceMergeKeysPartitionCountCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let target = configured_cuda_target()?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let items = nominal_4b_items(nominal_items, key_size);
    let tile_items = 512usize * items;
    let first_len = call.first_len;
    let second_len = call.second_len;
    audit::generated_expression(quote! {{
        match (#first_len).checked_add(#second_len) {
            ::core::option::Option::Some(0) => ::core::result::Result::Ok(0usize),
            ::core::option::Option::Some(__rccl_total) => {
                ::core::result::Result::Ok(__rccl_total.div_ceil(#tile_items) + 1)
            }
            ::core::option::Option::None => ::core::result::Result::Err(
                ::rccl_cub::DeviceAlgorithmError::SizeOverflow,
            ),
        }
    }})
}

fn expand_device_merge_pairs_partition_count(
    call: call::DeviceMergePairsPartitionCountCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let value_size = tmp_primitive_size(&call.value_type)?;
    let target = configured_cuda_target()?;
    let tune_size = key_size
        .checked_add(value_size)
        .ok_or_else(|| syn::Error::new_spanned(&call.value_type, "merge tuning size overflowed"))?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let items = nominal_4b_items(nominal_items, tune_size);
    let tile_items = 512usize * items;
    let first_len = call.first_len;
    let second_len = call.second_len;
    audit::generated_expression(quote! {{
        match (#first_len).checked_add(#second_len) {
            ::core::option::Option::Some(0) => ::core::result::Result::Ok(0usize),
            ::core::option::Option::Some(__rccl_total) => {
                ::core::result::Result::Ok(__rccl_total.div_ceil(#tile_items) + 1)
            }
            ::core::option::Option::None => ::core::result::Result::Err(
                ::rccl_cub::DeviceAlgorithmError::SizeOverflow,
            ),
        }
    }})
}

fn expand_device_merge_pairs(
    call: call::DeviceMergePairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let target = configured_cuda_target()?;
    let key_size = tmp_primitive_size(&key_type)?;
    let value_size = tmp_primitive_size(&value_type)?;
    let tune_size = key_size
        .checked_add(value_size)
        .ok_or_else(|| syn::Error::new_spanned(&value_type, "merge tuning size overflowed"))?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let items = nominal_4b_items(nominal_items, tune_size);
    let (use_load_to_shared_keys, use_load_to_shared_values) =
        merge_load_to_shared_policy(target, key_size, Some(value_size));
    let launch = merge_pairs_launch(
        &key_type,
        &value_type,
        key_size,
        value_size,
        use_load_to_shared_keys,
        use_load_to_shared_values,
        items,
    );
    let execution = call.execution;
    let first_keys = call.first_keys;
    let first_values = call.first_values;
    let second_keys = call.second_keys;
    let second_values = call.second_values;
    let workspace = call.workspace;
    let output_keys = call.output_keys;
    let output_values = call.output_values;
    let order = call.order;
    audit::generated_algorithm_expression(
        audit::Algorithm::Merge,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_first_keys = #first_keys;
            let __rccl_first_values = #first_values;
            let __rccl_second_keys = #second_keys;
            let __rccl_second_values = #second_values;
            let __rccl_workspace = #workspace;
            let __rccl_partitions = __rccl_workspace.__partitions_mut();
            let __rccl_output_keys = #output_keys;
            let __rccl_output_values = #output_values;
            let __rccl_order = #order;
            let __rccl_first_len = __rccl_first_keys.len();
            let __rccl_second_len = __rccl_second_keys.len();
            let __rccl_shape_error = if __rccl_first_values.len() != __rccl_first_len {
                ::core::option::Option::Some((
                    "first values",
                    __rccl_first_len,
                    __rccl_first_values.len(),
                ))
            } else if __rccl_second_values.len() != __rccl_second_len {
                ::core::option::Option::Some((
                    "second values",
                    __rccl_second_len,
                    __rccl_second_values.len(),
                ))
            } else {
                ::core::option::Option::None
            };
            if let ::core::option::Option::Some((__rccl_role, __rccl_expected, __rccl_actual)) =
                __rccl_shape_error
            {
                ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                    role: __rccl_role,
                    expected: __rccl_expected,
                    actual: __rccl_actual,
                })
            } else {
                match __rccl_first_len.checked_add(__rccl_second_len) {
                    ::core::option::Option::None => ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::SizeOverflow,
                    ),
                    ::core::option::Option::Some(__rccl_output_len)
                        if __rccl_output_keys.len() != __rccl_output_len =>
                    {
                        ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                role: "output keys",
                                expected: __rccl_output_len,
                                actual: __rccl_output_keys.len(),
                            },
                        )
                    }
                    ::core::option::Option::Some(__rccl_output_len)
                        if __rccl_output_values.len() != __rccl_output_len =>
                    {
                        ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                role: "output values",
                                expected: __rccl_output_len,
                                actual: __rccl_output_values.len(),
                            },
                        )
                    }
                    ::core::option::Option::Some(0) => ::core::result::Result::Ok(()),
                    ::core::option::Option::Some(__rccl_output_len) => {
                        let __rccl_required_partitions =
                            __rccl_output_len.div_ceil(#items * 512) + 1;
                        if __rccl_partitions.len() != __rccl_required_partitions {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                    role: "merge partitions",
                                    expected: __rccl_required_partitions,
                                    actual: __rccl_partitions.len(),
                                },
                            )
                        } else {
                            #launch
                        }
                    }
                }
            }
        }},
    )
}

fn expand_device_merge_pairs_workspace(
    call: call::DeviceMergePairsWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let value_size = tmp_primitive_size(&call.value_type)?;
    let target = configured_cuda_target()?;
    let tune_size = key_size
        .checked_add(value_size)
        .ok_or_else(|| syn::Error::new_spanned(&call.value_type, "merge tuning size overflowed"))?;
    let nominal_items = if target.capability >= 60 { 15 } else { 13 };
    let tile_items = 512usize * nominal_4b_items(nominal_items, tune_size);
    let execution = call.execution;
    let first_len = call.first_len;
    let second_len = call.second_len;
    audit::generated_expression(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceMergeWorkspace,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_total = (#first_len)
                .checked_add(#second_len)
                .ok_or(::rccl_cub::DeviceAlgorithmError::SizeOverflow)?;
            let __rccl_count = if __rccl_total == 0 {
                0usize
            } else {
                __rccl_total.div_ceil(#tile_items) + 1
            };
            let __rccl_partitions =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<usize>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_count,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            ::core::result::Result::Ok(
                ::rccl_cub::DeviceMergeWorkspace::__from_partitions(__rccl_partitions),
            )
        })()
    }})
}

fn expand_device_merge_sort_partition_count(
    call: call::DeviceMergeSortPartitionCountCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&call.key_type)?;
    let target = configured_cuda_target()?;
    let policy = merge_sort_policy(target, key_size);
    let tile_items = policy.threads * policy.items;
    let len = call.len;
    audit::generated_expression(quote! {{
        let __rccl_len: usize = #len;
        if __rccl_len == 0 {
            ::core::result::Result::<usize, ::rccl_cub::DeviceAlgorithmError>::Ok(0usize)
        } else {
            ::core::result::Result::<usize, ::rccl_cub::DeviceAlgorithmError>::Ok(
                __rccl_len.div_ceil(#tile_items) + 1,
            )
        }
    }})
}

fn expand_device_merge_sort_keys_workspace(
    call: call::DeviceMergeSortKeysWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = configured_cuda_target()?;
    let policy = merge_sort_policy(target, key_size);
    let tile_items = policy.threads * policy.items;
    let execution = call.execution;
    let len = call.len;
    audit::generated_expression(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceMergeSortKeysWorkspace<#key_type>,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_len: usize = #len;
            let __rccl_partition_count = if __rccl_len == 0 {
                0usize
            } else {
                __rccl_len.div_ceil(#tile_items) + 1
            };
            let __rccl_scratch_keys =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<#key_type>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_len,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_partitions =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<usize>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_partition_count,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            ::core::result::Result::Ok(
                ::rccl_cub::DeviceMergeSortKeysWorkspace::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_partitions,
                ),
            )
        })()
    }})
}

fn expand_device_merge_sort_pairs_workspace(
    call: call::DeviceMergeSortPairsWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = configured_cuda_target()?;
    let policy = merge_sort_policy(target, key_size);
    let tile_items = policy.threads * policy.items;
    let execution = call.execution;
    let len = call.len;
    audit::generated_expression(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceMergeSortPairsWorkspace<#key_type, #value_type>,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_len: usize = #len;
            let __rccl_partition_count = if __rccl_len == 0 {
                0usize
            } else {
                __rccl_len.div_ceil(#tile_items) + 1
            };
            let __rccl_scratch_keys =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<#key_type>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_len,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_scratch_values =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<#value_type>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_len,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_partitions =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<usize>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_partition_count,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            ::core::result::Result::Ok(
                ::rccl_cub::DeviceMergeSortPairsWorkspace::__from_parts(
                    __rccl_scratch_keys,
                    __rccl_scratch_values,
                    __rccl_partitions,
                ),
            )
        })()
    }})
}

fn expand_device_merge_sort_keys(
    call: call::DeviceMergeSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = configured_cuda_target()?;
    let policy = merge_sort_policy(target, key_size);
    let threads = policy.threads;
    let items = policy.items;
    let tile_items = threads * items;
    let shared_bytes =
        merge_sort_shared_bytes(threads, items, key_size, None).ok_or_else(|| {
            syn::Error::new_spanned(&key_type, "merge-sort shared storage overflowed")
        })?;
    let prepare_block = format_ident!("prepare_device_merge_sort_block_keys_{threads}");
    let launch_block = format_ident!("device_merge_sort_block_keys_{threads}");
    let prepare_block_in_place =
        format_ident!("prepare_device_merge_sort_block_keys_in_place_{threads}");
    let launch_block_in_place = format_ident!("device_merge_sort_block_keys_in_place_{threads}");
    let prepare_merge = format_ident!("prepare_device_merge_sort_merge_keys_{threads}");
    let launch_merge = format_ident!("device_merge_sort_merge_keys_{threads}");
    let execution = call.execution;
    let keys = call.keys;
    let workspace = call.workspace;
    let order = call.order;
    audit::generated_algorithm_expression(
        audit::Algorithm::MergeSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_keys = #keys;
            let __rccl_workspace = #workspace;
            let (__rccl_scratch_keys, __rccl_partitions) = __rccl_workspace.__parts_mut();
            let __rccl_order = #order;
            let __rccl_len = __rccl_keys.len();
            if __rccl_scratch_keys.len() != __rccl_len {
                ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                    role: "merge-sort scratch keys",
                    expected: __rccl_len,
                    actual: __rccl_scratch_keys.len(),
                })
            } else if __rccl_len == 0 {
                if __rccl_partitions.len() == 0 {
                    ::core::result::Result::Ok(())
                } else {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: 0,
                        actual: __rccl_partitions.len(),
                    })
                }
            } else {
                let __rccl_num_tiles = __rccl_len.div_ceil(#tile_items);
                let __rccl_num_partitions = __rccl_num_tiles + 1;
                if __rccl_partitions.len() != __rccl_num_partitions {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: __rccl_num_partitions,
                        actual: __rccl_partitions.len(),
                    })
                } else {
                    let __rccl_num_passes = usize::BITS as usize
                        - (__rccl_num_tiles - 1).leading_zeros() as usize;
                    let mut __rccl_ping = (__rccl_num_passes & 1) == 0;
                    let __rccl_agent_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_tiles.min(u32::MAX as usize) as u32,
                            #threads as u32,
                            #shared_bytes as u32,
                        );
                    let mut __rccl_result = if __rccl_ping {
                        match __rccl_execution
                            .module()
                            .#prepare_block_in_place::<#key_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                        {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                                .module()
                                .#launch_block_in_place::<#key_type, _, #items, #shared_bytes>(
                                    __rccl_execution.stream(),
                                    &__rccl_prepared,
                                    __rccl_order,
                                    &mut *__rccl_keys,
                                )
                                .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                        }
                    } else {
                        match __rccl_execution
                            .module()
                            .#prepare_block::<#key_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                        {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                                .module()
                                .#launch_block::<#key_type, _, #items, #shared_bytes>(
                                    __rccl_execution.stream(),
                                    &__rccl_prepared,
                                    &*__rccl_keys,
                                    __rccl_order,
                                    &mut *__rccl_scratch_keys,
                                )
                                .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                        }
                    };
                    let __rccl_partition_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_partitions.div_ceil(256).min(u32::MAX as usize) as u32,
                            256,
                            0,
                        );
                    let mut __rccl_pass = 0usize;
                    while __rccl_pass < __rccl_num_passes && __rccl_result.is_ok() {
                        let __rccl_target_merged_tiles = 2usize << __rccl_pass;
                        __rccl_result = match __rccl_execution
                            .module()
                            .prepare_device_merge_sort_partition::<#key_type, _, #threads, #items>(
                                __rccl_partition_launch,
                            )
                        {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => {
                                let __rccl_launch_result = if __rccl_ping {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_keys,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                } else {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_scratch_keys,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                };
                                __rccl_launch_result
                                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                            }
                        };
                        if __rccl_result.is_ok() {
                            __rccl_result = match __rccl_execution
                                .module()
                                .#prepare_merge::<#key_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                            {
                                ::core::result::Result::Err(__rccl_error) => {
                                    ::core::result::Result::Err(
                                        ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                    )
                                }
                                ::core::result::Result::Ok(__rccl_prepared) => {
                                    let __rccl_launch_result = if __rccl_ping {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_keys,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_scratch_keys,
                                        )
                                    } else {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_scratch_keys,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_keys,
                                        )
                                    };
                                    __rccl_launch_result
                                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                                }
                            };
                        }
                        __rccl_ping = !__rccl_ping;
                        __rccl_pass += 1;
                    }
                    __rccl_result
                }
            }
        }},
    )
}

fn expand_device_merge_sort_keys_copy(
    call: call::DeviceMergeSortKeysCopyCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let target = configured_cuda_target()?;
    let policy = merge_sort_policy(target, key_size);
    let threads = policy.threads;
    let items = policy.items;
    let tile_items = threads * items;
    let shared_bytes =
        merge_sort_shared_bytes(threads, items, key_size, None).ok_or_else(|| {
            syn::Error::new_spanned(&key_type, "merge-sort shared storage overflowed")
        })?;
    let prepare_block = format_ident!("prepare_device_merge_sort_block_keys_{threads}");
    let launch_block = format_ident!("device_merge_sort_block_keys_{threads}");
    let prepare_merge = format_ident!("prepare_device_merge_sort_merge_keys_{threads}");
    let launch_merge = format_ident!("device_merge_sort_merge_keys_{threads}");
    let execution = call.execution;
    let input = call.input;
    let workspace = call.workspace;
    let output = call.output;
    let order = call.order;
    audit::generated_algorithm_expression(
        audit::Algorithm::MergeSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_input = #input;
            let __rccl_workspace = #workspace;
            let (__rccl_scratch_keys, __rccl_partitions) = __rccl_workspace.__parts_mut();
            let __rccl_output = #output;
            let __rccl_order = #order;
            let __rccl_len = __rccl_input.len();
            let __rccl_shape_error = if __rccl_output.len() != __rccl_len {
                ::core::option::Option::Some(("output keys", __rccl_output.len()))
            } else if __rccl_scratch_keys.len() != __rccl_len {
                ::core::option::Option::Some((
                    "merge-sort scratch keys",
                    __rccl_scratch_keys.len(),
                ))
            } else {
                ::core::option::Option::None
            };
            if let ::core::option::Option::Some((__rccl_role, __rccl_actual)) = __rccl_shape_error {
                ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                    role: __rccl_role,
                    expected: __rccl_len,
                    actual: __rccl_actual,
                })
            } else if __rccl_len == 0 {
                if __rccl_partitions.len() == 0 {
                    ::core::result::Result::Ok(())
                } else {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: 0,
                        actual: __rccl_partitions.len(),
                    })
                }
            } else {
                let __rccl_num_tiles = __rccl_len.div_ceil(#tile_items);
                let __rccl_num_partitions = __rccl_num_tiles + 1;
                if __rccl_partitions.len() != __rccl_num_partitions {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: __rccl_num_partitions,
                        actual: __rccl_partitions.len(),
                    })
                } else {
                    let __rccl_num_passes = usize::BITS as usize
                        - (__rccl_num_tiles - 1).leading_zeros() as usize;
                    let mut __rccl_ping = (__rccl_num_passes & 1) == 0;
                    let __rccl_agent_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_tiles.min(u32::MAX as usize) as u32,
                            #threads as u32,
                            #shared_bytes as u32,
                        );
                    let mut __rccl_result = match __rccl_execution
                        .module()
                        .#prepare_block::<#key_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                    {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(__rccl_prepared) => {
                            let __rccl_launch_result = if __rccl_ping {
                                __rccl_execution.module().#launch_block::<#key_type, _, #items, #shared_bytes>(
                                    __rccl_execution.stream(),
                                    &__rccl_prepared,
                                    __rccl_input,
                                    __rccl_order,
                                    &mut *__rccl_output,
                                )
                            } else {
                                __rccl_execution.module().#launch_block::<#key_type, _, #items, #shared_bytes>(
                                    __rccl_execution.stream(),
                                    &__rccl_prepared,
                                    __rccl_input,
                                    __rccl_order,
                                    &mut *__rccl_scratch_keys,
                                )
                            };
                            __rccl_launch_result.map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                        }
                    };
                    let __rccl_partition_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_partitions.div_ceil(256).min(u32::MAX as usize) as u32,
                            256,
                            0,
                        );
                    let mut __rccl_pass = 0usize;
                    while __rccl_pass < __rccl_num_passes && __rccl_result.is_ok() {
                        let __rccl_target_merged_tiles = 2usize << __rccl_pass;
                        __rccl_result = match __rccl_execution
                            .module()
                            .prepare_device_merge_sort_partition::<#key_type, _, #threads, #items>(
                                __rccl_partition_launch,
                            )
                        {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => {
                                let __rccl_launch_result = if __rccl_ping {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_output,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                } else {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_scratch_keys,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                };
                                __rccl_launch_result
                                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                            }
                        };
                        if __rccl_result.is_ok() {
                            __rccl_result = match __rccl_execution
                                .module()
                                .#prepare_merge::<#key_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                            {
                                ::core::result::Result::Err(__rccl_error) => {
                                    ::core::result::Result::Err(
                                        ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                    )
                                }
                                ::core::result::Result::Ok(__rccl_prepared) => {
                                    let __rccl_launch_result = if __rccl_ping {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_output,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_scratch_keys,
                                        )
                                    } else {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_scratch_keys,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_output,
                                        )
                                    };
                                    __rccl_launch_result
                                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                                }
                            };
                        }
                        __rccl_ping = !__rccl_ping;
                        __rccl_pass += 1;
                    }
                    __rccl_result
                }
            }
        }},
    )
}

fn expand_device_merge_sort_pairs(
    call: call::DeviceMergeSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_merge_sort_pairs_operation(
        call.key_type,
        call.value_type,
        call.execution,
        None,
        None,
        call.keys,
        call.values,
        call.workspace,
        call.order,
    )
}

fn expand_device_merge_sort_pairs_copy(
    call: call::DeviceMergeSortPairsCopyCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_merge_sort_pairs_operation(
        call.key_type,
        call.value_type,
        call.execution,
        Some(call.input_keys),
        Some(call.input_values),
        call.output_keys,
        call.output_values,
        call.workspace,
        call.order,
    )
}

#[allow(clippy::too_many_arguments)]
fn expand_device_merge_sort_pairs_operation(
    key_type: syn::Type,
    value_type: syn::Type,
    execution: syn::Expr,
    input_keys: Option<syn::Expr>,
    input_values: Option<syn::Expr>,
    primary_keys: syn::Expr,
    primary_values: syn::Expr,
    workspace: syn::Expr,
    order: syn::Expr,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_size = tmp_primitive_size(&key_type)?;
    let value_size = tmp_primitive_size(&value_type)?;
    let target = configured_cuda_target()?;
    let policy = merge_sort_policy(target, key_size);
    let threads = policy.threads;
    let items = policy.items;
    let tile_items = threads * items;
    let shared_bytes = merge_sort_shared_bytes(threads, items, key_size, Some(value_size))
        .ok_or_else(|| {
            syn::Error::new_spanned(&value_type, "merge-sort shared storage overflowed")
        })?;
    if shared_bytes > 49_152 {
        return Err(syn::Error::new_spanned(
            &value_type,
            "this pinned CUB pair policy requires virtual shared memory, whose exact agent path is not implemented; refusing a smaller substitute",
        ));
    }

    let prepare_block = format_ident!("prepare_device_merge_sort_block_pairs_{threads}");
    let launch_block = format_ident!("device_merge_sort_block_pairs_{threads}");
    let prepare_block_in_place =
        format_ident!("prepare_device_merge_sort_block_pairs_in_place_{threads}");
    let launch_block_in_place = format_ident!("device_merge_sort_block_pairs_in_place_{threads}");
    let prepare_merge = format_ident!("prepare_device_merge_sort_merge_pairs_{threads}");
    let launch_merge = format_ident!("device_merge_sort_merge_pairs_{threads}");

    let (mode_bindings, shape_error, initial_block) = match (input_keys, input_values) {
        (Some(input_keys), Some(input_values)) => (
            quote! {
                let __rccl_input_keys = #input_keys;
                let __rccl_input_values = #input_values;
                let __rccl_primary_keys = #primary_keys;
                let __rccl_primary_values = #primary_values;
            },
            quote! {
                if __rccl_input_values.len() != __rccl_len {
                    ::core::option::Option::Some(("input values", __rccl_input_values.len()))
                } else if __rccl_primary_keys.len() != __rccl_len {
                    ::core::option::Option::Some(("output keys", __rccl_primary_keys.len()))
                } else if __rccl_primary_values.len() != __rccl_len {
                    ::core::option::Option::Some(("output values", __rccl_primary_values.len()))
                } else if __rccl_scratch_keys.len() != __rccl_len {
                    ::core::option::Option::Some(("merge-sort scratch keys", __rccl_scratch_keys.len()))
                } else if __rccl_scratch_values.len() != __rccl_len {
                    ::core::option::Option::Some(("merge-sort scratch values", __rccl_scratch_values.len()))
                } else {
                    ::core::option::Option::None
                }
            },
            quote! {
                match __rccl_execution
                    .module()
                    .#prepare_block::<#key_type, #value_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                {
                    ::core::result::Result::Err(__rccl_error) => {
                        ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                        )
                    }
                    ::core::result::Result::Ok(__rccl_prepared) => {
                        let __rccl_launch_result = if __rccl_ping {
                            __rccl_execution.module().#launch_block::<
                                #key_type,
                                #value_type,
                                _,
                                #items,
                                #shared_bytes,
                            >(
                                __rccl_execution.stream(),
                                &__rccl_prepared,
                                __rccl_input_keys,
                                __rccl_input_values,
                                __rccl_order,
                                &mut *__rccl_primary_keys,
                                &mut *__rccl_primary_values,
                            )
                        } else {
                            __rccl_execution.module().#launch_block::<
                                #key_type,
                                #value_type,
                                _,
                                #items,
                                #shared_bytes,
                            >(
                                __rccl_execution.stream(),
                                &__rccl_prepared,
                                __rccl_input_keys,
                                __rccl_input_values,
                                __rccl_order,
                                &mut *__rccl_scratch_keys,
                                &mut *__rccl_scratch_values,
                            )
                        };
                        __rccl_launch_result.map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                    }
                }
            },
        ),
        (None, None) => (
            quote! {
                let __rccl_primary_keys = #primary_keys;
                let __rccl_primary_values = #primary_values;
            },
            quote! {
                if __rccl_primary_values.len() != __rccl_len {
                    ::core::option::Option::Some(("values", __rccl_primary_values.len()))
                } else if __rccl_scratch_keys.len() != __rccl_len {
                    ::core::option::Option::Some(("merge-sort scratch keys", __rccl_scratch_keys.len()))
                } else if __rccl_scratch_values.len() != __rccl_len {
                    ::core::option::Option::Some(("merge-sort scratch values", __rccl_scratch_values.len()))
                } else {
                    ::core::option::Option::None
                }
            },
            quote! {
                if __rccl_ping {
                    match __rccl_execution
                        .module()
                        .#prepare_block_in_place::<#key_type, #value_type, _, #items, #shared_bytes>(
                            __rccl_agent_launch,
                        )
                    {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                            .module()
                            .#launch_block_in_place::<#key_type, #value_type, _, #items, #shared_bytes>(
                                __rccl_execution.stream(),
                                &__rccl_prepared,
                                __rccl_order,
                                &mut *__rccl_primary_keys,
                                &mut *__rccl_primary_values,
                            )
                            .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                    }
                } else {
                    match __rccl_execution
                        .module()
                        .#prepare_block::<#key_type, #value_type, _, #items, #shared_bytes>(__rccl_agent_launch)
                    {
                        ::core::result::Result::Err(__rccl_error) => {
                            ::core::result::Result::Err(
                                ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                            )
                        }
                        ::core::result::Result::Ok(__rccl_prepared) => __rccl_execution
                            .module()
                            .#launch_block::<#key_type, #value_type, _, #items, #shared_bytes>(
                                __rccl_execution.stream(),
                                &__rccl_prepared,
                                &*__rccl_primary_keys,
                                &*__rccl_primary_values,
                                __rccl_order,
                                &mut *__rccl_scratch_keys,
                                &mut *__rccl_scratch_values,
                            )
                            .map_err(::rccl_cub::DeviceAlgorithmError::Launch),
                    }
                }
            },
        ),
        _ => unreachable!("copy mode always supplies both input ranges"),
    };

    audit::generated_algorithm_expression(
        audit::Algorithm::MergeSort,
        quote! {{
            let __rccl_execution = #execution;
            #mode_bindings
            let __rccl_workspace = #workspace;
            let (__rccl_scratch_keys, __rccl_scratch_values, __rccl_partitions) =
                __rccl_workspace.__parts_mut();
            let __rccl_order = #order;
            let __rccl_len = __rccl_primary_keys.len();
            let __rccl_shape_error = #shape_error;
            if let ::core::option::Option::Some((__rccl_role, __rccl_actual)) = __rccl_shape_error {
                ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                    role: __rccl_role,
                    expected: __rccl_len,
                    actual: __rccl_actual,
                })
            } else if __rccl_len == 0 {
                if __rccl_partitions.len() == 0 {
                    ::core::result::Result::Ok(())
                } else {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: 0,
                        actual: __rccl_partitions.len(),
                    })
                }
            } else {
                let __rccl_num_tiles = __rccl_len.div_ceil(#tile_items);
                let __rccl_num_partitions = __rccl_num_tiles + 1;
                if __rccl_partitions.len() != __rccl_num_partitions {
                    ::core::result::Result::Err(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "merge-sort partitions",
                        expected: __rccl_num_partitions,
                        actual: __rccl_partitions.len(),
                    })
                } else {
                    let __rccl_num_passes = usize::BITS as usize
                        - (__rccl_num_tiles - 1).leading_zeros() as usize;
                    let mut __rccl_ping = (__rccl_num_passes & 1) == 0;
                    let __rccl_agent_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_tiles.min(u32::MAX as usize) as u32,
                            #threads as u32,
                            #shared_bytes as u32,
                        );
                    let mut __rccl_result = #initial_block;
                    let __rccl_partition_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            __rccl_num_partitions.div_ceil(256).min(u32::MAX as usize) as u32,
                            256,
                            0,
                        );
                    let mut __rccl_pass = 0usize;
                    while __rccl_pass < __rccl_num_passes && __rccl_result.is_ok() {
                        let __rccl_target_merged_tiles = 2usize << __rccl_pass;
                        __rccl_result = match __rccl_execution
                            .module()
                            .prepare_device_merge_sort_partition::<#key_type, _, #threads, #items>(
                                __rccl_partition_launch,
                            )
                        {
                            ::core::result::Result::Err(__rccl_error) => {
                                ::core::result::Result::Err(
                                    ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                )
                            }
                            ::core::result::Result::Ok(__rccl_prepared) => {
                                let __rccl_launch_result = if __rccl_ping {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_primary_keys,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                } else {
                                    __rccl_execution.module().device_merge_sort_partition::<
                                        #key_type,
                                        _,
                                        #threads,
                                        #items,
                                    >(
                                        __rccl_execution.stream(),
                                        &__rccl_prepared,
                                        &*__rccl_scratch_keys,
                                        __rccl_target_merged_tiles,
                                        __rccl_order,
                                        &mut *__rccl_partitions,
                                    )
                                };
                                __rccl_launch_result
                                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                            }
                        };
                        if __rccl_result.is_ok() {
                            __rccl_result = match __rccl_execution
                                .module()
                                .#prepare_merge::<#key_type, #value_type, _, #items, #shared_bytes>(
                                    __rccl_agent_launch,
                                )
                            {
                                ::core::result::Result::Err(__rccl_error) => {
                                    ::core::result::Result::Err(
                                        ::rccl_cub::DeviceAlgorithmError::Launch(__rccl_error),
                                    )
                                }
                                ::core::result::Result::Ok(__rccl_prepared) => {
                                    let __rccl_launch_result = if __rccl_ping {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            #value_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_primary_keys,
                                            &*__rccl_primary_values,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_scratch_keys,
                                            &mut *__rccl_scratch_values,
                                        )
                                    } else {
                                        __rccl_execution.module().#launch_merge::<
                                            #key_type,
                                            #value_type,
                                            _,
                                            #items,
                                            #shared_bytes,
                                        >(
                                            __rccl_execution.stream(),
                                            &__rccl_prepared,
                                            &*__rccl_scratch_keys,
                                            &*__rccl_scratch_values,
                                            &*__rccl_partitions,
                                            __rccl_target_merged_tiles,
                                            __rccl_order,
                                            &mut *__rccl_primary_keys,
                                            &mut *__rccl_primary_values,
                                        )
                                    };
                                    __rccl_launch_result
                                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)
                                }
                            };
                        }
                        __rccl_ping = !__rccl_ping;
                        __rccl_pass += 1;
                    }
                    __rccl_result
                }
            }
        }},
    )
}

fn expand_device_radix_sort_keys(
    call: call::DeviceRadixSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_via_buffer(call, false)
}

fn expand_device_radix_sort_workspace(
    call: call::DeviceRadixSortWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let primitive = primitive_radix(&key_type)?;
    let _ = primitive_radix_lowering(&key_type, primitive, false)?;
    require_policy800_workspace_target()?;
    let passes = primitive.bits.div_ceil(8) as usize;
    let bit_size = (primitive.bits / 8) as usize;
    let (_, single_items) = reg_bound_policy(256, 19, bit_size);
    let single_tile_items = 256usize * single_items;
    let (onesweep_threads, onesweep_items) = reg_bound_policy(384, 21, bit_size);
    let tile_items = onesweep_threads * onesweep_items;
    let execution = call.execution;
    let len = call.len;
    audit::generated_expression(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceRadixSortKeysWorkspace<#key_type>,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_len: usize = #len;
            let __rccl_uses_onesweep = __rccl_len > #single_tile_items;
            let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
            let __rccl_scratch_keys =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<#key_type>::zeroed(
                    __rccl_execution.stream(),
                    __rccl_len,
                )
                .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_bins = ::rccl_cub::__private::cuda_core::DeviceBuffer::<u64>::zeroed(
                __rccl_execution.stream(),
                if __rccl_uses_onesweep { #passes * 256usize } else { 0usize },
            )
            .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_lookback = ::rccl_cub::__private::cuda_core::DeviceBuffer::<i32>::zeroed(
                __rccl_execution.stream(),
                if __rccl_uses_onesweep { __rccl_blocks * 256usize } else { 0usize },
            )
            .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_counters = ::rccl_cub::__private::cuda_core::DeviceBuffer::<i32>::zeroed(
                __rccl_execution.stream(),
                if __rccl_uses_onesweep { #passes } else { 0usize },
            )
            .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            ::core::result::Result::Ok(::rccl_cub::DeviceRadixSortKeysWorkspace::__from_parts(
                __rccl_scratch_keys,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            ))
        })()
    }})
}

fn require_policy800_workspace_target() -> syn::Result<()> {
    let target = configured_cuda_target()?;
    if !(80..90).contains(&target.capability) {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "this workspace macro contains the pinned Policy800 layout only",
        ));
    }
    Ok(())
}

fn expand_device_radix_sort_workspace_counts(
    call: call::DeviceRadixSortWorkspaceCountsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    require_policy800_workspace_target()?;
    let primitive = primitive_radix(&call.key_type)?;
    let _ = primitive_radix_lowering(&call.key_type, primitive, false)?;
    let len = call.len;
    let passes = primitive.bits.div_ceil(8) as usize;
    let bit_size = (primitive.bits / 8) as usize;
    let (onesweep_threads, onesweep_items) = reg_bound_policy(384, 21, bit_size);
    let tile_items = onesweep_threads * onesweep_items;
    audit::generated_expression(quote! {{
        let __rccl_len: usize = #len;
        let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
        ::rccl_cub::RadixSortWorkspaceCounts {
            scratch_keys: __rccl_len,
            bins: #passes * 256usize,
            lookback: __rccl_blocks * 256usize,
            counters: #passes,
        }
    }})
}

fn expand_device_radix_sort_pairs_workspace(
    call: call::DeviceRadixSortPairsWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let value_type = call.value_type;
    let primitive = primitive_radix(&key_type)?;
    let _ = primitive_radix_lowering(&key_type, primitive, false)?;
    require_policy800_workspace_target()?;
    let key_size = tmp_primitive_size(&key_type)?;
    let value_size = tmp_primitive_size(&value_type)?;
    let passes = primitive.bits.div_ceil(8) as usize;
    let nominal_items = if key_size == 4 { 17 } else { 21 };
    let (_, single_items) = reg_bound_policy(256, 19, key_size.max(value_size));
    let single_tile_items = 256usize * single_items;
    let (onesweep_threads, onesweep_items) =
        reg_bound_policy(384, nominal_items, key_size.max(value_size));
    let tile_items = onesweep_threads * onesweep_items;
    let execution = call.execution;
    let len = call.len;
    audit::generated_expression(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceRadixSortPairsWorkspace<#key_type, #value_type>,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_len: usize = #len;
            let __rccl_uses_onesweep = __rccl_len > #single_tile_items;
            let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
            let __rccl_scratch_keys =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<#key_type>::zeroed(
                    __rccl_execution.stream(), __rccl_len,
                ).map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_scratch_values =
                ::rccl_cub::__private::cuda_core::DeviceBuffer::<#value_type>::zeroed(
                    __rccl_execution.stream(), __rccl_len,
                ).map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_bins = ::rccl_cub::__private::cuda_core::DeviceBuffer::<u64>::zeroed(
                __rccl_execution.stream(),
                if __rccl_uses_onesweep { #passes * 256usize } else { 0usize },
            ).map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_lookback = ::rccl_cub::__private::cuda_core::DeviceBuffer::<i32>::zeroed(
                __rccl_execution.stream(),
                if __rccl_uses_onesweep { __rccl_blocks * 256usize } else { 0usize },
            ).map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_counters = ::rccl_cub::__private::cuda_core::DeviceBuffer::<i32>::zeroed(
                __rccl_execution.stream(),
                if __rccl_uses_onesweep { #passes } else { 0usize },
            ).map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            ::core::result::Result::Ok(::rccl_cub::DeviceRadixSortPairsWorkspace::__from_parts(
                __rccl_scratch_keys,
                __rccl_scratch_values,
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            ))
        })()
    }})
}

fn expand_device_radix_sort_pairs_workspace_counts(
    call: call::DeviceRadixSortPairsWorkspaceCountsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    require_policy800_workspace_target()?;
    let primitive = primitive_radix(&call.key_type)?;
    let _ = primitive_radix_lowering(&call.key_type, primitive, false)?;
    let key_size = tmp_primitive_size(&call.key_type)?;
    let value_size = tmp_primitive_size(&call.value_type)?;
    let passes = primitive.bits.div_ceil(8) as usize;
    let nominal_items = if key_size == 4 { 17 } else { 21 };
    let (onesweep_threads, onesweep_items) =
        reg_bound_policy(384, nominal_items, key_size.max(value_size));
    let tile_items = onesweep_threads * onesweep_items;
    let len = call.len;
    audit::generated_expression(quote! {{
        let __rccl_len: usize = #len;
        let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
        ::rccl_cub::RadixSortPairsWorkspaceCounts {
            scratch_keys: __rccl_len,
            scratch_values: __rccl_len,
            bins: #passes * 256usize,
            lookback: __rccl_blocks * 256usize,
            counters: #passes,
        }
    }})
}

fn expand_device_radix_sort_keys_buffer_workspace(
    call: call::DeviceRadixSortKeysBufferWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let key_type = call.key_type;
    let primitive = primitive_radix(&key_type)?;
    let _ = primitive_radix_lowering(&key_type, primitive, false)?;
    require_policy800_workspace_target()?;
    let passes = primitive.bits.div_ceil(8) as usize;
    let bit_size = (primitive.bits / 8) as usize;
    let (_, single_items) = reg_bound_policy(256, 19, bit_size);
    let single_tile_items = 256usize * single_items;
    let (onesweep_threads, onesweep_items) = reg_bound_policy(384, 21, bit_size);
    let tile_items = onesweep_threads * onesweep_items;
    expand_radix_buffer_workspace(
        call.execution,
        call.len,
        passes,
        single_tile_items,
        tile_items,
    )
}

fn expand_device_radix_sort_pairs_buffer_workspace(
    call: call::DeviceRadixSortPairsBufferWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let primitive = primitive_radix(&call.key_type)?;
    let _ = primitive_radix_lowering(&call.key_type, primitive, false)?;
    require_policy800_workspace_target()?;
    let key_size = tmp_primitive_size(&call.key_type)?;
    let value_size = tmp_primitive_size(&call.value_type)?;
    let passes = primitive.bits.div_ceil(8) as usize;
    let nominal_items = if key_size == 4 { 17 } else { 21 };
    let (_, single_items) = reg_bound_policy(256, 19, key_size.max(value_size));
    let single_tile_items = 256usize * single_items;
    let (onesweep_threads, onesweep_items) =
        reg_bound_policy(384, nominal_items, key_size.max(value_size));
    let tile_items = onesweep_threads * onesweep_items;
    expand_radix_buffer_workspace(
        call.execution,
        call.len,
        passes,
        single_tile_items,
        tile_items,
    )
}

fn expand_radix_buffer_workspace(
    execution: syn::Expr,
    len: syn::Expr,
    passes: usize,
    single_tile_items: usize,
    tile_items: usize,
) -> syn::Result<proc_macro2::TokenStream> {
    audit::generated_expression(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceRadixSortBufferWorkspace,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_len: usize = #len;
            let __rccl_uses_onesweep = __rccl_len > #single_tile_items;
            let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
            let __rccl_bins = ::rccl_cub::__private::cuda_core::DeviceBuffer::<u64>::zeroed(
                __rccl_execution.stream(),
                if __rccl_uses_onesweep { #passes * 256usize } else { 0usize },
            ).map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_lookback = ::rccl_cub::__private::cuda_core::DeviceBuffer::<i32>::zeroed(
                __rccl_execution.stream(),
                if __rccl_uses_onesweep { __rccl_blocks * 256usize } else { 0usize },
            ).map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            let __rccl_counters = ::rccl_cub::__private::cuda_core::DeviceBuffer::<i32>::zeroed(
                __rccl_execution.stream(),
                if __rccl_uses_onesweep { #passes } else { 0usize },
            ).map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
            ::core::result::Result::Ok(::rccl_cub::DeviceRadixSortBufferWorkspace::__from_parts(
                __rccl_bins,
                __rccl_lookback,
                __rccl_counters,
            ))
        })()
    }})
}

fn expand_device_radix_sort_keys_descending(
    call: call::DeviceRadixSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_via_buffer(call, true)
}

fn expand_device_radix_sort_keys_buffer(
    call: call::DeviceRadixSortKeysBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_buffer_order(call, false)
}

fn expand_device_radix_sort_keys_buffer_descending(
    call: call::DeviceRadixSortKeysBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_buffer_order(call, true)
}

/// Lower CUB's `DoubleBuffer` overload without importing its pointer/selector
/// representation. The maximum policy shape is TMP; the requested bit window,
/// pass count, and selected side are ordinary runtime state.
fn expand_device_radix_sort_keys_buffer_order(
    call: call::DeviceRadixSortKeysBufferCall,
    descending: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let target = configured_cuda_target()?;
    if !(80..90).contains(&target.capability) {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "this lowering contains the exact pinned Policy800 one-sweep graph only; another architecture must add its own macro-selected CUB policy",
        ));
    }
    let primitive = primitive_radix(&call.key_type)?;
    let lowering = primitive_radix_lowering(&call.key_type, primitive, descending)?;
    let single_tile = primitive_single_tile_lowering(primitive, descending)?;
    let key_type = call.key_type;
    let bit_type = lowering.bit_type;
    let sentinel = lowering.sentinel;
    let default_key = lowering.default_key;
    let to_ordered = lowering.to_ordered;
    let from_ordered = lowering.from_ordered;
    let digit = lowering.digit;
    let rank_digit = lowering.rank_digit;
    let single_default_ordered = single_tile.default_ordered;
    let single_to_ordered = single_tile.to_ordered;
    let single_from_ordered = single_tile.from_ordered;
    let execution = call.execution;
    let buffers = call.buffers;
    let bit_range = call.bit_range;
    let workspace = call.workspace;
    let max_passes = primitive.bits.div_ceil(8) as usize;
    let histogram_kernel = format_ident!("device_radix_sort_histogram_{max_passes}");
    let prepare_histogram = format_ident!("prepare_device_radix_sort_histogram_{max_passes}_for");
    let key_bits = primitive.bits;
    let histogram_items = 16usize;
    let bit_size = (primitive.bits / 8) as usize;
    let (single_threads, single_items) = reg_bound_policy(256, 19, bit_size);
    let single_tile_items = single_threads * single_items;
    let (onesweep_threads, onesweep_items) = reg_bound_policy(384, 21, bit_size);
    let tile_items = onesweep_threads * onesweep_items;
    let union_bytes = tile_items * bit_size;
    let shared_bytes = union_bytes + 256usize * core::mem::size_of::<u64>();
    if ![34_304usize, 32_768usize].contains(&shared_bytes) {
        return Err(syn::Error::new_spanned(
            &key_type,
            "the selected radix key policy has no exact generated static shared capability",
        ));
    }
    let onesweep_kernel = format_ident!("device_radix_sort_onesweep_keys_{shared_bytes}");
    let prepare_onesweep =
        format_ident!("prepare_device_radix_sort_onesweep_keys_{shared_bytes}_for");
    let portion_size = ((1usize << 28) - 1) / tile_items * tile_items;
    audit::generated_algorithm_expression(
        audit::Algorithm::RadixSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_buffers = #buffers;
            let __rccl_bit_range: ::core::ops::Range<u32> = #bit_range;
            let __rccl_workspace = #workspace;
            let (__rccl_bins, __rccl_lookback, __rccl_counters) =
                __rccl_workspace.__parts_mut();
            let __rccl_len = __rccl_buffers.len();
            (|| -> ::core::result::Result<(), ::rccl_cub::DeviceAlgorithmError> {
                let __rccl_begin_bit = __rccl_bit_range.start;
                let __rccl_end_bit = __rccl_bit_range.end;
                if __rccl_begin_bit > __rccl_end_bit || __rccl_end_bit > #key_bits {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::InvalidBitRange {
                            begin: __rccl_begin_bit,
                            end: __rccl_end_bit,
                            key_bits: #key_bits,
                        },
                    );
                }
                if __rccl_len > #portion_size {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "Policy800 one-portion maximum",
                            expected: #portion_size,
                            actual: __rccl_len,
                        },
                    );
                }
                let __rccl_passes =
                    ((__rccl_end_bit - __rccl_begin_bit) as usize).div_ceil(8usize);
                if __rccl_len == 0 || __rccl_passes == 0 {
                    return ::core::result::Result::Ok(());
                }
                if __rccl_len <= #single_tile_items {
                    let __rccl_single_default_ordered = #single_default_ordered;
                    let __rccl_single_to_ordered = #single_to_ordered;
                    let __rccl_single_from_ordered = #single_from_ordered;
                    let __rccl_single_rank_digit = #rank_digit;
                    let __rccl_single_key_witness: #key_type = 0 as #key_type;
                    let __rccl_single_bit_witness: #bit_type = #sentinel;
                    let __rccl_single_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            1,
                            #single_threads as u32,
                            0,
                        );
                    let __rccl_single = __rccl_execution
                        .module()
                        .prepare_device_radix_sort_single_tile_keys_for::<
                            #key_type,
                            #bit_type,
                            _,
                            _,
                            _,
                            _,
                            #single_items,
                            #descending,
                        >(
                            &__rccl_single_key_witness,
                            &__rccl_single_bit_witness,
                            &__rccl_single_default_ordered,
                            &__rccl_single_to_ordered,
                            &__rccl_single_from_ordered,
                            &__rccl_single_rank_digit,
                            __rccl_single_launch,
                        )
                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                    let __rccl_single_result = {
                        let __rccl_current_is_first = __rccl_buffers.current_is_first();
                        let (__rccl_first, __rccl_second) =
                            __rccl_buffers.first_and_second_mut();
                        if __rccl_current_is_first {
                            __rccl_execution
                                .module()
                                .device_radix_sort_single_tile_keys::<
                                    #key_type, #bit_type, _, _, _, _, #single_items, #descending,
                                >(
                                    __rccl_execution.stream(),
                                    &__rccl_single,
                                    &**__rccl_first,
                                    &mut **__rccl_second,
                                    __rccl_begin_bit,
                                    __rccl_end_bit,
                                    __rccl_single_default_ordered,
                                    __rccl_single_to_ordered,
                                    __rccl_single_from_ordered,
                                    __rccl_single_rank_digit,
                                )
                        } else {
                            __rccl_execution
                                .module()
                                .device_radix_sort_single_tile_keys::<
                                    #key_type, #bit_type, _, _, _, _, #single_items, #descending,
                                >(
                                    __rccl_execution.stream(),
                                    &__rccl_single,
                                    &**__rccl_second,
                                    &mut **__rccl_first,
                                    __rccl_begin_bit,
                                    __rccl_end_bit,
                                    __rccl_single_default_ordered,
                                    __rccl_single_to_ordered,
                                    __rccl_single_from_ordered,
                                    __rccl_single_rank_digit,
                                )
                        }
                    };
                    __rccl_single_result
                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                    __rccl_buffers.swap();
                    return ::core::result::Result::Ok(());
                }
                let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
                let __rccl_required_bins = __rccl_passes * 256usize;
                if __rccl_bins.len() < __rccl_required_bins {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::WorkspaceTooSmall {
                            role: "radix bins",
                            required: __rccl_required_bins,
                            actual: __rccl_bins.len(),
                        },
                    );
                }
                if __rccl_lookback.len() != __rccl_blocks * 256usize {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "radix lookback",
                            expected: __rccl_blocks * 256usize,
                            actual: __rccl_lookback.len(),
                        },
                    );
                }
                if __rccl_counters.len() < __rccl_passes {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::WorkspaceTooSmall {
                            role: "radix pass counters",
                            required: __rccl_passes,
                            actual: __rccl_counters.len(),
                        },
                    );
                }

                let __rccl_init_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_passes as u32,
                        256,
                        0,
                    );
                let __rccl_init = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_init(__rccl_init_launch)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .device_radix_sort_init(
                        __rccl_execution.stream(),
                        &__rccl_init,
                        &mut *__rccl_counters,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_default_key = #default_key;
                let __rccl_to_ordered = #to_ordered;
                let __rccl_from_ordered = #from_ordered;
                let __rccl_digit = #digit;
                let __rccl_key_witness: #key_type = 0 as #key_type;
                let __rccl_bit_witness: #bit_type = #sentinel;
                let __rccl_histogram_probe_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        1,
                        128,
                        0,
                    );
                let __rccl_histogram_probe = __rccl_execution
                    .module()
                    .#prepare_histogram::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                    >(
                        &__rccl_key_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_digit,
                        __rccl_histogram_probe_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                let __rccl_histogram_blocks_per_sm = __rccl_histogram_probe
                    .function()
                    .max_active_blocks_per_multiprocessor(128, 0)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
                let __rccl_multiprocessors = __rccl_execution
                    .stream()
                    .context()
                    .multiprocessor_count()
                    .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
                let __rccl_histogram_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_histogram_blocks_per_sm * __rccl_multiprocessors,
                        128,
                        0,
                    );
                let __rccl_histogram = __rccl_execution
                    .module()
                    .#prepare_histogram::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                    >(
                        &__rccl_key_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_digit,
                        __rccl_histogram_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .#histogram_kernel::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                    >(
                        __rccl_execution.stream(),
                        &__rccl_histogram,
                        &**__rccl_buffers.first(),
                        __rccl_begin_bit,
                        __rccl_end_bit,
                        __rccl_default_key,
                        __rccl_to_ordered,
                        __rccl_digit,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_scan_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_passes as u32,
                        256,
                        0,
                    );
                let __rccl_scan = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_exclusive_sum::<#max_passes>(__rccl_scan_launch)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .device_radix_sort_exclusive_sum::<#max_passes>(
                        __rccl_execution.stream(),
                        &__rccl_scan,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_onesweep_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_blocks as u32,
                        #onesweep_threads as u32,
                        0,
                    );
                let __rccl_onesweep = __rccl_execution
                    .module()
                    .#prepare_onesweep::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        _,
                        #onesweep_items,
                    >(
                        &__rccl_key_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_from_ordered,
                        &__rccl_digit,
                        __rccl_onesweep_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                let mut __rccl_pass = 0usize;
                while __rccl_pass < __rccl_passes {
                    ::rccl_cub::__private::ZeroRadixLookback::zero_radix_lookback(
                        &mut *__rccl_lookback,
                        __rccl_execution.stream(),
                    )?;
                    let __rccl_bin_offset = __rccl_pass * 256usize;
                    let __rccl_current_bit =
                        __rccl_begin_bit + (__rccl_pass * 8usize) as u32;
                    let __rccl_num_bits =
                        (__rccl_end_bit - __rccl_current_bit).min(8u32);
                    let __rccl_pass_result = {
                        let __rccl_current_is_first = __rccl_buffers.current_is_first();
                        let (__rccl_first, __rccl_second) =
                            __rccl_buffers.first_and_second_mut();
                        if __rccl_current_is_first {
                            __rccl_execution
                                .module()
                                .#onesweep_kernel::<
                                    #key_type, #bit_type, _, _, _, _, #onesweep_items,
                                >(
                                    __rccl_execution.stream(),
                                    &__rccl_onesweep,
                                    &mut *__rccl_lookback,
                                    &mut *__rccl_counters,
                                    &*__rccl_bins,
                                    &**__rccl_first,
                                    &mut **__rccl_second,
                                    __rccl_bin_offset,
                                    __rccl_pass,
                                    __rccl_current_bit,
                                    __rccl_num_bits,
                                    __rccl_default_key,
                                    __rccl_to_ordered,
                                    __rccl_from_ordered,
                                    __rccl_digit,
                                )
                        } else {
                            __rccl_execution
                                .module()
                                .#onesweep_kernel::<
                                    #key_type, #bit_type, _, _, _, _, #onesweep_items,
                                >(
                                    __rccl_execution.stream(),
                                    &__rccl_onesweep,
                                    &mut *__rccl_lookback,
                                    &mut *__rccl_counters,
                                    &*__rccl_bins,
                                    &**__rccl_second,
                                    &mut **__rccl_first,
                                    __rccl_bin_offset,
                                    __rccl_pass,
                                    __rccl_current_bit,
                                    __rccl_num_bits,
                                    __rccl_default_key,
                                    __rccl_to_ordered,
                                    __rccl_from_ordered,
                                    __rccl_digit,
                                )
                        }
                    };
                    __rccl_pass_result
                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                    __rccl_buffers.swap();
                    __rccl_pass += 1usize;
                }
                ::core::result::Result::Ok(())
            })()
        }},
    )
}

fn expand_device_radix_sort_keys_via_buffer(
    call: call::DeviceRadixSortKeysCall,
    descending: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let primitive = primitive_radix(&call.key_type)?;
    let key_bits = primitive.bits;
    let call::DeviceRadixSortKeysCall {
        key_type,
        execution,
        keys,
        workspace,
    } = call;
    let operation = expand_device_radix_sort_keys_buffer_order(
        call::DeviceRadixSortKeysBufferCall {
            key_type: key_type.clone(),
            execution: syn::parse_quote!(__rccl_execution),
            buffers: syn::parse_quote!(&mut __rccl_buffers),
            bit_range: syn::parse_quote!(0u32..#key_bits),
            workspace: syn::parse_quote!(__rccl_controls),
        },
        descending,
    )?;
    audit::generated_algorithm_expression(
        audit::Algorithm::RadixSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_keys = #keys;
            let __rccl_workspace = #workspace;
            let (__rccl_scratch_keys, __rccl_controls) = __rccl_workspace.__parts_mut();
            let __rccl_len = __rccl_keys.len();
            (|| -> ::core::result::Result<(), ::rccl_cub::DeviceAlgorithmError> {
                let (__rccl_result, __rccl_current_is_first) = {
                    let __rccl_scratch_len = __rccl_scratch_keys.len();
                    let mut __rccl_buffers = ::rccl_cub::DoubleBuffer::new(
                        &mut *__rccl_keys,
                        &mut *__rccl_scratch_keys,
                    )
                    .ok_or(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "radix scratch keys",
                        expected: __rccl_len,
                        actual: __rccl_scratch_len,
                    })?;
                    let __rccl_result = #operation;
                    let __rccl_current_is_first = __rccl_buffers.current_is_first();
                    (__rccl_result, __rccl_current_is_first)
                };
                __rccl_result?;
                if !__rccl_current_is_first {
                    ::rccl_cub::__private::copy_device_range::<#key_type, _, _>(
                        &*__rccl_scratch_keys,
                        &mut *__rccl_keys,
                        __rccl_execution.stream(),
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Memory)?;
                }
                ::core::result::Result::Ok(())
            })()
        }},
    )
}

fn expand_device_radix_sort_pairs(
    call: call::DeviceRadixSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_via_buffer(call, false)
}

fn expand_device_radix_sort_pairs_descending(
    call: call::DeviceRadixSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_via_buffer(call, true)
}

fn expand_device_radix_sort_pairs_buffer(
    call: call::DeviceRadixSortPairsBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_buffer_order(call, false)
}

fn expand_device_radix_sort_pairs_buffer_descending(
    call: call::DeviceRadixSortPairsBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_buffer_order(call, true)
}

fn expand_device_radix_sort_pairs_buffer_order(
    call: call::DeviceRadixSortPairsBufferCall,
    descending: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let target = configured_cuda_target()?;
    if !(80..90).contains(&target.capability) {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "this lowering contains the exact pinned Policy800 one-sweep graph only; another architecture must add its own macro-selected CUB policy",
        ));
    }
    let primitive = primitive_radix(&call.key_type)?;
    let lowering = primitive_radix_lowering(&call.key_type, primitive, descending)?;
    let single_tile = primitive_single_tile_lowering(primitive, descending)?;
    let key_type = call.key_type;
    let value_type = call.value_type;
    let key_size = tmp_primitive_size(&key_type)?;
    let value_size = tmp_primitive_size(&value_type)?;
    let bit_type = lowering.bit_type;
    let sentinel = lowering.sentinel;
    let default_key = lowering.default_key;
    let to_ordered = lowering.to_ordered;
    let from_ordered = lowering.from_ordered;
    let digit = lowering.digit;
    let rank_digit = lowering.rank_digit;
    let single_default_ordered = single_tile.default_ordered;
    let single_to_ordered = single_tile.to_ordered;
    let single_from_ordered = single_tile.from_ordered;
    let execution = call.execution;
    let buffers = call.buffers;
    let bit_range = call.bit_range;
    let workspace = call.workspace;
    let max_passes = primitive.bits.div_ceil(8) as usize;
    let histogram_kernel = format_ident!("device_radix_sort_histogram_{max_passes}");
    let prepare_histogram = format_ident!("prepare_device_radix_sort_histogram_{max_passes}_for");
    let key_bits = primitive.bits;
    let histogram_items = 16usize;
    let nominal_items = if key_size == 4 { 17 } else { 21 };
    let (single_threads, single_items) = reg_bound_policy(256, 19, key_size.max(value_size));
    let single_tile_items = single_threads * single_items;
    let (onesweep_threads, onesweep_items) =
        reg_bound_policy(384, nominal_items, key_size.max(value_size));
    let tile_items = onesweep_threads * onesweep_items;
    let union_bytes = tile_items * key_size.max(value_size);
    let shared_bytes = union_bytes + 256usize * core::mem::size_of::<u64>();
    if ![28_160usize, 26_624usize, 32_768usize].contains(&shared_bytes) {
        return Err(syn::Error::new_spanned(
            &key_type,
            "the selected radix pair policy has no exact generated static shared capability",
        ));
    }
    let onesweep_kernel = format_ident!("device_radix_sort_onesweep_pairs_{shared_bytes}");
    let prepare_onesweep =
        format_ident!("prepare_device_radix_sort_onesweep_pairs_{shared_bytes}_for");
    let portion_size = ((1usize << 28) - 1) / tile_items * tile_items;
    audit::generated_algorithm_expression(
        audit::Algorithm::RadixSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_buffers = #buffers;
            let __rccl_bit_range: ::core::ops::Range<u32> = #bit_range;
            let __rccl_workspace = #workspace;
            let (__rccl_bins, __rccl_lookback, __rccl_counters) =
                __rccl_workspace.__parts_mut();
            let __rccl_len = __rccl_buffers.len();
            (|| -> ::core::result::Result<(), ::rccl_cub::DeviceAlgorithmError> {
                let __rccl_begin_bit = __rccl_bit_range.start;
                let __rccl_end_bit = __rccl_bit_range.end;
                if __rccl_begin_bit > __rccl_end_bit || __rccl_end_bit > #key_bits {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::InvalidBitRange {
                            begin: __rccl_begin_bit,
                            end: __rccl_end_bit,
                            key_bits: #key_bits,
                        },
                    );
                }
                if __rccl_len > #portion_size {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "Policy800 one-portion maximum",
                            expected: #portion_size,
                            actual: __rccl_len,
                        },
                    );
                }
                let __rccl_passes =
                    ((__rccl_end_bit - __rccl_begin_bit) as usize).div_ceil(8usize);
                if __rccl_len == 0 || __rccl_passes == 0 {
                    return ::core::result::Result::Ok(());
                }
                if __rccl_len <= #single_tile_items {
                    let __rccl_single_default_ordered = #single_default_ordered;
                    let __rccl_single_to_ordered = #single_to_ordered;
                    let __rccl_single_from_ordered = #single_from_ordered;
                    let __rccl_single_rank_digit = #rank_digit;
                    let __rccl_single_key_witness: #key_type = 0 as #key_type;
                    let __rccl_single_value_witness: #value_type =
                        ::core::default::Default::default();
                    let __rccl_single_bit_witness: #bit_type = #sentinel;
                    let __rccl_single_launch =
                        ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                            1,
                            #single_threads as u32,
                            0,
                        );
                    let __rccl_single = __rccl_execution
                        .module()
                        .prepare_device_radix_sort_single_tile_pairs_for::<
                            #key_type,
                            #value_type,
                            #bit_type,
                            _,
                            _,
                            _,
                            _,
                            #single_items,
                            #descending,
                        >(
                            &__rccl_single_key_witness,
                            &__rccl_single_value_witness,
                            &__rccl_single_bit_witness,
                            &__rccl_single_default_ordered,
                            &__rccl_single_to_ordered,
                            &__rccl_single_from_ordered,
                            &__rccl_single_rank_digit,
                            __rccl_single_launch,
                        )
                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                    let __rccl_single_result = {
                        let __rccl_current_is_first = __rccl_buffers.current_is_first();
                        let (__rccl_first, __rccl_second) =
                            __rccl_buffers.first_and_second_mut();
                        let (__rccl_first_keys, __rccl_first_values) = __rccl_first;
                        let (__rccl_second_keys, __rccl_second_values) = __rccl_second;
                        if __rccl_current_is_first {
                            __rccl_execution
                                .module()
                                .device_radix_sort_single_tile_pairs::<
                                    #key_type, #value_type, #bit_type, _, _, _, _, #single_items,
                                    #descending,
                                >(
                                    __rccl_execution.stream(),
                                    &__rccl_single,
                                    &**__rccl_first_keys,
                                    &**__rccl_first_values,
                                    &mut **__rccl_second_keys,
                                    &mut **__rccl_second_values,
                                    __rccl_begin_bit,
                                    __rccl_end_bit,
                                    __rccl_single_default_ordered,
                                    __rccl_single_to_ordered,
                                    __rccl_single_from_ordered,
                                    __rccl_single_rank_digit,
                                )
                        } else {
                            __rccl_execution
                                .module()
                                .device_radix_sort_single_tile_pairs::<
                                    #key_type, #value_type, #bit_type, _, _, _, _, #single_items,
                                    #descending,
                                >(
                                    __rccl_execution.stream(),
                                    &__rccl_single,
                                    &**__rccl_second_keys,
                                    &**__rccl_second_values,
                                    &mut **__rccl_first_keys,
                                    &mut **__rccl_first_values,
                                    __rccl_begin_bit,
                                    __rccl_end_bit,
                                    __rccl_single_default_ordered,
                                    __rccl_single_to_ordered,
                                    __rccl_single_from_ordered,
                                    __rccl_single_rank_digit,
                                )
                        }
                    };
                    __rccl_single_result
                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                    __rccl_buffers.swap();
                    return ::core::result::Result::Ok(());
                }
                let __rccl_blocks = __rccl_len.div_ceil(#tile_items);
                let __rccl_required_bins = __rccl_passes * 256usize;
                if __rccl_bins.len() < __rccl_required_bins {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::WorkspaceTooSmall {
                            role: "radix bins",
                            required: __rccl_required_bins,
                            actual: __rccl_bins.len(),
                        },
                    );
                }
                if __rccl_lookback.len() != __rccl_blocks * 256usize {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "radix lookback",
                            expected: __rccl_blocks * 256usize,
                            actual: __rccl_lookback.len(),
                        },
                    );
                }
                if __rccl_counters.len() < __rccl_passes {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::WorkspaceTooSmall {
                            role: "radix pass counters",
                            required: __rccl_passes,
                            actual: __rccl_counters.len(),
                        },
                    );
                }

                let __rccl_init_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_passes as u32,
                        256,
                        0,
                    );
                let __rccl_init = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_init(__rccl_init_launch)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .device_radix_sort_init(
                        __rccl_execution.stream(),
                        &__rccl_init,
                        &mut *__rccl_counters,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_default_key = #default_key;
                let __rccl_to_ordered = #to_ordered;
                let __rccl_from_ordered = #from_ordered;
                let __rccl_digit = #digit;
                let __rccl_key_witness: #key_type = 0 as #key_type;
                let __rccl_value_witness: #value_type = ::core::default::Default::default();
                let __rccl_bit_witness: #bit_type = #sentinel;
                let __rccl_histogram_probe_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        1,
                        128,
                        0,
                    );
                let __rccl_histogram_probe = __rccl_execution
                    .module()
                    .#prepare_histogram::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                    >(
                        &__rccl_key_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_digit,
                        __rccl_histogram_probe_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                let __rccl_histogram_blocks_per_sm = __rccl_histogram_probe
                    .function()
                    .max_active_blocks_per_multiprocessor(128, 0)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
                let __rccl_multiprocessors = __rccl_execution
                    .stream()
                    .context()
                    .multiprocessor_count()
                    .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
                let __rccl_histogram_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_histogram_blocks_per_sm * __rccl_multiprocessors,
                        128,
                        0,
                    );
                let __rccl_histogram = __rccl_execution
                    .module()
                    .#prepare_histogram::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                    >(
                        &__rccl_key_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_digit,
                        __rccl_histogram_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .#histogram_kernel::<
                        #key_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        #histogram_items,
                    >(
                        __rccl_execution.stream(),
                        &__rccl_histogram,
                        &**&__rccl_buffers.first().0,
                        __rccl_begin_bit,
                        __rccl_end_bit,
                        __rccl_default_key,
                        __rccl_to_ordered,
                        __rccl_digit,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_scan_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_passes as u32,
                        256,
                        0,
                    );
                let __rccl_scan = __rccl_execution
                    .module()
                    .prepare_device_radix_sort_exclusive_sum::<#max_passes>(__rccl_scan_launch)
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                __rccl_execution
                    .module()
                    .device_radix_sort_exclusive_sum::<#max_passes>(
                        __rccl_execution.stream(),
                        &__rccl_scan,
                        &mut *__rccl_bins,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;

                let __rccl_onesweep_launch =
                    ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                        __rccl_blocks as u32,
                        #onesweep_threads as u32,
                        0,
                    );
                let __rccl_onesweep = __rccl_execution
                    .module()
                    .#prepare_onesweep::<
                        #key_type,
                        #value_type,
                        #bit_type,
                        _,
                        _,
                        _,
                        _,
                        #onesweep_items,
                    >(
                        &__rccl_key_witness,
                        &__rccl_value_witness,
                        &__rccl_bit_witness,
                        &__rccl_default_key,
                        &__rccl_to_ordered,
                        &__rccl_from_ordered,
                        &__rccl_digit,
                        __rccl_onesweep_launch,
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                let mut __rccl_pass = 0usize;
                while __rccl_pass < __rccl_passes {
                    ::rccl_cub::__private::ZeroRadixLookback::zero_radix_lookback(
                        &mut *__rccl_lookback,
                        __rccl_execution.stream(),
                    )?;
                    let __rccl_bin_offset = __rccl_pass * 256usize;
                    let __rccl_current_bit =
                        __rccl_begin_bit + (__rccl_pass * 8usize) as u32;
                    let __rccl_num_bits =
                        (__rccl_end_bit - __rccl_current_bit).min(8u32);
                    let __rccl_pass_result = {
                        let __rccl_current_is_first = __rccl_buffers.current_is_first();
                        let (__rccl_first, __rccl_second) =
                            __rccl_buffers.first_and_second_mut();
                        let (__rccl_first_keys, __rccl_first_values) = __rccl_first;
                        let (__rccl_second_keys, __rccl_second_values) = __rccl_second;
                        if __rccl_current_is_first {
                            __rccl_execution
                                .module()
                                .#onesweep_kernel::<
                                    #key_type, #value_type, #bit_type, _, _, _, _, #onesweep_items,
                                >(
                                    __rccl_execution.stream(),
                                    &__rccl_onesweep,
                                    &mut *__rccl_lookback,
                                    &mut *__rccl_counters,
                                    &*__rccl_bins,
                                    &**__rccl_first_keys,
                                    &**__rccl_first_values,
                                    &mut **__rccl_second_keys,
                                    &mut **__rccl_second_values,
                                    __rccl_bin_offset,
                                    __rccl_pass,
                                    __rccl_current_bit,
                                    __rccl_num_bits,
                                    __rccl_default_key,
                                    __rccl_to_ordered,
                                    __rccl_from_ordered,
                                    __rccl_digit,
                                )
                        } else {
                            __rccl_execution
                                .module()
                                .#onesweep_kernel::<
                                    #key_type, #value_type, #bit_type, _, _, _, _, #onesweep_items,
                                >(
                                    __rccl_execution.stream(),
                                    &__rccl_onesweep,
                                    &mut *__rccl_lookback,
                                    &mut *__rccl_counters,
                                    &*__rccl_bins,
                                    &**__rccl_second_keys,
                                    &**__rccl_second_values,
                                    &mut **__rccl_first_keys,
                                    &mut **__rccl_first_values,
                                    __rccl_bin_offset,
                                    __rccl_pass,
                                    __rccl_current_bit,
                                    __rccl_num_bits,
                                    __rccl_default_key,
                                    __rccl_to_ordered,
                                    __rccl_from_ordered,
                                    __rccl_digit,
                                )
                        }
                    };
                    __rccl_pass_result
                        .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
                    __rccl_buffers.swap();
                    __rccl_pass += 1usize;
                }
                ::core::result::Result::Ok(())
            })()
        }},
    )
}

fn expand_device_radix_sort_pairs_via_buffer(
    call: call::DeviceRadixSortPairsCall,
    descending: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let primitive = primitive_radix(&call.key_type)?;
    let key_bits = primitive.bits;
    let call::DeviceRadixSortPairsCall {
        key_type,
        value_type,
        execution,
        keys,
        values,
        workspace,
    } = call;
    let operation = expand_device_radix_sort_pairs_buffer_order(
        call::DeviceRadixSortPairsBufferCall {
            key_type: key_type.clone(),
            value_type: value_type.clone(),
            execution: syn::parse_quote!(__rccl_execution),
            buffers: syn::parse_quote!(&mut __rccl_buffers),
            bit_range: syn::parse_quote!(0u32..#key_bits),
            workspace: syn::parse_quote!(__rccl_controls),
        },
        descending,
    )?;
    audit::generated_algorithm_expression(
        audit::Algorithm::RadixSort,
        quote! {{
            let __rccl_execution = #execution;
            let __rccl_keys = #keys;
            let __rccl_values = #values;
            let __rccl_workspace = #workspace;
            let (__rccl_scratch_keys, __rccl_scratch_values, __rccl_controls) =
                __rccl_workspace.__parts_mut();
            let __rccl_len = __rccl_keys.len();
            (|| -> ::core::result::Result<(), ::rccl_cub::DeviceAlgorithmError> {
                for (__rccl_role, __rccl_actual) in [
                    ("radix values", __rccl_values.len()),
                    ("radix scratch keys", __rccl_scratch_keys.len()),
                    ("radix scratch values", __rccl_scratch_values.len()),
                ] {
                    if __rccl_actual != __rccl_len {
                        return ::core::result::Result::Err(
                            ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                                role: __rccl_role,
                                expected: __rccl_len,
                                actual: __rccl_actual,
                            },
                        );
                    }
                }
                let (__rccl_result, __rccl_current_is_first) = {
                    let mut __rccl_buffers = ::rccl_cub::DoubleBuffer::paired(
                        (&mut *__rccl_keys, &mut *__rccl_values),
                        (&mut *__rccl_scratch_keys, &mut *__rccl_scratch_values),
                    )
                    .ok_or(::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                        role: "radix key/value buffers",
                        expected: __rccl_len,
                        actual: __rccl_len,
                    })?;
                    let __rccl_result = #operation;
                    let __rccl_current_is_first = __rccl_buffers.current_is_first();
                    (__rccl_result, __rccl_current_is_first)
                };
                __rccl_result?;
                if !__rccl_current_is_first {
                    ::rccl_cub::__private::copy_device_range::<#key_type, _, _>(
                        &*__rccl_scratch_keys,
                        &mut *__rccl_keys,
                        __rccl_execution.stream(),
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Memory)?;
                    ::rccl_cub::__private::copy_device_range::<#value_type, _, _>(
                        &*__rccl_scratch_values,
                        &mut *__rccl_values,
                        __rccl_execution.stream(),
                    )
                    .map_err(::rccl_cub::DeviceAlgorithmError::Memory)?;
                }
                ::core::result::Result::Ok(())
            })()
        }},
    )
}

fn expand_warp_reduce(call: call::WarpReduceCall) -> syn::Result<proc_macro2::TokenStream> {
    let value_type = call.value_type;
    let scalar = warp_scalar(&value_type)?;
    let width = call.logical_warp_threads;
    let value = call.input;
    let is_addition = call::is_addition_closure(&call.op);
    let op = type_binary_closure(call.op, &value_type);
    let is_builtin_sum = call.valid_items.is_none() && is_addition && scalar.redux_sum;
    let reduction = if is_builtin_sum {
        warp_redux_sum(scalar, &value_type, &width)
    } else {
        warp_shuffle_reduction(scalar, &value_type, &width, call.valid_items.as_ref())
    };
    audit::generated_expression(quote! {{
        let __rccl_op = #op;
        let mut __rccl_aggregate: #value_type = #value;
        #reduction
    }})
}

fn expand_warp_reduce_items(
    call: call::WarpReduceItemsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    let value_type = call.value_type;
    let scalar = warp_scalar(&value_type)?;
    let width = call.logical_warp_threads;
    let items_per_thread = call.items_per_thread;
    let items = call.items;
    let is_addition = call::is_addition_closure(&call.op);
    let op = type_binary_closure(call.op, &value_type);
    let reduction = if is_addition && scalar.redux_sum {
        warp_redux_sum(scalar, &value_type, &width)
    } else {
        warp_shuffle_reduction(scalar, &value_type, &width, None)
    };
    audit::generated_expression(quote! {{
        let __rccl_op = #op;
        let __rccl_items: [#value_type; #items_per_thread] = #items;
        let mut __rccl_aggregate: #value_type = __rccl_items[0];
        let mut __rccl_item = 1usize;
        while __rccl_item < #items_per_thread {
            __rccl_aggregate = __rccl_op(__rccl_aggregate, __rccl_items[__rccl_item]);
            __rccl_item += 1;
        }
        #reduction
    }})
}

fn type_binary_closure(mut operation: syn::Expr, value_type: &syn::Type) -> syn::Expr {
    if let syn::Expr::Closure(closure) = &mut operation {
        for input in &mut closure.inputs {
            if !matches!(input, syn::Pat::Type(_)) {
                let pattern = input.clone();
                *input = syn::Pat::Type(syn::PatType {
                    attrs: Vec::new(),
                    pat: Box::new(pattern),
                    colon_token: Default::default(),
                    ty: Box::new(value_type.clone()),
                });
            }
        }
    }
    operation
}

#[derive(Clone, Copy)]
struct WarpScalar {
    kind: WarpScalarKind,
    redux_sum: bool,
}

#[derive(Clone, Copy)]
enum WarpScalarKind {
    Unsigned32,
    Signed32,
    Unsigned64,
    Signed64,
    Float32,
    Float64,
}

fn warp_scalar(value_type: &syn::Type) -> syn::Result<WarpScalar> {
    let semantic_type = ungroup_type(value_type);
    let syn::Type::Path(path) = semantic_type else {
        return Err(syn::Error::new_spanned(
            value_type,
            "WarpReduce needs a macro-visible primitive scalar type",
        ));
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return Err(syn::Error::new_spanned(
            value_type,
            "WarpReduce aliases and custom values need explicit macro-visible shuffle metadata",
        ));
    }
    let (kind, redux_sum) = match path.path.segments[0].ident.to_string().as_str() {
        "u8" | "u16" | "u32" => (WarpScalarKind::Unsigned32, true),
        "i8" | "i16" | "i32" => (WarpScalarKind::Signed32, true),
        "u64" => (WarpScalarKind::Unsigned64, false),
        "i64" => (WarpScalarKind::Signed64, false),
        "f32" => (WarpScalarKind::Float32, false),
        "f64" => (WarpScalarKind::Float64, false),
        _ => {
            return Err(syn::Error::new_spanned(
                value_type,
                "WarpReduce aliases and custom values need explicit macro-visible shuffle metadata",
            ));
        }
    };
    Ok(WarpScalar { kind, redux_sum })
}

fn ungroup_type(value_type: &syn::Type) -> &syn::Type {
    match value_type {
        syn::Type::Group(group) => ungroup_type(&group.elem),
        syn::Type::Paren(paren) => ungroup_type(&paren.elem),
        other => other,
    }
}

fn warp_redux_sum(
    scalar: WarpScalar,
    value_type: &syn::Type,
    width: &syn::LitInt,
) -> proc_macro2::TokenStream {
    let result = match scalar.kind {
        WarpScalarKind::Unsigned32 => quote! {
            ::rccl_cub::__private::cuda_device::warp::redux_sync_add(
                __rccl_mask,
                __rccl_aggregate as u32,
            ) as #value_type
        },
        WarpScalarKind::Signed32 => quote! {
            ::rccl_cub::__private::cuda_device::warp::redux_sync_add(
                __rccl_mask,
                __rccl_aggregate as i32 as u32,
            ) as i32 as #value_type
        },
        WarpScalarKind::Unsigned64
        | WarpScalarKind::Signed64
        | WarpScalarKind::Float32
        | WarpScalarKind::Float64 => unreachable!("redux_sum is false for this scalar"),
    };
    quote! {{
        if false {
            let _: #value_type = __rccl_op(__rccl_aggregate, __rccl_aggregate);
        }
        let __rccl_first_lane =
            ::rccl_cub::__private::cuda_device::warp::lane_id() & !(#width - 1);
        let __rccl_mask = if #width == 32 {
            u32::MAX
        } else {
            ((1u32 << #width) - 1) << __rccl_first_lane
        };
        if #width == 1 {
            __rccl_aggregate
        } else {
            #result
        }
    }}
}

fn warp_shuffle_reduction(
    scalar: WarpScalar,
    value_type: &syn::Type,
    width: &syn::LitInt,
    valid_items: Option<&syn::Expr>,
) -> proc_macro2::TokenStream {
    let shuffled = warp_shuffle_down(scalar, value_type);
    let combine = if let Some(valid_items) = valid_items {
        quote! {
            let __rccl_valid_items = #valid_items;
            let __rccl_rank =
                ::rccl_cub::__private::cuda_device::cooperative_groups::ThreadGroup::thread_rank(
                    &__rccl_tile,
                );
            if __rccl_rank < __rccl_valid_items
                && __rccl_rank + __rccl_offset < __rccl_valid_items
            {
                __rccl_aggregate = __rccl_op(__rccl_aggregate, __rccl_other);
            }
        }
    } else {
        quote! {
            __rccl_aggregate = __rccl_op(__rccl_aggregate, __rccl_other);
        }
    };
    quote! {{
        let __rccl_tile =
            ::rccl_cub::__private::cuda_device::cooperative_groups::this_thread_block()
                .tiled_partition::<#width>();
        let mut __rccl_offset = 1u32;
        while __rccl_offset < #width {
            let __rccl_other: #value_type = #shuffled;
            #combine
            __rccl_offset <<= 1;
        }
        __rccl_aggregate
    }}
}

fn warp_shuffle_down(scalar: WarpScalar, value_type: &syn::Type) -> proc_macro2::TokenStream {
    match scalar.kind {
        WarpScalarKind::Unsigned32 | WarpScalarKind::Signed32 => quote! {
            ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down(
                &__rccl_tile,
                __rccl_aggregate as u32,
                __rccl_offset,
            ) as #value_type
        },
        WarpScalarKind::Unsigned64 | WarpScalarKind::Signed64 => quote! {{
            let __rccl_bits = __rccl_aggregate as u64;
            let __rccl_low =
                ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down(
                    &__rccl_tile,
                    __rccl_bits as u32,
                    __rccl_offset,
                ) as u64;
            let __rccl_high =
                ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down(
                    &__rccl_tile,
                    (__rccl_bits >> 32) as u32,
                    __rccl_offset,
                ) as u64;
            (__rccl_low | (__rccl_high << 32)) as #value_type
        }},
        WarpScalarKind::Float32 => quote! {
            ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down_f32(
                &__rccl_tile,
                __rccl_aggregate,
                __rccl_offset,
            )
        },
        WarpScalarKind::Float64 => quote! {{
            let __rccl_bits = __rccl_aggregate.to_bits();
            let __rccl_low =
                ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down(
                    &__rccl_tile,
                    __rccl_bits as u32,
                    __rccl_offset,
                ) as u64;
            let __rccl_high =
                ::rccl_cub::__private::cuda_device::cooperative_groups::WarpCollective::shfl_down(
                    &__rccl_tile,
                    (__rccl_bits >> 32) as u32,
                    __rccl_offset,
                ) as u64;
            f64::from_bits(__rccl_low | (__rccl_high << 32))
        }},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_program_owns_only_the_exact_scoped_cuda_oxide_bind() {
        let module: syn::ItemMod = syn::parse_quote! {
            mod kernels {
                use super::*;

                #[kernel]
                #[launch_contract(domain = 1, block = (32, 1, 1))]
                pub fn reduce(values: &mut [u32]) {
                    values[0] = 1;
                }
            }
        };
        let generated = expand_user_program(module)
            .expect("a crate-root user kernel program must lower")
            .to_string();
        assert!(generated.contains("cuda_device :: cuda_module"));
        assert!(generated.contains("macro_rules ! __rccl_cub_program"));
        assert!(generated.contains("$ crate :: kernels :: load (__rccl_context)"));
        assert!(generated.contains("use __rccl_cub_program as program"));
        assert_eq!(generated.matches("unsafe").count(), 1);
        assert!(!generated.contains("Arc"));
        assert!(!generated.contains("* mut"));
    }

    #[test]
    fn user_program_rejects_duplicate_or_uninspectable_module_ownership() {
        let duplicate: syn::ItemMod = syn::parse_quote! {
            #[cuda_module]
            mod kernels {
                #[kernel]
                fn reduce() {}
            }
        };
        assert!(
            expand_user_program(duplicate)
                .expect_err("the RCCL attribute must be the only module owner")
                .to_string()
                .contains("already supplies")
        );

        let external: syn::ItemMod = syn::parse_quote! { mod kernels; };
        assert!(
            expand_user_program(external)
                .expect_err("file-backed contents are invisible to the proc macro")
                .to_string()
                .contains("inline module")
        );
    }

    #[test]
    fn typed_catalog_is_complete_and_valid() {
        catalog::validate().expect("the typed CUB compiler catalog must be valid");
        assert_eq!(catalog::TARGETS.len(), 6);
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
        let call: call::WarpReduceCall = syn::parse_quote! {
            ::<i32, 32>(value, |left, right| left + right)
        };
        let output = expand_warp_reduce(call).expect("the operation must lower");
        let text = output.to_string();
        assert!(text.contains("warp :: redux_sync_add"));
        assert!(text.contains("let mut __rccl_aggregate : i32"));
        assert!(!text.contains("WarpReduceValue"));
        assert!(!text.contains("TypeId"));
        assert!(!text.contains("unsafe"));
        assert!(!text.contains("* mut"));
    }

    #[test]
    fn device_transform_lowering_owns_kernel_and_launch_rust() {
        let kernels = expand_device_kernels(syn::parse_quote!(__test_kernels))
            .expect("CUB must generate the transform kernel");
        let kernel_text = kernels.to_string();
        assert!(kernel_text.contains("transform_repeat_in_place"));
        assert!(kernel_text.contains("grid_stride_runs32"));
        assert!(kernel_text.contains("allow (clippy :: too_many_arguments)"));

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

    #[test]
    fn radix_default_key_is_macro_specialization_not_runtime_data() {
        let kernels = expand_device_kernels(syn::parse_quote!(__test_kernels))
            .expect("CUB must generate the radix kernels")
            .to_string();
        assert!(kernels.contains("DefaultKey : Fn () -> B + Copy"));
        assert!(kernels.contains("default_key : DefaultKey"));
        assert_eq!(
            kernels.matches("launch_bounds (256 , 1)").count(),
            2,
            "both single-tile entries must retain CUB's one-block launch bound"
        );
        assert_eq!(
            kernels.matches("33_856usize , 32usize").count(),
            2,
            "both single-tile entries must materialize BlockScan's exact static extent and alignment"
        );
        assert_eq!(
            kernels.matches("static_block_shared_memory !").count(),
            10,
            "every single-tile, histogram, exclusive-sum, and exact one-sweep specialization must materialize CUB's static shared union through the audited Rust capability"
        );
        assert!(!kernels.contains("sentinel : Uniform < B >"));

        let operation = expand_device_radix_sort_keys(syn::parse_quote! {
            ::<i32>(
                &exec,
                &mut keys,
                &mut workspace
            )
        })
        .expect("the macro must specialize the default key")
        .to_string();
        assert!(
            operation
                .contains("let __rccl_default_key = :: rccl_cub :: __private :: radix_default_u32")
        );
        assert!(operation.contains("& __rccl_default_key"));
        assert!(!operation.contains("let __rccl_default_key = ||"));
        assert!(!operation.contains("sentinel"));
    }

    #[test]
    fn radix_keys_emit_the_exact_policy800_single_tile_and_onesweep_graph() {
        let call: call::DeviceRadixSortKeysCall = syn::parse_quote! {
            ::<i32>(
                &exec,
                &mut keys,
                &mut workspace
            )
        };
        let operation = expand_device_radix_sort_keys(call)
            .expect("the pinned Policy800 key graph must lower")
            .to_string();
        assert!(operation.contains("prepare_device_radix_sort_init"));
        assert!(operation.contains("prepare_device_radix_sort_histogram"));
        assert!(operation.contains("prepare_device_radix_sort_exclusive_sum"));
        assert!(operation.contains("prepare_device_radix_sort_onesweep_keys"));
        assert!(operation.contains("prepare_device_radix_sort_single_tile_keys"));
        assert!(operation.contains(
            "prepare_device_radix_sort_histogram_4_for :: < i32 , u32 , _ , _ , _ , 16usize , >"
        ));
        assert!(
            operation
                .contains("device_radix_sort_histogram_4 :: < i32 , u32 , _ , _ , _ , 16usize , >")
        );
        assert!(operation.contains(
            "prepare_device_radix_sort_onesweep_keys_34304_for :: < i32 , u32 , _ , _ , _ , _ , 21usize"
        ));
        assert!(operation.contains("let __rccl_single_rank_digit ="));
        assert!(!operation.contains("let __rccl_rank_digit ="));
        assert!(operation.contains("radix_rank_digit_u32"));
        assert!(!operation.contains("| __rccl_ordered"));
        assert!(operation.contains("LaunchConfig1D :: new (__rccl_passes as u32 , 256 , 0"));
        assert!(operation.contains("max_active_blocks_per_multiprocessor (128 , 0"));
        assert!(
            !operation
                .contains("device_radix_sort_histogram_4 :: < i32 , u32 , _ , _ , _ , 21usize")
        );
        assert_eq!(operation.matches("zero_radix_lookback").count(), 1);
        assert!(operation.contains("copy_device_range :: < i32 , _ , _ >"));
        assert!(!operation.contains("core :: mem :: swap"));
        assert!(!operation.contains("core :: mem :: drop (__rccl_buffers)"));
        assert!(operation.contains("let (__rccl_result , __rccl_current_is_first) ="));
        assert!(operation.contains("while __rccl_pass < __rccl_passes"));
        assert!(operation.contains("let __rccl_bin_offset = __rccl_pass * 256usize"));
        assert!(
            operation.contains(
                "let __rccl_current_bit = __rccl_begin_bit + (__rccl_pass * 8usize) as u32"
            )
        );
        assert_eq!(
            operation
                .matches("device_radix_sort_onesweep_keys_34304 ::")
                .count(),
            2,
            "heterogeneous Rust ranges require one typed launch branch per selector state"
        );
        assert!(operation.contains("DoubleBuffer :: new"));
        assert!(operation.contains("first_and_second_mut"));
        assert!(operation.contains("__rccl_buffers . swap"));
        assert!(!operation.contains("if (__rccl_pass & 1usize) == 0"));
        assert!(!operation.contains("device_merge_sort"));
        assert!(!operation.contains("DeviceBuffer :: zeroed"));
        assert!(!operation.contains("compute_capability"));
        assert!(!operation.contains("__rccl_len as u32"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));
    }

    #[test]
    fn radix_key_buffer_keeps_bit_window_and_each_route_selection_runtime() {
        let operation = expand_device_radix_sort_keys_buffer(syn::parse_quote! {
            ::<u32>(
                &exec,
                &mut buffers,
                begin_bit..end_bit,
                &mut workspace
            )
        })
        .expect("the DoubleBuffer bit-window graph must lower")
        .to_string();

        assert!(operation.contains("Range < u32 >"));
        assert!(operation.contains("let __rccl_passes ="));
        assert!(operation.contains("div_ceil (8usize)"));
        assert!(operation.contains("__rccl_passes as u32"));
        assert!(operation.contains("__rccl_begin_bit , __rccl_end_bit"));
        assert!(operation.contains("min (8u32)"));
        assert!(operation.contains("first_and_second_mut"));
        assert_eq!(
            operation
                .matches("device_radix_sort_onesweep_keys_34304 ::")
                .count(),
            2
        );
        assert_eq!(
            operation
                .matches("device_radix_sort_single_tile_keys ::")
                .count(),
            2
        );
        assert_eq!(operation.matches("__rccl_buffers . swap").count(), 2);
        let single_begin = operation
            .find("if __rccl_len <= 4864usize")
            .expect("Policy800 u32 single-tile threshold must be macro materialized");
        let one_sweep_begin = operation[single_begin..]
            .find("let __rccl_blocks")
            .map(|offset| single_begin + offset)
            .expect("one-sweep workspace checks must follow the single-tile route");
        let single_route = &operation[single_begin..one_sweep_begin];
        assert!(single_route.contains("device_radix_sort_single_tile_keys ::"));
        assert!(single_route.contains("return :: core :: result :: Result :: Ok"));
        for masked_workspace in [
            "__rccl_bins . len",
            "__rccl_lookback . len",
            "__rccl_counters . len",
        ] {
            assert!(
                !single_route.contains(masked_workspace),
                "single-tile route must not require unused one-sweep workspace: {masked_workspace}"
            );
        }
        let launch_check = operation
            .rfind("map_err (:: rccl_cub :: DeviceAlgorithmError :: Launch) ?")
            .expect("launch failure must be propagated");
        let swap = operation
            .rfind("__rccl_buffers . swap")
            .expect("a successful pass must update selection");
        assert!(launch_check < swap);
        assert!(operation.contains("InvalidBitRange"));
        assert!(operation.contains("WorkspaceTooSmall"));
        assert!(!operation.contains("#max_passes"));
        assert!(!operation.contains("selector"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));
    }

    #[test]
    fn opaque_radix_workspace_preserves_the_single_tile_boundary() {
        let workspace = expand_device_radix_sort_workspace(syn::parse_quote! {
            ::<u32>(&exec, len)
        })
        .expect("the typed key workspace must lower")
        .to_string();
        assert!(workspace.contains("DeviceRadixSortKeysWorkspace"));
        assert!(workspace.contains("let __rccl_uses_onesweep = __rccl_len > 4864usize"));
        assert!(workspace.contains("if __rccl_uses_onesweep"));
        assert!(workspace.contains("else { 0usize }"));
        assert!(workspace.contains("DeviceBuffer :: < u32 > :: zeroed"));

        let buffer_workspace = expand_device_radix_sort_keys_buffer_workspace(syn::parse_quote! {
            ::<u32>(&exec, len)
        })
        .expect("the opaque DoubleBuffer workspace must lower")
        .to_string();
        assert!(buffer_workspace.contains("DeviceRadixSortBufferWorkspace"));
        assert!(!buffer_workspace.contains("DeviceBuffer :: < u32 > :: zeroed"));
        assert!(buffer_workspace.contains("DeviceBuffer :: < u64 > :: zeroed"));
    }

    #[test]
    fn radix_policy800_reg_bound_scaling_is_macro_selected_for_key_width() {
        assert_eq!(reg_bound_policy(384, 21, 4), (384, 21));
        assert_eq!(reg_bound_policy(384, 21, 8), (384, 10));
        assert_eq!(reg_bound_policy(256, 19, 4), (256, 19));
        assert_eq!(reg_bound_policy(256, 19, 8), (256, 9));

        let call: call::DeviceRadixSortKeysCall = syn::parse_quote! {
            ::<u64>(
                &exec,
                &mut keys,
                &mut workspace
            )
        };
        let operation = expand_device_radix_sort_keys(call)
            .expect("the pinned 64-bit Policy800 graph must lower")
            .to_string();
        assert!(operation.contains(
            "prepare_device_radix_sort_onesweep_keys_32768_for :: < u64 , u64 , _ , _ , _ , _ , 10usize"
        ));
        assert!(operation.contains("let __rccl_blocks = __rccl_len . div_ceil (3840usize)"));
        assert!(operation.contains("if __rccl_len <= 2304usize"));
        assert!(operation.contains(
            "prepare_device_radix_sort_single_tile_keys_for :: < u64 , u64 , _ , _ , _ , _ , 9usize , false"
        ));
        assert!(operation.contains("while __rccl_pass < __rccl_passes"));
        assert!(
            operation.contains("LaunchConfig1D :: new (__rccl_blocks as u32 , 384usize as u32")
        );
        assert!(operation.contains("prepare_device_radix_sort_histogram_8_for"));
        assert!(operation.contains("max_active_blocks_per_multiprocessor (128 , 0"));
        assert!(!operation.contains("compute_capability"));

        let workspace_call: call::DeviceRadixSortWorkspaceCountsCall =
            syn::parse_quote! { ::<u64>(len) };
        let workspace = expand_device_radix_sort_workspace_counts(workspace_call)
            .expect("the 64-bit workspace must use the same macro-selected tile")
            .to_string();
        assert!(workspace.contains("let __rccl_blocks = __rccl_len . div_ceil (3840usize)"));
    }

    #[test]
    fn radix_pairs_emit_the_exact_policy800_pair_graph_and_workspace() {
        let call: call::DeviceRadixSortPairsCall = syn::parse_quote! {
            ::<f32, i32>(
                &exec,
                &mut keys,
                &mut values,
                &mut workspace
            )
        };
        let operation = expand_device_radix_sort_pairs(call)
            .expect("the pinned Policy800 pair graph must lower")
            .to_string();
        assert!(operation.contains("prepare_device_radix_sort_histogram"));
        assert!(operation.contains("prepare_device_radix_sort_exclusive_sum"));
        assert!(operation.contains(
            "prepare_device_radix_sort_onesweep_pairs_28160_for :: < f32 , i32 , u32 , _ , _ , _ , _ , 17usize"
        ));
        assert!(operation.contains("let __rccl_blocks = __rccl_len . div_ceil (6528usize)"));
        assert!(operation.contains("__rccl_blocks as u32 , 384usize as u32 , 0"));
        assert!(!operation.contains("28160usize as u32"));
        assert!(operation.contains("radix_float_digit_masked_u32"));
        assert!(operation.contains("radix_float_rank_digit_u32"));
        assert!(!operation.contains("let __rccl_rank_digit ="));
        assert!(!operation.contains("| __rccl_ordered"));
        assert_eq!(
            operation
                .matches("device_radix_sort_onesweep_pairs_28160 ::")
                .count(),
            2
        );
        assert!(operation.contains("DoubleBuffer :: paired"));
        assert!(operation.contains("first_and_second_mut"));
        assert_eq!(operation.matches("copy_device_range").count(), 2);
        assert!(!operation.contains("core :: mem :: drop (__rccl_buffers)"));
        assert!(operation.contains("let (__rccl_result , __rccl_current_is_first) ="));
        assert!(!operation.contains("device_merge_sort"));
        assert!(!operation.contains("DeviceBuffer :: zeroed"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));

        let workspace = expand_device_radix_sort_pairs_workspace_counts(syn::parse_quote! {
            ::<f32, i32>(len)
        })
        .expect("pair workspace must use the same macro-selected tile")
        .to_string();
        assert!(workspace.contains("scratch_values : __rccl_len"));
        assert!(workspace.contains("div_ceil (6528usize)"));
    }

    #[test]
    fn radix_pairs_select_each_exact_policy800_static_union() {
        let u32_u64 = expand_device_radix_sort_pairs(syn::parse_quote! {
            ::<u32, u64>(
                &exec,
                &mut keys,
                &mut values,
                &mut workspace
            )
        })
        .expect("the pinned u32/u64 pair case must select its exact static union")
        .to_string();
        assert!(u32_u64.contains("prepare_device_radix_sort_onesweep_pairs_26624_for"));
        assert!(u32_u64.contains("device_radix_sort_onesweep_pairs_26624 ::"));
        assert!(u32_u64.contains("div_ceil (3072usize)"));

        let u64_u64 = expand_device_radix_sort_pairs(syn::parse_quote! {
            ::<u64, u64>(
                &exec,
                &mut keys,
                &mut values,
                &mut workspace
            )
        })
        .expect("a supported 64-bit key/value pair must not be rejected by a 32-bit extent")
        .to_string();
        assert!(u64_u64.contains("prepare_device_radix_sort_onesweep_pairs_32768_for"));
        assert!(u64_u64.contains("device_radix_sort_onesweep_pairs_32768 ::"));
        assert!(u64_u64.contains("div_ceil (3840usize)"));
    }

    #[test]
    fn radix_pair_buffer_has_one_structural_selector_per_pinned_route() {
        let operation = expand_device_radix_sort_pairs_buffer(syn::parse_quote! {
            ::<u32, i32>(
                &exec,
                &mut buffers,
                3u32..19u32,
                &mut workspace
            )
        })
        .expect("the paired DoubleBuffer graph must lower")
        .to_string();

        assert!(operation.contains("first_and_second_mut"));
        assert!(operation.contains("let (__rccl_first_keys , __rccl_first_values)"));
        assert!(operation.contains("let (__rccl_second_keys , __rccl_second_values)"));
        assert_eq!(
            operation
                .matches("device_radix_sort_onesweep_pairs_28160 ::")
                .count(),
            2
        );
        assert_eq!(
            operation
                .matches("device_radix_sort_single_tile_pairs ::")
                .count(),
            2
        );
        assert_eq!(operation.matches("__rccl_buffers . swap").count(), 2);
        let single_begin = operation
            .find("if __rccl_len <= 4864usize")
            .expect("u32/i32 dominant type keeps the Policy800 single-tile extent");
        let one_sweep_begin = operation[single_begin..]
            .find("let __rccl_blocks")
            .map(|offset| single_begin + offset)
            .expect("one-sweep graph must remain after the single-tile route");
        let single_route = &operation[single_begin..one_sweep_begin];
        assert!(single_route.contains("device_radix_sort_single_tile_pairs ::"));
        assert!(single_route.contains("__rccl_buffers . swap"));
        assert!(single_route.contains("return :: core :: result :: Result :: Ok"));
        assert!(!operation.contains("key_buffers"));
        assert!(!operation.contains("value_buffers"));
        assert!(!operation.contains("selector"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));
    }

    #[test]
    fn merge_sort_keys_emit_the_exact_block_partition_merge_topology() {
        let call: call::DeviceMergeSortKeysCall = syn::parse_quote! {
            ::<u32>(
                &exec,
                &mut keys,
                &mut workspace,
                |left: &u32, right: &u32| left.cmp(right)
            )
        };
        let operation = expand_device_merge_sort_keys(call)
            .expect("the pinned SM60+ merge-sort policy must lower")
            .to_string();
        assert!(operation.contains("prepare_device_merge_sort_block_keys_256"));
        assert!(operation.contains("device_merge_sort_block_keys_in_place_256"));
        assert!(
            operation.contains(
                "prepare_device_merge_sort_partition :: < u32 , _ , 256usize , 17usize >"
            )
        );
        assert!(operation.contains("prepare_device_merge_sort_merge_keys_256"));
        assert!(operation.contains("17424usize as u32"));
        assert!(!operation.contains("DeviceBuffer :: zeroed"));
        assert!(!operation.contains("device_merge_sort_keys_pass"));
        assert!(!operation.contains("compute_capability"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));

        let count: call::DeviceMergeSortPartitionCountCall = syn::parse_quote! { ::<u32>(len) };
        let count = expand_device_merge_sort_partition_count(count)
            .expect("partition storage must use the same macro-selected tile")
            .to_string();
        assert!(count.contains("div_ceil (4352usize)"));

        let padded_call: call::DeviceMergeSortKeysCall = syn::parse_quote! {
            ::<u64>(
                &exec,
                &mut keys,
                &mut workspace,
                |left: &u64, right: &u64| left.cmp(right)
            )
        };
        let padded = expand_device_merge_sort_keys(padded_call)
            .expect("u64 must lower through the padded Policy600 BlockExchange path")
            .to_string();
        assert!(
            padded
                .contains("prepare_device_merge_sort_partition :: < u64 , _ , 256usize , 8usize >")
        );
        assert!(padded.contains("16912usize as u32"));
    }

    #[test]
    fn merge_sort_pairs_emit_the_exact_agents_and_borrow_all_workspace() {
        let call: call::DeviceMergeSortPairsCall = syn::parse_quote! {
            ::<u32, u32>(
                &exec,
                &mut keys,
                &mut values,
                &mut workspace,
                |left: &u32, right: &u32| left.cmp(right)
            )
        };
        let operation = expand_device_merge_sort_pairs(call)
            .expect("the pinned key/value graph must lower")
            .to_string();
        assert!(operation.contains("prepare_device_merge_sort_block_pairs_256"));
        assert!(operation.contains("device_merge_sort_block_pairs_in_place_256"));
        assert!(operation.contains("prepare_device_merge_sort_partition"));
        assert!(operation.contains("prepare_device_merge_sort_merge_pairs_256"));
        assert!(operation.contains("17424usize as u32"));
        assert!(!operation.contains("DeviceBuffer :: zeroed"));
        assert!(!operation.contains("device_merge_sort_pairs_pass"));
        assert!(!operation.contains("compute_capability"));
        assert!(!operation.contains("unsafe"));
        assert!(!operation.contains("* mut"));
    }

    #[test]
    fn merge_tmp_layout_selects_exact_cccl_nominal_items_and_aliases_fail_closed() {
        let keys: call::DeviceMergeKeysCall = syn::parse_quote! {
            ::<u32>(
                &exec, &first, &second, &mut workspace, &mut output,
                |left, right| left.cmp(right)
            )
        };
        let key_text = expand_device_merge_keys(keys)
            .expect("primitive layout must select the pinned merge policies")
            .to_string();
        assert!(key_text.contains("prepare_device_partition_merge_path :: < u32 , _ , 15usize >"));
        assert!(key_text.contains(
            "prepare_device_merge_keys_agent :: < u32 , _ , 15usize , 4usize , false , 30736usize >"
        ));
        assert!(key_text.contains("30736usize as u32"));
        assert!(!key_text.contains("prepare_device_merge_keys ::"));
        assert!(!key_text.contains("compute_capability"));

        let sm52 = tmp_cuda_target(&syn::parse_quote!(sm_52))
            .expect("the internal policy table must retain SM52");
        let sm52_key_items = nominal_4b_items(13, 4);
        let sm52_key_load = merge_load_to_shared_policy(sm52, 4, None).0;
        assert_eq!(sm52_key_items, 13);
        assert!(!sm52_key_load);
        assert_eq!(
            merge_shared_bytes(512, sm52_key_items, 4, None, sm52_key_load, false),
            26_640
        );

        let pairs: call::DeviceMergePairsCall = syn::parse_quote! {
            ::<i32, u64>(
                &exec,
                &first_keys,
                &first_values,
                &second_keys,
                &second_values,
                &mut workspace,
                &mut output_keys,
                &mut output_values,
                |left, right| left.cmp(right),
            )
        };
        let pair_text = expand_device_merge_pairs(pairs)
            .expect("combined key/value size must be TMP input")
            .to_string();
        assert!(pair_text.contains("prepare_device_partition_merge_path :: < i32 , _ , 5usize >"));
        assert!(pair_text.contains(
            "prepare_device_merge_pairs_agent :: < i32 , u64 , _ , 5usize , 4usize , 8usize , false , false , 20496usize >"
        ));
        assert!(pair_text.contains("20496usize as u32"));

        let async_keys: call::DeviceMergeKeysCall = syn::parse_quote! {
            ::<u8>(
                &exec, &first, &second, &mut workspace, &mut output,
                |left, right| left.cmp(right)
            )
        };
        let async_key_text = expand_device_merge_keys(async_keys)
            .expect("the SM80 byte-key policy must select BlockLoadToShared")
            .to_string();
        assert!(async_key_text.contains(
            "prepare_device_merge_keys_agent :: < u8 , _ , 15usize , 1usize , true , 7760usize >"
        ));
        assert!(async_key_text.contains("7760usize as u32"));

        let async_pairs: call::DeviceMergePairsCall = syn::parse_quote! {
            ::<i32, u8>(
                &exec,
                &first_keys,
                &first_values,
                &second_keys,
                &second_values,
                &mut workspace,
                &mut output_keys,
                &mut output_values,
                |left, right| left.cmp(right),
            )
        };
        let async_pair_text = expand_device_merge_pairs(async_pairs)
            .expect("the SM80 i32/u8 policy must stage both arrays through shared memory")
            .to_string();
        assert!(async_pair_text.contains(
            "prepare_device_merge_pairs_agent :: < i32 , u8 , _ , 12usize , 4usize , 1usize , true , true , 24656usize >"
        ));
        assert!(async_pair_text.contains("24656usize as u32"));

        let sm52_pair_items = nominal_4b_items(13, 2 + 4);
        let (sm52_key_load, sm52_value_load) = merge_load_to_shared_policy(sm52, 2, Some(4));
        assert_eq!(sm52_pair_items, 8);
        assert_eq!((sm52_key_load, sm52_value_load), (false, false));
        assert_eq!(
            merge_shared_bytes(
                512,
                sm52_pair_items,
                2,
                Some(4),
                sm52_key_load,
                sm52_value_load,
            ),
            16_912
        );
        let alias: call::DeviceMergeKeysCall = syn::parse_quote! {
            ::<Key>(
                &exec, &first, &second, &mut workspace, &mut output,
                |left, right| left.cmp(right)
            )
        };
        let error = expand_device_merge_keys(alias)
            .expect_err("an opaque alias must not silently receive an invented CUB schedule");
        assert!(error.to_string().contains("macro-visible size/alignment"));
    }
}
