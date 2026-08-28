use crate::call;
use crate::target::{configured_cuda_target, tmp_primitive_size};
use quote::{format_ident, quote};

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
pub(super) fn reg_bound_policy(
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

pub(super) fn expand_device_radix_sort_keys(
    call: call::DeviceRadixSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_via_buffer(call, false)
}

pub(super) fn expand_device_radix_sort_workspace(
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
    Ok(quote! {{
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

pub(super) fn expand_device_radix_sort_workspace_counts(
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
    Ok(quote! {{
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

pub(super) fn expand_device_radix_sort_pairs_workspace(
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
    Ok(quote! {{
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

pub(super) fn expand_device_radix_sort_pairs_workspace_counts(
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
    Ok(quote! {{
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

pub(super) fn expand_device_radix_sort_keys_buffer_workspace(
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

pub(super) fn expand_device_radix_sort_pairs_buffer_workspace(
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

pub(super) fn expand_radix_buffer_workspace(
    execution: syn::Expr,
    len: syn::Expr,
    passes: usize,
    single_tile_items: usize,
    tile_items: usize,
) -> syn::Result<proc_macro2::TokenStream> {
    Ok(quote! {{
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

pub(super) fn expand_device_radix_sort_keys_descending(
    call: call::DeviceRadixSortKeysCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_via_buffer(call, true)
}

pub(super) fn expand_device_radix_sort_keys_buffer(
    call: call::DeviceRadixSortKeysBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_buffer_order(call, false)
}

pub(super) fn expand_device_radix_sort_keys_buffer_descending(
    call: call::DeviceRadixSortKeysBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_keys_buffer_order(call, true)
}

/// Lower CUB's `DoubleBuffer` overload without importing its pointer/selector
/// representation. The maximum policy shape is TMP; the requested bit window,
/// pass count, and selected side are ordinary runtime state.
pub(super) fn expand_device_radix_sort_keys_buffer_order(
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
    Ok(quote! {{
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
    }})
}

pub(super) fn expand_device_radix_sort_keys_via_buffer(
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
    Ok(quote! {{
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
    }})
}

pub(super) fn expand_device_radix_sort_pairs(
    call: call::DeviceRadixSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_via_buffer(call, false)
}

pub(super) fn expand_device_radix_sort_pairs_descending(
    call: call::DeviceRadixSortPairsCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_via_buffer(call, true)
}

pub(super) fn expand_device_radix_sort_pairs_buffer(
    call: call::DeviceRadixSortPairsBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_buffer_order(call, false)
}

pub(super) fn expand_device_radix_sort_pairs_buffer_descending(
    call: call::DeviceRadixSortPairsBufferCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_device_radix_sort_pairs_buffer_order(call, true)
}

pub(super) fn expand_device_radix_sort_pairs_buffer_order(
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
    Ok(quote! {{
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
    }})
}

pub(super) fn expand_device_radix_sort_pairs_via_buffer(
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
    Ok(quote! {{
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
    }})
}
