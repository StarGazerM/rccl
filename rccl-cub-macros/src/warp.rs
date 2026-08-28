use crate::call;
use crate::syntax::ungroup_type;
use quote::quote;

pub(super) fn expand_warp_reduce(
    call: call::WarpReduceCall,
) -> syn::Result<proc_macro2::TokenStream> {
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
    Ok(quote! {{
        let __rccl_op = #op;
        let mut __rccl_aggregate: #value_type = #value;
        #reduction
    }})
}

pub(super) fn expand_warp_reduce_items(
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
    Ok(quote! {{
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
