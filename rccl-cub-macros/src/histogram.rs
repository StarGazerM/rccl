use quote::{format_ident, quote};

use crate::{
    call,
    syntax::ungroup_type,
    target::{CudaTarget, configured_cuda_target, tmp_primitive_size},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scalar {
    Signed(u32),
    Unsigned(u32),
    Float(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HistogramPolicy {
    threads: usize,
    pixels: usize,
    rle: bool,
}

fn scalar(ty: &syn::Type) -> syn::Result<Scalar> {
    let syn::Type::Path(path) = ungroup_type(ty) else {
        return Err(syn::Error::new_spanned(
            ty,
            "DeviceHistogram currently requires a primitive arithmetic type",
        ));
    };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return Err(syn::Error::new_spanned(
            ty,
            "DeviceHistogram currently requires a primitive arithmetic type",
        ));
    }
    let ident = path.path.segments[0].ident.to_string();
    match ident.as_str() {
        "i8" => Ok(Scalar::Signed(8)),
        "i16" => Ok(Scalar::Signed(16)),
        "i32" => Ok(Scalar::Signed(32)),
        "i64" => Ok(Scalar::Signed(64)),
        "u8" => Ok(Scalar::Unsigned(8)),
        "u16" => Ok(Scalar::Unsigned(16)),
        "u32" => Ok(Scalar::Unsigned(32)),
        "u64" => Ok(Scalar::Unsigned(64)),
        "f32" => Ok(Scalar::Float(32)),
        "f64" => Ok(Scalar::Float(64)),
        _ => Err(syn::Error::new_spanned(
            ty,
            "DeviceHistogram supports i8/u8/i16/u16/i32/u32/i64/u64/f32/f64",
        )),
    }
}

fn policy(
    sample_type: &syn::Type,
    counter_type: &syn::Type,
    active_channels: usize,
    channels: usize,
    even: bool,
) -> syn::Result<HistogramPolicy> {
    policy_for(
        configured_cuda_target()?,
        sample_type,
        counter_type,
        active_channels,
        channels,
        even,
    )
}

fn policy_for(
    target: CudaTarget,
    sample_type: &syn::Type,
    counter_type: &syn::Type,
    active_channels: usize,
    channels: usize,
    even: bool,
) -> syn::Result<HistogramPolicy> {
    let sample_size = tmp_primitive_size(sample_type)?;
    let counter_size = tmp_primitive_size(counter_type)?;
    if target.capability >= 100
        && channels == 1
        && active_channels == 1
        && counter_size == 4
        && sample_size == 1
    {
        return Ok(if even {
            HistogramPolicy {
                threads: 928,
                pixels: 12,
                rle: false,
            }
        } else {
            HistogramPolicy {
                threads: 448,
                pixels: 12,
                rle: false,
            }
        });
    }
    if target.capability >= 90 && channels == 1 && active_channels == 1 && counter_size == 4 {
        if sample_size == 1 {
            return Ok(HistogramPolicy {
                threads: 768,
                pixels: 12,
                rle: false,
            });
        }
        if sample_size == 2 {
            return Ok(HistogramPolicy {
                threads: 960,
                pixels: 10,
                rle: true,
            });
        }
    }
    let sample_scale = sample_size.div_ceil(core::mem::size_of::<i32>());
    Ok(HistogramPolicy {
        threads: 384,
        pixels: (16 / active_channels / sample_scale).max(1),
        rle: true,
    })
}

fn counter_supported(counter_type: &syn::Type) -> syn::Result<()> {
    match scalar(counter_type)? {
        Scalar::Signed(32 | 64) | Scalar::Unsigned(32 | 64) => Ok(()),
        _ => Err(syn::Error::new_spanned(
            counter_type,
            "CUDA histogram atomic counters must be i32/u32/i64/u64",
        )),
    }
}

fn common_bindings(call: &call::DeviceHistogramWorkspaceCall) -> syn::Result<CommonExpansion> {
    counter_supported(&call.counter_type)?;
    scalar(&call.sample_type)?;
    scalar(&call.level_type)?;
    let channels = call.channels.base10_parse::<usize>()?;
    let active = call.active_channels.base10_parse::<usize>()?;
    Ok(CommonExpansion {
        sample_type: call.sample_type.clone(),
        counter_type: call.counter_type.clone(),
        level_type: call.level_type.clone(),
        channels,
        active,
        execution: call.execution.clone(),
        samples: call.samples.clone(),
        region: call.region.clone(),
        output: call.output.clone(),
    })
}

struct CommonExpansion {
    sample_type: syn::Type,
    counter_type: syn::Type,
    level_type: syn::Type,
    channels: usize,
    active: usize,
    execution: syn::Expr,
    samples: syn::Expr,
    region: syn::Expr,
    output: syn::Expr,
}

#[derive(Clone, Copy)]
struct HistogramLaunchShape<'a> {
    sample_type: &'a syn::Type,
    counter_type: &'a syn::Type,
    level_type: &'a syn::Type,
    channels: usize,
    active: usize,
    policy: HistogramPolicy,
}

fn sweep_name(even: bool, arithmetic: bool, threads: usize) -> syn::Ident {
    let input = if arithmetic { "_arithmetic" } else { "" };
    format_ident!(
        "{}{}_sweep_{}",
        if even {
            "device_histogram_even"
        } else {
            "device_histogram_range"
        },
        input,
        threads
    )
}

fn shared_bytes(counter_size: usize, active: usize, shared: bool) -> usize {
    let counters = if shared { 257 } else { 1 };
    // CUB's TempStorage follows the privatized counters with `int tile_idx`
    // and the one-byte empty `BlockLoadDirect::TempStorage` union member.
    let raw = active * counters * counter_size + core::mem::size_of::<i32>() + 1;
    raw.div_ceil(counter_size) * counter_size
}

fn workspace_branch(
    common: &CommonExpansion,
    even: bool,
    arithmetic: bool,
    shared: bool,
    policy: HistogramPolicy,
) -> syn::Result<proc_macro2::TokenStream> {
    let CommonExpansion {
        sample_type,
        counter_type,
        level_type,
        channels,
        active,
        ..
    } = common;
    let kernel = sweep_name(even, arithmetic, policy.threads);
    let prepare_kernel = format_ident!("prepare_{kernel}");
    let threads = policy.threads;
    let pixels = policy.pixels;
    let rle = policy.rle;
    let byte = tmp_primitive_size(sample_type)? == 1;
    let bytes = shared_bytes(tmp_primitive_size(counter_type)?, *active, shared);
    let generics = quote!(::<
        #sample_type,
        #counter_type,
        #level_type,
        #channels,
        #active,
        #pixels,
        #rle,
        #byte,
        #shared,
        #bytes
    >);
    Ok(quote! {{
        let __rccl_probe_launch = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
            1,
            #threads as u32,
            #bytes as u32,
        );
        let __rccl_probe = __rccl_execution.module().#prepare_kernel #generics(
            __rccl_probe_launch,
        ).map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
        let __rccl_blocks_per_sm = __rccl_probe
            .function()
            .max_active_blocks_per_multiprocessor(#threads as u32, #bytes as u32)
            .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
        let __rccl_sms = __rccl_execution
            .stream()
            .context()
            .multiprocessor_count()
            .map_err(::rccl_cub::DeviceAlgorithmError::Driver)?;
        let __rccl_occupancy = (__rccl_blocks_per_sm as usize)
            .checked_mul(__rccl_sms as usize)
            .ok_or(::rccl_cub::DeviceAlgorithmError::SizeOverflow)?;
        let __rccl_tile_pixels = #threads * #pixels;
        let __rccl_tiles_per_row = __rccl_num_row_pixels.div_ceil(__rccl_tile_pixels);
        let __rccl_blocks_per_row = __rccl_occupancy.min(__rccl_tiles_per_row);
        let __rccl_blocks_per_col = if __rccl_blocks_per_row == 0 {
            0usize
        } else {
            (__rccl_occupancy / __rccl_blocks_per_row).min(__rccl_num_rows)
        };
        ::rccl_cub::DeviceHistogramWorkspace::<#counter_type, #active>::__new(
            __rccl_execution.stream(),
            __rccl_privatized_lengths,
            __rccl_blocks_per_row,
            __rccl_blocks_per_col,
            (
                __rccl_num_row_pixels,
                __rccl_num_rows,
                __rccl_row_stride_samples,
            ),
        )
    }})
}

fn expand_workspace(
    call: call::DeviceHistogramWorkspaceCall,
    even: bool,
    arithmetic: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let common = common_bindings(&call)?;
    let policy = policy(
        &common.sample_type,
        &common.counter_type,
        common.active,
        common.channels,
        even,
    )?;
    let execution = &common.execution;
    let samples = &common.samples;
    let region = &common.region;
    let output = &common.output;
    let channels = common.channels;
    let active = common.active;
    let byte = tmp_primitive_size(&common.sample_type)? == 1;
    let shared = workspace_branch(&common, even, arithmetic, true, policy)?;
    let global = workspace_branch(&common, even, arithmetic, false, policy)?;
    Ok(quote! {{
        (|| -> ::core::result::Result<
            ::rccl_cub::DeviceHistogramWorkspace<_, #active>,
            ::rccl_cub::DeviceAlgorithmError,
        > {
            let __rccl_execution = #execution;
            let __rccl_samples = #samples;
            let __rccl_region = #region;
            let __rccl_output = #output;
            let (__rccl_num_row_pixels, __rccl_num_rows, __rccl_row_stride_samples) =
                __rccl_region.__resolve::<#channels>(__rccl_samples.len())?;
            let __rccl_output_lengths = *__rccl_output.lengths();
            if __rccl_output_lengths.iter().any(|&length| length == 0) {
                return ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::InvalidHistogramShape(
                        "every histogram channel requires at least one bin",
                    ),
                );
            }
            if __rccl_output_lengths
                .iter()
                .any(|&length| length >= i32::MAX as usize)
            {
                return ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::InvalidHistogramShape(
                        "histogram channel exceeds CUB's signed level-count domain",
                    ),
                );
            }
            let __rccl_max_bins = __rccl_output_lengths.iter().copied().max().unwrap_or(0);
            let __rccl_privatized_lengths = if #byte {
                [256usize; #active]
            } else {
                __rccl_output_lengths
            };
            if #byte || __rccl_max_bins <= 256 {
                #shared
            } else {
                #global
            }
        })()
    }})
}

pub(super) fn expand_device_histogram_even_workspace(
    call: call::DeviceHistogramWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_workspace(call, true, false)
}

pub(super) fn expand_device_histogram_range_workspace(
    call: call::DeviceHistogramWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_workspace(call, false, false)
}

pub(super) fn expand_device_histogram_even_arithmetic_workspace(
    call: call::DeviceHistogramWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_workspace(call, true, true)
}

pub(super) fn expand_device_histogram_range_arithmetic_workspace(
    call: call::DeviceHistogramWorkspaceCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_workspace(call, false, true)
}

pub(super) fn expand_device_histogram_even(
    call: call::DeviceHistogramEvenCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_even(call, false)
}

pub(super) fn expand_device_histogram_range(
    call: call::DeviceHistogramRangeCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_range(call, false)
}

pub(super) fn expand_device_histogram_even_arithmetic(
    call: call::DeviceHistogramEvenCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_even(call, true)
}

pub(super) fn expand_device_histogram_range_arithmetic(
    call: call::DeviceHistogramRangeCall,
) -> syn::Result<proc_macro2::TokenStream> {
    expand_range(call, true)
}

fn launch_common(
    shape: HistogramLaunchShape<'_>,
    even: bool,
    arithmetic: bool,
    shared: bool,
    extra_launch_args: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let HistogramLaunchShape {
        sample_type,
        counter_type,
        level_type,
        channels,
        active,
        policy,
    } = shape;
    let kernel = sweep_name(even, arithmetic, policy.threads);
    let prepare_kernel = format_ident!("prepare_{kernel}");
    let threads = policy.threads;
    let pixels = policy.pixels;
    let rle = policy.rle;
    let byte = tmp_primitive_size(sample_type)? == 1;
    let bytes = shared_bytes(tmp_primitive_size(counter_type)?, active, shared);
    let generics = quote!(::<
        #sample_type,
        #counter_type,
        #level_type,
        #channels,
        #active,
        #pixels,
        #rle,
        #byte,
        #shared,
        #bytes
    >);
    let sample_launch_args = if arithmetic {
        quote! {
            *__rccl_samples.start(),
            *__rccl_samples.step(),
            __rccl_samples.len(),
        }
    } else {
        quote! { __rccl_samples, }
    };
    Ok(quote! {{
        let __rccl_workspace_blocks = __rccl_blocks_per_row
            .checked_mul(__rccl_blocks_per_col)
            .ok_or(::rccl_cub::DeviceAlgorithmError::SizeOverflow)?;
        if __rccl_workspace_blocks != 0 {
            let __rccl_launch = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
                __rccl_workspace_blocks.min(u32::MAX as usize) as u32,
                #threads as u32,
                #bytes as u32,
            );
            let __rccl_prepared = __rccl_execution.module().#prepare_kernel #generics(
                __rccl_launch,
            ).map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
            __rccl_execution.module().#kernel #generics(
                __rccl_execution.stream(),
                &__rccl_prepared,
                #sample_launch_args
                &mut *__rccl_output_storage,
                &mut *__rccl_privatized_storage,
                #extra_launch_args
                __rccl_output_lengths,
                __rccl_output_offsets,
                __rccl_privatized_lengths,
                __rccl_privatized_offsets,
                __rccl_num_row_pixels,
                __rccl_num_rows,
                __rccl_row_stride_samples,
                __rccl_blocks_per_row,
            ).map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
        }
    }})
}

fn init_launch(counter_type: &syn::Type, active: usize) -> proc_macro2::TokenStream {
    quote! {
        let __rccl_max_bins = __rccl_output_lengths.iter().copied().max().unwrap_or(0);
        let __rccl_init_grid = __rccl_max_bins.div_ceil(256).max(1);
        let __rccl_init_config = ::rccl_cub::__private::cuda_core::LaunchConfig1D::new(
            __rccl_init_grid.min(u32::MAX as usize) as u32,
            256,
            0,
        );
        let __rccl_init = __rccl_execution
            .module()
            .prepare_device_histogram_init::<#counter_type, #active>(
                __rccl_init_config,
            )
            .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
        __rccl_execution
            .module()
            .device_histogram_init::<#counter_type, #active>(
                __rccl_execution.stream(),
                &__rccl_init,
                &mut *__rccl_output_storage,
                __rccl_output_lengths,
                __rccl_output_offsets,
                &mut *__rccl_queue,
            )
            .map_err(::rccl_cub::DeviceAlgorithmError::Launch)?;
    }
}

fn expand_even(
    call: call::DeviceHistogramEvenCall,
    arithmetic: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    counter_supported(&call.counter_type)?;
    scalar(&call.sample_type)?;
    scalar(&call.level_type)?;
    let channels = call.channels.base10_parse::<usize>()?;
    let active = call.active_channels.base10_parse::<usize>()?;
    let policy = policy(
        &call.sample_type,
        &call.counter_type,
        active,
        channels,
        true,
    )?;
    let sample_type = call.sample_type;
    let counter_type = call.counter_type;
    let level_type = call.level_type;
    let execution = call.execution;
    let samples = call.samples;
    let region = call.region;
    let output = call.output;
    let lower = call.lower_levels;
    let upper = call.upper_levels;
    let workspace = call.workspace;
    let byte = tmp_primitive_size(&sample_type)? == 1;
    let shape = HistogramLaunchShape {
        sample_type: &sample_type,
        counter_type: &counter_type,
        level_type: &level_type,
        channels,
        active,
        policy,
    };
    let shared = launch_common(
        shape,
        true,
        arithmetic,
        true,
        quote! {
            __rccl_lower_levels,
            __rccl_upper_levels,
        },
    )?;
    let global = launch_common(
        shape,
        true,
        arithmetic,
        false,
        quote! {
            __rccl_lower_levels,
            __rccl_upper_levels,
        },
    )?;
    let init = init_launch(&counter_type, active);
    Ok(quote! {{
        (|| -> ::core::result::Result<(), ::rccl_cub::DeviceAlgorithmError> {
            let __rccl_execution = #execution;
            let __rccl_samples = #samples;
            let __rccl_region = #region;
            let __rccl_output = #output;
            let __rccl_workspace = #workspace;
            let __rccl_lower_levels: [#level_type; #active] = #lower;
            let __rccl_upper_levels: [#level_type; #active] = #upper;
            let (__rccl_num_row_pixels, __rccl_num_rows, __rccl_row_stride_samples) =
                __rccl_region.__resolve::<#channels>(__rccl_samples.len())?;
            let (__rccl_output_storage, __rccl_output_lengths, __rccl_output_offsets) =
                __rccl_output.__parts_mut();
            let ::rccl_cub::__private::DeviceHistogramWorkspaceParts {
                privatized: __rccl_privatized_storage,
                queue: __rccl_queue,
                privatized_lengths: __rccl_privatized_lengths,
                privatized_offsets: __rccl_privatized_offsets,
                blocks_per_row: __rccl_blocks_per_row,
                blocks_per_col: __rccl_blocks_per_col,
                region: __rccl_workspace_region,
            } = __rccl_workspace.__parts_mut();
            if __rccl_workspace_region
                != (
                    __rccl_num_row_pixels,
                    __rccl_num_rows,
                    __rccl_row_stride_samples,
                )
            {
                return ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::InvalidHistogramShape(
                        "histogram workspace does not match this image region",
                    ),
                );
            }
            let __rccl_expected_privatized_lengths = if #byte {
                [256usize; #active]
            } else {
                __rccl_output_lengths
            };
            if __rccl_privatized_lengths != __rccl_expected_privatized_lengths {
                return ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::InvalidHistogramShape(
                        "histogram workspace does not match the output bin layout",
                    ),
                );
            }
            for channel in 0..#active {
                if <#sample_type as ::rccl_cub::__private::HistogramSample<#level_type>>::even_may_overflow(
                    __rccl_lower_levels[channel],
                    __rccl_upper_levels[channel],
                    __rccl_output_lengths[channel],
                ) {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::InvalidHistogramShape(
                            "even histogram bounds or bin arithmetic are invalid",
                        ),
                    );
                }
            }
            #init
            let __rccl_max_bins = __rccl_output_lengths.iter().copied().max().unwrap_or(0);
            if #byte || __rccl_max_bins <= 256 {
                #shared
            } else {
                #global
            }
            Ok(())
        })()
    }})
}

fn expand_range(
    call: call::DeviceHistogramRangeCall,
    arithmetic: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    counter_supported(&call.counter_type)?;
    scalar(&call.sample_type)?;
    scalar(&call.level_type)?;
    let channels = call.channels.base10_parse::<usize>()?;
    let active = call.active_channels.base10_parse::<usize>()?;
    let policy = policy(
        &call.sample_type,
        &call.counter_type,
        active,
        channels,
        false,
    )?;
    let sample_type = call.sample_type;
    let counter_type = call.counter_type;
    let level_type = call.level_type;
    let execution = call.execution;
    let samples = call.samples;
    let region = call.region;
    let output = call.output;
    let levels = call.levels;
    let workspace = call.workspace;
    let byte = tmp_primitive_size(&sample_type)? == 1;
    let shape = HistogramLaunchShape {
        sample_type: &sample_type,
        counter_type: &counter_type,
        level_type: &level_type,
        channels,
        active,
        policy,
    };
    let shared = launch_common(
        shape,
        false,
        arithmetic,
        true,
        quote! {
            __rccl_levels_storage,
            __rccl_level_lengths,
            __rccl_level_offsets,
        },
    )?;
    let global = launch_common(
        shape,
        false,
        arithmetic,
        false,
        quote! {
            __rccl_levels_storage,
            __rccl_level_lengths,
            __rccl_level_offsets,
        },
    )?;
    let init = init_launch(&counter_type, active);
    Ok(quote! {{
        (|| -> ::core::result::Result<(), ::rccl_cub::DeviceAlgorithmError> {
            let __rccl_execution = #execution;
            let __rccl_samples = #samples;
            let __rccl_region = #region;
            let __rccl_output = #output;
            let __rccl_levels = #levels;
            let __rccl_workspace = #workspace;
            let (__rccl_num_row_pixels, __rccl_num_rows, __rccl_row_stride_samples) =
                __rccl_region.__resolve::<#channels>(__rccl_samples.len())?;
            let (__rccl_output_storage, __rccl_output_lengths, __rccl_output_offsets) =
                __rccl_output.__parts_mut();
            let (__rccl_levels_storage, __rccl_level_lengths, __rccl_level_offsets) =
                __rccl_levels.__parts();
            for channel in 0..#active {
                let expected = __rccl_output_lengths[channel]
                    .checked_add(1)
                    .ok_or(::rccl_cub::DeviceAlgorithmError::SizeOverflow)?;
                if __rccl_level_lengths[channel] != expected {
                    return ::core::result::Result::Err(
                        ::rccl_cub::DeviceAlgorithmError::LengthMismatch {
                            role: "histogram range levels",
                            expected,
                            actual: __rccl_level_lengths[channel],
                        },
                    );
                }
            }
            let ::rccl_cub::__private::DeviceHistogramWorkspaceParts {
                privatized: __rccl_privatized_storage,
                queue: __rccl_queue,
                privatized_lengths: __rccl_privatized_lengths,
                privatized_offsets: __rccl_privatized_offsets,
                blocks_per_row: __rccl_blocks_per_row,
                blocks_per_col: __rccl_blocks_per_col,
                region: __rccl_workspace_region,
            } = __rccl_workspace.__parts_mut();
            if __rccl_workspace_region
                != (
                    __rccl_num_row_pixels,
                    __rccl_num_rows,
                    __rccl_row_stride_samples,
                )
            {
                return ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::InvalidHistogramShape(
                        "histogram workspace does not match this image region",
                    ),
                );
            }
            let __rccl_expected_privatized_lengths = if #byte {
                [256usize; #active]
            } else {
                __rccl_output_lengths
            };
            if __rccl_privatized_lengths != __rccl_expected_privatized_lengths {
                return ::core::result::Result::Err(
                    ::rccl_cub::DeviceAlgorithmError::InvalidHistogramShape(
                        "histogram workspace does not match the output bin layout",
                    ),
                );
            }
            #init
            let __rccl_max_bins = __rccl_output_lengths.iter().copied().max().unwrap_or(0);
            if #byte || __rccl_max_bins <= 256 {
                #shared
            } else {
                #global
            }
            Ok(())
        })()
    }})
}

#[cfg(test)]
mod tests {
    use super::{HistogramPolicy, policy_for, scalar, shared_bytes, sweep_name};
    use crate::target::CudaTarget;

    fn policy(
        capability: u16,
        sample: syn::Type,
        counter: syn::Type,
        active: usize,
        channels: usize,
        even: bool,
    ) -> HistogramPolicy {
        policy_for(
            CudaTarget { capability },
            &sample,
            &counter,
            active,
            channels,
            even,
        )
        .unwrap()
    }

    #[test]
    fn pinned_sm50_fallback_scales_pixels_by_active_channels_and_sample_words() {
        assert_eq!(
            policy(
                89,
                syn::parse_quote!(i32),
                syn::parse_quote!(i32),
                1,
                1,
                true,
            ),
            HistogramPolicy {
                threads: 384,
                pixels: 16,
                rle: true,
            }
        );
        assert_eq!(
            policy(
                89,
                syn::parse_quote!(i32),
                syn::parse_quote!(i32),
                3,
                4,
                false,
            )
            .pixels,
            5
        );
        assert_eq!(
            policy(
                89,
                syn::parse_quote!(i64),
                syn::parse_quote!(u64),
                3,
                4,
                true,
            )
            .pixels,
            2
        );
    }

    #[test]
    fn pinned_sm90_and_sm100_specializations_fail_closed_to_policy500() {
        assert_eq!(
            policy(
                90,
                syn::parse_quote!(u8),
                syn::parse_quote!(i32),
                1,
                1,
                true,
            ),
            HistogramPolicy {
                threads: 768,
                pixels: 12,
                rle: false,
            }
        );
        assert_eq!(
            policy(
                90,
                syn::parse_quote!(u16),
                syn::parse_quote!(u32),
                1,
                1,
                false,
            ),
            HistogramPolicy {
                threads: 960,
                pixels: 10,
                rle: true,
            }
        );
        assert_eq!(
            policy(
                100,
                syn::parse_quote!(u8),
                syn::parse_quote!(i32),
                1,
                1,
                true,
            ),
            HistogramPolicy {
                threads: 928,
                pixels: 12,
                rle: false,
            }
        );
        assert_eq!(
            policy(
                100,
                syn::parse_quote!(u8),
                syn::parse_quote!(i32),
                1,
                1,
                false,
            ),
            HistogramPolicy {
                threads: 448,
                pixels: 12,
                rle: false,
            }
        );
        assert_eq!(
            policy(
                100,
                syn::parse_quote!(u8),
                syn::parse_quote!(i32),
                3,
                4,
                true,
            ),
            HistogramPolicy {
                threads: 384,
                pixels: 5,
                rle: true,
            }
        );
        assert_eq!(
            policy(
                90,
                syn::parse_quote!(u8),
                syn::parse_quote!(i64),
                1,
                1,
                true,
            )
            .threads,
            384
        );
    }

    #[test]
    fn iterator_mode_changes_the_compile_time_kernel_family() {
        assert_eq!(
            sweep_name(true, false, 384).to_string(),
            "device_histogram_even_sweep_384"
        );
        assert_eq!(
            sweep_name(false, true, 960).to_string(),
            "device_histogram_range_arithmetic_sweep_960"
        );
        assert_eq!(shared_bytes(4, 3, true), 3_092);
        assert_eq!(shared_bytes(4, 3, false), 20);
    }

    #[test]
    fn primitive_parser_accepts_grouped_types_but_rejects_pointer_sized_aliases() {
        assert!(scalar(&syn::parse_quote!((i32))).is_ok());
        assert!(scalar(&syn::parse_quote!(usize)).is_err());
        assert!(scalar(&syn::parse_quote!(isize)).is_err());
    }
}
