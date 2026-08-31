use super::*;
use crate::merge::{merge_load_to_shared_policy, merge_shared_bytes, nominal_4b_items};
use crate::radix::reg_bound_policy;
use crate::set::set_policy_for;
use crate::target::{tmp_cuda_target, tmp_primitive_size};

#[test]
fn set_key_value_layout_and_target_policies_match_pinned_cccl() {
    let key_value: syn::Type = syn::parse_quote!(
        ::rccl_cub::KeyValue<i32, i32>
    );
    assert_eq!(tmp_primitive_size(&key_value).unwrap(), 8);

    let padded: syn::Type = syn::parse_quote!(
        ::rccl_cub::KeyValue<u8, u64>
    );
    assert_eq!(tmp_primitive_size(&padded).unwrap(), 16);

    let spoofed: syn::Type = syn::parse_quote!(application::KeyValue<u8, u64>);
    assert!(
        tmp_primitive_size(&spoofed)
            .unwrap_err()
            .to_string()
            .contains("only rccl_cub::KeyValue")
    );

    let sm52 = tmp_cuda_target(&syn::parse_quote!(sm_52)).unwrap();
    let policy = set_policy_for(sm52, 8, 8).unwrap();
    assert_eq!((policy.threads, policy.items), (256, 8));
    assert_eq!(policy.shared_bytes, 19_456);
    assert!(!policy.sm70);

    let sm89 = tmp_cuda_target(&syn::parse_quote!(sm_89)).unwrap();
    let policy = set_policy_for(sm89, 8, 8).unwrap();
    assert_eq!((policy.threads, policy.items), (512, 10));
    assert_eq!(policy.shared_bytes, 47_104);
    assert!(policy.sm70);

    let wide_value = set_policy_for(sm89, 4, 16).unwrap();
    assert_eq!(wide_value.items, 19, "value size must not tune items");
    assert_eq!(wide_value.shared_bytes, 165_872);
}

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
fn device_transform_kernel_is_owned_by_cub() {
    let kernels = expand_device_kernels(syn::parse_quote!(__test_kernels))
        .expect("CUB must generate the transform kernel");
    let kernel_text = kernels.to_string();
    assert!(kernel_text.contains("transform_repeat_in_place"));
    assert!(kernel_text.contains("grid_stride_runs32"));
    assert!(kernel_text.contains("expect (clippy :: too_many_arguments"));
    assert!(kernel_text.contains("CUDA entry points preserve the pinned CUB launch ABI"));
    assert!(!kernel_text.contains("allow (clippy :: too_many_arguments)"));
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
        !operation.contains("device_radix_sort_histogram_4 :: < i32 , u32 , _ , _ , _ , 21usize")
    );
    assert_eq!(operation.matches("zero_radix_lookback").count(), 1);
    assert!(operation.contains("copy_device_range :: < i32 , _ , _ >"));
    assert!(!operation.contains("core :: mem :: swap"));
    assert!(!operation.contains("core :: mem :: drop (__rccl_buffers)"));
    assert!(operation.contains("let (__rccl_result , __rccl_current_is_first) ="));
    assert!(operation.contains("while __rccl_pass < __rccl_passes"));
    assert!(operation.contains("let __rccl_bin_offset = __rccl_pass * 256usize"));
    assert!(
        operation
            .contains("let __rccl_current_bit = __rccl_begin_bit + (__rccl_pass * 8usize) as u32")
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
    assert!(operation.contains("LaunchConfig1D :: new (__rccl_blocks as u32 , 384usize as u32"));
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
        operation
            .contains("prepare_device_merge_sort_partition :: < u32 , _ , 256usize , 17usize >")
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
        padded.contains("prepare_device_merge_sort_partition :: < u64 , _ , 256usize , 8usize >")
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
