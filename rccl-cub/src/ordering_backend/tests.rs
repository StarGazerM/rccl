use super::{
    block_exchange_index, block_exchange_storage_items, load_to_shared_buffer_bytes,
    local_merge_path_begin, radix_digit_masked_u32, radix_digit_masked_u64, radix_digit_u32,
    radix_digit_u64, radix_f32_from_ordered_ascending, radix_f32_from_ordered_descending,
    radix_f32_to_ordered_ascending, radix_f32_to_ordered_descending,
    radix_f64_from_ordered_ascending, radix_f64_from_ordered_descending,
    radix_f64_to_ordered_ascending, radix_f64_to_ordered_descending, radix_float_digit_masked_u32,
    radix_float_digit_masked_u64, radix_float_digit_u32, radix_float_digit_u64,
    radix_float_rank_digit_u32, radix_float_rank_digit_u64, radix_i32_from_ordered_ascending,
    radix_i32_from_ordered_descending, radix_i32_to_ordered_ascending,
    radix_i32_to_ordered_descending, radix_i64_from_ordered_ascending,
    radix_i64_from_ordered_descending, radix_i64_to_ordered_ascending,
    radix_i64_to_ordered_descending, radix_min_ordered_u32, radix_min_ordered_u64,
    radix_rank_digit_u32, radix_rank_digit_u64, radix_u32_from_ordered_ascending,
    radix_u32_from_ordered_descending, radix_u32_to_ordered_ascending,
    radix_u32_to_ordered_descending, radix_u64_from_ordered_ascending,
    radix_u64_from_ordered_descending, radix_u64_to_ordered_ascending,
    radix_u64_to_ordered_descending,
};

const ORDERING_SOURCE: &str = concat!(
    include_str!("mod.rs"),
    include_str!("merge.rs"),
    include_str!("radix.rs"),
    "\n// __RCCL_TEST_BOUNDARY__\n",
);

#[test]
fn primitive_radix_function_items_round_trip_every_supported_representation() {
    for key in [0u32, 1, u32::MAX, 0x8000_0000, 0x1234_5678] {
        assert_eq!(
            radix_u32_from_ordered_ascending(radix_u32_to_ordered_ascending(key)),
            key
        );
        assert_eq!(
            radix_u32_from_ordered_descending(radix_u32_to_ordered_descending(key)),
            key
        );
    }
    for key in [i32::MIN, -1, 0, 1, i32::MAX] {
        assert_eq!(
            radix_i32_from_ordered_ascending(radix_i32_to_ordered_ascending(key)),
            key
        );
        assert_eq!(
            radix_i32_from_ordered_descending(radix_i32_to_ordered_descending(key)),
            key
        );
    }
    for raw in [
        0u32,
        0x8000_0000,
        1.0f32.to_bits(),
        (-1.0f32).to_bits(),
        f32::INFINITY.to_bits(),
        f32::NAN.to_bits(),
    ] {
        let key = f32::from_bits(raw);
        assert_eq!(
            radix_f32_from_ordered_ascending(radix_f32_to_ordered_ascending(key)).to_bits(),
            raw
        );
        assert_eq!(
            radix_f32_from_ordered_descending(radix_f32_to_ordered_descending(key)).to_bits(),
            raw
        );
    }
    for key in [
        0u64,
        1,
        u64::MAX,
        0x8000_0000_0000_0000,
        0x1234_5678_9abc_def0,
    ] {
        assert_eq!(
            radix_u64_from_ordered_ascending(radix_u64_to_ordered_ascending(key)),
            key
        );
        assert_eq!(
            radix_u64_from_ordered_descending(radix_u64_to_ordered_descending(key)),
            key
        );
    }
    for key in [i64::MIN, -1, 0, 1, i64::MAX] {
        assert_eq!(
            radix_i64_from_ordered_ascending(radix_i64_to_ordered_ascending(key)),
            key
        );
        assert_eq!(
            radix_i64_from_ordered_descending(radix_i64_to_ordered_descending(key)),
            key
        );
    }
    for raw in [
        0u64,
        0x8000_0000_0000_0000,
        1.0f64.to_bits(),
        (-1.0f64).to_bits(),
        f64::INFINITY.to_bits(),
        f64::NAN.to_bits(),
    ] {
        let key = f64::from_bits(raw);
        assert_eq!(
            radix_f64_from_ordered_ascending(radix_f64_to_ordered_ascending(key)).to_bits(),
            raw
        );
        assert_eq!(
            radix_f64_from_ordered_descending(radix_f64_to_ordered_descending(key)).to_bits(),
            raw
        );
    }
}

#[test]
fn radix_rank_digit_is_equivalent_for_every_valid_window_shape() {
    let ordered32 = 0xd3a5_7c19u32;
    for bit in 0..32u32 {
        for bits in 1..=(32 - bit).min(8) {
            assert_eq!(
                radix_rank_digit_u32(ordered32, bit, bits),
                radix_digit_u32(ordered32, bit, bits)
            );
        }
    }
    let ordered64 = 0xd3a5_7c19_2468_bdefu64;
    for bit in 0..64u32 {
        for bits in 1..=(64 - bit).min(8) {
            assert_eq!(
                radix_rank_digit_u64(ordered64, bit, bits),
                radix_digit_u64(ordered64, bit, bits)
            );
        }
    }
}

#[test]
fn floating_radix_digits_make_signed_zero_equivalent_in_both_directions() {
    for to_ordered in [
        radix_f32_to_ordered_ascending as fn(f32) -> u32,
        radix_f32_to_ordered_descending,
    ] {
        let positive = to_ordered(0.0);
        let negative = to_ordered(-0.0);
        for bit in 0..32u32 {
            for bits in 1..=(32 - bit).min(8) {
                assert_eq!(
                    radix_float_digit_u32(positive, bit, bits),
                    radix_float_digit_u32(negative, bit, bits)
                );
                assert_eq!(
                    radix_float_rank_digit_u32(positive, bit, bits),
                    radix_float_rank_digit_u32(negative, bit, bits)
                );
            }
        }
    }
    for to_ordered in [
        radix_f64_to_ordered_ascending as fn(f64) -> u64,
        radix_f64_to_ordered_descending,
    ] {
        let positive = to_ordered(0.0);
        let negative = to_ordered(-0.0);
        for bit in 0..64u32 {
            for bits in 1..=(64 - bit).min(8) {
                assert_eq!(
                    radix_float_digit_u64(positive, bit, bits),
                    radix_float_digit_u64(negative, bit, bits)
                );
                assert_eq!(
                    radix_float_rank_digit_u64(positive, bit, bits),
                    radix_float_rank_digit_u64(negative, bit, bits)
                );
            }
        }
    }
}

#[test]
fn single_tile_guard_keys_stay_in_cub_bit_ordered_representation() {
    assert_eq!(radix_min_ordered_u32(), 0);
    assert_eq!(radix_min_ordered_u64(), 0);
    assert_eq!(super::radix_default_u32(), u32::MAX);
    assert_eq!(super::radix_default_u64(), u64::MAX);
}

#[test]
fn rank_digits_keep_cub_bitfield_extraction_not_rotation() {
    let source = ORDERING_SOURCE;
    let begin = source
        .find("pub fn radix_rank_digit_u32")
        .expect("u32 rank digit helper must exist");
    let end = source[begin..]
        .find("fn warp_exclusive_sum_u32")
        .map(|offset| begin + offset)
        .expect("warp scan must follow primitive rank digit helpers");
    let rank_digits = &source[begin..end];
    assert!(rank_digits.contains("radix_digit_u32(ordered, bit, bits)"));
    assert!(rank_digits.contains("radix_digit_u64(ordered, bit, bits)"));
    assert!(!rank_digits.contains("rotate_right"));
}

#[test]
fn single_tile_radix_keeps_cub_union_phases_and_rust_ownership_boundary() {
    let source = ORDERING_SOURCE;
    let begin = source
        .find("fn radix_single_tile_rank")
        .expect("single-tile BlockRadixRank lowering must exist");
    let end = source[begin..]
        .find("/// Exact pinned SM80+ histogram agent")
        .map(|offset| begin + offset)
        .expect("one-sweep histogram must follow the single-tile route");
    let single_tile = &source[begin..end];

    for required in [
        "counter_lane * THREADS + thread_index",
        "thread_index as usize * RAKING_SEGMENT + word",
        "region::<u16>",
        "counter_indices",
        "for_each_radix_raking_word!",
        "for_each_single_tile_warp!",
        "let lane = warp::lane_id()",
        "let pass_bits = (bit_range.end - current_bit).min(6)",
        "radix_single_tile_scatter::<B, THREADS, ITEMS, true>",
        "radix_single_tile_scatter::<B, THREADS, ITEMS, false>",
        "BlockStripedBlockWrite",
        "MaybeUninit::<V>::uninit()",
        "MaybeUninit::<u16>::uninit()",
        "MaybeUninit::<u32>::uninit(); RAKING_SEGMENT",
        "warp::shuffle_sync(u32::MAX, input_values.len() as u32, 0)",
    ] {
        assert!(
            single_tile.contains(required),
            "single-tile source topology lost: {required}"
        );
    }
    assert!(
        single_tile.find("counter_lane * THREADS + thread_index")
            < single_tile.find("thread_index as usize * RAKING_SEGMENT + word"),
        "counter reset must precede the union's raking reinterpretation"
    );
    for forbidden in [
        "DeviceRegion",
        "get_unchecked",
        "as_mut_ptr",
        "sort_unstable",
        "device_merge_sort",
        "striped: bool",
        "let mut cached_segment = [0u32; RAKING_SEGMENT]",
        "let mut thread_prefixes = [0u16; ITEMS]",
        &["un", "safe"].concat(),
        &["*", "mut"].concat(),
    ] {
        assert!(
            !single_tile.contains(forbidden),
            "single-tile algorithm crossed its Rust safety/semantic boundary: {forbidden}"
        );
    }
}

#[test]
fn signed_local_merge_path_clamps_to_zero_like_cub() {
    assert_eq!(local_merge_path_begin(0, 17), 0);
    assert_eq!(local_merge_path_begin(8, 17), 0);
    assert_eq!(local_merge_path_begin(17, 17), 0);
    assert_eq!(local_merge_path_begin(23, 17), 6);
}

#[test]
fn global_merge_path_delegates_range_proof_without_recovering_indexes() {
    let source = ORDERING_SOURCE;
    let begin = source
        .find("pub fn global_merge_path")
        .expect("global MergePath wrapper must exist");
    let end = source[begin..]
        .find("/// CUB `MergePath` over the two adjacent ranges")
        .map(|offset| begin + offset)
        .expect("local MergePath helper must follow the global wrapper");
    let merge_path = &source[begin..end];

    for required in [
        "MergePathSearch::new(first, second, diagonal)",
        "search.step(|first_key, second_key|",
        "search.partition()",
    ] {
        assert!(
            merge_path.contains(required),
            "global MergePath lost its affine range capability: {required}"
        );
    }
    for forbidden in [
        "first[".to_owned(),
        "second[".to_owned(),
        "get_unchecked".to_owned(),
        ["un", "safe"].concat(),
    ] {
        assert!(
            !merge_path.contains(&forbidden),
            "global MergePath must not recreate C++ iterator arithmetic: {forbidden}"
        );
    }
}

#[test]
fn policy800_rank_keeps_the_pinned_ballot_loop_without_asm_shortcuts() {
    let source = ORDERING_SOURCE;
    let begin = source
        .find("fn radix_match_any_8")
        .expect("Policy800 match-any helper must exist");
    let end = source[begin..]
        .find("fn block_exclusive_sum_raking_memoize_384_i32")
        .map(|offset| begin + offset)
        .expect("Policy800 block scan must follow match-any");
    let match_any = &source[begin..end];
    assert!(match_any.contains("warp::ballot_sync"));
    assert!(match_any.contains("label & (1u32 << $bit) != 0"));
    assert!(match_any.contains("if bit_is_set"));
    assert!(match_any.contains("!set_lanes"));
    assert!(match_any.contains("let mut peers = matching_lanes_for_bit!(0u32)"));
    assert_eq!(
        match_any
            .matches("peers &= matching_lanes_for_bit!")
            .count(),
        7
    );
    for forbidden in [
        ["warp::", "match_any_sync"].concat(),
        "polarity".to_owned(),
        "as i32 >> 31".to_owned(),
        "wrapping_sub(1u32)".to_owned(),
        ["p", "tx_", "a", "sm", "!"].concat(),
        ["a", "sm", "!"].concat(),
    ] {
        assert!(
            !match_any.contains(&forbidden),
            "forbidden shortcut: {forbidden}"
        );
    }
}

#[test]
fn policy800_raking_scan_materializes_each_pinned_warp_step() {
    let source = ORDERING_SOURCE;
    let macro_begin = source
        .find("macro_rules! for_each_warp_scan_offset")
        .expect("the fixed warp-scan expansion must exist");
    let macro_end = source[macro_begin..]
        .find("macro_rules! for_each_radix_raking_word")
        .map(|offset| macro_begin + offset)
        .expect("the radix-raking expansion must follow the warp scan");
    let expansion = &source[macro_begin..macro_end];
    for offset in [
        "one!(1u32)",
        "one!(2u32)",
        "one!(4u32)",
        "one!(8u32)",
        "one!(16u32)",
    ] {
        assert!(
            expansion.contains(offset),
            "missing pinned warp-scan offset: {offset}"
        );
    }

    let scan_begin = source
        .find("fn warp_exclusive_sum_i32")
        .expect("the Policy800 warp scan must exist");
    let scan_end = source[scan_begin..]
        .find("fn radix_match_any_8")
        .map(|offset| scan_begin + offset)
        .expect("the match-any helper must follow the warp scan");
    let scan = &source[scan_begin..scan_end];
    assert!(scan.contains("for_each_warp_scan_offset!"));
    assert!(!scan.contains("while offset < 32"));

    let block_scan_begin = source
        .find("fn block_exclusive_sum_raking_memoize_384_i32")
        .expect("the Policy800 memoized block scan must exist");
    let block_scan_end = source[block_scan_begin..]
        .find("fn radix_lookback_global")
        .map(|offset| block_scan_begin + offset)
        .expect("the lookback helper must follow the block scan");
    let block_scan = &source[block_scan_begin..block_scan_end];
    assert!(block_scan.contains("let mut cached = [MaybeUninit::<i32>::uninit(); SEGMENT_LENGTH]"));
    assert!(!block_scan.contains("let mut cached = [0i32; SEGMENT_LENGTH]"));

    let u64_scan_begin = source
        .find("fn warp_exclusive_sum_u64")
        .expect("the bin-scan warp primitive must exist");
    let u64_scan_end = source[u64_scan_begin..]
        .find("pub fn radix_exclusive_sum_agent")
        .map(|offset| u64_scan_begin + offset)
        .expect("the bin-scan agent must follow its helpers");
    let u64_scan = &source[u64_scan_begin..u64_scan_end];
    assert!(u64_scan.contains("for_each_warp_scan_offset!"));
    assert!(u64_scan.contains("let mut cached = [MaybeUninit::<u64>::uninit(); SEGMENT_LENGTH]"));
    assert!(!u64_scan.contains("while offset < 32"));
    assert!(!u64_scan.contains("let mut cached = [0u64; SEGMENT_LENGTH]"));
}

#[test]
fn warp_specializations_read_the_lane_register_instead_of_rederiving_it() {
    let source = ORDERING_SOURCE;
    let backend = source
        .split("// __RCCL_TEST_BOUNDARY__")
        .next()
        .expect("device backend must precede its host tests");

    assert!(backend.contains("let lane = warp::lane_id()"));
    for forbidden in [
        "let lane = thread_index & 31",
        "let lane = thread_index % 32",
    ] {
        assert!(
            !backend.contains(forbidden),
            "fixed-width warp algorithms must use cuda-oxide's safe lane-register API: {forbidden}"
        );
    }
}

#[test]
fn policy800_onesweep_keeps_signed_counts_and_checked_slice_bands() {
    let source = ORDERING_SOURCE;
    let begin = source
        .find("pub fn radix_onesweep_keys_agent")
        .expect("Policy800 one-sweep agent must exist");
    let end = source[begin..]
        .find("// __RCCL_TEST_BOUNDARY__")
        .map(|offset| begin + offset)
        .expect("backend tests must follow one-sweep");
    let agent = &source[begin..end];

    for required in [
        "let num_items = input.len() as u32",
        "let mut warp_offsets =",
        "let mut local_count = 0i32",
        "let bin = thread_index",
        "let mut keys = [MaybeUninit::<B>::uninit(); ITEMS]",
        "let mut ranks = [MaybeUninit::<i32>::uninit(); ITEMS]",
        "let rank = initialized_value(&ranks[item])",
        "let tile_items = num_items - tile_base",
        "let digit_mask = (1u32 << num_bits) - 1",
        "bins.get(bin_offset..bin_offset + RADIX_DIGITS)",
        "FullWarpStripedInput::<K, ITEMS>::new",
        "PartialWarpStripedInput::<K, ITEMS>::new",
        ".map_or(item, sentinel, to_ordered)",
        "partial_values.load_into(item, &mut values[item])",
    ] {
        assert!(
            agent.contains(required),
            "missing one-sweep contract: {required}"
        );
    }
    for forbidden in [
        "num_items: u32",
        "input[source_index",
        "bins[bin_offset",
        "let mut ranks = [0i32; ITEMS]",
        "let mut keys = [sentinel; ITEMS]",
        "let valid_items =",
        "thread_index.min(RADIX_DIGITS as u32 - 1)",
        "digit(keys[item], current_bit, num_bits)",
    ] {
        assert!(
            !agent.contains(forbidden),
            "one-sweep must not duplicate a slice extent or its bounds checks: {forbidden}"
        );
    }

    let constructor_begin = agent
        .find("let dynamic_block = counter.fetch_add_device")
        .expect("the dynamic block-index publication must remain explicit");
    let constructor_end = agent[constructor_begin..]
        .find("let block_index =")
        .map(|offset| constructor_begin + offset)
        .expect("the published block index must be consumed");
    assert_eq!(
        agent[constructor_begin..constructor_end]
            .matches("thread::sync_threads()")
            .count(),
        1,
        "the pinned AgentRadixSortOnesweep constructor has exactly one CTA barrier"
    );

    for scatter_marker in [
        "MaybeUninit::new(ordered)",
        "MaybeUninit::new(initialized_value(&keys[item]))",
    ] {
        let scatter = agent
            .find(scatter_marker)
            .expect("both one-sweep agents must scatter keys into their shared union");
        let after_scatter = &agent[scatter + scatter_marker.len()..];
        let lookback = after_scatter
            .find("if active_bin")
            .expect("lookback must follow the shared key scatter");
        assert!(
            !after_scatter[..lookback].contains("thread::sync_threads()"),
            "an extra barrier must not serialize CUB's key-scatter/lookback overlap"
        );
    }
}

#[test]
fn masked_digit_extractors_match_the_bit_count_forms() {
    for bits in 1..=8 {
        let mask = (1u32 << bits) - 1;
        for bit in [0, 1, 7, 13, 24] {
            let value = 0xa5c3_791eu32;
            assert_eq!(
                radix_digit_masked_u32(value, bit, mask),
                radix_digit_u32(value, bit, bits)
            );
            assert_eq!(
                radix_float_digit_masked_u32(value, bit, mask),
                radix_float_digit_u32(value, bit, bits)
            );
        }
        for bit in [0, 1, 17, 41, 56] {
            let value = 0xa5c3_791e_d62b_480fu64;
            assert_eq!(
                radix_digit_masked_u64(value, bit, mask),
                radix_digit_u64(value, bit, bits)
            );
            assert_eq!(
                radix_float_digit_masked_u64(value, bit, mask),
                radix_float_digit_u64(value, bit, bits)
            );
        }
    }
}

#[test]
fn radix_histogram_keeps_runtime_pass_state_and_compile_time_item_extent() {
    let source = ORDERING_SOURCE;
    let begin = source
        .find("pub fn radix_histogram_agent")
        .expect("radix histogram agent must exist");
    let end = source[begin..]
        .find("fn warp_exclusive_sum_u64")
        .map(|offset| begin + offset)
        .expect("exclusive-sum helper must follow the histogram agent");
    let histogram = &source[begin..end];

    for required in [
        "const THREADS: usize",
        "const MAX_PASSES: usize",
        "let num_passes = (bit_range.end - bit_range.start).div_ceil(8) as usize",
        "let index = tile_offset + thread_index + item * THREADS",
        "if index < keys.len()",
        "let current_bit = bit_range.start + pass as u32 * 8",
        "let digit_mask = (1u32 << num_bits) - 1",
        "for_each_thread_item!(ITEMS, item",
    ] {
        assert!(
            histogram.contains(required),
            "missing pinned histogram staging contract: {required}"
        );
    }
    assert_eq!(
        histogram
            .matches("for_each_histogram_pass!(MAX_PASSES, num_passes, pass")
            .count(),
        3,
        "initialization, accumulation, and publication must preserve CUB's fully-unrolled pass traversal"
    );
    assert!(
        !histogram.contains("while pass < num_passes")
            && !histogram.contains("while current_bit < end_bit"),
        "CUB marks every histogram pass loop fully unrolled; a rolled loop would hide a performance mismatch"
    );
    assert!(
        !histogram.contains("blockDim_x"),
        "Policy800 histogram block width is a const policy, not runtime agent state"
    );
    assert!(
        !histogram.contains("digit(\n                    initialized_value(&ordered_keys[item]),\n                    current_bit,\n                    num_bits,"),
        "the extractor mask must be constructed once per pass, as in CUB's ShiftDigitExtractor"
    );
}

#[test]
fn serial_merge_does_not_duplicate_cursor_state_as_raw_pointers() {
    let source = ORDERING_SOURCE;
    let tests_begin = source
        .find("// __RCCL_TEST_BOUNDARY__")
        .expect("backend tests must follow the implementation");
    let implementation = &source[..tests_begin];
    for forbidden in [
        ["let mut first_", "pointer"].concat(),
        ["let mut second_", "pointer"].concat(),
        ["select_", "unpredictable"].concat(),
    ] {
        assert!(
            !implementation.contains(&forbidden),
            "serial merge must keep the measured Rust cursor/control-flow form: {forbidden}"
        );
    }
    for required in [
        "let oob_default = local[0]",
        "let oob_default = local_keys[0]",
        "let mut first_key = if first_count != 0",
        "let mut second_key = if second_count != 0",
        "local_keys[item] = second_key",
        "local_keys[item] = first_key",
        "indices[item] = second_cursor",
        "source_indices[item] = second_cursor",
    ] {
        assert!(
            implementation.contains(required),
            "BlockMergeSort serial merge must retain CUB's typed conditional head: {required}"
        );
    }
    assert!(
        !implementation.contains("local_keys[item] = if take_second"),
        "Rust pair serial merge must not keep selected keys and indices live together"
    );
}

#[test]
fn pair_compare_exchange_uses_one_borrow_safe_selection_predicate() {
    let source = ORDERING_SOURCE;
    let begin = source
        .find("fn stable_odd_even_sort_pairs")
        .expect("pair thread sort must exist");
    let end = source[begin..]
        .find("fn block_merge_round")
        .map(|offset| begin + offset)
        .expect("block merge must follow the thread sort");
    let pair_sort = &source[begin..end];

    for required in [
        "let exchange = ordered_before",
        "keys[left] = if exchange",
        "keys[right] = if exchange",
        "values[left] = if exchange",
        "values[right] = if exchange",
    ] {
        assert!(
            pair_sort.contains(required),
            "stable pair compare-exchange lost its single-predicate selection form: {required}"
        );
    }
    for forbidden in ["temporary_key", "temporary_value"] {
        assert!(
            !pair_sort.contains(forbidden),
            "sequential swap mutation regresses pair block-sort codegen: {forbidden}"
        );
    }
}

#[test]
fn merge_sort_agents_keep_cccl_tile_and_offset_predicates() {
    let source = ORDERING_SOURCE;
    let keys_begin = source
        .find("pub fn merge_sort_merge_keys_agent")
        .expect("keys merge agent must exist");
    let pairs_begin = source
        .find("pub fn merge_sort_merge_pairs_agent")
        .expect("pair merge agent must exist");
    let pairs_end = source[pairs_begin..]
        .find("fn merge_sort_merge_pairs_tile")
        .map(|offset| pairs_begin + offset)
        .expect("pair merge tile must follow its agent");
    let keys_agent = &source[keys_begin..pairs_begin];
    let pairs_agent = &source[pairs_begin..pairs_end];
    for agent in [keys_agent, pairs_agent] {
        assert!(agent.contains("if items_in_tile == tile_items"));
        assert!(
            !agent.contains("gridDim_x"),
            "AgentMerge selects its specialization from the actual remaining count"
        );
    }

    let partition_begin = source
        .find("pub fn merge_sort_partition_point")
        .expect("partition agent must exist");
    let partition = &source[partition_begin..keys_begin];
    assert!(!partition.contains("saturating_mul"));
    assert!(partition.contains("start.saturating_add(size)"));
    assert!(partition.contains("second_begin.saturating_add(size)"));
}

#[test]
fn merge_sort_agents_use_checked_concatenation_without_invalid_empty_ranges() {
    let source = ORDERING_SOURCE;
    let keys_begin = source
        .find("fn merge_sort_merge_keys_tile")
        .expect("keys merge tile must exist");
    let pairs_begin = source
        .find("fn merge_sort_merge_pairs_tile")
        .expect("pair merge tile must exist");
    let pairs_end = source[pairs_begin..]
        .find("fn merge_pairs_tile")
        .map(|offset| pairs_begin + offset)
        .expect("DeviceMerge must follow merge-sort agents");
    let keys = &source[keys_begin..pairs_begin];
    let pairs = &source[pairs_begin..pairs_end];

    for required in [
        "output.blocked_concatenated_input::<K, FULL>(",
        "start + first_begin",
        "start + size + second_begin",
    ] {
        assert!(
            keys.contains(required),
            "keys AgentMerge must preserve CUB's concatenated input boundary: {required}"
        );
    }
    for required in [
        "output_keys.blocked_concatenated_input::<K, FULL>(",
        "output_keys.blocked_concatenated_input::<V, FULL>(",
    ] {
        assert!(
            pairs.contains(required),
            "pair AgentMerge must preserve CUB's concatenated input boundary: {required}"
        );
    }
    for agent in [keys, pairs] {
        assert!(
            agent.contains(".load_into(item, &mut local"),
            "AgentMerge must load directly into its MaybeUninit register slot"
        );
        assert!(
            !agent.contains("let source_index = start +"),
            "an unrolled AgentMerge load must not rebuild the full-slice address for every item"
        );
        assert!(
            !agent.contains("&source[start + size + second_begin..]"),
            "a zero-count CUB iterator must not become an out-of-bounds Rust slice"
        );
        assert!(
            !agent.contains("get_unchecked"),
            "AgentMerge must consume the safe input capability instead of recovering unchecked indexing"
        );
    }
}

#[test]
fn device_merge_register_loads_use_the_same_checked_concatenation() {
    let source = ORDERING_SOURCE;
    let keys_begin = source
        .find("fn merge_keys_tile")
        .expect("DeviceMerge keys tile must exist");
    let pairs_agent = source[keys_begin..]
        .find("pub fn merge_pairs_agent")
        .map(|offset| keys_begin + offset)
        .expect("DeviceMerge pair agent must follow keys");
    let pairs_begin = source[pairs_agent..]
        .find("fn merge_pairs_tile")
        .map(|offset| pairs_agent + offset)
        .expect("DeviceMerge pair tile must exist");
    let pairs_end = source[pairs_begin..]
        .find("pub fn radix_init_bins_and_counters_agent")
        .map(|offset| pairs_begin + offset)
        .expect("radix agents must follow DeviceMerge");
    let keys = &source[keys_begin..pairs_agent];
    let pairs = &source[pairs_begin..pairs_end];

    assert!(keys.contains("output.blocked_concatenated_input::<K, FULL>("));
    assert_eq!(
        pairs
            .matches("tile.blocked_concatenated_input::<K, FULL>(")
            .count(),
        1
    );
    assert_eq!(
        pairs
            .matches("tile.blocked_concatenated_input::<V, FULL>(")
            .count(),
        1
    );
    assert_eq!(pairs.matches("tile.output(").count(), 2);
    for backend in [keys, pairs] {
        assert!(
            backend.contains(".load_into(item, &mut local"),
            "DeviceMerge must not return an Option for every unrolled register load"
        );
        for forbidden in [
            "first[first_begin +",
            "second[second_begin +",
            "first_keys[first_begin +",
            "second_keys[second_begin +",
            "first_values[first_begin +",
            "second_values[second_begin +",
            "get_unchecked",
        ] {
            assert!(
                !backend.contains(forbidden),
                "DeviceMerge must not bypass its checked concatenated input: {forbidden}"
            );
        }
    }
}

#[test]
fn block_sort_global_ranges_use_affine_capabilities_without_pointer_recovery() {
    let source = ORDERING_SOURCE;
    let begin = source
        .find("pub fn merge_sort_block_keys_agent")
        .expect("keys block-sort agent must exist");
    let end = source[begin..]
        .find("fn merge_sort_block_keys_tile")
        .map(|offset| begin + offset)
        .expect("keys tile implementation must follow all block agents");
    let agents_and_pair_io = &source[begin..end];

    for required in [
        "OneDimensionalLaunch<'_>",
        "WarpStripedBlockInput::<K",
        "WarpStripedBlockRead::<K",
        "WarpStripedBlockWrite::<K",
        "warp_striped_pair_into_write",
    ] {
        assert!(
            agents_and_pair_io.contains(required),
            "block-sort global range lost its Rust ownership capability: {required}"
        );
    }
    for forbidden in [
        ".as_ptr()".to_owned(),
        ".as_mut_ptr()".to_owned(),
        "cast_const".to_owned(),
        ["*", "const"].concat(),
        ["*", "mut"].concat(),
    ] {
        assert!(
            !agents_and_pair_io.contains(&forbidden),
            "block-sort global range recovered a raw pointer: {forbidden}"
        );
    }
}

#[test]
fn block_sort_guarded_load_has_one_bounds_decision() {
    let source = ORDERING_SOURCE;
    let pair_begin = source
        .find("fn merge_sort_block_pairs_load")
        .expect("pair block load must exist");
    let pair_end = source[pair_begin..]
        .find("fn merge_sort_block_pairs_loaded")
        .map(|offset| pair_begin + offset)
        .expect("pair loaded phase must follow pair block load");
    let pair_load = &source[pair_begin..pair_end];
    assert_eq!(pair_load.matches(".read_or(item,").count(), 2);
    assert!(
        !pair_load.contains("if !LAST || warp_striped < valid_items"),
        "the safe capability is already CUB's single guarded-load decision"
    );

    let keys_begin = source
        .find("fn merge_sort_block_keys_tile")
        .expect("keys block load must exist");
    let keys_end = source[keys_begin..]
        .find("fn merge_sort_block_keys_in_place_tile")
        .map(|offset| keys_begin + offset)
        .expect("in-place keys load must follow copy keys load");
    let keys_load = &source[keys_begin..keys_end];
    assert_eq!(keys_load.matches("input.read_or(item,").count(), 1);
    assert!(!keys_load.contains("if !LAST || warp_striped < valid_items"));
}

#[test]
fn radix_init_uses_affine_grid_stride_capabilities() {
    let source = ORDERING_SOURCE;
    let begin = source
        .find("pub fn radix_init_bins_and_counters_agent")
        .expect("radix init agent must exist");
    let end = source[begin..]
        .find("pub fn radix_histogram_agent")
        .map(|offset| begin + offset)
        .expect("histogram agent must follow radix init");
    let init = &source[begin..end];
    assert_eq!(init.matches("grid_stride_runs32").count(), 2);
    assert!(!init.contains("as_mut_ptr"));
    assert!(!init.contains(&["un", "safe"].concat()));
}

#[test]
fn radix_scan_uses_one_affine_bin_capability() {
    let source = ORDERING_SOURCE;
    let begin = source
        .find("pub fn radix_exclusive_sum_agent")
        .expect("radix scan agent must exist");
    let end = source[begin..]
        .find("fn warp_exclusive_sum_i32")
        .map(|offset| begin + offset)
        .expect("one-sweep helpers must follow radix scan");
    let scan = &source[begin..end];
    assert!(scan.contains("element_thread32"));
    assert!(!scan.contains("as_mut_ptr"));
}

#[test]
fn warp_transpose_store_formula_is_a_tile_bijection() {
    for items in 1usize..=15 {
        let mut seen = vec![false; 512 * items];
        for thread in 0usize..512 {
            let lane = thread & 31;
            let warp = thread >> 5;
            let warp_offset = warp * 32 * items;
            for item in 0usize..items {
                let output = warp_offset + item * 32 + lane;
                assert!(!seen[output]);
                seen[output] = true;
            }
        }
        assert!(seen.into_iter().all(|value| value));
    }
}

#[test]
fn block_exchange_padding_matches_the_pinned_power_of_two_rule() {
    assert_eq!(block_exchange_storage_items::<256, 17>(), 4_352);
    assert_eq!(block_exchange_storage_items::<256, 8>(), 2_112);
    assert_eq!(block_exchange_index::<8>(31), 31);
    assert_eq!(block_exchange_index::<8>(32), 33);

    let mut occupied = vec![false; block_exchange_storage_items::<256, 8>()];
    for logical in 0..(256 * 8) as u32 {
        let physical = block_exchange_index::<8>(logical);
        assert!(
            !occupied[physical],
            "padded exchange index must be injective"
        );
        occupied[physical] = true;
    }
    assert_eq!(occupied.iter().filter(|&&slot| slot).count(), 2_048);
    assert_eq!(occupied.iter().filter(|&&slot| !slot).count(), 64);
}

#[test]
fn block_merge_round_width_remains_runtime_algorithm_state() {
    let source = ORDERING_SOURCE;
    let tests_begin = source
        .find("// __RCCL_TEST_BOUNDARY__")
        .expect("backend tests must follow the implementation");
    let implementation = &source[..tests_begin];
    assert_eq!(
        implementation
            .matches("while target_merged_threads <=")
            .count(),
        4
    );
    assert!(implementation.contains("target_merged_threads *= 2"));
    assert!(!implementation.contains("for_each_merge_round"));
    assert!(!implementation.contains("const TARGET"));
}

#[test]
fn load_to_shared_layout_covers_every_alignment_without_overlap() {
    for item_size in [1u32, 2, 4, 8, 16] {
        for first_items in 0u32..=128 {
            for second_items in 0u32..=128 {
                let tile_items = first_items + second_items;
                let whole_buffer_bytes = load_to_shared_buffer_bytes(tile_items + 1, item_size)
                    + u32::from(item_size < 16) * 32;
                let second_buffer_bytes = load_to_shared_buffer_bytes(second_items, item_size);
                let second_buffer_offset = whole_buffer_bytes - second_buffer_bytes;

                for first_address_mod_16 in 0u32..16 {
                    for second_address_mod_16 in 0u32..16 {
                        if first_address_mod_16 % item_size.min(16) != 0
                            || second_address_mod_16 % item_size.min(16) != 0
                        {
                            continue;
                        }
                        let first_padding = u32::from(first_items != 0) * first_address_mod_16;
                        let second_padding = u32::from(second_items != 0) * second_address_mod_16;
                        let first_end = first_padding + first_items * item_size;
                        let second_begin = second_buffer_offset + second_padding;
                        let second_end = second_begin + second_items * item_size;
                        assert!(first_end <= second_begin);
                        assert!(second_end <= whole_buffer_bytes);
                    }
                }
            }
        }
    }
}
