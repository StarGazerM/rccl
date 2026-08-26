targets [sm80, sm89, sm90, sm100];

support span { rust: slice; ownership: borrowed; capability: contiguous; }
support array { rust: array; ownership: owned; capability: contiguous; }
support tuple { rust: tuple; ownership: owned; capability: product; }
support optional { rust: option; ownership: owned; capability: sum; }
support expected { rust: result; ownership: owned; capability: sum; }
support uninitialized { rust: maybe_uninit; ownership: owned; capability: storage; }
support device_vector { rust: device_buffer; ownership: owned; capability: contiguous; }
support device_reference { rust: indexed_access; ownership: borrowed; capability: proxy; }
support mdspan { rust: multidimensional_view; ownership: borrowed; capability: strided; }
support complex { rust: complex_value; ownership: owned; capability: scalar; }
support key_value_pair { rust: named_product; ownership: owned; capability: product; }
support double_buffer { rust: buffer_pair; ownership: borrowed; capability: storage; }
support future_value { rust: device_scalar; ownership: borrowed; capability: input; }
support temp_storage { rust: storage_layout; ownership: borrowed; capability: storage; }
support callable { rust: closure; ownership: owned; capability: callable; }
support input_range { rust: device_source; ownership: borrowed; capability: input; }
support output_range { rust: device_sink; ownership: borrowed; capability: output; }
support disjoint_output { rust: disjoint_slice; ownership: borrowed; capability: output; }
support offset { rust: integer; ownership: owned; capability: index; }
support device_bool { rust: rust_bool; ownership: owned; capability: scalar; }

schedule radix_sort {
    target sm80 { block_threads: 256; items_per_thread: 12; min_blocks_per_sm: 2; radix_bits: 8; unroll: 4; }
    target sm89 { block_threads: 256; items_per_thread: 16; min_blocks_per_sm: 2; radix_bits: 8; unroll: 4; }
    target sm90 { block_threads: 384; items_per_thread: 16; min_blocks_per_sm: 2; radix_bits: 8; unroll: 4; }
    target sm100 { block_threads: 512; items_per_thread: 16; min_blocks_per_sm: 1; radix_bits: 8; unroll: 4; }
}

schedule merge {
    target sm80 { block_threads: 256; items_per_thread: 15; fallback_threads: 128; unroll: 15; }
    target sm89 { block_threads: 256; items_per_thread: 15; fallback_threads: 128; unroll: 15; }
    target sm90 { block_threads: 256; items_per_thread: 15; fallback_threads: 128; unroll: 15; }
    target sm100 { block_threads: 256; items_per_thread: 15; fallback_threads: 128; unroll: 15; }
}

schedule comparison_sort {
    target sm80 { block_threads: 256; items_per_thread: 1; min_blocks_per_sm: 1; }
    target sm89 { block_threads: 256; items_per_thread: 1; min_blocks_per_sm: 1; }
    target sm90 { block_threads: 256; items_per_thread: 1; min_blocks_per_sm: 1; }
    target sm100 { block_threads: 256; items_per_thread: 1; min_blocks_per_sm: 1; }
}

primitive thread_load { status: planned; scope: thread; requires: []; schedule: none; kernels: []; }
primitive thread_store { status: planned; scope: thread; requires: []; schedule: none; kernels: []; }
primitive thread_reduce { status: planned; scope: thread; requires: []; schedule: none; kernels: []; }
primitive thread_scan { status: planned; scope: thread; requires: []; schedule: none; kernels: []; }
primitive thread_search { status: planned; scope: thread; requires: []; schedule: none; kernels: []; }
primitive thread_sort { status: planned; scope: thread; requires: []; schedule: none; kernels: []; }
primitive thread_simd { status: planned; scope: thread; requires: []; schedule: none; kernels: []; }
primitive thread_operators { status: native_rust; scope: thread; requires: []; schedule: none; kernels: []; }

primitive warp_load { status: planned; scope: warp; requires: [thread_load]; schedule: none; kernels: []; }
primitive warp_store { status: planned; scope: warp; requires: [thread_store]; schedule: none; kernels: []; }
primitive warp_exchange { status: planned; scope: warp; requires: [warp_load, warp_store]; schedule: none; kernels: []; }
primitive warp_reduce { status: planned; scope: warp; requires: [thread_reduce]; schedule: none; kernels: []; }
primitive warp_reduce_batched { status: planned; scope: warp; requires: [warp_reduce]; schedule: none; kernels: []; }
primitive warp_scan { status: planned; scope: warp; requires: [thread_scan]; schedule: none; kernels: []; }
primitive warp_bitonic_sort { status: planned; scope: warp; requires: [thread_sort, warp_exchange]; schedule: none; kernels: []; }
primitive warp_merge_sort { status: backend_blocked; scope: warp; requires: [thread_sort, warp_exchange]; schedule: none; kernels: []; }
primitive warp_utils { status: planned; scope: warp; requires: []; schedule: none; kernels: []; }

primitive block_load { status: planned; scope: block; requires: [warp_load]; schedule: none; kernels: []; }
primitive block_load_to_shared { status: planned; scope: block; requires: [block_load]; schedule: none; kernels: []; }
primitive block_store { status: planned; scope: block; requires: [warp_store]; schedule: none; kernels: []; }
primitive block_exchange { status: planned; scope: block; requires: [warp_exchange]; schedule: none; kernels: []; }
primitive block_shuffle { status: planned; scope: block; requires: [block_exchange]; schedule: none; kernels: []; }
primitive block_reduce { status: planned; scope: block; requires: [warp_reduce]; schedule: none; kernels: []; }
primitive block_scan { status: planned; scope: block; requires: [warp_scan]; schedule: none; kernels: []; }
primitive block_adjacent_difference { status: planned; scope: block; requires: [block_load, block_store]; schedule: none; kernels: []; }
primitive block_discontinuity { status: planned; scope: block; requires: [block_adjacent_difference]; schedule: none; kernels: []; }
primitive block_merge_sort { status: planned; scope: block; requires: [warp_merge_sort, block_exchange]; schedule: none; kernels: []; }
primitive block_radix_rank { status: planned; scope: block; requires: [block_scan]; schedule: none; kernels: []; }
primitive block_radix_sort { status: planned; scope: block; requires: [block_radix_rank, block_exchange]; schedule: none; kernels: []; }
primitive radix_rank_sort_operations { status: planned; scope: block; requires: [block_radix_rank]; schedule: none; kernels: []; }
primitive block_histogram { status: planned; scope: block; requires: [block_reduce]; schedule: none; kernels: []; }
primitive block_run_length_decode { status: planned; scope: block; requires: [block_scan]; schedule: none; kernels: []; }
primitive block_topk { status: planned; scope: block; requires: [block_merge_sort]; schedule: none; kernels: []; }
primitive block_raking_layout { status: planned; scope: block; requires: []; schedule: none; kernels: []; }

primitive device_for { status: planned; scope: device; requires: [block_load]; schedule: none; kernels: [for_each]; }
primitive device_transform { status: planned; scope: device; requires: [device_for]; schedule: none; kernels: [transform]; }
primitive device_copy { status: planned; scope: device; requires: [block_load, block_store]; schedule: none; kernels: [copy]; }
primitive device_memcpy { status: planned; scope: device; requires: [device_copy]; schedule: none; kernels: [memcpy]; }
primitive device_find { status: planned; scope: device; requires: [block_reduce]; schedule: none; kernels: [find]; }
primitive device_reduce { status: planned; scope: device; requires: [block_reduce]; schedule: none; kernels: [reduce]; }
primitive device_scan { status: planned; scope: device; requires: [block_scan]; schedule: none; kernels: [scan]; }
primitive device_segmented_reduce { status: planned; scope: device; requires: [device_reduce]; schedule: none; kernels: [segmented_reduce]; }
primitive device_segmented_scan { status: planned; scope: device; requires: [device_scan]; schedule: none; kernels: [segmented_scan]; }
primitive device_adjacent_difference { status: planned; scope: device; requires: [block_adjacent_difference]; schedule: none; kernels: [adjacent_difference]; }
primitive device_select { status: planned; scope: device; requires: [device_scan]; schedule: none; kernels: [select]; }
primitive device_partition { status: planned; scope: device; requires: [device_select]; schedule: none; kernels: [partition]; }
primitive device_run_length_encode { status: planned; scope: device; requires: [block_discontinuity, device_scan]; schedule: none; kernels: [run_length_encode]; }
primitive device_histogram { status: planned; scope: device; requires: [block_histogram, device_reduce]; schedule: none; kernels: [histogram]; }
primitive device_merge { status: backend_blocked; scope: device; requires: [block_load, block_store]; schedule: merge; kernels: [partition, merge_keys, merge_pairs]; }
primitive device_merge_sort { status: backend_blocked; scope: device; requires: [device_merge]; schedule: comparison_sort; kernels: [merge_pass]; }
primitive device_radix_sort { status: backend_blocked; scope: device; requires: [block_radix_sort, device_scan]; schedule: radix_sort; kernels: [histogram, prefix, scatter]; }
primitive device_segmented_radix_sort { status: planned; scope: device; requires: [device_radix_sort]; schedule: radix_sort; kernels: [segmented_histogram, segmented_scatter]; }
primitive device_segmented_sort { status: planned; scope: device; requires: [device_merge_sort]; schedule: comparison_sort; kernels: [segmented_merge_pass]; }
primitive device_topk { status: planned; scope: device; requires: [block_topk, device_select]; schedule: none; kernels: [topk]; }
primitive device_batched_topk { status: planned; scope: device; requires: [device_topk]; schedule: none; kernels: [batched_topk]; }

primitive arg_index_input_iterator { status: planned; scope: iterator; requires: []; schedule: none; kernels: []; }
primitive cache_modified_input_iterator { status: planned; scope: iterator; requires: []; schedule: none; kernels: []; }
primitive cache_modified_output_iterator { status: planned; scope: iterator; requires: []; schedule: none; kernels: []; }
primitive tex_obj_input_iterator { status: planned; scope: iterator; requires: []; schedule: none; kernels: []; }
