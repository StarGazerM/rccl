cub [
    device_adjacent_difference,
    device_copy,
    device_find,
    device_for,
    device_merge,
    device_merge_sort,
    device_partition,
    device_radix_sort,
    device_reduce,
    device_scan,
    device_select,
    device_transform,
];

support complex { rust: complex_value; status: planned; }
support pair { rust: tuple; status: native_rust; }
support tuple { rust: tuple; status: native_rust; }
support functional { rust: closure; status: native_rust; }
support zip_function { rust: product_closure; status: planned; }
support random { rust: random_state; status: planned; }
support addressof { rust: borrow; status: native_rust; }
support advance { rust: index_arithmetic; status: native_rust; }
support distance { rust: range_length; status: native_rust; }
support allocate_unique { rust: affine_owner; status: planned; }
support device_delete { rust: drop; status: native_rust; }
support device_free { rust: drop; status: native_rust; }
support device_make_unique { rust: affine_owner; status: planned; }
support device_malloc { rust: device_buffer; status: implemented; }
support device_new { rust: maybe_uninit; status: planned; }
support memory { rust: ownership; status: native_rust; }
support device_allocator { rust: device_allocator; status: implemented; }
support device_malloc_allocator { rust: device_allocator; status: implemented; }
support device_new_allocator { rust: device_allocator; status: implemented; }
support universal_allocator { rust: managed_allocator; status: implemented; }
support per_device_resource { rust: allocator_value; status: implemented; }
support device_ptr { rust: device_borrow; status: planned; }
support device_reference { rust: indexed_access; status: planned; }
support universal_ptr { rust: managed_borrow; status: planned; }
support device_vector { rust: device_buffer; status: implemented; }
support host_vector { rust: vec; status: native_rust; }
support universal_vector { rust: managed_buffer; status: planned; }
support execution_policy { rust: execution_borrow; status: implemented; }
support system_error { rust: result; status: native_rust; }
support version { rust: crate_version; status: native_rust; }

support discard_block_engine { rust: random_state; status: planned; }
support linear_congruential_engine { rust: random_state; status: planned; }
support linear_feedback_shift_engine { rust: random_state; status: planned; }
support normal_distribution { rust: random_distribution; status: planned; }
support subtract_with_carry_engine { rust: random_state; status: planned; }
support uniform_int_distribution { rust: random_distribution; status: planned; }
support uniform_real_distribution { rust: random_distribution; status: planned; }
support xor_combine_engine { rust: random_state; status: planned; }

support mr_allocator { rust: device_allocator; status: implemented; }
support device_memory_resource { rust: device_allocator; status: implemented; }
support disjoint_pool { rust: pool_allocator; status: implemented; }
support disjoint_sync_pool { rust: pool_allocator; status: implemented; }
support disjoint_tls_pool { rust: thread_local_allocator; status: planned; }
support fancy_pointer_resource { rust: allocator_adaptor; status: planned; }
support host_memory_resource { rust: host_allocator; status: implemented; }
support memory_resource { rust: device_allocator; status: implemented; }
support new_resource { rust: construction_allocator; status: planned; }
support polymorphic_adaptor { rust: generic_allocator; status: native_rust; }
support pool { rust: pool_allocator; status: implemented; }
support pool_options { rust: pool_configuration; status: implemented; }
support sync_pool { rust: synchronized_allocator; status: implemented; }
support tls_pool { rust: thread_local_allocator; status: planned; }
support universal_memory_resource { rust: managed_allocator; status: implemented; }
support validator { rust: checked_allocator; status: implemented; }

adaptor constant_iterator { status: implemented; capabilities: [input, random_access]; }
adaptor counting_iterator { status: implemented; capabilities: [input, random_access]; }
adaptor discard_iterator { status: implemented; capabilities: [output, random_access]; }
adaptor iterator_adaptor { status: native_rust; capabilities: [input]; }
adaptor iterator_categories { status: native_rust; capabilities: [input]; }
adaptor iterator_facade { status: native_rust; capabilities: [input]; }
adaptor iterator_traits { status: native_rust; capabilities: [input]; }
adaptor iterator_traversal_tags { status: native_rust; capabilities: [input]; }
adaptor offset_iterator { status: implemented; capabilities: [input, output, random_access]; }
adaptor permutation_iterator { status: implemented; capabilities: [input, random_access, permutation]; }
adaptor retag { status: native_rust; capabilities: [input]; }
adaptor reverse_iterator { status: implemented; capabilities: [input, output, random_access]; }
adaptor shuffle_iterator { status: implemented; capabilities: [input, random_access, permutation]; }
adaptor strided_iterator { status: implemented; capabilities: [input, output, random_access, strided]; }
adaptor tabulate_output_iterator { status: implemented; capabilities: [output, random_access]; }
adaptor transform_input_output_iterator { status: implemented; capabilities: [input, output, random_access]; }
adaptor transform_iterator { status: implemented; capabilities: [input, random_access]; }
adaptor transform_output_iterator { status: implemented; capabilities: [output, random_access]; }
adaptor zip_iterator { status: implemented; capabilities: [input, output, random_access, product]; }

algorithm adjacent_difference {
    status: planned; form: input_output; requires: [device_adjacent_difference];
    select { otherwise => device_adjacent_difference; }
}
algorithm binary_search {
    status: planned; form: query; requires: [device_find];
    select { otherwise => device_find; }
}
algorithm copy {
    status: planned; form: input_output; requires: [device_copy];
    select { otherwise => device_copy; }
}
algorithm count {
    status: planned; form: reduction; requires: [device_reduce];
    select { otherwise => device_reduce; }
}
algorithm equal {
    status: planned; form: query; requires: [device_reduce];
    select { otherwise => device_reduce; }
}
algorithm extrema {
    status: planned; form: reduction; requires: [device_reduce];
    select { otherwise => device_reduce; }
}
algorithm fill {
    status: planned; form: output; requires: [device_for];
    select { otherwise => device_for; }
}
algorithm find {
    status: planned; form: query; requires: [device_find];
    select { otherwise => device_find; }
}
algorithm for_each {
    status: planned; form: input_output; requires: [device_for];
    select { otherwise => device_for; }
}
algorithm gather {
    status: planned; form: input_output; requires: [device_transform];
    select { otherwise => device_transform; }
}
algorithm generate {
    status: planned; form: output; requires: [device_for];
    select { otherwise => device_for; }
}
algorithm inner_product {
    status: planned; form: reduction; requires: [device_reduce];
    select { otherwise => device_reduce; }
}
algorithm logical {
    status: planned; form: query; requires: [device_reduce];
    select { otherwise => device_reduce; }
}
algorithm merge {
    status: backend_blocked; form: input_output; requires: [device_merge];
    select { otherwise => device_merge; }
}
algorithm mismatch {
    status: planned; form: query; requires: [device_find];
    select { otherwise => device_find; }
}
algorithm partition {
    status: planned; form: in_place; requires: [device_partition];
    select { otherwise => device_partition; }
}
algorithm reduce {
    status: planned; form: reduction; requires: [device_reduce];
    select { otherwise => device_reduce; }
}
algorithm remove {
    status: planned; form: in_place; requires: [device_select];
    select { otherwise => device_select; }
}
algorithm replace {
    status: planned; form: input_output; requires: [device_transform];
    select { otherwise => device_transform; }
}
algorithm reverse {
    status: planned; form: in_place; requires: [device_for];
    select { otherwise => device_for; }
}
algorithm scan {
    status: planned; form: input_output; requires: [device_scan];
    select { otherwise => device_scan; }
}
algorithm scatter {
    status: planned; form: input_output; requires: [device_transform];
    select { otherwise => device_transform; }
}
algorithm sequence {
    status: planned; form: output; requires: [device_for];
    select { otherwise => device_for; }
}
algorithm set_operations {
    status: planned; form: input_output; requires: [device_merge];
    select { otherwise => device_merge; }
}
algorithm shuffle {
    status: planned; form: in_place; requires: [device_for];
    select { otherwise => device_for; }
}
algorithm sort {
    status: backend_blocked; form: in_place; requires: [device_radix_sort, device_merge_sort];
    select { natural_radix_key => device_radix_sort; otherwise => device_merge_sort; }
}
algorithm swap {
    status: planned; form: input_output; requires: [device_transform];
    select { otherwise => device_transform; }
}
algorithm tabulate {
    status: planned; form: output; requires: [device_for];
    select { otherwise => device_for; }
}
algorithm transform {
    status: planned; form: input_output; requires: [device_transform];
    select { otherwise => device_transform; }
}
algorithm transform_reduce {
    status: planned; form: reduction; requires: [device_transform, device_reduce];
    select { fused => device_reduce; otherwise => device_transform; }
}
algorithm transform_scan {
    status: planned; form: input_output; requires: [device_transform, device_scan];
    select { fused => device_scan; otherwise => device_transform; }
}
algorithm uninitialized_copy {
    status: planned; form: input_output; requires: [device_copy];
    select { otherwise => device_copy; }
}
algorithm uninitialized_fill {
    status: planned; form: output; requires: [device_for];
    select { otherwise => device_for; }
}
algorithm unique {
    status: planned; form: in_place; requires: [device_select];
    select { otherwise => device_select; }
}
