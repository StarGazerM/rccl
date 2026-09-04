const CUDA: &str = include_str!("cuda.rs");
const COPY: &str = include_str!("copy.rs");
const ORDERING: &str = include_str!("ordering.rs");
const MERGE: &str = include_str!("merge.rs");
const SET: &str = include_str!("set.rs");
const TRANSFORM: &str = include_str!("transform.rs");
const GENERATION: &str = include_str!("generation.rs");
const HISTOGRAM: &str = include_str!("histogram.rs");
const REDUCE: &str = include_str!("reduce.rs");
const GROUPED_REDUCTION: &str = include_str!("grouped_reduction.rs");
const RECORD_SORT: &str = include_str!("record_sort.rs");

#[test]
fn public_algorithm_calls_have_no_module_or_policy_type_parameter() {
    for source in [
        CUDA,
        COPY,
        ORDERING,
        MERGE,
        SET,
        TRANSFORM,
        HISTOGRAM,
        GENERATION,
        REDUCE,
        GROUPED_REDUCTION,
        RECORD_SORT,
    ] {
        assert!(!source.contains("Execution<'_, '_, A, M>"));
        assert!(!source.contains("pub fn write_into<A, M>"));
        assert!(!source.contains("Backend<"));
        assert!(!source.contains("#[macro_export]"));
    }
}

#[test]
fn program_is_a_normal_loader_not_a_codegen_boundary() {
    assert!(CUDA.contains("pub fn load("));
    for forbidden in [
        "__device_kernels!",
        "__thrust_backends!",
        "proc_macro",
        "quote!",
        "TokenStream",
    ] {
        assert!(!CUDA.contains(forbidden), "found obsolete `{forbidden}`");
    }
}

#[test]
fn public_signatures_do_not_reach_through_cubs_device_private_namespace() {
    for source in [
        ORDERING,
        MERGE,
        SET,
        TRANSFORM,
        HISTOGRAM,
        GENERATION,
        REDUCE,
        GROUPED_REDUCTION,
        RECORD_SORT,
    ] {
        assert!(!source.contains("rccl_cub::__private"));
    }
    assert!(!CUDA.contains("pub fn module("));
}

#[test]
fn transform_generation_histogram_and_reduce_are_ordinary_functions() {
    assert!(TRANSFORM.contains("pub fn transform<T, F, A>"));
    assert!(GENERATION.contains("pub fn tabulate<T, F, A>"));
    assert!(GENERATION.contains("pub fn fill<T, A>"));
    assert!(GENERATION.contains("pub fn sequence<T, A>"));
    assert!(HISTOGRAM.contains("pub fn histogram_even<"));
    assert!(REDUCE.contains("pub fn reduce<T, A>"));
    assert!(REDUCE.contains("pub trait DeviceReduceExt"));
    assert!(!TRANSFORM.contains("macro_rules!"));
    assert!(!HISTOGRAM.contains("macro_rules!"));
    assert!(!REDUCE.contains("macro_rules!"));
}

#[test]
fn async_fill_retains_execution_and_output_in_general_completion() {
    assert!(GENERATION.contains("pub fn fill_async<"));
    assert!(GENERATION.contains("fill_range_async(execution, output.view_mut(), value)"));
    assert!(COPY.contains("execution.submit((input, output)"));
    assert!(GENERATION.contains("fill_async(execution, output, value).wait()"));
    assert!(CUDA.contains("pub unsafe fn submit<'borrow, Resources, T, E, F>"));
}

#[test]
fn synchronous_transform_waits_for_stream_completion() {
    assert!(TRANSFORM.contains(".synchronize(result)"));
    assert!(!TRANSFORM.contains(".map_err(TransformError::Launch)"));
}

#[test]
fn rust_flavored_transform_api_stays_method_based() {
    assert!(TRANSFORM.contains("pub fn map_with_value_in_place<T, F, A>"));
    assert!(TRANSFORM.contains("pub trait DeviceTransformExt<T: DeviceCopy>"));
    assert!(TRANSFORM.contains("transform(execution, output, RepeatN::new(value, len), op)"));
}

#[test]
fn rust_device_slice_compat_layer_keeps_view_aliases() {
    assert!(include_str!("lib.rs").contains("DeviceSlice, DeviceSliceMut"));
    assert!(include_str!("range.rs").contains("pub type DeviceSlice<'a, T> = DeviceView<'a, T>;"));
    assert!(include_str!("range.rs").contains("fn as_device_slice(&self) -> DeviceSlice<'_, T>"));
    assert!(
        include_str!("range.rs")
            .contains("fn as_device_slice_mut(&mut self) -> DeviceSliceMut<'_, T>")
    );
}

#[test]
fn generation_matches_pinned_thrust_dependency_chain() {
    assert!(GENERATION.contains("tabulate_range(execution, output, move |index| {"));
    assert!(GENERATION.contains("T::sequence_value(init, step, index)"));
    assert!(GENERATION.contains("pub trait DeviceGenerateExt<T: DeviceCopy>"));
    assert!(!GENERATION.contains("to_host"));
    assert!(!GENERATION.contains("from_host"));
}

#[test]
fn histogram_facade_uses_execution_allocator() {
    assert!(HISTOGRAM.contains("DeviceHistogram::<C, ACTIVE, _>::new_in("));
    assert!(HISTOGRAM.contains("device_histogram_even_workspace_in"));
    assert!(HISTOGRAM.contains("device_histogram_even_in"));
    assert!(!HISTOGRAM.contains("DeviceHistogram::<C, ACTIVE>::new(execution.stream()"));
}

#[test]
fn reduce_facade_uses_cub_device_reduce_sum_workspace() {
    assert!(REDUCE.contains("device_reduce_sum_workspace_in"));
    assert!(REDUCE.contains("device_reduce_sum_in"));
    assert!(REDUCE.contains("output.get()"));
    assert!(REDUCE.contains("execution.synchronize(result)"));
    assert!(!REDUCE.contains("to_host_vec"));
    assert!(!REDUCE.contains("from_host"));
}
