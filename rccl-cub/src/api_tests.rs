const CUDA: &str = include_str!("cuda.rs");
const DISPATCH: &str = include_str!("dispatch.rs");
const HISTOGRAM: &str = include_str!("histogram_dispatch.rs");
const REDUCE: &str = include_str!("reduce.rs");
const REDUCE_KERNELS: &str = include_str!("reduce_kernels.rs");
const LIB: &str = include_str!("lib.rs");
const PRIVATE: &str = include_str!("__private.rs");
const DOUBLE_BUFFER: &str = include_str!("double_buffer.rs");

#[test]
fn fixed_kernels_are_loaded_by_an_ordinary_constructor() {
    assert!(CUDA.contains("pub fn load("));
    assert!(!CUDA.contains("pub struct KernelModules"));
    assert!(!CUDA.contains("pub fn module("));
    for source in [CUDA, DISPATCH, HISTOGRAM, REDUCE] {
        for forbidden in ["proc_macro", "TokenStream", "quote!", "CUDA_OXIDE_TARGET"] {
            assert!(!source.contains(forbidden), "found obsolete `{forbidden}`");
        }
    }
}

#[test]
fn launch_surface_can_retain_program_and_operation_resources() {
    assert!(CUDA.contains("pub unsafe fn submit<'borrow, Resources, T, E, F>"));
    assert!(CUDA.contains("CudaCompletion::from_submission(self.stream, (self, resources)"));
}

#[test]
fn cxx_read_only_range_descriptors_do_not_return() {
    for source in [CUDA, DISPATCH, HISTOGRAM, LIB, PRIVATE] {
        assert!(!source.contains("ReadOnlySlices"));
        assert!(!source.contains("ReadOnlyRange"));
    }
}

#[test]
fn operation_surface_is_functions_not_exported_macros() {
    for function in [
        "device_histogram_even",
        "device_merge_keys",
        "device_merge_sort_keys",
        "device_radix_sort_keys",
        "device_radix_sort_pairs",
        "device_reduce_sum",
    ] {
        assert!(
            LIB.contains(function),
            "missing ordinary function `{function}`"
        );
    }
    assert!(!LIB.contains("mod macros"));
    assert!(!DISPATCH.contains("#[macro_export]"));
    assert!(!HISTOGRAM.contains("#[macro_export]"));
}

#[test]
fn architecture_is_not_an_operation_type_parameter() {
    for source in [DISPATCH, HISTOGRAM, REDUCE] {
        assert!(!source.contains("Architecture"));
        assert!(!source.contains("TargetArchitecture"));
        assert!(!source.contains("ArchitecturePolicy"));
        assert!(!source.contains("Execution<'_, '_, A, M>"));
    }
}

#[test]
fn low_level_module_parameter_helpers_are_not_reexported() {
    for operation in [
        "device_histogram_even,",
        "device_histogram_range,",
        "device_merge_keys,",
        "device_merge_pairs,",
        "device_merge_sort_keys,",
        "device_merge_sort_pairs,",
        "device_radix_sort_keys,",
        "device_radix_sort_pairs,",
        "device_set_keys,",
        "device_set_pairs,",
    ] {
        assert!(
            !PRIVATE.contains(operation),
            "low-level operation `{operation}` leaked through __private"
        );
    }
}

#[test]
fn unchecked_double_buffer_constructor_stays_crate_private() {
    assert!(DOUBLE_BUFFER.contains("pub(crate) fn __from_equal_parts"));
    assert!(!DOUBLE_BUFFER.contains("pub fn __from_equal_parts"));
}

#[test]
fn device_reduce_sum_preserves_cub_two_pass_shape() {
    assert!(LIB.contains("device_reduce_sum"));
    assert!(REDUCE.contains("DeviceReduceSumWorkspaceCounts"));
    assert!(REDUCE.contains("REDUCE_SUBSCRIPTION_FACTOR"));
    assert!(REDUCE.contains("max_active_blocks_per_multiprocessor"));
    assert!(REDUCE.contains("multiprocessor_count"));
    assert!(REDUCE.contains("geometry.grid_size <= 1"));
    assert!(REDUCE.contains("reduce_sum_blocks"));
    assert!(REDUCE.contains("reduce_sum_single_tile"));
    assert!(REDUCE_KERNELS.contains("pub(crate) const REDUCE_SUBSCRIPTION_FACTOR: u32 = 5"));
    assert!(REDUCE_KERNELS.contains("tile_stride = grid_size * tile_items"));
    assert!(!REDUCE.contains("to_host_vec"));
}
