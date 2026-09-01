const BACKENDS: &str = include_str!("backends.rs");
const HISTOGRAM: &str = include_str!("../histogram.rs");
const PROGRAM: &str = include_str!("program.rs");
const TRANSFORM: &str = include_str!("transform.rs");

fn body<'a>(source: &'a str, name: &str) -> &'a str {
    let marker = format!("macro_rules! {name}");
    let start = source.find(&marker).expect("macro must exist");
    let rest = &source[start + marker.len()..];
    let end = rest.find("#[doc(hidden)]").unwrap_or(rest.len());
    &rest[..end]
}

fn exported_body<'a>(source: &'a str, name: &str) -> &'a str {
    let marker = format!("macro_rules! {name}");
    let start = source.find(&marker).expect("macro must exist");
    let rest = &source[start + marker.len()..];
    let end = rest.find("#[macro_export]").unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn declarative_backends_reach_the_exact_cub_operation_families() {
    let expected = [
        ("__thrust_sort_backend", "device_radix_sort_keys!"),
        (
            "__thrust_comparison_sort_backend",
            "device_merge_sort_keys!",
        ),
        ("__thrust_sort_pairs_backend", "device_radix_sort_pairs!"),
        (
            "__thrust_comparison_sort_pairs_backend",
            "device_merge_sort_pairs!",
        ),
        ("__thrust_merge_backend", "device_merge_keys!"),
        ("__thrust_merge_pairs_backend", "device_merge_pairs!"),
    ];
    for (backend, cub) in expected {
        assert!(
            body(BACKENDS, backend).contains(cub),
            "{backend} stopped lowering to {cub}"
        );
    }
}

#[test]
fn backend_catalog_covers_every_implemented_primitive_pair() {
    let catalog = body(BACKENDS, "__thrust_backends");
    assert_eq!(catalog.matches("__thrust_sort_backend!").count(), 6);
    assert_eq!(
        catalog.matches("__thrust_comparison_sort_backend!").count(),
        14
    );
    assert_eq!(catalog.matches("__thrust_merge_backends!").count(), 1);
    assert!(!catalog.contains("__thrust_merge_backend!"));
    assert_eq!(catalog.matches("__thrust_sort_pairs_for_key!").count(), 6);
    assert_eq!(
        catalog
            .matches("__thrust_comparison_sort_pairs_for_narrow_key!")
            .count(),
        7
    );
    assert_eq!(
        catalog
            .matches("__thrust_comparison_sort_pairs_for_wide_key!")
            .count(),
        7
    );
    assert!(!catalog.contains("__thrust_merge_pairs_for_key!"));
    assert_eq!(catalog.matches("__thrust_set_backends!").count(), 1);
    assert!(!catalog.contains("__thrust_set_backend!"));
    assert!(!catalog.contains("__thrust_set_pairs_for_key!"));

    let set_catalog = body(BACKENDS, "__thrust_set_backends");
    assert_eq!(set_catalog.matches("__thrust_set_backend!").count(), 22);
    assert_eq!(
        set_catalog
            .matches("__thrust_arithmetic_set_backend!")
            .count(),
        12
    );
    assert_eq!(
        set_catalog.matches("__thrust_set_pairs_for_key!").count(),
        14
    );

    let merge_catalog = body(BACKENDS, "__thrust_merge_backends");
    assert_eq!(merge_catalog.matches("__thrust_merge_backend!").count(), 14);
    assert_eq!(
        merge_catalog
            .matches("__thrust_merge_pairs_for_key!")
            .count(),
        14
    );

    let sort_values = body(BACKENDS, "__thrust_sort_pairs_for_key");
    let narrow_comparison_sort_values =
        body(BACKENDS, "__thrust_comparison_sort_pairs_for_narrow_key");
    let wide_comparison_sort_values = body(BACKENDS, "__thrust_comparison_sort_pairs_for_wide_key");
    let merge_values = body(BACKENDS, "__thrust_merge_pairs_for_key");
    let set_values = body(BACKENDS, "__thrust_set_pairs_for_key");
    assert_eq!(
        sort_values.matches("__thrust_sort_pairs_backend!").count(),
        12
    );
    assert!(sort_values.contains("$module, $key, usize"));
    assert!(sort_values.contains("$module, $key, isize"));
    assert!(!sort_values.contains("$module, $key, u128"));
    assert!(!sort_values.contains("$module, $key, i128"));
    assert_eq!(
        narrow_comparison_sort_values
            .matches("__thrust_comparison_sort_pairs_backend!")
            .count(),
        12
    );
    assert!(!narrow_comparison_sort_values.contains("$module, $key, u128"));
    assert!(!narrow_comparison_sort_values.contains("$module, $key, i128"));
    assert_eq!(
        wide_comparison_sort_values
            .matches("__thrust_comparison_sort_pairs_backend!")
            .count(),
        2
    );
    assert!(wide_comparison_sort_values.contains("__thrust_comparison_sort_pairs_for_narrow_key!"));
    assert!(wide_comparison_sort_values.contains("$module, $key, u128"));
    assert!(wide_comparison_sort_values.contains("$module, $key, i128"));
    assert_eq!(
        merge_values
            .matches("__thrust_merge_pairs_backend!")
            .count(),
        14
    );
    assert_eq!(
        set_values.matches("__thrust_set_pairs_backend!").count(),
        14
    );
}

#[test]
fn fixed_scaffolding_stays_declarative_policy_free_and_safe() {
    let all = [BACKENDS, HISTOGRAM, PROGRAM, TRANSFORM].join("\n");
    for forbidden in [
        "quote!",
        "syn::",
        "CUDA_OXIDE_TARGET",
        "CudaTarget",
        "sm_",
        "#[kernel",
        "TypeId",
        "Arc<",
        "*mut",
        "*const",
    ] {
        assert!(!all.contains(forbidden), "found forbidden `{forbidden}`");
    }
    assert!(PROGRAM.contains("__device_kernels!"));
    assert!(PROGRAM.contains("__rccl_thrust_kernels; histogram"));
    assert!(PROGRAM.contains("__rccl_thrust_kernels; set_operations"));
    assert!(PROGRAM.contains("__thrust_backends!"));
    assert!(PROGRAM.contains("__thrust_set_backends!"));
    assert!(PROGRAM.contains("__thrust_merge_backends!"));
    assert!(TRANSFORM.contains("__device_transform_repeat_in_place!"));
    for (thrust, cub) in [
        ("histogram_even", "device_histogram_even!"),
        ("histogram_range", "device_histogram_range!"),
        (
            "histogram_even_arithmetic",
            "device_histogram_even_arithmetic!",
        ),
        (
            "histogram_range_arithmetic",
            "device_histogram_range_arithmetic!",
        ),
    ] {
        assert!(
            exported_body(HISTOGRAM, thrust).contains(cub),
            "{thrust} stopped lowering to {cub}"
        );
    }
    assert_eq!(HISTOGRAM.matches("__synchronize(__rccl_result)").count(), 4);
    assert!(!HISTOGRAM.contains("DeviceHistogramWorkspace"));
    assert_eq!(PROGRAM.matches("unsafe").count(), 1);
}
