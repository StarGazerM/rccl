const ORDERING: &str = include_str!("ordering.rs");
const MERGE: &str = include_str!("merge.rs");
const PROGRAM: &str = include_str!("program.rs");
const TRANSFORM: &str = include_str!("transform.rs");

fn body<'a>(source: &'a str, name: &str) -> &'a str {
    let marker = format!("macro_rules! {name}");
    let start = source.find(&marker).expect("macro must exist");
    let rest = &source[start + marker.len()..];
    let end = rest.find("#[macro_export]").unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn every_ordering_form_reaches_its_cccl_cub_dependency() {
    let expected = [
        ("sort", "device_radix_sort_keys!"),
        ("sort_by", "device_merge_sort_keys!"),
        ("stable_sort", "device_radix_sort_keys!"),
        ("stable_sort_by", "device_stable_merge_sort_keys!"),
        ("sort_by_key", "device_radix_sort_pairs!"),
        ("sort_by_key_with", "device_merge_sort_pairs!"),
        ("stable_sort_by_key", "device_radix_sort_pairs!"),
        ("stable_sort_by_key_with", "device_stable_merge_sort_pairs!"),
    ];
    for (name, cub) in expected {
        let expansion = body(ORDERING, name);
        assert!(expansion.contains(cub), "{name} stopped lowering to {cub}");
        assert!(expansion.contains("__synchronize"));
        assert!(expansion.contains("TemporaryWorkspace"));
    }
}

#[test]
fn every_merge_form_reaches_device_merge() {
    for name in ["merge", "merge_by"] {
        assert!(body(MERGE, name).contains("device_merge_keys!"));
    }
    for name in ["merge_by_key", "merge_by_key_with"] {
        assert!(body(MERGE, name).contains("device_merge_pairs!"));
    }
}

#[test]
fn fixed_thrust_lowering_stays_declarative_and_policy_free() {
    let all = [ORDERING, MERGE, PROGRAM, TRANSFORM].join("\n");
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
    assert!(TRANSFORM.contains("__device_transform_repeat_in_place!"));
    assert_eq!(PROGRAM.matches("unsafe").count(), 1);
}
