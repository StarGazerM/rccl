const CUDA: &str = include_str!("cuda.rs");
const ORDERING: &str = include_str!("ordering.rs");
const MERGE: &str = include_str!("merge.rs");
const SET: &str = include_str!("set.rs");
const TRANSFORM: &str = include_str!("transform.rs");
const HISTOGRAM: &str = include_str!("histogram.rs");

#[test]
fn public_algorithm_calls_have_no_module_or_policy_type_parameter() {
    for source in [CUDA, ORDERING, MERGE, SET, TRANSFORM, HISTOGRAM] {
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
    for source in [ORDERING, MERGE, SET, TRANSFORM, HISTOGRAM] {
        assert!(!source.contains("rccl_cub::__private"));
    }
    assert!(!CUDA.contains("pub fn module("));
}

#[test]
fn transform_and_histogram_are_ordinary_functions() {
    assert!(TRANSFORM.contains("pub fn transform<T, F, A>"));
    assert!(HISTOGRAM.contains("pub fn histogram_even<"));
    assert!(!TRANSFORM.contains("macro_rules!"));
    assert!(!HISTOGRAM.contains("macro_rules!"));
}
