const PROGRAM: &str = include_str!("program.rs");
const TRANSFORM: &str = include_str!("transform.rs");

#[test]
fn fixed_cub_wrappers_are_declarative_not_quote_only_proc_macros() {
    let all = [PROGRAM, TRANSFORM].join("\n");
    assert!(PROGRAM.contains("macro_rules! cub_program"));
    assert!(TRANSFORM.contains("macro_rules! __device_transform_repeat_in_place"));
    assert!(!all.contains("quote!"));
    assert!(!all.contains("syn::"));
    assert!(!all.contains("CUDA_OXIDE_TARGET"));
    assert!(!all.contains("sm_"));
}

#[test]
fn program_has_one_exact_provenance_bind_and_transform_stays_safe() {
    assert_eq!(PROGRAM.matches("unsafe").count(), 1);
    assert!(PROGRAM.contains("__rccl_cub_kernels::load(__rccl_context)"));
    assert!(PROGRAM.contains("__device_kernels!(__rccl_cub_kernels)"));
    assert!(!TRANSFORM.contains("unsafe"));
    assert!(!TRANSFORM.contains("*mut"));
    assert!(!TRANSFORM.contains("*const"));
    assert!(TRANSFORM.contains("prepare_transform_repeat_in_place_for"));
}
