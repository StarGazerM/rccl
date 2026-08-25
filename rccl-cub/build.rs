use std::{env, fs, path::PathBuf};

fn target_sm() -> u32 {
    let Some(target) = env::var("CUDA_OXIDE_TARGET").ok() else {
        return 80;
    };
    let Some(suffix) = target.strip_prefix("sm_") else {
        panic!("CUDA_OXIDE_TARGET must have the form sm_<digits>, optionally followed by a or f");
    };
    let digits = suffix.trim_end_matches(['a', 'f']);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        panic!("CUDA_OXIDE_TARGET must have the form sm_<digits>, optionally followed by a or f");
    }
    let sm = digits
        .parse::<u32>()
        .expect("CUDA_OXIDE_TARGET architecture must fit in u32");
    assert!(sm >= 80, "RCCL requires sm_80 or newer");
    sm
}

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_OXIDE_TARGET");
    let sm = target_sm();
    let block_threads = if sm >= 100 {
        512
    } else if sm >= 90 {
        384
    } else {
        256
    };
    let items_per_thread = if sm >= 89 { 16 } else { 12 };
    let min_blocks_per_sm = if sm >= 100 { 1 } else { 2 };
    let output = format!(
        "pub const TARGET_SM: u32 = {sm};\n\
         pub const RADIX_BLOCK_THREADS: u32 = {block_threads};\n\
         pub const RADIX_ITEMS_PER_THREAD: u32 = {items_per_thread};\n\
         pub const RADIX_MIN_BLOCKS_PER_SM: u32 = {min_blocks_per_sm};\n"
    );
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    fs::write(out_dir.join("target_schedule.rs"), output)
        .expect("failed to write RCCL target schedule");
}
