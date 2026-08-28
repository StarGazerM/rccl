use super::*;

#[test]
fn typed_layout_uses_element_size_and_alignment() {
    let layout = DeviceBuffer::<u64, CudaMemoryResource>::layout_for(17).unwrap();
    assert_eq!(layout.size(), 17 * 8);
    assert_eq!(layout.align(), 8);
}

#[test]
fn typed_layout_detects_size_overflow() {
    assert!(matches!(
        DeviceBuffer::<u64, CudaMemoryResource>::layout_for(usize::MAX),
        Err(MemoryError::SizeOverflow)
    ));
}

#[test]
fn cub_region_layout_matches_rounded_alias_temporaries_contract() {
    let (offsets, total) =
        temporary_layout([1003 * 4, 1024 * 8, 33 * 4, 7 * 4], [128, 256, 256, 256]).unwrap();
    assert_eq!(offsets, [0, 4096, 12288, 12544]);
    assert_eq!(total, 13055);
}

#[test]
fn thrust_pair_prefix_preserves_128_byte_rounding_before_cub_alignment() {
    let (offsets, total) = temporary_layout([33 * 4, 33 * 8, 8 * 8], [128, 128, 256]).unwrap();
    assert_eq!(offsets, [0, 256, 768]);
    assert_eq!(total, 1279);
}

#[test]
fn default_pool_policy_retains_and_reuses_memory() {
    let config = PoolConfig::default();
    assert_eq!(config.release_threshold, u64::MAX);
    assert!(config.follow_event_dependencies);
    assert!(config.opportunistic_reuse);
    assert!(config.internal_dependencies);
}
