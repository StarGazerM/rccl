//! Host-oracle unit tests corresponding to pinned CCCL WarpReduce evidence.
//!
//! Source commit: `571f2fc3bc53cd710e306ad43d58995c1fe4219f`.
//! Sources: `cub/cub/warp/warp_reduce.cuh` and
//! `cub/test/warp/catch2_test_warp_reduce.cu`.

fn reference_groups<T: Copy>(
    input: &[T],
    logical_width: usize,
    valid_items: usize,
    op: impl Fn(T, T) -> T + Copy,
) -> Vec<T> {
    input
        .chunks_exact(logical_width)
        .map(|group| group[1..valid_items].iter().copied().fold(group[0], op))
        .collect()
}

#[test]
fn pinned_overview_reference_values_are_exact() {
    let input = (0..128).collect::<Vec<i32>>();
    assert_eq!(
        reference_groups(&input, 32, 32, |left, right| left + right),
        [496, 1520, 2544, 3568]
    );
}

#[test]
fn partial_reference_excludes_inactive_lanes() {
    let input = (0..64).collect::<Vec<i32>>();
    assert_eq!(
        reference_groups(&input, 32, 4, |left, right| left + right),
        [6, 134]
    );
}

#[test]
fn multiple_item_reference_keeps_thread_items_in_order() {
    let input = (0..128).collect::<Vec<i32>>();
    assert_eq!(
        reference_groups(&input, 128, 128, |left, right| left + right),
        [8128]
    );
}
