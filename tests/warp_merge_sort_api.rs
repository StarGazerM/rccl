use rccl::cub::{WarpMergeSort, WarpMergeSortConfig, WarpMergeSortTempStorage};

type DefaultLogicalWarp = WarpMergeSort<u32, 4>;
type ExplicitPairs = WarpMergeSort<i64, 7, 4, u32>;

fn assert_storage_type<T>() {}

#[test]
fn cub_generic_defaults_and_config_are_public() {
    assert_eq!(DefaultLogicalWarp::ITEMS_PER_THREAD, 4);
    assert_eq!(DefaultLogicalWarp::LOGICAL_WARP_THREADS, 32);
    assert_eq!(DefaultLogicalWarp::ITEMS_PER_TILE, 128);
    assert_eq!(ExplicitPairs::ITEMS_PER_THREAD, 7);
    assert_eq!(ExplicitPairs::LOGICAL_WARP_THREADS, 4);
    assert_eq!(ExplicitPairs::ITEMS_PER_TILE, 28);
    assert_storage_type::<WarpMergeSortTempStorage<u32, 4>>();
    assert_storage_type::<WarpMergeSortTempStorage<i64, 7, 4, u32>>();
}
