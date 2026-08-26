# Pinned CUB example and API-test contract

Baseline: NVIDIA CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`.

This is RCCL's source-of-truth input for porting CUB examples and focused
tests. It prevents an RCCL API sketch from becoming its own oracle. The
checked source inventory contains:

- 19 executable programs under `cub/examples`;
- 378 `example-begin` documentation regions under `cub/examples` and
  `cub/test`;
- 49 `catch2_test_*_api.cu` or `catch2_test_*_env_api.cu` files; and
- 105 public device/warp methods exercised by those programs and API tests.

`scripts/check-cccl-cub-examples.sh` normalizes and hashes all four sets. A
changed filename, example identifier, API-test file, or exercised method makes
the default gate fail and requires this contract to be reviewed again. The
checker also requires the 105 methods to match the typed
`EXERCISED_OPERATIONS` compiler catalog exactly.

This contract is not implementation credit. RCCL currently has no public CUB
algorithm macro. An example is ported only after its operation has a real
cuda-oxide lowering and the Rust test asserts the corresponding observable
result.

## Executable example programs

```text
block/example_block_radix_sort.cu
block/example_block_reduce.cu
block/example_block_reduce_dyn_smem.cu
block/example_block_scan.cu
device/example_device_decoupled_look_back.cu
device/example_device_env_memory_resource.cu
device/example_device_partition_flagged.cu
device/example_device_partition_if.cu
device/example_device_radix_sort.cu
device/example_device_radix_sort_custom.cu
device/example_device_reduce.cu
device/example_device_reduce_env.cu
device/example_device_scan.cu
device/example_device_select_flagged.cu
device/example_device_select_if.cu
device/example_device_select_unique.cu
device/example_device_sort_find_non_trivial_runs.cu
device/example_device_topk_keys.cu
device/example_device_topk_pairs.cu
```

## API-test files

```text
catch2_test_block_scan_api.cu
catch2_test_device_adjacent_difference_api.cu
catch2_test_device_adjacent_difference_env_api.cu
catch2_test_device_batched_topk_api.cu
catch2_test_device_batched_topk_env_api.cu
catch2_test_device_copy_api.cu
catch2_test_device_copy_env_api.cu
catch2_test_device_copy_mdspan_api.cu
catch2_test_device_find_api.cu
catch2_test_device_find_bound_sorted_values_env_api.cu
catch2_test_device_find_env_api.cu
catch2_test_device_for_api.cu
catch2_test_device_for_each_in_extents_api.cu
catch2_test_device_for_each_in_layout_api.cu
catch2_test_device_for_env_api.cu
catch2_test_device_histogram_api.cu
catch2_test_device_histogram_env_api.cu
catch2_test_device_memcpy_env_api.cu
catch2_test_device_merge_api.cu
catch2_test_device_merge_env_api.cu
catch2_test_device_merge_sort_api.cu
catch2_test_device_merge_sort_env_api.cu
catch2_test_device_partition_api.cu
catch2_test_device_partition_env_api.cu
catch2_test_device_radix_sort_env_api.cu
catch2_test_device_reduce_arg_minmax_env_api.cu
catch2_test_device_reduce_by_key_env_api.cu
catch2_test_device_reduce_env_api.cu
catch2_test_device_run_length_encode_api.cu
catch2_test_device_run_length_encode_env_api.cu
catch2_test_device_scan_api.cu
catch2_test_device_scan_by_key_env_api.cu
catch2_test_device_scan_env_api.cu
catch2_test_device_segmented_radix_sort_api.cu
catch2_test_device_segmented_radix_sort_env_api.cu
catch2_test_device_segmented_reduce_api.cu
catch2_test_device_segmented_reduce_env_api.cu
catch2_test_device_segmented_scan_api.cu
catch2_test_device_segmented_scan_env_api.cu
catch2_test_device_segmented_sort_keys_env_api.cu
catch2_test_device_segmented_sort_pairs_env_api.cu
catch2_test_device_select_api.cu
catch2_test_device_select_env_api.cu
catch2_test_device_topk_api.cu
catch2_test_device_topk_env_api.cu
catch2_test_device_transform_api.cu
catch2_test_device_transform_env_api.cu
catch2_test_warp_scan_api.cu
warp/catch2_test_warp_reduce_batched_api.cu
```

## Exercised public operation shapes

The source scan yields these method families. Every method receives its own
operand schema; RCCL must not collapse a family into one generic parser.

| Family | Methods exercised by the pinned programs/API tests |
|---|---|
| `DeviceAdjacentDifference` | `SubtractLeft`, `SubtractLeftCopy`, `SubtractRight`, `SubtractRightCopy` |
| `DeviceBatchedTopK` | `MaxKeys`, `MaxPairs`, `MinKeys`, `MinPairs` |
| `DeviceCopy` | `Batched`, `Copy` |
| `DeviceFind` | `FindIf`, `LowerBound`, `LowerBoundSortedValues`, `UpperBound`, `UpperBoundSortedValues` |
| `DeviceFor` | `Bulk`, `ForEach`, `ForEachCopy`, `ForEachCopyN`, `ForEachInExtents`, `ForEachInLayout`, `ForEachN` |
| `DeviceHistogram` | `HistogramEven`, `HistogramRange`, `MultiHistogramEven`, `MultiHistogramRange` |
| `DeviceMemcpy` | `Batched` |
| `DeviceMerge` | `MergeKeys`, `MergePairs` |
| `DeviceMergeSort` | `SortKeys`, `SortKeysCopy`, `SortPairs`, `SortPairsCopy`, `StableSortKeys`, `StableSortKeysCopy`, `StableSortPairs` |
| `DevicePartition` | `Flagged`, `If` |
| `DeviceRadixSort` | `SortKeys`, `SortKeysDescending`, `SortPairs`, `SortPairsDescending` |
| `DeviceReduce` | `ArgMax`, `ArgMin`, `ArgMinMax`, `Max`, `Min`, `Reduce`, `ReduceByKey`, `Sum`, `TransformReduce` |
| `DeviceRunLengthEncode` | `Encode`, `NonTrivialRuns` |
| `DeviceScan` | `ExclusiveScan`, `ExclusiveScanByKey`, `ExclusiveSum`, `ExclusiveSumByKey`, `InclusiveScan`, `InclusiveScanByKey`, `InclusiveScanInit`, `InclusiveSum`, `InclusiveSumByKey` |
| `DeviceSegmentedRadixSort` | `SortKeys`, `SortKeysDescending`, `SortPairs`, `SortPairsDescending` |
| `DeviceSegmentedReduce` | `ArgMax`, `ArgMin`, `Max`, `Min`, `Reduce`, `Sum` |
| `DeviceSegmentedScan` | `ExclusiveSegmentedScan`, `ExclusiveSegmentedSum`, `InclusiveSegmentedScan`, `InclusiveSegmentedScanInit`, `InclusiveSegmentedSum` |
| `DeviceSegmentedSort` | `SortKeys`, `SortKeysDescending`, `SortPairs`, `SortPairsDescending`, `StableSortKeys`, `StableSortKeysDescending`, `StableSortPairs`, `StableSortPairsDescending` |
| `DeviceSelect` | `Flagged`, `FlaggedIf`, `If`, `Unique`, `UniqueByKey` |
| `DeviceTopK` | `MaxKeys`, `MaxPairs`, `MinKeys`, `MinPairs` |
| `DeviceTransform` | `Fill`, `Generate`, `Transform`, `TransformIf`, `TransformStableArgumentAddresses` |
| `WarpReduceBatched` | `Reduce`, `ReduceToBlocked`, `ReduceToStriped`, `Sum`, `SumToBlocked`, `SumToStriped` |

The 378 embedded regions additionally carry block and warp calls through local
type aliases, including the complete `BlockScan` scalar/array, aggregate,
initial-value, and prefix-callback forms. Those regions are intentionally
tracked by exact `path:identifier` digest rather than inferred from the device
method regex.

## Required porting discipline

For each operation slice:

1. cite the exact executable example and every related `example-begin` region;
2. cite all related API-test files, including explicit-storage and environment
   overloads;
3. derive the Rust operand shape from those sources before adding a parser;
4. add no public macro until the actual cuda-oxide lowering exists;
5. check complete outputs, counts, offsets, stability, key/value association,
   allocation balance, stream behavior, and determinism wherever upstream
   checks them; and
6. seed an implementation or launch fault and prove the relevant test fails,
   so a skipped launch, default-filled output, partial check, or host fallback
   cannot masquerade as a pass.
