#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh

cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
if [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' "ordering parity requires pinned CCCL $cccl_revision" >&2
  exit 1
fi

foreign_compute="$(
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$foreign_compute" ]]; then
  printf '%s\n' 'ordering performance parity requires an exclusive GPU; active compute processes:' >&2
  printf '%s\n' "$foreign_compute" >&2
  exit 1
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-ordering-parity.XXXXXX")"
cleanup() {
  rm -rf -- "$build_dir"
  rm -f -- \
    device_ordering_benchmark.options \
    device_ordering_benchmark.target \
    device_merge_sort_pairs_benchmark.options \
    device_merge_sort_pairs_benchmark.target \
    ordering_benchmark.options \
    ordering_benchmark.target
}
trap cleanup EXIT

for source in benchmarks/merge-reference.cu benchmarks/ordering-reference.cu; do
  output="$build_dir/$(basename "$source" .cu)"
  nvcc -std=c++20 -O3 -arch="${RCCL_ARCH:-sm_89}" \
    -I"$cccl_root/cub" \
    -I"$cccl_root/thrust" \
    -I"$cccl_root/libcudacxx/include" \
    "$source" -o "$output"
done

cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
  --release --package rccl-cub --example device_ordering_benchmark
cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
  --release --package rccl-cub --example device_merge_sort_pairs_benchmark
cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
  --release --package rccl-thrust --example ordering_benchmark

items="${RCCL_ORDERING_ITEMS:-16777216}"
iterations="${RCCL_ORDERING_ITERATIONS:-31}"
rounds="${RCCL_ORDERING_ROUNDS:-5}"

cccl_merge="$build_dir/cccl-merge.txt"
cccl_ordering="$build_dir/cccl-ordering.txt"
rccl_cub="$build_dir/rccl-cub.txt"
rccl_cub_pairs="$build_dir/rccl-cub-pairs.txt"
rccl_thrust="$build_dir/rccl-thrust.txt"
for ((round = 0; round < rounds; ++round)); do
  "$build_dir/merge-reference" "$items" "$iterations" | tee -a "$cccl_merge"
  "$build_dir/ordering-reference" "$items" "$iterations" | tee -a "$cccl_ordering"
  ./target/release/examples/device_ordering_benchmark \
    --items "$items" --iterations "$iterations" | tee -a "$rccl_cub"
  ./target/release/examples/device_merge_sort_pairs_benchmark \
    --items "$items" --iterations "$iterations" | tee -a "$rccl_cub_pairs"
  ./target/release/examples/ordering_benchmark \
    --items "$items" --iterations "$iterations" | tee -a "$rccl_thrust"
done

median_metric() {
  local prefix="$1"
  local key="$2"
  local file="$3"
  awk -v prefix="$prefix" -v key="$key" '
    index($0, prefix) == 1 {
      for (field = 1; field <= NF; ++field) {
        split($field, pair, "=")
        if (pair[1] == key) {
          print pair[2] + 0
          seen = 1
        }
      }
    }
    END { if (!seen) exit 1 }
  ' "$file" | sort -g | awk '
    { samples[NR] = $1 }
    END {
      if (NR == 0) exit 1
      middle = int((NR + 1) / 2)
      if (NR % 2) value = samples[middle]
      else value = (samples[middle] + samples[middle + 1]) / 2
      printf "%.9f", value
    }
  '
}

require_equal_performance() {
  local name="$1"
  local cccl_ms="$2"
  local rccl_ms="$3"
  local limit=1.001
  awk -v name="$name" -v cccl="$cccl_ms" -v rccl="$rccl_ms" -v limit="$limit" 'BEGIN {
    ratio = rccl / cccl
    printf "%s median-of-medians ratio: %.6f (RCCL %.6f ms / CCCL %.6f ms; limit %.3f)\n", name, ratio, rccl, cccl, limit
    if (ratio > limit) exit 1
  }'
}

failed=0
require_equal_performance \
  'CUB DeviceMerge u32' \
  "$(median_metric 'CCCL CUB DeviceMerge u32:' median "$cccl_merge")" \
  "$(median_metric 'RCCL_CUB_ORDERING ' merge_gpu_median_ms "$rccl_cub")" || failed=1
require_equal_performance \
  'CUB DeviceMerge u32/u64 pairs' \
  "$(median_metric 'CCCL CUB DeviceMerge u32/u64 pairs:' median "$cccl_merge")" \
  "$(median_metric 'RCCL_CUB_ORDERING ' pair_merge_gpu_median_ms "$rccl_cub")" || failed=1
require_equal_performance \
  'CUB DeviceMergeSort::SortKeysCopy u32' \
  "$(median_metric 'CCCL_ORDERING ' cub_merge_sort_copy_gpu_median_ms "$cccl_ordering")" \
  "$(median_metric 'RCCL_CUB_ORDERING ' sort_copy_gpu_median_ms "$rccl_cub")" || failed=1
require_equal_performance \
  'CUB DeviceMergeSort::SortPairsCopy u32/u32' \
  "$(median_metric 'CCCL_ORDERING ' cub_merge_sort_pairs_copy_gpu_median_ms "$cccl_ordering")" \
  "$(median_metric 'RCCL_CUB_MERGE_SORT_PAIRS ' gpu_median_ms "$rccl_cub_pairs")" || failed=1
require_equal_performance \
  'CUB DeviceRadixSort::SortKeys u32' \
  "$(median_metric 'CCCL_ORDERING ' cub_radix_gpu_median_ms "$cccl_ordering")" \
  "$(median_metric 'RCCL_CUB_ORDERING ' radix_gpu_median_ms "$rccl_cub")" || failed=1
require_equal_performance \
  'CUB DeviceRadixSort::SortPairs u32/u32' \
  "$(median_metric 'CCCL_ORDERING ' cub_radix_pairs_gpu_median_ms "$cccl_ordering")" \
  "$(median_metric 'RCCL_CUB_ORDERING ' radix_pairs_gpu_median_ms "$rccl_cub")" || failed=1
require_equal_performance \
  'CUB DeviceRadixSort::SortKeys u32 DoubleBuffer 0..24' \
  "$(median_metric 'CCCL_ORDERING ' cub_radix_window_gpu_median_ms "$cccl_ordering")" \
  "$(median_metric 'RCCL_CUB_ORDERING ' radix_window_gpu_median_ms "$rccl_cub")" || failed=1
require_equal_performance \
  'CUB DeviceRadixSort::SortPairs u32/u32 DoubleBuffer 0..24' \
  "$(median_metric 'CCCL_ORDERING ' cub_radix_window_pairs_gpu_median_ms "$cccl_ordering")" \
  "$(median_metric 'RCCL_CUB_ORDERING ' radix_window_pairs_gpu_median_ms "$rccl_cub")" || failed=1
require_equal_performance \
  'CUB DeviceRadixSort::SortKeys u32 single tile' \
  "$(median_metric 'CCCL_ORDERING ' cub_radix_single_tile_gpu_median_ms "$cccl_ordering")" \
  "$(median_metric 'RCCL_CUB_ORDERING ' radix_single_tile_gpu_median_ms "$rccl_cub")" || failed=1
require_equal_performance \
  'CUB DeviceRadixSort::SortPairs u32/u32 single tile' \
  "$(median_metric 'CCCL_ORDERING ' cub_radix_single_tile_pairs_gpu_median_ms "$cccl_ordering")" \
  "$(median_metric 'RCCL_CUB_ORDERING ' radix_single_tile_pairs_gpu_median_ms "$rccl_cub")" || failed=1
require_equal_performance \
  'Thrust merge u32 through CUB' \
  "$(median_metric 'CCCL Thrust merge u32:' median "$cccl_merge")" \
  "$(median_metric 'RCCL_THRUST_ORDERING ' merge_gpu_median_ms "$rccl_thrust")" || failed=1
require_equal_performance \
  'Thrust merge_by_key u32/u64 pairs through CUB' \
  "$(median_metric 'CCCL Thrust merge_by_key u32/u64 pairs:' median "$cccl_merge")" \
  "$(median_metric 'RCCL_THRUST_ORDERING ' pair_merge_gpu_median_ms "$rccl_thrust")" || failed=1
require_equal_performance \
  'Thrust sort u32 through CUB smart sort' \
  "$(median_metric 'CCCL_ORDERING ' thrust_sort_gpu_median_ms "$cccl_ordering")" \
  "$(median_metric 'RCCL_THRUST_ORDERING ' sort_gpu_median_ms "$rccl_thrust")" || failed=1
require_equal_performance \
  'Thrust sort_by_key u32/u32 through CUB smart sort' \
  "$(median_metric 'CCCL_ORDERING ' thrust_sort_by_key_gpu_median_ms "$cccl_ordering")" \
  "$(median_metric 'RCCL_THRUST_ORDERING ' sort_by_key_gpu_median_ms "$rccl_thrust")" || failed=1

if ((failed != 0)); then
  printf '%s\n' 'Ordering performance parity: FAIL (implementation gap is not masked)' >&2
  exit 1
fi
printf '%s\n' 'Ordering performance parity: PASS'
