#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh

cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
if [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' "histogram parity requires pinned CCCL $cccl_revision" >&2
  exit 1
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-histogram.XXXXXX")"
cleanup() {
  rm -rf -- "$build_dir"
  rm -f -- \
    device_histogram_probe.options \
    device_histogram_probe.target \
    device_histogram_benchmark.options \
    device_histogram_benchmark.target
}
trap cleanup EXIT

cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
  --release --package rccl-cub --example device_histogram_probe
./target/release/examples/device_histogram_probe

compute_sanitizer="$CUDA_TOOLKIT_PATH/compute-sanitizer/compute-sanitizer"
if [[ ! -x "$compute_sanitizer" ]]; then
  printf '%s\n' "CUDA compute-sanitizer is unavailable at $compute_sanitizer" >&2
  exit 1
fi
"$compute_sanitizer" --tool memcheck --error-exitcode 97 \
  ./target/release/examples/device_histogram_probe
"$compute_sanitizer" --tool racecheck --error-exitcode 98 \
  ./target/release/examples/device_histogram_probe

foreign_compute="$(
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$foreign_compute" ]]; then
  printf '%s\n' 'histogram performance parity requires an exclusive GPU; active compute processes:' >&2
  printf '%s\n' "$foreign_compute" >&2
  exit 1
fi

nvcc -std=c++20 -O3 -arch="${RCCL_ARCH:-sm_89}" \
  -I"$cccl_root/cub" \
  -I"$cccl_root/thrust" \
  -I"$cccl_root/libcudacxx/include" \
  benchmarks/histogram-reference.cu \
  -o "$build_dir/histogram-reference"
cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
  --release --package rccl-cub --example device_histogram_benchmark

items="${RCCL_HISTOGRAM_ITEMS:-16777216}"
bins="${RCCL_HISTOGRAM_BINS:-128}"
iterations="${RCCL_HISTOGRAM_ITERATIONS:-31}"
rounds="${RCCL_HISTOGRAM_ROUNDS:-5}"
cccl_results="$build_dir/cccl.txt"
rccl_results="$build_dir/rccl.txt"
for ((round = 0; round < rounds; ++round)); do
  "$build_dir/histogram-reference" "$items" "$bins" "$iterations" | tee -a "$cccl_results"
  ./target/release/examples/device_histogram_benchmark \
    --items "$items" --bins "$bins" --iterations "$iterations" | tee -a "$rccl_results"
done

median_metric() {
  local key="$1"
  local file="$2"
  awk -v key="$key" '
    index($0, "_CUB_HISTOGRAM ") != 0 {
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
  local metric="$2"
  local cccl_ms
  local rccl_ms
  cccl_ms="$(median_metric "$metric" "$cccl_results")"
  rccl_ms="$(median_metric "$metric" "$rccl_results")"
  awk -v name="$name" -v cccl="$cccl_ms" -v rccl="$rccl_ms" 'BEGIN {
    limit = 1.001
    ratio = rccl / cccl
    printf "%s median-of-medians ratio: %.6f (RCCL %.6f ms / CCCL %.6f ms; limit %.3f)\n", name, ratio, rccl, cccl, limit
    if (ratio > limit) exit 1
  }'
}

failed=0
require_equal_performance 'CUB HistogramEven i32' even_gpu_median_ms || failed=1
require_equal_performance 'CUB HistogramRange i32' range_gpu_median_ms || failed=1
require_equal_performance 'CUB MultiHistogramEven 4/3 i32' multi_even_gpu_median_ms || failed=1
require_equal_performance 'CUB MultiHistogramRange 4/3 i32' multi_range_gpu_median_ms || failed=1
if ((failed != 0)); then
  printf '%s\n' 'Histogram performance parity: FAIL (implementation gap is not masked)' >&2
  exit 1
fi
printf '%s\n' 'Histogram correctness, sanitizer, and performance parity: PASS'
