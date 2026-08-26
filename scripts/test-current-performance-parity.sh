#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh

cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
if [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' "performance reference must use pinned CCCL $cccl_revision" >&2
  exit 1
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-current-parity.XXXXXX")"
cleanup() {
  rm -rf -- "$build_dir"
  rm -f -- \
    constant_iterator_benchmark.options \
    constant_iterator_benchmark.target \
    warp_reduce_benchmark.options \
    warp_reduce_benchmark.target
}
trap cleanup EXIT

for source in \
  benchmarks/thrust-transform-constant-reference.cu \
  benchmarks/warp-reduce-reference.cu; do
  output="$build_dir/$(basename "$source" .cu)"
  nvcc -std=c++20 -O3 -arch="${RCCL_ARCH:-sm_89}" \
    -I"$cccl_root/cub" \
    -I"$cccl_root/thrust" \
    -I"$cccl_root/libcudacxx/include" \
    "$source" -o "$output"
done

cargo oxide build --features cuda-oxide-example -- \
  --release --package rccl-cub --example warp_reduce_benchmark
cargo oxide build --features cuda-oxide-example -- \
  --release --package rccl-thrust --example constant_iterator_benchmark

minimum_metric() {
  local key="$1"
  awk -v key="$key" '
    {
      for (field = 1; field <= NF; ++field) {
        split($field, pair, "=")
        if (pair[1] == key && (!seen || pair[2] + 0 < minimum)) {
          minimum = pair[2] + 0
          seen = 1
        }
      }
    }
    END { if (!seen) exit 1; printf "%.9f", minimum }
  '
}

require_no_regression() {
  local name="$1"
  local cccl="$2"
  local rccl="$3"
  local relative_limit="$4"
  awk -v name="$name" -v cccl="$cccl" -v rccl="$rccl" -v relative_limit="$relative_limit" 'BEGIN {
    ratio = rccl / cccl
    printf "%s best-median ratio: %.6f (RCCL %.6f ms / CCCL %.6f ms; limit %.3f)\n", name, ratio, rccl, cccl, relative_limit
    if (ratio > relative_limit) exit 1
  }'
}

# CUDA event samples fluctuate with clock and thermal state even when the
# device instruction sequence is unchanged. One tenth of one percent is the
# measurement equivalence band, not permission to use a slower lowering: the seeded
# shuffle-tree fault is substantially outside it and is recorded in the two
# upstream correspondence reviews.

warp_cccl="$build_dir/warp-cccl.txt"
warp_rccl="$build_dir/warp-rccl.txt"
transform_cccl="$build_dir/transform-cccl.txt"
transform_rccl="$build_dir/transform-rccl.txt"
for _ in 1 2 3 4 5 6 7; do
  "$build_dir/warp-reduce-reference" | tee -a "$warp_cccl"
  ./target/release/examples/warp_reduce_benchmark | tee -a "$warp_rccl"
  "$build_dir/thrust-transform-constant-reference" | tee -a "$transform_cccl"
  ./target/release/examples/constant_iterator_benchmark | tee -a "$transform_rccl"
done

require_no_regression \
  'CUB WarpReduce i32 sum' \
  "$(minimum_metric median_ms < "$warp_cccl")" \
  "$(minimum_metric median_ms < "$warp_rccl")" \
  '1.001'
require_no_regression \
  'Thrust transform plus constant_iterator' \
  "$(minimum_metric median_ms < "$transform_cccl")" \
  "$(minimum_metric median_ms < "$transform_rccl")" \
  '1.001'

printf '%s\n' 'Current implemented CCCL result and no-performance-regression gate: PASS'
