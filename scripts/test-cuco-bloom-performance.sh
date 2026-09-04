#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh
readonly cuco_revision=1d538b6befc286ab450d9e27d7ffbc4f1415c641
readonly cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cuco_root="${CUCO_ROOT:-$task_root/target/cuco-reference-$cuco_revision}"
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
for pinned in "$cuco_root:$cuco_revision" "$cccl_root:$cccl_revision"; do
  root="${pinned%%:*}"; revision="${pinned##*:}"
  if [[ ! -d "$root/.git" || "$(git -C "$root" rev-parse HEAD)" != "$revision" ]]; then
    printf '%s\n' "run scripts/test-cuco-bloom.sh first to install pinned reference $revision" >&2
    exit 1
  fi
done

foreign_compute="$(
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$foreign_compute" ]]; then
  printf '%s\n' 'exclusive GPU required; active compute processes:' "$foreign_compute" >&2
  exit 1
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-cuco-bloom-performance.XXXXXX")"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT
"$CUDA_HOME/bin/nvcc" -std=c++20 -O3 --expt-extended-lambda -arch="${RCCL_ARCH:-sm_89}" \
  -I"$cuco_root/include" -I"$cccl_root/cub" -I"$cccl_root/thrust" \
  -I"$cccl_root/libcudacxx/include" benchmarks/cuco-bloom-reference.cu \
  -o "$build_dir/cuco-bloom-reference"
rust_benchmark="${RCCL_CUCO_BLOOM_BENCHMARK:-$task_root/target/release/examples/bloom_benchmark}"
if [[ ! -x "$rust_benchmark" ]]; then
  cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
    --release --package rccl-cuco --example bloom_benchmark
fi

metric() {
  local key="$1"
  awk -v key="$key" '{
    for (field = 1; field <= NF; ++field) {
      split($field, pair, "=")
      if (pair[1] == key) print pair[2] + 0
    }
  }'
}
median_file() {
  sort -n "$1" | awk '{ values[NR] = $1 } END {
    if (NR == 0) exit 1
    print values[int(NR / 2) + 1]
  }'
}

reference_add="$build_dir/reference-add.txt"
rust_add="$build_dir/rust-add.txt"
reference_contains="$build_dir/reference-contains.txt"
rust_contains="$build_dir/rust-contains.txt"
for round in 0 1 2 3 4; do
  if (( round % 2 == 0 )); then order=(reference rust); else order=(rust reference); fi
  for implementation in "${order[@]}"; do
    if [[ "$implementation" == reference ]]; then
      output="$("$build_dir/cuco-bloom-reference" --items 8000000 --blocks 10000000 --iterations 10)"
      metric add_median_ms <<<"$output" >> "$reference_add"
      metric contains_median_ms <<<"$output" >> "$reference_contains"
    else
      output="$("$rust_benchmark" --items 8000000 --blocks 10000000 --iterations 10)"
      metric add_median_ms <<<"$output" >> "$rust_add"
      metric contains_median_ms <<<"$output" >> "$rust_contains"
    fi
    printf '%s\n' "$output" >&2
  done
done

for algorithm in add contains; do
  reference="$(median_file "$build_dir/reference-$algorithm.txt")"
  rust="$(median_file "$build_dir/rust-$algorithm.txt")"
  awk -v algorithm="$algorithm" -v reference="$reference" -v rust="$rust" 'BEGIN {
    ratio = rust / reference
    printf "Bloom %s median-of-medians ratio=%.6f (RCCL %.9f ms / cuco %.9f ms; limit 1.001)\n", algorithm, ratio, rust, reference
    if (ratio > 1.001) exit 1
  }'
done
printf '%s\n' 'Pinned HISA/cuCollections Bloom equal-work performance parity: PASS'
