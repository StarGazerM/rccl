#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh
readonly cuco_revision=1d538b6befc286ab450d9e27d7ffbc4f1415c641
readonly cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cuco_root="${CUCO_ROOT:-$task_root/target/cuco-reference-$cuco_revision}"
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
[[ -d "$cuco_root/.git" && "$(git -C "$cuco_root" rev-parse HEAD)" == "$cuco_revision" ]]
[[ -d "$cccl_root/.git" && "$(git -C "$cccl_root" rev-parse HEAD)" == "$cccl_revision" ]]

foreign_compute="$(
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$foreign_compute" ]]; then
  printf '%s\n' 'exact row-set performance is unverified: an exclusive GPU is required; active compute processes:' >&2
  printf '%s\n' "$foreign_compute" >&2
  exit 1
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-cuco-exact-set-performance.XXXXXX")"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT
nvcc="$CUDA_HOME/bin/nvcc"
"$nvcc" -std=c++20 -O3 --expt-extended-lambda -arch="${RCCL_ARCH:-sm_89}" \
  -I"$cuco_root/include" -I"$cccl_root/cub" -I"$cccl_root/thrust" \
  -I"$cccl_root/libcudacxx/include" benchmarks/cuco-exact-row-set-reference.cu \
  -o "$build_dir/reference"

rust_benchmark="${RCCL_CUCO_EXACT_SET_BENCHMARK:-$task_root/target/release/examples/exact_row_set_benchmark}"
if [[ ! -x "$rust_benchmark" ]]; then
  cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
    --release --package rccl-cuco --example exact_row_set_benchmark
fi

metric() {
  local key="$1"
  awk -v key="$key" '{ for (field=1; field<=NF; ++field) { split($field,pair,"="); if (pair[1]==key) print pair[2]+0 } }'
}
median_file() {
  sort -n "$1" | awk '{ values[NR]=$1 } END { if (NR==0) exit 1; print values[int(NR/2)+1] }'
}

for operation in insert contains; do
  for occupancy in 0.50 0.75 0.90; do
    reference_samples="$build_dir/reference-$operation-$occupancy.txt"
    rust_samples="$build_dir/rust-$operation-$occupancy.txt"
    reference_capacities="$build_dir/reference-capacity-$operation-$occupancy.txt"
    rust_capacities="$build_dir/rust-capacity-$operation-$occupancy.txt"
    for round in 0 1 2 3 4; do
      if (( round % 2 == 0 )); then order=(reference rust); else order=(rust reference); fi
      for implementation in "${order[@]}"; do
        if [[ "$implementation" == reference ]]; then
          output="$("$build_dir/reference" 8000000 "$occupancy" 10 "$operation")"
          metric median_ms <<<"$output" >> "$reference_samples"
          metric capacity <<<"$output" >> "$reference_capacities"
        else
          output="$("$rust_benchmark" --items 8000000 --occupancy "$occupancy" --iterations 10 --operation "$operation")"
          metric median_ms <<<"$output" >> "$rust_samples"
          metric capacity <<<"$output" >> "$rust_capacities"
        fi
        printf '%s\n' "$output" >&2
      done
    done
    reference_capacity="$(sort -nu "$reference_capacities")"
    rust_capacity="$(sort -nu "$rust_capacities")"
    [[ "$reference_capacity" != *$'\n'* && "$rust_capacity" != *$'\n'* \
      && "$reference_capacity" == "$rust_capacity" ]] || {
      printf '%s\n' "capacity mismatch for $operation at $occupancy: cuco={$reference_capacity} RCCL={$rust_capacity}" >&2
      exit 1
    }
    reference="$(median_file "$reference_samples")"
    rust="$(median_file "$rust_samples")"
    awk -v operation="$operation" -v occupancy="$occupancy" -v reference="$reference" -v rust="$rust" 'BEGIN {
      ratio=rust/reference
      printf "exact_row_set %s occupancy=%s median-of-medians ratio=%.6f (RCCL %.9f ms / cuco %.9f ms; limit 1.001)\n", operation, occupancy, ratio, rust, reference
      if (ratio > 1.001) exit 1
    }'
  done
done
printf '%s\n' 'Pinned cuCollections exact HISA row-set strict equal-work performance parity: PASS'
