#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh

readonly cuco_revision=1d538b6befc286ab450d9e27d7ffbc4f1415c641
readonly cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
readonly cuco_origin=https://github.com/NVIDIA/cuCollections.git
readonly cccl_origin=https://github.com/NVIDIA/cccl.git
cuco_root="${CUCO_ROOT:-$task_root/target/cuco-reference-$cuco_revision}"
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
if [[ ! -d "$cuco_root/.git" ]]; then
  mkdir -p "$task_root/target"
  staging="$(mktemp -d "$task_root/target/.cuco-reference.XXXXXX")"
  cleanup_staging() { rm -rf -- "$staging"; }
  trap cleanup_staging EXIT
  git -C "$staging" init --quiet
  git -C "$staging" remote add origin "$cuco_origin"
  git -C "$staging" fetch --quiet --depth 1 origin "$cuco_revision"
  git -C "$staging" checkout --quiet --detach FETCH_HEAD
  mv "$staging" "$cuco_root"
  trap - EXIT
fi
if [[ ! -d "$cccl_root/.git" ]]; then
  mkdir -p "$task_root/target"
  staging="$(mktemp -d "$task_root/target/.cccl-reference.XXXXXX")"
  cleanup_staging() { rm -rf -- "$staging"; }
  trap cleanup_staging EXIT
  git -C "$staging" init --quiet
  git -C "$staging" remote add origin "$cccl_origin"
  git -C "$staging" fetch --quiet --depth 1 origin "$cccl_revision"
  git -C "$staging" checkout --quiet --detach FETCH_HEAD
  mv "$staging" "$cccl_root"
  trap - EXIT
fi
if [[ "$(git -C "$cuco_root" rev-parse HEAD)" != "$cuco_revision" ]] \
  || [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' 'performance comparison requires the pinned cuCollections and CCCL checkouts' >&2
  exit 1
fi
nvcc="$CUDA_HOME/bin/nvcc"
if [[ ! -x "$nvcc" ]]; then
  printf '%s\n' "CUDA nvcc is unavailable at $nvcc" >&2
  exit 1
fi
foreign_compute="$(
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$foreign_compute" ]]; then
  printf '%s\n' 'u32/u64 static-map performance gate requires an exclusive GPU; active compute processes:' >&2
  printf '%s\n' "$foreign_compute" >&2
  exit 1
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-cuco-u64-performance.XXXXXX")"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

"$nvcc" -std=c++20 -O3 --expt-extended-lambda -arch="${RCCL_ARCH:-sm_89}" \
  -I"$cuco_root/include" \
  -I"$cccl_root/cub" \
  -I"$cccl_root/thrust" \
  -I"$cccl_root/libcudacxx/include" \
  benchmarks/cuco-static-map-u64-benchmark.cu \
  -o "$build_dir/cuco-static-map-u64-benchmark"
rust_benchmark="${RCCL_CUCO_U64_BENCHMARK:-$task_root/target/release/examples/static_map_u64_benchmark}"
if [[ ! -x "$rust_benchmark" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    printf '%s\n' \
      "Rust benchmark binary is unavailable at $rust_benchmark and cargo is not on PATH" >&2
    exit 1
  fi
  cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
    --release --package rccl-cuco --example static_map_u64_benchmark
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
  local file="$1"
  sort -n "$file" | awk '{ values[NR] = $1 } END {
    if (NR == 0) exit 1
    print values[int(NR / 2) + 1]
  }'
}

max_ratio="${RCCL_CUCO_U64_MAX_RATIO:-1.050}"

for occupancy in 0.50 0.75 0.90; do
  reference_samples="$build_dir/reference-$occupancy.txt"
  rust_samples="$build_dir/rust-$occupancy.txt"
  reference_capacities="$build_dir/reference-capacity-$occupancy.txt"
  rust_capacities="$build_dir/rust-capacity-$occupancy.txt"
  for round in 0 1 2 3 4; do
    if (( round % 2 == 0 )); then
      order=(reference rust)
    else
      order=(rust reference)
    fi
    for implementation in "${order[@]}"; do
      if [[ "$implementation" == reference ]]; then
        output="$("$build_dir/cuco-static-map-u64-benchmark" 8000000 "$occupancy" 10)"
        printf '%s\n' "$output" >&2
        metric median_ms <<<"$output" >> "$reference_samples"
        metric capacity <<<"$output" >> "$reference_capacities"
      else
        output="$("$rust_benchmark" --items 8000000 --occupancy "$occupancy" --iterations 10)"
        printf '%s\n' "$output" >&2
        metric median_ms <<<"$output" >> "$rust_samples"
        metric capacity <<<"$output" >> "$rust_capacities"
      fi
    done
  done
  reference_capacity="$(sort -nu "$reference_capacities")"
  rust_capacity="$(sort -nu "$rust_capacities")"
  if [[ "$reference_capacity" == *$'\n'* || "$rust_capacity" == *$'\n'* \
    || "$reference_capacity" != "$rust_capacity" ]]; then
    printf '%s\n' \
      "capacity mismatch at occupancy $occupancy: cuco={$reference_capacity} RCCL={$rust_capacity}" >&2
    exit 1
  fi
  reference="$(median_file "$reference_samples")"
  rust="$(median_file "$rust_samples")"
  awk -v occupancy="$occupancy" -v reference="$reference" -v rust="$rust" \
    -v limit="$max_ratio" 'BEGIN {
    ratio = rust / reference
    printf "static_map u32_u64 insert occupancy=%s median-of-medians ratio=%.6f (RCCL %.9f ms / cuco %.9f ms; limit %.3f)\n", occupancy, ratio, rust, reference, limit
    if (ratio > limit) exit 1
  }'
done

printf '%s\n' 'Pinned cuCollections static_map<u32,u64> insertion performance gate: PASS'
