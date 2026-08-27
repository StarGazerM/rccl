#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh

cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
if [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' "ordering result parity requires pinned CCCL $cccl_revision" >&2
  exit 1
fi

cub_examples=(
  device_ordering_probe
  device_merge_sort_keys_probe
  device_merge_sort_pairs_probe
  device_radix_sort_keys_probe
  device_radix_sort_pairs_probe
)
thrust_examples=(
  merge
  comparison_sort
  sort
)

cleanup() {
  local example
  for example in "${cub_examples[@]}" "${thrust_examples[@]}"; do
    rm -f -- "$example.options" "$example.target"
  done
}
trap cleanup EXIT

for example in "${cub_examples[@]}"; do
  cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
    --release --package rccl-cub --example "$example"
  "./target/release/examples/$example"
done

for example in "${thrust_examples[@]}"; do
  cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
    --release --package rccl-thrust --example "$example"
  "./target/release/examples/$example"
done

compute_sanitizer="$CUDA_TOOLKIT_PATH/compute-sanitizer/compute-sanitizer"
if [[ ! -x "$compute_sanitizer" ]]; then
  printf '%s\n' "CUDA compute-sanitizer is unavailable at $compute_sanitizer" >&2
  exit 1
fi
for example in device_radix_sort_keys_probe device_radix_sort_pairs_probe; do
  "$compute_sanitizer" --tool memcheck --error-exitcode 99 \
    "./target/release/examples/$example"
  "$compute_sanitizer" --tool racecheck --error-exitcode 99 \
    "./target/release/examples/$example"
done

printf '%s\n' 'Pinned CUB and Thrust ordering result/sanitizer parity: PASS'
