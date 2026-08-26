#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"

source scripts/oxide-env.sh

readonly cccl_commit=571f2fc3bc53cd710e306ad43d58995c1fe4219f
readonly cccl_origin=https://github.com/NVIDIA/cccl.git

if [[ -n "${CCCL_ROOT:-}" ]]; then
  cccl_root="$(cd "$CCCL_ROOT" && pwd)"
else
  cccl_root="$task_root/target/cccl-reference-$cccl_commit"
  if [[ ! -d "$cccl_root/.git" ]]; then
    mkdir -p "$task_root/target"
    checkout_staging="$(mktemp -d "$task_root/target/.cccl-reference.XXXXXX")"
    cleanup_checkout_staging() {
      rm -rf -- "$checkout_staging"
    }
    trap cleanup_checkout_staging EXIT
    git -C "$checkout_staging" init --quiet
    git -C "$checkout_staging" remote add origin "$cccl_origin"
    git -C "$checkout_staging" fetch --quiet --depth 1 origin "$cccl_commit"
    git -C "$checkout_staging" checkout --quiet --detach FETCH_HEAD
    mv "$checkout_staging" "$cccl_root"
    trap - EXIT
  fi
fi

actual_commit="$(git -C "$cccl_root" rev-parse HEAD)"
if [[ "$actual_commit" != "$cccl_commit" ]]; then
  printf '%s\n' \
    "CCCL_ROOT resolves to $actual_commit; reference tests require pinned commit $cccl_commit" >&2
  exit 1
fi

scripts/check-cccl-coverage.sh "$cccl_root"

reference_build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-cccl-reference.XXXXXX")"
cleanup_reference_build() {
  rm -rf -- "$reference_build_dir"
}
trap cleanup_reference_build EXIT

mapfile -t reference_sources < <(
  find "$task_root/benchmarks" -maxdepth 1 -type f -name '*-reference.cu' | sort
)
expected_reference_sources=(
  "$task_root/benchmarks/merge-reference.cu"
  "$task_root/benchmarks/warp-merge-sort-reference.cu"
)
if ! diff -u \
  <(printf '%s\n' "${expected_reference_sources[@]}") \
  <(printf '%s\n' "${reference_sources[@]}"); then
  printf '%s\n' 'a CUDA reference source has no explicit compile-and-run route' >&2
  exit 1
fi

for source in "${reference_sources[@]}"; do
  output="$reference_build_dir/$(basename "$source" .cu)"
  nvcc -std=c++20 -O3 -arch="${RCCL_ARCH:-sm_89}" \
    -I"$cccl_root/cub" \
    -I"$cccl_root/thrust" \
    -I"$cccl_root/libcudacxx/include" \
    "$source" -o "$output"
done

"$reference_build_dir/merge-reference" 65536 1
"$reference_build_dir/warp-merge-sort-reference" 262144 1
