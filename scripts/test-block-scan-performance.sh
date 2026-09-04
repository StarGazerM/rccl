#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh

readonly cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
if [[ ! -d "$cccl_root/.git" ]] || [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' "performance reference requires pinned CCCL $cccl_revision" >&2
  exit 1
fi

foreign_compute="$(
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$foreign_compute" ]]; then
  printf '%s\n%s\n' \
    'PERFORMANCE_UNVERIFIED: exclusive GPU unavailable; no cuda-oxide ownership claim.' \
    "$foreign_compute" >&2
  exit 77
fi

nvcc="$CUDA_HOME/bin/nvcc"
[[ -x "$nvcc" ]] || { printf '%s\n' "required CUDA compiler unavailable: $nvcc" >&2; exit 1; }
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-block-scan-performance.XXXXXX")"
cleanup() {
  rm -rf -- "$build_dir"
  rm -f -- block_scan_benchmark.options block_scan_benchmark.target
}
trap cleanup EXIT
arch="${RCCL_ARCH:-sm_89}"

"$nvcc" -std=c++17 -O3 -arch="$arch" \
  -I "$cccl_root/cub" -I "$cccl_root/thrust" -I "$cccl_root/libcudacxx/include" \
  benchmarks/cub-block-scan-reference.cu -o "$build_dir/cub-block-scan-reference"
cargo oxide build --arch "$arch" --features cuda-oxide-example -- \
  --release --package rccl-cub --example block_scan_benchmark

cccl_results="$build_dir/cccl.txt"
rccl_results="$build_dir/rccl.txt"
for _ in 1 2 3 4 5 6 7; do
  "$build_dir/cub-block-scan-reference" >>"$cccl_results"
  ./target/release/examples/block_scan_benchmark >>"$rccl_results"
done

minimum_median() {
  local type="$1" items="$2" file="$3" implementation="$4"
  awk -v type="$type" -v items="$items" -v implementation="$implementation" '
    {
      matched = 0
      metric = 0
      for (field = 1; field <= NF; ++field) {
        split($field, pair, "=")
        if (pair[1] == "implementation" && pair[2] == implementation) matched++
        if (pair[1] == "algorithm" && pair[2] == "block_exclusive_sum") matched++
        if (pair[1] == "type" && pair[2] == type) matched++
        if (pair[1] == "threads" && pair[2] == 256) matched++
        if (pair[1] == "items" && pair[2] == items) matched++
        if (pair[1] == "blocks" && pair[2] == 32768) matched++
        if (pair[1] == "launches" && pair[2] == 100) matched++
        if (pair[1] == "median_ms") metric = pair[2] + 0
      }
      if (matched == 7 && (!seen || metric < minimum)) {
        minimum = metric
        seen = 1
      }
    }
    END { if (!seen) exit 1; printf "%.9f", minimum }
  ' "$file"
}

blocked=0
for type in u32 u64; do
  for items in 8388608 8388545; do
    cccl="$(minimum_median "$type" "$items" "$cccl_results" cccl)"
    rccl="$(minimum_median "$type" "$items" "$rccl_results" rccl)"
    ratio="$(awk -v rccl="$rccl" -v cccl="$cccl" 'BEGIN { printf "%.9f", rccl / cccl }')"
    printf 'algorithm=block_exclusive_sum type=%s items=%s cccl_median_ms=%s rccl_median_ms=%s ratio=%s\n' \
      "$type" "$items" "$cccl" "$rccl" "$ratio"
    if ! awk -v ratio="$ratio" 'BEGIN { exit !(ratio <= 1.001) }'; then
      blocked=1
    fi
  done
done

if [[ "$blocked" -ne 0 ]]; then
  printf '%s\n' \
    'BACKEND_BLOCKED: identical 256-thread BLOCK_SCAN_WARP_SCANS work, types, full/partial grids, resets, warmups, samples, and launch counts exceed the strict 1.001 gate; stop without changing production semantics or benchmark policy.' >&2
  exit 1
fi
printf '%s\n' 'Pinned equal-work CUB BlockScan strict performance gate: PASS'
