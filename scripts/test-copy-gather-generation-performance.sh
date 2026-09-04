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
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-copy-gather-generation-performance.XXXXXX")"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

"$nvcc" -std=c++20 -O3 --expt-extended-lambda -arch="${RCCL_ARCH:-sm_89}" \
  -I"$cccl_root/cub" -I"$cccl_root/thrust" -I"$cccl_root/libcudacxx/include" \
  benchmarks/thrust-copy-gather-generation-reference.cu \
  -o "$build_dir/thrust-copy-gather-generation-reference"
cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
  --release --package rccl-thrust --example copy_gather_generation_benchmark

cccl_results="$build_dir/cccl.txt"
rccl_results="$build_dir/rccl.txt"
for round in 1 2 3 4 5 6 7; do
  "$build_dir/thrust-copy-gather-generation-reference" \
    | tee -a "$cccl_results" \
    | awk -v round="$round" '{ print "round=" round, $0 }'
  ./target/release/examples/copy_gather_generation_benchmark \
    | tee -a "$rccl_results" \
    | awk -v round="$round" '{ print "round=" round, $0 }'
done

minimum_median() {
  local algorithm="$1" file="$2" implementation="$3"
  awk -v algorithm="$algorithm" -v implementation="$implementation" '
    {
      matched = 0
      metric = 0
      for (field = 1; field <= NF; ++field) {
        split($field, pair, "=")
        if (pair[1] == "implementation" && pair[2] == implementation) matched++
        if (pair[1] == "algorithm" && pair[2] == algorithm) matched++
        if (pair[1] == "type" && pair[2] == "u32") matched++
        if (pair[1] == "items" && pair[2] == 16777216) matched++
        if (pair[1] == "median_ms") metric = pair[2] + 0
      }
      if (matched == 4 && (!seen || metric < minimum)) {
        minimum = metric
        seen = 1
      }
    }
    END { if (!seen) exit 1; printf "%.9f", minimum }
  ' "$file"
}

blocked=0
for algorithm in copy gather fill sequence; do
  cccl="$(minimum_median "$algorithm" "$cccl_results" cccl)"
  rccl="$(minimum_median "$algorithm" "$rccl_results" rccl)"
  ratio="$(awk -v rccl="$rccl" -v cccl="$cccl" 'BEGIN { printf "%.9f", rccl / cccl }')"
  printf 'algorithm=%s cccl_median_ms=%s rccl_median_ms=%s ratio=%s\n' \
    "$algorithm" "$cccl" "$rccl" "$ratio"
  if ! awk -v ratio="$ratio" 'BEGIN { exit !(ratio <= 1.001) }'; then
    blocked=1
  fi
done

if [[ "$blocked" -ne 0 ]]; then
  printf '%s\n' \
    'BACKEND_BLOCKED: matched exclusive-GPU microkernels exceed the strict 1.001 gate; stop without an unsafe production workaround.' >&2
  exit 1
fi
printf '%s\n' 'Pinned equal-work copy/gather/generation strict performance gate: PASS'
