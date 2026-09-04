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
foreign_compute="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
if [[ -n "$foreign_compute" ]]; then
  printf '%s\n%s\n' 'PERFORMANCE_UNVERIFIED: exclusive GPU unavailable; no cuda-oxide ownership claim.' "$foreign_compute" >&2
  exit 77
fi

nvcc="$CUDA_HOME/bin/nvcc"
[[ -x "$nvcc" ]] || { printf '%s\n' "required CUDA compiler unavailable: $nvcc" >&2; exit 1; }
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-compaction-performance.XXXXXX")"
cleanup() { rm -rf -- "$build_dir"; rm -f -- compaction_benchmark.options compaction_benchmark.target; }
trap cleanup EXIT
arch="${RCCL_ARCH:-sm_89}"
"$nvcc" -std=c++17 -O3 -arch="$arch" \
  -I "$cccl_root/cub" -I "$cccl_root/thrust" -I "$cccl_root/libcudacxx/include" \
  benchmarks/cub-compaction-reference.cu -o "$build_dir/cub-compaction-reference"
cargo oxide build --arch "$arch" --features cuda-oxide-example -- \
  --release --package rccl-thrust --example compaction_benchmark
cccl_results="$build_dir/cccl.txt"
rccl_results="$build_dir/rccl.txt"
for _ in 1 2 3 4 5 6 7; do
  "$build_dir/cub-compaction-reference" >>"$cccl_results"
  ./target/release/examples/compaction_benchmark >>"$rccl_results"
done
printf '%s\n' 'raw_cccl_rounds_begin'
cat "$cccl_results"
printf '%s\n' 'raw_cccl_rounds_end'
printf '%s\n' 'raw_rccl_rounds_begin'
cat "$rccl_results"
printf '%s\n' 'raw_rccl_rounds_end'
allocation_bytes() {
  local file="$1" implementation="$2"
  awk -v implementation="$implementation" '
    { matched=0; bytes=0; for (field=1; field<=NF; ++field) { split($field,pair,"=");
      if (pair[1]=="implementation" && pair[2]==implementation) matched=1;
      if (pair[1]=="temp_bytes") bytes=pair[2]+0; }
      if (matched) { if (seen && bytes != expected) exit 1; expected=bytes; seen=1; } }
    END { if (!seen) exit 1; printf "%.0f", expected }
  ' "$file"
}
cccl_allocation="$(allocation_bytes "$cccl_results" cccl)"
rccl_allocation="$(allocation_bytes "$rccl_results" rccl)"
if [[ "$cccl_allocation" != "$rccl_allocation" ]]; then
  printf 'allocation mismatch: cccl=%s rccl=%s\n' "$cccl_allocation" "$rccl_allocation" >&2
  exit 1
fi
minimum_median() {
  local algorithm="$1" file="$2" implementation="$3"
  awk -v algorithm="$algorithm" -v implementation="$implementation" '
    { matched=0; metric=0; for (field=1; field<=NF; ++field) { split($field,pair,"=");
      if (pair[1]=="implementation" && pair[2]==implementation) matched++;
      if (pair[1]=="algorithm" && pair[2]==algorithm) matched++;
      if (pair[1]=="type" && pair[2]=="u32") matched++;
      if (pair[1]=="items" && pair[2]==1048589) matched++;
      if (pair[1]=="threads" && pair[2]==256) matched++;
      if (pair[1]=="launches" && pair[2]==10) matched++;
      if (pair[1]=="median_ms") metric=pair[2]+0; }
      if (matched==6 && (!seen || metric<minimum)) { minimum=metric; seen=1; } }
    END { if (!seen) exit 1; printf "%.9f", minimum }
  ' "$file"
}
blocked=0
for algorithm in select_if select_flagged unique_copy stable_partition; do
  cccl="$(minimum_median "$algorithm" "$cccl_results" cccl)"
  rccl="$(minimum_median "$algorithm" "$rccl_results" rccl)"
  ratio="$(awk -v rccl="$rccl" -v cccl="$cccl" 'BEGIN { printf "%.9f", rccl/cccl }')"
  printf 'algorithm=%s cccl_median_ms=%s rccl_median_ms=%s ratio=%s\n' "$algorithm" "$cccl" "$rccl" "$ratio"
  if ! awk -v ratio="$ratio" 'BEGIN { exit !(ratio <= 1.001) }'; then blocked=1; fi
done
if [[ "$blocked" -ne 0 ]]; then
  printf '%s\n' 'PERFORMANCE_GAP: identical inputs, marks, stable scatter outputs, allocation exclusion, synchronization, warmups, samples, and launch counts exceed 1.001; optimize the production scan/scatter path rather than relaxing semantics or benchmark policy.' >&2
  exit 1
fi
printf '%s\n' 'Pinned equal-work compaction strict performance gate: PASS'
