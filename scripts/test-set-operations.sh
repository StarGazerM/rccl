#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh

cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
if [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' "set-operation parity requires pinned CCCL $cccl_revision" >&2
  exit 1
fi

if rg -n 'require_output|maximum_output|OutputTooSmall' \
  rccl-thrust/src/set.rs rccl-cub/src/ordering_backend/set.rs rccl-cub/src/error.rs; then
  printf '%s\n' 'set calls may not invent a worst-case output-capacity restriction absent from CCCL' >&2
  exit 1
fi
if ! rg -U -q '#\[repr\(C\)\]\n#\[derive\(Clone, Copy, Debug, Default, PartialEq, Eq, DeviceCopy\)\]\npub struct KeyValue' \
  rccl-cub/src/range.rs; then
  printf '%s\n' 'KeyValue must expose macro-visible C layout without a dishonest key-only Rust Ord' >&2
  exit 1
fi
if rg -n '\bunsafe\b|\*mut|\*const|\bArc\b' \
  rccl-cub/src/ordering_backend/set.rs rccl-cub-macros/src/set.rs rccl-thrust/src/set.rs; then
  printf '%s\n' 'set algorithms and generated lowering must retain the safe Rust boundary' >&2
  exit 1
fi
if rg -n '#\[allow\(' \
  rccl-cub/src/ordering_backend/set.rs rccl-cub-macros/src/kernels.rs rccl-thrust/src/set.rs \
  || ! rg -q 'reason = "the final cursor loads are part of CCCL' \
    rccl-cub/src/ordering_backend/set.rs \
  || ! rg -q 'reason = "CUDA entry points preserve the pinned CUB launch ABI' \
    rccl-cub-macros/src/kernels.rs; then
  printf '%s\n' 'set lint exceptions must be checked CCCL algorithm or CUDA ABI expectations, never broad allows' >&2
  exit 1
fi
if rg -n 'F: Fn\([^;]*\+ Copy[,;]' rccl-thrust/src/set.rs \
  || ! rg -q "F: Fn\(&K, &K\) -> Ordering \+ Copy \+ 'static" rccl-thrust/src/set.rs \
  || ! rg -q 'let order = move |' rccl-thrust/examples/set_operations_tests.rs; then
  printf '%s\n' 'set comparators must own captures; borrowed host closures may not cross the kernel ABI' >&2
  exit 1
fi
for example in \
  rccl-thrust/examples/set_operations.rs \
  rccl-thrust/examples/set_operations_tests.rs \
  rccl-thrust/examples/set_operations_benchmark.rs; do
  if ! rg -q 'thrust_program!\(&context; .*set_operations' "$example"; then
    printf '%s\n' 'set programs must request only their compile-time backend capability bundle' >&2
    exit 1
  fi
done

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-set-operations.XXXXXX")"
examples=(set_operations set_operations_tests set_operations_benchmark)
cleanup() {
  rm -rf -- "$build_dir"
  local example
  for example in "${examples[@]}"; do
    rm -f -- "$example.options" "$example.target"
  done
}
trap cleanup EXIT

nvcc -std=c++20 -O3 -arch="${RCCL_ARCH:-sm_89}" \
  -I"$cccl_root/cub" \
  -I"$cccl_root/thrust" \
  -I"$cccl_root/libcudacxx/include" \
  benchmarks/set-operations-reference.cu \
  -o "$build_dir/set-operations-reference"

for example in "${examples[@]}"; do
  cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
    --release --package rccl-thrust --example "$example"
done

./target/release/examples/set_operations
./target/release/examples/set_operations_tests

compute_sanitizer="$CUDA_TOOLKIT_PATH/compute-sanitizer/compute-sanitizer"
if [[ ! -x "$compute_sanitizer" ]]; then
  printf '%s\n' "CUDA compute-sanitizer is unavailable at $compute_sanitizer" >&2
  exit 1
fi
"$compute_sanitizer" --tool memcheck --error-exitcode 99 \
  ./target/release/examples/set_operations_tests --skip-wide-index
"$compute_sanitizer" --tool racecheck --error-exitcode 99 \
  ./target/release/examples/set_operations_tests --skip-wide-index
"$compute_sanitizer" --tool memcheck --error-exitcode 99 \
  "$build_dir/set-operations-reference" 1000003 1 keys i32 none 1.000 50
"$compute_sanitizer" --tool memcheck --error-exitcode 99 \
  "$build_dir/set-operations-reference" 1000003 1 pairs i32 i64 1.000 50

# Strict timing cannot be inferred from a shared accelerator. Refuse to turn
# another workload's scheduling noise into either a parity pass or a kernel
# regression. Correctness and sanitizer checks above still run first.
foreign_compute="$(
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "$foreign_compute" ]]; then
  printf '%s\n' 'set-operation performance parity requires an exclusive GPU; active compute processes:' >&2
  printf '%s\n' "$foreign_compute" >&2
  exit 1
fi

iterations="${RCCL_SET_ITERATIONS:-21}"
rounds="${RCCL_SET_ROUNDS:-5}"
cccl_samples="$build_dir/cccl.txt"
rccl_samples="$build_dir/rccl.txt"

if [[ "${RCCL_SET_FULL_MATRIX:-0}" == 1 ]]; then
  items_axis=(65536 1048576 16777216 268435456)
  entropy_axis=(1.000 0.201)
  ratio_axis=(25 50 75)
  rows=(
    keys:i8:none keys:i16:none keys:i32:none keys:i64:none
    pairs:i8:i8 pairs:i8:i64 pairs:i16:i8 pairs:i16:i64
    pairs:i32:i8 pairs:i32:i64 pairs:i64:i8 pairs:i64:i64
  )
else
  items_axis=("${RCCL_SET_ITEMS:-16777216}")
  entropy_axis=("${RCCL_SET_ENTROPY:-1.000}")
  ratio_axis=("${RCCL_SET_RATIO:-50}")
  rows=(keys:i32:none pairs:i32:i64)
fi

run_reference() {
  local items="$1" mode="$2" key="$3" value="$4" entropy="$5" ratio="$6"
  "$build_dir/set-operations-reference" \
    "$items" "$iterations" "$mode" "$key" "$value" "$entropy" "$ratio" \
    | tee -a "$cccl_samples"
}

run_rccl() {
  local items="$1" mode="$2" key="$3" value="$4" entropy="$5" ratio="$6"
  ./target/release/examples/set_operations_benchmark \
    --items "$items" --iterations "$iterations" --mode "$mode" --key "$key" \
    --value "$value" --entropy "$entropy" --ratio "$ratio" \
    | tee -a "$rccl_samples"
}

for items in "${items_axis[@]}"; do
  for entropy in "${entropy_axis[@]}"; do
    for ratio in "${ratio_axis[@]}"; do
      for row in "${rows[@]}"; do
        IFS=: read -r mode key value <<<"$row"
        for ((round = 0; round < rounds; ++round)); do
          if ((round % 2 == 0)); then
            run_reference "$items" "$mode" "$key" "$value" "$entropy" "$ratio"
            run_rccl "$items" "$mode" "$key" "$value" "$entropy" "$ratio"
          else
            run_rccl "$items" "$mode" "$key" "$value" "$entropy" "$ratio"
            run_reference "$items" "$mode" "$key" "$value" "$entropy" "$ratio"
          fi
        done
      done
    done
  done
done

median_metric() {
  local prefix="$1"
  local key="$2"
  local file="$3"
  awk -v prefix="$prefix" -v key="$key" '
    index($0, prefix) == 1 {
      for (field = 1; field <= NF; ++field) {
        split($field, pair, "=")
        if (pair[1] == key) print pair[2] + 0
      }
    }
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
  local prefix="$2"
  local metric="$3"
  local cccl_ms rccl_ms
  cccl_ms="$(median_metric "CCCL_THRUST_SET $prefix " "$metric" "$cccl_samples")"
  rccl_ms="$(median_metric "RCCL_THRUST_SET $prefix " "$metric" "$rccl_samples")"
  awk -v name="$name" -v cccl="$cccl_ms" -v rccl="$rccl_ms" 'BEGIN {
    limit = 1.001
    ratio = rccl / cccl
    printf "%s median-of-medians ratio: %.6f (RCCL %.6f ms / CCCL %.6f ms; limit %.3f)\n", name, ratio, rccl, cccl, limit
    if (ratio > limit) exit 1
  }'
}

failed=0
for items in "${items_axis[@]}"; do
  for entropy in "${entropy_axis[@]}"; do
    for ratio in "${ratio_axis[@]}"; do
      for row in "${rows[@]}"; do
        IFS=: read -r mode key value <<<"$row"
        prefix="mode=$mode key=$key value=$value items=$items entropy=$entropy ratio=$ratio"
        for operation in union intersection difference symmetric; do
          label="$operation $mode $key/$value items=$items entropy=$entropy ratio=$ratio"
          require_equal_performance "$label" "$prefix" "${operation}_gpu_median_ms" || failed=1
        done
      done
    done
  done
done

if ((failed != 0)); then
  printf '%s\n' 'Set-operation performance parity: FAIL (implementation gap is not masked)' >&2
  exit 1
fi
printf '%s\n' 'Pinned Thrust set-operation result, sanitizer, and performance parity: PASS'
