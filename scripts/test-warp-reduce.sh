#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"

cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
header="$cccl_root/cub/cub/warp/warp_reduce.cuh"
test_source="$cccl_root/cub/test/warp/catch2_test_warp_reduce.cu"

if [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' "CCCL checkout is not pinned to $cccl_revision" >&2
  exit 1
fi
printf '%s  %s\n' \
  92e55aa35a74c0faca2de3ba609b27167ead04062533c20976dc20fd16734a95 \
  "$header" \
  | sha256sum --check --status
printf '%s  %s\n' \
  25bec1964bd85f8e9b1000af379b636218aebd713c42719cdf23e03b502c5295 \
  "$test_source" \
  | sha256sum --check --status

overview_example=rccl-cub/examples/warp_reduce.rs
matrix_example=rccl-cub/examples/warp_reduce_catch2.rs
unit_test=rccl-cub/tests/warp_reduce_upstream.rs
for required in \
  'rccl_cub::warp_reduce!(32' \
  'rccl_cub::warp_reduce!(16' \
  'rccl_cub::warp_reduce_items!(32' \
  '496, 1520, 2544, 3568'; do
  if ! rg -q -F "$required" "$overview_example"; then
    printf '%s\n' "WarpReduce port lost required evidence: $required" >&2
    exit 1
  fi
done
for required in \
  'rccl_cub::warp_reduce!($width' \
  'rccl_cub::warp_reduce_items!($width' \
  'full_integer_operation!(operation, 1' \
  'items_integer_operation!(operation, 16' \
  'items_integer_operation!(operation, 1' \
  'assert_eq!(executed, 150)' \
  'u8_matrix' \
  'u16_matrix' \
  'i32_matrix' \
  'i64_matrix' \
  'f32_min_max'; do
  if ! rg -q -F "$required" "$matrix_example"; then
    printf '%s\n' "WarpReduce Catch2 matrix lost required evidence: $required" >&2
    exit 1
  fi
done
for required in \
  'pinned_overview_reference_values_are_exact' \
  'partial_reference_excludes_inactive_lanes' \
  'multiple_item_reference_keeps_thread_items_in_order' \
  'supported_catch2_matrix_executes_every_translated_row' \
  'warp_reduce_accepts_a_literal_forwarded_by_a_rust_macro'; do
  if ! rg -q -F "$required" "$unit_test" rccl-cub-macros/src/call.rs; then
    printf '%s\n' "WarpReduce unit-test port lost required evidence: $required" >&2
    exit 1
  fi
done

if rg -n '\b(Arc|Comparator|Less|Greater|TypeId|Any)\b|\*mut|\*const' \
  rccl-cub/src \
  rccl-cub-macros/src/lib.rs \
  rccl-cub-macros/src/call.rs \
  rccl-cub-macros/src/catalog.rs; then
  printf '%s\n' 'WarpReduce implementation reintroduced erased or raw-pointer machinery' >&2
  exit 1
fi

source scripts/oxide-env.sh

cleanup_sidecars() {
  rm -f -- \
    warp_reduce.options \
    warp_reduce.target \
    warp_reduce_catch2.options \
    warp_reduce_catch2.target
}
trap cleanup_sidecars EXIT

for example in warp_reduce warp_reduce_catch2; do
  cargo oxide build \
    --features cuda-oxide-example \
    -- \
    --release \
    --package rccl-cub \
    --example "$example"
done

./target/release/examples/warp_reduce
./target/release/examples/warp_reduce_catch2

printf '%s\n' 'Pinned CCCL WarpReduce Rust port through standard cuda-oxide build: PASS'
