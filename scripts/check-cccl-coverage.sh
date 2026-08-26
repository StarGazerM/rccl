#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ledger="$task_root/cccl-coverage.toml"
cccl_root="${1:-${CCCL_ROOT:-$task_root/target/cccl-reference-571f2fc3bc53cd710e306ad43d58995c1fe4219f}}"

if [[ ! -f "$cccl_root/thrust/thrust/sort.h" || ! -f "$cccl_root/cub/cub/cub.cuh" ]]; then
  printf '%s\n' "CCCL source tree is unavailable at $cccl_root" >&2
  exit 1
fi

ledger_entries="$(mktemp)"
actual_entries="$(mktemp)"
cleanup() {
  rm -f -- "$ledger_entries" "$actual_entries"
}
trap cleanup EXIT

rg -o '"(thrust/thrust/[^" ]+\.h|cub/cub/[^" ]+\.cuh)"' "$ledger" \
  | tr -d '"' \
  | sort >"$ledger_entries"

if duplicate="$(uniq -d "$ledger_entries")" && [[ -n "$duplicate" ]]; then
  printf '%s\n' 'duplicate CCCL coverage-ledger entries:' "$duplicate" >&2
  exit 1
fi

find "$cccl_root/thrust/thrust" -maxdepth 1 -type f -name '*.h' -printf 'thrust/thrust/%f\n' \
  >>"$actual_entries"
for area in iterator mr type_traits random; do
  find "$cccl_root/thrust/thrust/$area" -maxdepth 1 -type f -name '*.h' \
    -printf "thrust/thrust/$area/%f\n" >>"$actual_entries"
done
for area in block device warp thread iterator agent grid detail; do
  find "$cccl_root/cub/cub/$area" -maxdepth 1 -type f -name '*.cuh' \
    -printf "cub/cub/$area/%f\n" >>"$actual_entries"
done
find "$cccl_root/cub/cub" -maxdepth 1 -type f -name '*.cuh' -printf 'cub/cub/%f\n' \
  >>"$actual_entries"
sort -o "$actual_entries" "$actual_entries"

if ! diff -u "$actual_entries" "$ledger_entries"; then
  printf '%s\n' 'the CCCL coverage ledger does not match the pinned public/support inventory' >&2
  exit 1
fi

surface_count="$(rg -c '^\[\[surface\]\]$' "$ledger")"
for field in id kind owner status entries evidence; do
  field_count="$(rg -c "^$field = " "$ledger")"
  if [[ "$field_count" -ne "$surface_count" ]]; then
    printf '%s\n' \
      "coverage ledger has $surface_count surfaces but $field_count $field fields" >&2
    exit 1
  fi
done

if rg -n '^status = "' "$ledger" \
  | rg -v 'status = "(missing|partial|implemented|native-rust|backend-blocked)"$'; then
  printf '%s\n' 'coverage ledger contains an unknown status' >&2
  exit 1
fi

while IFS= read -r review_path; do
  if [[ ! -f "$task_root/$review_path" ]]; then
    printf '%s\n' "coverage ledger review is unavailable: $review_path" >&2
    exit 1
  fi
done < <(rg -o '^review = "[^"]+"$' "$ledger" | cut -d '"' -f 2)

printf 'CCCL coverage ledger: PASS (%s pinned headers in %s surfaces)\n' \
  "$(wc -l <"$ledger_entries")" "$surface_count"
