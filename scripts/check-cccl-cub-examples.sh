#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cccl_root="${1:-${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}}"

if [[ ! -d "$cccl_root/.git" && ! -f "$cccl_root/.git" ]]; then
  printf '%s\n' "pinned CCCL checkout is unavailable at $cccl_root" >&2
  exit 1
fi
if [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' "CCCL checkout is not pinned to $cccl_revision" >&2
  exit 1
fi

temporary_directory="$(mktemp -d)"
cleanup() {
  rm -rf -- "$temporary_directory"
}
trap cleanup EXIT

example_files="$temporary_directory/example-files"
example_regions="$temporary_directory/example-regions"
api_tests="$temporary_directory/api-tests"
contract_sources="$temporary_directory/contract-sources"
public_methods="$temporary_directory/public-methods"

find "$cccl_root/cub/examples" -type f -name '*.cu' -printf '%P\n' \
  | sort >"$example_files"

rg --no-heading '// example-begin' \
  "$cccl_root/cub/examples" "$cccl_root/cub/test" --glob '*.cu' \
  | sed "s#^$cccl_root/##" \
  | sed -E 's/:[[:space:]]*\/\/ example-begin[[:space:]]+/:/' \
  | sort >"$example_regions"

find "$cccl_root/cub/test" -maxdepth 2 -type f \
  \( -name 'catch2_test_*_api.cu' -o -name 'catch2_test_*_env_api.cu' \) \
  -printf '%P\n' | sort >"$api_tests"

{
  find "$cccl_root/cub/examples" -type f -name '*.cu'
  while IFS= read -r test_path; do
    printf '%s\n' "$cccl_root/cub/test/$test_path"
  done <"$api_tests"
} | sort -u >"$contract_sources"

xargs -d '\n' rg -o --no-filename --pcre2 \
  '(?:cub::)?(?:Device[A-Za-z0-9]+|WarpReduceBatched)::[A-Z][A-Za-z0-9_]+' \
  <"$contract_sources" \
  | sed 's/^cub:://' \
  | rg -v '::TempStorage$' \
  | sort -u >"$public_methods"

check_inventory() {
  local description="$1"
  local path="$2"
  local expected_count="$3"
  local expected_digest="$4"
  local actual_count actual_digest
  actual_count="$(wc -l <"$path")"
  actual_digest="$(sha256sum "$path" | cut -d ' ' -f 1)"
  if [[ "$actual_count" != "$expected_count" || "$actual_digest" != "$expected_digest" ]]; then
    printf '%s\n' \
      "$description drifted: expected $expected_count entries/$expected_digest, got $actual_count/$actual_digest" >&2
    printf '%s\n' "normalized inventory retained at $path until this process exits" >&2
    exit 1
  fi
}

check_inventory 'CUB executable examples' "$example_files" 19 \
  1360c7b22993eb985d00cf496cbdf18ae865521738b5c1c17e9fc7ba659d01ae
check_inventory 'CUB embedded example regions' "$example_regions" 378 \
  169860d280d40be39a52b969e35c26f6fd479758dcfa2efcb602218afa2d14c5
check_inventory 'CUB API-test files' "$api_tests" 49 \
  f912154f0cca0b95e4f94d2157719effb188fd6b2124a0bd188be9ad95a537f8
check_inventory 'CUB public methods exercised by examples/API tests' "$public_methods" 105 \
  a213bf7e1282806bbd08f2a3e2641fabbaa0d86dffc6daeae4203706a8267c9b

printf '%s\n' \
  'Pinned CUB source contract: PASS (19 programs, 378 regions, 49 API tests, 105 exercised methods)'
