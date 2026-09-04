#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source scripts/oxide-env.sh

metadata="$(cargo metadata --locked --no-deps --format-version=1)"
mapfile -t cuda_packages < <(
  jq -r '
    .packages[]
    | select(.features | has("cuda-oxide-example"))
    | [
        .name,
        ([.targets[]
          | select(.kind | index("example"))
          | select(.["required-features"] | index("cuda-oxide-example"))]
         | length)
      ]
    | @tsv
  ' <<<"$metadata" | sort
)

if ((${#cuda_packages[@]} == 0)); then
  printf '%s\n' 'no workspace packages expose cuda-oxide example targets' >&2
  exit 1
fi

printf '%s\n' 'SM86 compile-only gate; produced binaries are not executed'
for package_record in "${cuda_packages[@]}"; do
  IFS=$'\t' read -r package example_count <<<"$package_record"
  if [[ -z "$example_count" || "$example_count" == 0 ]]; then
    printf 'package %s exposes cuda-oxide-example but has no gated examples\n' "$package" >&2
    exit 1
  fi
  printf '  compiling %s (%s CUDA examples) for sm_86\n' "$package" "$example_count"
  cargo oxide build --arch sm_86 --features cuda-oxide-example -- \
    --locked --package "$package" --examples
done

printf '%s\n' 'SM86 cuda-oxide examples: COMPILE-ONLY PASS'
printf '%s\n' 'No SM86 runtime correctness or performance claim is implied.'
