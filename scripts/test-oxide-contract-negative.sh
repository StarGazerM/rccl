#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
fixture="$scratch/repo"
trap 'rm -rf -- "$scratch"' EXIT

fixture_paths=(
  .cargo/config.toml
  Cargo.lock
  README.md
  rccl-cub/Cargo.toml
  rccl-cuco/Cargo.toml
  rccl-memory/Cargo.toml
  rccl-thrust/Cargo.toml
  rust-toolchain.toml
  scripts/check-oxide-contract.sh
  scripts/oxide-env.sh
  scripts/setup-cuda-oxide.sh
)

populate_fixture() {
  rm -rf -- "$fixture"
  mkdir -p -- "$fixture"
  tar -C "$repo_root" -cf - "${fixture_paths[@]}" | tar -C "$fixture" -xf -
}

expect_rejection() {
  local name=$1
  local expected_message=$2
  shift 2
  local output
  if output="$("$@" 2>&1)"; then
    printf 'negative fixture unexpectedly passed: %s\n' "$name" >&2
    exit 1
  fi
  if ! grep -Fq "$expected_message" <<<"$output"; then
    printf 'negative fixture failed for the wrong reason: %s\n%s\n' "$name" "$output" >&2
    exit 1
  fi
  printf '  %-31s: rejected\n' "$name"
}

populate_fixture
"$fixture/scripts/check-oxide-contract.sh" >/dev/null

expected_commit=5f7c34fddadcf250f2e45d0cffe83a3d27f2bea3
mismatched_commit=0000000000000000000000000000000000000000
sed -i "s/$expected_commit/$mismatched_commit/g" "$fixture/rccl-cub/Cargo.toml"
expect_rejection 'manifest pin mismatch' 'manifest pin mismatch in rccl-cub/Cargo.toml' \
  "$fixture/scripts/check-oxide-contract.sh"

populate_fixture
sed -i 's#https://github.com/StarGazerM/cuda-oxide.git#https://github.com/NVlabs/cuda-oxide.git#g' \
  "$fixture/rccl-memory/Cargo.toml"
expect_rejection 'manifest origin mismatch' 'manifest pin mismatch in rccl-memory/Cargo.toml' \
  "$fixture/scripts/check-oxide-contract.sh"

printf '%s\n' 'cuda-oxide negative fixture checks: PASS'
