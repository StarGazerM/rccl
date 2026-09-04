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
  Cargo.toml
  README.md
  patches/cuda-oxide-read-only-i32x4.patch
  rccl-cub/Cargo.toml
  rccl-cuco/Cargo.toml
  rccl-memory/Cargo.toml
  rccl-thrust/Cargo.toml
  rust-toolchain.toml
  scripts/check-oxide-contract.sh
  scripts/oxide-env.sh
  scripts/setup-cuda-oxide.sh
  vendor/cuda-device/Cargo.toml
  vendor/cuda-host/Cargo.toml
  vendor/cuda-macros/Cargo.toml
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

expected_commit=97f8b2b7882f0c15ad9ce9b53abed5553920caa8
mismatched_commit=0000000000000000000000000000000000000000
sed -i "s/$expected_commit/$mismatched_commit/g" "$fixture/rccl-cub/Cargo.toml"
expect_rejection 'manifest pin mismatch' 'manifest pin mismatch in rccl-cub/Cargo.toml' \
  "$fixture/scripts/check-oxide-contract.sh"

populate_fixture
rm -- "$fixture/patches/cuda-oxide-read-only-i32x4.patch"
expect_rejection 'missing patch (contract)' 'required cuda-oxide patch is missing' \
  "$fixture/scripts/check-oxide-contract.sh"
expect_rejection 'missing patch (setup)' 'missing cuda-oxide patch' \
  "$fixture/scripts/setup-cuda-oxide.sh" "$scratch/unused-checkout"

populate_fixture
printf '\n# deterministic fixture mutation\n' >>"$fixture/patches/cuda-oxide-read-only-i32x4.patch"
expect_rejection 'modified patch (contract)' 'cuda-oxide patch checksum mismatch' \
  "$fixture/scripts/check-oxide-contract.sh"
expect_rejection 'modified patch (setup)' 'cuda-oxide patch checksum mismatch' \
  "$fixture/scripts/setup-cuda-oxide.sh" "$scratch/unused-checkout"

printf '%s\n' 'cuda-oxide negative fixture checks: PASS'
