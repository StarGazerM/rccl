#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
setup_script="$repo_root/scripts/setup-cuda-oxide.sh"
oxide_env="$repo_root/scripts/oxide-env.sh"
rust_toolchain_file="$repo_root/rust-toolchain.toml"
rust_config="$repo_root/.cargo/config.toml"
lockfile="$repo_root/Cargo.lock"
readme="$repo_root/README.md"

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

toml_string() {
  local key=$1
  local file=$2
  awk -F'"' -v key="$key" '
    $0 ~ "^[[:space:]]*" key "[[:space:]]*=" { print $2; exit }
  ' "$file"
}

sh_assignment() {
  local key=$1
  local file=$2
  awk -v key="$key" '
    index($0, key "=") == 1 {
      value = substr($0, length(key) + 2)
      if ((substr(value, 1, 1) == "\"" && substr(value, length(value), 1) == "\"") ||
          (substr(value, 1, 1) == "\047" && substr(value, length(value), 1) == "\047")) {
        value = substr(value, 2, length(value) - 2)
      }
      print value
      exit
    }
  ' "$file"
}

for required_file in \
  "$setup_script" "$oxide_env" "$rust_toolchain_file" "$rust_config" "$lockfile" "$readme"; do
  [[ -f "$required_file" ]] ||
    fail "required cuda-oxide contract file is missing: $required_file"
done

expected_toolchain="$(toml_string channel "$rust_toolchain_file")"
expected_origin="$(sh_assignment oxide_origin "$setup_script")"
expected_commit="$(sh_assignment oxide_commit "$setup_script")"
setup_toolchain="$(sh_assignment oxide_toolchain "$setup_script")"
[[ "$expected_toolchain" =~ ^nightly-[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] ||
  fail "invalid nightly pin in rust-toolchain.toml: $expected_toolchain"
[[ "$setup_toolchain" == "$expected_toolchain" ]] ||
  fail "mismatched nightly pin: rust-toolchain.toml=$expected_toolchain, setup-cuda-oxide.sh=$setup_toolchain"
[[ "$expected_origin" == 'https://github.com/StarGazerM/cuda-oxide.git' ]] ||
  fail "unexpected cuda-oxide origin in setup-cuda-oxide.sh: $expected_origin"
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] ||
  fail "invalid cuda-oxide commit pin in setup-cuda-oxide.sh: $expected_commit"
for identity in "$expected_toolchain" "$expected_origin" "$expected_commit"; do
  grep -Fq "$identity" "$readme" ||
    fail "README.md is missing cuda-oxide contract identity: $identity"
done

default_cuda_home="$(toml_string CUDA_HOME "$rust_config")"
default_cuda_toolkit="$(toml_string CUDA_TOOLKIT_PATH "$rust_config")"
default_libnvvm="$(toml_string LIBNVVM_PATH "$rust_config")"
default_libdevice="$(toml_string CUDA_OXIDE_LIBDEVICE "$rust_config")"
[[ -n "$default_cuda_home" && -n "$default_cuda_toolkit" &&
  -n "$default_libnvvm" && -n "$default_libdevice" ]] ||
  fail 'failed to parse CUDA defaults from .cargo/config.toml'
if [[ "$default_cuda_home" =~ /hpc_sdk/Linux_x86_64/([^/]+)/cuda/([^/]+)$ ]]; then
  documented_cuda_matrix="NVIDIA HPC SDK ${BASH_REMATCH[1]}, CUDA ${BASH_REMATCH[2]}"
else
  fail "CUDA_HOME default does not identify the pinned HPC SDK/CUDA matrix: $default_cuda_home"
fi
grep -Fq "$documented_cuda_matrix" "$readme" ||
  fail "README.md is missing CUDA compatibility matrix: $documented_cuda_matrix"
[[ "$default_cuda_toolkit" == "$default_cuda_home" ]] ||
  fail "toolkit and CUDA_HOME defaults are inconsistent"
[[ "$default_libnvvm" == "$default_cuda_home/nvvm/lib64/libnvvm.so" ]] ||
  fail "unexpected LIBNVVM_PATH default: $default_libnvvm"
[[ "$default_libdevice" == "$default_cuda_home/nvvm/libdevice/libdevice.10.bc" ]] ||
  fail "unexpected CUDA_OXIDE_LIBDEVICE default: $default_libdevice"
grep -Fq "$default_cuda_home" "$oxide_env" ||
  fail "oxide-env.sh does not include the configured CUDA_HOME default"

required_dependency_manifests=(
  rccl-cub/Cargo.toml
  rccl-cuco/Cargo.toml
  rccl-memory/Cargo.toml
  rccl-thrust/Cargo.toml
)
for manifest_rel in "${required_dependency_manifests[@]}"; do
  manifest="$repo_root/$manifest_rel"
  [[ -f "$manifest" ]] || fail "required cuda-oxide manifest is missing: $manifest_rel"
  awk -v origin="$expected_origin" -v revision="$expected_commit" -v file="$manifest_rel" '
    /cuda-(core|device|host)[[:space:]]*=/ {
      if (index($0, "git = \"" origin "\"") == 0 ||
          index($0, "rev = \"" revision "\"") == 0) {
        printf "manifest pin mismatch in %s: %s\n", file, $0 > "/dev/stderr"
        bad = 1
      }
      count += 1
    }
    END {
      if (count != 3) {
        printf "expected three cuda-oxide dependencies in %s, found %d\n", file, count > "/dev/stderr"
        bad = 1
      }
      if (bad) exit 1
    }
  ' "$manifest"
done

expected_lock_source="git+$expected_origin?rev=$expected_commit#$expected_commit"
lock_sources="$(
  awk -F'"' -v origin="$expected_origin" '
    index($0, "source = \"git+" origin "?rev=") { print $2 }
  ' "$lockfile" | sort -u
)"
[[ -n "$lock_sources" ]] || fail 'Cargo.lock has no RCCL cuda-oxide source revision'
while IFS= read -r lock_source; do
  [[ "$lock_source" == "$expected_lock_source" ]] ||
    fail "Cargo.lock contains unexpected cuda-oxide source: $lock_source"
done <<<"$lock_sources"

checkout="$repo_root/target/cuda-oxide-rccl"
if [[ -e "$checkout" ]]; then
  [[ -d "$checkout/.git" ]] || fail 'target/cuda-oxide-rccl is not a git checkout'
  [[ "$(git -C "$checkout" rev-parse HEAD)" == "$expected_commit" ]] ||
    fail 'target/cuda-oxide-rccl is at the wrong revision'
  [[ -z "$(git -C "$checkout" status --porcelain --untracked-files=normal)" ]] ||
    fail 'target/cuda-oxide-rccl is dirty'
fi

printf '%s\n' 'cuda-oxide reproducibility contract'
printf '  rustup toolchain      : %s\n' "$expected_toolchain"
printf '  cuda-oxide origin     : %s\n' "$expected_origin"
printf '  cuda-oxide commit     : %s\n' "$expected_commit"
printf '  CUDA_HOME default     : %s\n' "$default_cuda_home"
printf '  LIBNVVM_PATH default  : %s\n' "$default_libnvvm"
printf '  LIBDEVICE default     : %s\n' "$default_libdevice"
printf '%s\n' '  contract checks       : PASS'