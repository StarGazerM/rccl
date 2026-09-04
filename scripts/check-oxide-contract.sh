#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
setup_script="$repo_root/scripts/setup-cuda-oxide.sh"
oxide_env="$repo_root/scripts/oxide-env.sh"
rust_toolchain_file="$repo_root/rust-toolchain.toml"
rust_config="$repo_root/.cargo/config.toml"
root_manifest="$repo_root/Cargo.toml"
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

required_files=(
  "$setup_script"
  "$oxide_env"
  "$rust_toolchain_file"
  "$rust_config"
  "$root_manifest"
  "$lockfile"
  "$readme"
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || fail "required cuda-oxide contract file is missing: $required_file"
done

expected_toolchain="$(toml_string channel "$rust_toolchain_file")"
setup_toolchain="$(sh_assignment oxide_toolchain "$setup_script")"
expected_commit="$(sh_assignment oxide_commit "$setup_script")"
expected_patch_rel="$(sh_assignment oxide_patch_rel "$setup_script")"
expected_patch_sha256="$(sh_assignment oxide_patch_sha256 "$setup_script")"
setup_patch_expression="$(sh_assignment oxide_patch "$setup_script")"

[[ "$expected_toolchain" =~ ^nightly-[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] ||
  fail "invalid nightly pin in rust-toolchain.toml: $expected_toolchain"
[[ "$setup_toolchain" == "$expected_toolchain" ]] ||
  fail "mismatched nightly pin: rust-toolchain.toml=$expected_toolchain, setup-cuda-oxide.sh=$setup_toolchain"
[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] ||
  fail "invalid cuda-oxide commit pin in setup-cuda-oxide.sh: $expected_commit"
[[ "$expected_patch_sha256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "invalid cuda-oxide patch checksum in setup-cuda-oxide.sh: $expected_patch_sha256"
[[ -n "$expected_patch_rel" && "$expected_patch_rel" != /* &&
  "/$expected_patch_rel/" != *"/../"* && "/$expected_patch_rel/" != *"/./"* ]] ||
  fail "cuda-oxide patch path must be repository-relative: $expected_patch_rel"
[[ "$setup_patch_expression" == '$repo_root/$oxide_patch_rel' ]] ||
  fail "setup-cuda-oxide.sh must resolve oxide_patch from the repository root"

expected_patch="$repo_root/$expected_patch_rel"
[[ -f "$expected_patch" ]] ||
  fail "required cuda-oxide patch is missing: $expected_patch_rel"
actual_patch_sha256="$(sha256sum "$expected_patch" | awk '{print $1}')"
[[ "$actual_patch_sha256" == "$expected_patch_sha256" ]] ||
  fail "cuda-oxide patch checksum mismatch: expected $expected_patch_sha256, found $actual_patch_sha256"
for documented_identity in \
  "$expected_toolchain" \
  "$expected_commit" \
  "$expected_patch_rel" \
  "$expected_patch_sha256"; do
  grep -Fq "$documented_identity" "$readme" ||
    fail "README.md is missing cuda-oxide contract identity: $documented_identity"
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
  fail "toolkit and CUDA_HOME defaults are inconsistent: $default_cuda_toolkit vs $default_cuda_home"
[[ "$default_libnvvm" == "$default_cuda_home/nvvm/lib64/libnvvm.so" ]] ||
  fail "unexpected LIBNVVM_PATH default: $default_libnvvm"
[[ "$default_libdevice" == "$default_cuda_home/nvvm/libdevice/libdevice.10.bc" ]] ||
  fail "unexpected CUDA_OXIDE_LIBDEVICE default: $default_libdevice"
grep -Fq "$default_cuda_home" "$oxide_env" ||
  fail "oxide-env.sh does not include the configured CUDA_HOME default: $default_cuda_home"

grep -Eq '^\[patch\."https://github\.com/NVlabs/cuda-oxide\.git"\]$' "$root_manifest" ||
  fail 'missing root [patch] section for cuda-oxide source override'
for crate_name in cuda-device cuda-host cuda-macros; do
  grep -Eq "^${crate_name}[[:space:]]*=[[:space:]]*\\{[[:space:]]*path[[:space:]]*=[[:space:]]*\"vendor/${crate_name}\"[[:space:]]*\\}$" \
    "$root_manifest" ||
    fail "root [patch] is missing vendor path for $crate_name"
done

required_dependency_manifests=(
  rccl-cub/Cargo.toml
  rccl-cuco/Cargo.toml
  rccl-memory/Cargo.toml
  rccl-thrust/Cargo.toml
  vendor/cuda-device/Cargo.toml
  vendor/cuda-host/Cargo.toml
  vendor/cuda-macros/Cargo.toml
)
for manifest_rel in "${required_dependency_manifests[@]}"; do
  [[ -f "$repo_root/$manifest_rel" ]] ||
    fail "required cuda-oxide manifest is missing: $manifest_rel"
done

mapfile -d '' manifests < <(
  find "$repo_root" \
    \( -path "$repo_root/.git" -o -path "$repo_root/target" \) -prune -o \
    -name Cargo.toml -print0 | sort -z
)
pinned_manifest_count=0
for manifest in "${manifests[@]}"; do
  if ! grep -Eq 'git[[:space:]]*=[[:space:]]*"https://github\.com/NVlabs/cuda-oxide\.git"' "$manifest"; then
    continue
  fi
  manifest_rel="${manifest#"$repo_root/"}"
  pinned_manifest_count=$((pinned_manifest_count + 1))
  awk -v expected="$expected_commit" -v file="$manifest_rel" '
    /git[[:space:]]*=[[:space:]]*"https:\/\/github.com\/NVlabs\/cuda-oxide.git"/ {
      pin = "rev[[:space:]]*=[[:space:]]*\"" expected "\""
      if ($0 !~ pin) {
        printf "manifest pin mismatch in %s: %s\n", file, $0 > "/dev/stderr"
        bad = 1
      }
    }
    END { if (bad) exit 1 }
  ' "$manifest" || exit $?
done
((pinned_manifest_count >= ${#required_dependency_manifests[@]})) ||
  fail "expected at least ${#required_dependency_manifests[@]} cuda-oxide dependency manifests, found $pinned_manifest_count"

lock_sources="$(
  awk -F'"' '
    /source = "git\+https:\/\/github.com\/NVlabs\/cuda-oxide.git\?rev=/ {
      print $2
    }
  ' "$lockfile" | sort -u
)"
[[ -n "$lock_sources" ]] ||
  fail 'Cargo.lock has no cuda-oxide source revisions'
expected_lock_source="git+https://github.com/NVlabs/cuda-oxide.git?rev=$expected_commit#$expected_commit"
while IFS= read -r lock_source; do
  [[ "$lock_source" == "$expected_lock_source" ]] ||
    fail "Cargo.lock contains unexpected cuda-oxide source: $lock_source"
done <<<"$lock_sources"

patched_checkout="$repo_root/target/cuda-oxide-patched"
if [[ -e "$patched_checkout" ]]; then
  [[ -d "$patched_checkout/.git" ]] ||
    fail 'target/cuda-oxide-patched exists but is not a git checkout'
  patched_head="$(git -C "$patched_checkout" rev-parse HEAD)"
  [[ "$patched_head" == "$expected_commit" ]] ||
    fail "target/cuda-oxide-patched is at $patched_head, expected $expected_commit"

  expected_dir="$(mktemp -d)"
  expected_index="$expected_dir/index"
  cleanup_index() {
    rm -rf -- "$expected_dir"
  }
  trap cleanup_index EXIT
  GIT_INDEX_FILE="$expected_index" git -C "$patched_checkout" read-tree HEAD
  GIT_INDEX_FILE="$expected_index" git -C "$patched_checkout" apply --cached "$expected_patch"
  expected_tree="$(GIT_INDEX_FILE="$expected_index" git -C "$patched_checkout" write-tree)"
  actual_tree="$(git -C "$patched_checkout" write-tree)"
  [[ -z "$(git -C "$patched_checkout" ls-files --others --exclude-standard)" ]] &&
    git -C "$patched_checkout" diff --quiet &&
    [[ "$actual_tree" == "$expected_tree" ]] ||
    fail 'target/cuda-oxide-patched does not exactly match the required local patch'
  cleanup_index
  trap - EXIT
fi

printf '%s\n' 'cuda-oxide reproducibility contract'
printf '  rustup toolchain      : %s\n' "$expected_toolchain"
printf '  cuda-oxide commit     : %s\n' "$expected_commit"
printf '  local patch file      : %s\n' "$expected_patch_rel"
printf '  local patch sha256    : %s\n' "$expected_patch_sha256"
printf '  CUDA_HOME default     : %s\n' "$default_cuda_home"
printf '  LIBNVVM_PATH default  : %s\n' "$default_libnvvm"
printf '  LIBDEVICE default     : %s\n' "$default_libdevice"
printf '%s\n' '  contract checks       : PASS'