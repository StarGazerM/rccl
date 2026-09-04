#!/usr/bin/env bash
set -euo pipefail

readonly expected_toolchain='nightly-2026-08-28'
readonly expected_oxide='97f8b2b7882f0c15ad9ce9b53abed5553920caa8'
readonly expected_patch='9d92f1e838614af92514b062fc3f0540bf22dee6dcbe1596cfd045d6341f163f'
readonly expected_cccl='571f2fc3bc53cd710e306ad43d58995c1fe4219f'
readonly expected_cuco='1d538b6befc286ab450d9e27d7ffbc4f1415c641'
readonly oxide_root='/opt/rccl/toolchains/cuda-oxide'
readonly patch_path='/opt/rccl/patches/cuda-oxide-read-only-i32x4.patch'
readonly cccl_root='/opt/rccl/sources/cccl'
readonly cuco_root='/opt/rccl/sources/cuCollections'

fail() {
  printf 'rccl-image-self-check: %s\n' "$*" >&2
  exit 1
}

require_tool() {
  command -v "$1" >/dev/null 2>&1 || fail "missing tool: $1"
}

[[ "$(uname -m)" == x86_64 ]] || fail "image architecture is not x86_64"
[[ "${CUDA_HOME:-}" == /usr/local/cuda ]] || fail 'CUDA_HOME is not /usr/local/cuda'
[[ "${CUDA_TOOLKIT_PATH:-}" == /usr/local/cuda ]] || fail 'CUDA_TOOLKIT_PATH is not /usr/local/cuda'
[[ "${LIBCLANG_PATH:-}" == /usr/lib/llvm-18/lib ]] || fail 'LIBCLANG_PATH is not pinned to LLVM 18'
[[ "${RUSTUP_TOOLCHAIN:-}" == "$expected_toolchain" ]] || fail 'RUSTUP_TOOLCHAIN mismatch'
[[ ":${PATH}:" == *':/opt/cargo/bin:'* ]] || fail 'cargo is absent from PATH'
[[ ":${PATH}:" == *':/usr/local/cuda/bin:'* ]] || fail 'CUDA bin is absent from PATH'

for tool in nvcc ptxas cuobjdump nvdisasm compute-sanitizer clang clang++ git cmake ninja python3 pip3 rsync zstd sqlite3 rustup rustc cargo cargo-oxide; do
  require_tool "$tool"
done
nvcc --version | grep -Fq 'release 12.9' || fail 'nvcc is not CUDA 12.9'
ptxas --version | grep -Fq 'release 12.9' || fail 'ptxas is not CUDA 12.9'
cuobjdump --version | grep -Fq 'release 12.9' || fail 'cuobjdump is not CUDA 12.9'
nvdisasm --version | grep -Fq 'release 12.9' || fail 'nvdisasm is not CUDA 12.9'
compute-sanitizer --version | grep -Fq '2025.2' || fail 'compute-sanitizer is not the CUDA 12.9 release'
[[ -e "${LIBCLANG_PATH}/libclang.so" ]] || fail "libclang.so is absent from ${LIBCLANG_PATH}"
python3 -c 'import venv' || fail 'python venv module is unavailable'

[[ "$(rustup show active-toolchain | awk '{print $1}')" == "$expected_toolchain-x86_64-unknown-linux-gnu" ]] ||
  fail 'active Rust toolchain mismatch'
for component in rust-src rustc-dev rust-analyzer clippy llvm-tools rustfmt; do
  rustup component list --toolchain "$expected_toolchain" --installed |
    grep -Eq "^${component}(-x86_64-unknown-linux-gnu)? \(installed\)$" ||
    fail "missing Rust component: $component"
done
for target in x86_64-unknown-linux-gnu nvptx64-nvidia-cuda; do
  rustup target list --toolchain "$expected_toolchain" --installed |
    grep -Fxq "$target" || fail "missing Rust target: $target"
done
cargo oxide --version >/dev/null
[[ -s /opt/cargo/cuda-oxide/librustc_codegen_cuda.so ]] || fail 'prebuilt cuda-oxide backend cache is absent'

[[ "$(sha256sum "$patch_path" | awk '{print $1}')" == "$expected_patch" ]] || fail 'cuda-oxide patch checksum mismatch'
[[ "$(git -C "$oxide_root" rev-parse HEAD)" == "$expected_oxide" ]] || fail 'cuda-oxide revision mismatch'
[[ -z "$(git -C "$oxide_root" ls-files --others --exclude-standard)" ]] || fail 'cuda-oxide has untracked files'
git -C "$oxide_root" diff --quiet || fail 'cuda-oxide worktree differs from its patched index'
git -C "$oxide_root" diff HEAD --check || fail 'cuda-oxide patch has whitespace errors'
expected_dir="$(mktemp -d)"
expected_index="$expected_dir/index"
cleanup() {
  rm -rf "$expected_dir"
}
trap cleanup EXIT HUP INT TERM
GIT_INDEX_FILE="$expected_index" git -C "$oxide_root" read-tree HEAD
GIT_INDEX_FILE="$expected_index" git -C "$oxide_root" apply --cached "$patch_path"
expected_tree="$(GIT_INDEX_FILE="$expected_index" git -C "$oxide_root" write-tree)"
actual_tree="$(git -C "$oxide_root" write-tree)"
[[ "$actual_tree" == "$expected_tree" ]] || fail 'cuda-oxide indexed tree is not the exact canonical patch state'
cleanup
trap - EXIT HUP INT TERM

for pinned in "$cccl_root:$expected_cccl" "$cuco_root:$expected_cuco"; do
  root="${pinned%%:*}"
  revision="${pinned##*:}"
  [[ "$(git -C "$root" rev-parse HEAD)" == "$revision" ]] || fail "source revision mismatch: $root"
  [[ -z "$(git -C "$root" status --porcelain --untracked-files=normal)" ]] || fail "source checkout is dirty: $root"
done

[[ -x /opt/nvidia/nvidia_entrypoint.sh ]] || fail 'Runpod NVIDIA entrypoint is not inherited'
[[ -x /start.sh ]] || fail 'Runpod /start.sh is not inherited'
require_tool sshd
grep -Eiq '(ssh|authorized_keys)' /start.sh || fail 'Runpod /start.sh no longer contains SSH startup handling'

printf 'rccl-image-self-check: ok cuda=12.9 rust=%s oxide=%s cccl=%s cuco=%s startup=/start.sh\n' \
  "$expected_toolchain" "$expected_oxide" "$expected_cccl" "$expected_cuco"
