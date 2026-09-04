#!/usr/bin/env sh
# Build the exact cuda-oxide compiler required by RCCL's architecture cfgs,
# read-only vector loads, and bounds-proved per-thread tiles.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
oxide_commit='97f8b2b7882f0c15ad9ce9b53abed5553920caa8'
oxide_toolchain='nightly-2026-08-28'
oxide_patch_rel='patches/cuda-oxide-read-only-i32x4.patch'
oxide_patch_sha256='9d92f1e838614af92514b062fc3f0540bf22dee6dcbe1596cfd045d6341f163f'
oxide_checkout=${1:-"$repo_root/target/cuda-oxide-patched"}
oxide_patch="$repo_root/$oxide_patch_rel"

if [ ! -f "$oxide_patch" ]; then
  printf 'missing cuda-oxide patch: %s\n' "$oxide_patch" >&2
  exit 1
fi
actual_patch_sha256=$(sha256sum "$oxide_patch" | awk '{print $1}')
if [ "$actual_patch_sha256" != "$oxide_patch_sha256" ]; then
  printf 'cuda-oxide patch checksum mismatch: expected %s, found %s\n' \
    "$oxide_patch_sha256" "$actual_patch_sha256" >&2
  exit 1
fi

if [ ! -d "$oxide_checkout/.git" ]; then
  if [ -e "$oxide_checkout" ]; then
    printf 'refusing non-git cuda-oxide checkout: %s\n' "$oxide_checkout" >&2
    exit 1
  fi
  git clone https://github.com/NVlabs/cuda-oxide.git "$oxide_checkout"
fi

if [ -n "$(git -C "$oxide_checkout" status --porcelain --untracked-files=normal)" ]; then
  expected_dir=$(mktemp -d)
  expected_index="$expected_dir/index"
  trap 'test ! -e "$expected_index" || unlink "$expected_index"; rmdir "$expected_dir"' EXIT HUP INT TERM
  GIT_INDEX_FILE="$expected_index" git -C "$oxide_checkout" read-tree HEAD
  GIT_INDEX_FILE="$expected_index" git -C "$oxide_checkout" apply --cached "$oxide_patch"
  expected_tree=$(GIT_INDEX_FILE="$expected_index" git -C "$oxide_checkout" write-tree)
  actual_tree=$(git -C "$oxide_checkout" write-tree)
  unlink "$expected_index"
  rmdir "$expected_dir"
  trap - EXIT HUP INT TERM
  if [ -n "$(git -C "$oxide_checkout" ls-files --others --exclude-standard)" ] ||
    ! git -C "$oxide_checkout" diff --quiet ||
    [ "$actual_tree" != "$expected_tree" ]; then
    printf 'refusing cuda-oxide checkout with unrelated changes: %s\n' "$oxide_checkout" >&2
    exit 1
  fi
else
  git -C "$oxide_checkout" fetch --depth 1 origin "$oxide_commit"
  git -C "$oxide_checkout" checkout --detach "$oxide_commit"
  git -C "$oxide_checkout" apply --index "$oxide_patch"
fi

if [ "$(git -C "$oxide_checkout" rev-parse HEAD)" != "$oxide_commit" ]; then
  printf 'cuda-oxide checkout is not at required commit %s\n' "$oxide_commit" >&2
  exit 1
fi

git -C "$oxide_checkout" diff HEAD --check
. "$script_dir/oxide-env.sh"

rustup toolchain install "$oxide_toolchain" --profile minimal \
  --component rust-src --component rustc-dev --component llvm-tools
cargo +"$oxide_toolchain" install \
  --path "$oxide_checkout/crates/cargo-oxide" --locked --force
(
  cd "$oxide_checkout"
  cargo +"$oxide_toolchain" oxide setup
)

printf '%s\n' "cuda-oxide $oxide_commit with RCCL architecture cfgs, read-only vectors, and ThreadTile lowering is installed"
