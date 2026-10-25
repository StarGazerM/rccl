#!/usr/bin/env sh
# Build the exact RCCL-maintained cuda-oxide compiler/runtime revision.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
oxide_origin='https://github.com/StarGazerM/cuda-oxide.git'
oxide_commit='496c01bfc5df0dad510bc4ca76e2260d6f586b8f'
oxide_toolchain='1.99.1'
oxide_checkout=${1:-"$repo_root/target/cuda-oxide-rccl"}

if [ ! -d "$oxide_checkout/.git" ]; then
  if [ -e "$oxide_checkout" ]; then
    printf 'refusing non-git cuda-oxide checkout: %s\n' "$oxide_checkout" >&2
    exit 1
  fi
  git clone "$oxide_origin" "$oxide_checkout"
fi

if [ -n "$(git -C "$oxide_checkout" status --porcelain --untracked-files=normal)" ]; then
  printf 'refusing dirty cuda-oxide checkout: %s\n' "$oxide_checkout" >&2
  exit 1
fi
git -C "$oxide_checkout" remote set-url origin "$oxide_origin"
git -C "$oxide_checkout" fetch --depth 1 origin "$oxide_commit"
git -C "$oxide_checkout" checkout --detach "$oxide_commit"
test "$(git -C "$oxide_checkout" rev-parse HEAD)" = "$oxide_commit"

. "$script_dir/oxide-env.sh"
rustup toolchain install "$oxide_toolchain" --profile minimal \
  --component rust-src --component rustc-dev --component llvm-tools
cargo +"$oxide_toolchain" install \
  --path "$oxide_checkout/cuda-oxide/crates/cargo-oxide" --locked --force
(
  cd "$oxide_checkout/cuda-oxide"
  cargo +"$oxide_toolchain" oxide setup
)

printf '%s\n' "RCCL cuda-oxide $oxide_commit is installed"
