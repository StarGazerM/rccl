#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh

readonly cuco_revision=1d538b6befc286ab450d9e27d7ffbc4f1415c641
readonly cuco_origin=https://github.com/NVIDIA/cuCollections.git
readonly cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
readonly cccl_origin=https://github.com/NVIDIA/cccl.git
cuco_root="${CUCO_ROOT:-$task_root/target/cuco-reference-$cuco_revision}"
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"

if [[ ! -d "$cuco_root/.git" ]]; then
  mkdir -p "$task_root/target"
  staging="$(mktemp -d "$task_root/target/.cuco-reference.XXXXXX")"
  cleanup_staging() { rm -rf -- "$staging"; }
  trap cleanup_staging EXIT
  git -C "$staging" init --quiet
  git -C "$staging" remote add origin "$cuco_origin"
  git -C "$staging" fetch --quiet --depth 1 origin "$cuco_revision"
  git -C "$staging" checkout --quiet --detach FETCH_HEAD
  mv "$staging" "$cuco_root"
  trap - EXIT
fi
if [[ ! -d "$cccl_root/.git" ]]; then
  mkdir -p "$task_root/target"
  staging="$(mktemp -d "$task_root/target/.cccl-reference.XXXXXX")"
  cleanup_staging() { rm -rf -- "$staging"; }
  trap cleanup_staging EXIT
  git -C "$staging" init --quiet
  git -C "$staging" remote add origin "$cccl_origin"
  git -C "$staging" fetch --quiet --depth 1 origin "$cccl_revision"
  git -C "$staging" checkout --quiet --detach FETCH_HEAD
  mv "$staging" "$cccl_root"
  trap - EXIT
fi
if [[ "$(git -C "$cuco_root" rev-parse HEAD)" != "$cuco_revision" ]]; then
  printf '%s\n' "CUCO_ROOT must resolve to pinned cuCollections $cuco_revision" >&2
  exit 1
fi
if [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' "CCCL_ROOT must resolve to pinned CCCL $cccl_revision" >&2
  exit 1
fi

nvcc="${CUDA_HOME}/bin/nvcc"
compute_sanitizer="$CUDA_TOOLKIT_PATH/compute-sanitizer/compute-sanitizer"
if [[ ! -x "$nvcc" ]]; then
  printf '%s\n' "CUDA nvcc is unavailable at $nvcc" >&2
  exit 1
fi
if [[ ! -x "$compute_sanitizer" ]]; then
  printf '%s\n' "CUDA compute-sanitizer is unavailable at $compute_sanitizer" >&2
  exit 1
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-cuco-xxhash64.XXXXXX")"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

cargo test -p rccl-cuco --test xxhash64
cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
  --release --package rccl-cuco --example xxhash64_probe
./target/release/examples/xxhash64_probe > "$build_dir/rust-vectors"
"$compute_sanitizer" --tool memcheck --error-exitcode 97 \
  ./target/release/examples/xxhash64_probe > /dev/null

strings target/release/examples/xxhash64_probe \
  | awk '
      !done && /\.visible \.entry xxhash64_vectors\(/ { capture=1; found=1 }
      capture { print }
      capture && /-- End function/ { capture=0; done=1 }
      END { if (!found || !done) exit 1 }
    ' \
  > "$build_dir/xxhash64-vectors.ptx"
strings target/release/examples/xxhash64_probe \
  | awk '
      !done && /\.visible \.entry xxhash64_values\(/ { capture=1; found=1 }
      capture { print }
      capture && /-- End function/ { capture=0; done=1 }
      END { if (!found || !done) exit 1 }
    ' \
  > "$build_dir/xxhash64-values.ptx"
value_u64_loads="$(rg -c 'ld\.global\.b64' "$build_dir/xxhash64-values.ptx")"
row_u32_loads="$(rg -c 'ld\.global\.b32' "$build_dir/xxhash64-values.ptx")"
if ! rg -q '\.reqntid 256, 1, 1' "$build_dir/xxhash64-vectors.ptx" \
  || ! rg -q '\.reqntid 32, 1, 1' "$build_dir/xxhash64-values.ptx" \
  || ! rg -q 'ld\.global.*\.u8|ld\.global.*\.b8' "$build_dir/xxhash64-vectors.ptx" \
  || ! rg -q 'st\.global.*\.u64|st\.global.*\.b64' "$build_dir/xxhash64-vectors.ptx" \
  || ! rg -q 'st\.global.*\.u64|st\.global.*\.b64' "$build_dir/xxhash64-values.ptx" \
  || [[ "$value_u64_loads" -ne 1 || "$row_u32_loads" -ne 4 ]]; then
  printf '%s\n' 'XXHash64 probes lost direct byte loads, one-load u64 hashing, or four-load u32-row hashing' >&2
  exit 1
fi
if rg -n '(^|[[:space:]])(ld|st)\.local|(^|[[:space:]])call(\.|[[:space:]])|\.visible \.func' \
  "$build_dir/xxhash64-vectors.ptx" "$build_dir/xxhash64-values.ptx"; then
  printf '%s\n' 'XXHash64 probes contain a local spill or out-of-line helper/fallback call' >&2
  exit 1
fi

"$nvcc" -std=c++20 -O3 --expt-extended-lambda -arch="${RCCL_ARCH:-sm_89}" \
  -I"$cuco_root/include" \
  -I"$cccl_root/cub" \
  -I"$cccl_root/thrust" \
  -I"$cccl_root/libcudacxx/include" \
  benchmarks/cuco-xxhash64-reference.cu \
  -o "$build_dir/cuco-xxhash64-reference"
"$build_dir/cuco-xxhash64-reference" > "$build_dir/cuco-vectors"
diff -u "$build_dir/cuco-vectors" "$build_dir/rust-vectors"

printf '%s\n' 'Pinned cuCollections XXHash64 host/device/oracle/code-shape matrix: PASS'
