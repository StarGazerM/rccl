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
if [[ ! -x "$nvcc" ]]; then
  printf '%s\n' "CUDA nvcc is unavailable at $nvcc" >&2
  exit 1
fi


build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-cuco-static-map.XXXXXX")"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

cargo test -p rccl-cuco
cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
  --package rccl-cuco --example static_map
./target/debug/examples/static_map
compute_sanitizer="$CUDA_TOOLKIT_PATH/compute-sanitizer/compute-sanitizer"
if [[ ! -x "$compute_sanitizer" ]]; then
  printf '%s\n' "CUDA compute-sanitizer is unavailable at $compute_sanitizer" >&2
  exit 1
fi
"$compute_sanitizer" --tool memcheck --error-exitcode 97 \
  ./target/debug/examples/static_map
"$compute_sanitizer" --tool racecheck --error-exitcode 98 \
  ./target/debug/examples/static_map
"$compute_sanitizer" --tool synccheck --error-exitcode 99 \
  ./target/debug/examples/static_map
cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
  --release --package rccl-cuco --example static_map_benchmark
strings target/release/examples/static_map_benchmark \
  | awk '
      !done && /\.visible \.entry static_map_insert_fast_TID_/ { capture=1; found=1 }
      capture { print }
      capture && /-- End function/ { capture=0; done=1 }
      END { if (!found || !done) exit 1 }
    ' \
  > "$build_dir/static-map-insert.ptx"
if ! rg -q '\.reqntid 128, 1, 1' "$build_dir/static-map-insert.ptx" \
  || ! rg -q 'atom\.relaxed\.gpu\.global\.cas\.b64' "$build_dir/static-map-insert.ptx" \
  || ! rg -q 'ld\.global\.b32' "$build_dir/static-map-insert.ptx" \
  || ! rg -q 'vote\.sync\.ballot\.b32' "$build_dir/static-map-insert.ptx" \
  || ! rg -q 'shfl\.sync\.idx\.b32' "$build_dir/static-map-insert.ptx"; then
  printf '%s\n' 'static-map fast kernel lost its 128-thread packed-CAS warp schedule' >&2
  exit 1
fi
cas_count="$(rg -c 'atom\.relaxed\.gpu\.global\.cas\.b64' "$build_dir/static-map-insert.ptx")"
shuffle_count="$(rg -c 'shfl\.sync\.idx\.b32' "$build_dir/static-map-insert.ptx")"
if [[ "$cas_count" -ne 1 || "$shuffle_count" -ne 1 ]]; then
  printf '%s\n' 'static-map no-erase path must use one elected-lane CAS and one status shuffle' >&2
  exit 1
fi
if rg -n 'atom\..*\.sys|ld\.relaxed\.gpu\.b64|ld\.global\.b64|rem\.u64|membar|[ls]t?\.local|\bcall\b|\.visible \.func' \
  "$build_dir/static-map-insert.ptx"; then
  printf '%s\n' 'static-map fast kernel contains a wide/coherent probe, system atomic, fence, spill, division, or helper call' >&2
  exit 1
fi

"$nvcc" -std=c++20 -O3 --expt-extended-lambda -arch="${RCCL_ARCH:-sm_89}" \
  -I"$cuco_root/include" \
  -I"$cccl_root/cub" \
  -I"$cccl_root/thrust" \
  -I"$cccl_root/libcudacxx/include" \
  benchmarks/cuco-static-map-reference.cu \
  -o "$build_dir/cuco-static-map-reference"
"$build_dir/cuco-static-map-reference" 65536 0.5 3

printf '%s\n' 'Pinned cuCollections static-map build and correctness matrix: PASS'
