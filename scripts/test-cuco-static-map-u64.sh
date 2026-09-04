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

nvcc="$CUDA_HOME/bin/nvcc"
compute_sanitizer="$CUDA_TOOLKIT_PATH/compute-sanitizer/compute-sanitizer"
ptxas="$CUDA_HOME/bin/ptxas"
nvdisasm="$CUDA_HOME/bin/nvdisasm"
for tool in "$nvcc" "$compute_sanitizer" "$ptxas" "$nvdisasm"; do
  if [[ ! -x "$tool" ]]; then
    printf '%s\n' "required CUDA tool unavailable: $tool" >&2
    exit 1
  fi
done

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/rccl-cuco-static-map-u64.XXXXXX")"
cleanup() { rm -rf -- "$build_dir"; }
trap cleanup EXIT

cargo test -p rccl-cuco
cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
  --package rccl-cuco --example static_map_u64
./target/debug/examples/static_map_u64
"$compute_sanitizer" --tool memcheck --error-exitcode 97 ./target/debug/examples/static_map_u64
"$compute_sanitizer" --tool racecheck --error-exitcode 98 ./target/debug/examples/static_map_u64
"$compute_sanitizer" --tool synccheck --error-exitcode 99 ./target/debug/examples/static_map_u64

strings target/debug/examples/static_map_u64 \
  | awk '
      !done && /\.visible \.entry static_map_insert_TID_/ { capture=1; found=1 }
      capture { print }
      capture && /-- End function/ { capture=0; done=1 }
      END { if (!found || !done) exit 1 }
    ' \
  > "$build_dir/static-map-u64-insert.ptx"
if ! rg -q '\.reqntid 128, 1, 1' "$build_dir/static-map-u64-insert.ptx" \
  || ! rg -q 'ld\.acquire\.gpu\.b64' "$build_dir/static-map-u64-insert.ptx" \
  || ! rg -q 'atom\.acq_rel\.gpu\.global\.cas\.b64' "$build_dir/static-map-u64-insert.ptx" \
  || ! rg -q 'st\.relaxed\.gpu\.b64' "$build_dir/static-map-u64-insert.ptx" \
  || ! rg -q 'st\.release\.gpu\.b64' "$build_dir/static-map-u64-insert.ptx" \
  || ! rg -q 'vote\.sync\.ballot\.b32' "$build_dir/static-map-u64-insert.ptx" \
  || ! rg -q 'shfl\.sync\.idx\.b32' "$build_dir/static-map-u64-insert.ptx"; then
  printf '%s\n' 'u32/u64 static-map kernel lost acquire/control-CAS/release two-word protocol' >&2
  exit 1
fi
if rg -n 'atom\..*\.sys|membar|[ls]t\.local|\bcall\b|\.visible \.func' \
  "$build_dir/static-map-u64-insert.ptx"; then
  printf '%s\n' 'u32/u64 static-map kernel contains system atomics, fences, spills, or helper calls' >&2
  exit 1
fi
if ! awk '
  /\.reg \.b32/ { split($0, parts, /%r<|>/); if (parts[2] + 0 > 96) exit 1 }
  /\.reg \.b64/ { split($0, parts, /%rd<|>/); if (parts[2] + 0 > 112) exit 1 }
' "$build_dir/static-map-u64-insert.ptx"; then
  printf '%s\n' 'u32/u64 static-map PTX register use exceeded audit threshold' >&2
  exit 1
fi
{
  printf '%s\n' '.version 7.8' '.target sm_89' '.address_size 64'
  strings target/debug/examples/static_map_u64 \
    | awk '
        !done && /\.visible \.entry static_map_insert_TID_/ { capture=1; found=1 }
        capture && /^\.reqntid/ { print ")"; print; print "{"; next }
        capture && /-- End function/ { print "}"; capture=0; done=1; next }
        capture && $0 !~ /^[[:space:]]*\.loc/ && $0 !~ /^\$L__tmp/ \
          && $0 !~ /function_name/ && $0 !~ /inlined_at/ \
          && $0 !~ /-- Begin function/ { print }
        END { if (!found || !done) exit 1 }
      '
} > "$build_dir/static-map-u64-insert-standalone.ptx"
"$ptxas" -v -arch="${RCCL_ARCH:-sm_89}" \
  "$build_dir/static-map-u64-insert-standalone.ptx" \
  -o "$build_dir/static-map-u64-insert.cubin" 2> "$build_dir/static-map-u64-ptxas.txt"
if rg -n '[1-9][0-9]* bytes (stack frame|spill stores|spill loads)' \
  "$build_dir/static-map-u64-ptxas.txt"; then
  printf '%s\n' 'u32/u64 static-map ptxas reports stack frame or spills' >&2
  exit 1
fi
"$nvdisasm" "$build_dir/static-map-u64-insert.cubin" > "$build_dir/static-map-u64.sass"
if rg -n '\b(LDL|STL|ATOM\.E\.SYS)\b' "$build_dir/static-map-u64.sass"; then
  printf '%s\n' 'u32/u64 static-map SASS contains local-memory traffic or system atomics' >&2
  exit 1
fi
if ! rg -q 'ATOM.*CAS|ATOMG.*CAS' "$build_dir/static-map-u64.sass"; then
  printf '%s\n' 'u32/u64 static-map SASS contains no control CAS' >&2
  exit 1
fi
rg 'Function properties|Used [0-9]+ registers|spill' "$build_dir/static-map-u64-ptxas.txt" || true

"$nvcc" -std=c++20 -O3 --expt-extended-lambda -arch="${RCCL_ARCH:-sm_89}" \
  -I"$cuco_root/include" \
  -I"$cccl_root/cub" \
  -I"$cccl_root/thrust" \
  -I"$cccl_root/libcudacxx/include" \
  benchmarks/cuco-static-map-u64-reference.cu \
  -o "$build_dir/cuco-static-map-u64-reference"
"$build_dir/cuco-static-map-u64-reference"

printf '%s\n' 'Pinned cuCollections static_map<u32,u64> compressed-range matrix: PASS'
