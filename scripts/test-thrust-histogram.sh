#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh

cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
if [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' "Thrust histogram coverage requires pinned CCCL $cccl_revision" >&2
  exit 1
fi

cleanup() {
  rm -f -- histogram.options histogram.target
}
trap cleanup EXIT

cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features cuda-oxide-example -- \
  --release --package rccl-thrust --example histogram
./target/release/examples/histogram

compute_sanitizer="$CUDA_TOOLKIT_PATH/compute-sanitizer/compute-sanitizer"
if [[ ! -x "$compute_sanitizer" ]]; then
  printf '%s\n' "CUDA compute-sanitizer is unavailable at $compute_sanitizer" >&2
  exit 1
fi
"$compute_sanitizer" --tool memcheck --error-exitcode 97 \
  ./target/release/examples/histogram
"$compute_sanitizer" --tool racecheck --error-exitcode 98 \
  ./target/release/examples/histogram

printf '%s\n' 'Pinned Thrust histogram host facade and sanitizer paths: PASS'
