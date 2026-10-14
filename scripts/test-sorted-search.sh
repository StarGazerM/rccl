#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh
arch="${RCCL_ARCH:-sm_89}"
sanitizer="$CUDA_TOOLKIT_PATH/compute-sanitizer/compute-sanitizer"
[[ -x "$sanitizer" ]] || { printf 'required CUDA sanitizer unavailable: %s\n' "$sanitizer" >&2; exit 1; }
export CUDA_OXIDE_PTX_DIR="${CUDA_OXIDE_PTX_DIR:-$task_root/target/oxide-ir}"
cargo oxide build --arch "$arch" --features cuda-oxide-example -- \
  --release --package rccl-thrust --example sorted_search
./target/release/examples/sorted_search
"$sanitizer" --tool memcheck --error-exitcode 97 ./target/release/examples/sorted_search
"$sanitizer" --tool racecheck --error-exitcode 98 ./target/release/examples/sorted_search
# Expected device faults need their own process and are not sanitizer-clean cases.
./target/release/examples/sorted_search --range-oob
printf 'Sorted-search CPU oracle, memcheck, racecheck and async range-fault propagation: PASS (architecture=%s; not a performance claim)\n' "$arch"
