#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"

cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
example_source="$cccl_root/thrust/examples/constant_iterator.cu"
test_source="$cccl_root/thrust/testing/constant_iterator.cu"
if [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]]; then
  printf '%s\n' "constant-iterator reference must use pinned CCCL $cccl_revision" >&2
  exit 1
fi

printf '%s  %s\n' \
  fd7953b9bde0ce933106cbe54fab7434b25903d6db774c6d8b8cc14c5cfd242b \
  "$example_source" | sha256sum --check --status
printf '%s  %s\n' \
  40fdf9632920c0c89fab7c2ddcd2a5d524c09dbccce3ea6302e7b544f61be88a \
  "$test_source" | sha256sum --check --status

for required in \
  'rccl_thrust::thrust_program!(&context)' \
  'repeat_n(10_i32, data.len())' \
  'rccl_thrust::transform!(&exec, &mut data, repeated' \
  '[13, 17, 12, 15]'; do
  if ! rg -q -F "$required" rccl-thrust/examples/constant_iterator.rs; then
    printf '%s\n' "constant_iterator port lost required evidence: $required" >&2
    exit 1
  fi
done

source scripts/oxide-env.sh
cleanup() {
  rm -f -- constant_iterator.options constant_iterator.target
}
trap cleanup EXIT
cargo oxide build --features cuda-oxide-example -- \
  --release --package rccl-thrust --example constant_iterator
./target/release/examples/constant_iterator

printf '%s\n' 'Pinned CCCL Thrust constant_iterator Rust port: PASS'
