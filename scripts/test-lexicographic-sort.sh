#!/usr/bin/env bash
# Correctness of sort_columns_lexicographic: contract edges, then every
# strategy on a duplicate-heavy and a wide-key matrix (each run checks all
# permutations against a host stable sort).
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh
arch="${RCCL_ARCH:-sm_89}"
cargo oxide build --arch "$arch" --features cuda-oxide-example -- \
  --release --package rccl-thrust --example lexicographic_sort_strategies
binary=./target/release/examples/lexicographic_sort_strategies
"$binary" --edges
for items in 1 7 3001 100003 1000003; do
  for bits in 1 3,3 2,2,2,2,2,2 32 17,32 32,32,32,32,32,32; do
    "$binary" --items "$items" --bits "$bits" --samples 1
  done
done
