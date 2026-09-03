#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"

metadata="$(cargo metadata --no-deps --format-version 1)"
mapfile -t workspace_packages < <(jq -r '.packages[].name' <<<"$metadata" | sort)
expected_packages=(
  rccl
  rccl-cub
  rccl-memory
  rccl-thrust
)
if ! diff -u \
  <(printf '%s\n' "${expected_packages[@]}") \
  <(printf '%s\n' "${workspace_packages[@]}"); then
  printf '%s\n' 'workspace must contain only the four runtime crates' >&2
  exit 1
fi

package_dependencies() {
  jq -r --arg package "$1" \
    '.packages[] | select(.name == $package) | .dependencies[] | select(.optional == false and .kind == null) | .name' \
    <<<"$metadata" | sort | paste -sd ' ' -
}

if [[ "$(package_dependencies rccl-cub)" != 'cuda-core cuda-device cuda-host rccl-memory' ]]; then
  printf '%s\n' 'rccl-cub must depend only on cuda-oxide and rccl-memory' >&2
  exit 1
fi
if [[ "$(package_dependencies rccl-thrust)" != 'cuda-core cuda-device cuda-host rccl-cub rccl-memory' ]]; then
  printf '%s\n' 'rccl-thrust must use rccl-cub fixed kernels and may not depend on a proc-macro facade' >&2
  exit 1
fi
if [[ "$(package_dependencies rccl)" != 'rccl-cub rccl-thrust' ]]; then
  printf '%s\n' 'rccl must remain a thin runtime re-export boundary' >&2
  exit 1
fi

if [[ -e rccl-cub/build.rs || -e rccl-thrust/build.rs || -e rccl-cub-macros ]]; then
  printf '%s\n' 'runtime build scripts may not select targets or generate schedules' >&2
  exit 1
fi
if [[ -e rccl-cub/src/macros || -e rccl-thrust/src/macros ]]; then
  printf '%s\n' 'deleted public macro modules returned' >&2
  exit 1
fi

mapfile -t runtime_sources < <(
  find rccl-cub/src rccl-thrust/src src -type f -name '*.rs' \
    ! -name 'api_tests.rs' ! -name 'tests.rs' | sort
)
forbidden_runtime='proc_macro2::|use[[:space:]]+proc_macro|use[[:space:]]+quote|quote!|TokenStream|CUDA_OXIDE_TARGET|CudaTarget|configured_cuda_target|TargetArchitecture|ArchitecturePolicy|AotKernelSet|TargetSchedule|DispatchEngine|SortConstInvocation|MergeConstInvocation|ReadOnlySlices|ReadOnlyRange'
if rg -n "$forbidden_runtime" "${runtime_sources[@]}"; then
  printf '%s\n' 'runtime code contains a parser, target value, or obsolete compiler-policy type' >&2
  exit 1
fi
if rg -n '#\[macro_export\]' rccl-cub/src rccl-thrust/src \
  --glob '*.rs' --glob '!api_tests.rs'; then
  printf '%s\n' 'CUB and Thrust operations must not be exported as macros' >&2
  exit 1
fi
if rg -n '^proc-macro[[:space:]]*=' Cargo.toml rccl-*/Cargo.toml; then
  printf '%s\n' 'RCCL must not contain a procedural-macro crate' >&2
  exit 1
fi
if find rccl-cub/src rccl-thrust/src -type f -name '*.dsl' | rg .; then
  printf '%s\n' 'RCCL runtime crates must not contain a compiler DSL' >&2
  exit 1
fi
for operation in \
  device_merge_keys device_merge_pairs \
  device_merge_sort_keys device_merge_sort_pairs \
  device_radix_sort_keys device_radix_sort_pairs \
  device_histogram_even device_histogram_range; do
  if ! rg -q "^pub fn $operation" rccl-cub/src; then
    printf '%s\n' "rccl-cub is missing ordinary operation function $operation" >&2
    exit 1
  fi
done
for operation in transform histogram_even histogram_range; do
  if ! rg -q "^pub fn $operation" rccl-thrust/src; then
    printf '%s\n' "rccl-thrust is missing ordinary operation function $operation" >&2
    exit 1
  fi
done
if ! rg -q '^pub fn warp_reduce<T, const N: u32>' rccl-cub/src/warp.rs \
  || ! rg -q '^pub fn warp_reduce_items<T, const N: u32, const ITEMS: usize>' rccl-cub/src/warp.rs; then
  printf '%s\n' 'warp reduction must remain ordinary const-generic functions' >&2
  exit 1
fi
if ! rg -q 'pub trait DeviceSortExt' rccl-thrust/src/ordering.rs \
  || ! rg -q 'pub struct Merge<' rccl-thrust/src/merge.rs \
  || ! rg -q 'pub trait ArithmeticSetRangeExt' rccl-thrust/src/set.rs \
  || ! rg -q 'define_key_set_expression!\(Union,' rccl-thrust/src/set.rs \
  || ! rg -q '^define_key_set_expression!\($' rccl-thrust/src/set.rs \
  || ! rg -q '^    Intersection,$' rccl-thrust/src/set.rs \
  || ! rg -q '^    Difference,$' rccl-thrust/src/set.rs \
  || ! rg -q '^    SymmetricDifference,$' rccl-thrust/src/set.rs \
  || ! rg -q 'pub trait DeviceBufferExt' rccl-thrust/src/range.rs \
  || ! rg -q 'pub trait DeviceRangeExt' rccl-thrust/src/range.rs \
  || rg -n 'pub trait Device(Merge|Set)' rccl-thrust/src \
  || rg -n 'macro_rules! (sort|sort_by|stable_sort|merge|merge_by|set_union|set_intersection|set_difference|set_symmetric_difference)' rccl-thrust/src; then
  printf '%s\n' 'Thrust ordering, merge, sets, and fancy ranges must remain borrowed Rust expression APIs' >&2
  exit 1
fi
if rg -n '(exec|execution)\.(merge|union|intersection|difference|symmetric)|\.with_values_mut\(' \
  rccl-thrust/examples --glob '*.rs'; then
  printf '%s\n' 'Thrust examples regressed to execution-receiver or cursor-like range APIs' >&2
  exit 1
fi

if rg -n '__rccl_(cub|thrust)_kernels|\bunsafe\b|::<[^>]*sm_[0-9]+|cub_program!|thrust_program!' \
  rccl-cub/examples/device_*.rs rccl-thrust/examples --glob '*.rs'; then
  printf '%s\n' 'public device-algorithm examples expose macro-generated backends, unsafe, or target tokens' >&2
  exit 1
fi
if [[ "$(rg -l '#\[cuda_device::cuda_module\]' rccl-cub/examples/warp_reduce*.rs | wc -l)" -ne 4 ]] \
  || [[ "$(rg -l 'unsafe \{ kernels::load\(&context\)' rccl-cub/examples/warp_reduce*.rs | wc -l)" -ne 4 ]] \
  || rg -n '#\[rccl_cub::program\]|kernels::program!|cub_program!' rccl-cub/examples/warp_reduce*.rs; then
  printf '%s\n' 'warp examples must use cuda-oxide modules directly, without an RCCL syntax layer' >&2
  exit 1
fi
if rg -n 'device_merge_(keys|pairs)_partition_count!|device_merge_sort_partition_count!|\.(scratch_keys|scratch_values|bins|lookback|counters)|&mut [A-Za-z0-9_]*(scratch|bins|lookback|counters|partitions)' \
  rccl-cub/examples/device_*.rs --glob '*.rs'; then
  printf '%s\n' 'direct CUB examples expose private temporary-storage components' >&2
  exit 1
fi
for workspace_function in \
  device_histogram_even_workspace \
  device_histogram_even_arithmetic_workspace \
  device_histogram_range_workspace \
  device_histogram_range_arithmetic_workspace \
  device_merge_workspace \
  device_merge_pairs_workspace \
  device_merge_sort_keys_workspace \
  device_merge_sort_pairs_workspace \
  device_radix_sort_keys_workspace \
  device_radix_sort_pairs_workspace \
  device_radix_sort_keys_buffer_workspace \
  device_radix_sort_pairs_buffer_workspace; do
  if ! rg -q "^pub fn $workspace_function" rccl-cub/src; then
    printf '%s\n' "rccl-cub is missing ordinary workspace function $workspace_function" >&2
    exit 1
  fi
done
if rg -n 'fn (get|write)\(' rccl-cub/src/range.rs rccl-thrust/src/range.rs; then
  printf '%s\n' 'range descriptors may not execute device access on the host' >&2
  exit 1
fi
if rg -n 'DeviceRange for \[T\]' rccl-cub/src; then
  printf '%s\n' 'host slices must not masquerade as host-side device ranges' >&2
  exit 1
fi

mapfile -t rust_examples < <(
  find rccl-cub/examples rccl-thrust/examples examples src/bin tests/fixtures \
    -type f -name '*.rs' 2>/dev/null | sort
)
expected_rust_examples=(
  rccl-cub/examples/device_histogram_benchmark.rs
  rccl-cub/examples/device_histogram_counters_32.rs
  rccl-cub/examples/device_histogram_counters_64.rs
  rccl-cub/examples/device_histogram_probe.rs
  rccl-cub/examples/device_histogram_types_16.rs
  rccl-cub/examples/device_histogram_types_32.rs
  rccl-cub/examples/device_histogram_types_64.rs
  rccl-cub/examples/device_histogram_types_8.rs
  rccl-cub/examples/device_merge_sort_keys_benchmark.rs
  rccl-cub/examples/device_merge_sort_keys_probe.rs
  rccl-cub/examples/device_merge_sort_pairs_benchmark.rs
  rccl-cub/examples/device_merge_sort_pairs_probe.rs
  rccl-cub/examples/device_ordering_benchmark.rs
  rccl-cub/examples/device_ordering_probe.rs
  rccl-cub/examples/device_radix_sort_keys_probe.rs
  rccl-cub/examples/device_radix_sort_pairs_probe.rs
  rccl-cub/examples/device_radix_sort_single_tile_pairs_probe.rs
  rccl-cub/examples/device_radix_sort_single_tile_probe.rs
  rccl-cub/examples/support/histogram_counter_probe.rs
  rccl-cub/examples/support/histogram_scalar_probe.rs
  rccl-cub/examples/warp_reduce.rs
  rccl-cub/examples/warp_reduce_benchmark.rs
  rccl-cub/examples/warp_reduce_catch2.rs
  rccl-cub/examples/warp_reduce_catch2_f32.rs
  rccl-thrust/examples/comparison_sort.rs
  rccl-thrust/examples/constant_iterator.rs
  rccl-thrust/examples/constant_iterator_benchmark.rs
  rccl-thrust/examples/histogram.rs
  rccl-thrust/examples/merge.rs
  rccl-thrust/examples/ordering_benchmark.rs
  rccl-thrust/examples/set_operations.rs
  rccl-thrust/examples/set_operations_benchmark.rs
  rccl-thrust/examples/set_operations_tests.rs
  rccl-thrust/examples/sort.rs
)
if ! diff -u \
  <(printf '%s\n' "${expected_rust_examples[@]}") \
  <(printf '%s\n' "${rust_examples[@]}"); then
  printf '%s\n' 'runtime example set drifted from the registered upstream-corresponding ports' >&2
  exit 1
fi

scripts/check-cccl-cub-examples.sh

if [[ "${RCCL_STRUCTURE_ONLY:-0}" == 1 ]]; then
  printf '%s\n' 'RCCL compiler-cfg and ordinary-Rust operation boundary: PASS'
  exit 0
fi

cargo fmt --all -- --check
RUSTFLAGS='-D warnings' cargo check --workspace --all-targets
RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets -- -D warnings
RUSTFLAGS='-D warnings' cargo test --workspace --no-fail-fast
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
scripts/test-warp-reduce.sh
scripts/test-thrust-constant-iterator.sh
scripts/test-thrust-histogram.sh
scripts/test-histogram.sh
scripts/test-ordering-results.sh
scripts/test-set-operations.sh
scripts/test-current-performance-parity.sh
scripts/test-ordering-performance-parity.sh

printf '%s\n' 'RCCL compiler-cfg and ordinary-Rust operation boundary gate: PASS'
printf '%s\n' 'Pinned implemented CUB and Thrust examples plus result/performance parity executed.'
