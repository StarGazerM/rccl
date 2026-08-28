#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"

metadata="$(cargo metadata --no-deps --format-version 1)"
mapfile -t workspace_packages < <(jq -r '.packages[].name' <<<"$metadata" | sort)
expected_packages=(
  rccl
  rccl-cub
  rccl-cub-macros
  rccl-memory
  rccl-thrust
)
if ! diff -u \
  <(printf '%s\n' "${expected_packages[@]}") \
  <(printf '%s\n' "${workspace_packages[@]}"); then
  printf '%s\n' 'workspace must contain one compile-time policy crate and four runtime crates' >&2
  exit 1
fi

package_dependencies() {
  jq -r --arg package "$1" \
    '.packages[] | select(.name == $package) | .dependencies[] | select(.optional == false and .kind == null) | .name' \
    <<<"$metadata" | sort | paste -sd ' ' -
}

if [[ "$(package_dependencies rccl-cub)" != 'cuda-core cuda-device cuda-host rccl-cub-macros rccl-memory' ]]; then
  printf '%s\n' 'rccl-cub must keep CUB policy expansion in rccl-cub-macros' >&2
  exit 1
fi
if [[ "$(package_dependencies rccl-thrust)" != 'cuda-core cuda-device cuda-host rccl-cub rccl-memory' ]]; then
  printf '%s\n' 'rccl-thrust must use declarative macros and may not depend on a proc-macro facade' >&2
  exit 1
fi
if [[ "$(package_dependencies rccl)" != 'rccl-cub rccl-thrust' ]]; then
  printf '%s\n' 'rccl must remain a thin runtime re-export boundary' >&2
  exit 1
fi

if [[ -e rccl-cub/build.rs || -e rccl-thrust/build.rs ]]; then
  printf '%s\n' 'runtime build scripts may not select targets or generate schedules' >&2
  exit 1
fi
if rg -n 'emit_support' rccl-cub rccl-thrust rccl-cub-macros --glob '*.rs' --glob 'Cargo.toml'; then
  printf '%s\n' 'ordinary support code must not be emitted by a proc macro' >&2
  exit 1
fi

mapfile -t runtime_sources < <(
  find rccl-cub/src rccl-thrust/src src -type f -name '*.rs' \
    ! -path '*/macros/program.rs' ! -path '*/tests.rs' | sort
)
forbidden_runtime='\b(TypeId|Any|Arc|AotKernelSet|TargetSchedule|KernelSet|DispatchEngine|SortConstInvocation|MergeConstInvocation|Comparator)\b|\bunsafe\b|\*mut|\*const|cuda_module|load_all_ptx|load_module_from_ptx|nvJit|PTX|ptx'
if rg -n "$forbidden_runtime" "${runtime_sources[@]}"; then
  printf '%s\n' 'runtime algorithm code contains compiler metadata, unsafe ownership, or backend compiler control' >&2
  exit 1
fi
if [[ "$(rg -n '\bunsafe\b' rccl-cub/src/macros/program.rs rccl-thrust/src/macros/program.rs | wc -l)" -ne 2 ]] \
  || ! rg -q 'unsafe \{ __rccl_cub_kernels::load\(__rccl_context\) \}' rccl-cub/src/macros/program.rs \
  || ! rg -q 'unsafe \{ __rccl_thrust_kernels::load\(__rccl_context\) \}' rccl-thrust/src/macros/program.rs; then
  printf '%s\n' 'only the two exact cuda-oxide artifact-provenance binds may be unsafe' >&2
  exit 1
fi

forbidden_codegen_loader='cuda_host::cuda_module|load_all_ptx_bundles_merged|load_module_from_ptx|nvJit|cuModuleLoadData|load_function\(|push_kernel_scalar|launch_kernel_on_stream'
if rg -n "$forbidden_codegen_loader" rccl-cub-macros/src rccl-cub/src/macros rccl-thrust/src/macros --glob '*.rs'; then
  printf '%s\n' 'macro expansion contains a runtime loader or manual launch ABI' >&2
  exit 1
fi

if ! rg -q '^syn = \{ version = "3\.0\.4", features = \[' rccl-cub-macros/Cargo.toml \
  || ! rg -q '^syn_derive = "0\.2\.0"$' rccl-cub-macros/Cargo.toml; then
  printf '%s\n' 'the CUB policy macro crate must use current syn and syn_derive parsers' >&2
  exit 1
fi
if find rccl-cub-macros/src rccl-cub/src rccl-thrust/src -type f -name '*.dsl' | rg .; then
  printf '%s\n' 'RCCL uses shallow Rust call macros, not separate DSL source files' >&2
  exit 1
fi
if ! rg -q 'DeviceReduceSumCall' rccl-cub-macros/src/call.rs \
  || ! rg -q 'DeviceReduceArgMinMaxCall' rccl-cub-macros/src/call.rs \
  || ! rg -q 'DeviceReduceByKeyCall' rccl-cub-macros/src/call.rs \
  || ! rg -q '#\[derive\(Debug, syn_derive::Parse\)\]' rccl-cub-macros/src/call.rs \
  || ! rg -q 'Punctuated<Expr, Token!\[,]>' rccl-cub-macros/src/call.rs \
  || ! rg -q 'Punctuated<Type' rccl-cub-macros/src/call.rs \
  || ! rg -q 'pub\(crate\).*: Expr' rccl-cub-macros/src/call.rs; then
  printf '%s\n' 'CUB operation parsing must retain Rust expressions and macro-visible TMP types' >&2
  exit 1
fi

# Proc-macro entry points only parse and dispatch. Fixed quote-only wrappers
# belong to the declarative runtime macro files.
if rg -n 'quote!|use quote' rccl-cub-macros/src/lib.rs \
  || rg -n '^pub fn (cub_program|__device_transform_repeat_in_place)\(' rccl-cub-macros/src/lib.rs; then
  printf '%s\n' 'a quote-only public proc macro returned; use macro_rules for fixed lowering' >&2
  exit 1
fi
if ! rg -q '^pub fn program\(' rccl-cub-macros/src/lib.rs \
  || ! rg -q '^pub fn warp_reduce\(' rccl-cub-macros/src/lib.rs \
  || ! rg -q '^pub fn warp_reduce_items\(' rccl-cub-macros/src/lib.rs \
  || ! rg -q '^pub fn __device_kernels\(' rccl-cub-macros/src/lib.rs \
  || ! rg -q 'device_merge_keys' rccl-cub-macros/src/lib.rs \
  || ! rg -q 'device_radix_sort_keys' rccl-cub-macros/src/lib.rs; then
  printf '%s\n' 'the remaining proc macros must be the item/kernel/policy specializers' >&2
  exit 1
fi
for declarative_macro in \
  cub_program \
  __device_transform_repeat_in_place; do
  if ! rg -q "macro_rules! $declarative_macro" rccl-cub/src/macros/*.rs; then
    printf '%s\n' "missing declarative CUB macro $declarative_macro" >&2
    exit 1
  fi
done
for declarative_macro in \
  thrust_program transform sort sort_by stable_sort stable_sort_by \
  sort_by_key sort_by_key_with stable_sort_by_key stable_sort_by_key_with \
  merge merge_by merge_by_key merge_by_key_with; do
  if ! rg -q "macro_rules! $declarative_macro" rccl-thrust/src/macros/*.rs; then
    printf '%s\n' "missing declarative Thrust macro $declarative_macro" >&2
    exit 1
  fi
done
if rg -n '\$(key|value)_type:ty' rccl-thrust/src/macros --glob '*.rs'; then
  printf '%s\n' 'opaque ty fragments would hide primitive policy input from the inner CUB macro' >&2
  exit 1
fi
if rg -n 'sm_[0-9]+|CUDA_OXIDE_TARGET|CudaTarget|configured_cuda_target' "${runtime_sources[@]}" \
  || [[ "$(rg -l 'configured_cuda_target' rccl-cub-macros/src/*.rs | sort | paste -sd ' ' -)" != 'rccl-cub-macros/src/merge.rs rccl-cub-macros/src/radix.rs rccl-cub-macros/src/target.rs' ]]; then
  printf '%s\n' 'architecture selection escaped the CUB policy expansion boundary' >&2
  exit 1
fi
if rg -n '#\[kernel|cuda_module' \
  rccl-thrust/src/macros/{program,transform,ordering,merge}.rs \
  || ! rg -q '__device_kernels' rccl-thrust/src/macros/program.rs \
  || ! rg -q '__device_transform_repeat_in_place' rccl-thrust/src/macros/transform.rs; then
  printf '%s\n' 'Thrust declarative macros must lower through CUB and never own CUDA kernels' >&2
  exit 1
fi

if rg -n '__rccl_(cub|thrust)_kernels|::load\(|\bunsafe\b|::<[^>]*sm_[0-9]+' \
  rccl-cub/examples/device_*.rs rccl-thrust/examples --glob '*.rs'; then
  printf '%s\n' 'public device-algorithm examples expose backend loading, unsafe, or target tokens' >&2
  exit 1
fi
if rg -n 'cuda_module|::load\(|\bunsafe\b' rccl-cub/examples/warp_reduce*.rs --glob '*.rs' \
  || [[ "$(rg -l '#\[rccl_cub::program\]' rccl-cub/examples/warp_reduce*.rs | wc -l)" -ne 4 ]] \
  || [[ "$(rg -l 'kernels::program!\(&context\)' rccl-cub/examples/warp_reduce*.rs | wc -l)" -ne 4 ]]; then
  printf '%s\n' 'public block/warp examples must use the scoped Rust program macro, never a raw cuda-oxide loader' >&2
  exit 1
fi
if rg -n 'device_merge_(keys|pairs)_partition_count!|device_merge_sort_partition_count!|\.(scratch_keys|scratch_values|bins|lookback|counters)|&mut [A-Za-z0-9_]*(scratch|bins|lookback|counters|partitions)' \
  rccl-cub/examples/device_*.rs --glob '*.rs'; then
  printf '%s\n' 'direct CUB examples expose private temporary-storage components' >&2
  exit 1
fi
for workspace_macro in \
  device_merge_keys_workspace \
  device_merge_pairs_workspace \
  device_merge_sort_keys_workspace \
  device_merge_sort_pairs_workspace \
  device_radix_sort_workspace \
  device_radix_sort_pairs_workspace \
  device_radix_sort_keys_buffer_workspace \
  device_radix_sort_pairs_buffer_workspace; do
  if ! rg -q "$workspace_macro" rccl-cub/src/lib.rs; then
    printf '%s\n' "rccl-cub is missing opaque temporary-storage macro $workspace_macro" >&2
    exit 1
  fi
done
if ! rg -q 'pub\(crate\) const PRIMITIVES: &\[Primitive\]' rccl-cub-macros/src/catalog.rs; then
  printf '%s\n' 'the closed CUB inventory must remain typed compiler input' >&2
  exit 1
fi
if rg -n 'fn (get|write)\(' rccl-cub/src rccl-thrust/src; then
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
  rccl-cub/examples/warp_reduce.rs
  rccl-cub/examples/warp_reduce_benchmark.rs
  rccl-cub/examples/warp_reduce_catch2.rs
  rccl-cub/examples/warp_reduce_catch2_f32.rs
  rccl-thrust/examples/comparison_sort.rs
  rccl-thrust/examples/constant_iterator.rs
  rccl-thrust/examples/constant_iterator_benchmark.rs
  rccl-thrust/examples/merge.rs
  rccl-thrust/examples/ordering_benchmark.rs
  rccl-thrust/examples/sort.rs
)
if ! diff -u \
  <(printf '%s\n' "${expected_rust_examples[@]}") \
  <(printf '%s\n' "${rust_examples[@]}"); then
  printf '%s\n' 'runtime example set drifted from the registered upstream-corresponding ports' >&2
  exit 1
fi

scripts/check-cccl-cub-examples.sh

cargo fmt --all -- --check
RUSTFLAGS='-D warnings' cargo check --workspace --all-targets
RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets -- -D warnings
RUSTFLAGS='-D warnings' cargo test --workspace --no-fail-fast
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
scripts/test-warp-reduce.sh
scripts/test-thrust-constant-iterator.sh
scripts/test-ordering-results.sh
scripts/test-current-performance-parity.sh
scripts/test-ordering-performance-parity.sh

printf '%s\n' 'RCCL compile-time policy and declarative-macro boundary gate: PASS'
printf '%s\n' 'Pinned implemented CUB and Thrust examples plus result/performance parity executed.'
