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
  rccl-thrust-macros
)
if ! diff -u \
  <(printf '%s\n' "${expected_packages[@]}") \
  <(printf '%s\n' "${workspace_packages[@]}"); then
  printf '%s\n' 'workspace must contain the two compiler layers and four runtime layers only' >&2
  exit 1
fi

package_dependencies() {
  jq -r --arg package "$1" \
    '.packages[] | select(.name == $package) | .dependencies[] | select(.optional == false) | .name' \
    <<<"$metadata" | sort | paste -sd ' ' -
}

if [[ "$(package_dependencies rccl-cub)" != 'cuda-core rccl-memory' ]]; then
  printf '%s\n' 'rccl-cub support must be ordinary Rust with no proc-macro or host-loader dependency' >&2
  exit 1
fi
if [[ "$(package_dependencies rccl-thrust)" != 'cuda-core rccl-cub rccl-memory' ]]; then
  printf '%s\n' 'rccl-thrust support must be ordinary Rust and depend on CUB, not a support macro' >&2
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
if rg -n 'emit_support|rccl_(cub|thrust)_macros::' rccl-cub rccl-thrust --glob '*.rs' --glob 'Cargo.toml'; then
  printf '%s\n' 'ordinary support code must not be emitted by a proc macro' >&2
  exit 1
fi

runtime_roots=(rccl-cub/src rccl-thrust/src src)
forbidden_runtime='\b(TypeId|Any|Arc|AotKernelSet|TargetSchedule|KernelSet|DispatchEngine|SortConstInvocation|MergeConstInvocation|Comparator|Less|Greater)\b|\bunsafe\b|\*mut|\*const|cuda_module|load_all_ptx|load_module_from_ptx|nvJit|PTX|ptx'
if rg -n "$forbidden_runtime" "${runtime_roots[@]}" --glob '*.rs'; then
  printf '%s\n' 'runtime algorithm crates contain compile-time metadata, unsafe ownership, or JIT code' >&2
  exit 1
fi

forbidden_codegen_loader='cuda_host::cuda_module|load_all_ptx_bundles_merged|load_module_from_ptx|nvJit|cuModuleLoadData|load_function\(|push_kernel_scalar|launch_kernel_on_stream'
if rg -n "$forbidden_codegen_loader" \
  rccl-cub-macros/src rccl-thrust-macros/src --glob '*.rs'; then
  printf '%s\n' 'proc-macro expansion contains a runtime loader or manual launch ABI' >&2
  exit 1
fi
if ! cmp -s rccl-cub-macros/src/audit.rs rccl-thrust-macros/src/audit.rs; then
  printf '%s\n' 'CUB and Thrust generated-Rust safety audits have drifted apart' >&2
  exit 1
fi
for audit_rule in \
  visit_attribute \
  visit_expr_cast \
  visit_expr_method_call \
  visit_expr_raw_addr \
  visit_expr_unsafe \
  visit_item_foreign_mod \
  visit_item_impl \
  visit_item_static \
  visit_item_trait \
  visit_item_union \
  visit_macro \
  visit_path \
  visit_signature \
  visit_type_bare_fn \
  visit_type_ptr; do
  if ! rg -q "fn $audit_rule" rccl-cub-macros/src/audit.rs; then
    printf '%s\n' "generated-Rust safety audit is missing $audit_rule" >&2
    exit 1
  fi
done

for manifest in rccl-cub-macros/Cargo.toml rccl-thrust-macros/Cargo.toml; do
  if ! rg -q '^syn = \{ version = "2", features = \[' "$manifest"; then
    printf '%s\n' "$manifest must use the typed syn 2 parser" >&2
    exit 1
  fi
done
if find rccl-cub-macros/src rccl-thrust-macros/src -type f -name '*.dsl' | rg .; then
  printf '%s\n' 'RCCL uses shallow Rust call macros, not separate DSL source files' >&2
  exit 1
fi
if ! rg -q 'DeviceReduceSumCall' rccl-cub-macros/src/call.rs \
  || ! rg -q 'DeviceReduceArgMinMaxCall' rccl-cub-macros/src/call.rs \
  || ! rg -q 'DeviceReduceByKeyCall' rccl-cub-macros/src/call.rs \
  || ! rg -q 'SortByCall' rccl-thrust-macros/src/call.rs \
  || ! rg -q 'StableSortByKeyCall' rccl-thrust-macros/src/call.rs \
  || ! rg -q 'Punctuated::<Expr' rccl-cub-macros/src/call.rs \
  || ! rg -q 'Punctuated::<Expr' rccl-thrust-macros/src/call.rs \
  || ! rg -q 'pub\(crate\).*: Expr' rccl-cub-macros/src/call.rs \
  || ! rg -q 'pub\(crate\).*: Expr' rccl-thrust-macros/src/call.rs; then
  printf '%s\n' 'operation DSLs must be shallow syn 2 call ASTs retaining Rust expressions' >&2
  exit 1
fi
if rg -n 'emit_support' rccl-cub-macros/src rccl-thrust-macros/src --glob '*.rs'; then
  printf '%s\n' 'deleted support-generation syntax returned' >&2
  exit 1
fi
if rg -n '^#\[proc_macro\]$' rccl-cub-macros/src/lib.rs rccl-thrust-macros/src/lib.rs; then
  printf '%s\n' 'no operation macro may be exported before its cuda-oxide lowering exists' >&2
  exit 1
fi
if ! rg -q 'pub\(crate\) const PRIMITIVES: &\[Primitive\]' rccl-cub-macros/src/catalog.rs \
  || ! rg -q 'pub\(crate\) const ALGORITHMS: &\[Algorithm\]' rccl-thrust-macros/src/catalog.rs; then
  printf '%s\n' 'closed CCCL inventories must be typed Rust compiler catalogs' >&2
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
if ! rg -q 'generated_rust' rccl-cub-macros/src/audit.rs \
  || ! rg -q 'generated_rust' rccl-thrust-macros/src/audit.rs; then
  printf '%s\n' 'both operation generators must retain their generated-Rust audit' >&2
  exit 1
fi

if find examples src/bin tests/fixtures -type f -name '*.rs' 2>/dev/null | rg .; then
  printf '%s\n' 'stale runtime examples, compatibility fixtures, or GPU binaries survived the compiler rewrite' >&2
  exit 1
fi

scripts/check-cccl-cub-examples.sh

cargo fmt --all -- --check
RUSTFLAGS='-D warnings' cargo check --workspace --all-targets
RUSTFLAGS='-D warnings' cargo clippy --workspace --all-targets -- -D warnings
RUSTFLAGS='-D warnings' cargo test --workspace --no-fail-fast
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps

printf '%s\n' 'RCCL proc-macro compiler and runtime-boundary gate: PASS'
printf '%s\n' 'GPU execution intentionally not run: generated algorithm launch paths are not implemented yet.'
