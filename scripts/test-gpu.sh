#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"

source scripts/oxide-env.sh

metadata="$(cargo metadata --no-deps --format-version 1)"

mapfile -t workspace_packages < <(jq -r '.packages[].name' <<<"$metadata" | sort)
expected_packages=(
  rccl
  rccl-cub
  rccl-memory
  rccl-renamed-dependency-fixture
  rccl-thrust
  rccl-thrust-direct-fixture
)
if ! diff -u \
  <(printf '%s\n' "${expected_packages[@]}") \
  <(printf '%s\n' "${workspace_packages[@]}"); then
  printf '%s\n' 'the workspace package inventory changed or an obsolete layer returned' >&2
  exit 1
fi

package_dependencies() {
  jq -r --arg package "$1" \
    '.packages[] | select(.name == $package) | .dependencies[] | select(.optional == false) | .name' \
    <<<"$metadata" | sort | paste -sd ' ' -
}

if [[ "$(package_dependencies rccl-cub)" != 'cuda-core cuda-device cuda-host rccl-memory' ]]; then
  printf '%s\n' 'rccl-cub must depend only on CUDA and rccl-memory' >&2
  exit 1
fi
if [[ "$(package_dependencies rccl-thrust)" != 'cuda-core rccl-cub rccl-memory' ]]; then
  printf '%s\n' 'rccl-thrust must own the host API and depend directly on rccl-cub' >&2
  exit 1
fi
if [[ "$(package_dependencies rccl)" != 'rccl-cub rccl-thrust' ]]; then
  printf '%s\n' 'the rccl umbrella must remain a two-crate re-export boundary' >&2
  exit 1
fi

obsolete_crates=(rccl-device-core rccl-macros rccl-runtime)
for obsolete in "${obsolete_crates[@]}"; do
  if [[ -e "$obsolete" ]]; then
    printf '%s\n' "obsolete crate directory $obsolete still exists" >&2
    exit 1
  fi
done
architecture_roots=(
  Cargo.toml Cargo.lock README.md STRICT_CCCL_API_AUDIT.md
  CCCL_MERGE_COMPATIBILITY.md CCCL_WARP_MERGE_SORT_COMPATIBILITY.md
  rccl-cub rccl-memory rccl-thrust src tests examples
)
if rg -n 'rccl[_-](device[_-]core|macros|runtime)|sort_module!|merge_module!' \
  "${architecture_roots[@]}"; then
  printf '%s\n' 'an obsolete crate or generated-module API is still referenced' >&2
  exit 1
fi
if rg -n '\bExecution\b|\b(use|extern crate)\s+rccl_thrust\b' rccl-cub/src rccl-cub/Cargo.toml; then
  printf '%s\n' 'CUB imports a Thrust execution policy or host facade' >&2
  exit 1
fi
if rg -n '\b(proc_macro|proc-macro-crate)\b' Cargo.toml rccl-thrust rccl-cub src; then
  printf '%s\n' 'procedural-macro infrastructure survived the ordinary API refactor' >&2
  exit 1
fi
if rg -n '__private' rccl-cub rccl-thrust src tests examples; then
  printf '%s\n' 'the obsolete compiler-ABI namespace survived the refactor' >&2
  exit 1
fi
if rg -n '#\[cuda_host::cuda_module\]|#\[kernel\]|struct Device(Merge|RadixSort)' src/lib.rs; then
  printf '%s\n' 'the rccl umbrella owns implementation kernels instead of remaining a thin re-export crate' >&2
  exit 1
fi
if (( $(wc -l < src/lib.rs) > 40 )); then
  printf '%s\n' 'the rccl umbrella grew beyond a thin re-export boundary' >&2
  exit 1
fi

mapfile -t example_sources < <(
  find "$task_root/examples" -maxdepth 1 -type f -name '*.rs' ! -name 'gpu_support.rs' | sort
)
mapfile -t registered_example_sources < <(
  jq -r '.packages[] | select(.name == "rccl") | .targets[] | select(.name | startswith("rccl-example-")) | .src_path' \
    <<<"$metadata" | sort
)
if ! diff -u <(printf '%s\n' "${example_sources[@]}") <(printf '%s\n' "${registered_example_sources[@]}"); then
  printf '%s\n' 'an RCCL example is unregistered or a registered example source is missing' >&2
  exit 1
fi

mapfile -t memory_examples < <(
  jq -r '.packages[] | select(.name == "rccl-memory") | .targets[] | select(.kind[] == "example") | .name' \
    <<<"$metadata" | sort
)
if [[ "${memory_examples[*]}" != 'allocation_trace private_pool' ]]; then
  printf '%s\n' "the rccl-memory runtime example inventory changed: ${memory_examples[*]}" >&2
  exit 1
fi

mapfile -t registered_bins < <(
  jq -r '.packages[] | select(.name == "rccl") | .targets[] | select(.kind[] == "bin") | .name' \
    <<<"$metadata" | sort
)
runtime_bins=(
  cccl-comparison-tests
  cccl-merge-tests
  cccl-sort-tests
  cccl-warp-merge-sort-tests
  merge-benchmark
  rccl
  warp-merge-sort-benchmark
)
for source in "${registered_example_sources[@]}"; do
  stem="$(basename "$source" .rs)"
  runtime_bins+=("rccl-example-${stem//_/-}")
done
mapfile -t runtime_bins < <(printf '%s\n' "${runtime_bins[@]}" | sort)
if ! diff -u <(printf '%s\n' "${registered_bins[@]}") <(printf '%s\n' "${runtime_bins[@]}"); then
  printf '%s\n' 'an RCCL binary is built but omitted from the runtime gate' >&2
  exit 1
fi

if find "$task_root" -maxdepth 1 -type f \
  \( -name 'rccl_example_cooperative_sort.*' -o -name 'rccl_example_custom_radix_sort.*' \
     -o -name 'rccl_example_iterator_sort.*' -o -name 'rccl_example_segmented_sort.*' \
     -o -name 'rccl_example_smart_sort.*' \) | rg .; then
  printf '%s\n' 'obsolete generated artifacts from deleted compatibility examples remain' >&2
  exit 1
fi

cleanup_codegen_artifacts() {
  find "$task_root" -maxdepth 1 -type f \
    \( -name 'rccl*.ll' -o -name 'rccl*.ptx' -o -name 'rccl*.cubin' -o -name 'rccl*.ltoir' \
       -o -name 'cccl*.ll' -o -name 'cccl*.ptx' -o -name 'cccl*.cubin' -o -name 'cccl*.ltoir' \
       -o -name 'merge_benchmark*.ll' -o -name 'merge_benchmark*.ptx' \
       -o -name 'merge_benchmark*.cubin' -o -name 'merge_benchmark*.ltoir' \
       -o -name 'warp_merge_sort_benchmark*.ll' -o -name 'warp_merge_sort_benchmark*.ptx' \
       -o -name 'warp_merge_sort_benchmark*.cubin' -o -name 'warp_merge_sort_benchmark*.ltoir' \) \
    -delete
}
cleanup_codegen_artifacts
trap cleanup_codegen_artifacts EXIT

sort_consumers=(
  src/main.rs
  src/bin/cccl-sort-tests.rs
  src/bin/cccl-comparison-tests.rs
  src/bin/cccl-merge-tests.rs
  tests/sort_api.rs
  tests/shared_execution.rs
  tests/fixtures/renamed-dependency/src/lib.rs
  tests/fixtures/thrust-direct/src/lib.rs
  examples/automatic-sort.rs
  examples/bucket-sort2d.rs
  examples/find-non-trivial-runs.rs
  examples/lexicographical-sort.rs
  examples/mode.rs
  examples/resource_sort.rs
  examples/sort-by-key.rs
  examples/sort.rs
  examples/sorting-aos-vs-soa.rs
)
for consumer in "${sort_consumers[@]}"; do
  if ! rg -q '\b(sort|sort_by|stable_sort|stable_sort_by|sort_by_key|stable_sort_by_key|sort_by_key_with|stable_sort_by_key_with)\s*\(' "$consumer"; then
    printf '%s\n' "declared sort consumer $consumer contains no ordinary sort call" >&2
    exit 1
  fi
done
forbidden_sort_api='\b(CudaExecutionPolicy|Sorter|SortSchedule|ComparisonPolicy|DefaultSortKey|KeySortDispatch|PairSortDispatch|OneSweepAscending|OneSweepDescending|RadixCodeMerge|RadixKey|DeviceU32IteratorMut|DeviceIteratorMapping|DeviceRandomAccessMut|DeviceIteratorMut|sort_iter|stable_sort_iter|sort_by_key_iter|stable_sort_by_key_iter|gpu_sort)\b|rccl::testing'
if rg -n "$forbidden_sort_api" "${sort_consumers[@]}"; then
  printf '%s\n' 'retained sort consumers use a removed compatibility or TMP API' >&2
  exit 1
fi

forbidden_sort_lowering='struct CompiledSortSchedule|struct U32RadixKernelSet|struct PolicyConfig|enum SortOrder|descending: bool|fn __radix_config\(|const __RADIX_CONFIG|const __MERGE_BLOCK_THREADS|FA: Fn|FD: Fn|\.compute_capability\(\)'
if rg -n "$forbidden_sort_lowering" rccl-cub/src/radix.rs rccl-thrust/src/algorithm.rs rccl-cub/build.rs; then
  printf '%s\n' 'sort lowering retained a descriptor, runtime architecture selector, or runtime order' >&2
  exit 1
fi

merge_consumers=(
  src/bin/cccl-merge-tests.rs
  src/bin/merge-benchmark.rs
  tests/merge_api.rs
  tests/shared_execution.rs
  tests/fixtures/renamed-dependency/src/lib.rs
  tests/fixtures/thrust-direct/src/lib.rs
)

for consumer in "${merge_consumers[@]}"; do
  if ! rg -q '\b(merge|merge_by|merge_by_key|merge_by_key_with)\s*\(' "$consumer"; then
    printf '%s\n' "declared merge consumer $consumer contains no ordinary merge call" >&2
    exit 1
  fi
done

if rg -n '^#!\[feature\(generic_const_exprs\)\]|^#!\[allow\(incomplete_features' \
  "${sort_consumers[@]}" "${merge_consumers[@]}" tests/shared_execution.rs; then
  printf '%s\n' 'a sort/merge consumer still requires generic_const_exprs' >&2
  exit 1
fi

forbidden_merge_api='\b(CudaExecutionPolicy|CudaMergeExecutionPolicy|MergeSchedule|DeviceMerge|MergeWorkspace|MERGE_POLICIES|MergeInputIterator|MergeOutputIterator|MergeCountingIterator|MergeDiscardIterator|merge_iterators|merge_by_key_iterators|merge_descending|merge_by_key_descending|gpu_merge)\b|\.advanced\(\)|\.on\('
if rg -n "$forbidden_merge_api" "${merge_consumers[@]}"; then
  printf '%s\n' 'retained merge consumers use a removed compatibility or TMP API' >&2
  exit 1
fi

forbidden_merge_selector='struct MergePolicyConfig|const __CONFIG|struct MergeLauncher|F: Fn|select_config|launcher\.sm\(\)|fn __config\(|\.compute_capability\(\)|const MIN_SM|UnsupportedArchitecture'
if rg -n "$forbidden_merge_selector" rccl-cub/src/merge.rs rccl-cub/build.rs; then
  printf '%s\n' 'merge still contains runtime architecture-selection plumbing' >&2
  exit 1
fi

if rg -n 'pub struct CudaExecutionPolicy|pub struct CudaMergeExecutionPolicy|__inner:.*ExecutionPolicy' rccl-cub/src rccl-thrust/src src; then
  printf '%s\n' 'an algorithm-specific execution wrapper survived' >&2
  exit 1
fi

if rg -n 'trait (RadixPolicy|SortPolicy|SortConstInvocation)|impl .*?(RadixPolicy|SortPolicy|SortConstInvocation)|struct __RcclSortSchedule|type __RcclSortSchedule' rccl-cub/src rccl-thrust/src src; then
  printf '%s\n' 'sort retained a policy-type schedule or generated schedule ZST' >&2
  exit 1
fi

if rg -n 'struct DispatchEngine|SortDispatchError' rccl-cub/src rccl-thrust/src src; then
  printf '%s\n' 'sort retained the eager runtime dispatch wrapper' >&2
  exit 1
fi

if rg -n 'trait (MergeKernelPolicy|MergeConstInvocation)|impl .*?(MergeKernelPolicy|MergeConstInvocation)|type __(Key|Pair)Policy|PolicyId::new|fn stable_name_hash' rccl-cub/src/merge.rs; then
  printf '%s\n' 'merge retained policy-type TMP or manufactured policy identity' >&2
  exit 1
fi

if rg -n 'rccl::testing|\.on\(|par_in\(&context' rccl-memory/README.md; then
  printf '%s\n' 'memory documentation retained a deleted execution-policy API' >&2
  exit 1
fi

lint_roots=(rccl-cub rccl-memory rccl-thrust src tests examples)
mapfile -t allow_files < <(rg -l '#!?\[allow\(' --glob '*.rs' "${lint_roots[@]}" | sort)
expected_allow_files=(
  rccl-cub/src/comparison.rs
  rccl-cub/src/merge.rs
  rccl-cub/src/radix.rs
  src/bin/cccl-warp-merge-sort-tests.rs
)
if ! diff -u \
  <(printf '%s\n' "${expected_allow_files[@]}") \
  <(printf '%s\n' "${allow_files[@]}"); then
  printf '%s\n' 'the narrow kernel-macro lint-suppression inventory changed' >&2
  exit 1
fi
allow_site_count="$(rg -o '#!?\[allow\(' --glob '*.rs' "${lint_roots[@]}" | wc -l)"
if [[ "$allow_site_count" -ne 5 ]]; then
  printf '%s\n' "found $allow_site_count allow attributes; expected five kernel-macro sites" >&2
  exit 1
fi
if rg -U --pcre2 '#!?\[allow\([^]]*\b(dead_code|unused(?:_\w+)?|warnings)\b[^]]*\)\]' \
  --glob '*.rs' "${lint_roots[@]}"; then
  printf '%s\n' 'blanket dead/unused-code warning suppression is forbidden' >&2
  exit 1
fi

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- \
  -D warnings \
  -D clippy::allow_attributes_without_reason
for round in 1 2; do
  printf 'host test round %d/2\n' "$round"
  cargo test --workspace --no-fail-fast
done
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
cargo oxide doctor
cargo oxide build --arch "${RCCL_ARCH:-sm_89}" --features gpu-examples

merge_entry_count="$(strings target/release/merge-benchmark | rg -c '^\.visible \.entry merge_(keys|pairs)')"
if [[ "$merge_entry_count" -ne 4 ]]; then
  printf '%s\n' "merge benchmark embedded $merge_entry_count merge kernels; expected one normal and one fallback entry per key/pair shape" >&2
  exit 1
fi

for round in 1 2; do
  printf 'GPU runtime and example round %d/2\n' "$round"
  target/release/cccl-sort-tests
  target/release/cccl-comparison-tests
  target/release/cccl-merge-tests
  target/release/cccl-warp-merge-sort-tests

  target/release/rccl --elements 65536 --iterations 1
  target/release/merge-benchmark --elements-per-input 65536 --iterations 1
  target/release/warp-merge-sort-benchmark --elements 262144 --iterations 1

  cargo run --quiet --release -p rccl-memory --example private_pool
  cargo run --quiet --release -p rccl-memory --example allocation_trace -- \
    --allocations=64 --max-allocation=4096 --max-live-bytes=1048576 --warmups=0 --repetitions=1

  for example in "${registered_bins[@]}"; do
    if [[ "$example" != rccl-example-* ]]; then
      continue
    fi
    "target/release/$example"
  done
done

scripts/test-cccl-references.sh
