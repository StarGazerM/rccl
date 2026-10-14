#!/usr/bin/env bash
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
source scripts/oxide-env.sh
readonly cccl_revision=571f2fc3bc53cd710e306ad43d58995c1fe4219f
cccl_root="${CCCL_ROOT:-$task_root/target/cccl-reference-$cccl_revision}"
if [[ ! -d "$cccl_root/.git" ]] || [[ "$(git -C "$cccl_root" rev-parse HEAD)" != "$cccl_revision" ]] || [[ -n "$(git -C "$cccl_root" status --porcelain --untracked-files=no)" ]]; then
  printf 'performance reference requires clean pinned CCCL %s\n' "$cccl_revision" >&2
  exit 1
fi
if [[ "${RCCL_EXCLUSIVE_GPU:-0}" != 1 ]]; then
  printf '%s\n' 'PERFORMANCE_UNVERIFIED: reserve an exclusive GPU and set RCCL_EXCLUSIVE_GPU=1; shared-GPU measurements cannot establish parity.' >&2
  exit 77
fi
check_exclusive() {
  local processes
  if ! processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"; then
    printf '%s\n' 'PERFORMANCE_UNVERIFIED: cannot establish GPU ownership.' >&2
    exit 77
  fi
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    printf 'PERFORMANCE_UNVERIFIED: foreign GPU compute processes: %s\n' "$processes" >&2
    exit 77
  fi
}
check_exclusive
arch="${RCCL_ARCH:-sm_89}"
nvcc="$CUDA_HOME/bin/nvcc"
[[ -x "$nvcc" ]] || { printf 'required CUDA compiler unavailable: %s\n' "$nvcc" >&2; exit 1; }
[[ -x /usr/bin/time ]] || { printf '%s\n' '/usr/bin/time is required for compile time and peak RSS reporting.' >&2; exit 1; }
# Keep raw rounds, setup measurements, build logs, and generated code even on failure.
mkdir -p "$task_root/target"
results="${RCCL_SEARCH_RESULTS_DIR:-$(mktemp -d "$task_root/target/sorted-search-performance.XXXXXX")}"
mkdir -p "$results"
results="$(cd "$results" && pwd)"
export CUDA_OXIDE_PTX_DIR="${CUDA_OXIDE_PTX_DIR:-$task_root/target/oxide-ir}"
mkdir -p "$CUDA_OXIDE_PTX_DIR"
export CUDA_OXIDE_PTX_DIR="$(cd "$CUDA_OXIDE_PTX_DIR" && pwd)"
printf 'Retaining sorted-search artifacts: %s\n' "$results"
{
  printf 'cccl_revision=%s architecture=%s backend=cuda-oxide llvm-nvptx\n' "$cccl_revision" "$arch"
  printf 'CUDA_HOME=%s CUDA_OXIDE_PTX_DIR=%s CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_HOME" "$CUDA_OXIDE_PTX_DIR" "${CUDA_VISIBLE_DEVICES:-unset}"
  printf '%s\n' 'seed=19 key=u64 index=u64 sizes=1024/256,1048576/262144 datasets=uniform-hits,32-way-duplicates,87.5-percent-misses orders=ascending,descending warmups=10 samples=31 rounds=7 stream=default resets=none'
  printf '%s\n' 'Equal-range reference is composed pinned Thrust lower_bound + upper_bound; RCCL uses its production equal_range. GPU event timing excludes allocations/uploads/oracle checks. Cold context and per-case setup are separate. No packed/row-key performance coverage is claimed.'
  printf '%s\n' 'Build timings are first-observed and repeated incremental builds, NOT clean-cache cold-build evidence; existing compiler/dependency caches are retained.'
  nvidia-smi --query-gpu=index,uuid,name,compute_cap,driver_version,memory.total --format=csv
  nvidia-smi
  "$nvcc" --version
  rustc --version --verbose
  cargo oxide --version
  git rev-parse HEAD
  git status --short
} >"$results/metadata.txt" 2>&1
cat "$results/metadata.txt"
/usr/bin/time -v -o "$results/reference-build-time.txt" \
  "$nvcc" -std=c++20 -O3 -arch="$arch" --ptxas-options=-v \
  -I "$cccl_root/cub" -I "$cccl_root/thrust" -I "$cccl_root/libcudacxx/include" \
  benchmarks/thrust-sorted-search-reference.cu -o "$results/thrust-sorted-search-reference" \
  >"$results/reference-build.stdout" 2>"$results/reference-build.stderr"
for build in first-observed repeated-incremental; do
  /usr/bin/time -v -o "$results/rccl-build-$build-time.txt" \
    cargo oxide build --arch "$arch" --features cuda-oxide-example -- \
    --release --package rccl-thrust --example sorted_search_benchmark \
    >"$results/rccl-build-$build.stdout" 2>"$results/rccl-build-$build.stderr"
done
check_exclusive
python3 - "$results" "$task_root/target/release/examples/sorted_search_benchmark" <<'PY'
import itertools
import math
import pathlib
import statistics
import subprocess
import sys

results = pathlib.Path(sys.argv[1])
commands = {"rccl": sys.argv[2], "cccl": str(results / "thrust-sorted-search-reference")}

def require_exclusive(allowed_pid=None):
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    try:
        pids = {int(value.strip()) for value in query.stdout.splitlines() if value.strip()}
    except ValueError:
        pids = {-1}
    if query.returncode or pids - {allowed_pid}:
        raise RuntimeError("PERFORMANCE_UNVERIFIED: GPU ownership unavailable or foreign process detected: " + query.stdout + query.stderr)

try:
    for round_number in range(1, 8):
        order = ("cccl", "rccl") if round_number % 2 else ("rccl", "cccl")
        for implementation in order:
            require_exclusive()
            path = results / f"round-{round_number}-{implementation}.txt"
            with path.open("w") as output, path.with_suffix(".stderr").open("w") as errors:
                process = subprocess.Popen([commands[implementation]], stdout=output, stderr=errors)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=0.25)
                            break
                        except subprocess.TimeoutExpired:
                            require_exclusive(process.pid)
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            require_exclusive()
            if code:
                raise SystemExit(f"{implementation} round {round_number} failed ({code}); see {path} and its .stderr")
            print(f"round={round_number} execution_order={','.join(order)} implementation={implementation} raw={path}", flush=True)
except RuntimeError as error:
    print(error, file=sys.stderr)
    raise SystemExit(77)

expected = set(itertools.product(
    ("lower_bound", "upper_bound", "equal_range"),
    ("small", "large"), ("uniform", "duplicates", "misses"),
    ("ascending", "descending"),
))
records = {}
for round_number in range(1, 8):
    for implementation in commands:
        cases = {}
        for line in (results / f"round-{round_number}-{implementation}.txt").read_text().splitlines():
            if not line.startswith("implementation="):
                continue
            fields = dict(word.split("=", 1) for word in line.split())
            key = tuple(fields[name] for name in ("algorithm", "size", "dataset", "order"))
            if key in cases:
                raise SystemExit(f"duplicate case: {round_number} {implementation} {key}")
            size = fields["size"]
            contract = dict(implementation=implementation, key="u64", index="u64", seed="19",
                            warmups="10", samples="31", stream="default", resets="none",
                            haystack="1024" if size == "small" else "1048576",
                            needles="256" if size == "small" else "262144")
            if any(fields.get(name) != value for name, value in contract.items()):
                raise SystemExit(f"equal-work metadata mismatch: {fields}")
            samples = [float(value) for value in fields["sample_ms"].split(",")]
            median = float(fields["median_ms"])
            if len(samples) != 31 or any(not math.isfinite(value) or value <= 0 for value in samples):
                raise SystemExit(f"invalid timing samples: {fields}")
            if not math.isclose(statistics.median(samples), median, abs_tol=1e-9):
                raise SystemExit(f"reported median differs from samples: {fields}")
            cases[key] = median
        if set(cases) != expected:
            raise SystemExit(f"missing/unexpected cases: round={round_number} implementation={implementation} {set(cases) ^ expected}")
        records[round_number, implementation] = cases

failed = []
with (results / "paired-summary.txt").open("w") as summary:
    for key in sorted(expected):
        ratios = [records[r, "rccl"][key] / records[r, "cccl"][key] for r in range(1, 8)]
        ratio = statistics.median(ratios)
        passed = ratio <= 1.001
        if not passed:
            failed.append(key)
        line = (f"algorithm={key[0]} size={key[1]} dataset={key[2]} order={key[3]} "
                f"paired_median_ratio={ratio:.9f} ratio_min={min(ratios):.9f} ratio_max={max(ratios):.9f} "
                f"paired_ratios={','.join(f'{value:.9f}' for value in ratios)} gate={'PASS' if passed else 'PERFORMANCE_GAP'}")
        print(line)
        print(line, file=summary)
if failed:
    print(f"PERFORMANCE_GAP: {len(failed)}/{len(expected)} cases exceed 1.001; no aggregate parity claim. "
          "Retain this evidence and inspect generated PTX/SASS, registers, local memory/spills, launch counts, "
          "and temporary bytes before optimization; production code, not benchmark semantics, must change.", file=sys.stderr)
    raise SystemExit(1)
print("Pinned equal-work sorted-search paired-round performance gate: PASS (u64 cases only, current GPU/architecture only)")
PY
