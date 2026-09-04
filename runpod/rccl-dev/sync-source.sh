#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'usage: %s DESTINATION [SOURCE_ROOT]\n' "$0" >&2
  printf 'example: %s root@203.0.113.10:/workspace/rccl\n' "$0" >&2
  exit 64
fi

destination="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_root="${2:-$(cd -- "$script_dir/../.." && pwd)}"

[[ -d "$source_root" ]] || {
  printf 'source root does not exist: %s\n' "$source_root" >&2
  exit 66
}
command -v rsync >/dev/null 2>&1 || {
  printf 'rsync is required\n' >&2
  exit 69
}

# Intentionally no --delete: a source refresh must not erase remote benchmark
# results. The image already owns toolchains and pinned reference checkouts.
exec rsync \
  --archive \
  --compress \
  --human-readable \
  --info=stats2,progress2 \
  --prune-empty-dirs \
  --protect-args \
  --exclude='.git/' \
  --exclude='**/.git/***' \
  --exclude='target/' \
  --exclude='**/target/***' \
  --exclude='.oxide-artifacts/' \
  --exclude='**/.oxide-artifacts/***' \
  --exclude='*.ll' \
  --exclude='*.ptx' \
  --exclude='*.cubin' \
  --exclude='*.fatbin' \
  --exclude='*.ltoir' \
  --exclude='*.sass' \
  --exclude='*.nsys-rep' \
  --exclude='*.ncu-rep' \
  --exclude='*.qdrep' \
  --exclude='*.nvvp' \
  --exclude='*.profraw' \
  --exclude='*.profdata' \
  --exclude='nsys*.sqlite' \
  --exclude='ncu*.csv' \
  --exclude='runpod-watchdog*.log' \
  --exclude='.cargo/credentials' \
  --exclude='.cargo/credentials.toml' \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='.git-credentials' \
  --exclude='.netrc' \
  --exclude='.ssh/' \
  "$source_root/" "$destination"
