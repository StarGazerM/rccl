#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
requested_revision="${RCCL_CLEAN_CLONE_REV:-HEAD}"

commit="$(git -C "$repo_root" rev-parse --verify "${requested_revision}^{commit}")" ||
  {
    printf 'clean-clone revision is not a commit: %s\n' "$requested_revision" >&2
    exit 1
  }

required_committed_paths=(
  scripts/check-clean-clone.sh
  scripts/check-oxide-contract.sh
  scripts/test-oxide-contract-negative.sh
)
missing_committed_paths=()
for path in "${required_committed_paths[@]}"; do
  if ! git -C "$repo_root" cat-file -e "$commit:$path" 2>/dev/null; then
    missing_committed_paths+=("$path")
  fi
done
if ((${#missing_committed_paths[@]} != 0)); then
  printf 'clean-clone gate cannot validate commit %s because required gate files are not committed:\n' \
    "$commit" >&2
  printf '  %s\n' "${missing_committed_paths[@]}" >&2
  printf '%s\n' \
    'Working-tree and untracked files are intentionally not injected into the committed-tree check.' >&2
  exit 2
fi

archive_dir="$(mktemp -d)"
workdir="$archive_dir/repo"
trap 'rm -rf -- "$archive_dir"' EXIT

printf 'cloning committed tree %s into temporary checkout\n' "$commit"
git clone --quiet --no-local --no-hardlinks "$repo_root" "$workdir"
git -C "$workdir" checkout --quiet --detach "$commit"

if [[ -n "$(git -C "$workdir" status --porcelain --untracked-files=all)" ]]; then
  printf 'temporary checkout is unexpectedly dirty: %s\n' "$workdir" >&2
  exit 1
fi

printf '%s\n' 'caller working-tree changes were excluded; no files are injected into the clone'
printf '%s\n' 'running pinned cuda-oxide contract check in committed clone'
"$workdir/scripts/check-oxide-contract.sh"

printf '%s\n' 'running deterministic negative contract fixtures in committed clone'
"$workdir/scripts/test-oxide-contract-negative.sh"

printf '%s\n' 'running cargo metadata --locked --no-deps in committed clone'
cargo metadata --manifest-path "$workdir/Cargo.toml" --locked --no-deps --format-version=1 >/dev/null

printf '%s\n' 'running cargo check --workspace --locked --all-targets in committed clone'
cargo check --manifest-path "$workdir/Cargo.toml" --workspace --locked --all-targets

printf 'clean committed checkout %s: PASS\n' "$commit"
