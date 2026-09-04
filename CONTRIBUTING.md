# Contributing

RCCL is pre-1.0 and focused on small, evidence-backed Rust-native CUDA
capability slices. Read `README.md` and the capability ledgers before making a
change. Do not claim full CCCL parity for a partial or unverified surface.

## Changes

- Keep public APIs Rust-native; do not add C++ iterator/facade compatibility
  layers or exported operation macros.
- Preserve the pinned nightly, cuda-oxide revision, and canonical patch.
- Add or update a focused executable probe for a new observable contract.
- Keep CUDA work stream- and allocator-safe; document synchronization and
  ownership assumptions.
- Do not commit `target/`, PTX/cubins, logs, profiling databases, credentials,
  or machine-specific paths.

## Verification

Run `cargo fmt --all`, the relevant workspace checks/tests, and the narrow
CUDA example or script for the changed contract when hardware is available.
For host-only changes, report exactly which checks were run. GPU execution is
not required by CI and must not be implied by a green CI result.

## Pull requests

Describe the contract, implementation boundary, evidence, and remaining gaps.
Keep unrelated formatting or generated output out of the change. Contributions
are accepted under the repository's Apache-2.0 OR MIT license choice.
