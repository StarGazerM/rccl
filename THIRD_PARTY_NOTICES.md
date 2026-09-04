# Third-party notices

This repository contains Rust source and a paused local checkout of selected
cuda-oxide crates. It does not redistribute the CUDA SDK, NVIDIA HPC SDK,
CUDA driver, NVVM, libdevice, or generated device binaries; those components
remain subject to their respective NVIDIA license terms and must be installed
separately by users.

## cuda-oxide

The vendored `cuda-core`, `cuda-device`, `cuda-host`, and `cuda-macros`
packages identify themselves as Apache-2.0 in their local `Cargo.toml` files.
They are from NVlabs/cuda-oxide at revision
`97f8b2b7882f0c15ad9ce9b53abed5553920caa8`:

<https://github.com/NVlabs/cuda-oxide>

The repository's local vendor copy is a build/reproducibility candidate, not a
new license grant. The upstream license and notices must be retained when the
vendor copy is redistributed.

## Rust dependencies

The project does not vendor its crates.io dependencies. Their license and
notice text is supplied by each upstream package and is resolved through the
Cargo lockfile. Before a crates.io release, run the release dependency-license
inventory against the final lockfile and include any required notices. No
additional license terms are asserted here for transitive dependencies.

## Project license

RCCL source is offered under the choice of Apache-2.0 or MIT; see
`LICENSE-APACHE` and `LICENSE-MIT`.
