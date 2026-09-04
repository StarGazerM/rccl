# Third-party notices

This repository depends on RCCL's pinned fork of selected cuda-oxide crates. It
does not redistribute a local cuda-oxide source copy or the CUDA SDK,
CUDA driver, NVVM, libdevice, or generated device binaries; those components
remain subject to their respective NVIDIA license terms and must be installed
separately by users.

## cuda-oxide

The `cuda-core`, `cuda-device`, `cuda-host`, and `cuda-macros` dependencies
identify themselves as Apache-2.0. RCCL pins its compatibility branch at
`5f7c34fddadcf250f2e45d0cffe83a3d27f2bea3`:

<https://github.com/StarGazerM/cuda-oxide>

The fork derives from NVlabs/cuda-oxide and retains upstream license notices:

<https://github.com/NVlabs/cuda-oxide>

## Rust dependencies

The project does not vendor its crates.io dependencies. Their license and
notice text is supplied by each upstream package and is resolved through the
Cargo lockfile. Before a crates.io release, run the release dependency-license
inventory against the final lockfile and include any required notices. No
additional license terms are asserted here for transitive dependencies.

## Project license

RCCL source is offered under the choice of Apache-2.0 or MIT; see
`LICENSE-APACHE` and `LICENSE-MIT`.
