# RCCL Runpod development image

This image keeps the expensive CUDA/Rust compiler setup in immutable Docker
layers. RCCL itself is never copied into the image; synchronize only source
files into `/workspace/rccl` for each pod.

## Published image

Tag: `docker.io/stargazermiao/rccl-dev:v1.0.0`

The base is
`runpod/pytorch:1.1.0-cu1290-torch260-ubuntu2404@sha256:c23076b7331b4f5b4bb2978740358a574f602473e77706c9f1747cd6b75f8485`.
It retains the base image's `/opt/nvidia/nvidia_entrypoint.sh` entrypoint and
`/start.sh` command, including Runpod SSH startup.

The image pins CUDA 12.9, Rust `nightly-2026-08-28`, cuda-oxide
`97f8b2b7882f0c15ad9ce9b53abed5553920caa8` with the repository patch
SHA-256 `9d92f1e838614af92514b062fc3f0540bf22dee6dcbe1596cfd045d6341f163f`,
CCCL `571f2fc3bc53cd710e306ad43d58995c1fe4219f`, and cuCollections
`1d538b6befc286ab450d9e27d7ffbc4f1415c641`.

## Rebuild and push

Run from the repository root. The Dockerfile-specific ignore file admits only
the Dockerfile, self-check, and canonical patch; local Cargo credentials,
Docker credentials, SSH keys, environment files, the mutable RCCL tree, and
other repository data cannot enter the build context.

```bash
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --file runpod/rccl-dev/Dockerfile \
  --tag stargazermiao/rccl-dev:v1.0.0 \
  --push \
  .
```

Never overwrite this tag. Change `org.opencontainers.image.version` and use a
new semantic version tag for every image change. Cargo authentication is not
needed: do not run `cargo login`/`cargo publish` and do not add credential
mounts or credential files to the Dockerfile.

## Create a development pod

Check live availability and choose the cheapest GPU that matches the required
architecture. For Ada/SM89, one example is the RTX 2000 Ada:

```bash
runpodctl gpu list --include-unavailable
runpodctl pod create \
  --name rccl-dev-v1-sm89 \
  --image docker.io/stargazermiao/rccl-dev:v1.0.0 \
  --gpu-id 'NVIDIA RTX 2000 Ada Generation' \
  --ports '22/tcp' \
  --container-disk-in-gb 40 \
  --wait \
  --wait-timeout 20m
```

Start a local deletion watchdog immediately after recording the returned pod
ID. This is a fallback cost guard, not a replacement for explicit cleanup:

```bash
POD_ID='<returned-id>'
(
  sleep 3600
  runpodctl pod delete "$POD_ID"
) >"/tmp/rccl-runpod-watchdog-$POD_ID.log" 2>&1 &
WATCHDOG_PID=$!
```

`runpodctl ssh info "$POD_ID"` returns the public IP, mapped port, and exact
private-key path. Use those values for SSH. On the pod, the following command
must succeed without running any installer:

```bash
rccl-image-self-check
nvidia-smi
```

## Synchronize source only

Configure the nonstandard Runpod SSH port for rsync, then run the helper from
the workstation:

```bash
export RSYNC_RSH='ssh -i /path/from/runpodctl-ssh-info -p MAPPED_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
runpod/rccl-dev/sync-source.sh root@POD_IP:/workspace/rccl/
```

The helper does not use `--delete`, so benchmark results on the pod survive a
source refresh. It excludes every `.git` and nested `target` directory,
`.oxide-artifacts`, generated LLVM/PTX/CUDA files, profiler reports/databases,
watchdog logs, and credential/secret files.

## Compile and benchmark

The image exports `CUDA_HOME`, `CUDA_TOOLKIT_PATH`, `LIBCLANG_PATH`,
`RUSTUP_TOOLCHAIN`, `CCCL_ROOT`, and `CUCO_ROOT`; no setup script or package
installation is needed after SSH login.

```bash
cd /workspace/rccl
cargo oxide build --arch sm_89 --features cuda-oxide-example -- \
  --package rccl-cub --example warp_reduce
RCCL_ARCH=sm_89 scripts/test-block-scan-performance.sh
```

For repeatable timing, use an otherwise idle exclusive GPU, warm up first, and
run the repository performance script's complete round set rather than timing
a single process launch.

## Cleanup

Delete the test pod even if verification fails, stop the local watchdog, and
confirm the API returns `not_found` for the deleted ID:

```bash
runpodctl pod delete "$POD_ID"
kill "$WATCHDOG_PID" 2>/dev/null || true
runpodctl pod get "$POD_ID"
```
