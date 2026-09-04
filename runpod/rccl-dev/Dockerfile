# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=runpod/pytorch:1.1.0-cu1290-torch260-ubuntu2404@sha256:c23076b7331b4f5b4bb2978740358a574f602473e77706c9f1747cd6b75f8485
FROM ${BASE_IMAGE}

ARG RUSTUP_VERSION=1.28.2
ARG RUSTUP_SHA256=20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c
ARG RUST_TOOLCHAIN=nightly-2026-08-28
ARG CUDA_OXIDE_REVISION=97f8b2b7882f0c15ad9ce9b53abed5553920caa8
ARG CUDA_OXIDE_PATCH_SHA256=9d92f1e838614af92514b062fc3f0540bf22dee6dcbe1596cfd045d6341f163f
ARG CCCL_REVISION=571f2fc3bc53cd710e306ad43d58995c1fe4219f
ARG CUCO_REVISION=1d538b6befc286ab450d9e27d7ffbc4f1415c641

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_TOOLKIT_PATH=/usr/local/cuda \
    LIBNVVM_PATH=/usr/local/cuda/nvvm/lib64/libnvvm.so \
    CUDA_OXIDE_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc \
    LIBCLANG_PATH=/usr/lib/llvm-18/lib \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    RUSTUP_TOOLCHAIN=nightly-2026-08-28 \
    CUDA_OXIDE_ROOT=/opt/rccl/toolchains/cuda-oxide \
    CCCL_ROOT=/opt/rccl/sources/cccl \
    CUCO_ROOT=/opt/rccl/sources/cuCollections \
    PATH=/opt/cargo/bin:/usr/local/cuda/bin:/usr/local/cuda/compute-sanitizer:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

LABEL org.opencontainers.image.title="RCCL Runpod development and benchmark toolchain" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.base.name="runpod/pytorch:1.1.0-cu1290-torch260-ubuntu2404@sha256:c23076b7331b4f5b4bb2978740358a574f602473e77706c9f1747cd6b75f8485" \
      io.rccl.cuda.version="12.9" \
      io.rccl.rust.toolchain="nightly-2026-08-28" \
      io.rccl.cuda-oxide.revision="97f8b2b7882f0c15ad9ce9b53abed5553920caa8" \
      io.rccl.cuda-oxide.patch.sha256="9d92f1e838614af92514b062fc3f0540bf22dee6dcbe1596cfd045d6341f163f" \
      io.rccl.cccl.revision="571f2fc3bc53cd710e306ad43d58995c1fe4219f" \
      io.rccl.cucollections.revision="1d538b6befc286ab450d9e27d7ffbc4f1415c641" \
      io.rccl.runpod.startup="inherited:/opt/nvidia/nvidia_entrypoint.sh -> /start.sh"

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

RUN apt-get update \
 && apt-get install --yes --no-install-recommends \
      build-essential \
      ca-certificates \
      clang \
      clang-tools \
      cmake \
      curl \
      git \
      libclang-dev \
      libssl-dev \
      ninja-build \
      pkg-config \
      python3 \
      python3-pip \
      python3-venv \
      rsync \
      sqlite3 \
      zstd \
 && rm -rf /var/lib/apt/lists/*

COPY patches/cuda-oxide-read-only-i32x4.patch /opt/rccl/patches/cuda-oxide-read-only-i32x4.patch

RUN printf '%s  %s\n' "${CUDA_OXIDE_PATCH_SHA256}" /opt/rccl/patches/cuda-oxide-read-only-i32x4.patch | sha256sum --check - \
 && curl --fail --location --proto '=https' --tlsv1.2 \
      "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" \
      --output /tmp/rustup-init \
 && printf '%s  %s\n' "${RUSTUP_SHA256}" /tmp/rustup-init | sha256sum --check - \
 && chmod 0755 /tmp/rustup-init \
 && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain none \
 && rm /tmp/rustup-init \
 && rustup toolchain install "${RUST_TOOLCHAIN}" --profile minimal \
      --component rust-src \
      --component rustc-dev \
      --component rust-analyzer \
      --component clippy \
      --component llvm-tools-preview \
      --component rustfmt \
      --target x86_64-unknown-linux-gnu \
      --target nvptx64-nvidia-cuda \
 && rustup default "${RUST_TOOLCHAIN}"

RUN mkdir -p /opt/rccl/toolchains /opt/rccl/sources \
 && git init --quiet "${CUDA_OXIDE_ROOT}" \
 && git -C "${CUDA_OXIDE_ROOT}" remote add origin https://github.com/NVlabs/cuda-oxide.git \
 && git -C "${CUDA_OXIDE_ROOT}" fetch --quiet --depth 1 origin "${CUDA_OXIDE_REVISION}" \
 && git -C "${CUDA_OXIDE_ROOT}" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD \
 && test "$(git -C "${CUDA_OXIDE_ROOT}" rev-parse HEAD)" = "${CUDA_OXIDE_REVISION}" \
 && git -C "${CUDA_OXIDE_ROOT}" apply --index /opt/rccl/patches/cuda-oxide-read-only-i32x4.patch \
 && git -C "${CUDA_OXIDE_ROOT}" diff HEAD --check \
 && cargo +"${RUST_TOOLCHAIN}" install --path "${CUDA_OXIDE_ROOT}/crates/cargo-oxide" --locked --force \
 && (cd "${CUDA_OXIDE_ROOT}" && cargo +"${RUST_TOOLCHAIN}" oxide setup) \
 && test -s "${CARGO_HOME}/cuda-oxide/librustc_codegen_cuda.so" \
 && rm -rf "${CUDA_OXIDE_ROOT}/target" \
 && git init --quiet "${CCCL_ROOT}" \
 && git -C "${CCCL_ROOT}" remote add origin https://github.com/NVIDIA/cccl.git \
 && git -C "${CCCL_ROOT}" fetch --quiet --depth 1 origin "${CCCL_REVISION}" \
 && git -C "${CCCL_ROOT}" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD \
 && test "$(git -C "${CCCL_ROOT}" rev-parse HEAD)" = "${CCCL_REVISION}" \
 && git init --quiet "${CUCO_ROOT}" \
 && git -C "${CUCO_ROOT}" remote add origin https://github.com/NVIDIA/cuCollections.git \
 && git -C "${CUCO_ROOT}" fetch --quiet --depth 1 origin "${CUCO_REVISION}" \
 && git -C "${CUCO_ROOT}" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD \
 && test "$(git -C "${CUCO_ROOT}" rev-parse HEAD)" = "${CUCO_REVISION}"

COPY runpod/rccl-dev/self-check.sh /usr/local/bin/rccl-image-self-check
RUN chmod 0755 /usr/local/bin/rccl-image-self-check \
 && /usr/local/bin/rccl-image-self-check

# Deliberately inherit the Runpod base image's ENTRYPOINT and CMD. In particular,
# /opt/nvidia/nvidia_entrypoint.sh and /start.sh remain responsible for SSH.
