#!/usr/bin/env sh
# Source this file before invoking cargo-oxide on this NVIDIA HPC SDK host.
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
export CUDA_TOOLKIT_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
export LIBNVVM_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/nvvm/lib64/libnvvm.so
export CUDA_OXIDE_LIBDEVICE=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/nvvm/libdevice/libdevice.10.bc
export LIBCLANG_PATH=/lib/x86_64-linux-gnu

