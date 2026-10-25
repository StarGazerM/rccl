#!/usr/bin/env sh
# Source this file before invoking cargo-oxide. Explicit caller settings win;
# otherwise prefer this development host's HPC SDK and then CUDA's conventional
# system installation. Architecture selection remains a cargo-oxide input.
export RUSTC_BOOTSTRAP=1
if [ -z "${CUDA_HOME:-}" ]; then
  for rccl_cuda_root in \
    /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9 \
    /usr/local/cuda
  do
    if [ -d "$rccl_cuda_root" ]; then
      CUDA_HOME="$rccl_cuda_root"
      break
    fi
  done
fi

if [ -z "${CUDA_HOME:-}" ]; then
  printf '%s\n' 'CUDA toolkit not found; set CUDA_HOME before sourcing scripts/oxide-env.sh' >&2
  return 1 2>/dev/null || exit 1
fi

export CUDA_HOME
export CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-$CUDA_HOME}"
export LIBNVVM_PATH="${LIBNVVM_PATH:-$CUDA_TOOLKIT_PATH/nvvm/lib64/libnvvm.so}"
export CUDA_OXIDE_LIBDEVICE="${CUDA_OXIDE_LIBDEVICE:-$CUDA_TOOLKIT_PATH/nvvm/libdevice/libdevice.10.bc}"

if [ -z "${LIBCLANG_PATH:-}" ]; then
  for rccl_clang_root in \
    /lib/x86_64-linux-gnu \
    /usr/lib/llvm-20/lib \
    /usr/lib/llvm-19/lib \
    /usr/lib/llvm-18/lib
  do
    if ls "$rccl_clang_root"/libclang.so* >/dev/null 2>&1; then
      LIBCLANG_PATH="$rccl_clang_root"
      break
    fi
  done
fi

if [ -n "${LIBCLANG_PATH:-}" ]; then
  export LIBCLANG_PATH
fi

unset rccl_cuda_root rccl_clang_root
