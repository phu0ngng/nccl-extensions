#!/usr/bin/env bash
# Reuse finished NCCL output across pipelines. Sources are always from checkout.
set -euo pipefail
BUILD_JOBS="${KUBERNETES_CPU_REQUEST:-$(nproc)}"

cache_dir=.ci/nccl-build-cache
build_dir=third_party/nccl/build

# GitLab restores caches before script execution, so it cannot use a key computed
# here. Keep one output per CI recipe and validate all runtime inputs on restore.
# A different NCCL pin or variable override replaces that output on success.
fingerprint=$(
  set -e
  {
    git -C third_party/nccl rev-parse HEAD
    printf '%s\n' "${CI_JOB_IMAGE:?}" "${NVCC_GENCODE:?}"
    uname -m
    ${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc} --version
    ${CXX:-g++} --version
    sha256sum .gitlab/ci/build-nccl.sh .gitlab/ci/nccl-build.yml
    # Include optional Make overrides and NVCC environment flags.
    for name in CUDA_HOME CUDA_LIB CUDA_INC NVCC CXX CC CXXSTD CXXFLAGS \
      CPPFLAGS LDFLAGS NVCCFLAGS NVCUFLAGS NVCC_APPEND_FLAGS NVCC_PREPEND_FLAGS \
      CUDARTLIB DEBUG TRACE VERBOSE \
      KEEP NVTX ASAN UBSAN TSAN WERROR PROFAPI RDMA_CORE NET_PROFILER \
      MLX5DV MAX_EXT_NET_PLUGINS EMIT_LLVM_IR DOCA_HOME \
      NCCL_ONLY_FUNCS NCCL_ONLY_TYPES NCCL_ONLY_ALGOS \
      MAKEFLAGS MAKEOVERRIDES; do
      printf '%s=%s\n' "${name}" "${!name-}"
    done
  } | sha256sum | cut -d ' ' -f 1
)

if [[ -f "${cache_dir}/fingerprint" ]] &&
   [[ "$(cat "${cache_dir}/fingerprint")" == "${fingerprint}" ]] &&
   [[ -s "${cache_dir}/build/include/nccl.h" ]] &&
   [[ -s "${cache_dir}/build/lib/libnccl.so" ]] &&
   [[ -s "${cache_dir}/build/lib/libnccl_static.a" ]]; then
  echo "NCCL build cache hit: ${fingerprint}"
  mkdir -p "${build_dir}"
  cp -a "${cache_dir}/build/." "${build_dir}/"
else
  echo "NCCL build cache miss: ${fingerprint}"
  # Do not let leftover objects or an incomplete cache affect a fresh build.
  rm -rf "${build_dir}" "${cache_dir}"
  make -C third_party/nccl -j"${BUILD_JOBS}" NVCC_GENCODE="${NVCC_GENCODE}" src.build
  test -s "${build_dir}/include/nccl.h"
  test -s "${build_dir}/lib/libnccl.so"
  test -s "${build_dir}/lib/libnccl_static.a"
  mkdir -p "${cache_dir}/build"
  # Preserve finished outputs while excluding the large intermediate objects.
  cp -a "${build_dir}/include" "${build_dir}/lib" "${cache_dir}/build/"
  if [[ -d "${build_dir}/bin" ]]; then
    cp -a "${build_dir}/bin" "${cache_dir}/build/"
  fi
  printf '%s\n' "${fingerprint}" > "${cache_dir}/fingerprint"
fi
