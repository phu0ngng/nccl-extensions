#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
INSTALL="${CFT_INSTALL_DIR:-${ROOT}/cft_install}"
: "${CUDA_HOME:?CUDA 13.4 toolkit required}"
export MPI_HOME="${MPI_HOME:-/usr/local}"
export PATH="${CUDA_HOME}/bin:${MPI_HOME}/bin:${PATH}"
export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${INSTALL}/lib:${CUDA_HOME}/lib64:${MPI_HOME}/lib:${LD_LIBRARY_PATH:-}"
GENCODE='-gencode=arch=compute_100,code=sm_100'
NCCL_SOURCE="${NCCL_CI_SOURCE:-${ROOT}/third_party/nccl}"
GTEST_SOURCE="${GTEST_CI_SOURCE:-${ROOT}/third_party/googletest/googletest}"
# The parent's "build:nccl: [arm, cu13]" artifact (same image and gencode); unset builds NCCL here.
NCCL_PREBUILT_DIR="${NCCL_PREBUILT_DIR:-}"

BUILD_JOBS="${KUBERNETES_CPU_REQUEST:-$(nproc)}"
echo "build host: $(uname -m), parallel jobs=${BUILD_JOBS}"
mkdir -p "${INSTALL}"
nvcc --version
if [[ -n "${NCCL_PREBUILT_DIR}" ]]; then
  mkdir -p "${INSTALL}/include" "${INSTALL}/lib"
  cp -a "${NCCL_PREBUILT_DIR}/include/." "${INSTALL}/include/"
  cp -a "${NCCL_PREBUILT_DIR}/lib/." "${INSTALL}/lib/"
  if [[ -d "${NCCL_PREBUILT_DIR}/bin" ]]; then
    mkdir -p "${INSTALL}/bin"
    cp -a "${NCCL_PREBUILT_DIR}/bin/." "${INSTALL}/bin/"
  fi
  test -s "${INSTALL}/include/nccl.h"
  test -e "${INSTALL}/lib/libnccl.so"
  echo "NCCL restored from prebuilt directory ${NCCL_PREBUILT_DIR}"
else
  make -C "${NCCL_SOURCE}" -j"${BUILD_JOBS}" CUDA_HOME="${CUDA_HOME}" \
    BUILDDIR="${INSTALL}" NVCC_GENCODE="${GENCODE}" src.build
fi
# Correctness jobs do not need a CUPTI dependency on the older driver.
EP_BUILD=(make -C "${ROOT}/nccl_ep" -j"${BUILD_JOBS}" CUDA_HOME="${CUDA_HOME}"
  NCCL_HOME="${INSTALL}" NCCL_SUBMODULE_HOME="${NCCL_SOURCE}"
  BUILDDIR="${INSTALL}" MPI=1 MPI_HOME="${MPI_HOME}"
  NVCC_GENCODE="${GENCODE}" CUPTI_CFLAGS= CUPTI_LDFLAGS=)
# Finish shared-library symlinks before -lnccl_ep can select the static archive.
"${EP_BUILD[@]}" lib
"${EP_BUILD[@]}" ep_test ep_bench

"${MPI_HOME}/bin/mpicxx" -std=c++17 \
  -I"${CUDA_HOME}/include" -I"${INSTALL}/include" -I"${ROOT}/nccl_ep" \
  "${ROOT}/nccl_ep/ci/test_cft_compatibility.cc" \
  -L"${INSTALL}/lib" -lnccl_ep -Wl,--no-as-needed -lnccl -Wl,--as-needed \
  -L"${CUDA_HOME}/lib64" -lcudart_static -L"${CUDA_HOME}/lib64/stubs" -lcuda \
  -ldl -lrt -pthread -o "${INSTALL}/test/nccl_ep/test_cft_compatibility"

make -C "${ROOT}/nccl_ep/tests" -j"${BUILD_JOBS}" NVCC="${CUDA_HOME}/bin/nvcc" \
  NCCL_HOME="${INSTALL}" BUILDDIR="${INSTALL}" GTEST_DIR="${GTEST_SOURCE}" \
  NVCC_GENCODE="${GENCODE}" "${INSTALL}/test/nccl_ep/test_cft_driver_api" \
  "${INSTALL}/test/nccl_ep/test_ht_chunk_config"
NCCL_EP_BUILDDIR="${INSTALL}" TEST_SUITE=test_ht_chunk_config \
  bash "${ROOT}/nccl_ep/tests/run_tests.sh"

# GPU-free builders still need a loadable driver library for the mocked tests.
# Scope the compatibility library to this process; real GPU tests use host drivers.
mock_library_path="${LD_LIBRARY_PATH}"
if [[ -n "${CFT_MOCK_DRIVER_LIB_DIR:-}" ]]; then
  test -r "${CFT_MOCK_DRIVER_LIB_DIR}/libcuda.so.1"
  mock_library_path="${CFT_MOCK_DRIVER_LIB_DIR}:${mock_library_path}"
fi
LD_LIBRARY_PATH="${mock_library_path}" \
NCCL_EP_BUILDDIR="${INSTALL}" TEST_SUITE=test_cft_driver_api \
  bash "${ROOT}/nccl_ep/tests/run_tests.sh"

# A prebuilt NCCL is identified by the artifact's NCCL_COMMIT beside its build directory.
nccl_commit() {
  local commit_file=""
  if [[ -n "${NCCL_PREBUILT_DIR}" ]]; then
    commit_file="$(dirname "${NCCL_PREBUILT_DIR}")/NCCL_COMMIT"
  fi
  if [[ -n "${commit_file}" && -s "${commit_file}" ]]; then
    cat "${commit_file}"
  else
    git -C "${NCCL_SOURCE}" rev-parse HEAD
  fi
}
{
  git -C "${ROOT}" rev-parse HEAD
  nccl_commit
  nvcc --version
} > "${INSTALL}/provenance.txt"
(
  cd "${INSTALL}"
  sha256sum lib/libnccl.so.2 lib/libnccl_ep.so.0 \
    test/nccl_ep/ep_bench test/nccl_ep/ep_test test/nccl_ep/test_cft_compatibility \
    > SHA256SUMS
)
