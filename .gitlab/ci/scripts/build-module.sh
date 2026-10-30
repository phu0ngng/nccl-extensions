#!/usr/bin/env bash
# Build one extension module with the release recipe (build_lib.py), build its
# test binaries against that same tree, and stage the artifact the test
# children install. Usage: build-module.sh <ep|m2n>
#
# Inputs (environment):
#   CI_PROJECT_DIR       repository checkout                          required
#   NCCL_HOME            NCCL artifact's build dir (include/, lib/)   required
#   NVCC_GENCODE         -gencode flags to compile                    required
#   MODULE_ARTIFACT_DIR  where to stage the artifact (recreated)      required
#   MODULE_BUILD_DIR     make tree shared by the library and test-binary steps   default ${CI_PROJECT_DIR}/build
#   (relative directories resolve against CI_PROJECT_DIR)
#   CUDA_HOME                                                default /usr/local/cuda
#   MPI_HOME                                                 default /usr/local
#   GTEST_DIR                                                default third_party/googletest/googletest
#   KUBERNETES_CPU_REQUEST  make -j                          default nproc
set -euo pipefail

module="${1:?usage: build-module.sh <ep|m2n>}"
case "${module}" in
  ep) component=nccl_ep ;;
  m2n) component=nccl_m2n ;;
  *) echo "build-module.sh: unknown module '${module}' (expected ep or m2n)" >&2; exit 2 ;;
esac

: "${CI_PROJECT_DIR:?CI_PROJECT_DIR must point at the repository checkout}"
: "${NCCL_HOME:?NCCL_HOME must point at the NCCL artifact build directory (include/ and lib/)}"
: "${NVCC_GENCODE:?NVCC_GENCODE must hold the -gencode flags to compile}"
: "${MODULE_ARTIFACT_DIR:?MODULE_ARTIFACT_DIR must name the directory to stage the artifact in}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MPI_HOME="${MPI_HOME:-/usr/local}"
MODULE_BUILD_DIR="${MODULE_BUILD_DIR:-${CI_PROJECT_DIR}/build}"
GTEST_DIR="${GTEST_DIR:-${CI_PROJECT_DIR}/third_party/googletest/googletest}"
BUILD_JOBS="${KUBERNETES_CPU_REQUEST:-$(nproc)}"
export NCCL_HOME NVCC_GENCODE CUDA_HOME MPI_HOME

[[ -s "${NCCL_HOME}/include/nccl_device.h" ]] ||
  { echo "build-module.sh: NCCL_HOME=${NCCL_HOME} has no include/nccl_device.h" >&2; exit 1; }
compgen -G "${NCCL_HOME}/lib/libnccl.so*" >/dev/null ||
  { echo "build-module.sh: NCCL_HOME=${NCCL_HOME} has no lib/libnccl.so*" >&2; exit 1; }
nccl_commit_file="$(dirname "${NCCL_HOME}")/NCCL_COMMIT"
if [[ -s "${nccl_commit_file}" ]]; then
  echo ">>> [${module}] NCCL commit $(cat "${nccl_commit_file}") from ${NCCL_HOME}"
fi

# One canonical path for both steps: make compares target names textually.
[[ "${MODULE_BUILD_DIR}" == /* ]] || MODULE_BUILD_DIR="${CI_PROJECT_DIR}/${MODULE_BUILD_DIR}"
[[ "${MODULE_ARTIFACT_DIR}" == /* ]] || MODULE_ARTIFACT_DIR="${CI_PROJECT_DIR}/${MODULE_ARTIFACT_DIR}"
mkdir -p "${MODULE_BUILD_DIR}" "${MODULE_ARTIFACT_DIR}"
MODULE_BUILD_DIR="$(cd "${MODULE_BUILD_DIR}" && pwd -P)"
MODULE_ARTIFACT_DIR="$(cd "${MODULE_ARTIFACT_DIR}" && pwd -P)"
if [[ "${MODULE_ARTIFACT_DIR}" == "${MODULE_BUILD_DIR}" ]]; then
  echo "build-module.sh: MODULE_ARTIFACT_DIR is recreated from scratch and cannot be the build tree ${MODULE_BUILD_DIR}; set MODULE_BUILD_DIR elsewhere" >&2
  exit 1
fi

# Fresh tree per run: make's timestamps do not notice a changed toolchain, gencode
# or NCCL, so a reused MODULE_BUILD_DIR could stage stale objects.
rm -rf "${MODULE_BUILD_DIR}"
mkdir -p "${MODULE_BUILD_DIR}"

# Library step: exactly as the release builds it (no MPI, no CUPTI, static
# cudart); the make tree stays in MODULE_BUILD_DIR for the test-binary step.
echo ">>> [${module}] library: ${component} via build_lib.py into ${MODULE_BUILD_DIR} (-j${BUILD_JOBS})"
# The NCCL CI image's python3 is 3.6; uv runs the script's PEP 723 header
# (python >= 3.10 + packaging). Images without uv must provide python3 >= 3.10.
if command -v uv >/dev/null 2>&1; then
  build_lib=(uv run --script)
else
  python3 -c 'import packaging' 2>/dev/null ||
    { echo ">>> [${module}] installing the 'packaging' module build_lib.py imports"; python3 -m pip install -q packaging; }
  build_lib=(python3)
fi
MAKEFLAGS="-j${BUILD_JOBS}" "${build_lib[@]}" "${CI_PROJECT_DIR}/python/build_assets/build_lib.py" "${component}" \
  --cuda-home "${CUDA_HOME}" \
  --build-dir "${MODULE_BUILD_DIR}" \
  --output-dir "${MODULE_BUILD_DIR}/python"

# Test-binary step: link against the library the previous step left in BUILDDIR.
# Every test target depends on the library file, so make links without recompiling.
# The m2n children install exactly these binaries by name.
M2N_BINARIES=(multi_dim_geometry_test multi_dim_test_mpi basic_api_test_local
  basic_api_test_mpi reshard_bench reshard_batch_bench_user_window)
echo ">>> [${module}] test binaries: make -j${BUILD_JOBS} (MPI_HOME=${MPI_HOME}, GTEST_DIR=${GTEST_DIR})"
case "${module}" in
  ep)
    EP_MAKE=(make -C "${CI_PROJECT_DIR}/nccl_ep" -j"${BUILD_JOBS}"
      NCCL_HOME="${NCCL_HOME}" CUDA_HOME="${CUDA_HOME}" BUILDDIR="${MODULE_BUILD_DIR}"
      NVCC_GENCODE="${NVCC_GENCODE}" MPI=1 MPI_HOME="${MPI_HOME}")
    # lib is up to date after the library step; keeping it first keeps the shared-library
    # symlinks ahead of the parallel executable links.
    "${EP_MAKE[@]}" lib
    "${EP_MAKE[@]}" ep_test ep_bench af_test
    make -C "${CI_PROJECT_DIR}/nccl_ep/tests" -j"${BUILD_JOBS}" \
      NCCL_HOME="${NCCL_HOME}" NVCC="${CUDA_HOME}/bin/nvcc" BUILDDIR="${MODULE_BUILD_DIR}" \
      GTEST_DIR="${GTEST_DIR}" NVCC_GENCODE="${NVCC_GENCODE}"
    for binary in ep_test ep_bench af_test; do
      test -x "${MODULE_BUILD_DIR}/test/nccl_ep/${binary}" ||
        { echo "build-module.sh: ${component} build left no executable test/nccl_ep/${binary}" >&2; exit 1; }
    done
    ;;
  m2n)
    make -C "${CI_PROJECT_DIR}/nccl_m2n" -j"${BUILD_JOBS}" \
      NCCL_HOME="${NCCL_HOME}" CUDA_HOME="${CUDA_HOME}" BUILDDIR="${MODULE_BUILD_DIR}" \
      NVCC_GENCODE="${NVCC_GENCODE}" MPI_HOME="${MPI_HOME}" GTEST_DIR="${GTEST_DIR}" \
      lib tests bench reshard reshard_batch_user_window
    for file in lib/libnccl_m2n.so include/nccl_m2n.h; do
      test -s "${MODULE_BUILD_DIR}/${file}" ||
        { echo "build-module.sh: ${component} build left no ${file}" >&2; exit 1; }
    done
    for binary in "${M2N_BINARIES[@]}"; do
      test -x "${MODULE_BUILD_DIR}/bin/${binary}" ||
        { echo "build-module.sh: ${component} build left no executable bin/${binary}" >&2; exit 1; }
    done
    ;;
esac

# Artifact step: stage the layout the test children install.
echo ">>> [${module}] artifact: staging ${MODULE_ARTIFACT_DIR}"
rm -rf "${MODULE_ARTIFACT_DIR}"
case "${module}" in
  ep)
    mkdir -p "${MODULE_ARTIFACT_DIR}/lib" "${MODULE_ARTIFACT_DIR}/include" "${MODULE_ARTIFACT_DIR}/test/nccl_ep"
    cp -a "${MODULE_BUILD_DIR}/lib"/libnccl_ep.* "${MODULE_ARTIFACT_DIR}/lib/"
    cp -a "${MODULE_BUILD_DIR}/include/nccl_ep.h" "${MODULE_ARTIFACT_DIR}/include/"
    cp -a "${MODULE_BUILD_DIR}/include/nccl_ep" "${MODULE_ARTIFACT_DIR}/include/"
    cp -a "${MODULE_BUILD_DIR}/test/nccl_ep/ep_test" "${MODULE_BUILD_DIR}/test/nccl_ep/ep_bench" \
      "${MODULE_BUILD_DIR}/test/nccl_ep/af_test" "${MODULE_ARTIFACT_DIR}/test/nccl_ep/"
    for unit_test in "${MODULE_BUILD_DIR}/test/nccl_ep"/test_*; do
      if [[ -f "${unit_test}" && -x "${unit_test}" ]]; then
        cp -a "${unit_test}" "${MODULE_ARTIFACT_DIR}/test/nccl_ep/"
      fi
    done
    ;;
  m2n)
    mkdir -p "${MODULE_ARTIFACT_DIR}/bin" "${MODULE_ARTIFACT_DIR}/include" "${MODULE_ARTIFACT_DIR}/lib"
    cp -a "${MODULE_BUILD_DIR}/lib/libnccl_m2n.so" "${MODULE_ARTIFACT_DIR}/lib/"
    cp -a "${MODULE_BUILD_DIR}/include/nccl_m2n.h" "${MODULE_ARTIFACT_DIR}/include/"
    for binary in "${M2N_BINARIES[@]}"; do
      cp -a "${MODULE_BUILD_DIR}/bin/${binary}" "${MODULE_ARTIFACT_DIR}/bin/"
    done
    ;;
esac
echo ">>> [${module}] done: $(find "${MODULE_ARTIFACT_DIR}" -type f -o -type l | wc -l | tr -d ' ') files staged in ${MODULE_ARTIFACT_DIR}"
