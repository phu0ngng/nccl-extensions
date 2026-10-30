#!/usr/bin/env bash
# Build nccl_m2n through its CMake path (library, tests and benches, hidden
# visibility) against a prebuilt NCCL and install it: a public entry point
# missing NCCL_M2N_API fails the link here instead of at a consumer. Nothing
# built here runs. Usage: build-cmake.sh m2n
#
# Inputs (environment):
#   CI_PROJECT_DIR            repository checkout                              required
#   NCCL_HOME                 NCCL artifact's build dir (include/, lib/)       required
#   (relative directories resolve against CI_PROJECT_DIR)
#   CUDA_HOME                 nvcc at ${CUDA_HOME}/bin/nvcc       default /usr/local/cuda
#   MPI_HOME                  FindMPI hint                        default /usr/local
#   GTEST_DIR                 googletest source root              default third_party/googletest/googletest
#   CMAKE_CUDA_ARCHITECTURES                                      default 90
#   CMAKE_BUILD_DIR           CMake binary dir (recreated)        default .ci/m2n-cmake-build
#   CMAKE_INSTALL_PREFIX      install prefix (recreated)          default .ci/m2n-cmake-install
#   KUBERNETES_CPU_REQUEST    cmake --build --parallel            default nproc
set -euo pipefail

module="${1:?usage: build-cmake.sh m2n}"
case "${module}" in
  m2n) ;;
  ep)
    echo "build-cmake.sh: nccl_ep/CMakeLists.txt is not standalone (no project())" >&2
    exit 2 ;;
  *) echo "build-cmake.sh: unknown module '${module}' (expected m2n)" >&2; exit 2 ;;
esac

fail() { echo "build-cmake.sh: $*" >&2; exit 1; }

: "${CI_PROJECT_DIR:?CI_PROJECT_DIR must point at the repository checkout}"
: "${NCCL_HOME:?NCCL_HOME must point at the NCCL artifact build directory (include/ and lib/)}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
MPI_HOME="${MPI_HOME:-/usr/local}"
GTEST_DIR="${GTEST_DIR:-${CI_PROJECT_DIR}/third_party/googletest/googletest}"
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90}"
CMAKE_BUILD_DIR="${CMAKE_BUILD_DIR:-${CI_PROJECT_DIR}/.ci/m2n-cmake-build}"
CMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX:-${CI_PROJECT_DIR}/.ci/m2n-cmake-install}"
BUILD_JOBS="${KUBERNETES_CPU_REQUEST:-$(nproc)}"
# CMake resolves relative -D paths against its own cwd; hand it absolute ones.
for var in NCCL_HOME GTEST_DIR CMAKE_BUILD_DIR CMAKE_INSTALL_PREFIX; do
  [[ "${!var}" == /* ]] || printf -v "${var}" '%s/%s' "${CI_PROJECT_DIR}" "${!var}"
done

# Preflight: inputs CMake reports late or, for googletest, not at all
# (tests/CMakeLists.txt skips the gtest binaries with a STATUS message).
for header in nccl.h nccl_device.h; do
  [[ -s "${NCCL_HOME}/include/${header}" ]] || fail "NCCL_HOME=${NCCL_HOME} has no include/${header}"
done
compgen -G "${NCCL_HOME}/lib/libnccl.so*" >/dev/null || fail "NCCL_HOME=${NCCL_HOME} has no lib/libnccl.so*"
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || fail "CUDA_HOME=${CUDA_HOME} has no bin/nvcc"
[[ -s "${GTEST_DIR}/include/gtest/gtest.h" ]] ||
  fail "GTEST_DIR=${GTEST_DIR} has no include/gtest/gtest.h; nccl_m2n/tests would skip basic_api_test_* silently (git submodule update --init third_party/googletest)"

# cmake_minimum_required is 3.20; an older or missing system cmake falls back
# to the PyPI wheel through uvx. The pipe never fails so set -e stays quiet.
cmake_version() { { "$@" --version 2>/dev/null || true; } | sed -n '1s/^cmake version \([0-9][0-9.]*\).*/\1/p'; }
version_ok() { [[ "$1" =~ ^([0-9]+)\.([0-9]+) ]] && (( BASH_REMATCH[1] > 3 || (BASH_REMATCH[1] == 3 && BASH_REMATCH[2] >= 20) )); }
CMAKE=(cmake)
cmake_ver="$(cmake_version "${CMAKE[@]}")"
if ! version_ok "${cmake_ver}"; then
  command -v uvx >/dev/null 2>&1 || fail "cmake >= 3.20 required, found '${cmake_ver:-none}' and no uvx to fetch one"
  CMAKE=(uvx --from cmake cmake)
  cmake_ver="$(cmake_version "${CMAKE[@]}")"
  version_ok "${cmake_ver}" || fail "'uvx --from cmake cmake' gave version '${cmake_ver:-none}', need >= 3.20"
fi
cmake_path="${CMAKE[*]}"
[[ "${CMAKE[0]}" != cmake ]] || cmake_path="$(command -v cmake)"
echo ">>> [m2n] preflight: cmake ${cmake_ver} (${cmake_path}); NCCL_HOME=${NCCL_HOME} GTEST_DIR=${GTEST_DIR} CUDA_HOME=${CUDA_HOME}"

# Fresh trees: a cached configure or a stale install would mask changed inputs.
rm -rf "${CMAKE_BUILD_DIR}" "${CMAKE_INSTALL_PREFIX}"
echo ">>> [m2n] configure: -S nccl_m2n -B ${CMAKE_BUILD_DIR} (sm ${CMAKE_CUDA_ARCHITECTURES}, MPI_HOME=${MPI_HOME}, prefix ${CMAKE_INSTALL_PREFIX})"
"${CMAKE[@]}" -S "${CI_PROJECT_DIR}/nccl_m2n" -B "${CMAKE_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
  -DNCCL_HOME="${NCCL_HOME}" \
  -DNCCL_M2N_BUILD_TESTS=ON \
  -DNCCL_M2N_BUILD_BENCH=ON \
  -DGTEST_DIR="${GTEST_DIR}" \
  -DMPI_HOME="${MPI_HOME}" \
  -DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}" \
  -DCMAKE_INSTALL_LIBDIR=lib

echo ">>> [m2n] build: --parallel ${BUILD_JOBS}"
"${CMAKE[@]}" --build "${CMAKE_BUILD_DIR}" --parallel "${BUILD_JOBS}"
# Every test and bench target tests/ and benchmarks/ define; a missing one
# means FindMPI or the gtest lookup skipped it without failing the configure.
M2N_BINARIES=(basic_api_test_local basic_api_test_mpi multi_dim_geometry_test multi_dim_test_mpi
  reshard_bench reshard_batch_bench_user_window reshard_model_bench)
for binary in "${M2N_BINARIES[@]}"; do
  [[ -x "${CMAKE_BUILD_DIR}/bin/${binary}" ]] || fail "cmake build left no executable bin/${binary}"
done
# pipe_commb_resources_test sets no RUNTIME_OUTPUT_DIRECTORY, so it stays in tests/.
[[ -x "${CMAKE_BUILD_DIR}/tests/pipe_commb_resources_test" ]] ||
  fail "cmake build left no executable tests/pipe_commb_resources_test"

echo ">>> [m2n] install: ${CMAKE_INSTALL_PREFIX}"
"${CMAKE[@]}" --install "${CMAKE_BUILD_DIR}"
M2N_INSTALLED=(lib/libnccl_m2n.so lib/libnccl_m2n.a include/nccl_m2n.h
  lib/cmake/NCCLM2N/NCCLM2NConfig.cmake lib/cmake/NCCLM2N/NCCLM2NConfigVersion.cmake
  lib/cmake/NCCLM2N/NCCLM2NTargets.cmake lib/cmake/NCCLM2N/NCCLM2NTargets-release.cmake)
for file in "${M2N_INSTALLED[@]}"; do
  [[ -s "${CMAKE_INSTALL_PREFIX}/${file}" ]] || fail "cmake --install left no ${file} under ${CMAKE_INSTALL_PREFIX}"
done
echo ">>> [m2n] cmake path OK"
