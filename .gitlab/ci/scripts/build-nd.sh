#!/usr/bin/env bash
# Build nccl_nd against a prebuilt NCCL and check what a consumer links.
# Usage: build-nd.sh make|cmake
#
#   make   build and install the library with Make, compile nd_test against the
#          installed prefix only, and stage that prefix (lib/, include/, test/)
#          as the artifact the test child installs. test/ also holds
#          nd_fault_test, which compiles the library source with fault hooks.
#   cmake  configure with NCCL_ND_BUILD_TESTS=ON and NCCL_ND_BUILD_BENCH=ON,
#          build, install, and check the installed layout. Nothing built here
#          runs.
#
# Both modes assert that the library exports exactly the public ncclNd* entry
# points and does not link libcuda directly.
#
# Inputs (environment):
#   CI_PROJECT_DIR            repository checkout                              required
#   NCCL_HOME                 NCCL artifact's build dir (include/, lib/)       required
#   NVCC_GENCODE              -gencode flags (make)                            required for make
#   MODULE_ARTIFACT_DIR       staged install prefix (make, recreated)          required for make
#   CMAKE_CUDA_ARCHITECTURES  (cmake)                                          default 90
#   ND_BUILD_DIR              build tree (recreated)        default .ci/nd-<mode>-build
#   CMAKE_INSTALL_PREFIX      install prefix (cmake, recreated)  default .ci/nd-cmake-install
#   (relative directories resolve against CI_PROJECT_DIR)
#   CUDA_HOME                                                      default /usr/local/cuda
#   KUBERNETES_CPU_REQUEST    parallel build jobs                  default nproc
set -euo pipefail

mode="${1:?usage: build-nd.sh make|cmake}"
case "${mode}" in
  make|cmake) ;;
  *) echo "build-nd.sh: unknown mode '${mode}' (expected make or cmake)" >&2; exit 2 ;;
esac

fail() { echo "build-nd.sh: $*" >&2; exit 1; }

: "${CI_PROJECT_DIR:?CI_PROJECT_DIR must point at the repository checkout}"
: "${NCCL_HOME:?NCCL_HOME must point at the NCCL artifact build directory (include/ and lib/)}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
ND_BUILD_DIR="${ND_BUILD_DIR:-${CI_PROJECT_DIR}/.ci/nd-${mode}-build}"
BUILD_JOBS="${KUBERNETES_CPU_REQUEST:-$(nproc)}"
abs() { [[ "$1" == /* ]] && printf '%s' "$1" || printf '%s/%s' "${CI_PROJECT_DIR}" "$1"; }
NCCL_HOME="$(abs "${NCCL_HOME}")"
ND_BUILD_DIR="$(abs "${ND_BUILD_DIR}")"

[[ -s "${NCCL_HOME}/include/nccl_device.h" ]] || fail "NCCL_HOME=${NCCL_HOME} has no include/nccl_device.h"
compgen -G "${NCCL_HOME}/lib/libnccl.so*" >/dev/null || fail "NCCL_HOME=${NCCL_HOME} has no lib/libnccl.so*"
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || fail "CUDA_HOME=${CUDA_HOME} has no bin/nvcc"
nccl_commit_file="$(dirname "${NCCL_HOME}")/NCCL_COMMIT"
if [[ -s "${nccl_commit_file}" ]]; then
  echo ">>> [nd] NCCL commit $(cat "${nccl_commit_file}") from ${NCCL_HOME}"
fi

# The installed library must export the public API and nothing else, and must
# reach the driver only through the CUDA runtime.
check_library() {
  local lib="$1" exported
  [[ -s "${lib}" ]] || fail "no library at ${lib}"
  exported="$(nm -D --defined-only "${lib}" | awk '$2 == "T" && $3 ~ /^ncclNd/ { print $3 }' | sort | tr '\n' ' ')"
  local expected="ncclNdAllToAll ncclNdCreate ncclNdDestroy ncclNdGetVersion ncclNdGroupEnd ncclNdGroupStart "
  [[ "${exported}" == "${expected}" ]] || fail "${lib} exports '${exported}', expected '${expected}'"
  if readelf -d "${lib}" | grep -q 'NEEDED.*libcuda\.so'; then
    fail "${lib} links libcuda directly"
  fi
  echo ">>> [nd] ${lib}: exports ${exported}; no direct libcuda"
}

# Fresh trees: make timestamps and a cached configure do not notice a changed
# toolchain, gencode or NCCL.
rm -rf "${ND_BUILD_DIR}"

if [[ "${mode}" == make ]]; then
  : "${NVCC_GENCODE:?NVCC_GENCODE must hold the -gencode flags to compile}"
  : "${MODULE_ARTIFACT_DIR:?MODULE_ARTIFACT_DIR must name the directory to stage the artifact in}"
  MODULE_ARTIFACT_DIR="$(abs "${MODULE_ARTIFACT_DIR}")"
  rm -rf "${MODULE_ARTIFACT_DIR}"
  echo ">>> [nd] make lib tests bench (${NVCC_GENCODE})"
  make -C "${CI_PROJECT_DIR}/nccl_nd" -j "${BUILD_JOBS}" lib tests bench \
    BUILDDIR="${ND_BUILD_DIR}" NCCL_HOME="${NCCL_HOME}" CUDA_HOME="${CUDA_HOME}" NVCC_GENCODE="${NVCC_GENCODE}"
  echo ">>> [nd] make install: ${MODULE_ARTIFACT_DIR}"
  make -C "${CI_PROJECT_DIR}/nccl_nd" install \
    BUILDDIR="${ND_BUILD_DIR}" NCCL_HOME="${NCCL_HOME}" CUDA_HOME="${CUDA_HOME}" NVCC_GENCODE="${NVCC_GENCODE}" \
    PREFIX="${MODULE_ARTIFACT_DIR}"
  check_library "${MODULE_ARTIFACT_DIR}/lib/libnccl_nd.so"
  # Consumer smoke: the staged nd_test sees only the installed header and library.
  mkdir -p "${MODULE_ARTIFACT_DIR}/test"
  # shellcheck disable=SC2086  # NVCC_GENCODE holds several flags.
  "${CUDA_HOME}/bin/nvcc" -std=c++17 ${NVCC_GENCODE} -O2 \
    -I"${MODULE_ARTIFACT_DIR}/include" -isystem "${NCCL_HOME}/include" \
    "${CI_PROJECT_DIR}/nccl_nd/tests/nd_test.cu" \
    -L"${MODULE_ARTIFACT_DIR}/lib" -lnccl_nd -L"${NCCL_HOME}/lib" -lnccl \
    -o "${MODULE_ARTIFACT_DIR}/test/nd_test"
  install -m 755 "${ND_BUILD_DIR}/test/nccl_nd/nd_fault_test" "${MODULE_ARTIFACT_DIR}/test/"
  echo ">>> [nd] staged ${MODULE_ARTIFACT_DIR}"
  exit 0
fi

# cmake_minimum_required is 3.20; an older or missing system cmake falls back
# to the PyPI wheel through uvx, as build-cmake.sh does.
cmake_version() { { "$@" --version 2>/dev/null || true; } | sed -n '1s/^cmake version \([0-9][0-9.]*\).*/\1/p'; }
version_ok() { [[ "$1" =~ ^([0-9]+)\.([0-9]+) ]] && (( BASH_REMATCH[1] > 3 || (BASH_REMATCH[1] == 3 && BASH_REMATCH[2] >= 20) )); }
CMAKE=(cmake)
if ! version_ok "$(cmake_version "${CMAKE[@]}")"; then
  command -v uvx >/dev/null 2>&1 || fail "cmake >= 3.20 required and no uvx to fetch one"
  CMAKE=(uvx --from cmake cmake)
  version_ok "$(cmake_version "${CMAKE[@]}")" || fail "'uvx --from cmake cmake' is older than 3.20"
fi
CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90}"
CMAKE_INSTALL_PREFIX="$(abs "${CMAKE_INSTALL_PREFIX:-${CI_PROJECT_DIR}/.ci/nd-cmake-install}")"
rm -rf "${CMAKE_INSTALL_PREFIX}"
echo ">>> [nd] cmake configure (sm ${CMAKE_CUDA_ARCHITECTURES}, prefix ${CMAKE_INSTALL_PREFIX})"
"${CMAKE[@]}" -S "${CI_PROJECT_DIR}/nccl_nd" -B "${ND_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
  -DNCCL_HOME="${NCCL_HOME}" \
  -DNCCL_ND_BUILD_TESTS=ON \
  -DNCCL_ND_BUILD_BENCH=ON \
  -DCMAKE_INSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}" \
  -DCMAKE_INSTALL_LIBDIR=lib
"${CMAKE[@]}" --build "${ND_BUILD_DIR}" --parallel "${BUILD_JOBS}"
for built in test/nccl_nd/nd_test test/nccl_nd/nd_fault_test bench/nccl_nd/qkv_bench; do
  [[ -x "${ND_BUILD_DIR}/${built}" ]] || fail "cmake build left no executable ${built}"
done
"${CMAKE[@]}" --install "${ND_BUILD_DIR}"
[[ -s "${CMAKE_INSTALL_PREFIX}/include/nccl_nd.h" ]] || fail "cmake --install left no include/nccl_nd.h"
check_library "${CMAKE_INSTALL_PREFIX}/lib/libnccl_nd.so"
echo ">>> [nd] cmake path OK"
