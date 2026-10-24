#!/usr/bin/env bash
# Generate LL JIT kernel sources for one source tree without a GPU.
#
# Usage: jit_enumerate.sh <source-tree> <builddir> <out-cache-dir>
#
# Generates the header-derived glue (jit_enumerate_gen.py), builds tools/jit_enumerate.cu
# against the tree's headers and its built libnccl_ep.so (for resolveDispatchKernelSpec),
# then runs it. The output has
# the JIT cache layout (<out>/<variant>/kernel.cu, metadata.json) that
# jit_variants.py compiles. Needs CUDA_HOME and NCCL_HOME. Prints the number of
# variants written and any kernel enumerator the matrix did not cover; exits
# non-zero if the glue cannot be generated or the driver does not build against
# this tree, which the caller reports as skipped.
set -euo pipefail

TREE="${1:?source tree}"
BUILD="${2:?builddir}"
OUT="${3:?out cache dir}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NCCL_HOME="${NCCL_HOME:?NCCL_HOME}"
ARCH="${REVIEW_ARCH:-sm_90}"

# The generators' argument lists and the kernel enumerators are read from this tree's headers;
# a parameter the script has no binding for stops here with a message (exit 3, reported as skipped).
mkdir -p "${OUT}"
python3 -I "${HERE}/jit_enumerate_gen.py" "${TREE}" "${OUT}/jit_enumerate_gen.inc" || exit 3

# sm_100a -> 100: the numeric target, for the CUDA-version and architecture gates in the driver.
ARCH_NUM="${ARCH#sm_}"
ARCH_NUM="${ARCH_NUM%%[a-z]*}"

BIN="${OUT}/.jit_enumerate"
"${CUDA_HOME}/bin/nvcc" -std=c++17 --expt-relaxed-constexpr -arch="${ARCH}" \
    -DREVIEW_ARCH_NUM="${ARCH_NUM}" -I"${OUT}" \
    -I"${BUILD}/include" -I"${BUILD}/include/nccl_ep" -I"${TREE}/nccl_ep" -I"${NCCL_HOME}/include" \
    "${HERE}/jit_enumerate.cu" -o "${BIN}" \
    -L"${BUILD}/lib" -lnccl_ep -L"${NCCL_HOME}/lib" -lnccl -lcuda -lpthread -ldl \
    > "${OUT}/.jit_enumerate.build.log" 2>&1 || {
    echo "jit_enumerate: driver does not build against ${TREE} (see ${OUT}/.jit_enumerate.build.log)" >&2
    exit 3
}
LD_LIBRARY_PATH="${BUILD}/lib:${NCCL_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" "${BIN}" "${OUT}"
