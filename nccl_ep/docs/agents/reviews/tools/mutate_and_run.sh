#!/usr/bin/env bash
# Apply a defect patch to a head worktree, rebuild, run one test, revert.
#
# Usage:
#   mutate_and_run.sh <head-worktree> <builddir> <mutation.patch> <test-binary-name> <nranks> [gtest args...]
#
# A test-validity finding says "if X were dropped the test would still pass".
# Write X as a patch against the head tree and run this. The result line is
#   MUTATION <name>: test PASSED  -> the finding stands (the test cannot see X)
#   MUTATION <name>: test FAILED  -> the finding is refuted (the test catches X)
#   MUTATION <name>: build FAILED -> the mutation is not a valid experiment
# The patch is reverted whatever happens. Requires NCCL_HOME for the build.
set -euo pipefail

TREE="${1:?head worktree}"
BUILDDIR="${2:?builddir}"
PATCH="${3:?mutation patch}"
TEST="${4:?test binary name}"
NRANKS="${5:?nranks}"
shift 5
NAME="$(basename "${PATCH}" .patch)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${REVIEW_JOBS:-$(nproc)}"

cd "${TREE}"
if ! git apply --check "${PATCH}"; then
    echo "MUTATION ${NAME}: patch does not apply"
    exit 2
fi
git apply "${PATCH}"
revert() { git apply -R "${PATCH}" || git checkout -- . ; }
trap revert EXIT

STATUS=0
if ! make -s -C nccl_ep -j"${JOBS}" lib BUILDDIR="${BUILDDIR}" > "${BUILDDIR}/mutation_${NAME}_build.log" 2>&1 \
   || ! make -s -C nccl_ep/tests -j"${JOBS}" BUILDDIR="${BUILDDIR}" "${BUILDDIR}/test/nccl_ep/${TEST}" >> "${BUILDDIR}/mutation_${NAME}_build.log" 2>&1; then
    echo "MUTATION ${NAME}: build FAILED (see ${BUILDDIR}/mutation_${NAME}_build.log)"
    exit 3
fi

export LD_LIBRARY_PATH="${BUILDDIR}/lib:${NCCL_HOME:-${TREE}/third_party/nccl/build}/lib:${LD_LIBRARY_PATH:-}"
export NCCL_EP_JIT_CACHE_DIR="${BUILDDIR}/jit_mutation_${NAME}"
rm -rf "${NCCL_EP_JIT_CACHE_DIR}"
if REVIEW_LOG_DIR="${BUILDDIR}/mutation_${NAME}_logs" \
   "${HERE}/run_tests_sanitized.sh" "${BUILDDIR}/test/nccl_ep/${TEST}" "${NRANKS}" none "$@" > "${BUILDDIR}/mutation_${NAME}.log" 2>&1; then
    echo "MUTATION ${NAME}: test PASSED (the test does not detect this defect; log ${BUILDDIR}/mutation_${NAME}.log)"
    STATUS=0
else
    echo "MUTATION ${NAME}: test FAILED (the test detects this defect; log ${BUILDDIR}/mutation_${NAME}.log)"
    STATUS=1
fi
exit "${STATUS}"
