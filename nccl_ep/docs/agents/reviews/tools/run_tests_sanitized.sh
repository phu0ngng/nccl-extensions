#!/usr/bin/env bash
# Run one nccl_ep gtest binary on N ranks, optionally under compute-sanitizer.
#
# Usage:
#   run_tests_sanitized.sh <test-binary> <nranks> [sanitizer-tool|none] [gtest args...]
#
# Mirrors tests/run_tests.sh: one process per rank, --rank/--nranks/--uid-file,
# GPU chosen inside the binary. Logs go to $REVIEW_LOG_DIR (default: mktemp -d);
# the rank-0 log and every sanitizer error summary are printed. Exit code is
# non-zero if any rank fails or the sanitizer reports errors.
set -euo pipefail

BIN="${1:?test binary}"
NRANKS="${2:?nranks}"
TOOL="${3:-none}"
shift 3 || shift $#
LOG_DIR="${REVIEW_LOG_DIR:-$(mktemp -d)}"
mkdir -p "${LOG_DIR}"
UID_FILE="${TMPDIR:-/tmp}/te_ep_uid_$(basename "${BIN}")_$$"
rm -f "${UID_FILE}"
trap 'rm -f "${UID_FILE}"' EXIT INT TERM

WRAP=()
if [[ "${TOOL}" != "none" ]]; then
    SAN="${COMPUTE_SANITIZER:-$(command -v compute-sanitizer || echo "${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer")}"
    # --print-limit keeps one bad kernel from producing megabytes; racecheck needs
    # shared-memory hazard reporting on, which is its default.
    WRAP=("${SAN}" --tool "${TOOL}" --print-limit 50 --error-exitcode 99)
fi

PIDS=()
for i in $(seq 0 $((NRANKS - 1))); do
    "${WRAP[@]}" "${BIN}" --rank="${i}" --nranks="${NRANKS}" --uid-file="${UID_FILE}" "$@" \
        > "${LOG_DIR}/rank_${i}.log" 2>&1 &
    PIDS+=($!)
done
FAIL=0
for i in $(seq 0 $((NRANKS - 1))); do
    wait "${PIDS[$i]}" || FAIL=1
done

echo "=== $(basename "${BIN}") nranks=${NRANKS} tool=${TOOL} logs=${LOG_DIR} ==="
cat "${LOG_DIR}/rank_0.log"
if [[ "${TOOL}" != "none" ]]; then
    echo "--- sanitizer summaries (all ranks) ---"
    grep -h "ERROR SUMMARY\|RACECHECK SUMMARY\|SYNCCHECK SUMMARY\|INITCHECK SUMMARY" "${LOG_DIR}"/rank_*.log || true
    if grep -qh "SUMMARY: [1-9]" "${LOG_DIR}"/rank_*.log; then
        FAIL=1
    fi
fi
exit "${FAIL}"
