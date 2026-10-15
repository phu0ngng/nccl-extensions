#!/usr/bin/env bash
# Adapt a Slurm task to the unit tests' existing command-line interface.
set -euo pipefail

: "${SLURM_PROCID:?must run inside a Slurm task}"
: "${SLURM_NTASKS:?must run inside a Slurm task}"
: "${SLURM_JOB_ID:?must run inside a Slurm job}"
: "${SLURM_STEP_ID:?must run inside a Slurm step}"

TEST_BINARY="${1:?usage: run_unit_test.sh BINARY [ARGS...]}"
shift
UID_FILE="${TMPDIR:-/tmp}/te_ep_uid_${TEST_BINARY##*/}_${SLURM_JOB_ID}_${SLURM_STEP_ID}"

exec "${TEST_BINARY}" \
    --rank="${SLURM_PROCID}" \
    --nranks="${SLURM_NTASKS}" \
    --uid-file="${UID_FILE}" \
    "$@"
