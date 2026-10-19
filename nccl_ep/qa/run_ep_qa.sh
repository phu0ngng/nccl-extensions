#!/usr/bin/env bash
# Standalone NCCL-EP QA runner for machines without Slurm or nvsubmit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INSTALL="${NCCL_HOME:-${REPO_ROOT}/build}"
SUITE="smoke"
NP=""
UNIT_GPUS=""
HOSTFILE=""
MPIEXEC="${MPIEXEC:-mpirun}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR=""
CASE_FILTER="*"
CASE_TIMEOUT="${NCCL_EP_QA_TIMEOUT:-1800}"
PYTHON_TIMEOUT="${NCCL_EP_QA_PYTHON_TIMEOUT:-300}"
ENABLE_HT=1
ENABLE_NVFP4=0
DRY_RUN=0
LIST_ONLY=0
FAIL_FAST=0
MPI_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_ep_qa.sh [OPTIONS]

Run NCCL-EP tests on a standalone machine using mpirun/mpiexec. This script
never builds NCCL or NCCL-EP; --install must point at prepared artifacts.

Options:
  --install DIR         NCCL/NCCL-EP install root (default: $NCCL_HOME or build/)
  --suite NAME          smoke, unit, bench, python, or full (default: smoke)
  --np N                MPI ranks (default: detected local GPU count)
  --unit-gpus N         Local GPU count for unit tests (default: detected)
  --hostfile FILE       MPI hostfile for multi-node runs
  --mpiexec PATH        MPI launcher (default: mpirun)
  --mpi-arg ARG         Extra MPI launcher argument; may be repeated
  --python PATH         Python interpreter for the Python suite
  --log-dir DIR         Result directory (default: ep-qa-results-<timestamp>)
  --case GLOB           Run only matching case names (default: *)
  --timeout SEC         C++/unit case timeout (default: 1800)
  --python-timeout SEC  Python case timeout (default: 300)
  --disable-ht          Skip high-throughput benchmark cases
  --enable-nvfp4       Enable the Blackwell NVFP4 benchmark case
  --fail-fast           Stop after the first failed case
  --list                List selected case names without preflight checks
  --dry-run             Print selected commands without executing them
  -h, --help            Show this help

Examples:
  run_ep_qa.sh --install /opt/nccl-ep --np 4 --suite smoke
  run_ep_qa.sh --install /opt/nccl-ep --np 16 --hostfile hosts --suite bench
  run_ep_qa.sh --suite bench --case 'bench-ll-*' --dry-run
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_uint() {
  local flag="$1" value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${flag} requires a positive integer"
}

while (( $# > 0 )); do
  case "$1" in
    --install) [[ $# -ge 2 ]] || die "--install requires a value"; INSTALL="$2"; shift 2 ;;
    --suite) [[ $# -ge 2 ]] || die "--suite requires a value"; SUITE="$2"; shift 2 ;;
    --np) [[ $# -ge 2 ]] || die "--np requires a value"; require_uint --np "$2"; NP="$2"; shift 2 ;;
    --unit-gpus) [[ $# -ge 2 ]] || die "--unit-gpus requires a value"; require_uint --unit-gpus "$2"; UNIT_GPUS="$2"; shift 2 ;;
    --hostfile) [[ $# -ge 2 ]] || die "--hostfile requires a value"; HOSTFILE="$2"; shift 2 ;;
    --mpiexec) [[ $# -ge 2 ]] || die "--mpiexec requires a value"; MPIEXEC="$2"; shift 2 ;;
    --mpi-arg) [[ $# -ge 2 ]] || die "--mpi-arg requires a value"; MPI_EXTRA_ARGS+=("$2"); shift 2 ;;
    --python) [[ $# -ge 2 ]] || die "--python requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --log-dir) [[ $# -ge 2 ]] || die "--log-dir requires a value"; LOG_DIR="$2"; shift 2 ;;
    --case) [[ $# -ge 2 ]] || die "--case requires a value"; CASE_FILTER="$2"; shift 2 ;;
    --timeout) [[ $# -ge 2 ]] || die "--timeout requires a value"; require_uint --timeout "$2"; CASE_TIMEOUT="$2"; shift 2 ;;
    --python-timeout) [[ $# -ge 2 ]] || die "--python-timeout requires a value"; require_uint --python-timeout "$2"; PYTHON_TIMEOUT="$2"; shift 2 ;;
    --disable-ht) ENABLE_HT=0; shift ;;
    --enable-nvfp4) ENABLE_NVFP4=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --list) LIST_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

NEED_MPI=0
NEED_UNIT=0
case "${SUITE}" in
  smoke|bench|python) NEED_MPI=1 ;;
  unit) NEED_UNIT=1 ;;
  full) NEED_MPI=1; NEED_UNIT=1 ;;
  *) die "unsupported suite '${SUITE}' (expected smoke, unit, bench, python, or full)" ;;
esac

if (( NEED_MPI )); then
  if [[ -z "${NP}" ]]; then
    if (( LIST_ONLY || DRY_RUN )); then
      NP=4
    elif command -v nvidia-smi >/dev/null 2>&1; then
      NP="$(nvidia-smi -L 2>/dev/null | wc -l)"
    fi
    [[ "${NP:-0}" =~ ^[1-9][0-9]*$ ]] || die "could not detect GPUs; pass --np"
  fi
  [[ -z "${HOSTFILE}" || -f "${HOSTFILE}" ]] || die "hostfile not found: ${HOSTFILE}"
fi

if (( NEED_UNIT )); then
  if [[ -z "${UNIT_GPUS}" ]]; then
    if (( LIST_ONLY || DRY_RUN )); then
      UNIT_GPUS=4
    elif command -v nvidia-smi >/dev/null 2>&1; then
      UNIT_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
    elif (( NEED_MPI )) && [[ -z "${HOSTFILE}" ]]; then
      UNIT_GPUS="${NP}"
    fi
    [[ "${UNIT_GPUS:-0}" =~ ^[1-9][0-9]*$ ]] || die "could not detect local GPUs; pass --unit-gpus"
  fi
fi
if [[ -d "${INSTALL}" ]]; then
  INSTALL="$(cd "${INSTALL}" && pwd)"
elif (( ! LIST_ONLY && ! DRY_RUN )); then
  die "install directory not found: ${INSTALL}"
fi
LOG_DIR="${LOG_DIR:-${PWD}/ep-qa-results-$(date -u +%Y%m%dT%H%M%SZ)}"

EP_TEST="${INSTALL}/test/nccl_ep/ep_test"
EP_BENCH="${INSTALL}/test/nccl_ep/ep_bench"
AF_TEST="${INSTALL}/test/nccl_ep/af_test"
EP_TEST_PY="${REPO_ROOT}/nccl_ep/ep_test.py"
AFD_BINDINGS_TEST="${REPO_ROOT}/python/ci/verify_ep_afd_bindings.py"
UNIT_RUNNER="${REPO_ROOT}/nccl_ep/tests/run_tests.sh"

export NCCL_HOME="${INSTALL}"
export NCCL_EP_HOME="${INSTALL}"
export NCCL_EP_BUILDDIR="${INSTALL}"
export NCCL_EP_JIT_SOURCE_DIR="${INSTALL}/include/nccl_ep"
export NCCL_EP_JIT_BUILD_INCLUDE_DIR="${INSTALL}/include"
export NCCL_GIN_TYPE="${NCCL_GIN_TYPE:-3}"
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  export CUDA_HOME
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export NCCL_EP_JIT_CUDA_INCLUDE_DIR="${CUDA_HOME}/include"
  export NCCL_EP_JIT_NVCC="${CUDA_HOME}/bin/nvcc"
  export LD_LIBRARY_PATH="${INSTALL}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${INSTALL}/lib:${LD_LIBRARY_PATH:-}"
fi

MPI_CMD=()
if (( NEED_MPI )); then
  MPI_CMD=("${MPIEXEC}" -np "${NP}")
  [[ -z "${HOSTFILE}" ]] || MPI_CMD+=(--hostfile "${HOSTFILE}")
  MPI_CMD+=("${MPI_EXTRA_ARGS[@]}")
fi

RESULT_NAMES=()
RESULT_STATUS=()
RESULT_SECONDS=()
RESULT_LOGS=()
PASS_COUNT=0
FAIL_COUNT=0

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

case_selected() {
  [[ "$1" == ${CASE_FILTER} ]]
}

record_result() {
  RESULT_NAMES+=("$1")
  RESULT_STATUS+=("$2")
  RESULT_SECONDS+=("$3")
  RESULT_LOGS+=("$4")
  if [[ "$2" == PASS ]]; then
    ((PASS_COUNT += 1))
  else
    ((FAIL_COUNT += 1))
  fi
}

run_case() {
  local name="$1" timeout_s="$2"
  shift 2
  case_selected "${name}" || return 0
  if (( LIST_ONLY )); then
    printf '%s\n' "${name}"
    return 0
  fi
  if (( DRY_RUN )); then
    printf '%s\n' "${name}"
    print_command timeout --signal=TERM --kill-after=30 "${timeout_s}" "$@"
    return 0
  fi

  local log="${LOG_DIR}/${name}.log" start end rc status
  printf '\n===== %s =====\n' "${name}"
  print_command "$@"
  start="$(date +%s)"
  set +e
  timeout --signal=TERM --kill-after=30 "${timeout_s}" "$@" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  end="$(date +%s)"
  if (( rc == 0 )); then
    status=PASS
    printf 'CASE_RESULT: PASS (%s)\n' "${name}"
  else
    status=FAIL
    printf 'CASE_RESULT: FAIL (%s, rc=%d)\n' "${name}" "${rc}"
  fi
  record_result "${name}" "${status}" "$((end - start))" "${log}"
  if (( rc != 0 && FAIL_FAST )); then
    write_results
    exit "${rc}"
  fi
}

run_mpi_case() {
  local name="$1" timeout_s="$2" binary="$3"
  shift 3
  run_case "${name}" "${timeout_s}" "${MPI_CMD[@]}" "${binary}" "$@"
}

run_mpi_case_np() {
  local name="$1" timeout_s="$2" case_np="$3" binary="$4"
  shift 4
  local case_mpi_cmd=("${MPIEXEC}" -np "${case_np}")
  [[ -z "${HOSTFILE}" ]] || case_mpi_cmd+=(--hostfile "${HOSTFILE}")
  case_mpi_cmd+=("${MPI_EXTRA_ARGS[@]}")
  run_case "${name}" "${timeout_s}" "${case_mpi_cmd[@]}" "${binary}" "$@"
}

run_smoke_suite() {
  run_mpi_case smoke-ep-test-ll "${CASE_TIMEOUT}" "${EP_TEST}" -a ll -t 128 -d 7168
  run_mpi_case smoke-ep-test-ht-flat "${CASE_TIMEOUT}" "${EP_TEST}" -a ht -L fl -t 4096 -d 7168
  run_mpi_case smoke-ep-test-ht-expert-major "${CASE_TIMEOUT}" "${EP_TEST}" -a ht -L em -t 4096 -d 7168
  run_mpi_case smoke-ep-test-ht-flat-query "${CASE_TIMEOUT}" "${EP_TEST}" -a ht -L fl -q -t 4096 -d 7168
  run_mpi_case smoke-ep-test-ht-expert-major-query "${CASE_TIMEOUT}" "${EP_TEST}" -a ht -L em -q -t 4096 -d 7168
}

run_bench_sweep() {
  local algorithm="$1" hidden="$2"
  shift 2
  local layouts="$1" tokens="$2" dtypes="$3"
  local layout token dtype
  for layout in ${layouts}; do
    for token in ${tokens}; do
      for dtype in ${dtypes}; do
        run_mpi_case "bench-${algorithm}-${layout}-${token}-${dtype}" "${CASE_TIMEOUT}" "${EP_BENCH}" \
          --algorithm "${algorithm}" --layout "${layout}" --tokens "${token}" \
          --hidden "${hidden}" --top-k 8 --experts 256 --validate --datatype "${dtype}"
      done
    done
  done
}

run_benchmark_suite() {
  # AFD requires an even, square attention:FFN rank partition. The larger
  # top-k case mirrors the fixed 32-expert CI geometry where it is valid.
  if (( NP >= 4 && NP % 2 == 0 )); then
    run_mpi_case bench-afd-staged "${CASE_TIMEOUT}" "${AF_TEST}" -d 2048 -t 32 -i 2
    run_mpi_case bench-afd-zero-copy-auto "${CASE_TIMEOUT}" "${AF_TEST}" -d 2048 -t 32 -i 2 -z
    if (( 32 > NP && 32 % (NP / 2) == 0 )); then
      run_mpi_case bench-afd-topk8 "${CASE_TIMEOUT}" "${AF_TEST}" -d 2048 -t 32 -e 32 -k 8 -i 2
    fi
  fi

  run_bench_sweep low-latency 7168 "em rm" "128 256 1024 4096" "bf16 fp16"
  run_bench_sweep low-latency 7168 "em rm" "128 256 1024 2048" "fp32"

  run_mpi_case bench-ll-less-than-max "${CASE_TIMEOUT}" "${EP_BENCH}" \
    --algorithm low-latency --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
    --validate --dispatch-less-than-max-tokens 64
  run_mpi_case bench-ll-non-uniform "${CASE_TIMEOUT}" "${EP_BENCH}" \
    --algorithm low-latency --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
    --validate --non-uniform-tokens
  run_mpi_case bench-ll-independent-sm-counts "${CASE_TIMEOUT}" "${EP_BENCH}" \
    --algorithm low-latency --layout em --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
    --dispatch-num-sms 21 --combine-num-sms 23 --validate
  for layout in rm em; do
    run_mpi_case "bench-ll-partial-warp-${layout}" "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm low-latency --layout "${layout}" --tokens 1 --hidden 2880 --top-k 4 \
      --experts 128 --datatype bf16 --topk-idx-int32 --validate
    run_mpi_case "bench-ll-topk31-${layout}" "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm low-latency --layout "${layout}" --tokens 128 --hidden 7168 --top-k 31 \
      --experts 256 --validate
  done
  run_mpi_case bench-ll-topk31-int32 "${CASE_TIMEOUT}" "${EP_BENCH}" \
    --algorithm low-latency --layout em --tokens 128 --hidden 7168 --top-k 31 \
    --experts 256 --topk-idx-int32 --validate
  run_mpi_case bench-ll-quant-fp8-fp32 "${CASE_TIMEOUT}" "${EP_BENCH}" \
    --algorithm low-latency --layout em --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
    --validate --dispatch-only --dispatch-quantization scales-forward \
    --scales-forward-token-dtype fp8e4m3 --scales-forward-scale-dtype fp32
  run_mpi_case bench-ll-quant-bf16-fp16 "${CASE_TIMEOUT}" "${EP_BENCH}" \
    --algorithm low-latency --layout em --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
    --validate --dispatch-only --dispatch-quantization scales-forward \
    --scales-forward-token-dtype bf16 --scales-forward-scale-dtype fp16
  run_mpi_case bench-ll-zcopy-fp4 "${CASE_TIMEOUT}" "${EP_BENCH}" \
    --algorithm low-latency --layout rm --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
    --validate --dispatch-only --zcopy --dispatch-quantization scales-forward \
    --scales-forward-token-dtype fp4x2 --scales-forward-scale-dtype uint8
  run_mpi_case bench-ll-ds-fp8 "${CASE_TIMEOUT}" "${EP_BENCH}" \
    --algorithm low-latency --layout em --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
    --validate --dispatch-only --dispatch-quantization ds-fp8e3m4

  # LL DSFP8 zero-copy is an NVLink-only path; a hostfile denotes the RDMA lane.
  if [[ -z "${HOSTFILE}" ]]; then
    run_mpi_case bench-ll-ds-fp8-zcopy "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm low-latency --layout rm --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
      --validate --dispatch-only --zcopy --dispatch-quantization ds-fp8e3m4
  fi

  if (( ENABLE_NVFP4 )); then
    run_mpi_case bench-ll-nvfp4-combine "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm low-latency --layout em --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
      --datatype bf16 --combine-quantization nvfp4 --disable-token-dropping --validate
  fi

  if (( ENABLE_HT )); then
    run_bench_sweep high-throughput 7168 "fl em" "128 256 1024 4096" "bf16 fp16"
    run_bench_sweep high-throughput 3584 "fl em" "128 256 1024 2048" "fp32"

    # QA boundary: a non-aligned token count across both HT layouts and all wire dtypes.
    if (( 512 % NP == 0 )); then
      local nonaligned_capacity_args=()
      for layout in fl em; do
        nonaligned_capacity_args=()
        [[ "${layout}" == em ]] && nonaligned_capacity_args=(--max-recv-token-slots-per-rank 65536)
        for dtype in bf16 fp16 fp32; do
          run_mpi_case "bench-ht-nonaligned-${layout}-${dtype}" "${CASE_TIMEOUT}" "${EP_BENCH}" \
            --algorithm high-throughput --layout "${layout}" --tokens 4100 --hidden 7168 \
            --top-k 8 --experts 512 --datatype "${dtype}" "${nonaligned_capacity_args[@]}" --validate
        done
      done
    fi

    # QA boundary: four ranks with 96 local experts each.
    if (( NP >= 4 )); then
      local high_expert_capacity_args=()
      for layout in fl em; do
        high_expert_capacity_args=()
        [[ "${layout}" == em ]] && high_expert_capacity_args=(--max-recv-token-slots-per-rank 65536)
        run_mpi_case_np "bench-ht-high-local-experts-${layout}" "${CASE_TIMEOUT}" 4 "${EP_BENCH}" \
          --algorithm high-throughput --layout "${layout}" --tokens 4096 --hidden 7168 \
          --top-k 8 --experts 384 "${high_expert_capacity_args[@]}" --validate
      done
    fi
    run_mpi_case bench-ht-less-than-max "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm high-throughput --tokens 4096 --hidden 7168 --top-k 8 --experts 256 \
      --validate --dispatch-less-than-max-tokens 2048
    run_mpi_case bench-ht-non-uniform "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm high-throughput --tokens 4096 --hidden 7168 --top-k 8 --experts 256 \
      --validate --non-uniform-tokens
    run_mpi_case bench-ht-independent-sm-counts "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm high-throughput --layout fl --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
      --dispatch-num-sms 15 --combine-num-sms 17 --validate
    run_mpi_case bench-ht-meta-force-scan "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm high-throughput --layout em --tokens 4096 --hidden 7168 --top-k 8 --experts 256 \
      --meta-exchange-policy force-scan --validate
    # REQUIRE_FUSED is defined only for a single LSA team, so keep it out of the RDMA lane.
    if [[ -z "${HOSTFILE}" ]]; then
      run_mpi_case bench-ht-meta-require-fused "${CASE_TIMEOUT}" "${EP_BENCH}" \
        --algorithm high-throughput --layout fl --tokens 4096 --hidden 7168 --top-k 8 --experts 256 \
        --ht-flow push-pull --meta-exchange-policy require-fused --validate
    fi
    run_mpi_case bench-ht-zcopy-fp4 "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm high-throughput --layout fl --tokens 128 --hidden 7168 --top-k 8 --experts 256 \
      --validate --dispatch-only --zcopy --dispatch-quantization scales-forward \
      --scales-forward-token-dtype fp4x2 --scales-forward-scale-dtype uint8
    run_case bench-ht-pull-push-bf16 "${CASE_TIMEOUT}" env \
      NCCL_LSA_TEAM_SIZE="${NP}" "${MPI_CMD[@]}" "${EP_BENCH}" \
      --algorithm high-throughput --layout em --tokens 4096 --hidden 7168 --top-k 8 \
      --experts 256 --validate --ht-flow pull-push
    run_case bench-ht-pull-push-mxfp8 "${CASE_TIMEOUT}" env \
      NCCL_LSA_TEAM_SIZE="${NP}" "${MPI_CMD[@]}" "${EP_BENCH}" \
      --algorithm high-throughput --layout em --tokens 4096 --hidden 7168 --top-k 8 \
      --experts 256 --validate --dispatch-only --mxfp8 --ht-flow pull-push
    run_mpi_case bench-ht-mxfp8-combine "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm high-throughput --layout em --tokens 4096 --hidden 7168 --top-k 8 --experts 256 \
      --validate --combine-quantization mxfp8
    run_mpi_case bench-ht-mxfp8-end-to-end "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm high-throughput --layout em --tokens 4096 --hidden 7168 --top-k 8 --experts 256 \
      --validate --mxfp8
    run_mpi_case bench-ht-mxfp8-backward "${CASE_TIMEOUT}" "${EP_BENCH}" \
      --algorithm high-throughput --layout em --tokens 4096 --hidden 7168 --top-k 8 --experts 256 \
      --validate --backward --combine-quantization mxfp8

    # QA multi-node boundary: ten experts per rank and non-power-of-two totals.
    # Pair the two requested hidden/top-k extremes to avoid a redundant cross product.
    if [[ -n "${HOSTFILE}" ]] && (( NP == 8 || NP == 16 )); then
      local nonpower_experts=$((NP * 10)) layout shape hidden top_k
      local recv_capacity_args=()
      for layout in fl em; do
        for shape in "8192 8" "16384 22"; do
          read -r hidden top_k <<< "${shape}"
          recv_capacity_args=()
          if [[ "${layout}" == em ]]; then
            # Bound the random-routing receive allocation without using the huge theoretical worst case.
            recv_capacity_args=(--max-recv-token-slots-per-rank 131072)
          fi
          run_mpi_case "bench-ht-nonpower-experts-np${NP}-${layout}-h${hidden}-k${top_k}" \
            "${CASE_TIMEOUT}" "${EP_BENCH}" --algorithm high-throughput --layout "${layout}" \
            --tokens 4096 --hidden "${hidden}" --top-k "${top_k}" --experts "${nonpower_experts}" \
            "${recv_capacity_args[@]}" --validate
        done
      done
    fi
  fi
}

run_unit_suite() {
  run_case unit-matrix "${CASE_TIMEOUT}" env NCCL_HOME="${INSTALL}" NCCL_EP_BUILDDIR="${INSTALL}" \
    bash "${UNIT_RUNNER}" "${UNIT_GPUS}"
}

run_python_case() {
  local name="$1"
  shift
  run_mpi_case "python-${name}" "${PYTHON_TIMEOUT}" "${PYTHON_BIN}" "${EP_TEST_PY}" \
    -t 16 -d 1024 "$@"
}

run_python_suite() {
  run_case python-afd-bindings "${PYTHON_TIMEOUT}" "${PYTHON_BIN}" "${AFD_BINDINGS_TEST}"
  run_python_case ll-em -a ll -L em
  run_python_case ll-rm -a ll -L rm
  run_python_case ht-fl -a ht -L fl
  run_python_case ht-em -a ht -L em --ht-flow push-pull --meta-exchange-policy force-scan
  run_python_case ll-quant -a ll -Q
  run_python_case ht-quant -a ht -Q
  run_python_case ll-em-staged-dispatch -a ll -L em -s dispatch
  run_python_case ll-em-staged-combine -a ll -L em -s combine
  run_python_case ll-em-staged-both -a ll -L em -s both
  run_python_case ll-rm-staged-both -a ll -L rm -s both
  run_python_case ht-fl-cached -a ht -L fl -c
  run_python_case ll-em-update -a ll -L em --update
  run_python_case ll-rm-update -a ll -L rm --update
  run_python_case ht-fl-update -a ht -L fl --update
  run_python_case ht-em-update -a ht -L em --update
  run_python_case ht-fl-eager -a ht -L fl -q
  run_python_case ht-em-eager -a ht -L em -q
  run_python_case ht-fl-backward -a ht -L fl --backward
  run_python_case ht-em-backward -a ht -L em --backward
  run_python_case ht-fl-overflow-drop -a ht -L fl --overflow-drop
  run_python_case ll-rm-local-expert-ids -a ll -L rm --expert-id-kind local
  run_python_case ll-rm-global-expert-ids -a ll -L rm --expert-id-kind global
  run_python_case ht-fl-local-expert-ids -a ht -L fl --expert-id-kind local
  run_python_case ht-fl-global-expert-ids -a ht -L fl --expert-id-kind global
  run_python_case ht-em-alignment -a ht -L em --alignment 8
  run_python_case ll-em-mask -a ll -L em --mask
  run_python_case ll-em-random -a ll -L em -r
  run_python_case ht-fl-random -a ht -L fl -r
}

preflight() {
  command -v timeout >/dev/null 2>&1 || die "timeout is required"
  command -v tee >/dev/null 2>&1 || die "tee is required"
  if (( NEED_MPI )); then
    command -v "${MPIEXEC}" >/dev/null 2>&1 || die "MPI launcher not found: ${MPIEXEC}"
  fi
  case "${SUITE}" in
    smoke|full)
      [[ -x "${EP_TEST}" ]] || die "ep_test not found or executable: ${EP_TEST}"
      ;;
  esac
  case "${SUITE}" in
    bench|full)
      [[ -x "${EP_BENCH}" ]] || die "ep_bench not found or executable: ${EP_BENCH}"
      [[ -x "${AF_TEST}" ]] || die "af_test not found or executable: ${AF_TEST}"
      ;;
  esac
  case "${SUITE}" in
    unit|full)
      [[ -x "${UNIT_RUNNER}" || -f "${UNIT_RUNNER}" ]] || die "unit runner not found: ${UNIT_RUNNER}"
      [[ -x "${INSTALL}/test/nccl_ep/test_cft_driver_api" ]] || \
        die "unit binaries are incomplete under ${INSTALL}/test/nccl_ep"
      ;;
  esac
  case "${SUITE}" in
    python|full)
      command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python interpreter not found: ${PYTHON_BIN}"
      [[ -f "${EP_TEST_PY}" ]] || die "Python test not found: ${EP_TEST_PY}"
      [[ -f "${AFD_BINDINGS_TEST}" ]] || die "AFD bindings test not found: ${AFD_BINDINGS_TEST}"
      ;;
  esac
}

capture_environment() {
  {
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'repo_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf 'suite=%s\nnp=%s\nunit_gpus=%s\ninstall=%s\n' \
      "${SUITE}" "${NP:-unused}" "${UNIT_GPUS:-unused}" "${INSTALL}"
    printf 'mpiexec=%s\npython=%s\n' "${MPIEXEC}" "${PYTHON_BIN}"
    printf 'cuda_home=%s\n' "${CUDA_HOME:-unset}"
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version --format=csv || true
    command -v nvcc >/dev/null 2>&1 && nvcc --version || true
    if (( NEED_MPI )); then
      "${MPIEXEC}" --version 2>/dev/null | head -n 3 || true
    fi
  } > "${LOG_DIR}/environment.txt"
}

xml_escape() {
  local value="$1"
  value="${value//&/&amp;}"
  value="${value//</&lt;}"
  value="${value//>/&gt;}"
  value="${value//\"/&quot;}"
  printf '%s' "${value}"
}

write_results() {
  (( LIST_ONLY || DRY_RUN )) && return 0
  local tsv="${LOG_DIR}/results.tsv" junit="${LOG_DIR}/junit.xml" i
  printf 'case\tstatus\tseconds\tlog\n' > "${tsv}"
  for ((i = 0; i < ${#RESULT_NAMES[@]}; ++i)); do
    printf '%s\t%s\t%s\t%s\n' \
      "${RESULT_NAMES[$i]}" "${RESULT_STATUS[$i]}" "${RESULT_SECONDS[$i]}" "${RESULT_LOGS[$i]}" >> "${tsv}"
  done
  {
    printf '<?xml version="1.0" encoding="UTF-8"?>\n'
    printf '<testsuite name="nccl-ep-qa" tests="%d" failures="%d">\n' "${#RESULT_NAMES[@]}" "${FAIL_COUNT}"
    for ((i = 0; i < ${#RESULT_NAMES[@]}; ++i)); do
      printf '  <testcase name="%s" time="%s">' "$(xml_escape "${RESULT_NAMES[$i]}")" "${RESULT_SECONDS[$i]}"
      if [[ "${RESULT_STATUS[$i]}" == FAIL ]]; then
        printf '<failure message="case failed">See %s</failure>' "$(xml_escape "${RESULT_LOGS[$i]}")"
      fi
      printf '</testcase>\n'
    done
    printf '</testsuite>\n'
  } > "${junit}"
}

if (( ! LIST_ONLY && ! DRY_RUN )); then
  preflight
  mkdir -p "${LOG_DIR}"
  capture_environment
fi

case "${SUITE}" in
  smoke) run_smoke_suite ;;
  unit) run_unit_suite ;;
  bench) run_benchmark_suite ;;
  python) run_python_suite ;;
  full) run_unit_suite; run_smoke_suite; run_benchmark_suite; run_python_suite ;;
esac

write_results
if (( LIST_ONLY || DRY_RUN )); then
  exit 0
fi

printf '\n===== NCCL-EP QA SUMMARY =====\n'
printf 'passed=%d failed=%d total=%d\n' "${PASS_COUNT}" "${FAIL_COUNT}" "$((PASS_COUNT + FAIL_COUNT))"
printf 'results=%s\n' "${LOG_DIR}"
(( FAIL_COUNT == 0 ))
