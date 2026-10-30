#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the CI wheel in the release's shape from CI-built libraries: EP and M2N
# for cu12 and cu13 staged as build_wheels.sh stages them, one cp312 wheel plus
# the sdist in python/dist/, then content, install and import checks.
#
# Inputs: EP_CU12_DIR, EP_CU13_DIR, M2N_CU12_DIR, M2N_CU13_DIR (each the module
# build's lib/ and include/; relative paths resolve against CI_PROJECT_DIR),
# CUDA_HOME (setup.py compiles against it; its nvcc major picks the extra the
# install check uses) and CI_PROJECT_DIR (the checkout to package).

set -euo pipefail

: "${EP_CU12_DIR:?EP_CU12_DIR must hold the cu12 EP build (lib/, include/)}"
: "${EP_CU13_DIR:?EP_CU13_DIR must hold the cu13 EP build (lib/, include/)}"
: "${M2N_CU12_DIR:?M2N_CU12_DIR must hold the cu12 M2N build (lib/, include/)}"
: "${M2N_CU13_DIR:?M2N_CU13_DIR must hold the cu13 M2N build (lib/, include/)}"
: "${CUDA_HOME:?CUDA_HOME must be set}"
: "${CI_PROJECT_DIR:?CI_PROJECT_DIR must be set}"
export CUDA_HOME

ROOT="$(cd "${CI_PROJECT_DIR}" && pwd)"
PACKAGE_DIR="${ROOT}/python"
EP_PACKAGE_DIR="${PACKAGE_DIR}/nccl/ep"
M2N_PACKAGE_DIR="${PACKAGE_DIR}/nccl/m2n"
DIST_DIR="${PACKAGE_DIR}/dist"
WORK_DIR="${PACKAGE_DIR}/.ci-wheel-${CI_JOB_ID:-local}"
TOOLS_VENV_DIR="${WORK_DIR}/tools"
RUNTIME_VENV_DIR="${WORK_DIR}/runtime"
UV_CACHE_DIR="${WORK_DIR}/uv-cache"
UV_PYTHON_INSTALL_DIR="${WORK_DIR}/uv-python"
PYTHON="${TOOLS_VENV_DIR}/bin/python"
WHEEL=""
SDIST=""

# Headers each library owns, relative to its build include/ (build_lib.py's patterns).
EP_HEADER_PATTERNS=(nccl_ep.h 'nccl_ep/*.h' 'nccl_ep/*.hpp' 'nccl_ep/device/*.cuh' 'nccl_ep/device/ll/*.cuh')
M2N_HEADER_PATTERNS=(nccl_m2n.h)

log() { printf '>>> %s\n' "$*"; }

resolve_dir() {
    local dir=$1
    [[ "${dir}" == /* ]] || dir="${ROOT}/${dir}"
    cd "${dir}" && pwd
}

cleanup_staged() {
    rm -rf "${EP_PACKAGE_DIR:?}/lib" "${EP_PACKAGE_DIR:?}/include" \
        "${M2N_PACKAGE_DIR:?}/lib" "${M2N_PACKAGE_DIR:?}/include"
}

cleanup() {
    cleanup_staged
    rm -rf "${WORK_DIR}"
}

# stage_library <build dir> <package dir> <variant> <library>: the loader opens
# lib/<variant>/<library>, so the symlink chain is flattened into that one file.
stage_library() {
    local build_dir=$1 package_dir=$2 variant=$3 library=$4
    test -s "${build_dir}/lib/${library}"
    mkdir -p "${package_dir}/lib/${variant}"
    cp -L "${build_dir}/lib/${library}" "${package_dir}/lib/${variant}/${library}"
}

# stage_headers <include dir> <destination> <pattern>...: every pattern must match a file.
stage_headers() {
    local include_dir=$1 destination=$2
    shift 2
    local pattern header found
    rm -rf "${destination}"
    for pattern in "$@"; do
        found=0
        while IFS= read -r header; do
            [[ -f "${include_dir}/${header}" ]] || continue
            mkdir -p "${destination}/$(dirname "${header}")"
            cp -L "${include_dir}/${header}" "${destination}/${header}"
            found=1
        done < <(cd "${include_dir}" && compgen -G "${pattern}" || true)
        if (( !found )); then
            echo "ERROR: no header matched ${pattern} under ${include_dir}" >&2
            return 1
        fi
    done
}

# check_headers_match <staged include> <other include>: the wheel stores one header
# set, so the cu13 build must have produced the same files; EP's config.h records
# toolkit paths and is excluded, as in build_wheels.sh.
check_headers_match() {
    local staged_dir=$1 other_include_dir=$2
    local header rel
    while IFS= read -r header; do
        rel="${header#"${staged_dir}/"}"
        if ! cmp -s "${header}" "${other_include_dir}/${rel}"; then
            echo "ERROR: ${rel} differs between the cu12 and cu13 builds" >&2
            return 1
        fi
    done < <(find "${staged_dir}" -type f ! -name config.h | sort)
}

# Stage the four libraries and the cu12 headers where the release puts them:
#   nccl/ep/lib/{cu12,cu13}/libnccl_ep.so    nccl/ep/include/{nccl_ep.h,nccl_ep/**}
#   nccl/m2n/lib/{cu12,cu13}/libnccl_m2n.so  nccl/m2n/include/nccl_m2n.h
stage_libs() {
    local ep_cu12 ep_cu13 m2n_cu12 m2n_cu13
    ep_cu12="$(resolve_dir "${EP_CU12_DIR}")"
    ep_cu13="$(resolve_dir "${EP_CU13_DIR}")"
    m2n_cu12="$(resolve_dir "${M2N_CU12_DIR}")"
    m2n_cu13="$(resolve_dir "${M2N_CU13_DIR}")"
    cleanup_staged
    stage_library "${ep_cu12}" "${EP_PACKAGE_DIR}" cu12 libnccl_ep.so
    stage_library "${ep_cu13}" "${EP_PACKAGE_DIR}" cu13 libnccl_ep.so
    stage_library "${m2n_cu12}" "${M2N_PACKAGE_DIR}" cu12 libnccl_m2n.so
    stage_library "${m2n_cu13}" "${M2N_PACKAGE_DIR}" cu13 libnccl_m2n.so
    stage_headers "${ep_cu12}/include" "${EP_PACKAGE_DIR}/include" "${EP_HEADER_PATTERNS[@]}"
    check_headers_match "${EP_PACKAGE_DIR}/include" "${ep_cu13}/include"
    stage_headers "${m2n_cu12}/include" "${M2N_PACKAGE_DIR}/include" "${M2N_HEADER_PATTERNS[@]}"
    check_headers_match "${M2N_PACKAGE_DIR}/include" "${m2n_cu13}/include"
}

detect_cuda_variant() {
    local nvcc="${CUDA_HOME}/bin/nvcc" version
    if ! version="$("${nvcc}" --version 2>&1)"; then
        echo "ERROR: failed to run ${nvcc}" >&2
        echo "${version}" >&2
        return 1
    fi
    if [[ "${version}" =~ release[[:space:]]+(12|13)\. ]]; then
        printf 'cu%s\n' "${BASH_REMATCH[1]}"
    else
        echo "ERROR: expected CUDA 12 or 13 at ${CUDA_HOME}" >&2
        echo "${version}" >&2
        return 1
    fi
}

# The CI image ships uv in /usr/local/bin; elsewhere bootstrap it with the system pip.
# Either way the wheel is built by an isolated Python 3.12, hence the cp312 tag.
bootstrap_tools() {
    local -a uv_cmd
    export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR
    if command -v uv >/dev/null; then
        uv_cmd=(uv)
    else
        python3 -m pip install --disable-pip-version-check --target "${WORK_DIR}/bootstrap" uv
        uv_cmd=(env "PYTHONPATH=${WORK_DIR}/bootstrap" python3 -m uv)
    fi
    "${uv_cmd[@]}" venv --python 3.12 --seed "${TOOLS_VENV_DIR}"
    "${PYTHON}" -m pip install --disable-pip-version-check build
}

# Wheel from the staged checkout, sdist separately: build's default wheel-from-sdist
# path would drop the staged libraries. The wheel is not auditwheel-repaired: the
# x86 cu12 EP library is built on glibc 2.35, above the manylinux_2_28 policy.
build_dists() {
    export NCCL_EXTENSIONS_REQUIRE_NATIVE_LIBS=1
    "${PYTHON}" -m build --wheel --outdir "${DIST_DIR}" "${PACKAGE_DIR}"
    "${PYTHON}" -m build --sdist --outdir "${DIST_DIR}" "${PACKAGE_DIR}"
}

# One cp312 wheel carrying every staged file and the extras; an sdist carrying the
# generated Cython sources and none of the staged native files.
check_dists() {
    local wheels=("${DIST_DIR}"/*.whl)
    if [[ "${#wheels[@]}" -ne 1 || ! -f "${wheels[0]}" ]]; then
        echo "ERROR: expected exactly one wheel in ${DIST_DIR}" >&2
        return 1
    fi
    WHEEL="${wheels[0]}"
    if [[ "${WHEEL}" != *-cp312-cp312-*.whl ]]; then
        echo "ERROR: expected a cp312 wheel, got ${WHEEL##*/}" >&2
        return 1
    fi
    SDIST="$(find "${DIST_DIR}" -maxdepth 1 -name '*.tar.gz' -print -quit)"
    test -n "${SDIST}"

    "${PYTHON}" - "${WHEEL}" "${PACKAGE_DIR}" <<'PY'
import os
import sys
import zipfile

wheel_path, package_dir = sys.argv[1:3]
expected = {
    "nccl/ep/lib/cu12/libnccl_ep.so",
    "nccl/ep/lib/cu13/libnccl_ep.so",
    "nccl/m2n/lib/cu12/libnccl_m2n.so",
    "nccl/m2n/lib/cu13/libnccl_m2n.so",
    "nccl/ep/include/nccl_ep.h",
    "nccl/m2n/include/nccl_m2n.h",
}
for staged in ("nccl/ep/lib", "nccl/ep/include", "nccl/m2n/lib", "nccl/m2n/include"):
    for dirpath, _, files in os.walk(os.path.join(package_dir, staged)):
        for name in files:
            expected.add(os.path.relpath(os.path.join(dirpath, name), package_dir))
with zipfile.ZipFile(wheel_path) as wheel:
    names = set(wheel.namelist())
    missing = sorted(expected - names)
    if missing:
        raise SystemExit("wheel is missing: " + ", ".join(missing))
    metadata = next(name for name in names if name.endswith(".dist-info/METADATA"))
    text = wheel.read(metadata).decode()
    for extra in ("bench", "cu12", "cu13"):
        if f"Provides-Extra: {extra}" not in text:
            raise SystemExit(f"wheel metadata is missing the {extra} extra")
print(f"wheel carries {len(expected)} staged files")
PY

    local sdist_list="${WORK_DIR}/sdist-files.txt" pyx
    tar -tzf "${SDIST}" > "${sdist_list}"
    for pyx in nccl_ep cynccl_ep nccl_m2n cynccl_m2n; do
        grep -qE "/nccl/_extensions/bindings/${pyx}\.pyx$" "${sdist_list}"
    done
    if grep -E '/nccl/(ep|m2n)/(lib|include)/|\.so$' "${sdist_list}"; then
        echo "ERROR: the source distribution contains staged native files" >&2
        return 1
    fi
}

# Install the wheel into a clean venv through the public extras and import every
# facade and binding module. The native libraries load lazily, so this proves the
# package layout and dependency closure, not GPU function.
check_install() {
    local variant=$1
    "${PYTHON}" -m venv "${RUNTIME_VENV_DIR}"
    "${RUNTIME_VENV_DIR}/bin/pip" install --disable-pip-version-check \
        "${WHEEL}[${variant},bench]"
    "${RUNTIME_VENV_DIR}/bin/python" -c \
        'import nccl.ep, nccl.m2n; from nccl._extensions.bindings import nccl_ep, nccl_m2n'
    "${RUNTIME_VENV_DIR}/bin/python" -m nccl.m2n.benchmarks.reshard_bench --help
}

main() {
    local cuda_variant
    cuda_variant="$(detect_cuda_variant)"
    trap cleanup EXIT
    rm -rf "${DIST_DIR}" "${WORK_DIR}"
    mkdir -p "${DIST_DIR}" "${WORK_DIR}"
    log "staging EP and M2N cu12/cu13 libraries and headers into ${PACKAGE_DIR}/nccl"
    stage_libs
    log "provisioning the Python 3.12 build environment in ${WORK_DIR}"
    bootstrap_tools
    log "building the wheel and sdist into ${DIST_DIR}"
    build_dists
    log "checking the wheel and sdist contents"
    check_dists
    log "installing ${WHEEL##*/}[${cuda_variant},bench] into a clean venv"
    check_install "${cuda_variant}"
    log "built ${WHEEL##*/} and ${SDIST##*/}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
