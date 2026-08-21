#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Build wheels for nccl-extensions
# This script is intended for CI use with uv and cibuildwheel
#
# Usage: build_wheels.sh [LIB[@VERSION] ...]
#
# Targets are passed through to build_lib.py, which stages each library's
# shared object and public headers into the package before the wheels are
# built. A version selects that library's release tag; without one the current
# checkout is used. Defaults to every library the wheel packages.
#
# Requires: CUDA_HOME environment variable set

set -euo pipefail

# Libraries the wheel packages; also the default set to build
DEFAULT_TARGETS=(nccl_ep nccl_m2n)

if [ $# -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=("${DEFAULT_TARGETS[@]}")
fi

# Quote each selector for cibuildwheel's shell command. build_lib.py remains
# the single source of truth for selector validation.
TARGET_ARGUMENTS=()
for target in "${TARGETS[@]}"; do
    printf -v quoted_target '%q' "$target"
    TARGET_ARGUMENTS+=("$quoted_target")
done
TARGET_COMMAND="${TARGET_ARGUMENTS[*]}"

# Check CUDA_HOME is set
if [ -z "${CUDA_HOME:-}" ]; then
    echo "Error: CUDA_HOME is not set" >&2
    exit 1
fi
if [ "$CUDA_HOME" = "~" ]; then
    CUDA_HOME="$HOME"
elif [[ "$CUDA_HOME" = "~/"* ]]; then
    CUDA_HOME="$HOME/${CUDA_HOME:2}"
fi
if ! CUDA_HOME="$(cd "$CUDA_HOME" 2>/dev/null && pwd -P)"; then
    echo "Error: CUDA_HOME is not a directory: $CUDA_HOME" >&2
    exit 1
fi
export CUDA_HOME

# Check uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install from: https://docs.astral.sh/uv/" >&2
    exit 1
fi

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PYTHON_DIR/.." && pwd)"
BUILDDIR="${BUILDDIR:-$REPO_ROOT/build}"
if [ "$BUILDDIR" = "~" ]; then
    BUILDDIR="$HOME"
elif [[ "$BUILDDIR" = "~/"* ]]; then
    BUILDDIR="$HOME/${BUILDDIR:2}"
fi
mkdir -p "$BUILDDIR"
BUILDDIR="$(cd "$BUILDDIR" && pwd -P)"
DIST_DIR="$BUILDDIR/dist"

mkdir -p "$DIST_DIR"

echo "========================================="
echo "Building nccl-extensions wheels"
echo "========================================="
echo "CUDA_HOME: $CUDA_HOME"
echo "Targets: ${TARGETS[*]}"
echo "Output directory: $DIST_DIR"
echo ""

cd "$PYTHON_DIR"

# Build wheels using cibuildwheel. CUDA's libcuda stub satisfies the native
# link; the real driver library is supplied at runtime. Runtime libraries are
# excluded from auditwheel repair via pyproject.toml's repair-wheel-command.
printf -v QUOTED_CUDA_HOME '%q' "$CUDA_HOME"
export CIBW_ENVIRONMENT="CUDA_HOME=$QUOTED_CUDA_HOME NCCL_EXTENSIONS_REQUIRE_NATIVE_LIBS=1"

# The repository is mounted because cibuildwheel copies only the Python package
# into the container, while build_lib.py needs the native sources next to it.
printf -v CUDA_MOUNT '%q' "$CUDA_HOME:$CUDA_HOME:ro"
printf -v REPO_MOUNT '%q' "$REPO_ROOT:$REPO_ROOT"
export CIBW_CONTAINER_ENGINE="docker; create_args: -v $CUDA_MOUNT -v $REPO_MOUNT"

# Build the native libraries inside the manylinux container and stage each .so
# and its owned headers into the package cibuildwheel is about to build. The
# libraries ship in the wheel, so building them on the host would tie the wheel
# to the host glibc and fail auditwheel's manylinux check.
# The container runs as root, so git rejects the mounted checkout as
# dubiously owned; a LIB@VERSION target needs git to create its worktree.
CONTAINER_PYTHON=/opt/python/cp312-cp312/bin/python
printf -v QUOTED_REPO_ROOT '%q' "$REPO_ROOT"
printf -v QUOTED_BUILD_SCRIPT '%q' "$SCRIPT_DIR/build_lib.py"
export CIBW_BEFORE_ALL="git config --global --add safe.directory $QUOTED_REPO_ROOT && \
    $CONTAINER_PYTHON -m pip install -q packaging && \
    $CONTAINER_PYTHON $QUOTED_BUILD_SCRIPT $TARGET_COMMAND \
    --cuda-home \"\$CUDA_HOME\" --output-dir \"{package}\""

uv tool run cibuildwheel --output-dir "$DIST_DIR" --platform linux .

echo ""
echo "========================================="
echo "Build completed successfully!"
echo "========================================="
echo "Output directory: $DIST_DIR"
echo ""
echo "Built packages:"
find "$DIST_DIR" -type f -name "*.whl" | sort
