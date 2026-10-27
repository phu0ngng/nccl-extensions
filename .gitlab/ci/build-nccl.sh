#!/usr/bin/env bash
# Reuse finished NCCL output across pipelines. Sources are always from checkout.
set -euo pipefail
BUILD_JOBS="${KUBERNETES_CPU_REQUEST:-$(nproc)}"

cache_dir=.ci/nccl-build-cache
build_dir=third_party/nccl/build

# One registry version per NCCL pin, arch and toolchain; the full fingerprint is
# re-checked after restore so a stale or foreign tarball is never a hit.
fingerprint=$(
  set -e
  {
    git -C third_party/nccl rev-parse HEAD
    printf '%s\n' "${CI_JOB_IMAGE:?}" "${NVCC_GENCODE:?}"
    uname -m
    ${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc} --version
    ${CXX:-g++} --version
    sha256sum .gitlab/ci/build-nccl.sh .gitlab/ci/nccl-build.yml
    # Include optional Make overrides and NVCC environment flags.
    for name in CUDA_HOME CUDA_LIB CUDA_INC NVCC CXX CC CXXSTD CXXFLAGS \
      CPPFLAGS LDFLAGS NVCCFLAGS NVCUFLAGS NVCC_APPEND_FLAGS NVCC_PREPEND_FLAGS \
      CUDARTLIB DEBUG TRACE VERBOSE \
      KEEP NVTX ASAN UBSAN TSAN WERROR PROFAPI RDMA_CORE NET_PROFILER \
      MLX5DV MAX_EXT_NET_PLUGINS EMIT_LLVM_IR DOCA_HOME \
      NCCL_ONLY_FUNCS NCCL_ONLY_TYPES NCCL_ONLY_ALGOS \
      MAKEFLAGS MAKEOVERRIDES; do
      printf '%s=%s\n' "${name}" "${!name-}"
    done
  } | sha256sum | cut -d ' ' -f 1
)

# Finished output lives in the project's generic package registry; GitLab's runner
# cache never hits on the k8s runners (no shared cache backend). Fail-open: any
# registry error degrades to a miss, and a failed publish never fails the build.
NCCL_CACHE="${NCCL_CACHE:-1}"
NCCL_CACHE_PACKAGE="${NCCL_CACHE_PACKAGE:-nccl-ci-build-cache}"
nccl_commit=$(git -C third_party/nccl rev-parse HEAD)
arch=$(uname -m)
version="${nccl_commit:0:12}-${arch}-${fingerprint:0:16}"
scratch=.ci/nccl-build-cache-xfer
if [[ "${NCCL_CACHE}" == 1 ]]; then
  pkg_url="${CI_API_V4_URL:?}/projects/${CI_PROJECT_ID:?}/packages/generic/${NCCL_CACHE_PACKAGE}/${version}"
  curl_cmd=(curl -fs --retry 1 --connect-timeout 15 --max-time 600 -H "JOB-TOKEN: ${CI_JOB_TOKEN:?}")
fi

# Called as a condition, so errexit is off inside: every step is chained explicitly.
# cache_dir is per-job scratch on fresh k8s pods; the registry is the only cross-job cache to check.
restore_cache() {
  [[ "${NCCL_CACHE}" == 1 ]] || return 1
  rm -rf "${scratch}"
  if mkdir -p "${scratch}" &&
     "${curl_cmd[@]}" -o "${scratch}/nccl.tar.gz" "${pkg_url}/nccl.tar.gz" &&
     "${curl_cmd[@]}" -o "${scratch}/nccl.tar.gz.sha256" "${pkg_url}/nccl.tar.gz.sha256" &&
     (cd "${scratch}" && sha256sum -c --quiet nccl.tar.gz.sha256) &&
     rm -rf "${cache_dir}" && mkdir -p "${cache_dir}" &&
     tar -xzf "${scratch}/nccl.tar.gz" -C "${cache_dir}"; then
    rm -rf "${scratch}"
    return 0
  fi
  rm -rf "${cache_dir}" "${scratch}"
  echo "NCCL build cache: registry restore failed (non-fatal)"
  return 1
}

publish_cache() {
  [[ "${NCCL_CACHE}" == 1 ]] || return 0
  rm -rf "${scratch}"
  mkdir -p "${scratch}" || return 1
  tar -czf "${scratch}/nccl.tar.gz" -C "${cache_dir}" build fingerprint || return 1
  (cd "${scratch}" && sha256sum nccl.tar.gz > nccl.tar.gz.sha256) || return 1
  cat > "${scratch}/metadata.json" <<EOF || return 1
{
  "nccl_commit": "${nccl_commit}",
  "arch": "${arch}",
  "image": "${CI_JOB_IMAGE}",
  "nvcc_gencode": "${NVCC_GENCODE}",
  "fingerprint": "${fingerprint}",
  "version": "${version}",
  "pipeline_id": "${CI_PIPELINE_ID-}",
  "pipeline_url": "${CI_PIPELINE_URL-}",
  "commit_sha": "${CI_COMMIT_SHA-}",
  "ref": "${CI_COMMIT_REF_NAME-}",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
  local name
  for name in nccl.tar.gz nccl.tar.gz.sha256 metadata.json; do
    "${curl_cmd[@]}" -S --upload-file "${scratch}/${name}" "${pkg_url}/${name}" >/dev/null || return 1
  done
  echo "NCCL build cache published: ${version} ($(du -h "${scratch}/nccl.tar.gz" | cut -f1))"
}

cache_source=local
if restore_cache; then
  cache_source="registry ${version}"
fi

if [[ -f "${cache_dir}/fingerprint" ]] &&
   [[ "$(cat "${cache_dir}/fingerprint")" == "${fingerprint}" ]] &&
   [[ -s "${cache_dir}/build/include/nccl.h" ]] &&
   [[ -s "${cache_dir}/build/lib/libnccl.so" ]] &&
   [[ -s "${cache_dir}/build/lib/libnccl_static.a" ]]; then
  echo "NCCL build cache hit: ${fingerprint} (${cache_source})"
  mkdir -p "${build_dir}"
  cp -a "${cache_dir}/build/." "${build_dir}/"
else
  echo "NCCL build cache miss: ${fingerprint}"
  # Do not let leftover objects or an incomplete cache affect a fresh build.
  rm -rf "${build_dir}" "${cache_dir}"
  make -C third_party/nccl -j"${BUILD_JOBS}" NVCC_GENCODE="${NVCC_GENCODE}" src.build
  test -s "${build_dir}/include/nccl.h"
  test -s "${build_dir}/lib/libnccl.so"
  test -s "${build_dir}/lib/libnccl_static.a"
  mkdir -p "${cache_dir}/build"
  # Preserve finished outputs while excluding the large intermediate objects.
  cp -a "${build_dir}/include" "${build_dir}/lib" "${cache_dir}/build/"
  if [[ -d "${build_dir}/bin" ]]; then
    cp -a "${build_dir}/bin" "${cache_dir}/build/"
  fi
  printf '%s\n' "${fingerprint}" > "${cache_dir}/fingerprint"
  publish_cache || echo "NCCL build cache: publish failed (non-fatal)"
  rm -rf "${scratch}"
fi
