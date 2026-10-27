"""Exercise cache decisions without CUDA, a GitLab runner, or a package registry."""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest


CI_DIR = Path(__file__).resolve().parents[1]

# Fake generic package registry: one directory per version (the URL's second-to-last
# segment), so every published version stays retrievable as in the real registry.
CURL_SHIM = """
printf '%s\\n' "$*" >> "${MOCK_CURL_LOG}"
[ "${MOCK_CURL_EXIT:-0}" = 0 ] || exit "${MOCK_CURL_EXIT}"
dest=; src=
while [ $# -gt 1 ]; do
  case "$1" in
    -o) dest=$2; shift ;;
    --upload-file) src=$2; shift ;;
    -H|--retry|--connect-timeout|--max-time) shift ;;
  esac
  shift
done
name=${1##*/}
version=${1%/*}; version=${version##*/}
if [ -n "${src}" ]; then
  [ "${MOCK_CURL_UPLOAD_EXIT:-0}" = 0 ] || exit "${MOCK_CURL_UPLOAD_EXIT}"
  mkdir -p "${MOCK_REGISTRY}/${version}" && cp "${src}" "${MOCK_REGISTRY}/${version}/${name}"
else
  [ -f "${MOCK_REGISTRY}/${version}/${name}" ] || exit 22
  cp "${MOCK_REGISTRY}/${version}/${name}" "${dest}"
fi
"""


class BuildCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.registry = base / "registry"
        self.registry.mkdir()
        self.curl_log = base / "curl.log"
        self.bin = base / "tools"
        self.bin.mkdir()
        self.tool("git", 'echo "${MOCK_COMMIT:-commit-a}"')
        self.tool("nproc", "echo 2")
        self.tool("compiler", 'echo "${MOCK_COMPILER:-compiler-v1}"')
        self.tool("make", """
printf '%s\\n' "$@" > make-args
echo build >> builds
if [ "${MOCK_FAIL:-0}" = 1 ]; then exit 1; fi
mkdir -p third_party/nccl/build/include third_party/nccl/build/lib \
  third_party/nccl/build/obj third_party/nccl/build/bin
echo header > third_party/nccl/build/include/nccl.h
echo device > third_party/nccl/build/include/nccl_device.h
echo library > third_party/nccl/build/lib/libnccl.so.2.30.7
ln -sf libnccl.so.2.30.7 third_party/nccl/build/lib/libnccl.so
echo static > third_party/nccl/build/lib/libnccl_static.a
echo object > third_party/nccl/build/obj/unused.o
echo binary > third_party/nccl/build/bin/ncclras
""")
        self.tool("curl", CURL_SHIM)
        self.root = self.checkout("checkout")
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        CI_JOB_IMAGE="image-v1", NVCC_GENCODE="sm_100",
                        NVCC=str(self.bin / "compiler"),
                        CXX=str(self.bin / "compiler"),
                        CI_API_V4_URL="https://gitlab.example/api/v4",
                        CI_PROJECT_ID="42", CI_JOB_TOKEN="job-token",
                        CI_PIPELINE_ID="7", CI_PIPELINE_URL="https://gitlab.example/p/7",
                        CI_COMMIT_SHA="abc123", CI_COMMIT_REF_NAME="main",
                        MOCK_REGISTRY=str(self.registry),
                        MOCK_CURL_LOG=str(self.curl_log))

    def tool(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/sh\nset -eu\n" + body + "\n")
        path.chmod(0o755)

    def checkout(self, name):
        root = Path(self.temp.name) / name
        (root / ".gitlab/ci").mkdir(parents=True)
        for item in ("build-nccl.sh", "nccl-build.yml"):
            shutil.copyfile(CI_DIR / item, root / ".gitlab/ci" / item)
        return root

    def run_build(self, root=None, **overrides):
        env = {key: value for key, value in dict(self.env, **overrides).items()
               if value is not None}
        return subprocess.run(
            ["bash", ".gitlab/ci/build-nccl.sh"], cwd=root or self.root,
            env=env, capture_output=True, text=True,
        )

    def assert_build(self, message, **overrides):
        result = self.run_build(**overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(message, result.stdout)
        return result

    def curl_calls(self):
        if not self.curl_log.exists():
            return []
        return self.curl_log.read_text().splitlines()

    def registry_files(self):
        """Published file names across every version directory."""
        return sorted(path.name for path in self.registry.rglob("*") if path.is_file())

    def package_dir(self):
        """The single published version; callers expect exactly one."""
        (version_dir,) = self.registry.iterdir()
        return version_dir

    def drop_from_registry(self, member):
        """Re-pack the published tarball without one member, checksum kept valid."""
        package = self.package_dir()
        path = package / "nccl.tar.gz"
        with tarfile.open(path) as tar:
            members = [item for item in tar.getmembers() if item.name != member]
            data = {item.name: tar.extractfile(item).read()
                    for item in members if item.isfile()}
        with tarfile.open(path, "w:gz") as tar:
            for item in members:
                tar.addfile(item, io.BytesIO(data[item.name]) if item.isfile() else None)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (package / "nccl.tar.gz.sha256").write_text(f"{digest}  nccl.tar.gz\n")

    def test_hit_restores_outputs_without_compiling(self):
        self.assert_build("cache miss")
        shutil.rmtree(self.root / "third_party/nccl/build")
        self.assert_build("cache hit")
        build = self.root / "third_party/nccl/build"
        self.assertTrue((build / "lib/libnccl.so").is_symlink())
        self.assertTrue((build / "lib/libnccl.so").is_file())
        self.assertTrue((build / "include/nccl_device.h").is_file())
        self.assertTrue((build / "bin/ncclras").is_file())
        self.assertFalse((build / "obj").exists())
        self.assertEqual((self.root / "builds").read_text().splitlines(), ["build"])

    def test_changed_inputs_rebuild(self):
        self.assert_build("cache miss")
        overrides = (
            {"MOCK_COMMIT": "commit-b"}, {"CI_JOB_IMAGE": "image-v2"},
            {"NVCC_GENCODE": "sm_90"}, {"MOCK_COMPILER": "compiler-v2"},
            {"DEBUG": "1"},
            {"NVCC_APPEND_FLAGS": "--use_fast_math -lineinfo"},
            {"NVCC_PREPEND_FLAGS": "--use_fast_math -lineinfo"},
        )
        for override in overrides:
            with self.subTest(override=override):
                self.assert_build("cache miss", **override)
                self.assert_build("cache hit", **override)
                # The first version is still in the registry, so going back is a hit.
                self.assert_build("cache hit")
        # One compile for the first inputs and one per change; the hits compiled nothing.
        self.assertEqual(len((self.root / "builds").read_text().splitlines()),
                         1 + len(overrides))

    def test_cpu_request_controls_parallelism_without_invalidating_cache(self):
        self.assert_build("cache miss", KUBERNETES_CPU_REQUEST="3")
        self.assertIn("-j3", (self.root / "make-args").read_text().splitlines())
        self.assert_build("cache hit", KUBERNETES_CPU_REQUEST="4")

    def test_incomplete_cache_rebuilds(self):
        self.assert_build("cache miss")
        self.drop_from_registry("build/lib/libnccl_static.a")
        self.assert_build("cache miss")

    def test_failed_build_does_not_publish_cache(self):
        self.assert_build("cache miss")
        published = (self.package_dir() / "metadata.json").read_text()
        self.assertNotEqual(self.run_build(DEBUG="1", MOCK_FAIL="1").returncode, 0)
        self.assertFalse((self.root / ".ci/nccl-build-cache/fingerprint").exists())
        self.assertEqual((self.package_dir() / "metadata.json").read_text(), published)
        self.assert_build("cache miss", DEBUG="1")

    def test_recipe_change_rebuilds(self):
        self.assert_build("cache miss")
        with (self.root / ".gitlab/ci/nccl-build.yml").open("a") as recipe:
            recipe.write("\n# changed recipe\n")
        self.assert_build("cache miss")

    def test_missing_compiler_fails_even_with_cache(self):
        self.assert_build("cache miss")
        self.assertNotEqual(self.run_build(NVCC="/missing/nvcc").returncode, 0)

    def test_registry_round_trip_across_checkouts(self):
        result = self.assert_build("cache miss")
        fingerprint = re.search(r"cache miss: (\w+)", result.stdout).group(1)
        arch = subprocess.check_output(["uname", "-m"], text=True).strip()
        version = f"commit-a-{arch}-{fingerprint[:16]}"
        self.assertIn(f"cache published: {version} (", result.stdout)
        self.assertEqual(self.registry_files(),
                         ["metadata.json", "nccl.tar.gz", "nccl.tar.gz.sha256"])
        self.assertFalse((self.root / ".ci/nccl-build-cache-xfer").exists())
        metadata = json.loads((self.package_dir() / "metadata.json").read_text())
        self.assertEqual(metadata["fingerprint"], fingerprint)
        self.assertEqual(metadata["version"], version)
        # The upload URL carried the computed key: the registry directory is that version.
        self.assertEqual(self.package_dir().name, metadata["version"])
        self.assertEqual(metadata["nccl_commit"], "commit-a")
        self.assertEqual(metadata["arch"], arch)
        self.assertEqual(metadata["image"], "image-v1")
        self.assertEqual(metadata["nvcc_gencode"], "sm_100")
        self.assertEqual(metadata["pipeline_id"], "7")
        self.assertEqual(metadata["ref"], "main")
        self.assertRegex(metadata["created_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        with tarfile.open(self.package_dir() / "nccl.tar.gz") as tar:
            names = tar.getnames()
        self.assertIn("fingerprint", names)
        self.assertIn("build/lib/libnccl_static.a", names)
        self.assertNotIn("build/obj", names)
        calls_before = len(self.curl_calls())

        other = self.checkout("checkout-2")
        result = self.run_build(root=other)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"cache hit: {fingerprint} (registry {version})", result.stdout)
        self.assertNotIn("cache published", result.stdout)
        self.assertFalse((other / "builds").exists())
        self.assertTrue((other / "third_party/nccl/build/lib/libnccl.so").is_file())
        new_calls = self.curl_calls()[calls_before:]
        self.assertEqual(len(new_calls), 2)
        self.assertFalse(any("--upload-file" in call for call in new_calls))

    def test_unreachable_registry_builds(self):
        result = self.assert_build("cache miss", MOCK_CURL_EXIT="7")
        self.assertIn("registry restore failed (non-fatal)", result.stdout)
        self.assertIn("publish failed (non-fatal)", result.stdout)
        self.assertEqual((self.root / "builds").read_text().splitlines(), ["build"])
        self.assertEqual(self.registry_files(), [])
        self.assertFalse((self.root / ".ci/nccl-build-cache-xfer").exists())

    def test_failed_publish_keeps_build(self):
        result = self.assert_build("cache miss", MOCK_CURL_UPLOAD_EXIT="22")
        self.assertIn("publish failed (non-fatal)", result.stdout)
        self.assertNotIn("cache published", result.stdout)
        self.assertTrue((self.root / "third_party/nccl/build/lib/libnccl.so").is_file())
        self.assertTrue((self.root / ".ci/nccl-build-cache/fingerprint").is_file())
        self.assertEqual(self.registry_files(), [])

    def test_cache_off_never_contacts_registry(self):
        self.assert_build("cache miss", NCCL_CACHE="0", CI_API_V4_URL=None,
                          CI_PROJECT_ID=None, CI_JOB_TOKEN=None)
        self.assertEqual(self.curl_calls(), [])
        self.assertEqual((self.root / "builds").read_text().splitlines(), ["build"])
        self.assertEqual(self.registry_files(), [])

    def test_checksum_mismatch_rebuilds(self):
        self.assert_build("cache miss")
        (self.package_dir() / "nccl.tar.gz.sha256").write_text("0" * 64 + "  nccl.tar.gz\n")
        shutil.rmtree(self.root / "third_party/nccl/build")
        result = self.assert_build("cache miss")
        self.assertIn("registry restore failed (non-fatal)", result.stdout)
        self.assertEqual((self.root / "builds").read_text().splitlines(), ["build", "build"])
        subprocess.run(["sha256sum", "-c", "--quiet", "nccl.tar.gz.sha256"],
                       cwd=self.package_dir(), check=True)


if __name__ == "__main__":
    unittest.main()
