"""Exercise cache decisions without CUDA or a GitLab runner."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


CI_DIR = Path(__file__).resolve().parents[1]


class BuildCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / ".gitlab/ci").mkdir(parents=True)
        for name in ("build-nccl.sh", "nccl-build.yml"):
            shutil.copyfile(CI_DIR / name, self.root / ".gitlab/ci" / name)
        self.bin = self.root / "tools"
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
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        CI_JOB_IMAGE="image-v1", NVCC_GENCODE="sm_100",
                        NVCC=str(self.bin / "compiler"),
                        CXX=str(self.bin / "compiler"))

    def tool(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/sh\nset -eu\n" + body + "\n")
        path.chmod(0o755)

    def run_build(self, **overrides):
        return subprocess.run(
            ["bash", ".gitlab/ci/build-nccl.sh"], cwd=self.root,
            env=dict(self.env, **overrides), capture_output=True, text=True,
        )

    def assert_build(self, message, **overrides):
        result = self.run_build(**overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(message, result.stdout)

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
        for override in (
            {"MOCK_COMMIT": "commit-b"}, {"CI_JOB_IMAGE": "image-v2"},
            {"NVCC_GENCODE": "sm_90"}, {"MOCK_COMPILER": "compiler-v2"},
            {"DEBUG": "1"},
            {"NVCC_APPEND_FLAGS": "--use_fast_math -lineinfo"},
            {"NVCC_PREPEND_FLAGS": "--use_fast_math -lineinfo"},
        ):
            with self.subTest(override=override):
                self.assert_build("cache miss", **override)
                self.assert_build("cache hit", **override)
                self.assert_build("cache miss")

    def test_cpu_request_controls_parallelism_without_invalidating_cache(self):
        self.assert_build("cache miss", KUBERNETES_CPU_REQUEST="3")
        self.assertIn("-j3", (self.root / "make-args").read_text().splitlines())
        self.assert_build("cache hit", KUBERNETES_CPU_REQUEST="4")

    def test_incomplete_cache_rebuilds(self):
        self.assert_build("cache miss")
        (self.root / ".ci/nccl-build-cache/build/lib/libnccl_static.a").unlink()
        self.assert_build("cache miss")

    def test_failed_build_does_not_publish_cache(self):
        self.assert_build("cache miss")
        self.assertNotEqual(self.run_build(DEBUG="1", MOCK_FAIL="1").returncode, 0)
        self.assertFalse((self.root / ".ci/nccl-build-cache/fingerprint").exists())
        self.assert_build("cache miss", DEBUG="1")

    def test_recipe_change_rebuilds(self):
        self.assert_build("cache miss")
        with (self.root / ".gitlab/ci/nccl-build.yml").open("a") as recipe:
            recipe.write("\n# changed recipe\n")
        self.assert_build("cache miss")

    def test_missing_compiler_fails_even_with_cache(self):
        self.assert_build("cache miss")
        self.assertNotEqual(self.run_build(NVCC="/missing/nvcc").returncode, 0)


if __name__ == "__main__":
    unittest.main()
