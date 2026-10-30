"""Exercise build_lib.py's build-directory handling without CUDA or a compiler."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "build_lib.py"

# Stands in for make: records the call and lays out what the real build installs
# under BUILDDIR for the component named by -C.
MAKE_SHIM = r"""
printf '%s\n' "$*" >> "${MOCK_MAKE_LOG}"
src=; build=
while [ $# -gt 0 ]; do
  case "$1" in
    -C) src=$2; shift ;;
    BUILDDIR=*) build=${1#BUILDDIR=} ;;
  esac
  shift
done
case "${src##*/}" in
  nccl_ep)
    mkdir -p "${build}/lib" "${build}/include/nccl_ep/device/ll" "${build}/obj/nccl_ep"
    echo so > "${build}/lib/libnccl_ep.so.0.3.0"
    ln -sf libnccl_ep.so.0.3.0 "${build}/lib/libnccl_ep.so.0"
    ln -sf libnccl_ep.so.0 "${build}/lib/libnccl_ep.so"
    echo a > "${build}/lib/libnccl_ep.a"
    echo h > "${build}/include/nccl_ep.h"
    echo h > "${build}/include/nccl_ep/ep_enums.h"
    echo h > "${build}/include/nccl_ep/common.hpp"
    echo h > "${build}/include/nccl_ep/device/ht_ep.cuh"
    echo h > "${build}/include/nccl_ep/device/ll/ll_common.cuh"
    echo o > "${build}/obj/nccl_ep/nccl_ep.o" ;;
  nccl_m2n)
    mkdir -p "${build}/lib" "${build}/include" "${build}/obj/nccl_m2n"
    echo so > "${build}/lib/libnccl_m2n.so"
    echo h > "${build}/include/nccl_m2n.h"
    echo o > "${build}/obj/nccl_m2n/m2n_init.o" ;;
  *) echo "unexpected make -C ${src}" >&2; exit 1 ;;
esac
"""

# build_lib.py's only third-party import; this stand-in is used only where the
# packaging distribution is not installed for the test interpreter.
PACKAGING_STUB = """
class Version:
    def __init__(self, text):
        self.text = str(text)

    def __eq__(self, other):
        return isinstance(other, Version) and self.text == other.text

    def __hash__(self):
        return hash(self.text)

    def __str__(self):
        return self.text
"""


class BuildDirTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.bin = base / "tools"
        self.tool(self.bin / "make", MAKE_SHIM)
        self.cuda = base / "cuda"
        self.tool(self.cuda / "bin" / "nvcc",
                  'echo "Cuda compilation tools, release 12.9, V12.9.86"')
        nccl_home = base / "nccl"
        nccl_home.mkdir()
        # Where the script's temporary directories go, so leftovers are visible.
        self.tmpdir = base / "tmp"
        self.tmpdir.mkdir()
        self.output = base / "output"
        self.make_log = base / "make.log"
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}",
                        NCCL_HOME=str(nccl_home), TMPDIR=str(self.tmpdir),
                        MOCK_MAKE_LOG=str(self.make_log),
                        PYTHONDONTWRITEBYTECODE="1")
        self.env.pop("CUDA_PATH", None)
        if importlib.util.find_spec("packaging") is None:
            stub = base / "pystubs" / "packaging"
            stub.mkdir(parents=True)
            (stub / "__init__.py").write_text("")
            (stub / "version.py").write_text(PACKAGING_STUB)
            self.env["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(stub.parent), os.environ.get("PYTHONPATH"))
                if part)

    def tool(self, path, body):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nset -eu\n" + body + "\n")
        path.chmod(0o755)

    def run_build(self, *args):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args,
             "--cuda-home", str(self.cuda), "--output-dir", str(self.output)],
            env=self.env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def build_dirs_passed(self):
        """BUILDDIR handed to make, one entry per make call."""
        if not self.make_log.exists():
            return []
        return [arg[len("BUILDDIR="):]
                for call in self.make_log.read_text().splitlines()
                for arg in call.split() if arg.startswith("BUILDDIR=")]

    def test_default_build_is_temporary(self):
        self.run_build("nccl_ep")
        (build_dir,) = self.build_dirs_passed()
        self.assertTrue(build_dir.startswith(str(self.tmpdir)), build_dir)
        self.assertFalse(Path(build_dir).exists())
        self.assertEqual(list(self.tmpdir.iterdir()), [])
        self.assertTrue((self.output / "nccl/ep/lib/cu12/libnccl_ep.so").is_file())
        self.assertTrue(
            (self.output / "nccl/ep/include/nccl_ep/device/ll/ll_common.cuh").is_file())

    def test_build_dir_is_kept_and_reused(self):
        build = Path(self.temp.name) / "kept" / "build"
        self.run_build("nccl_ep", "--build-dir", str(build))
        self.assertEqual(self.build_dirs_passed(), [str(build.resolve())])
        for relative in ("lib/libnccl_ep.so", "lib/libnccl_ep.a", "include/nccl_ep.h",
                         "include/nccl_ep/ep_enums.h", "obj/nccl_ep/nccl_ep.o"):
            self.assertTrue((build / relative).exists(), relative)
        self.assertTrue((self.output / "nccl/ep/lib/cu12/libnccl_ep.so").is_file())
        self.assertTrue((self.output / "nccl/ep/include/nccl_ep.h").is_file())
        # The staging temporary directory is still cleaned up in this mode.
        self.assertEqual(list(self.tmpdir.iterdir()), [])

        leftover = build / "obj" / "nccl_ep" / "nccl_ep.d"
        leftover.write_text("kept")
        self.run_build("nccl_ep", "--build-dir", str(build))
        self.assertEqual(self.build_dirs_passed(), [str(build.resolve())] * 2)
        self.assertEqual(leftover.read_text(), "kept")

    def test_build_dir_is_shared_by_all_targets(self):
        build = Path(self.temp.name) / "build"
        self.run_build("nccl_ep", "nccl_m2n", "--build-dir", str(build))
        self.assertEqual(self.build_dirs_passed(), [str(build.resolve())] * 2)
        self.assertTrue((build / "lib/libnccl_ep.so").exists())
        self.assertTrue((build / "lib/libnccl_m2n.so").is_file())
        self.assertTrue((self.output / "nccl/m2n/lib/cu12/libnccl_m2n.so").is_file())
        self.assertTrue((self.output / "nccl/m2n/include/nccl_m2n.h").is_file())


if __name__ == "__main__":
    unittest.main()
