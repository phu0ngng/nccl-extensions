# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only regression for SONAME lookup from an unpacked wheel."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile


@unittest.skipUnless(shutil.which("cc") and shutil.which("readelf"),
                     "requires a host C compiler and readelf")
class WheelSonameTest(unittest.TestCase):
    def test_wheel_consumer_resolves_soname(self):
        # Use the production package-data patterns with a tiny native fixture;
        # neither CUDA nor NCCL is needed to exercise the ELF loader contract.
        from setuptools import Distribution
        from setuptools.command.bdist_wheel import bdist_wheel
        from setuptools.config.pyprojecttoml import load_file

        config = load_file(Path(__file__).resolve().parents[2] / "pyproject.toml")
        patterns = config["tool"]["setuptools"]["package-data"]["nccl.ep"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ep.c"
            source.write_text("int ep_test(void) { return 42; }\n")
            build = root / "build"
            (build / "lib").mkdir(parents=True)
            library = build / "lib/libnccl_ep.so.0.3.0"
            # Mirror the Makefile/CMake link chain for the bundled ABI major.
            subprocess.run(["cc", "-shared", "-fPIC", "-Wl,-soname,libnccl_ep.so.0",
                            str(source), "-o", str(library)], check=True)
            (build / "lib/libnccl_ep.so.0").symlink_to(library.name)
            (build / "lib/libnccl_ep.so").symlink_to("libnccl_ep.so.0")
            package = root / "nccl/ep"
            repo = Path(__file__).resolve().parents[3]
            staging_env = dict(os.environ, CI_PROJECT_DIR=str(repo), CUDA_HOME=str(root),
                               EP_CU12_DIR=str(build), EP_CU13_DIR=str(build),
                               M2N_CU12_DIR=str(build), M2N_CU13_DIR=str(build))
            for variant in ("cu12", "cu13"):
                directory = package / "lib" / variant
                subprocess.run(
                    ["bash", "-c", 'source "$1"; stage_library "$2" "$3" "$4" libnccl_ep.so',
                     "stage-wheel-fixture", str(repo / "python/ci/build_ci_wheel.sh"),
                     str(build), str(package), variant], check=True, env=staging_env,
                )
                self.assertEqual([p.name for p in directory.iterdir()], ["libnccl_ep.so.0"])
                self.assertFalse((package / "lib" / variant / "libnccl_ep.so.0").is_symlink())
            (package / "__init__.py").write_text("")
            previous = Path.cwd()
            try:
                os.chdir(root)
                dist = Distribution(dict(name="soname-fixture", version="0",
                                         packages=["nccl.ep"],
                                         package_data={"nccl.ep": patterns}))
                dist.script_name = "setup.py"
                command = bdist_wheel(dist)
                command.ensure_finalized()
                command.run()
            finally:
                os.chdir(previous)
            unpacked = root / "unpacked"
            with zipfile.ZipFile(next((root / "dist").glob("*.whl"))) as wheel:
                self.assertEqual(sorted(n for n in wheel.namelist() if "/lib/" in n),
                                 [f"nccl/ep/lib/{v}/libnccl_ep.so.0" for v in ("cu12", "cu13")])
                wheel.extractall(unpacked)
            consumer = root / "consumer.c"
            consumer.write_text("int ep_test(void); int main(void) { return ep_test() != 42; }\n")
            env = {k: v for k, v in os.environ.items()
                   if k not in {"LD_LIBRARY_PATH", "LD_PRELOAD"}}
            for variant in ("cu12", "cu13"):
                libdir = unpacked / "nccl/ep/lib" / variant
                binary = root / "consumer"
                subprocess.run(["cc", str(consumer), f"-L{libdir}", "-l:libnccl_ep.so.0",
                                f"-Wl,-rpath,{libdir}", "-o", str(binary)], check=True)
                dynamic = subprocess.check_output(["readelf", "-d", str(binary)], text=True)
                self.assertIn("[libnccl_ep.so.0]", dynamic)
                subprocess.run([str(binary)], check=True, env=env)
                (libdir / "libnccl_ep.so.0").unlink()
                result = subprocess.run([str(binary)], env=env, capture_output=True)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
