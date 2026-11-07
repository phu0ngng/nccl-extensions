# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test bundled library selection without importing CUDA or loading a GPU library."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class BundledLibraryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bindings = types.SimpleNamespace(__version__="12.9.0")
        cuda = types.ModuleType("cuda")
        cuda.bindings = self.bindings
        source = Path(__file__).resolve().parents[2] / "nccl/_extensions/_runtime.py"
        spec = importlib.util.spec_from_file_location("runtime_under_test", source)
        self.runtime = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"cuda": cuda}):
            spec.loader.exec_module(self.runtime)
        self.runtime.__file__ = str(self.root / "nccl/_extensions/_runtime.py")

    def library(self, name, variant, filename):
        path = self.root / "nccl" / name / "lib" / variant / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    def test_soname_and_cuda_variant(self):
        for major in (12, 13):
            expected = self.library("ep", f"cu{major}", "libnccl_ep.so.0")
            self.bindings.__version__ = f"{major}.0.0"
            self.assertEqual(self.runtime.bundled_library("nccl_ep"), expected)

    def test_missing_variant_does_not_use_another_cuda(self):
        self.library("ep", "cu13", "libnccl_ep.so.0")
        self.assertIsNone(self.runtime.bundled_library("nccl_ep"))

    def test_m2n_keeps_unversioned_filename(self):
        expected = self.library("m2n", "cu12", "libnccl_m2n.so")
        self.assertEqual(self.runtime.bundled_library("nccl_m2n"), expected)


if __name__ == "__main__":
    unittest.main()
