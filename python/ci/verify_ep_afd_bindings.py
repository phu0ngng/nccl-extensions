#!/usr/bin/env python3

"""Verify that the installed wheel exposes the complete AFD API surface."""

from __future__ import annotations

import ctypes

import nccl.ep as ep
from nccl._extensions.bindings import nccl_ep as low_level


LOW_LEVEL_FUNCTIONS = ("af_dispatch", "af_recv", "af_combine", "af_return")
HANDLE_METHODS = LOW_LEVEL_FUNCTIONS
C_SYMBOLS = (
    "ncclEpAfDispatch",
    "ncclEpAfRecv",
    "ncclEpAfCombine",
    "ncclEpAfReturn",
)


def main() -> None:
    # Force the generated loader to resolve the packaged NCCL EP library.
    ep.get_lib_version()
    library_path = ep.get_lib_path()
    if library_path is None:
        raise RuntimeError("could not determine the loaded NCCL EP library path")

    for name in LOW_LEVEL_FUNCTIONS:
        if not callable(getattr(low_level, name, None)):
            raise RuntimeError(f"low-level binding is missing {name}")
    for name in HANDLE_METHODS:
        if not callable(getattr(ep.Handle, name, None)):
            raise RuntimeError(f"nccl.ep.Handle is missing {name}")

    library = ctypes.CDLL(str(library_path))
    for name in C_SYMBOLS:
        if getattr(library, name, None) is None:
            raise RuntimeError(f"NCCL EP library is missing {name}")

    print("AFD Python bindings: PASSED")


if __name__ == "__main__":
    main()
