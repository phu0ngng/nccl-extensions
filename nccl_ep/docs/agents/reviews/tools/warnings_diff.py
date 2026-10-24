#!/usr/bin/env python3
"""Diff compiler warnings between two build logs.

Usage: warnings_diff.py <base.log> <head.log> > warnings_new.md

Warnings are normalized (build paths and line numbers removed) before the set
difference, so a warning that only moved does not count as new.
"""
import re
import sys

WARN_RE = re.compile(r"^(?P<file>[^:\s]+):(?P<line>\d+)(?::\d+)?:\s*warning:\s*(?P<msg>.*)$")
NVCC_WARN_RE = re.compile(r"^(?P<file>[^(\s]+)\((?P<line>\d+)\):\s*warning(?: #\d+-D)?:\s*(?P<msg>.*)$")
# nvcc flags the members GoogleTest's TEST/TEST_P macros generate for every test case as
# unreferenced; every new test adds one, so these carry no information about the change.
NOISE_RE = re.compile(r"::(gtest_registering_dummy_|test_info_)\" was declared but never referenced")


def collect(path):
    seen = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = WARN_RE.match(line) or NVCC_WARN_RE.match(line)
            if not m or NOISE_RE.search(m.group("msg")):
                continue
            file = re.sub(r"^.*?/(nccl_ep|nccl_m2n)/", r"\1/", m.group("file"))
            key = (file, m.group("msg").strip())
            seen.setdefault(key, []).append(int(m.group("line")))
    return seen


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base = collect(sys.argv[1])
    head = collect(sys.argv[2])
    new = sorted(k for k in head if k not in base)
    gone = sorted(k for k in base if k not in head)
    print(f"# Compiler warnings: base {sum(len(v) for v in base.values())}, head {sum(len(v) for v in head.values())}\n")
    print(f"## New in head ({len(new)})\n")
    for file, msg in new:
        lines = ", ".join(str(n) for n in sorted(set(head[(file, msg)]))[:6])
        print(f"- `{file}:{lines}`: {msg}")
    print(f"\n## Gone in head ({len(gone)})\n")
    for file, msg in gone:
        print(f"- `{file}`: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
