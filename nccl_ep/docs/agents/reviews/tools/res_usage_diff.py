#!/usr/bin/env python3
"""Diff per-kernel resource usage of two shared libraries.

Usage: res_usage_diff.py <base/libnccl_ep.so> <head/libnccl_ep.so> > res_usage_diff.md

Runs `cuobjdump -res-usage` on both, demangles kernel names with c++filt, and
prints every kernel whose registers, spills, shared, constant or stack usage
changed, then kernels only in one side. Kernels in the JIT path are not in the
library; see jit_variants.py.
"""
import re
import shutil
import subprocess
import sys

FUNC_RE = re.compile(r"^\s*Function\s+(\S+):")
RES_RE = re.compile(r"^\s*(REG|STACK|SHARED|LOCAL|CONSTANT\[\d+\]|TEXTURE|SURFACE|SAMPLER):(\d+)")
ARCH_RE = re.compile(r"arch\s*=\s*(sm_\d+a?)")


def cuobjdump():
    for cand in ("cuobjdump", "/usr/local/cuda/bin/cuobjdump"):
        if shutil.which(cand) or cand.startswith("/"):
            return cand
    return "cuobjdump"


def parse(path):
    out = subprocess.run([cuobjdump(), "-res-usage", path], capture_output=True, text=True, check=False).stdout
    kernels = {}
    arch = "?"
    func = None
    for line in out.splitlines():
        m = ARCH_RE.search(line)
        if m:
            arch = m.group(1)
        m = FUNC_RE.match(line)
        if m:
            func = (arch, m.group(1))
            kernels[func] = {}
            continue
        if func is None:
            continue
        for part in line.split():
            m = RES_RE.match(part)
            if m:
                kernels[func][m.group(1)] = int(m.group(2))
    return kernels


def demangle(names):
    if not names or not shutil.which("c++filt"):
        return {n: n for n in names}
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=False).stdout
    pretty = out.splitlines()
    return {n: (p if p else n) for n, p in zip(names, pretty + [""] * (len(names) - len(pretty)))}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base = parse(sys.argv[1])
    head = parse(sys.argv[2])
    names = demangle(sorted({k[1] for k in list(base) + list(head)}))
    fields = ("REG", "STACK", "SHARED", "LOCAL", "CONSTANT[0]")
    changed, only_base, only_head = [], [], []
    for key in sorted(set(base) | set(head)):
        if key not in base:
            only_head.append(key)
        elif key not in head:
            only_base.append(key)
        else:
            diffs = [(f, base[key].get(f, 0), head[key].get(f, 0)) for f in fields if base[key].get(f, 0) != head[key].get(f, 0)]
            if diffs:
                changed.append((key, diffs))
    print(f"# Static kernel resource usage: {len(base)} base, {len(head)} head kernels\n")
    print(f"## Changed ({len(changed)})\n")
    print("| Arch | Kernel | Field | Base | Head |")
    print("|---|---|---|---|---|")
    for (arch, name), diffs in changed:
        for f, b, h in diffs:
            print(f"| {arch} | `{names[name][:120]}` | {f} | {b} | {h} |")
    print(f"\n## Only in head ({len(only_head)})\n")
    for arch, name in only_head:
        r = head[(arch, name)]
        print(f"- {arch} `{names[name][:120]}`: REG {r.get('REG', 0)}, STACK {r.get('STACK', 0)}, SHARED {r.get('SHARED', 0)}, LOCAL {r.get('LOCAL', 0)}")
    print(f"\n## Only in base ({len(only_base)})\n")
    for arch, name in only_base:
        print(f"- {arch} `{names[name][:120]}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
