#!/usr/bin/env python3
"""Compile every JIT variant in a cache directory offline and tabulate ptxas output.

Usage:
  jit_variants.py compile <cache-dir> <out.tsv> [--arch sm_100a] [--source-dir DIR]
                  [--build-include-dir DIR] [--cuda-include-dir DIR] [--jobs N]
  jit_variants.py diff <base.tsv> <head.tsv> > jit_diff.md

`compile` walks `<cache-dir>/<key>/kernel.cu` (the layout the library's JIT
cache uses, filled by running tests or ep_bench with NCCL_EP_JIT_CACHE_DIR set
to a fresh directory) and runs nvcc with the same options the library's
compiler backend uses (device/jit/nvcc_compiler.cc), plus `-Xptxas -v`, then
`cuobjdump -sass` for an instruction count. Variant names are taken from
metadata.json when present, else from the kernel source's first __global__.

`diff` joins two tables on variant name and prints variants that are new,
gone, or changed in registers, spills, shared, stack or instruction count.
"""
import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

PTXAS_FUNC_RE = re.compile(r"ptxas info\s+: Compiling entry function '([^']+)' for '(sm_\d+a?)'")
PTXAS_REG_RE = re.compile(r"Used (\d+) registers(?:, used \d+ barriers)?(?:, (\d+) bytes smem)?(?:, (\d+) bytes cmem\[0\])?")
PTXAS_SPILL_RE = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
PTXAS_STACK_RE = re.compile(r"(\d+) bytes stack frame")
SASS_INSTR_RE = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/\s+\S")
GLOBAL_RE = re.compile(r"__global__[^;{]*?\b(\w+)\s*\(", re.S)

FIELDS = ["variant", "arch", "entry", "registers", "spill_stores", "spill_loads", "smem", "stack", "sass_instructions", "status"]


def nvcc_path():
    for env in ("NCCL_EP_JIT_NVCC", "NVCC"):
        if os.environ.get(env):
            return os.environ[env]
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    cand = os.path.join(cuda, "bin", "nvcc")
    return cand if os.path.exists(cand) else "nvcc"


def tool(name):
    cuda = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    cand = os.path.join(cuda, "bin", name)
    return cand if os.path.exists(cand) else name


def include_options(source_dir, build_include_dir, cuda_include_dir):
    # Mirrors include_options() in device/jit/nvcc_compiler.cc.
    opts = []
    if source_dir:
        opts += ["-I" + os.path.dirname(source_dir.rstrip("/")), "-I" + source_dir, "-I" + os.path.join(source_dir, "device")]
    if build_include_dir:
        opts += ["-I" + build_include_dir, "-I" + os.path.join(build_include_dir, "nccl_device")]
    if cuda_include_dir:
        opts += ["-I" + cuda_include_dir, "-I" + os.path.join(cuda_include_dir, "cccl")]
    return opts


def variant_name(variant_dir, source):
    meta = os.path.join(variant_dir, "metadata.json")
    if os.path.exists(meta):
        try:
            with open(meta, encoding="utf-8") as f:
                data = json.load(f)
            for key in ("variant_name", "variant", "name"):
                if isinstance(data.get(key), str):
                    return data[key]
        except (OSError, ValueError):
            pass
    m = GLOBAL_RE.search(source)
    if m:
        return m.group(1)
    return os.path.basename(variant_dir)


def compile_one(variant_dir, args, opts):
    src = os.path.join(variant_dir, "kernel.cu")
    with open(src, encoding="utf-8", errors="replace") as f:
        source = f.read()
    name = variant_name(variant_dir, source)
    row = {k: "" for k in FIELDS}
    row["variant"] = name
    with tempfile.TemporaryDirectory() as tmp:
        cubin = os.path.join(tmp, "kernel.cubin")
        cmd = [nvcc_path(), "--cubin", src, "-o", cubin, "-Xptxas", "-v"] + opts
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        log = proc.stdout + proc.stderr
        with open(os.path.join(variant_dir, "review_ptxas.log"), "w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n" + log)
        if proc.returncode != 0:
            row["status"] = "compile_failed"
            return [row]
        rows = []
        entry, arch = None, None
        cur = None
        for line in log.splitlines():
            m = PTXAS_FUNC_RE.search(line)
            if m:
                entry, arch = m.group(1), m.group(2)
                cur = dict(row, entry=entry, arch=arch, status="ok")
                rows.append(cur)
                continue
            if cur is None:
                continue
            m = PTXAS_REG_RE.search(line)
            if m:
                cur["registers"] = m.group(1)
                cur["smem"] = m.group(2) or "0"
            m = PTXAS_SPILL_RE.search(line)
            if m:
                cur["spill_stores"], cur["spill_loads"] = m.group(1), m.group(2)
            m = PTXAS_STACK_RE.search(line)
            if m:
                cur["stack"] = m.group(1)
        sass = subprocess.run([tool("cuobjdump"), "-sass", cubin], capture_output=True, text=True, check=False).stdout
        count = sum(1 for line in sass.splitlines() if SASS_INSTR_RE.match(line))
        if not rows:
            rows.append(dict(row, status="ok_no_ptxas_info"))
        if len(rows) == 1:
            rows[0]["sass_instructions"] = str(count)
        else:
            per_entry = {}
            current = None
            for line in sass.splitlines():
                m = re.search(r"Function : (\S+)", line)
                if m:
                    current = m.group(1)
                    per_entry[current] = 0
                elif current and SASS_INSTR_RE.match(line):
                    per_entry[current] += 1
            for r in rows:
                r["sass_instructions"] = str(per_entry.get(r["entry"], ""))
        return rows


def cmd_compile(args):
    arch = args.arch or os.environ.get("REVIEW_ARCH") or "sm_90"
    opts = ["--std=c++17", "--extended-lambda", "--expt-relaxed-constexpr", "-arch=" + arch]
    opts += include_options(
        args.source_dir or os.environ.get("NCCL_EP_JIT_SOURCE_DIR", ""),
        args.build_include_dir or os.environ.get("NCCL_EP_JIT_BUILD_INCLUDE_DIR", ""),
        args.cuda_include_dir or os.environ.get("NCCL_EP_JIT_CUDA_INCLUDE_DIR", ""),
    )
    opts += os.environ.get("NVCC_EXTRA_FLAGS", "").split()
    dirs = sorted(
        os.path.join(args.cache_dir, d)
        for d in os.listdir(args.cache_dir)
        if os.path.exists(os.path.join(args.cache_dir, d, "kernel.cu"))
    )
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for result in ex.map(lambda d: compile_one(d, args, opts), dirs):
            rows.extend(result)
    rows.sort(key=lambda r: (r["variant"], r["entry"]))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\t".join(FIELDS) + "\n")
        for r in rows:
            f.write("\t".join(r[k] for k in FIELDS) + "\n")
    failed = [r for r in rows if r["status"] == "compile_failed"]
    print(f"{len(dirs)} variants, {len(rows)} entries, {len(failed)} compile failures -> {args.out}")
    for r in failed:
        print("  FAILED:", r["variant"])
    return 1 if failed else 0


def read_tsv(path):
    table = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            table[(row["variant"], row.get("entry", ""))] = row
    return table


def cmd_diff(args):
    base = read_tsv(args.base)
    head = read_tsv(args.head)
    fields = ["registers", "spill_stores", "spill_loads", "smem", "stack", "sass_instructions", "status"]
    keys = sorted(set(base) | set(head))
    print(f"# JIT variants: {len(base)} base, {len(head)} head\n")
    failed = [k for k in head if head[k]["status"] == "compile_failed"]
    print(f"## Compile failures in head ({len(failed)})\n")
    for k in failed:
        print(f"- `{k[0]}`")
    print("\n## Changed\n")
    print("| Variant | Field | Base | Head |")
    print("|---|---|---|---|")
    for k in keys:
        if k in base and k in head:
            for f in fields:
                if base[k].get(f, "") != head[k].get(f, ""):
                    print(f"| `{k[0]}` | {f} | {base[k].get(f, '')} | {head[k].get(f, '')} |")
    print("\n## Only in head\n")
    print("| Variant | Registers | Spills (st/ld) | Smem | Stack | SASS instr |")
    print("|---|---|---|---|---|---|")
    for k in keys:
        if k not in base:
            r = head[k]
            print(f"| `{k[0]}` | {r['registers']} | {r['spill_stores']}/{r['spill_loads']} | {r['smem']} | {r['stack']} | {r['sass_instructions']} |")
    print("\n## Only in base\n")
    for k in keys:
        if k not in head:
            print(f"- `{k[0]}`")
    print("\n## Unchanged\n")
    same = [k for k in keys if k in base and k in head and all(base[k].get(f, "") == head[k].get(f, "") for f in fields)]
    print(f"{len(same)} variants identical in every field.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile")
    c.add_argument("cache_dir")
    c.add_argument("out")
    c.add_argument("--arch")
    c.add_argument("--source-dir")
    c.add_argument("--build-include-dir")
    c.add_argument("--cuda-include-dir")
    c.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    c.set_defaults(func=cmd_compile)
    d = sub.add_parser("diff")
    d.add_argument("base")
    d.add_argument("head")
    d.set_defaults(func=cmd_diff)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
