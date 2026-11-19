#!/usr/bin/env python3
"""Show how a kernel orders its memory loads and stores, from SASS with line info.

Usage:
  sass_ldst.py hot  <cubin | lineinfo.sass> [--spaces global,generic] [--top N]
  sass_ldst.py show <cubin | lineinfo.sass> (--src FILE[:FIRST[-LAST]] | --addr HEX-HEX)
                    [--spaces global,generic,local,shared] [--context]

The input is a cubin compiled with `-lineinfo` (the script runs `nvdisasm -c -g`
on it) or the saved text of that command.

`hot` ranks source lines by the number of load and store instructions attributed
to them. Inlined helpers are attributed to the helper's own line, not the call
site, so use it to find where a copy landed, then pass that file and line range
(or the address range) to `show`.

`show` prints the selected loads and stores in address order, one row per basic
block, and for each block the longest run of global or generic loads issued before a store:

  L128 S128                 run 1: every load is followed by the store that needs it
  L128 L128 L128 L128 S128 S128 S128 S128    run 4: four loads are in flight together

A store waits for the load that feeds it, so run 1 in a loop means one load
latency per element; run k means one load latency per k elements. The output is
static: it gives the order of issue, not the time. `?` marks a predicated
instruction (tail handling), `*` a block inside a loop (target of a backward
branch).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

FILE_RE = re.compile(r'//## File "([^"]+)", line (\d+)')
INSTR_RE = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
LABEL_RE = re.compile(r"^(\.L_x_\d+):")
FUNC_RE = re.compile(r"^\s*\.type\s+(\S+),@function")
MEM_RE = re.compile(r"^(?:@!?U?P\w+\s+)?((?:LD|ST)[GLS]?)((?:\.[A-Z0-9]+)*)\s")
BRANCH_RE = re.compile(r"^(?:@!?U?P\w+\s+)?(BRA|BRX|JMP|JMX|RET|EXIT|CALL\S*)\b(?:.*?`\((\.L_x_\d+)\))?")
SPACE_OF = {"": "generic", "G": "global", "L": "local", "S": "shared"}


def read_sass(path):
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"\x7fELF":
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    tool = shutil.which("nvdisasm") or os.path.join(cuda, "bin", "nvdisasm")
    proc = subprocess.run([tool, "-c", "-g", path], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.exit(f"nvdisasm failed on {path}: {proc.stderr.strip()}")
    return proc.stdout


def parse(text):
    """Return a list of instruction dicts in file order."""
    out = []
    func, src, line, pending = "", "", 0, []
    for raw in text.splitlines():
        m = FUNC_RE.match(raw)
        if m:
            func = m.group(1)
            continue
        m = FILE_RE.search(raw)
        if m:
            src, line = m.group(1), int(m.group(2))
            continue
        m = LABEL_RE.match(raw)
        if m:
            pending.append(m.group(1))
            continue
        m = INSTR_RE.match(raw)
        if not m:
            continue
        ins = {"addr": int(m.group(1), 16), "text": m.group(2), "func": func, "src": src, "line": line, "labels": pending}
        pending = []
        mem = MEM_RE.match(ins["text"] + " ")
        if mem:
            op, mods = mem.group(1), mem.group(2).split(".")[1:]
            ins["kind"] = "L" if op.startswith("LD") else "S"
            ins["space"] = SPACE_OF[op[2:]]
            width = [w for w in mods if w in ("128", "64", "U8", "S8", "U16", "S16")]
            ins["width"] = width[0].lstrip("US") if width else "32"
            ins["pred"] = ins["text"].startswith("@")
        br = BRANCH_RE.match(ins["text"])
        if br:
            ins["branch"] = br.group(1)
            ins["target"] = br.group(2)
        out.append(ins)
    return out


def loop_spans(instrs):
    """Address spans [target, branch] of backward branches, per function."""
    label_addr = {}
    for ins in instrs:
        for lab in ins["labels"]:
            label_addr[(ins["func"], lab)] = ins["addr"]
    spans = []
    for ins in instrs:
        tgt = label_addr.get((ins["func"], ins.get("target")))
        if tgt is not None and tgt <= ins["addr"]:
            spans.append((ins["func"], tgt, ins["addr"]))
    return spans


def src_matches(ins, name, first, last):
    if not ins["src"].endswith(name):
        return False
    return first is None or first <= ins["line"] <= last


def cmd_hot(args, instrs):
    spaces = set(args.spaces.split(","))
    counts = {}
    for ins in instrs:
        if ins.get("space") in spaces:
            key = (os.path.basename(ins["src"]), ins["line"])
            c = counts.setdefault(key, {"L": 0, "S": 0})
            c[ins["kind"]] += 1
    rows = sorted(counts.items(), key=lambda kv: -(kv[1]["L"] + kv[1]["S"]))[: args.top]
    print(f"{'loads':>6} {'stores':>6}  source line  (spaces: {args.spaces})")
    for (name, line), c in rows:
        print(f"{c['L']:>6} {c['S']:>6}  {name}:{line}")
    return 0


def cmd_show(args, instrs):
    spaces = set(args.spaces.split(","))
    if args.addr:
        lo, hi = (int(x, 16) for x in args.addr.split("-"))
        selected = lambda ins: lo <= ins["addr"] <= hi
    else:
        name, _, rng = args.src.partition(":")
        first = last = None
        if rng:
            a, _, b = rng.partition("-")
            first, last = int(a), int(b or a)
        selected = lambda ins: src_matches(ins, name, first, last)
    spans = loop_spans(instrs)

    blocks, cur = [], None
    for ins in instrs:
        if ins["labels"] or cur is None:
            cur = {"func": ins["func"], "addr": ins["addr"], "ops": [], "other": 0}
            blocks.append(cur)
        if selected(ins):
            if ins.get("space") in spaces:
                cur["ops"].append(ins)
            elif "kind" in ins:
                cur["other"] += 1
            if args.context:
                cur.setdefault("ctx", []).append(ins)
        if "branch" in ins:
            cur = None

    total = {"L": 0, "S": 0}
    local_ops, loop_runs, printed = 0, [], 0
    for b in blocks:
        if not b["ops"]:
            continue
        printed += 1
        in_loop = any(f == b["func"] and lo_ <= b["addr"] <= hi_ for f, lo_, hi_ in spans)
        seq, run, max_run, fed_store = [], 0, 0, False
        for ins in b["ops"]:
            total[ins["kind"]] += 1
            far = ins["space"] in ("global", "generic")
            local_ops += ins["space"] == "local"
            seq.append(ins["kind"] + ins["width"] + ("" if far else ins["space"][0]) + ("?" if ins["pred"] else ""))
            if not far:
                continue
            if ins["kind"] == "L":
                run += 1
                max_run = max(max_run, run)
            else:
                fed_store = fed_store or run > 0
                run = 0
        if in_loop and fed_store:
            loop_runs.append(max_run)
        extra = f"  (+{b['other']} ops in other spaces)" if b["other"] else ""
        print(f"{b['addr']:#07x}{'*' if in_loop else ' '} run {max_run:<3} {' '.join(seq)}{extra}")
        if args.context:
            for ins in b["ctx"]:
                print(f"            {ins['addr']:#07x}  {ins['text']}")
    if not printed:
        print("no matching loads or stores; was the cubin built with -lineinfo, and is the copy inlined from a helper (try `hot`)?")
        return 2
    print(f"\nloads {total['L']}, stores {total['S']} in {printed} blocks (spaces: {args.spaces})")
    if loop_runs:
        print(f"loop blocks where a global load is followed by a global store: {len(loop_runs)}, longest load run {min(loop_runs)} to {max(loop_runs)}")
        if max(loop_runs) == 1:
            print("=> interleaved: each element pays one load latency before its store can issue")
    if local_ops:
        print(f"{local_ops} local-memory ops (suffix l): a spill or a per-thread array indexed at run time is on this path")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hot")
    h.add_argument("input")
    h.add_argument("--spaces", default="global,generic")
    h.add_argument("--top", type=int, default=25)
    s = sub.add_parser("show")
    s.add_argument("input")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--src")
    g.add_argument("--addr")
    s.add_argument("--spaces", default="global,generic,local,shared")
    s.add_argument("--context", action="store_true", help="also print every selected instruction")
    args = p.parse_args()
    instrs = parse(read_sass(args.input))
    return cmd_hot(args, instrs) if args.cmd == "hot" else cmd_show(args, instrs)


if __name__ == "__main__":
    sys.exit(main())
