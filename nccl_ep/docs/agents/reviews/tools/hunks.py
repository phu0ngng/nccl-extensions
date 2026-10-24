#!/usr/bin/env python3
"""Summarize a unified diff per file and hunk.

Usage: hunks.py <diff.patch> [--max-hunks N] > hunks.md

The output is the short form of the change for critics in confirm mode and for
triage: file, added/removed line counts, and one line per hunk with its header
context and the first changed line.
"""
import argparse
import re
import sys

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def parse(lines):
    files = []
    current = None
    hunk = None
    for line in lines:
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[1].rstrip("\n") if " b/" in line else line.rstrip("\n")
            current = {"path": path, "added": 0, "removed": 0, "hunks": [], "status": "modified"}
            files.append(current)
            hunk = None
            continue
        if current is None:
            continue
        if line.startswith("new file mode"):
            current["status"] = "added"
        elif line.startswith("deleted file mode"):
            current["status"] = "deleted"
        elif line.startswith("rename to "):
            current["status"] = "renamed"
        m = HUNK_RE.match(line)
        if m:
            hunk = {
                "new_start": int(m.group(3)),
                "new_len": int(m.group(4) or 1),
                "context": m.group(5).strip(),
                "first": "",
                "added": 0,
                "removed": 0,
            }
            current["hunks"].append(hunk)
            continue
        if hunk is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            hunk["added"] += 1
            current["added"] += 1
            if not hunk["first"]:
                hunk["first"] = line[1:].strip()
        elif line.startswith("-") and not line.startswith("---"):
            hunk["removed"] += 1
            current["removed"] += 1
            if not hunk["first"]:
                hunk["first"] = line[1:].strip()
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patch")
    ap.add_argument("--max-hunks", type=int, default=40, help="hunks listed per file")
    args = ap.parse_args()
    with open(args.patch, encoding="utf-8", errors="replace") as f:
        files = parse(f)
    total_add = sum(x["added"] for x in files)
    total_rm = sum(x["removed"] for x in files)
    print(f"# Change summary: {len(files)} files, +{total_add} -{total_rm}\n")
    print("| File | Status | + | - | Hunks |")
    print("|---|---|---|---|---|")
    for x in files:
        print(f"| `{x['path']}` | {x['status']} | {x['added']} | {x['removed']} | {len(x['hunks'])} |")
    print()
    for x in files:
        print(f"## `{x['path']}` ({x['status']}, +{x['added']} -{x['removed']})\n")
        for h in x["hunks"][: args.max_hunks]:
            end = h["new_start"] + max(h["new_len"] - 1, 0)
            ctx = f" `{h['context']}`" if h["context"] else ""
            first = h["first"].replace("|", "\\|")
            if len(first) > 110:
                first = first[:107] + "..."
            print(f"- L{h['new_start']}-{end} (+{h['added']} -{h['removed']}){ctx}: `{first}`")
        if len(x["hunks"]) > args.max_hunks:
            print(f"- ... {len(x['hunks']) - args.max_hunks} more hunks")
        print()


if __name__ == "__main__":
    sys.exit(main())
