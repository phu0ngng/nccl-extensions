#!/usr/bin/env python3
"""Generate the header-derived glue that jit_enumerate.cu includes.

Usage: jit_enumerate_gen.py <source-tree> <out.inc>

jit_enumerate.cu is a fixed matrix of configurations. What used to drift with
the code is everything it hard-coded about the library: the argument list of
the JIT source generators, the argument list of the kernel selectors, and the
set of kernel enumerators with their names. This script reads those from the
tree's headers, so the driver follows the code:

  * Functions (ll_dispatch_jit_source, ll_dispatch_select_algo,
    ll_combine_jit_source, ll_combine_lsa_eligible): the parameter list is
    parsed and each parameter is bound BY NAME to a field of the driver's
    Cfg struct or to a fixed expression (BINDINGS below). Reordered, added or
    removed parameters need no edit. Parameters with a default value are left
    out of the call. A new parameter without a binding stops the run with a
    message naming it; that is the one place a human adds a line.
  * Enums (LlDispatchAlgo, LlCombineKernel, NcclEpDispatchLsaMode): the
    enumerators are parsed, so a new kernel is enumerated and named without an
    edit. Sentinel enumerators (kNone, kInvalid) are skipped.

Exit status 3 when a header or function cannot be parsed or a parameter has no
binding; the caller reports the pack's JIT step as skipped with the message.
"""
import os
import re
import sys

DISPATCH_HDR = "nccl_ep/device/jit/ll_dispatch_jit.cuh"
COMBINE_HDR = "nccl_ep/device/jit/ll_combine_jit.cuh"
DEVICE_DIR = "nccl_ep/device"

SENTINELS = {"kNone", "kInvalid"}

# Parameter name -> C++ expression in the context of the driver (`c` is a Cfg).
# A name here applies to every function above, so one name means one thing.
BINDINGS = {
    # Shared
    "hidden": "c.hidden",
    "num_topk": "c.topk",
    "layout": "c.layout",
    "nvlinkOnly": "c.nvlinkOnly",
    "tokenDtype": "ncclBfloat16",
    "topkIdxIsInt64": "false",
    "cftEnabled": "c.cftEnabled",
    "weightDtype": "c.weightDtype",
    "lsaMode": "c.lsaMode",
    # Dispatch source
    "kernel_spec": "spec",
    "algo": "c.algo",
    "compactQuant": "false",
    "stageQuant": "false",
    "rankOwnsExperts": "true",
    "rankOwnsTokens": "true",
    "numTmaStages": "c.tmaStages",
    "numTmaQuantWarps": "0",
    "recordDstInfo": "false",
    "cftZeroCopy": "c.cftZeroCopy",
    "cftCountedPayload": "c.cftCounted",
    "cftNumWarps": "4",
    "cftOneWordMask": "false",
    # Dispatch selector
    "recipe": "NCCL_EP_DISP_QUANT_NONE",
    # Combine source and eligibility
    "useLogFmt": "false",
    "recipe_literal": "c.recipeLiteral",
    "kernel": "c.combineKernel",
    "qrecipe": "NCCL_EP_COMB_QUANT_NONE",
}

FUNCTIONS = [
    ("ll_dispatch_jit_source", DISPATCH_HDR),
    ("ll_dispatch_select_algo", DISPATCH_HDR),
    ("ll_combine_jit_source", COMBINE_HDR),
    ("ll_combine_lsa_eligible", COMBINE_HDR),
]

ENUMS = ["LlDispatchAlgo", "LlCombineKernel", "NcclEpDispatchLsaMode"]


class GenError(Exception):
    pass


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def read(path):
    try:
        with open(path) as f:
            return strip_comments(f.read())
    except OSError as e:
        raise GenError("cannot read %s: %s" % (path, e))


def split_top_level(text):
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def parse_function(text, name, hdr):
    """Return [(param_name, has_default)] of the first definition of `name`."""
    m = re.search(r"\b%s\s*\(" % re.escape(name), text)
    if not m:
        raise GenError("%s: no definition of %s" % (hdr, name))
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
    if depth:
        raise GenError("%s: unbalanced parentheses in %s" % (hdr, name))
    params = []
    for raw in split_top_level(text[m.end():i - 1]):
        raw = " ".join(raw.split())
        has_default = "=" in raw
        decl = raw.split("=")[0].strip()
        pm = re.search(r"(\w+)\s*$", decl)
        if not pm:
            raise GenError("%s: cannot read parameter '%s' of %s" % (hdr, raw, name))
        params.append((pm.group(1), has_default))
    return params


def call_args(name, hdr, tree, used):
    params = parse_function(read(os.path.join(tree, hdr)), name, hdr)
    args, missing = [], []
    for pname, has_default in params:
        if has_default:
            continue
        if pname not in BINDINGS:
            missing.append(pname)
            continue
        args.append(BINDINGS[pname])
        used.add(BINDINGS[pname])
    if missing:
        raise GenError(
            "%s: parameter(s) %s of %s have no binding; add them to BINDINGS in %s"
            % (hdr, ", ".join(missing), name, os.path.basename(__file__)))
    return args


def find_enum(tree, name):
    for root, _, files in os.walk(os.path.join(tree, DEVICE_DIR)):
        for fn in sorted(files):
            if not fn.endswith((".cuh", ".h", ".hpp")):
                continue
            m = re.search(r"enum\s+class\s+%s\b[^{]*\{([^}]*)\}" % name,
                          read(os.path.join(root, fn)))
            if m:
                items = [e.split("=")[0].strip() for e in m.group(1).split(",")]
                return [e for e in items if e]
    raise GenError("no 'enum class %s' under %s" % (name, DEVICE_DIR))


def tag(enumerator):
    return re.sub(r"^k", "", enumerator).lower()


def emit(tree):
    used = set()
    out = ["// Generated by jit_enumerate_gen.py from the source tree's headers. Do not edit.", ""]

    for enum in ENUMS:
        items = [e for e in find_enum(tree, enum) if e not in SENTINELS]
        out.append("inline constexpr %s kAll%s[] = {%s};" % (
            enum, enum, ", ".join("%s::%s" % (enum, e) for e in items)))
        out.append("inline const char* %sTag(%s v) {" % (enum, enum))
        out.append("    switch (v) {")
        for e in items:
            out.append('        case %s::%s: return "%s";' % (enum, e, tag(e)))
        out.append("        default: return \"unknown\";")
        out.append("    }")
        out.append("}")
        out.append("")

    sigs = {
        "dispatchSource": ("std::string", "ll_dispatch_jit_source", DISPATCH_HDR),
        "selectAlgo": ("LlDispatchAlgo", "ll_dispatch_select_algo", DISPATCH_HDR),
        "combineSource": ("std::string", "ll_combine_jit_source", COMBINE_HDR),
        "lsaEligible": ("bool", "ll_combine_lsa_eligible", COMBINE_HDR),
    }
    for wrapper, (ret, fn, hdr) in sigs.items():
        args = call_args(fn, hdr, tree, used)
        out.append("inline %s %s(const Cfg& c, const nccl_ep::DispatchKernelSpec& spec) {" % (ret, wrapper))
        out.append("    (void)c; (void)spec;")
        out.append("    return %s(%s);" % (fn, ", ".join(args)))
        out.append("}")
        out.append("")

    out.append("#define REVIEW_HAS_WEIGHT_DTYPE %d" % (1 if "c.weightDtype" in used else 0))
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    try:
        text = emit(sys.argv[1])
    except GenError as e:
        sys.stderr.write("jit_enumerate_gen: %s\n" % e)
        return 3
    with open(sys.argv[2], "w") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
