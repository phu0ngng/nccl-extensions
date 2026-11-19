---
name: review-build-evidence
description: "Build-mode step of an nccl_ep review: when a CUDA toolkit is present, probe the machine, ask the user for compiler access when none is present and the diff can change codegen, ask whether to build and run, then produce the evidence pack (base and head builds, warning diff, static and JIT kernel resource diffs, touched tests on each side, compute-sanitizer, optional mutation runs) that the critics cite instead of asking the author. Use at the start of any nccl_ep review; it decides between reading mode and build mode, and it never builds or runs anything without the user's explicit yes."
---

# Build evidence

Runs before triage. It settles the review mode and, in build mode, produces the
evidence pack described in `tools/README.md` and consumed through
`REVIEW_METHOD.md` ("Mechanical evidence").

## Two modes

| Mode | When | What the critics have |
|---|---|---|
| **reading** | no toolchain, the user declined, or the pack failed | the diff and the repository; cost claims follow `REVIEW_METHOD.md` step 7 (ask when the diff can change codegen); test-validity findings are hypotheses |
| **build** | toolchain present and the user said yes | `INDEX.md` and the artifacts; cost claims cite rows; "the test cannot see X" is checked by a mutation run when GPUs are present |

State the mode in the first line of the review and, in build mode, which pack
steps were skipped. A critic never claims an artifact that `INDEX.md` does not list.

## Step 1. Probe, do not act

Check and report, without building anything:

- `nvcc` (under `CUDA_HOME`, default `/usr/local/cuda`, or on `PATH`) and its version;
- an NCCL build at `NCCL_HOME` (`third_party/nccl/build` by default): `include/nccl.h` and `lib/libnccl.so`;
- GPUs: `nvidia-smi -L | wc -l` (the unit tests need 4), and whether `compute-sanitizer` exists;
- disk and time: two source worktrees plus two builds with tests; a full `nccl_ep` build is minutes
  per side on a many-core host, longer on a laptop; the test runs add a few minutes; sanitizer
  runs multiply test time by 5 to 20.

If `nvcc` or the NCCL build is missing, do not install a toolkit or build NCCL on your own
initiative. If the diff cannot change code generation (`REVIEW_METHOD.md` step 7), the mode is
reading: say so in one line, name what is missing, and go to triage. Otherwise ask the user
explicitly, once, whether a compiler can be made available: name what is missing and what
would do (a toolkit path or module of the CI version, or a machine or cluster node where a
compile-only job may run; no GPU needed), and say that it improves the review, because
register, spill, occupancy and load/store-order claims become numbers instead of questions
to the author (`PERF_COMPLEXITY_REVIEW_SKILL.md` section 6). If the user provides one, re-probe
and go to step 2. If not, the mode is reading; say in the review header that the resource
checks were not done.

## Step 2. Ask the user

**Mandatory.** Building and running is the user's decision, every time, even if the user
agreed in an earlier review or the toolchain is obviously present. Do not infer consent from
"review this", from a CI configuration, or from the presence of `review-evidence/` output of
an earlier run. Ask one question with these options, in this order, and wait for the answer:

1. **Full pack**: build base and head (lib in `DEBUG=0` and `DEBUG=1`, tests), run the touched
   tests on both sides with a fresh JIT cache, compile all JIT variants offline, run
   `compute-sanitizer` (`initcheck`, `memcheck`) on the head tests.
2. **Build only**: builds, warning diff, static resource diff, host-only touched tests,
   offline JIT compile of variants from sources generated for a fixed configuration matrix;
   nothing executes on a GPU.
3. **Reuse**: an evidence directory already exists for these exact base and head SHAs
   (`INDEX.md` names them); read it, run nothing.
4. **Skip**: reading mode.

State with the question: the two SHAs, the output directory, the GPU count that will be used,
and the estimate from the probe. If the answer is anything other than a clear choice, treat it
as **Skip**. A user who says "do whatever is fastest" has chosen Skip; a user who says "yes"
without picking has chosen **Build only**, and the GPU steps still need their own yes.

Mutation runs (`tools/mutate_and_run.sh`) and sanitizer tools beyond the two defaults are a
second, separate question, asked only when a critic has a concrete finding that a mutation
would settle; name the patch and the test in the question.

Never run `ep_bench` from a review: it needs MPI and a reservation, and performance numbers
from a shared machine are not evidence.

## Step 3. Run

    REVIEW_GPUS=<n or 0> NCCL_HOME=<path> tools/review_prep.sh <base> <head> <out-dir>

- Full pack: `REVIEW_GPUS` as probed. Build only: `REVIEW_GPUS=0 REVIEW_SANITIZERS=`.
- `out-dir` outside the source tree (default `../review-evidence`); never inside `nccl_ep/`.
- Run it in the foreground and wait; do not start critics on a half-written pack.
- If a build fails, that is the first finding of the review (`blocker` if head fails and base
  builds; a note if both fail for an environment reason). Do not fix the tree to make it build.
- Hand the critics `INDEX.md`; they open only the artifacts they need.

### Keep the JIT enumeration glue current

`tools/jit_enumerate_gen.py` parses the JIT generator and selector signatures and the kernel
enumerators from the tree's headers. Check it before trusting the JIT step, and after any run
whose `INDEX.md` lists the JIT step as skipped with a `jit_enumerate_gen` message:

- Run `tools/jit_enumerate_gen.py <tree> <out.inc>` on the base and the head tree. Exit 3 names
  the parameter that has no binding.
- If the change adds or renames a parameter of `ll_dispatch_jit_source`,
  `ll_dispatch_select_algo`, `ll_combine_jit_source` or `ll_combine_lsa_eligible`, or adds a
  kernel enumerator, the script is stale until proven otherwise. Add the binding to `BINDINGS`
  (a parameter takes the one value that keeps the other variants comparable) and re-run.
- Read `NOT COVERED` lines in the enumeration output. A kernel the matrix does not build is a
  gap in `jit_enumerate.cu`: extend the matrix, or state in the review that the JIT diff does
  not reach that kernel.
- Edit only `tools/`, never the tree under review, and report the edit in the review header as a
  tooling change separate from the findings.

## Step 4. Report what the pack says, not what it implies

In the review header, three lines: the mode; which steps ran and which were skipped with the
reason copied from `INDEX.md`; the test and sanitizer verdicts (pass or fail per side). Then
triage.

## Scope rules

- The pack reads and writes only its own output directory and the two worktrees it creates
  under it. It does not touch the user's working tree or any remote.
- Remove nothing. The user decides when to delete `out-dir`; say its size at the end.
- The evidence is for this change. Do not use the pack to review untouched kernels, and do
  not report a static-resource difference in a kernel the change does not reach.
