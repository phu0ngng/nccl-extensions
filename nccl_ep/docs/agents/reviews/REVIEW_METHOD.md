# Review method

Shared by every critic in this directory. The reviewers whose comments held up best
worked the same way; apply these steps before writing a finding.

## How to review

1. **Build the smallest counterexample, at the extreme of the supported range.** Name the input or
   interleaving that breaks the code: the largest rank count, a one-byte buffer at
   the end of a page, a full queue, the last CTA exiting early. "Might race" is not a
   finding; a concrete failing case is.
2. **Audit every exit path.** For each resource a function acquires (buffer, handle,
   lock, epoch, slot), walk every return, error and early-exit path and check it is
   released or rolled back.
3. **Diff against the sibling or legacy path.** When code replaces or parallels an
   existing path, list what the old path did that the new one drops: a check, a
   fence, a cleanup, a default.
4. **Ask what a value means.** Who produces it, in what unit, and can a legal value
   collide with a sentinel (for example 0)? Check polarity and units on every
   threshold.
5. **Check what a fence, barrier or lock covers.** Name the memory and the peers it
   orders. A guard that covers some peers or one proxy does not cover the rest.
6. **Check both sides of a contract.** Allocation against selection, host against
   device, producer against consumer, documentation against code. A change on one side
   without the other is the usual bug.
7. **Demand evidence for cost claims, and get it yourself when you can.** Do not
   assert register use, spills, divergence or speedup without SASS, `ptxas -v` or a
   measurement.
   - If `nvcc` is available, produce the evidence: compile the changed kernels with the
     project's own flags and target architecture and read `-Xptxas -v` (registers,
     spills, shared memory, stack) and the SASS (`cuobjdump -sass`). Do the same on the
     base revision and compare. Cite the numbers in the finding.
   - Use the architecture, CUDA version and flags the build uses, and say which. Results
     for another architecture or toolkit do not transfer. For a JIT-compiled kernel,
     compile its generated source with the options the JIT uses.
   - Static output shows registers, spills and instruction mix, not latency or
     throughput. A speedup needs a measurement on a GPU.
   - Compile only the changed translation units or the instantiations the diff touches,
     and say which you skipped.
   - If `nvcc` is not available, or the kernel cannot be built there, first decide
     whether the diff can change code generation at all. Templating an existing path on a
     type while the instantiation used today stays the same, renames, moves and host-only
     changes do not; say so and stop. A changed loop, branch, memory width, launch bound,
     shared-memory size or new per-thread array does. Only in that case report the claim
     as a question and ask the author for the numbers, naming the kernel and
     instantiation.
8. **Separate blocking from non-blocking.** Mark what must change before merge and
   what is a suggestion. Drop a finding when the author shows a counterexample to it.
9. **Control scope.** An unrelated change belongs in a separate commit. Do not demand
   redesigns outside the diff. The one exception is in the README ("Staying quiet").

## Questions for the author

A question costs the author a round trip. Before asking one, try to answer it from the
repository: trace the value, read the function you are unsure about, grep for the consumers
of a name, read the design document the code points at. Budget a handful of tool calls per
question. Then:

- If the repository answers it, there is no question. Either it becomes a finding or it
  is dropped.
- If a premise of a finding was the open point and it turned out false, drop the finding.
- Ask only when the answer is not in the repository: the author's intent, a measurement, a
  hardware or toolchain fact you cannot reproduce. State what you read and why it did not
  settle the point. "Is this intended?" without that is not a question.

## Mechanical evidence

In build mode (`BUILD_EVIDENCE_SKILL.md`: toolchain present and the user agreed),
`tools/review_prep.sh` has built an evidence pack once for the change
(`tools/README.md`). Read its `INDEX.md` before the diff. In reading mode this section
does not apply and step 7 governs cost claims. Cite the artifact instead of recomputing
or asking:

| Claim | Artifact |
|---|---|
| the change compiles, in debug and release flags; new warnings | `build_<side>.log`, `warnings_new.md` |
| registers, spills, shared memory, stack of a statically compiled kernel, base against head | `res_usage_diff.md` |
| the same for a JIT kernel, per variant; which JIT variants the touched tests reach at all | `jit_diff.md`, `jit_<side>/variants.tsv` |
| the touched tests pass before and after | `tests_<side>.log` |
| uninitialized reads, out-of-bounds access, shared-memory races, barrier misuse in the touched tests | `sanitizer_<tool>.log` |
| a test still passes with a given defect applied | `mutation_<name>.log` via `tools/mutate_and_run.sh` |

Rules for using it:

- A cost claim with a row in `res_usage_diff.md` or `jit_diff.md` cites the row. Without a row,
  step 7 applies. Static numbers are not latency; a speedup still needs a measurement.
- When `INDEX.md` says the JIT cache was filled by a test run, a JIT variant the change adds
  that is absent from `jit_head/variants.tsv` was not executed by any touched test. That is a
  test-validity finding, not a question to the author. When the sources were generated for a
  fixed matrix instead, the table says only which variants compile; test reach is unknown.
- A test-validity finding of the form "the test would still pass if X" is checked with a
  mutation run when GPUs are present and the user has agreed to that run (it is a separate
  question, see `BUILD_EVIDENCE_SKILL.md`); report the result, not the hypothesis. Without
  it, state that the mutation was not run.
- The pack lists what it skipped and why. A critic that needs a skipped artifact says so
  and falls back to reading the code; it does not pretend the artifact exists.
- Formatting is outside the pack: `nccl_ep/` is excluded from `clang-format` by the
  repository, so formatting stays a code-organization judgment under `CODING_SKILL.md`.

## Mechanical checks

Do these with tools, not from memory.

- API, intrinsic, architecture and version claims: grep the headers and gating macros
  (toolkit, NCCL and architecture version macros) and cite the file. Never say something
  "does not exist" from recall. Find the version that introduced an API before calling a
  guard a typo.
- Callers and consumers: grep every caller and the free or consumer path before claiming a
  leak, a race or a null.
- Guards: read the code above the line and up to the next barrier or reset before claiming
  an input is unchecked.
- Arithmetic: run the expression on one concrete value (signed, unsigned, wrap, stride,
  rank layout).
- ABI: compute `sizeof` and `offsetof` of the struct before and after, and check the
  version bump and padding.
- Compare with the sibling or reference implementation, and say what the new path drops.

## Before asserting a finding

These are the patterns behind the wrong comments in past reviews. Check them first.

- Asserting from memory: PTX operand widths, API availability, architecture limits.
  Verify against the toolchain or the code.
- Misreading units or polarity, or reasoning from a stale premise (an earlier
  revision, a removed path).
- Treating a question as a defect. If you do not know, ask and mark it as a question.
- Judging a guard without asking what release builds do when it rejects an input.
- Calling code dead without checking every kernel family and build variant that can
  instantiate it.
- Arguing "sibling code does the same" as a rebuttal. Consistency with other code
  does not make unbounded waits or missing fences safe.
