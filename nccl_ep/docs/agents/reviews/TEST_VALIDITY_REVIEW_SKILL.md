---
name: review-test-validity
description: "Critic for test and benchmark value and validity in nccl_ep changes. First calls out tests that should not be committed (one-time verification of new code, per-function tests of an e2e-covered feature, near-duplicates, cases that differ only in constants, tests of internals or of the compiler, CI-matrix growth with no regression value), then checks the tests that stay: do they reach the new path, can they fail, is the reference honest (including byte-exact transport payloads), are skips reported and covered by a CI row, are tile-size and pointer-alignment cases covered, and are env leakage, rank divergence, seeds, tolerances, diagnostics, tier and sanitizer coverage sound. Use when reviewing any nccl_ep change that adds or changes tests, ep_bench validation, or CI configuration, and especially for agent-generated code, which adds many tests."
---

# Test review

Apply `REVIEW_METHOD.md` first.

A test is a claim: "if this passes, behavior X holds." It also has a standing cost:
every commit pays for it in CI time, GPU allocation, build time and maintenance.
For each added or changed test ask two questions, in order:

1. Is the claim worth paying for on every pipeline? (section 1)
2. If yes, can the test actually fail, and does it reach the new path? (sections 2-6)

Report in the `README.md` format. For section 1 the "Suggest" line must say what to do:
**delete**, **merge** into a named existing test, **run once** (result goes in the change
description), or **demote** to a nightly or manual tier.

## 1. Should this test exist?

Agent-generated changes tend to add a test for every line they add. Most verify that the
new code works today, which a developer does once, not forever. The CI unit suite already
expands many binaries into multi-GPU cases, and each new binary or mode variant adds to
every pipeline. Keep a test only when it guards a **regression that is plausible and
costly**.

### 1.1 Decision rule

For each test, finish the sentence: "This test fails if someone later changes ____, and
that would hurt users because ____." Name the concrete edit and the user-visible failure.
If you cannot, or if the blank is "the code I just wrote", report it.

### 1.2 Cost

For each new binary or CI case state the build cost, GPUs and ranks, and rough
wall-clock. Test code larger than the product change is a prompt to explain, not a
finding by itself.

### 1.3 Call these out

- **Run-once verification.** The test checks a property no plausible later edit can
  change, or only repeats the author's manual check of the new code. Run it once and put
  the result in the change description. If it is the only check of a feature's main flow,
  convert it into **one** end-to-end test.
- **One test per new function.** Unit tests of helpers that an end-to-end test already
  drives, *when a failure of that test would still localize the cause*. Keep one direct
  test for a helper whose failures an end-to-end run would not localize.
- **Near-duplicates.** The same flow repeated across layout, dtype, mode or rank count
  where the code under test does not branch on the difference. Name the branch (or its
  absence) in the finding. Merge into one parameterized case or keep the representative.
- **Constant-only variants.** Cases differing only in sizes with no boundary behind them.
  Keep the boundaries: zero, one, the limit and limit + 1, the value where the code
  switches path, wraparound and reuse after overflow. Derive them from the diff's
  branches. Merge duplicates of the same boundary.
- **New binary for a small feature.** Prefer a case in the existing binary for that
  layer, or in the bench validation, since each binary costs a build and a job.
- **Matrix growth.** A new env, mode or layout added to every test, or a CI row per
  variant, when only the new path differs. Restrict it to cases that reach the new path,
  and keep CI registration in sync with the local runner (section 2).
- **Tests of internals.** Asserts on private layout, kernel parameter structs or constants
  that change with tuning. They fail on refactors and catch no user-visible bug. Test the
  observable contract.
- **Tests of the language or library.** Checks that `static_assert`, `sizeof`, `memcpy` or
  CUDA behave as documented. A `static_assert` in the header is cheaper. One ABI layout
  test for the public structs is justified; not one per struct (extend it).
- **Mirror tests.** The expected value comes from the same code or formula as the
  implementation, so the test cannot disagree with it.
- **Duplicate bug tests.** A fixed bug needs one regression test at the lowest level that
  reproduces it, not one per symptom. Keep it when the bug can return through ordinary
  edits or needs the full stack to show up.
- **Slow tests for fast questions.** Multi-GPU, multi-iteration or large-shape runs for a
  property that does not depend on rank count or size. Do not flag tests of inherently
  multi-rank behavior.
- **Perf checks in unit tests.** Timing assertions are flaky and slow; perf belongs in the
  bench and the perf-regression pipeline.

### 1.4 Guard rails against over-deletion

- Before recommending **delete**, name the surviving test that fails for the same change
  and confirm it runs in CI. If it is skipped or gated off on CI hardware, mark the
  finding "unverified" and do not delete.
- Do not delete a regression test that carries a bug or design-document reference unless
  the cause is eliminated.
- **Merge** must not drop boundary, mode or error-path coverage.
- A test that sections 2-6 require is a keep, not a run-once: the rejection test for a new
  parameter, coverage of each affected transport and dtype, and a test of a protocol or
  wire-format change.

### 1.5 Keep these

Regressions a reviewer already caught once (stale handle memory, epoch parity, ordering,
wrong-branch selection); public-contract behavior callers depend on (error codes,
overflow policy, ABI layout, documented defaults, mixed-mode rejection); boundaries where
code changes path; and one end-to-end test per feature that drives the real flow with
checks on the output.

### 1.6 What to suggest instead

Verify once locally and put the command and result in the change description; add a
numerical comparison to the existing bench validation; add a `static_assert` or a host
validation that returns an error; move long or rare sweeps to a nightly tier.

### 1.7 Tiers

New tests name their tier and a wall-clock estimate: pull-request (small, few GPUs, fast),
nightly (sweeps, seed fuzz, sanitizer runs), manual (perf). Do not put nightly-sized
tests in the pull-request tier.

The standalone QA runner (`nccl_ep/qa/run_ep_qa.sh`, see `nccl_ep/qa/README.md`) is a second
place tests live, with its own tiers: `smoke`, `unit` (the `tests/run_tests.sh` matrix),
`bench` (including the multi-node and boundary-size cases), `python`, and `full` for release
qualification. The two have different budgets, so the bar for asking for a test differs:

- **CI has a small time budget; the bar for adding a test there is high.** Ask for a CI test
  only when the change adds a path nothing else reaches and a short, few-GPU test can prove
  it. Select carefully: prefer extending an existing CI case or a host-only test over a new
  case, and do not ask for sweeps or large shapes in CI. A reviewer who asks for several new
  CI tests names which existing ones to drop or merge, or says why the budget is justified.
- **QA has a large budget; the bar is lower.** A new mode variant, dtype, transport or
  boundary size belongs in the QA matrix even when it is too slow for CI. It is fine to ask
  for the QA case, but do not make a QA-only addition a `blocker`.
- **Ask the author to run QA cases and post the result.** A QA run needs GPUs the reviewer
  may not have, so the request names the exact command, and the author pastes the summary
  (`results.tsv`, or the failing case logs) in the change. Example:
  `run_ep_qa.sh --install <prefix> --np 8 --suite bench --case 'bench-ll-*'`.
- **Existing tests count, not only added ones.** For a change to code that existing tests
  already reach, the reviewer may request the author run the existing cases that cover it,
  to show the behavior is preserved. This applies to refactors, performance changes and any
  change to a shared path. Run `run_ep_qa.sh --suite full --list` (no GPU needed) and
  `tests/run_tests.sh` names, and match the touched code to case names: kernel family,
  layout (`rm`/`em`, `fl`/`em`), dtype (`bf16`/`fp16`/`fp32`), quantization (`quant`,
  `ds-fp8`, `fp4`), zero-copy (`zcopy`), AFD, non-aligned or boundary sizes. Hand the author
  a short `--case` glob list, not "run everything", and say which transport (single NVLink
  domain, or `--hostfile` for RDMA) each one needs.
- **Keep the two registered consistently.** The README says CI coverage changes update the
  runner in the same change. A new CI test with no QA counterpart is a question to the
  author; the reverse is expected and not a finding.
- A case that needs a condition the runner gates on (hostfile, rank count, NVFP4, HT) says
  so, and is not silently absent from the configuration that was meant to exercise it.

## 2. Does it reach the new path, and does it run?

- Every new fused or new-mode path has a test that actually reaches it (for example a
  drop policy with multi-expert hits). A gate that routes the test to the old path hides
  the new one. Wire-format changes need coverage on every affected transport and dtype.
- Reach the path that motivated the change, not only the default. Tests assert the mode
  the library actually resolved and do not infer it from env variables; a hard-coded env
  value collapses the CI matrix.
- **Skips are not coverage.** A skip uses the framework's skip mechanism so it is
  reported; returning success with a "skipped" message hides it as a pass. A new test
  declares its required GPUs, ranks and env, carries a named reason for each skip, and
  says which CI job proves it ran rather than skipped: an env-gated or arch-gated test
  has a CI row that sets the condition. A suite that returns success when there are too
  few GPUs, or tests that skip on a gating mistake, leave the new path with no coverage
  and a green pipeline. Count skips in CI, or fail a row that skips everything. A
  forced-path test removed from CI because it fails is a finding, not a cleanup: the
  failure may be the silent fallback it was written to catch.
- New tests and mode variants are registered in the CI configuration, and it stays in
  sync with the local runner. Test gating helpers mirror the library's own gate.
- Capability-gated fallbacks are tested on an environment that lacks the capability, not
  only on a build that compiles it. When a test sizes a resource from a query, assert the
  smaller size was actually used.
- Tests vary data per iteration so they can observe epoch and parity flips.
- **Sizes around a tile.** Where a kernel moves data in fixed-size tiles or stages (a
  staging buffer, a vector width, a chunk of tokens), tests cover the smallest supported
  size, a size that is not a multiple of the tile (a partial last tile), one step below,
  at and one step above each tile boundary the diff introduces or changes, and the
  largest supported size. Derive the tile size from the diff, not from the test, and
  assert the observable result at those sizes, not the tile constant itself.
- **Pointer alignment is a separate axis.** Where the API accepts buffers at non-default
  alignment (a window offset, a sliced tensor), test one aligned and one deliberately
  misaligned buffer and assert which copy path was taken. Where it does not accept them,
  test the rejection with the exact error code.

## 3. Can it fail?

- A test must not pass by reading stale or recycled memory. Fill output and receive
  buffers with a sentinel (for example NaN) before each operation and assert that padding
  and untouched regions still hold it, and that written regions changed.
- No tautological branches: assert a post-condition that distinguishes right from wrong.
- When an assertion becomes invalid, fix or relocate it. Never skip or weaken the only
  test of a behavior; keep the one slot-exact or value-exact check that exists.
- Comparisons fail on NaN and Inf: write them so a NaN cannot pass.
- Tolerances are derived from the dtype's epsilon and the number of accumulated terms, or
  the comparison is bit-exact where the kernel is deterministic; comment the derivation
  beside the constant. Do not relax a tolerance to make a mismatch pass; trace a value
  end to end first. State whether reduction order is fixed; if not, compare against a
  higher-precision reference instead of a reordered sum.
- Every new public parameter or mode has a rejection test for an invalid value that
  checks the exact error code and that handles and buffers are still reusable and
  destroyable afterwards. Error paths need tests, not only happy paths.
- Tests of validation do not pass through the kernel (a device trap is not an error
  return).

## 4. Is the reference honest and the data realistic?

- The reference reproduces the rounding or quantization granularity of the code under
  test and is otherwise as simple as possible: same input, check the output, do not
  mimic the implementation. Validity or expected values come from the same inputs the API
  contract uses, not from payload data.
- For transport-only kernels (the dispatch and combine data path), validate with a
  deterministic payload that encodes source rank, token and element, and compare bytes
  exactly. A misroute is then identifiable and no tolerance is needed. Do not quantize in
  a transport test. This does not apply where the library itself quantizes or decodes;
  use the tolerance rules there.
- Validation covers all layouts and paths, with validators named after the real layout.
  Test-only setup lives in test or validation code, never in library or shared bench
  paths.
- Routing-dependent changes include a skewed or empty case (hot expert, zero-token
  ranks, duplicate routes) and one near real size, or state why the change is
  unaffected. A capacity-versus-actual bug does not show at tiny balanced shapes.
- Log the random seed on failure; add a random-seed pass in the nightly tier. A fixed
  seed makes an input-dependent bug pass identically every run.

## 5. Isolation, ranks and diagnostics

- A test that sets an environment variable restores it through an RAII guard. Run new
  tests alone and in shuffled order.
- A rank-divergent assertion must be collective: reduce a pass or fail flag across ranks
  or abort the group. Otherwise one rank returns while peers block, and the job burns its
  whole timeout with no failing assertion in the log. Launchers have a per-suite timeout
  that kills every rank.
- A change to group or handle configuration says what happens when ranks disagree, with a
  bounded-time test or an explicit "unsupported" with a host check.
- Per-element checks report the first N mismatches (rank, indices, expected, actual,
  resolved mode) and cap output. A bare count is not a diagnostic.

## 6. Sanitizers, goldens and benchmarks

- A change to a synchronization protocol or buffer layout includes a small-shape
  race, memory and initialization sanitizer run, with the result in the change
  description, and the nightly tier runs them.
- When a golden constant (struct offset, byte size, boundary) changes in a diff, the
  description states why that is intentional. Tests assert documented behavior; a header
  or doc change that alters behavior needs a matching test change.
- Benchmarks are isolated from other work in the process, benchmark each new mode, and
  must not add per-iteration copies that bias the new path. Bandwidth accounting uses the
  bytes actually moved per dtype and format (payload plus scales). Report bus and
  algorithm bandwidth explicitly, and say how correctness is validated.
- Perf rows come from same-node A-B-B-A runs, isolate the cause of a claimed gain, and
  cover the targeted regime and every transport or topology the change touches. Publish
  regression results for setup or routing changes even when no impact is expected.
- State which claims CI verifies and which were checked by hand. Re-measure after a
  correctness or synchronization fix.
