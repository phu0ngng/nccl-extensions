# Review critics

Critic skills for reviewing nccl_ep changes. Each one encodes rules that reviewers
applied in past reviews. They complement the writing
skills in `../CODING_SKILL.md` and `../PLC_DOCS_SKILL.md`.

| Skill | File | Use for |
|---|---|---|
| `review-memory-model` | `MEMORY_MODEL_REVIEW_SKILL.md` | fences, proxies, epochs, banks, buffer lifetime, handle state, and the API contract they imply |
| `review-code-organization` | `CODE_ORGANIZATION_REVIEW_SKILL.md` | modularity, duplication, names, constants, scope of the change |
| `review-perf-complexity` | `PERF_COMPLEXITY_REVIEW_SKILL.md` | cost of each new loop, launch, copy or sync versus actual data |
| `review-api-change` | `API_CHANGE_REVIEW_SKILL.md` | `nccl_ep.h`, env knobs, config fields, versioning, documented contract |
| `review-error-handling` | `ERROR_HANDLING_REVIEW_SKILL.md` | returned errors, validation, device guards, selection and budgets |
| `review-test-validity` | `TEST_VALIDITY_REVIEW_SKILL.md` | whether a test is worth its CI cost (delete, merge, run once, demote), then whether it can actually fail |
| `review-build-evidence` | `BUILD_EVIDENCE_SKILL.md` | the step before triage: probe the toolchain, ask the user, and in build mode produce the evidence pack (`tools/`) |

## Method

`REVIEW_METHOD.md` holds the reasoning steps and the pre-assertion checks shared by all
critics. Each skill points to it.

## Running a review

A review runs in one of two modes and has four stages: mode selection, triage, critics,
consolidation. Skipping triage runs every critic at full cost on diffs most of them cannot
apply to; skipping consolidation posts duplicates and unverified claims.

### 0. Mode

- **Reading mode**: the critics have the diff and the repository. Cost claims follow
  `REVIEW_METHOD.md` step 7; test-validity findings are hypotheses stated as such.
- **Build mode**: `BUILD_EVIDENCE_SKILL.md` has produced an evidence pack (`tools/`), and the
  critics cite its artifacts (`REVIEW_METHOD.md`, "Mechanical evidence").

`BUILD_EVIDENCE_SKILL.md` decides: it probes the toolchain and **asks the user** before
building or running anything. No build, test run, sanitizer or mutation run happens without
that explicit yes, in any review, whatever an earlier review was told. The first line of the
review names the mode.

### 1. Triage

In build mode read the pack's `INDEX.md` and `hunks.md` first; in reading mode read the diff
once. Classify what the diff changes, then pick the critics. A clean `res_usage_diff.md` and
`jit_diff.md` rule out perf-complexity on a fact rather than on a reading of the diff; a
build failure or a sanitizer error is a finding before any critic runs.

| The diff changes | Critics that apply |
|---|---|
| a flag, counter, epoch, bank, fence, barrier, signal, buffer lifetime or handle state | memory model, error handling |
| a kernel body, loop bound, launch geometry, copy or division | perf-complexity, memory model if it touches shared state |
| validation, error returns, asserts, selection or budgets | error handling, API if a documented contract moves |
| `nccl_ep.h`, env knobs, config fields, docs of a contract, python bindings | API, error handling |
| tests, benchmarks, CI rows | test validity |
| only the type, signature or plumbing of existing values (templating, renames, moves) | code organization, test validity; API if a public type changes |
| any of the above | code organization |

A critic that does not apply is not run. Triage records it as **ruled out**, with the
reasons per area, hunk by hunk ("no fence, counter, epoch or buffer-lifetime line changes;
the two sync-adjacent hunks change only a template signature"). A bare label is not a
verdict; a ruled-out area without reasons is reviewed in full. Triage is one reading by one
agent, so it is checked twice more:

- every critic that runs says in one line when it sees a rule from a ruled-out area apply
  (an added `static` in a templated function, an assert that aborts across the C ABI, a
  changed count), naming the area;
- consolidation reads those lines and starts the named critic in full before merging.

The ruled-out areas and their reasons appear in the review header, so the author can see
what was not reviewed and why.

### 2. Critics

Review a diff with the selected critics, or name one:

    Review the current branch against main with the review-memory-model skill.

Read the diff and the surrounding code before judging. A finding that does not point at
a changed line, or at an unchanged line the change makes wrong, is dropped. Do not search
for something to report: "no findings" is a valid result for any skill.

### 3. Consolidation

Before anything is posted, one pass over all critics' output:

- **Merge.** Findings that rest on the same fact (the same stale comment, the same check,
  the same missing test) become one finding naming every area that raised it. Keep the
  strongest evidence and the most specific suggestion.
- **Refute.** For every remaining finding, try to disprove it from the code: a guard
  above the line, a caller that never produces the input, a path that is already null or
  unreachable, a sibling that already handles it. Spend tool calls on this; do not judge
  from the finding's own text. A finding that is disproved is dropped, and one whose
  premise cannot be confirmed either way is demoted to a question. State in the finding
  that it survived this check and what was tried.
- **Reopen.** If any critic flagged a rule from an area triage ruled out, run that critic
  in full on the flagged hunks first, and merge its output with the rest.
- **Answer questions.** Apply the question rule from `REVIEW_METHOD.md` to every question
  a critic raised: answer it from the repository if that is possible, or drop it. A
  question one critic answered is not asked by another.
- **Order.** Majors first, then minors grouped by the fix they need, then questions.
  Put what was checked and found clean in a short summary at the end.

## Finding format

One entry per finding, most severe first:

    [severity] file:line - rule (skill section)
    Claim: one testable sentence.
    Evidence: the quoted changed line, and the other side (caller, consumer, sibling or
      documented contract) with file:line.
    Trigger: the input, interleaving, rank layout or size that fails, and the failing
      outcome. Optional for readability, naming, documentation and test-value findings.
    Checked: what you read or grepped to rule out "already handled" and "pre-existing".
    Suggest: the smallest change, or "no fix proposed".

A runtime finding with no trigger or no stated other side is dropped. A claim you cannot verify
from the repository goes in a separate **Questions for the author** list, subject to the
question rule in `REVIEW_METHOD.md` (answer it yourself first; ask only about intent,
measurements or hardware). It is not a finding. A vague "consider" remark is neither.

Severities: `blocker` (hang, corruption, ABI break, error across the C ABI, with a stated
trigger), `major` (silent wrong behavior, regression, contract mismatch), `minor`
(readability, naming, wording of docs; a missing or false contract statement follows the
severity guide of the skill that owns it). A style item is never above `minor`. Report every finding
that meets the evidence bar above; there is no cap on the number. Say "no findings" for a
skill that found none.

## Staying quiet

Do not report:

- a line that is not in the diff, unless the diff makes it wrong;
- a pre-existing issue the change does not touch or worsen;
- something a linter, formatter or CI already enforces;
- an issue that needs an input or scale no caller can produce;
- a claim that depends on a toolchain or API fact you did not check;
- a finding the author already answered with a reason you have verified is true.

Two exceptions. The code-organization checks on duplicated logic, hand-rolled helpers and
inline assembly may reach existing code that the change relates to (the new code repeats
it, needs the same operation, or sits next to it); an unrelated existing instance is not
reported. This is the only place the scope rule is relaxed. And "sibling code does the
same" never silences a safety rule such as a bounded wait, a fence or validation.

## When there is an earlier review

If earlier findings or author replies are available, read them first.

- Post only new findings, or "fix verified at <commit>" or "fix incomplete" for an earlier
  one. Do not repeat a finding the author declined.
- Treat an author's reason ("out of scope", "pre-existing", "invariant X holds") as a
  hypothesis: verify it once in the code. If it is true, withdraw the finding. If it is
  false, reply with the line that contradicts it.
- After a fix lands, re-read the new code for the same invariant, and name any remaining
  case. Follow-ups of this kind caught the most real holes in past reviews.
- Say once, in a single line, what you saw outside the scope of the change; do not
  review whole files.

## Source and limits

Rules come from a large set of past review threads on nccl_ep changes. They were
tagged by automated reading and the ones kept were checked against the code history; they
have not been fully reviewed by a person. Some rules are design
principles with no past review behind them. When a rule here conflicts with the current
code or a newer decision, raise the conflict as a question to the author; the newer
decision wins and the skill should be updated.
