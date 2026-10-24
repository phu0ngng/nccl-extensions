---
name: review-error-handling
description: "Critic for error handling, validation and robustness in nccl_ep changes: exceptions or aborts crossing the C API, assert-only checks that vanish under NDEBUG, new device asserts and traps that kill the CUDA context, teardown and rollback on failure, sticky-error and get-last-error handling, rank-mask and timeout flows, integer narrowing and overflow, null/zero/alignment inputs, device-computed capacity checks, env-knob parsing, JIT failure semantics, selection and eligibility logic, forced versus AUTO paths and default-change rules, resource budgets, state after failure, and error-message quality. Use when reviewing any nccl_ep change that adds validation, a new mode, a selection or fallback path, a new kernel parameter, allocation, or JIT launch configuration."
---

# Error-handling and guard review

Apply `REVIEW_METHOD.md` first.

For each new input, mode or resource, ask what the caller sees when it is wrong: a
returned error, a trap, a hang, or silently wrong data. Only the first is acceptable.
Report in the `README.md` format.

## 1. Public paths return errors, never throw or abort

- Public C entry points return an error status. Any assert, throw, `abort` or process
  exit reachable from them is a finding (existing instances follow the known-idioms rule
  below). Where internal helpers throw, one wrapper per
  entry point maps the exception to a specific status; check that new throwing helpers
  are covered by it.
- Recoverable failures at any layer (JIT launch, undersized buffer, bad env value,
  library call failing during init) return an error; they do not terminate the
  application. Errors name the failing combination.
- Enforce each invariant once. Input-derived invariants are validated on the host with a
  returned error that behaves the same in debug and release.
- **Avoid adding `EP_DEVICE_ASSERT`.** Today it traps in every build and kills the CUDA
  context, and the project intends to move to a release model where device asserts are
  compiled out. An assert then protects nothing in production, and a trap is a worse
  failure than an error. Every newly added device assert is a finding unless all of
  these hold:
  - the invariant is visible only on the device (for example a count computed there), so
    the host cannot check it;
  - no compile-time check, host check or clamp can replace it;
  - the comment says why, and what breaks if it is violated.

  Prefer, in order: `static_assert`; a host check with a returned error; a clamp or drop
  that sets an error flag the host reads at the next call (the existing async-error-flag
  pattern). An invariant that must hold in production does not rest on an assert alone.
  Existing asserts on lines the change does not touch are out of scope. If the change
  edits an assert or adds one next to it, ask whether it can be removed.
- **Known assert idioms on public paths.** Today the library validates caller input with
  a throwing host assert (null public arguments, unset layout, counter tensor rank,
  dtype and null checks, "feature not enabled" checks, mode-combination checks) and with
  plain `assert` for tensor descriptors, which disappears under `NDEBUG`. These are
  known debt and are being removed. Flag new code that copies these idioms. Do not
  re-report an existing instance unless the change touches it. Check that every public
  entry point maps a thrown exception to a returned status.
- Validation runs **before** work is enqueued, so a failure never leaves a half-launched
  operation.

## 2. Inputs

- Validate every precondition kernels assume, with a returned error, and document it:
  dtype, shape, limits, alignment, non-null for nonzero size, number of dimensions,
  contiguity, and the legality of a zero-size call (a no-op success, not a launch with an
  empty grid).
- Take user-supplied wide dimensions through a checked multiply in a wide type and
  range-check before narrowing to a smaller type. Use one shared overflow helper, not
  one-off checks. Products such as rows x hidden x element size must not wrap. The same
  holds for caller-supplied index arrays (expert or token ids) that a kernel narrows to
  a smaller integer to pack or bucket them: grep each cast, and require the range check
  on the wide value before the cast. A check after the cast cannot see a wrapped value,
  which then lands in a valid bucket. Expected evidence: the input value just above the
  narrow type's maximum and where it ends up.
- Reject unsupported combinations and unsupported groups at creation, collectively on all
  ranks, not on first use. Also reject them at each per-call entry point (dispatch,
  combine, and any other op) that can see the unsupported mode or topology, with a
  returned status that names the combination; a creation-time check alone does not cover
  a handle or mode that reaches the call by another route. Reject user buffers that are
  too small as host errors, not as device traps.
- Entry validation covers every field the inner helpers dereference. A defensive check
  names what it compares against: comparing a value to one derived from the same source
  is a tautology. Do not add a runtime null fallback behind a compile-time "required"
  flag.
- A host check on a device-computed quantity (a received count against a capacity) that
  needs a device-to-host sync is not deleted because the sync is costly; it protects
  memory. Move it into the kernel: clamp or drop the excess and set the async error flag
  the host reads at the next call. A trap is allowed only if the device-assert exception
  conditions in section 1 hold. Record the checked value, and compare selection and
  allocation against that same recorded value.
- Parse env knobs through the single env module with a strict parser, a range check and a
  returned error at creation. Resolve and log the effective value once, and make it
  identical on all ranks or verified collectively. Do not silently ignore or partially
  parse invalid values.
- Struct and config fields added later are read only through the size-checked decode path
  with a defined default; never read past the caller's declared size.
- A fixed limit (rank count, token count, mask width, arrival bits) is rejected on the
  host before launch with a returned error that names the limit; a device assert as the
  only check is a finding. When a kernel maps a per-token list (top-k entries, expert
  slots) onto warp lanes or a fixed array, list every kernel in the diff and nearby that
  uses the same mapping and check that each one enforces the same bound (static assert
  or one host check covering all of them); a bound enforced in one sibling kernel only
  is a finding on the others. The shared-definition rule and the rule against limits that
  encode a topology are in `PERF_COMPLEXITY_REVIEW_SKILL.md` section 3.
- When the user forced a path (for example by an env setting) and it cannot be honored,
  fail with an error. Fall back silently only when the user left the choice to AUTO, and
  log the fallback.

## 3. Device-side guards

- Bounds-check indices and limits that device code relies on before the first dependent
  memory access or atomic (for example an expert id before an atomic add).
- `static_assert` shape constraints that prevent overrun into a neighboring slot.
- Launch configuration is derived from, or checked against, what the kernel structurally
  requires (for example warps per role).
- Comment guards that cover edge cases. Do not convert a guard into an assert, and do not
  add an assert where a host check or a flag-and-continue path is possible.
- Print a runtime diagnostic when a required instruction or capability is unsupported;
  unsupported-feature paths fail loudly through a stub, not silently.

## 4. Failure paths, cleanup and state

- Teardown and cleanup do not use throwing check macros. Release everything best-effort,
  remember the first error, return it at the end, and make destroy idempotent (null
  pointers after freeing).
- Every allocation in create or init is owned by an RAII wrapper or scope guard, or is
  listed in the destroy path. Check that every return and throw after it releases it,
  and that the order mirrors allocation. Collective calls cannot be skipped by one
  rank's error.
- A failing call leaves the handle or group either unchanged (validate before mutating)
  or explicitly poisoned (later calls return an error). Test a call after a failed call.
- Check immediately after each launch and attribute the result to that launch. Never
  clear a sticky error or an unexpected failure with a "get last error" call (illegal
  address, trap, a failed launch): it is fatal, so mark the group poisoned. For an
  expected, recoverable, non-sticky failure of a probe or opt-in call that the code
  handles (for example by falling back), clear it immediately after that call, in one
  place, so it is not blamed on a later launch.
- Distinguish launch-time errors from device errors that surface at a later sync. Do not
  claim an API validated something the device can still trap on; error text says it may
  be asynchronous.

## 5. Cross-rank and bounded waits

- Validation or JIT failures that depend on per-rank data are made collective-consistent
  (a flag reduction before launch) or documented as fatal. One rank erroring while peers
  block is a hang, not an error.
- Every device spin-wait has a timeout whose expiry is reported through an error channel.
  A timeout saturated to a value the loop can never reach is not a timeout.
- Any change touching the rank mask or a timeout path is reviewed against two flows, and
  the code must handle both. (a) Failed ranks are already present when the call starts
  (pre-known): they are skipped in sends, counts, waits and barriers, their outputs are
  defined, and participant counts are computed over the active set. (b) A new rank fails
  during the call: the timeout marks the mask and sets the async error flag, every
  protocol row ends in a consistent state, and the next call sees the updated mask. State
  the post-state of each row for both flows. A timeout that changes shared state
  documents that masks can differ across ranks and who reconciles them.

## 6. JIT and cache failures

- Compile and launch errors return a status carrying a truncated compiler message; they
  never abort. A failed compile is not cached as a success, a corrupt or stale cache
  entry is detected (checksum or version and architecture key) and evicted, and failure
  is made collective.

## 7. Selection, eligibility and fallback

- Auto-selection is an ordered list of eligibility predicates, each checked against what
  the previous stage actually produced. State read by a later stage comes from whichever
  path really ran; every fallback initializes or invalidates it.
- Do not reuse layout-specific math for another layout; guard with an explicit flag.
- Check inputs against the limits of whatever consumes them.
- Plumb a new kernel parameter through every layer that copies it, with a check that
  fails if one is left default.
- Arch or feature gates are enumerated against the full supported-target list; guard on
  the capability itself (version, arch), not a proxy such as header presence.
- Probe a capability with a query, never by creating the resource: a failing create can
  poison state. Gate a feature that needs a newer driver on the runtime driver version as
  well as the toolkit version it was built with.
- On a rebase, re-check predicates the old code used.
- Every combination the host can select must compile and launch, and every combination
  the kernel rejects must be unselectable or return an error before launch. Derive the
  host predicate from the same condition the kernel asserts instead of copying it.
- A change that adds an alternative path or an adaptive knob states whether the default
  behavior changes, and the code, docs and tests agree.
  - If the default is meant to stay the same, the knob-off path matches the previous
    behavior value by value, including everything derived from the knob (buffer sizes,
    header fields, extra stores, grid sizes). Check the derived values, not only that
    the feature does not run.
  - If the default is meant to change, that is an explicit decision: documented as a
    default change with a release note, measured against the old path including
    regressions (not enabled on a size saving alone), and the old path stays reachable
    through an override for a release.
  - A knob-off path that differs from the previous behavior with no statement of intent
    is a finding.

## 8. Resource budgets

- Every new static shared-memory allocation, with alignment, and every other resource the
  selector budgets, is counted, or launch fails.

## 9. Messages and logging

- Errors include the rank, the group or handle, the offending value and the allowed
  range, the mode or path where they fire, and the underlying error name (not only a
  generic string). Use one logger with levels (warn on soft fallback, error on failure),
  API-level names first (name the env variable only when it is the override that
  applied), and no logging in hot paths.
