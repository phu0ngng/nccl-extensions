---
name: review-api-change
description: "Critic for public API and ABI changes in nccl_ep (nccl_ep.h, env knobs, config structs, Python bindings, release notes): can it be smaller, can it yield undefined or silently wrong behavior, zero and AUTO semantics and env precedence, outputs and ignored fields, over-rejection of valid input, knobs for optimizations still in flux, is it consistent across paths, ranks and versions, is it versioned and layout-safe (size, magic, feature version, widths), LSA and rail sizing and symmetry assumptions, are lifetime, thread-safety, stream and error contracts stated, and does the documented contract match what is enforced. Use when a change touches nccl_ep.h, ncclEp* signatures or structs, enums, env variables or config fields, defaults, bindings or public docs."
---

# API-change review

Apply `REVIEW_METHOD.md` first.

Treat the API as a contract with callers you cannot see. For each added or changed
element, answer: is it needed, is it the smallest form, what does a caller get when
they misuse it, and does behavior depend on a hidden mode. Report in the `README.md`
format. Severity guide: ABI break or silent behavior change is `blocker`; missing
contract, inconsistent error code or hang on misuse is `major`.

## 1. Is it needed, and is it minimal?

- No public knob without a concrete consumer, and no benchmark-only knob. Make the
  better path the default, or auto-detect. Do not commit to maintaining two paths.
- Do not publish a public field or enum value for an optimization that is still in flux.
  Use an env variable or an internal flag until the design settles: a public value cannot
  be removed without an ABI break, and a value whose path gets replaced stays behind as
  dead weight. A value that names a transitional path (a "legacy" or "force the old"
  option) is a signal. Remove unused flags before the ABI freezes. Existing public values
  of this kind are known debt: do not re-report them, but flag a new one.
- Keep implementation types out of the public header. A change that adds an internal
  include or type to it is a finding.
- Document AUTO as library-chosen and allowed to change in a later release, and the
  explicit values as the stable contract.
- Do not require inputs the library can derive (for example a global scale).
- Express mutually exclusive choices as one enumerated selector, not several flags or
  booleans that can conflict. This holds for env variables too: when the diff adds two or
  more boolean env variables of one mode family, or a boolean that overlaps an existing
  selector, propose one variable with one value per mode. Name the combination (both
  set, or A set while B selects another mode) whose outcome is decided by code order
  and not by the contract; a selector has no such silent precedence. Prefer a documented
  implicit contract over a new option
  when a cheap always-on behavior suffices. Prefer a group-config field to an env
  variable, and give a documented env fallback a deprecation plan. An undocumented env
  override for a size cutoff or tuning constant is acceptable (see the perf skill).
- Env-variable hygiene: for each new env variable, grep the env module for existing
  knobs and answer four questions in the finding. (1) Does an existing knob already
  bound the same resource (for example an SM budget for a sibling op)? Then reuse it, or
  extend it, instead of adding a parallel one. (2) Does it overlap an existing mode
  family? Then generalize the name or fold it into that selector, and have it return an
  error in modes where it has no effect. (3) Is it a performance feature the user must
  opt into? Ask why it is not selected automatically, and whether the legacy path is
  still needed once it is. (4) Does it have a sibling for the parallel kernel or
  direction, and is its value derivable from the kernel's own parallelization (SMs,
  pipelines, batch)? A knob added for one side only is a finding, as is a constraint it
  inherits from another knob (divisibility, a shared value) that the new knob's own
  semantics do not need: it should be independent, and may override the shared one.
  Expected output: the existing knob or selector named, or the reason none fits.
- Take the underlying quantity, not a lossy flag derived from it. Name fields for the
  contract (what a value bounds), not for how it is used today.
- Set required environment inside the library rather than burden users.
- Defaults are deliberate and documented: a default preserves prior behavior, and no
  default is a silent "none" that disables something. Ask what AUTO resolves to when a
  new capability lands.

## 2. One name, one meaning; no silent behavior

- An output has one meaning across all paths and modes (for example counts before or
  after dropping). Behavior must not depend on a hidden mode the caller cannot observe.
- Dispatch-time and update-time contracts agree on types and ignored fields, or the
  split is documented. API versus env precedence is the same across settings.
- Never silently ignore a caller-provided output or argument on some paths: error, or
  honor it everywhere. Every output the contract promises is written, or the missing
  precondition is rejected with a message; an output conditional on another argument
  needs an explicit check.
- Data a peer writes into the receiver (received top-k indices and weights, counts,
  offsets) are declared outputs in the outputs struct, never a side buffer or an input.
  State per layout and mode when each is required, and return an error naming the field
  when a required one is missing.
- Per-token companion tensors share the main tensor's leading dimensions, and the header
  states the shape.
- When an existing field would change meaning for a new mode (padded or unpadded, before
  or after a drop), prefer a separate field.
- A field that does not apply to the chosen mode may be documented as ignored only while
  it holds its zero, AUTO or NULL value. A non-default value that would have no effect
  is a finding: the call should return an error naming the field and the mode. Do not
  let a mode accept a tensor its algorithm cannot consume; document who owns the step
  instead. Drop the finding only after weighing all of these, and say so in the report:
  the value is a pure performance hint with no correctness effect; the field comment
  says it is ignored in that mode; rejecting would break a legitimate caller (for
  example a framework that reuses one config across modes) and no better remedy
  exists. "It is documented" alone is not enough.
- Before accepting a new rejection, check that the mode does not already support the
  input. A valid input is not rejected only because an optional path is off: under AUTO
  or OFF an optional path is opportunistic (used when the input is compatible, otherwise
  the general path runs), and only an explicit ON is strict. A strict failure returns an
  error that names the real limitation and, where one exists, the working alternative.
  Do not add a second rejection for a case an existing host check already refuses with
  the same error.
- An explicitly requested configuration is honored or rejected, never overridden,
  ignored or downgraded without notice. Renamed or removed names (env variables, fields,
  flags) fail loudly or warn, not silently ignored.
- Validate with an inclusive allowlist of supported modes and dtypes, not a denylist.
  Reject a type or mode on any path that does not implement it. Validate call order and
  state preconditions with a usage error (for example backward before forward).
- Document direction, units and semantics of numeric parameters where a plausible
  inversion would silently produce wrong results (scale versus its reciprocal), and any
  numeric property callers might assume but does not hold.
- Document or opt-in-gate default changes that alter results, such as output ordering or
  count availability.

## 3. ABI, versioning and layout

- Any change to a public struct's layout bumps the ABI version, appends fields only, and
  updates the frozen per-version size macros and static asserts in the same change. Never
  append to a released boundary.
  - When a change freezes a size or layout as version N, diff the public header against
    the last released tag and list every field added after that release. Each one must
    sit after the released boundary, in a later version with its own size macro; a
    frozen boundary taken from the current layout silently includes unreleased fields
    and breaks callers built against the release. Evidence: the field list, and
    `sizeof` at the tag versus the new frozen size.
  - A macro that declares per-struct boundary versions must statically assert that each
    boundary version is at most the current API version, so a field added without a
    version bump fails at compile time.
  - When a per-struct version scheme replaces or supplements a global API version, grep
    every version and feature check (group creation, handle creation, dispatch) and
    confirm each uses the new scheme and the new boundaries, not the old constant.
- New fields use fixed-width types and natural alignment, with explicit padding. Do not
  rely on enum width or compiler padding.
- Extents and byte sizes that can exceed 2^32 (anything that scales with tokens x hidden
  x element size) use `size_t` or `uint64_t` in structs and signatures, not `unsigned int`.
- A new variant or option is a struct field or a new entry point. Never change a released
  signature in place, add an overload, or use a default argument.
- **Zero means old behavior.** Old callers pass zero (or a smaller struct) for fields they
  do not know. State the zero meaning of every appended field and check it equals
  pre-change behavior. Never renumber or reuse the zero enumerator.
  - Zero may mean AUTO or UNSET only when 0 is not itself a legal value of the field. A
    count, threshold or mode switch where 0 is meaningful needs another form. Prefer
    inverting the field so that zero is still the default (a "disable" flag, not an
    "enable" flag). Otherwise use a distinct "unset" value outside the legal range, and
    make validation accept every documented value.
  - A required field left at zero is rejected with a returned error naming the field.
  - Where an env variable can also set the value, an explicit API value is authoritative
    and the env variable applies only under AUTO. State this in the field's comment.
    SM budgets are the documented exception: an operation's env variable and
    `NCCL_EP_COMM_SMS` beat its field (README SM-budget controls).
- A newer caller on an older library has its unknown tail dropped. If a field is a pure
  hint, dropping it is fine. If it changes semantics, the caller must be able to detect
  that, and a warning shown only under verbose logging does not satisfy this. Use the
  group-level version, a returned error for a nonzero unknown tail, or a capability
  query, and state which.
- Struct `size` says which fields are present. Do not add a version field to a struct
  that the existing group-level feature version or `size` can cover. Add one only when
  size cannot signal a semantic change and the existing version cannot carry it, and say
  why. Specify precedence and the error when the declared size and the version disagree.
  Bump the API version and update the frozen size macros in the same change that adds
  the field.
- Enums are append-only with explicit values for released ones; zero is AUTO or UNSET in
  one documented way, only where 0 is not a legal value (see above). No `_MAX` or
  `_COUNT` sentinel in the public header. Bindings must
  not cast unvalidated integers.
- Version checks fail by default and allowlist supported cases.
- A change to a handle's memory needs updates the size query in the same change, with
  documented invariants (monotone in inputs, alignment, behavior on reuse or re-import).
- Do not embed runtime-sized or non-trivially copyable objects by value in public or host
  structs; use a handle or pointer.

## 4. Lifetime, threading, streams and errors

- For every pointer or array argument or field, say whether it is copied at call time or
  retained until a named point (destroy, completion, stream completion), who frees it,
  and whether the host reads it at enqueue or on the stream.
- State thread-safety per entry point: thread-safe, one thread per object, or externally
  synchronized, and which calls may overlap. Default is unsupported.
- State stream and graph semantics per entry point: capture-safe, allocates, host-syncs,
  collective, and what is enqueued on a stream argument.
- State the contract after failure: outputs are untouched or null, the object remains
  usable or must be destroyed, and what other ranks observe. Initialize out-parameters
  before the first return.
- Use the error taxonomy consistently: invalid argument for a bad value or struct,
  invalid usage for wrong call order or state, system or internal for the environment.
  Public entry points never throw or abort (see `ERROR_HANDLING_REVIEW_SKILL.md`).

## 5. Cross-rank and cross-version behavior

- Config that changes symmetric layout or rank roles must agree across ranks: add it to
  a check, or state it is the caller's responsibility (preferred over an added
  consistency collective). Mismatched settings must not hang silently.
- Size intra-domain buffers, loops and checks by the LSA team, and inter-domain ones by
  the rail team, never by GPUs per host or host count. Name the exact quantity in
  variables and messages ("LSA team size", not "ranks per node"). A variable named for
  nodes that holds a team size is a naming finding.
- State and validate implicit symmetry assumptions at creation, collectively where they
  depend on per-rank values: a homogeneous topology (every host or team has the same
  size), contiguous rank ranges per team, and identical settings that select a wire
  protocol. A new setting of that kind joins the existing collective vote.
- Optional note, not a required finding: identical launch dimensions across ranks when a
  local index (such as a block index) is reused as a remote slot or barrier index. The
  library does not enforce this today and it is hard to enforce, so report it as `minor`
  or a question and do not insist. If the synchronization does not depend on the launch
  dimensions (the index only selects local work, or the remote index is derived from
  something the ranks agree on), there is nothing to flag as an issue. A question to the
  author is still fine to bring the assumption to their attention.
- State derived from an unvalidated assumption and never read is dead: remove it, with
  the work that computes it.
- Fields silently dropped by older libraries need a version check and documented
  contract.
- Runtime-dependent decisions that change which collective a rank runs need an agreement
  check.

## 6. Documented contract equals enforced contract

- Documented optional outputs and restrictions match what is enforced, reproduced on
  hardware.
- When a change starts rejecting something (or stops), update the doc comment on that
  field or enum in the same change, and re-read the comments next to every changed
  validation. A comment that says a field is ignored while the code returns an error for
  it is a finding.
- Document overflow, truncation and drop behavior for every capacity-bounded output and
  what the count outputs report then. Do not couple library semantics to how the caller
  invokes it (eager or graph); document capacity bounds as caller-visible.
- Public docs do not reference internal non-exported APIs. Document preconditions the
  change newly enforces. Name the actor (caller or library) for any required sync or copy.
- Check zero-size and boundary cases of every new size parameter.
- For window-backed tensors, validate that the view's offset plus extent lies inside the
  registered window before any remote operation, not only that the offset is aligned.
  Keep the choice of window and offset for a user-visible tensor in one function, and
  resolve a handle to a pointer late, at the call that owns the context.

## 7. Bindings, release and tests

- Bindings mirror the C contract: names, shapes, units, and sizes taken from the C
  header, never literals. A C change lists the matching binding edits, and the package
  imports without the optional compiled component. Regenerated artifacts build.
- A new, changed or removed public element needs a release-note entry; removal is
  deprecated for at least one release first.
- A new public field or argument has a compatibility test with an older-sized struct and
  a larger one with a nonzero tail, and a rejection test for an invalid value that checks
  the exact error code.
