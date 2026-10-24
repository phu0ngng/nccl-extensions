---
name: review-code-organization
description: "Critic for code organization and readability in nccl_ep changes (including design-document review): growing monoliths (structs, files, functions, kernels), duplicated logic and missed reuse of existing helpers and primitives, inline assembly, wrong layering and include direction, template-parameter explosion, policy mixed with mechanism, scattered config and env access, unclear ownership of state, magic numbers and units, misleading names, scope creep in the change, and comment quality (including issue ids that the public mirror cannot show). Use when reviewing any nccl_ep C++/CUDA diff; pair with cpp-style (CODING_SKILL.md) for mechanical style rules."
---

# Code-organization review

Apply `REVIEW_METHOD.md` first.

The mechanical rules (no default parameters, no nested ternaries, braced single
statements, short comments, no inline asm outside `device_primitives.cuh`, reuse and
duplication basics) live in `../CODING_SKILL.md`. Apply them, then look for the
structural issues below; section 2 extends the reuse and assembly rules. Report in the
`README.md` format.

## 1. Structure stays manageable

- A feature that adds fields groups them (sub-struct or params struct with helpers), in
  its own header or source when self-contained. Do not append to flat lists, giant
  argument lists or one huge init function. On a changed kernel or JIT launch call site,
  count the scalar arguments added: when an existing params struct already carries them,
  pass the struct whole.
- A function or kernel that gains another mode splits by mode instead of branching
  inside. On the added lines, count the new `if`/`?:` tests on the mode flag inside
  existing functions; two or more scattered tests are a finding that asks for one
  per-mode `switch` or a helper per mode. Public entry points and long bodies get named
  helpers for self-contained
  blocks. Split only where it removes duplication, a mode branch or a distinct job;
  more files is not a goal.
- Do not add a new subsystem to a file that is already very large. Put it in its own
  file, with includes at the top.
- Extend an existing abstraction for a new variant instead of adding a parallel one.
  Code shared by two kernels or host paths lives in one function.

## 2. Reuse, duplication and primitives

**Search before accepting new code.** For every new block of logic, search the existing
kernels and host paths for the same logic, using the distinguishing operation and not the
names. When it exists, report each location and propose the modularization: a shared
helper or a template parameter, with the extracted signature and the call sites that
change. Aim for an almost-ready solution in the Suggest line, small enough to apply as a
patch. When a pattern is found in one kernel, grep every sibling kernel for it in the same
change.

**Scan the primitives and utilities aggressively.** Before accepting any new helper,
arithmetic idiom or small utility, read the device primitives header, the shared common
header, and the LL and HT primitive headers. Report any custom implementation of something
that exists, naming the existing one. Near-copies of an existing primitive with another
scope, order or width are findings: extend the existing family.

**No inline assembly.** Any `asm` on a changed line outside the primitives header is a
finding. Look for a
primitive-based solution, in this order: an existing wrapper in the primitives header; a
standard or libcu++ facility (`cuda::atomic_ref`, `cuda::ptx`); a new named wrapper added
to the primitives header. The suggested fix may touch code outside the diff, for example
moving an existing use site into the primitives header so that both callers share one
definition. Existing assembly outside the primitives header is known debt: do not
re-report an instance the change does not touch, but when the new code needs the same
operation, propose moving the existing site too.

**Scope exception.** The README ("Staying quiet") relaxes the scope rule for these three
checks only, for existing code the change relates to. A fix that moves or merges existing
code to avoid a second copy goes in a
separate commit before the feature commit, in the same change.

## 3. Layers and dependencies

- One include direction: public API, then core, then adapters, then device. Device
  headers do not include host-only headers (env parsing, streams, string building).
  JIT headers generate code; they do not carry host policy.
- Policy and mechanism are separate. Mode and kernel selection is a pure function of its
  inputs (no allocation, no launch), usable by one selection point and testable without
  a GPU. Setup and launch consume its result.
- Parsing and policy are separate layers. The env/config layer reports what the user
  set; the core decides. Every knob is read in the one env/config module, resolved once
  at creation into a config struct, and passed down. No direct environment reads in
  device or adapter code.
- Every internal symbol lives in the library's namespace structure; file-local code is
  in an anonymous namespace. Only the public header is public.
- A concept that exists in more than one variant (low-latency and high-throughput paths)
  is changed in all of them, or the change says why only one is affected. Shared
  concepts live in shared headers.

## 4. Compile-time parameters

- A diff that adds a template bool or enum states which combinations are reachable and
  rejects the rest in the selector. Each parameter multiplies instantiations, JIT keys
  and the test matrix. Past about three flags, use a traits struct, or a runtime
  argument when the cost is off the hot path.
- Generated or string-built kernel source comes from one shared prologue and a typed
  instantiation descriptor, not hand-pasted signatures. Never hand-edit generated or
  embedded sources; regenerate and say so.
- Define each feature or version gate once, as a named macro or constant in one config
  header, and use only that name. Wrap repeated instrumentation in one helper. A new
  gate needs a build leg that exercises it.

## 5. State and ownership

- Every new resource, cache or global names its owner and lifetime (group, handle,
  process) and is created and destroyed by that owner's create and destroy pair.
  Allocate and free in one place; no hidden globals or thread-locals.
- Remove fields, parameters, conditions and code paths the change makes redundant or
  always-constant. A new debug or escape-hatch knob needs an owner and a removal
  condition.
  - When the diff deletes a workaround or code block, list what only that code used
    (includes, helpers, locals, constants, flags) and grep each for remaining uses; an
    unused one left behind is a finding, as is a stray blank line at the deletion.
  - When a flag or predicate is repeated or wrapped at several sites, grep for every
    place that can set it; if none can, report it as dead and ask for removal.
- State each invariant once, beside the constant or struct it constrains, backed by a
  `static_assert` or host validation. Other sites say "see X". A device-side range check
  that silently encodes a scope restriction (for example one team, one rail, one mode)
  needs a comment at the check and a host-side rejection of the unsupported case.

## 6. Constants, units and names

- Name every non-obvious literal and derive dependent constants from their source.
  Shared constants live in a shared header; use typed `constexpr` over macros and
  templates over macros. Non-trivial bit manipulation sits behind a named helper with
  the layout documented.
- Encode units in names (`_bytes`, `_elems`, `_tokens`), use `size_t` for byte counts,
  convert only at a named boundary helper.
- Derive each offset or index from the value it depends on, in one place.
- A name matches what the code does and where it lives: domain terms, one term per
  concept across the change, functions named for what they validate or compute. No
  feature or mechanism names in generic code, no reuse of one feature's constant for
  another, no names for mechanisms the code does not use.
  - Legacy names: when the change alters what a variable, field, kernel or invoker holds
    (for example a "dense" buffer that may now be compact), grep its uses and ask for a
    rename that matches the new meaning; avoid dense/sparse-style wording for new formats.
  - New kernel and invoker names start with the owning operation, then the semantics.
  - Do not reuse a domain term (lane, warp, rank, slot) for a different granularity than
    the established one in device code.
  - Abbreviations in new identifiers and enum values are expanded in a doc comment at the
    definition, or spelled out; the same holds for comments (see section 8).
- Inputs are explicit and mean one thing. Do not infer a mode from an unrelated null
  pointer, overload one parameter for two meanings, or add special cases that exist only
  for tests. No test-only code in production headers.

## 7. Scope of the change

- Unrelated changes (refactors, renames, buffer-sharing changes, out-of-scope features)
  go in a separate change. Generic code does not check mode-specific conditions.
- After a rebase, migrate to the current helpers and extend shared validators to the
  new case; do not leave both old and new mechanisms.

## 8. Comments

- Comments describe current behavior and the reason, briefly. No history, no remarks
  about other libraries, no narration of one test or change.
- No issue or tracker ids, MR numbers or links in comments: they are not visible in the
  public mirror. A temporary workaround is marked in plain words together with its
  removal condition (for example "remove when the minimum supported toolkit provides X"),
  so it can be found and removed without an id.
- Document non-obvious invariants where they live: thread or lane mapping of
  collectives, layout kept for compatibility, variant contracts.
- Keep comments accurate: update those the change invalidates.
- Comments and docs name things in full: no slang or unexplained shorthand ("A/B" for two
  modes, bare acronyms) and no undefined terms. Name each mode or path explicitly.
- Update every build list a new file belongs to (Make, CMake, test lists).

## 9. Formatting of added lines

- Indentation is checked on every added or changed line, not only on braces: `else`,
  `else if`, continuation lines and `case` bodies align with the surrounding block (see
  CODING_SKILL section 5). Compare each added line's indent with its neighbours and report
  each mismatch; also ask for `clang-format` on the touched lines or a pre-commit hook
  when more than one occurs.

## 10. Design documents

PLC documents are written under `../PLC_DOCS_SKILL.md` (never edit an existing one; a new
directory carries the MR number). When the diff adds or changes a document under
`docs/plc`, read it against the code and check:

- Directory name, title and index entry carry the MR number, and no existing PLC
  `README.md` was edited. Report any that do not.
- Scope: every described element is added or altered by this change; a pre-existing
  element presented as new, or a path the change does not support (name the transport
  or mode the document mentions), is a finding. The supported scope is stated once.
- Why before how: each section gives the high-level rationale (who needs which
  information, when it must be available, and so which pattern follows) and not field
  names or low-level dumps. Report a section that lists raw fields without a reason.
- Terms and constants are defined before first use (meaning, scope, origin), and data
  structures before the diagram that uses them. Each sentence is understandable on its
  own; flag jargon and out-of-scope mentions.
- Precision: each step names the API call or kernel launch it runs in, a collective is
  named exactly (for example AllGather versus All-to-All) with what each rank sends, and
  who performs any sync or copy (caller or library) is stated. Check each against the
  code and report mismatches.
- A change to a wire or buffer format opens with an end-to-end data-flow diagram of each
  leg (buffers, conversion kernels, writer and reader).
