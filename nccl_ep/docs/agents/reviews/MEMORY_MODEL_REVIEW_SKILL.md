---
name: review-memory-model
description: "Critic for memory-model, ordering, lifetime and graph-safety bugs in nccl_ep changes: missing or misplaced fences and their scope and strength, library puts, waits and GIN signals (strong versus weak), barrier coverage across LSA and GIN peers, proxy (generic vs async) races, epoch/bank protocol holes and stale words after timeouts, state read before it is published, buffers that outlive or are freed under their users, error and timeout exits that leave a protocol dirty, CUDA graph capture and replay hazards, programmatic dependent launch, and the API contract those imply (stream ordering, handle lifetime, cross-rank agreement). Use when reviewing any change that touches device kernels, epoch or bank logic, async copies (TMA, cp.async.bulk), handle-carried state, buffer allocation or registration, or host-side deferred work in nccl_ep."
---

# Memory-model review

Apply `REVIEW_METHOD.md` first.

Review a diff as a hostile reader of the protocol. For every piece of shared state,
ask who writes it, who reads it, what orders the two, what happens on the *next*
call, and what happens on an error or timeout. Report findings in the format in
`README.md`.

## 1. Build the protocol table first

Before judging any line, list for each flag, counter, bank and staging buffer:

| State | Writer (scope) | Reader (scope) | Transport | Ordering mechanism | Reset point |
|---|---|---|---|---|---|

- A finding is any row with no ordering mechanism, no reset before the next epoch, or
  a reset that some call order, mode or error exit skips.
- If the diff adds a row, extend the table; if it changes a row, re-derive it.
- Name the transport per flag (NVLink/LSA load-store, GIN/RDMA signal, host-visible
  memory). Each has its own visibility mechanism; when one flag serves two transports,
  argue for both. The weaker one decides.
- Initialization is a write and needs ordering like any other. Do not initialize a
  shared counter in one block while other blocks may already be updating it.

## 2. Publish and acquire

Ask first: after the code sees this flag, does it read memory that was written before
the flag was set? If yes, the writer needs release and the reader acquire. If no, the
reader usually needs no acquire. Before reporting any missing fence, also work out what
ordering the surrounding code already provides (2.5).

### 2.1 Payload, then flag: both sides are required

- Data a receiver depends on (counts, metadata) is written **before** the readiness
  signal so it rides the same release.
- The writer's flag store is release, or follows a release fence. The reader's flag
  load is acquire, or is followed by an acquire fence. A missing half is a `blocker`
  only if no larger scope supplies it: a fence may be implicit (a kernel boundary, a
  barrier, a release/acquire chain through a second flag, stream order). Apply the rules
  and the verification list in 2.5 before reporting one.
- One release fence followed by relaxed publish stores is valid. It does not need a
  release on every store.
- The position of a fence decides which accesses it orders. A release fence orders only
  the writes before it, and an acquire fence only the reads after it, so moving a fence
  across a store or load, or merging, hoisting or sinking one, changes the memory model
  the code implements even when every fence is still present. Verify the position
  explicitly every time: for each fence in the diff, name the accesses that sit before
  (release) or after (acquire) it and confirm they are the payload and the flag. Do not
  accept "there is a fence nearby" as evidence.

### 2.2 Flags that publish no data: no acquire on the waiter

- **Slot free (write-after-read hazard):** the consumer finishes reading a slot and
  sets "free"; the producer waits, then overwrites. The producer reads nothing the
  consumer wrote, so acquire on its poll protects nothing. The consumer still needs
  release or a fence so its reads complete before it sets the flag.
- **Credit:** a counter of freed slots, used as permission to send. Same analysis.
- **Abort poll:** a flag meaning "stop". No payload is published, so the load can be
  relaxed. The real risk is an unbounded wait (see `ERROR_HANDLING_REVIEW_SKILL.md`).
- If the same flag also guards data read afterwards, apply 2.1 and 2.3 instead.
- Do not report a missing acquire on these waiters. Do report a missing release on the
  signaler when its earlier accesses must complete first.

### 2.3 Polling loops: relaxed poll, then one acquire

- A waiter can poll with relaxed loads and issue one acquire fence after the loop
  exits. A relaxed poll followed by reads of the guarded data with no acquire after the
  loop is a `blocker`.
- Acquire inside the loop is a performance question, not a correctness one, and can be
  deliberate (its cache invalidation can act as backoff). Report it as `minor` and ask
  for a measurement, not a change.
- A volatile load is relaxed: use it to poll counters, not as the guard for data that
  follows. Do not read data that a peer can write in the same epoch through a
  non-coherent load path.

### 2.4 Library waits: verify the exact call

The code may use library waits (GIN signal wait, request wait) and hand-written polls.
Do not assume a library wait fences, and do not assume it does not.

1. Find what this exact call guarantees in the NCCL headers or implementation for the
   version being built against. Cite it in the evidence line.
2. Check that the guarantee covers the memory and scope the code reads afterward.
3. If it does, do not ask for an extra fence. If it does not, or the guarantee is not
   documented and cannot be read from the implementation, treat it as absent and require
   an explicit acquire after the wait (`major`).
4. If you cannot read the documentation or the implementation at all, state it as a
   question to the author.

For a library put or signal, also read what scope it releases at and how the caller
requests another one (GIN puts take a given and a required release scope). Device scope
is the library's default for sources in GPU memory, so do not ask for system scope on
such a put by default. Ask for it when the source is host-backed memory, or when another
agent outside the GPU and the NIC reads the data. Verify per version and backend.

For GIN signals, state whether each is strong (visibility implies all preceding puts to
that peer on the context are settled) or weak (only the bundled put). An unqualified
legacy signal type takes its strength from a communicator-level flag; name where that
flag is set. Do not accept "same-QP ordering" stated only in a comment.

### 2.5 Barriers, and ordering from a bigger scope

- The entry side of a barrier that gates peer reads is acquire; the exit side is
  release. Check the barrier's scope against the peers whose memory is read. A
  CTA-scope barrier guarding cross-GPU reads is a `blocker`.
- Barrier then publish is not cumulative by default. When one thread publishes for a
  group, each writer releases at the scope the final reader needs, or the publisher
  acquires at that scope after the barrier.
- Ordering acquired by a subset of threads (one lane polls, then shuffles) must reach
  every thread that reads the guarded data (`__syncwarp`, or a per-thread acquire).
- Before removing or shortening a barrier or wait, state which peer classes the remaining
  wait covers (LSA or GIN, rail or world) and what ordering requirement each had. A
  subset-team barrier after a superset barrier is removable only if it creates no new
  ordering requirement.
- When one thread flushes and another signals, a cooperative sync is needed between flush
  and signal; same-thread flush and signal need none. Check that the coop type passed to
  the flush and to the signal match. Fix every sibling variant, ideally in the shared
  flush helper.
- A completion or progress signal published by one thread needs a barrier or `__syncwarp`
  showing that all threads have left the work body.
- A missing fence is a finding only if no larger scope already provides the ordering.
  List the surrounding ordering from smallest to largest: the fence itself, the barrier
  (CTA, cluster, grid, system), a release/acquire chain through a second flag, the
  kernel boundary (implicit release at completion, acquire at the next start), and
  stream order, events or graph edges. If a bigger scope completes the pairing, name it
  in the evidence line as an assumption, and verify it:
  - the writes are in a later kernel, not the same kernel;
  - every participating rank crosses the boundary (one GPU's kernel boundary does not
    order another GPU's accesses);
  - both kernels are in the same stream, or ordered by an event or graph edge;
  - programmatic dependent launch does not overlap them (see 3);
  - the ordering also holds on error and timeout exits.
- If the code's own comments express doubt (a TODO or "double check"), report it as a
  question to the author.

*Example.* A receive path polls peer flags relaxed and issues one system-scope acquire
fence. It issues no release, because nothing it wrote earlier must be ordered before a
later signal. Resets of sender-side counters are not fenced and rely on the next
epoch's writers being a later launch on the same stream. That holds only while the
launches stay ordered. Adding a second stream, enabling PDL or fusing the kernels breaks
it, so name the assumption in review.

### 2.6 Wait coverage: GIN does not cover LSA

A wait that completes remote GIN peers does not cover LSA peers, whose memory is read by
load/store. If the code reads both kinds of peer after one wait, ask which wait covers
each. A single GIN wait guarding an LSA read is a `blocker` once you have confirmed for this
version that the wait does not cover LSA peers; until then ask. This rule comes from past
reviews, not from a guarantee stated here.

### 2.7 Strength, scope and compiler ordering

- Fence scope equals the widest agent that shares the data: CTA, device, or system (NIC,
  peer GPU, host). Name the consumer and its scope in the evidence line. A block-scope
  fence after observing something written by a NIC or another CTA is wrong unless the
  structure is single-CTA by design; that assumption is written next to the fence. A
  flag or counter in shared memory read by warps of the same CTA needs only block scope.
- A zero-byte or otherwise empty operation keeps the declared scope: a signal publishes
  earlier writes whatever the payload size, so do not skip the release because the byte
  count is zero when the API's scope contract is payload-independent. A per-chunk
  override starts from the caller's requirement and only strengthens it. A fence on the
  first chunk only is valid for a single issuing thread with no writes in between.
  Annotate a deliberate strengthening.
- Fence strength is justified by causality and scope. Cross-GPU visibility needs
  system scope, not GPU scope. A stronger fence already in place covers weaker stores:
  use the relaxed form and say why. A barrier that only guards "readers are done with
  an input" can be relaxed once the data is consumed.
- Fence and barrier asm primitives always carry the `memory` clobber, so the compiler
  cannot reorder around them. Never make it an opt-in.

### 2.8 Checklist, per fence in the diff

1. What data does this flag guard, if any?
2. Who reads it, and on which proxy, scope and peer set?
3. What ordering does the surrounding code already provide, at every scope up to the
   kernel boundary, and does each assumption hold for every rank and path?
4. What exactly does any library wait guarantee, verified for this version?
5. Is the fence present on every path, including timeout and error exits?
6. Does the fence match the proxy of the write (section 3)?
7. Does the wait cover every peer that is read?

## 3. Async copies and proxies

- Name the producer and consumer proxy at every crossing. Generic writes read by async
  (TMA) loads, and async-filled data read by generic code, each need the matching proxy
  fence. Question a fence you cannot attach to a proxy pair.
- Before reusing a staging buffer or pipeline stage, wait for the operation that reads
  or writes it to **complete** (not merely for its source to be released), and show the
  wait distance covers the reuse distance.
- Before publishing readiness for data an async store wrote, wait for store completion
  (the bulk-group wait), then fence the async proxy, and only then publish; the proxy
  fence does not replace the wait. A flag store after an async write with no wait, or a
  wait with no proxy fence before a generic read, is a `blocker`.
- Producers and consumers of a pipeline agree on a participant set fixed before the
  loop starts. A mask that changes mid-reduction (timeouts, peer loss) can deadlock;
  snapshot it per CTA.
- In-place rewrites never overwrite elements other threads have not yet read.
- **Programmatic dependent launch (PDL) and any relaxed launch ordering** remove the
  guarantee that kernel exit orders everything for the next kernel. When it is enabled,
  or when a change makes it possible, check:
  - Before its wait, the dependent overlaps the primary. Code a thread runs before its
    `griddepcontrol.wait` must not read anything the primary produces, and must not
    write memory the primary still reads or writes. This includes protocol state:
    counters, epoch and bank selection, flags, and anything reset by a last-arriving CTA.
  - Every thread that touches predecessor-produced data is ordered after a wait, by
    executing it or by a barrier after a thread that did. The wait belongs before the
    first such access, not necessarily first in the kernel.
  - The trigger (`launch_dependents`) is a scheduling hint, not a visibility mechanism:
    it does not make data visible and does not guarantee an early start. A dependent of a
    kernel that triggers must wait. Reading data without a wait through the
    fence-before-trigger path requires the memory-domain and scope conditions in the PTX
    ISA; verify them.
  - Assumptions of the form "the next call is a later launch" no longer hold for the
    overlapped part, for example a counter reset done by the previous kernel's elected
    CTA. Name each such assumption and check it against the wait position.
  - Re-justify co-residency and stream-order arguments (sections 2.5 and 4) for the
    overlapped kernels.

## 4. Epochs, banks, counters and participation

- Every flag and counter has a defined reset point reached on **every** path: all call
  orders (send-only, recv-only, back-to-back, either op first), all modes, and all
  error exits. Use one shared helper for cleanup; reset every bank the next operation
  reads, including read-done acknowledgements, before publishing readiness.
- All CTAs capture the epoch before any CTA advances it; advancing needs a grid sync or
  an election that provably has all CTAs resident.
- Inter-CTA waits and "last arriver" elections require co-residency. Prove it
  (cooperative launch, occupancy bound) or use a scheme that does not need it. See
  `PERF_COMPLEXITY_REVIEW_SKILL.md` section 5 for launch-side checks.
- Collective barriers are reached by every participant on every path, independent of
  per-role or per-mode conditions.
- Reduce the per-thread status of a cooperative call (a collective result, an error flag)
  before any divergent early return, so a thread that returns does not leave others
  waiting on or acting on a value it would have supplied.
- Keep barriers outside index guards. A warp or group with no work still participates in
  a CTA-wide barrier, or the participant count is recomputed over the active set (see
  `PERF_COMPLEXITY_REVIEW_SKILL.md` section 5).
- A function that contains a bootstrap or collective barrier runs with one thread per
  rank, or the barrier counts every thread that executes it.
- Counters are balanced per role (every add has a matching reset or decrement, inactive
  roles included) and width-checked. Show the worst-case call count to wraparound, use
  wrap-safe comparison, and test with the counter seeded near the maximum.
- Do not reuse a counter across phases without a sync between them.
- For every flag or count word that the next use of a bank treats as "arrived", show it
  is cleared on every path that did not consume it (timeout, masked peer, skipped
  receive) before the bank is reused. If the design relies on consumer clears instead of
  a bank wipe, state which condition guarantees no unconsumed word (for example, masks
  are rejected on that path).
- For epoch-tagged flags with a finite space, show a wrapped value cannot match a stale
  line (clear at rollover or add sequence bits). Do not share receive buffers across
  algorithms that keep independent epoch or flag counters.
- When a feature breaks an index-space identity (for example experts = local x ranks),
  audit every bank, counter and offset sized from it.
- Size every buffer or bank to the maximum over the modes and consumers that can use it on
  this group; do not size for modes that cannot occur there (see the perf skill, section 2).
- When two capacity bounds exist for the same data (for example a receive-slot limit and
  a smaller limit the index maps or staging were built for), the count published to
  readers is clamped to the smaller bound. For each published count, name the bound its
  consumers' maps cover; a reader that trusts the larger count reads slots no writer
  filled. Expected evidence: the count's producer, the map or buffer extent, and the
  case where the first exceeds the second.
- A flag-based gate (ready value, expected count, phase number) takes at every call site
  the value the helper's documented protocol requires for that mode, given whether the
  flag is reset or monotonic. For each call site in the diff, read the helper's comment
  and the flag's reset path, and show the passed value matches in every mode branch
  (fused or unfused, with or without the extra sync). A value that is right for one
  branch and passed unchanged in another is a finding.
- Before kernel exit, confirm remote ranks have finished reading buffers the next call
  may overwrite.
- Degenerate inputs (duplicates, zero counts, empty ranks) are rejected, defined, or
  zero-filled so no kernel reads memory that was never written.

## 5. Errors, timeouts and partial failure

- For every timeout, abort or masked-peer exit, state the post-state of every row of the
  protocol table. Either the protocol heals itself (epochs still advance, banks still
  reset) or the group is marked poisoned, with a sticky flag checked at the next entry.
- Validity flags ("maps built", "handle ready") are set last and stay false on every
  later error and fallback path.
- A failure partway through init or teardown must not leave a collective half-entered
  or a resource double-freed. Teardown order mirrors allocation order. Collective
  register, deregister and allocate calls cannot be skipped by one rank's error.
- Zeroing memory that peers may write must complete, and be followed by a cross-rank
  barrier, before it is exposed. A peer's first write must not be wiped.

## 6. Lifetime, ownership and graphs

- Mutable per-operation state lives in the handle or the call, never in group-level
  state that concurrent handles or calls can race on. Rewriting group-level pointer
  state on every call is a race.
- State that crosses a boundary (handle import, caller-built blob, saved snapshot)
  records how it was produced (version, layout parameters, chunking), is validated or
  recomputed on use, and is captured only after the producing operation completed with
  the copy stream-ordered.
- Deferred continuations (a send-only phase, a lambda run later) capture by value, never
  pointers to stack locals.
- State read from freshly allocated memory is zeroed once at init; never assume
  counters read as 0.
- Free, move or re-register a buffer only after proving no live handle or captured
  graph still holds its address or window. A change that can reallocate must say which
  handles and graphs it invalidates, and reject the call while any is live or capturing.
- Free memory through the allocator that produced it (a user allocator hook through the
  hook, `cudaMalloc` with `cudaFree`, `ncclMemAlloc` with `ncclMemFree`, host-pinned
  with its host free). When a change moves a buffer to another allocation path, change
  every free site.
- Every host function reachable from a launch path is either capture-safe (async calls
  on the user stream only, no allocation, no sync) or explicitly rejects capture.
- Host-side state advanced at capture time is baked in once. Per-replay state (epochs,
  banks, counters) lives in device memory and is advanced by a kernel.
- Remotely targetable resources are torn down only after a cross-rank quiescence point;
  a local device sync drains only local work. User-owned zero-copy tensors state when
  the library, and its peers, stop touching them.
- Host-visible flags written by the device are released at system scope, read only
  after stream completion, and the memory is freed only after the stream completes.
- Reject mixed-capability operands (some window-backed, some not) up front, naming the
  argument.

## 7. API contract implied by the memory model

Check that the public header and docs state what the protocol needs:

- The stream and ordering contract: which streams may use a handle or group, whether
  concurrent use is supported (default: no), and what the caller must do to order
  consumers after producers.
- What is and is not supported on handle reuse, reordering and replay. Do not overstate:
  if the library cannot control launch order, say so.
- Dispatch and combine agree on routing-metadata format; kernel selection accounts for
  the preceding operation or fails on mismatch.
- Config that changes symmetric layout or rank roles is agreed across ranks (checked,
  or documented as the caller's responsibility).
- A change to ordering or synchronization invalidates earlier performance numbers;
  re-measure.

## 8. Comment the non-obvious ordering

One line per fence or barrier saying what it orders against what. Comment intentional
over-writes into padding. Require the ordering to be explained, not only named:
- A split load and store used instead of an atomic read-modify-write is valid only with
  exactly one writer; say who it is.
- For a benign race, say who races and why the result is the same either way.
- For a deliberate scope escalation, say why a wider scope is used where a narrower one
  looks sufficient.
- When a fence is omitted or deleted because a callee already provides the ordering, say
  which callee and what it provides, and keep that true when the callee changes.
