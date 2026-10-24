---
name: review-perf-complexity
description: "Critic for algorithmic and overhead cost in nccl_ep changes: work sized by buffer capacity instead of the actual batch, wrong bound for the regime (latency vs bandwidth), recomputation, extra launches, passes or serial network/fence hops, skew and scale blindness, atomics contention, optional features that cost the default kernel (pay-for-play), alignment and tails, registers, spills and integer width, warp collectives, division and rank masks, topology-size assumptions, early CTA exit and cooperative launch, JIT and host hot-path overhead, graph capture and signalling cost, env override versus automatic default, and weak evidence. Use when reviewing any change that adds a kernel, a launch, a copy, a sync, a loop over peers/experts/tokens, or a new mode on a hot path in nccl_ep."
---

# Algorithmic-performance review

Apply `REVIEW_METHOD.md` first.

For every new element in the diff (loop, kernel, launch, copy, sync, allocation, message),
state its cost as a function of **actual** quantities (tokens received, experts hit,
ranks present), then ask whether the same result needs less. Put the cost expression in
the evidence line of each finding. Report in the `README.md` format.

## 1. Which bound, which regime

- State the bound the change targets (latency, link or memory bandwidth, issue rate) per
  regime (small latency-critical batches vs large throughput batches) and the expected
  change as bytes over bandwidth or serial hops times round-trip time. Reject changes
  that do not move that term: a bandwidth saving in a latency-bound path is noise.
- Count dependent hops on the critical path per message: system-scope releases, signal
  or counter-then-flag handshakes, extra RDMA signals. Justify each added one; fold
  counts and flags into the payload where possible.

## 2. Capacity versus actual (the most repeated bug class)

Work, grids, copies, checks or buffers sized by capacity when a device-side count bounds
the real work.

- Launch grids, copy sizes and pre-checks use the actual count, not the buffer capacity,
  especially where the true count arrives late. Fold an extra capacity-sized pass into
  an existing one.
- Loops over experts, ranks or peers are sized by who is active; role-specific work is
  gated so inactive roles issue no traffic.
- Size the grid from the actual work items, not from capacity or from a peer count that
  includes excluded peers. Map CTA ids to resources modulo the resource count.
- A feature claimed to leave a mode unchanged preserves that mode's exact sizing and
  behavior. Gate padded strides and extra buffers on the mode that uses them; do not
  allocate buffers nothing reads.

## 3. Complexity

- Do not recompute per iteration or chunk what can carry state; do not nest passes where
  one pass over the smaller set gives the same result. Ask which guarantee the API does
  *not* make (ordering, gap-free output) and whether relaxing it simplifies the work.
  For each per-thread loop in a decode or conversion kernel, write its trip count; a
  loop whose bound is data-dependent and that contains another data-dependent scan is a
  finding when one pass over the smaller set (such as top-k) gives the same result.
- Replace serial cross-thread loops with warp-level collectives, or with a strided share
  of the work (lane i takes items i, i+32, ...) instead of scan-and-filter, and loops
  with closed forms where available. Compute per-block and per-token quantities once.
  Check the preconditions: all 32 lanes must participate for `__match_any_sync`, one
  ballot covers at most 32 items, and a dedup that negates an index leaves index 0
  unchanged.
- No runtime integer division on the per-token path for a loop-invariant divisor. Use a
  precomputed reciprocal (multiply-high), shifts or unsigned operands, and keep 64-bit
  helpers 64-bit.
- Do not broadcast to every peer or block when each consumer reads a slice.
- Cost the change at the target extremes (for example 8 / 72 / 256 ranks, hundreds of
  experts, large top-k) for representability as well as work. Flag work linear in ranks
  per thread, or quadratic overall. Never index a fixed-width mask by rank or expert id:
  size masks from the topology maximum as multiple words and `static_assert` the
  supported limit.
- Do not bake the NVLink domain size into the code. A fixed rank-count constant that
  encodes a product's domain size is a finding, even if asserted and host-checked.
  A limit that is a real resource limit (shared memory for per-rank arrays, mask words,
  arrival bits) is derived from that resource, computed or queried at init, and
  explained in a comment. A limit that cannot be removed comes from one definition
  shared by host and device, and the host rejects larger values before launch with a
  returned error that names the limit; matching literals on each side are a finding.
  When the user has not chosen a path (AUTO), a fast path that cannot support the team
  may fall back to a general path and log it. When the user forced the path, for example
  by an env setting, fail with an error; never fall back silently. For every fixed cap,
  check the largest supported team, who sets the limit, and what happens above it.

## 4. Skew and contention

- Require a skewed-routing run (hot expert, empty ranks, duplicate routes) beside the
  uniform one, and state the worst-case work per CTA and per rank. The slowest rank or
  expert sets the time.
- Reduce within the warp or CTA first, then issue one atomic per warp or CTA. Gate
  diagnostic atomics and counters so they cost nothing when off.

## 5. Launches, syncs and hot-path additions

- New functionality is pay-for-play. No unconditional launch or graph node in an
  existing hot-path API, and no added cost on one path to serve another. A new mode
  adds no compile, launch or capture cost to existing modes.
- Optional features (instrumentation, timing, stats counters, abort or timeout handling,
  extra checks) cost the default kernel nothing. Put them in a separate host-selected
  variant or compile them out; a runtime flag inside the hot kernel is acceptable only
  for a single predictable store that provably folds away. Demand registers and
  small-message latency of the default kernel with the feature off, before and after.
  Keep the original wait loop when abort or timeout handling is off, and avoid
  host-memory writes in the kernel prologue or epilogue and clock reads around fences
  beyond what the feature needs. Ask for a CI check that compares the default kernel.
  Variants for observability multiply instantiations, so require the measured overhead
  of the flag when off before adding one. Report a suspected cost as a question unless
  there is a measurement.
- No auxiliary SM-wide kernel next to a communication kernel; prefer copy engine or
  fuse. For each new standalone conversion, layout or reformat kernel before or after
  dispatch or combine, state the extra launch and the extra read and write of the full
  tensor (bytes over bandwidth), and ask whether the work fits in the producing or
  consuming kernel's prologue or epilogue, or in the neighboring conversion kernel;
  accept a separate kernel only with that reason or a measurement. Avoid grid-wide syncs;
  prefer per-CTA counters or election where the protocol
  allows.
- Flag any new host sync or blocking query on the hot path and require its side effects
  to be documented.
- Resolve mode choices on the host and pass them by value; do not make the device load
  data to decide a branch. Prefer compile-time selection over runtime branches when the
  mode is known at launch, but:
  - count the instantiations the change adds. Each new template parameter multiplies
    existing variants and JIT cache entries. Do not compile combinations no caller can
    produce, and compare with the measured cost of the runtime branch;
  - when a check is moved out of a loop or to the call site, list every caller and
    confirm each guarantees what the check handled. A bound computed on the device
    cannot be validated on the host;
  - represent an absent option with a tag type or an explicit flag, not a sentinel value
    that a legal input can equal;
  - keep fast-path kernels free of tail handling and route irregular shapes to a
    general kernel.
- Spin-waits need backoff, and independent waits overlap instead of serializing. Defer
  waits and setup to just before first use; keep minimal lookahead so loads overlap
  in-flight stores.
- After any early CTA exit or role split, recompute every partition and barrier
  participant count over the active set. An early return must not skip a barrier that
  other threads wait on.
- Use cooperative launch when the kernel needs co-residency: a grid sync, or inter-CTA
  waiting or a last-arriver election. State which, from the protocol (who waits for
  whom) and not from a comment. Do not flag a cooperative launch for a kernel without a
  grid sync until you have checked for the others. A kernel that needs co-residency and
  is not launched cooperatively can deadlock. Cap the CTA count by the kernel's own
  limits, and check what the launch does when the grid cannot be resident. See
  `MEMORY_MODEL_REVIEW_SKILL.md` section 4.
- Tuning heuristics are per kernel and per op; do not borrow another kernel's picker or
  knob. When the diff calls an existing launch-config, occupancy or chunk-size helper
  from a new kernel, name the bottleneck that helper was tuned for (load latency, shared
  memory limited stages, issue rate) and check it is the new kernel's bottleneck at the
  smallest and largest supported hidden size; list the warps per CTA, CTAs per SM and
  waits the helper yields for the new kernel. If it differs or is unmeasured, ask for a
  kernel-specific picker or an A/B against it. Ship a measured automatic default that is
  good without any env setting: a new opt-in performance feature (not an override of an
  existing automatic choice) that leaves the default path unchanged is a finding unless a
  sweep shows the default is already best. Where variants cross over by size, select with
  a measured model, not a fixed pick. An env override for a size cutoff or tuning
  constant is acceptable. It applies when the config field is AUTO (an explicit config
  value is authoritative), and then it is obeyed exactly: the automatic logic must not
  alter it. A forced setting that cannot be honored returns an error or logs, and does
  not silently fall back. Report a knob
  that is the only way to avoid a performance cliff, since users who set nothing then
  get the slow behavior, and a new knob for an optimization still in flux (use an
  internal flag until the design settles). Check that a changed threshold is conditioned
  on the hardware or feature that justifies it.

## 6. Memory access, resources and compilation

- State the access width and alignment (for example 16 B) of each new global, shared or
  remote load or store. Check what happens at unaligned sizes and tails, coalescing,
  shared-memory bank mapping and swizzle, and stage depth against latency.
  - Check every pointer and the row stride: source, destination and, for window-backed
    tensors, the window offset, in both directions of a transfer.
  - Bound the last access of a wide or emulated vector load by bytes against the end of
    the allocation.
  - Tail behavior is the same in debug and release. If a guard exists only in debug
    builds, state what release does with the rejected input (the device assert is not such
    a guard today: it traps in every build, see the error-handling skill). Prefer
    rejecting on the
    host with a returned error, so the kernel is never reached.
  - For packed or sub-byte types, assert the element size and the union layout at
    compile time.
  - Accept a complete host-side alignment check on both sides of a transfer; do not ask
    for a device-side duplicate.
- Reserve only the memory a variant uses, whether shared memory, registers or global
  buffers, and count only that in budgets. Do not move hot data to a slower space (for
  example kernel arguments to global memory) without a measurement.
- Report registers, spills and shared memory per kernel before and after. When `nvcc`
  is available, build both revisions and report them yourself (`REVIEW_METHOD.md`).
  Require explicit launch bounds on new kernels. Derive launch sizing from the resource
  budget, not constants.
- A per-thread array is free only if every index is a compile-time constant after
  unrolling. A runtime index puts it in local memory. Check how each new array is
  indexed and check the spill count; do not flag an array because it exists.
- Keep device integers 32-bit unless a bound proves they stay below 2^31. Widen values
  that scale with tokens x hidden x element size, and keep address math in `size_t` or
  `uintptr_t`. Never narrow inside a function whose contract is 64-bit, even to save
  registers. Before asking for a narrower type, check the largest value the variable
  can hold.
- Count distinct JIT specialization keys; state which values are in the key and which
  are runtime arguments. The first call must be off any timed path and the cache must
  stay hit across handles.
- Hot host paths allocate nothing, use no maps, vectors or string building, and size
  everything at init. Query device attributes and occupancy once and cache them. The
  update path asserts and does not recompute layout. A per-call query that is required
  for correctness, such as a capture-status check, is acceptable; measure its host cost.
  Measure host launch time, not only GPU time.
- Keep a path graph-friendly: stable pointers and arguments, no host-visible count, no
  data-dependent grid. Compare replay timing. An operation that can be captured does no
  blocking D2H copy or sync. Size for the worst case on the device, or detect capture
  and return a clear error for the mode that needs the host count. State which, and cost
  the worst-case sizing against section 2. Check that library-internal eager work runs
  under relaxed capture mode.

## 7. Network behavior

State messages per peer per call, bytes per message, in-flight depth and the flow-control
or credit mechanism. Many small puts or unbounded queue depth risk WQE-rate limits and
head-of-line blocking; batching trades latency for rate.

- Signal per pipeline chunk so receivers start early. For barriers across many ranks,
  compare constant-size state with a phase bit against per-peer slots, and ask for
  latency numbers in the small regime. Coalesce only counter-style trailing signals.
- Serve in-domain peers by load/store and remote peers by the NIC path, with separate
  buffers so both progress concurrently. Ask the reason for any change that merges
  them.

## 8. Overlap

State the SMs used and the kernel duration, and the end-to-end effect with concurrent
compute. Compare SM copies against copy-engine or NIC offload as a trade-off, not a
default.

## 9. Evidence required

- Regression data for every shared kernel the feature touches, including paths that
  should be unaffected. Verify that the benchmark measures only the path under test.
- Sweeps across SM budgets, thread counts, platforms and representative model shapes
  before choosing a default. Tuning on one architecture does not transfer.
- Quantify the benefit before adding machinery, and prefer the existing implementation
  when gains are inconsistent. Do not enable a path by default on a size saving alone
  when measured regressions exist.
- A cost claim needs data: `ptxas -v`, SASS or a measurement. Produce it when you can
  (`REVIEW_METHOD.md`), otherwise ask. A suggestion without a stated cost, or one that
  depends on an architecture-specific measurement (tail unroll, staging depth, JIT
  versus precompiled), is a question; with a stated small cost it is `minor`; it is never
  `major` without a measurement. Before citing a past
  decision as the standard, check what the code does now.
- Report the statistic compared (mean, p50, tail) and use the same one on both sides,
  with the first iterations trimmed, per-rank min and max, and a profiler artifact
  (timeline or counter) for any claimed cause.

## 10. Worked example: compacting a sparse row

A change compacted per-expert weights into a per-token row. Its decoder scanned local
experts in ascending order and searched the token's top-k list for each match, costing
O(experts per rank + hits x top-k). Review asked whether ascending order or gap-free
rows were promised by the API (they were not) and for fewer passes and less divergence;
a follow-up walked the top-k list once and packed matches at the start of the row,
cutting the work to O(top-k). Method: write the cost, list what the API does not
guarantee, then ask what a single pass over the smaller set would cost.

## 11. Worked example: ranking tokens per expert inside a warp

A lane needs its position among the lanes whose tokens route to the same expert. A
serial version loops over earlier lanes: O(lanes) per lane, and it diverges. Instead take
one `__ballot_sync` of the predicate. Each lane's rank is the popcount of the ballot bits
below its lane id, and the popcount of the whole ballot advances a running per-warp
offset, read before it is updated. Cost: one ballot and two popcounts per expert per
iteration. Preconditions: every lane executes the ballot, and one ballot covers at most
32 items. Method: write the serial cost, name the preconditions of the collective, then
check that the running offset is read before it is updated and that "not routed" does
not collide with a valid encoding.
