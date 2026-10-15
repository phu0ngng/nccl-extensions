# Low-Latency 1-Sided Combine with Automatic Kernel Selection (MR !302)

LSA-only, rank-major, BF16-only, unquantized combine can pull its result
directly out of a peer's `inData` instead of waiting for that peer to push it
("1-sided"), avoiding a host round-trip of send + receive. Whether a given
`ncclEpCombine` call actually uses this is resolved per call, mostly
on-device.

## 1-sided combine algorithm

`combine_kernel_impl_1sided_rm_lsa` (`device/ll/ll_combine_lsa.cuh`) pairs
with a dispatch kernel that ran with `kRecordDstInfo=true`
(`device/ll/ll_dispatch_lsa.cuh`), which additionally records each token's
`(dstRank, slotIdx)` into `dstInfo` (a `-1` `dstRank` sentinel marks
duplicate-rank/invalid top-k slots).

- **SEND phase** is reduced to a single-shot readiness broadcast: "my
  `inData` is ready to be pulled from." No payload is sent.
- **RECV phase**: each CTA reads its assigned tokens' results directly from
  the owning rank's `inData` window (`ncclGetPeerPointer(inDataWindow, ...)`
  + TMA), using `dstInfo` to locate them. Once every CTA on this GPU has
  confirmed its own reads are complete (CTA-local `mbarrier`/`__syncthreads`,
  then a grid-wide election via the `ctaSync[1]` counter), the elected last
  CTA broadcasts a read-done flag to every peer it read from
  (`sendRankFinishFlagLsa`).
- Before retiring, the kernel also waits (`waitForAllReadDoneDistributedLsa`)
  for every peer that could have pulled from *this* rank's `inData` to send
  its own read-done flag — the caller's `inData` buffer must not be
  overwritten by the next round's expert compute until every reader has
  confirmed it's done.

Three kernel bodies exist (`LlCombineKernel`, `device/ll_ep_adapter.cuh`):
`k2SidedRmLsa` (push, the LSA-only general case), `k1SidedRmLsa` (pull, above),
and `kAutoRmLsa`, which wraps both bodies behind one on-device runtime branch.

## Kernel selection

`NCCL_EP_COMBINE_LSA` (`nccl_ep_env.h`) requests `"2sided"`, `"1sided"`,
`"universal"` (always the general kernel), or unset (`"auto"`).
`ll_combine_select_kernel` (`device/jit/ll_combine_jit.cuh`) is the sole
place a request resolves to a concrete kernel:

- An enforced LSA kernel (`"2sided"`/`"1sided"`) falls back to the general
  kernel **silently** when the call isn't LSA-eligible at all (wrong
  topology/layout/recipe/CFT) — the env var only governs the pick *within*
  the LSA domain, it doesn't force LSA eligibility.
- `kInvalid` (an error) is reserved for an in-domain request that still can't
  be honored — `"1sided"` enforced but not `oneSidedEligible` (see below).
- Unset (`kNone`) picks `kAutoRmLsa` when `oneSidedEligible`, else falls back
  the same way an enforced kernel would.

### Auto mode's on-device pick

For unenforced, eligible calls, **dispatch** — not combine — makes the
1-sided-vs-2-sided decision, so it's ready before combine runs (no host
round-trip):

```cpp
// device/ll/ll_dispatch_lsa.cuh, dispatch RECV phase, kRecordDstInfo only
bool exceedsThreshold = false;
for (int peer = laneId; peer < numRanks; peer += 32) {
    if (isRankMasked(rankMask, peer)) continue;
    exceedsThreshold = exceedsThreshold || totalTokenRecvBuf[peer] > combineLsaThreshold;
}
const bool anyExceedsThreshold = __ballot_sync(0xffffffffu, exceedsThreshold) != 0;
if (laneId == 0) *want1SidedBuf = anyExceedsThreshold ? 1 : 0;
```

`max(counts) > threshold` is equivalent to `any(count > threshold)`, so this
needs one warp-wide ballot instead of a shuffle-reduction to a max. Runs on
the **last warp of the last SM** (not `smId`/`warpId == 0`): SM 0 already
carries the SEND phase's next-bank cleanup + count-publish duties, and
`warpGroupId 0` is never idle in `recordRecvTokensLsa`, so placing this there
would add the extra global-memory latency to the busiest warp on the busiest
SM. The last warp of the last SM is usually genuine spare capacity instead
(`recordRecvTokensLsa` already skips `responsibleExpertIdx >= numExperts`
there whenever the expert count doesn't divide evenly).

`combine_kernel_impl_auto_rm_lsa` reads the published decision as a plain,
no-reset-needed int (dispatch overwrites it fresh every round):
`want1Sided = *p.want1SidedBuf != 0`.

### Disqualifying 1-sided

`handle->ll.dispatch_1sided_disqualified` gates `oneSidedEligible` for the
*next* combine call. It's input/output across one `ncclEpDispatch` call
(`DispatchParams::dstInfoDisqualifiedOut`): `nccl_ep.cc` seeds it, and
`launch_ll_dispatch` can only add disqualifying reasons on top, never clear
one the caller already set:

```cpp
// device/jit/ll_dispatch_jit.cuh
*dstInfoDisqualifiedOut = *dstInfoDisqualifiedOut || algo == LlDispatchAlgo::kDefault;
```

Seeded true (`nccl_ep.cc`, dispatch call site) when
`max_dispatch_tokens_per_rank <= combine_lsa_threshold` — no round can ever
dispatch more tokens than that ceiling, so if even the ceiling can't clear
the threshold (`<=`, matching the on-device check's strict `>`), 1-sided is
statically unreachable this round regardless of which dispatch kernel runs.
Otherwise seeded false, then OR'd with `algo == kDefault` once
`ll_dispatch_select_algo` actually picks a kernel — covering every reason it
could fall back to the general kernel (`NCCL_EP_DISPATCH_LSA=universal`,
unsupported recipe/layout, CFT, hidden-size misalignment, `!nvlinkOnly`), not
just the explicit `"universal"` override.

`call_combine` (`device/ll_ep_adapter.cu`) combines this with the rest of
1-sided's requirements: `oneSidedEligible = lsaEligible && tokenDtype ==
ncclBfloat16 && groupZeroCopyOn && inDataWindow != ncclWindow_t{} &&
!dispatch1SidedDisqualified`.
