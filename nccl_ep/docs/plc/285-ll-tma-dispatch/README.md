# LL TMA Dispatch for LSA-only Cases (MR !285)

## Summary

This change adds a TMA-pipelined SEND phase to the LL (low-latency), LSA-only,
rank-major, unquantized dispatch kernel family. It replaces the existing
per-token atomics-driven SEND loop with a warp-0-owned, N-stage software
pipeline that overlaps TMA loads from the attention input with TMA stores into
each destination rank's receive region, while every other warp in the CTA
concurrently performs the routing-sentinel init and next-epoch
recv-count-buffer clear that would otherwise run serially afterward.

The new kernel (`dispatch_kernel_impl_2sided_rm_lsa_tma`) becomes the default
for LSA-eligible, unquantized, rank-major dispatch calls
(`NCCL_EP_DISPATCH_LSA=tma`, the zero-init default). It falls back to the
existing non-TMA LSA kernel for LSA-eligible calls it does not support (fused
DS-FP8E3M4), and to the general dispatch kernel for anything not LSA-eligible
at all. There is no new public API or quantization recipe; only an internal
kernel-family selector is added.

## Problem and scope

The existing LSA-only rank-major dispatch SEND phase
(`dispatch_kernel_impl_2sided_rm_lsa`) issues per-token NVLink stores directly
from registers/shared staging, using a per-token atomic slot allocation to
serialize destination-slot numbering. This change targets the same call
shape and replaces only the SEND phase's data-movement strategy:

```text
nvlinkOnly && layout == RANK_MAJOR && recipe == NONE
```

`DS_FP8E3M4` (fused quantized) and anything not LSA-eligible keep using their
existing kernels unchanged. The RECV phase, completion protocol, epoch
handling, and output formats are untouched -- the TMA kernel's RECV phase
reuses the same `recordRecvTokensLsa` / `syncAndRecvCounts` /
`completeFullLowLatencyEpoch` path as the non-TMA LSA kernel, instantiated at
`kRecipe = NCCL_EP_DISP_QUANT_NONE`.

## Design

### Overview

One CTA owns a grid-strided slice of this rank's tokens
(`t = smId; t < numTokens; t += numSms`). Within a CTA, warp 0 alone drives
the TMA pipeline across its token range; every other warp spends the whole
SEND phase on unrelated per-CTA bookkeeping that has to happen anyway, so the
two run concurrently rather than one after the other. The idle warps' routing
sentinel clear writes through `args.outRecvTopkIdx`, a caller-owned output
tensor that `ncclEpDispatch` validates only for ndim/datatype, not 16-byte
alignment -- the vectorized `int4` clear path is only taken when the pointer
happens to be aligned for it, falling back to a scalar per-element loop
otherwise:

```text
SEND phase
--------------------------------------------+------------------------------------
warp 0                                      | warps 1..numWarps-1  (concurrent)
                                            |
kNumStages-deep ring in shared memory       | outRecvTopkIdx = -1
(kNumStages = dispatch_tma_smem::           | (routing-sentinel init)
  choose_config; stage = payload /          |
  header / mbarrier)                        | next-epoch recv-count-buffer clear
prefetch token 0 into stage 0               |
for each token t in this CTA's range:       |
  wait this stage's mbarrier (load landed)  |
  assemble routing header -> stage header   |
  tma_store_fence (header write visible     |
    to the async/TMA proxy)                 |
  dedup winning lane per destination rank   |
    (warpFirstTopKOccurence, shuffle-only)  |
  every winning lane resolves its OWN       |
    destination address, in parallel        |
    (P2P pointer cached once per CTA)       |
  lane 0 alone issues every winner's        |
    (header, payload) store, shuffled in    |
    from the winning lanes; commits ONCE    |
    per token (lane 0 only, even if zero    |
    winners -- an empty commit-group)       |
  lane 0: tma_store_wait<kNumStages-1>      |
    -> reload the next stage, one           |
    iteration ahead                         |
--------------------------------------------+------------------------------------
                      |
                      v
syncAndSendCounts
  (__syncthreads; publishes per-rank sent counts;
   elected last CTA resets both count arrays for the next epoch)
                      |
                      v

RECV phase (all warps, shared with the non-TMA LSA kernel)
  completeFullLowLatencyEpoch -> syncAndRecvCounts -> recordRecvTokensLsa
```

A CTA with zero tokens skips the pipeline entirely. A CTA with fewer tokens
than `kNumStages` simply never wraps the ring around (every token gets its
own never-reused stage) -- the ring itself is always the full
`kNumStages` deep regardless of this CTA's token count; there is no
separate "active" ring size to track.

### Stage ring and TMA pipelining

Each shared-memory stage holds one whole token: its payload (TMA-loaded from
`args.inData`), its routing header (assembled in-place by lanes, not loaded),
and one mbarrier tracking the payload load only. The pipeline prefetches only
the very first token before the loop starts; each loop iteration then issues
the *next* token's load one iteration ahead, cycling through the
`kNumStages` physical stage buffers (`stage = iter % kNumStages`) as it goes.
A stage is only ever reloaded once `kNumStages` iterations have elapsed since
it was last used, so a store for token *i* can be in flight while the load
for token *i + kNumStages* is issued, hiding per-token NVLink/TMA latency
behind issue rate.

Because a token's `topK` experts can span up to `topK` distinct destination
ranks, more than one lane can independently be the dedup "winner" for a
different destination in the same iteration. `cp.async.bulk`'s commit-group
counters are tracked per-thread, not per-warp, so `tma_store_wait<N>`'s "is
this stage free" check is only meaningful if every store for a token is
issued and committed by the same lane: a lane that doesn't win a given
iteration would otherwise sit at a commit-group count of zero, regardless of
how many *other* lanes still have stores reading that stage's smem in
flight. Every winning lane therefore resolves its own destination address in
parallel (plain global-memory arithmetic, not a TMA op), then shuffles that
address to lane 0, which alone issues every store for the token and commits
them as a single group -- including an empty commit when a token happens to
have zero winners (e.g. every destination masked) -- so
`tma_store_wait<kNumStages - 1>`, called by lane 0 only, corresponds exactly
to one commit-group per loop iteration with no gaps.

This one-commit-per-iteration invariant is also why the pipeline prefetches
only one token ahead instead of filling the whole `kNumStages`-deep ring up
front: a stage is never reloaded until `kNumStages` real iterations (and
therefore `kNumStages` commit-groups) have elapsed since it was last used, so
`tma_store_wait<kNumStages - 1>` is always checked at a point where it is
already a meaningful bound.

### Shared memory sizing

The stage count `kNumStages` is resolved once per JIT variant
(`dispatch_tma_smem::choose_config`, keyed only by hidden size, token dtype,
and the group's cached shared-memory budget -- not by the call's token
count); `dispatch_tma_smem::kStaticSmemBytes` is reserved out of that budget
first, to cover `syncAndSendCounts`'s static `__shared__ bool shIsLastCta`
(which isn't part of the dynamic allocation `choose_config` sizes), the same
split `call_combine` already applies via `combine_smem::kLsaStaticSmemBytes`.
Row-byte sizing for `choose_config`'s input goes through
`token_row_bytes(dtype, hidden)` (`device_primitives.cuh`), not a plain
`hidden * bytes_per_element` multiply, so it stays correct if a packed
sub-byte wire dtype (e.g. `ncclFloat4x2`) is ever reachable here --
`NCCL_EP_DISP_QUANT_NONE`'s dtype validation currently restricts this kernel
to BF16/FP16/FP32, so that case isn't reachable today, but the sizing helper
doesn't assume it never will be. A per-CTA rank-pointer cache (one `uint64_t`
per destination rank) is carved out of the same dynamic allocation ahead of
the stage ring, since `ncclGetP2pPtr`'s result for a given destination is
loop-invariant for the whole SEND phase and is otherwise worth resolving
only once per CTA rather than once per `(token, winning lane)`.

### Selection and fallback

| `NCCL_EP_DISPATCH_LSA` | Kernel | When |
| --- | --- | --- |
| `tma` (default) | `dispatch_kernel_impl_2sided_rm_lsa_tma` | LSA-eligible, rank-major, `NONE` recipe |
| `simt` | `dispatch_kernel_impl_2sided_rm_lsa` | LSA-eligible, rank-major; also the silent fallback from `tma` for `DS_FP8E3M4` (the TMA kernel takes no recipe template) |
| `universal` | the general `dispatch_kernel_impl` | forces the RDMA/GIN-capable kernel regardless of LSA eligibility |

Combine is untouched by this MR and has no matching selector: combine's LSA
kernel (`combine_kernel_impl_2sided_rm_lsa`) is already TMA-pipelined.

## Performance evaluation

p50 latency (microseconds), comparing the prior kernel (`baseline`), this
change (`tophead`), and an external FlashInfer reference kernel, across
EP8/EP16 world size and CUDA Graph capture on/off.
Both the dispatch call alone (`DISPATCH`) and a full dispatch+combine
round trip (`ROUNDTRIP`) are reported. `v_base%` is `tophead` vs `baseline`,
`(tophead - baseline) / baseline`; `v_fi%` is `tophead` vs `flashinfer`,
`(tophead - flashinfer) / flashinfer`.

`tophead` is faster than `baseline` in the large majority of cells, with the
gain generally growing with token count -- DISPATCH reaches -11.3% at
EP16/graph=0, T=2, and -7.5% at EP8/graph=1, T=8192. The exceptions are
EP8/graph=0's ROUNDTRIP numbers, which run slightly *slower* than `baseline`
(up to +1.1%) through most of the token-count range before crossing over to
faster at T>=512, and a couple of isolated +0.0/+0.2% DISPATCH points at
mid-range T in EP8/graph=0 and EP16/graph=1. Against FlashInfer, `tophead` is
decisively faster at low-to-mid token counts (DISPATCH down to roughly -70%
at T=1-2 in both EP8 and EP16, graph=0), and the comparison inverts at the
high end: FlashInfer is ahead by low-to-mid single digits on most DISPATCH
T>=4096 cells, and by a wider margin (up to +15.6%) on most ROUNDTRIP
T>=2048 cells.

### EP8, graph=0 (job 3247086)

**DISPATCH** (p50, microseconds)

```
     T | baseline |  tophead | flashinfer |  v_base% |    v_fi%
     1 |    41.29 |    40.70 |     139.11 |    -1.4% |   -70.7%
     2 |    41.13 |    40.69 |     139.17 |    -1.1% |   -70.8%
     4 |    41.38 |    40.94 |     139.33 |    -1.1% |   -70.6%
     8 |    41.59 |    40.97 |     139.09 |    -1.5% |   -70.5%
    16 |    42.19 |    41.59 |     138.90 |    -1.4% |   -70.1%
    32 |    42.63 |    42.40 |     139.19 |    -0.5% |   -69.5%
    64 |    45.29 |    45.19 |     139.35 |    -0.2% |   -67.6%
   128 |    50.35 |    50.44 |     139.66 |    +0.2% |   -63.9%
   256 |    60.99 |    61.03 |     139.96 |    +0.1% |   -56.4%
   512 |    83.43 |    82.02 |     140.91 |    -1.7% |   -41.8%
  1024 |   126.62 |   124.53 |     143.51 |    -1.7% |   -13.2%
  2048 |   223.74 |   211.19 |     213.97 |    -5.6% |    -1.3%
  4096 |   407.27 |   381.64 |     379.84 |    -6.3% |    +0.5%
  8192 |   773.58 |   724.67 |     708.87 |    -6.3% |    +2.2%
```

**ROUNDTRIP** (p50, microseconds)

```
     T | baseline |  tophead | flashinfer |  v_base% |    v_fi%
     1 |    61.82 |    62.29 |     155.79 |    +0.8% |   -60.0%
     2 |    61.96 |    62.61 |     156.29 |    +1.0% |   -59.9%
     4 |    62.57 |    63.10 |     156.60 |    +0.8% |   -59.7%
     8 |    62.67 |    63.38 |     156.52 |    +1.1% |   -59.5%
    16 |    63.25 |    63.67 |     157.03 |    +0.7% |   -59.5%
    32 |    63.80 |    64.32 |     156.79 |    +0.8% |   -59.0%
    64 |    68.70 |    68.82 |     157.57 |    +0.2% |   -56.3%
   128 |    78.65 |    78.92 |     158.68 |    +0.3% |   -50.3%
   256 |   102.24 |   102.67 |     162.99 |    +0.4% |   -37.0%
   512 |   150.20 |   149.28 |     184.27 |    -0.6% |   -19.0%
  1024 |   249.77 |   248.71 |     236.13 |    -0.4% |    +5.3%
  2048 |   456.50 |   443.84 |     400.90 |    -2.8% |   +10.7%
  4096 |   825.67 |   799.29 |     735.01 |    -3.2% |    +8.7%
  8192 |  1609.21 |  1559.48 |    1399.90 |    -3.1% |   +11.4%
```

### EP8, graph=1 (job 3247087)

**DISPATCH** (p50, microseconds)

```
     T | baseline |  tophead | flashinfer |  v_base% |    v_fi%
     1 |    23.32 |    22.56 |      26.45 |    -3.3% |   -14.7%
     2 |    22.69 |    22.35 |      24.33 |    -1.5% |    -8.1%
     4 |    23.77 |    22.81 |      24.41 |    -4.0% |    -6.6%
     8 |    23.98 |    22.88 |      24.64 |    -4.6% |    -7.1%
    16 |    24.07 |    22.99 |      24.54 |    -4.5% |    -6.3%
    32 |    24.26 |    24.03 |      24.55 |    -0.9% |    -2.1%
    64 |    26.81 |    26.63 |      25.63 |    -0.7% |    +3.9%
   128 |    32.18 |    32.03 |      30.18 |    -0.5% |    +6.1%
   256 |    42.69 |    42.60 |      41.05 |    -0.2% |    +3.8%
   512 |    64.96 |    63.47 |      64.43 |    -2.3% |    -1.5%
  1024 |   108.43 |   106.35 |     108.55 |    -1.9% |    -2.0%
  2048 |   207.26 |   191.84 |     190.58 |    -7.4% |    +0.7%
  4096 |   391.99 |   363.25 |     356.01 |    -7.3% |    +2.0%
  8192 |   756.81 |   700.32 |     687.10 |    -7.5% |    +1.9%
```

**ROUNDTRIP** (p50, microseconds)

```
     T | baseline |  tophead | flashinfer |  v_base% |    v_fi%
     1 |    39.53 |    38.94 |      41.96 |    -1.5% |    -7.2%
     2 |    39.59 |    38.98 |      46.18 |    -1.5% |   -15.6%
     4 |    39.81 |    39.35 |      46.53 |    -1.2% |   -15.4%
     8 |    40.15 |    39.54 |      47.57 |    -1.5% |   -16.9%
    16 |    41.26 |    40.13 |      48.51 |    -2.7% |   -17.3%
    32 |    41.88 |    41.70 |      49.32 |    -0.4% |   -15.5%
    64 |    47.41 |    47.22 |      50.98 |    -0.4% |    -7.4%
   128 |    57.62 |    57.62 |      57.86 |    +0.0% |    -0.4%
   256 |    81.32 |    81.66 |      76.01 |    +0.4% |    +7.4%
   512 |   130.02 |   129.21 |     121.77 |    -0.6% |    +6.1%
  1024 |   226.60 |   224.57 |     215.99 |    -0.9% |    +4.0%
  2048 |   429.63 |   414.09 |     382.08 |    -3.6% |    +8.4%
  4096 |   808.96 |   779.86 |     715.86 |    -3.6% |    +8.9%
  8192 |  1595.52 |  1538.54 |    1381.72 |    -3.6% |   +11.3%
```

### EP16, graph=0 (job 3247088)

**DISPATCH** (p50, microseconds)

```
     T | baseline |  tophead | flashinfer |  v_base% |    v_fi%
     1 |    52.58 |    46.89 |     148.16 |   -10.8% |   -68.4%
     2 |    52.76 |    46.79 |     148.33 |   -11.3% |   -68.5%
     4 |    52.52 |    46.78 |     147.84 |   -10.9% |   -68.4%
     8 |    52.76 |    46.92 |     148.16 |   -11.1% |   -68.3%
    16 |    53.34 |    47.72 |     148.19 |   -10.5% |   -67.8%
    32 |    53.70 |    48.98 |     148.00 |    -8.8% |   -66.9%
    64 |    56.39 |    52.04 |     148.01 |    -7.7% |   -64.8%
   128 |    62.93 |    59.04 |     148.10 |    -6.2% |   -60.1%
   256 |    76.15 |    71.90 |     148.98 |    -5.6% |   -51.7%
   512 |   103.33 |    99.54 |     150.02 |    -3.7% |   -33.6%
  1024 |   158.35 |   153.45 |     161.67 |    -3.1% |    -5.1%
  2048 |   279.67 |   263.22 |     264.17 |    -5.9% |    -0.4%
  4096 |   508.26 |   477.17 |     475.27 |    -6.1% |    +0.4%
  8192 |   956.99 |   899.94 |     894.04 |    -6.0% |    +0.7%
```

**ROUNDTRIP** (p50, microseconds)

```
     T | baseline |  tophead | flashinfer |  v_base% |    v_fi%
     1 |    74.17 |    69.17 |     165.16 |    -6.7% |   -58.1%
     2 |    74.49 |    69.40 |     165.45 |    -6.8% |   -58.1%
     4 |    74.84 |    69.49 |     165.30 |    -7.1% |   -58.0%
     8 |    75.40 |    69.79 |     165.33 |    -7.4% |   -57.8%
    16 |    75.83 |    70.17 |     166.28 |    -7.5% |   -57.8%
    32 |    77.15 |    71.96 |     166.20 |    -6.7% |   -56.7%
    64 |    81.73 |    77.40 |     166.72 |    -5.3% |   -53.6%
   128 |    94.33 |    91.03 |     168.86 |    -3.5% |   -46.1%
   256 |   123.40 |   119.67 |     176.67 |    -3.0% |   -32.3%
   512 |   182.39 |   179.77 |     205.13 |    -1.4% |   -12.4%
  1024 |   308.76 |   305.02 |     285.11 |    -1.2% |    +7.0%
  2048 |   577.39 |   560.53 |     495.04 |    -2.9% |   +13.2%
  4096 |  1063.02 |  1031.43 |     917.17 |    -3.0% |   +12.5%
  8192 |  2088.01 |  2030.42 |    1759.48 |    -2.8% |   +15.4%
```

### EP16, graph=1 (job 3247089)

**DISPATCH** (p50, microseconds)

```
     T | baseline |  tophead | flashinfer |  v_base% |    v_fi%
     1 |    26.50 |    25.73 |      24.73 |    -2.9% |    +4.0%
     2 |    26.70 |    25.97 |      26.05 |    -2.7% |    -0.3%
     4 |    26.65 |    26.06 |      26.03 |    -2.2% |    +0.1%
     8 |    26.97 |    26.12 |      26.06 |    -3.2% |    +0.2%
    16 |    26.87 |    26.15 |      26.26 |    -2.7% |    -0.4%
    32 |    27.89 |    27.69 |      26.21 |    -0.7% |    +5.6%
    64 |    30.75 |    30.70 |      28.81 |    -0.2% |    +6.6%
   128 |    37.61 |    37.50 |      35.74 |    -0.3% |    +4.9%
   256 |    50.56 |    50.31 |      48.64 |    -0.5% |    +3.4%
   512 |    78.56 |    77.75 |      78.46 |    -1.0% |    -0.9%
  1024 |   131.66 |   130.46 |     133.03 |    -0.9% |    -1.9%
  2048 |   249.87 |   237.08 |     239.04 |    -5.1% |    -0.8%
  4096 |   482.85 |   452.65 |     450.19 |    -6.3% |    +0.5%
  8192 |   939.78 |   878.21 |     869.37 |    -6.6% |    +1.0%
```

**ROUNDTRIP** (p50, microseconds)

```
     T | baseline |  tophead | flashinfer |  v_base% |    v_fi%
     1 |    42.85 |    41.86 |      47.96 |    -2.3% |   -12.7%
     2 |    43.62 |    42.18 |      49.05 |    -3.3% |   -14.0%
     4 |    43.85 |    42.59 |      49.30 |    -2.9% |   -13.6%
     8 |    44.34 |    43.31 |      49.69 |    -2.3% |   -12.8%
    16 |    44.66 |    43.57 |      50.53 |    -2.4% |   -13.8%
    32 |    46.88 |    46.33 |      51.42 |    -1.2% |    -9.9%
    64 |    53.41 |    53.23 |      54.99 |    -0.3% |    -3.2%
   128 |    67.38 |    67.41 |      65.19 |    +0.0% |    +3.4%
   256 |    96.38 |    95.90 |      88.92 |    -0.5% |    +7.8%
   512 |   156.59 |   155.72 |     148.85 |    -0.6% |    +4.6%
  1024 |   283.49 |   281.81 |     262.90 |    -0.6% |    +7.2%
  2048 |   548.36 |   535.34 |     473.02 |    -2.4% |   +13.2%
  4096 |  1031.07 |  1001.14 |     894.69 |    -2.9% |   +11.9%
  8192 |  2071.31 |  2009.07 |    1737.37 |    -3.0% |   +15.6%
```

All four jobs use the same world-size/token-count sweep; `graph=1` captures
the call(s) inside a CUDA Graph. This capture did not record stdev/CV%, so
run-to-run variance is not characterized here.

## Validation

This evaluation is a performance comparison, not a dedicated correctness
regression; no new test file was added alongside this kernel. Because
`dispatch_kernel_impl_2sided_rm_lsa_tma` becomes the default kernel for every
LSA-eligible, unquantized, rank-major LL dispatch call, the existing LL
regression suite (`test_lifecycle.cu`, `test_ll_epoch_protocol.cu`,
`test_ll_group_epoch.cu`) now exercises it implicitly for any call shape it
already covered under the old default -- but none of those tests name the
TMA kernel or its stage-ring sizing explicitly, and `ep_bench --validate`'s
dispatch-output check is the only thing that has actually been run against
the configurations in the table above. A dedicated
test exercising `NCCL_EP_DISPATCH_LSA=tma` directly (stage-count boundaries,
the DS_FP8E3M4 fallback to `simt`, the concurrent idle-warp
sentinel/recv-count-buffer clear, and the misaligned-`outRecvTopkIdx`
scalar-fallback path) is a gap worth closing before this is relied on as
regression-tested, not just benchmark-validated.

## Implementation and packaging

| Responsibility | Source |
| --- | --- |
| TMA stage-ring pipeline, dedup, idle-warp concurrent bookkeeping, RECV-phase reuse | `nccl_ep/device/ll/ll_dispatch_lsa.cuh` |
| Stage-ring/rank-cache sizing (`dispatch_tma_smem::choose_config`, `kStaticSmemBytes`) | `nccl_ep/device/ll/ll_ep_smem.cuh` |
| Kernel-family selection, JIT source generation, static-smem reservation | `nccl_ep/device/jit/ll_dispatch_jit.cuh` |
| FP4-aware row-byte sizing (`token_row_bytes`) | `nccl_ep/device/device_primitives.cuh` |
| `NcclEpDispatchLsaMode`, `DispatchParams::lsaMode` | `nccl_ep/device/ll_ep_adapter.cuh` |
| `lsaMode`/`maxDynamicSmem` threading into `launch_ll_dispatch` | `nccl_ep/device/ll_ep_adapter.cu` |
| `NCCL_EP_DISPATCH_LSA` env var | `nccl_ep/nccl_ep_env.h`, `nccl_ep/nccl_ep_env.cc` |

No public struct or ABI changes. `NCCL_EP_DISPATCH_LSA=simt` remains
available as a direct rollback path to the prior kernel without reverting
the change; `NCCL_EP_DISPATCH_LSA=universal` remains available as a full
fallback to the general dispatch kernel.
