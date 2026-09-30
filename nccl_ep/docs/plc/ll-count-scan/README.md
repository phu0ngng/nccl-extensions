# Counting-Warp Scan of the Top-k Table on the CFT Dispatch Path

## Overview

Every low-latency dispatch CTA must know, for each destination rank it serves,
how many tokens the current call sends to that rank. The counting warp derives
that number from the top-k routing table while the send warps move payloads,
and the count publisher forwards it to the destination. The store path's scan
(`countTokensPerRank_mask`, source preserved) consumes each routing
entry as soon as it is loaded: an integer division, a divergent branch, and two
arrays in local memory, and only then issues the next load. Every CTA reads the
whole table, so that scan is bound by one load round trip per entry.

This design adds a second scan, `countTokensPerRankCft`, selected by an explicit
`if constexpr (kCft)` at the counting-warp site and used only when the dispatch
runs over CFT (`NCCL_EP_LL_CFT=1`). It keeps the role, the inputs, and the
output of the counting warp unchanged. Each lane issues a batch of loads before
it consumes any of them, maps an entry to a rank with a range check and a
multiply instead of a division, keeps the per-token rank masks in registers, and
counts with warp ballots. The result is identical to the store path's scan for
every input it accepts.

The trade is a fixed batch shape and a few more registers in the counting warp.
The 32-rank limit per counting warp already existed in the store path's scan and
is now stated by an assert. The STORE JIT specialization keeps
`countTokensPerRank_mask`; the CFT specialization compiles only the new scan.
The host chooses the specialization and its cache entry before launch, so
there is no runtime transport branch in the counting warp. This PLC covers the
counting warp only; count publication and the receive side are unchanged.

## Where the count comes from and where it goes

### The counting warp

The dispatch grid is sized from the number of experts. Each CTA owns a
contiguous range of experts, and the last warp of each CTA is the counting warp.
Its window of destination ranks covers the ranks that host the CTA's experts.
The exclusive expert-end index conservatively includes the next rank when it
falls exactly on a rank boundary:

```text
numLocalExperts = numExperts / numRanks

expertBegin = ctaIndex × warpGroupsPerCta
expertEnd   = min(expertBegin + warpGroupsPerCta, numExperts)

rankBegin   = expertBegin / numLocalExperts
rankEnd     = expertEnd   / numLocalExperts + 1
```

For example, with 8 local experts per rank and 4 warp groups per CTA:

```text
CTA 0: experts  0..3   → rank 0            → window [0, 1)
CTA 1: experts  4..7   → rank 0            → window [0, 2)
CTA 2: experts  8..11  → rank 1            → window [1, 2)
CTA 3: experts 12..15  → rank 1            → window [1, 3)
```

Several CTAs can cover the same rank. Each of them computes the same count from
the same table; only the CTA that holds the rank's first local expert publishes
it. Publication checks that the rank's first expert is in the CTA's half-open
expert range, so a conservatively included next rank is not published by that
CTA. The scan itself runs on every CTA in parallel, so its latency is paid once
per dispatch, not once per CTA in sequence.

### What the counting warp produces

For each rank `r` in the window, the counting warp produces `N_r`: the number of
tokens that have at least one top-k entry on rank `r`. A token counts once per
rank however many of its experts land there. This matches the sender, which
sends each token to a rank once, regardless of how many local experts the
token selects there.

```text
Token 0 selects experts on ranks A, A, B  → counts once for A, once for B
Token 1 selects experts on ranks B, C, C  → counts once for B, once for C
Token 2 selects experts on rank  A        → counts once for A

N_A = 2   N_B = 2   N_C = 1
```

The counts are written to shared memory, one slot per rank in the window.

### Who consumes the counts

On the store path, let `F = FINISHED_SUM_TAG`. The counts feed the
`rankDone` protocol and the count publisher:

```text
Counting warp:  N_r known ─────────────────→ +(F − N_r) into rankDone[r]

Count publisher (first warp of the group that owns rank r's first expert):
                waits until rankDone[r] reaches 2F, then publishes −N_r − 1
                into the destination's count channels
```

The cleaner contributes `+F` and every sent token contributes `+1`, so
`rankDone[r]` reaches `2F` only when the counting contribution, the cleanup,
and all `N_r` sends have arrived. The count publisher therefore cannot run
before the counting warp has finished. Everything downstream of the scan waits
for it.

On the CFT path, the counting warp calls `cftPublishCountEarly` after its
shared-memory count stores and a warp synchronization. It publishes the count
as soon as the scan finishes, without waiting for `rankDone` or all payload
sends; the hardware byte counter tracks payload completion.

## The store path's scan (`countTokensPerRank_mask`, source preserved)

Lane `l` of the counting warp owns tokens `l`, `l + 32`, `l + 64`, and so on.
For each of its tokens it walks the top-k entries one at a time:

```text
For each owned token:
   For each top-k entry k:
      load entry
      rank = entry / numLocalExperts          (runtime integer division;
                                               −1 for an unused entry)
      if rank < 0: continue                   (divergent branch)
      if rank in window: rankMap[batch] |= 1 << (rank − rankBegin)
                                              (64-bit mask array, local memory)
   Every 32 tokens: for each mask, for each rank bit set: rankCount[rank]++
                                              (counter array, local memory)
Warp-reduce rankCount for each rank; lane 0 writes the shared-memory slot
```

Four properties make it slow:

- The load of the next entry is issued only after the current entry has been
  divided, tested, and merged into the mask. Each entry costs one round trip to
  L2 per lane.
- Division by a runtime value compiles to a multi-instruction sequence with a
  subroutine call on this architecture.
- `continue` on an unused entry diverges the warp.
- Both arrays are indexed dynamically, so they live in local memory. Merging an
  entry into a mask and bumping a counter are read-modify-write operations on
  local memory.

```text
entry k       load ──→ divide ──→ test ──→ RMW rankMap ──┐
                                                         │ dependency
entry k+1                                                └──→ load ──→ divide ──→ ...
```

## The CFT-path scan (`countTokensPerRankCft`)

### Batches of loads before any consume

Lane `l` handles four tokens per batch: `i0 + l`, `i0 + l + 32`, `i0 + l + 64`,
and `i0 + l + 96`, where `i0` advances by 128 tokens per batch. The top-k
entries of a token are visited in chunks of eight. For one chunk the lane issues
all `4 × 8` loads first and consumes them afterwards. Nothing in the consume
phase feeds back into the load phase, so the 32 loads are independent and all
of them are in flight at the same time.

```text
                 issue 32 loads ─────────────────────────────┐
  token t0: e0 e1 e2 e3 e4 e5 e6 e7                          │
  token t1: e0 e1 e2 e3 e4 e5 e6 e7                          │ one round trip
  token t2: e0 e1 e2 e3 e4 e5 e6 e7                          │ for the batch
  token t3: e0 e1 e2 e3 e4 e5 e6 e7                          │
                 consume 32 entries ◀────────────────────────┘
```

A token index at or beyond `numTokens`, or an entry index at or beyond
`numTopk`, is not loaded; the lane substitutes `−1`, which the mapping below
rejects. The loop bounds are the same for every lane, so every lane reaches
every ballot.

### Branch-free mapping of an entry to a rank bit

Define, once per scan:

```text
d      = numLocalExperts
lo     = rankBegin × d                    first expert index in the window
width  = (rankEnd − rankBegin) × d        number of expert indices in the window
magic  = ⌊(2^32 − 1) / d⌋
```

For an entry `e`, let `x = e − lo` computed as a 32-bit unsigned value.

```text
in window   ⇔  x < width          (a negative e wraps to a huge x and fails)
q0          =  mulhi(x, magic)    (high 32 bits of the 64-bit product)
q           =  q0 + 1  if  x − q0 × d ≥ d  else  q0
rankBits[t] |= 1 << q             (predicated on "in window")
```

`q` is exactly `x / d`. The product `magic × d` lies between `2^32 − d` and
`2^32 − 1`, so `x × magic / 2^32` lies between `x / d − x / 2^32` and `x / d`.
For `x` below `2^31` the lower end is above `x / d − 1`, so the truncated
estimate `q0` is either `x / d` or one less, and a single compare of the
remainder repairs it. Expert indices are far below `2^31`, and out-of-window
values are discarded by the predicate before their quotient matters.

The quotient is computed for every entry, in or out of window. This keeps the
in-window test a predicate on a single store into the mask instead of a branch
around the whole body.

Worked example with `d = 8`, window `[2, 4)`, so `lo = 16`, `width = 16`,
`magic = 536870911`:

```text
entry 21  →  x = 5    →  in window  →  q0 = 0, remainder 5  →  q = 0  →  bit 0 (rank 2)
entry 31  →  x = 15   →  in window  →  q0 = 1, remainder 7  →  q = 1  →  bit 1 (rank 3)
entry 40  →  x = 24   →  x ≥ width  →  no bit
entry −1  →  x = 2^32 − 17  →  x ≥ width  →  no bit
```

### Counting with ballots

After a batch, each lane holds four masks, one per token it handled. Lane `r`
owns rank `rankBegin + r`. For each rank bit `r` in the window and each token
slot `t`, one ballot across the warp collects the lanes whose token `t` targets
rank `r`, and lane `r` adds the population count of that ballot:

```text
for r in 0 .. windowSize − 1:
   for t in 0 .. 3:
      hits = ballot( bit r of rankBits[t] )      every lane participates
      if lane == r: count_r += popc(hits)
```

There is no per-lane counter array and no final warp reduction. After the last
batch, lane `r` writes `count_r` into the shared-memory slot of rank
`rankBegin + r`, followed by a warp synchronization so the consumers described
above can read the slots.

### End-to-end picture

```text
                 top-k table in gMem
                        │
        batch of 128 tokens, chunk of 8 entries
                        │
      ┌─────────────────┴─────────────────┐
      │  32 independent loads per lane    │   (round trip paid once per batch)
      └─────────────────┬─────────────────┘
                        ▼
      x = e − lo;  in = x < width;  q = exact x / d       (no branch, no divide)
                        ▼
      rankBits[t] |= 1 << q   if in                       (registers)
                        ▼
      per rank bit: ballot + popc, lane r accumulates rank r
                        ▼
      lane r writes N_r to shared memory
```

## Correctness

The output is defined as before: for each rank in the window, the number of
tokens whose set of top-k ranks contains that rank. The new scan reaches the
same value for every input the previous scan accepted:

- Unused or negative entries are rejected by the unsigned range check. The
  previous scan mapped them to `−1` and skipped them.
- `d = 1` gives `magic = 2^32 − 1`; the estimate is one below `x` for `x ≥ 1`
  and the fix-up restores it.
- Token counts that are not a multiple of 128 and top-k values that are not a
  multiple of 8 are handled by the `−1` substitution; the trip counts stay
  uniform across the warp.
- 32-bit and 64-bit index types are narrowed to `int` exactly as before.
- A window of more than 32 ranks is rejected by an assert. The previous code
  shifted a 32-bit `1` by the rank offset, so it could not represent such a
  window either.
- CTAs that share a rank compute from the same table and reach the same count.

## Interaction with the send paths

On the store-based send path the count publisher waits until `rankDone[r]`
reaches `2F`, which includes every payload send of the dispatch. The scan
overlaps the sends and its latency is normally hidden behind them. The store
path therefore keeps `countTokensPerRank_mask`. A version of this scan shared
by both paths measured a small-batch store-path gain (STORE/STORE+scan
1.04–1.09 at 128–512 tokens, 128/256 experts, 4 ranks, hecate); that gain is
forgone by design so that the store-path source is preserved.

On the CFT path payload completion is tracked by the hardware byte counter and
the count is published as soon as it is known (LL CFT dispatch PLC, "Publish the
token count early"). The counting warp then sits directly in front of the
publish, and the scan is on the receiver's critical path. This is where the
shorter scan matters: at 128 issuing SMs with top-k 8 the CFT path is
scan-bound without it (kernel-only CFT dispatch at 128 experts, 2048 tokens,
4 ranks, hecate0342: 208.8 µs at `2355f2b` with the store path's scan,
164.7 µs at `50ce4e4` with the shared scan reapplied; the store path measured
187.9 / 184.8 µs on the same cells). A separate predecessor comparison on
hecate0337 measured CFT at 176.0 µs for the older bundled-scan head `f37ad50`
and 208.9 µs for `2355f2b` (store: 194.5 / 186.9 µs).

## Measured performance

Measured source: `aecc9215c7e17c243c004541fdf38dcc2a050cfb` (NEW), based on
`2355f2b` (OLD, MR !60). COMBO is `50ce4e4`, the same baseline plus the
original shared-scan MR !244. The final publication commit differs from
the tested source only in documentation and commit metadata; executable
source and device comments are identical to the tested revision. The
historical SASS comparison and 40/40 validation remain labeled separately.

Setup: 2026-09-30, hecate0350, job 669325, 4 ranks on 1 node, sm_107,
CUDA 13.4.44, NCCL 2.30.7, expert-major layout, bf16, hidden size 7168,
top-k 8. Times are CUPTI kernel-only LL dispatch averages per run (10
warmups, 50 timed iterations), pooled as the median of 3 rotated process
runs. There were 108 matrix runs and 12 adjacent A/A control runs.
Ratios use unrounded medians; absolute medians below are in microseconds.
Sources: `publication-mr244/PERF_TABLES.md`, `PERF_SUMMARY.json`,
`perf_results.txt`, and the per-run logs.

The predeclared exclusion rule removes a repetition above 3 times its
cell/arm's minimum average. One repetition was excluded: COMBO CFT,
e128/t512, repetition 2, 1308.05 µs (run min/max 49.86/1728.83 µs).
Its retained averages are 52.36 and 51.24 µs, giving a 51.80 µs median.
All other matrix medians use 3 repetitions. `*` marks the affected
comparison, which uses 2 retained COMBO repetitions and needs a
double-check. Subthreshold spikes are retained in the source tables.

### (a) Store path: preserved source

Both binaries have `NCCL_EP_LL_CFT` unset. Each cell shows OLD → NEW µs
(ratio), where ratio = **NEW store µs / OLD store µs**; >1 means NEW is
slower. OLD = `2355f2b`; NEW = `aecc921`.

| Experts | 128 tokens | 512 tokens | 2048 tokens |
|---:|---:|---:|---:|
| 64 | 30.82 → 30.70 (0.996) | 70.87 → 70.46 (0.994) | 309.49 → 309.32 (0.999) |
| 128 | 33.36 → 33.23 (0.996) | 60.10 → 59.97 (0.998) | 184.34 → 183.59 (0.996) |
| 256 | 33.34 → 33.35 (1.000) | 60.37 → 60.02 (0.994) | 184.50 → 184.03 (0.997) |

Ratios range from 0.994 to 1.000. No clear performance change was detected
in these measured cells; this does not establish zero impact. At e128/t512,
adjacent independent runs of the same NEW store binary had second/first
ratios of 0.9890–1.0043.

### (b) CFT path: CFT-only versus shared scan

Both binaries have `NCCL_EP_LL_CFT=1`. Each cell shows COMBO → NEW µs
(ratio), where ratio = **NEW CFT µs / COMBO CFT µs**; <1 means NEW is
faster. COMBO = `50ce4e4`; NEW = `aecc921`.

| Experts | 128 tokens | 512 tokens | 2048 tokens |
|---:|---:|---:|---:|
| 64 | 25.71 → 25.27 (0.983) | 59.43 → 58.83 (0.990) | 195.76 → 194.71 (0.995) |
| 128 | 25.53 → 25.36 (0.993) | 51.80 → 51.49 (0.994)* | 161.47 → 161.39 (1.000) |
| 256 | 26.81 → 26.32 (0.982) | 52.72 → 52.19 (0.990) | 161.87 → 161.45 (0.997) |

Ratios range from 0.982 to 1.000. The two implementations perform similarly
in the measured cells. At e128/t512, adjacent independent runs of the same
NEW CFT binary had second/first ratios of 0.9882–1.0254. Across all matrix
arms, retained repetition spread `(max-min)/median` ranges from 0.03% to
15.60%; the largest spread is COMBO CFT e64/t128.

### (c) Same binary: store versus CFT

Both arms use the `aecc921` binary. Each cell shows **store µs / CFT µs =
ratio**; >1 means CFT is faster. The store arm has `NCCL_EP_LL_CFT` unset;
the CFT arm sets it to `1`.

| Experts | 128 tokens | 512 tokens | 2048 tokens |
|---:|---:|---:|---:|
| 64 | 30.70 / 25.27 = **1.215** | 70.46 / 58.83 = **1.198** | 309.32 / 194.71 = **1.589** |
| 128 | 33.23 / 25.36 = **1.310** | 59.97 / 51.49 = **1.165** | 183.59 / 161.39 = **1.138** |
| 256 | 33.35 / 26.32 = **1.267** | 60.02 / 52.19 = **1.150** | 184.03 / 161.45 = **1.140** |

The measured store/CFT ratio ranges from **1.138 to 1.589**.

## Validation evidence

Fresh normal functional validation of `aecc921` passed 3/3 `ep_test`
invocations on four ranks (LL with CFT off/on and HT sanity) and 12/12
four-rank `ep_bench --validate` gates. Those gates cover expert-major and
rank-major with CFT off/on at e128/k8/t512; rank-major k2/k6/k16 with CFT
off/on at e128/t137; and CFT expert-major e64/k8/t2048 and e256/k8/t32.
CFT debug layout gates initialized on 4/4 ranks when enabled and 0/4 when
unset. Eight-rank validation across two nodes passed 4/4 gates at
e128/k8/t512 (expert-major/rank-major × CFT off/on), with CFT initialization
on 8/8 ranks when enabled and 0/8 when unset. Sources:
`publication-mr244/validate_results.txt`, `rank8_results.txt`, and per-run logs.

Historical evidence remains separate: implementation `c433242` passed
40/40 four-rank `ep_bench --validate` cases on hecate0356 (both layouts,
CFT off/on, e64/e128/e256 × t32/t512/t2048 at k8, plus rank-major k2/k16
with CFT off/on). The original scan's host emulation reported identical
outputs for 4,000 random configurations; it was not rerun for this revision.

The sanitizer gate remains **FAILED/unresolved**. Four-rank `ep_test -a ll`
under memcheck encountered application receive-count/combine mismatches
on both `aecc921` and baseline `2355f2b`. MPI-filtered repeats reported no
CUDA API errors but still failed application checks: full passes out of
two attempts were OLD store 1/2, OLD CFT 1/2, NEW store 0/2, NEW CFT 1/2.
Baseline reproduction does not resolve the cause or exclude a regression;
no sanitizer-tool defect is assumed. These failures remain separate from
the normal functional passes above. Source:
`publication-mr244/SANITIZER_NOTE.md` and `sanitizer_retry_results.txt`.

## Current scope and remaining limitations

- The scan is compiled only into the `kCft` specialization; STORE keeps
  `countTokensPerRank_mask`. Both scans produce identical output (host
  emulation over 4000 random configurations; `--validate` passes on both paths
  in both layouts).
- Each JIT variant contains one scan. STORE and CFT share the source template,
  but the compiler discards the other scan and its local state.
- A counting warp's window is limited to 32 ranks. Larger windows would need
  a wider mask or a second pass.
- The batch shape, four tokens by eight entries per lane, is fixed at compile
  time. It was chosen for the register budget of the dispatch kernel.
- Every CTA still reads the whole top-k table. Sharing one scan result across
  the CTAs that serve the same rank is a possible extension and is not part of
  this design.
- Both expert-major and rank-major layouts use the same scan; the layout does
  not change the count.
