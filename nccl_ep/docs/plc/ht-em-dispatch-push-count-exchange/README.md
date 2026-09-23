# HT EM Push Dispatch with Count Exchange

This design removes the routing-bitmap AllGather and global scan from the
critical path of high-throughput (HT), expert-major (EM) local-permute
dispatch. Its fused delivery mode is also called *fused-meta-dispatch*.

## Problem

The scan path communicates every token's routing decision before it can start
dispatch:

1. `ncclEpUpdateHandle` AllGathers the per-token expert bitmap.
2. A global scan turns that bitmap into receive slots, expert offsets, and
   routing maps.
3. Dispatch moves the token payload using those precomputed maps.

The communicated bitmap grows with tokens and experts. On wide EP domains,
metadata preparation can cost more than moving the payload and must finish
before dispatch starts.

## Decision

Count exchange communicates only aggregate routing information before payload
movement. The source retains token-level routing in its handle until forward
dispatch, where MAP warps use it to build payload slots and send each
destination only the local-expert membership for rows delivered there. No rank
receives another source's full token-to-expert bitmap.

```text
Scan path
  global token/expert bitmap AllGather -> global map scan -> payload dispatch

Count path
  local histogram -> exchange compact counts -> dispatch {
      reserve FLAT slots + build s2d + send payload and destination LERM
  } -> receiver builds flat2em while relocating FLAT -> EM
```

The tradeoff is deliberate: count mode adds MAP work and peer LERM writes to
dispatch, but removes token-dependent collective metadata and the standalone
global scan from the pre-dispatch critical path.

Counts reserve non-overlapping receive ranges. The source's private routing
data then identifies which token belongs in each range. This is enough to
rebuild the exact maps needed by dispatch without sharing the full routing
bitmap.

For source rank `s`, the exchanged row is

```text
count_row[s] = [ rank_counts[R] | expert_counts[E] ]
```

- `rank_counts[d]` is the number of source tokens routed to destination rank
  `d`. Multiple expert hits on the same destination rank count once.
- `expert_counts[e]` is the number of assignments to global expert `e`.

The two halves answer different questions. Rank counts assign deterministic
FLAT receive ranges to each source. Expert counts size and offset the receiver's
expert-major zones. The global token-to-expert matrix is never exchanged; each
destination receives LERM only for the FLAT rows routed to it.

The routing artifacts are:

| Term | Meaning | Producer and consumer |
| --- | --- | --- |
| FLAT | Rank-major receive staging with one payload row per `(source token, destination rank)` | Dispatch writes it; the dispatch epilogue and local permute read it |
| EM | Final expert-major output with one row per local-expert assignment | `local_permute_dup` writes it for the caller |
| s2d (sparse-to-dense map) | `(source token, destination rank) -> FLAT slot` | MAP builds it; the payload store uses it to place rows |
| LERM (local-expert routing map) | Packed bitmap `FLAT slot -> local experts` | Source MAP writes it to the destination; the receiver extracts weights and builds EM placement from it |
| `flat2em` | Per-FLAT-row list of up to `num_topk` EM slots | Forward local permute persists it for backward and combine |

One FLAT row can therefore fan out to several EM rows on the same destination.

## Scope and selection

`dispatch_push_count_capable(group, layout)` selects count exchange when all of
the following hold:

```text
HT expert-major layout
&& local-permute dispatch
&& one LSA team
&& NCCL_EP_HT_EM_AG_SCAN_MODE is disabled
```

A single LSA team can span multiple physical nodes. `is_internode_available`
means that the group contains multiple LSA teams connected by the GIN path;
those configurations still use the scan path. Pull-push, local-dup, NVLink-dup,
FLAT layout, and explicit scan mode also use scan.

Once count exchange is selected, count rows have two delivery modes:

| Mode | Selected when | Count delivery | Layout available |
| --- | --- | --- | --- |
| Fused | No early layout output is requested, the group is not eager, and `NCCL_EP_HT_EM_COUNT_UNFUSED` is off | Dispatch writes each source row directly into every LSA rank's landing table | During dispatch |
| Unfused | Any layout output is requested, eager mode is active, or `NCCL_EP_HT_EM_COUNT_UNFUSED=1` | `ncclEpUpdateHandle` AllGathers the compact rows | Before dispatch |
| Scan | Count exchange is not capable or scan mode is forced | AllGather of per-token routing metadata | Before dispatch |

Fused and unfused describe only when count rows reach the receiver. Both use
the same count histogram and rebuild routing maps during forward dispatch;
backward dispatch reuses the maps saved by forward. Neither is the scan
algorithm.

An allocated `layout_info` wrapper with no output tensors does not force the
unfused path. `NCCL_EP_HT_EM_COUNT_UNFUSED` changes count-row delivery only; it
does not enable count exchange for an unsupported configuration.

Eager EM requires the unfused path because it sizes receive buffers from
`getNumRecvTokens()` before dispatch. It still avoids the full routing-bitmap
exchange and scan.

## Contract

Count exchange preserves the observable routing contract of the scan path:

- Subject to the configured overflow policy, every valid source token reaches
  the same destination ranks and local experts, with the same token,
  probability, and scale data.
- `expert_counters`, `expert_offsets`, and `recv_total_counter` keep their
  dtype, padding, and overflow semantics.
- Forward dispatch persists the maps needed by backward dispatch and combine.
- DROP never writes beyond the receive budget; TRAP reports overflow instead of
  allowing a peer write out of bounds.
- CUDA graph replay advances synchronization state from captured device work.

The physical order of rows within an expert is not part of this contract.
Count mode assigns expert-major slots with atomic cursors, so its row order can
differ from scan. The persisted `flat2em` map keeps forward, backward, and
combine consistent with the order chosen by forward dispatch.

## Design

### Metadata built at UpdateHandle

`build_count_metadata` walks the source's `topk_idx` once and creates:

- the source's count row;
- per-chunk destination-rank counts;
- a per-token destination-rank bitmap;
- a `cached_topk_idx` snapshot, kept local to the source rank. In count mode,
  forward MAP warps read it during dispatch to rebuild s2d and LERM, standing
  in for the routing bitmap scan mode would otherwise have precomputed; and
- the existing payload gates and source token count.

All token-level outputs are handle-owned. Of these preprocessing outputs, only
the count row is exchanged before map construction. This separation is the
central design choice: global aggregates establish ownership of receive ranges,
while source-local detail fills those ranges.

Each rank then plays both sides of the protocol. As a sender, it uses all count
rows to place its local tokens in every destination's FLAT staging buffer and
writes their local-expert membership. As a receiver, it uses the expert-count
columns to publish the layout of its own expert-major output.

### Forward fused flow

The following is a schematic schedule, not a measured timeline. Solid arrows
carry metadata; double arrows cross an LSA peer mapping; the payload path is
labelled separately.

`HEAD` is the dispatch block's head warp, which fans out the count row.
`MAP` denotes the metadata-building warps. `G2S` and `S2G` are the payload
pipeline's global-to-shared load and shared-to-peer-global store. LERM is the
local-expert routing map that associates each received FLAT row with this
rank's experts.

```text
UpdateHandle, source rank s
  topk_idx
      |
      v
  count histogram
      +--> private count_row[s]
      +--> private chunk counts, rank bitmap, topk snapshot
  return; no collective

Dispatch kernel
  HEAD producer
    count_row[s] ==IPC write==> each LSA rank's group-owned landing table
                              [system fence + per-CTA LSA barrier]
                                                   |
  MAP consumer/producer                            v
    all source rows --------------------------> build sparse-to-dense map
    source-private bitmap + top-k + s2d -------> write destination LERM rows
    expert counts --------------------------> publish local expert layout
                         |
                         +-- release readiness per chunk --+
                                                           |
  PAYLOAD                                                  v
    token + dense probability (+ scale)
        -> G2S shared-memory queue -> S2G ==IPC==> destination FLAT staging

  [dispatch tail completes grid + inter-rank rendezvous; kernel exits]

Dispatch epilogue
  landed dense probability + landed LERM
      -> dense_to_sparse_prob -> FLAT sparse weights

Receiver local_permute_dup
  FLAT tokens + FLAT sparse weights + landed LERM
      -> caller expert-major output
      -> handle-private LERM and flat2em maps for combine/backward
```

The MAP work overlaps the payload pipeline. It publishes each s2d chunk before
G2S or S2G consumes routing state for that chunk. The later LERM construction
is off the S2G critical path and must finish before the dispatch epilogue reads
it.

### Forward unfused flow

Unfused mode uses the same histogram and the same dispatch map builder. It
moves compact count delivery earlier so the caller can observe the layout.

```text
UpdateHandle
  count histogram
      -> AllGather(count rows)
      -> cached rows in handle scratch
      -> compute_layout_info
      -> caller layout tensors ready

Dispatch kernel
  no count-row fan-out
  MAP reads cached rows -> sparse-to-dense map, peer LERM, layout outputs
  MAP readiness -------+-> G2S/S2G payload pipeline

  [dispatch tail completes grid + inter-rank rendezvous; kernel exits]

Dispatch epilogue
  dense probability + LERM -> dense_to_sparse_prob -> FLAT sparse weights

Receiver local_permute_dup
  FLAT tokens + FLAT sparse weights + LERM
      -> expert-major output + persistent maps
```

Dispatch writes any layout tensors supplied to that call. If
`ncclEpUpdateHandle` received layout output tensors, count-row delivery is
unfused; dispatch may still write the same or different tensors. Supplying the
outputs only to dispatch does not itself disable fused delivery.

### Why the reconstruction is sufficient

For destination rank `d`, prefix sums of `rank_counts[d]` across source ranks
give each source a disjoint FLAT segment. Per-chunk rank counts refine that
segment into chunk bases. Within a chunk, the source's rank bitmap supplies the
final token offset. This produces the sparse-to-dense map consumed by the
payload pipeline.

For each routed token, the source combines its private top-k row with the
assigned FLAT slot and writes a packed local-expert routing map (LERM) to the
destination. The receiver therefore learns expert membership without receiving
the source's full routing bitmap. `dense_to_sparse_prob` first uses LERM to
compact the landed dense probability row into FLAT sparse weights. Then
`local_permute_dup` uses the same LERM to assign expert-major slots and saves
that assignment for later operations.

### Work partitioning

Forward count mode uses the same chunk schedule in fused and unfused delivery.
Let `B` be the dispatch CTA count, `C` the tokens per chunk, and `P` the fitted
payload-pipeline count:

```text
chunk c -> CTA (c % B)
CTA-local chunk q = floor(c / B) -> payload pipeline (q % P)
```

`B` is a grid size, not a physical-SM mapping. CUDA schedules those CTAs onto
available SMs; the chunk assignment stays with the CTA wherever it runs.
Each payload pipeline is one G2S warp paired with one S2G warp. The pair walks
the selected tokens in its chunks; it does not assign one token permanently to
each warp or lane. For s2d, the two MAP warps alternate CTA-owned chunks. For
LERM, they alternate 32-token waves, with one lane owning one token in a wave.
The automatic single-LSA configuration normally resolves to `B=16`, `C=64`,
and `P=2`; environment overrides and shared-memory fitting can change these
values.

`local_permute_dup` has a separate launch and schedule. Its grid has
`shuffle_sms` CTAs when configured, otherwise `device_sm_count` CTAs. Eight
permute warps per CTA each own one received FLAT token at a time, grid-stride
through later tokens, and cooperate across the token row and its local-expert
destinations. A ninth warp zeros expert padding. Backward dispatch skips MAP
construction and consumes the maps persisted by forward.

## Ownership, ordering, and reuse

| State | Owner | Lifetime and use |
| --- | --- | --- |
| Count row, chunk counts, rank bitmap, top-k snapshot, gathered rows | Handle | Produced by UpdateHandle; consumed by that handle's dispatch |
| s2d, expert offsets/counts, persisted LERM and `flat2em` | Handle | Produced or finalized by forward dispatch; reused by backward/combine |
| Count-row landing table, incoming LERM table, FLAT token/probability/weight staging, cursors, barrier state | Group | Reused by ordered dispatches on the group |
| Expert-major output and optional layout tensors | Caller | Written according to the API's stream ordering |

The landing table is indexed by sender rank, not by handle or CTA. Preparing
multiple handles ahead is safe because UpdateHandle writes handle-private
scratch. Their dispatches must obey the group's existing ordered-use contract;
arbitrary concurrent dispatches on different streams would alias group-owned
landing, staging, and cursor state.

The ordering chain is:

1. Stream order makes the histogram visible to dispatch.
2. In fused mode, a system fence and, for teams larger than one rank, an LSA
   barrier make each count row visible to the destination MAP consumers. Every
   CTA receives the rank-count prefix it needs; CTA 0 also owns expert-layout
   publication.
3. MAP publishes a per-chunk readiness counter with release semantics after
   the sparse-to-dense stores it protects are ordered and visible. G2S and S2G
   acquire that counter before reading the chunk.
4. MAP system-fences peer LERM writes. After all CTAs arrive, the elected
   dispatch tail performs the inter-rank completion rendezvous. Kernel
   completion is then the receiver boundary for incoming LERM and FLAT
   token/probability payload.
5. On the same stream, `dense_to_sparse_prob` consumes the landed probability
   and LERM to produce FLAT sparse weights.
6. `local_permute_dup` consumes FLAT tokens, weights, and LERM, then copies LERM
   into handle-private storage and persists `flat2em` before group buffers can
   be reused.

Device-resident expected-generation counters preserve this sequence across
CUDA graph replays without host-side generation updates.

## Overflow and padding

FLAT capacity and padded expert-major capacity are related but distinct.
Count mode applies the same policy at every place that can create or consume a
slot:

- s2d and LERM construction use the same FLAT-slot clamp.
- Payload stores skip a slot marked as dropped.
- Published expert counts and offsets are clamped to the receive budget.
- `recv_total_counter` reports the pre-drop padded requirement used for eager
  sizing.
- With TRAP, an over-budget FLAT or padded total emits the scan-parity
  diagnostic and traps before an unsafe peer access.

Under DROP, sender-rank FLAT ordering and expert-major padding can retain
different token subsets. That can leave a published expert zone larger than
the number of rows actually written. The last `local_permute_dup` block zeros
this phantom-row gap and corrects the handle's active counts. Backward uses the
corrected counts and forward's saved map, including when its output buffer is a
different allocation.

## Cost and resource model

Let `R` be ranks in the single LSA team, `E` global experts, `T` maximum source
tokens, `K` top-k, and `C` tokens per dispatch chunk.

| Resource | Scan path | Count path |
| --- | --- | --- |
| Logical routing input received per rank | `R * T * ceil(E/8)` bytes of per-token expert bitmap | `4 * R * (R + E)` bytes of count rows |
| Global preprocessing | Separate scan before dispatch | Routing reconstruction overlapped with dispatch |
| Source-local routing detail | Derived from gathered bitmap | `O(T * ceil(R/64))` rank bitmap, `O(ceil(T/C) * R)` chunk counts, and `O(T * K)` top-k snapshot |
| Routing maps | Prebuilt s2d, LERM, `flat2em`, and expert layout | s2d built at the source, LERM written to the receiver, and `flat2em` built at the receiver |
| Peer landing storage | Routing/scan storage | Group-owned `4 * R * (R + E)`-byte count-row region plus LERM region |

The current fused implementation republishes the rank-count prefix from every
dispatch CTA because every CTA needs its own prefix before it can release a
payload chunk. CTA 0 alone publishes the expert-count tail. This spends extra
token-count-independent metadata traffic beyond the logical row footprint to
avoid a grid-wide dependency in front of S2G.

Current limits are deliberate: count exchange is single-LSA-team only,
`num_topk` is capped at the compile-time constant `kEpCountMaxTopk` (`64`, not
a per-group setting), and destination-rank metadata is bounded at `1024`
ranks. The top-k limit protects top-k-indexed storage and the per-token
deduplicated destination-rank list, both statically sized to `kEpCountMaxTopk`
entries since a token can touch at most `num_topk` distinct ranks; count-row
width itself is `R + E`.

## Historical performance signal

The measurements below use BF16 expert-major HT, `experts = WORLD_SIZE * 8`,
warmup 10, and 50 measured iterations. `Sum = UpdateHandle + Dispatch +
DispatchEpilogue`; speedup is `scan.sum / mode.sum`. In the configuration
column, `k` is top-k, `h` is hidden width, and `t` is tokens per rank.

All listed multi-node configurations were one LSA team (`rdma_team_size == 1`),
so count exchange ran for every value. The `4n` and `8n` labels describe
physical placement, not the unsupported multiple-LSA-team case. `nvl` denotes
single-node NVLink, `mnnvl` multi-node NVLink, and `ib` the IB/GDAKI runs.

These numbers are historical directional data. The document does not retain
their source SHA, GPU SKU, CUDA/NCCL versions, exact commands, or raw logs.
Performance sign-off therefore remains pending a reproducible rerun.

Fused versus scan:

| Config | 1n_nvl | 4n_mnnvl | 4n_ib | 8n_mnnvl | 8n_ib |
| --- | ---: | ---: | ---: | ---: | ---: |
| k=8 h=7168 t=4096 | 1.01x | 1.04x | 1.06x | 1.13x | 1.07x |
| k=8 h=7168 t=8192 | 1.10x | 1.04x | 1.06x | 1.09x | 1.04x |
| k=22 h=3072 t=4096 | 1.13x | 1.22x | 1.23x | 1.26x | 1.18x |
| k=22 h=3072 t=8192 | 1.17x | 1.20x | 1.19x | 1.25x | 1.15x |

Unfused versus scan:

| Config | 1n_nvl | 4n_mnnvl | 4n_ib | 8n_mnnvl | 8n_ib |
| --- | ---: | ---: | ---: | ---: | ---: |
| k=8 h=7168 t=4096 | 1.01x | 0.98x | 1.04x | 1.12x | 1.02x |
| k=8 h=7168 t=8192 | 1.02x | 1.02x | 1.04x | 1.07x | 1.02x |
| k=22 h=3072 t=4096 | 1.12x | 1.17x | 1.18x | 1.22x | 1.14x |
| k=22 h=3072 t=8192 | 1.23x | 1.18x | 1.14x | 1.24x | 1.13x |

The benefit grows with routing width because the removed bitmap depends on
tokens and experts while a count row depends only on ranks and experts.
Single-node, low-top-k cases are approximately neutral. Multiple-LSA-team
performance is unchanged because both selections use scan.

## Validation and sign-off

The repository test driver exercises the mode-sensitive suites under default
count mode, forced scan, and forced unfused count. The table describes current
coverage and required external runs; it is not a pass record for a particular
source revision.

| Area | Modes | Coverage |
| --- | --- | --- |
| Automated mode matrix | Fused count, unfused count, scan | `run_tests.sh` reruns EM-affected suites with `NCCL_EP_HT_EM_AG_SCAN_MODE=1` and `NCCL_EP_HT_EM_COUNT_UNFUSED=1` |
| Layout outputs | Fused count, unfused count, scan | `test_output_layout.cu`: dispatch metadata, int32/int64 totals, alignment, combine, and early/deferred publication |
| Overflow | Fused and unfused cases in the default count run; forced scan rerun | `test_ht_overflow_drop.cu`: aligned DROP, deep FLAT overflow, and forward/backward phantom rows |
| Map reuse | Fused count, unfused count, scan | Forward maps consumed by combine and backward dispatch |
| CUDA Graph replay | Required regression | Capture forward dispatch/combine and launch the same graph repeatedly; `ep_test -g` currently launches it only once |
| End-to-end topology matrix | Required external run | `ep_bench -a ht -L em -V` on 1n NVL and single-LSA 4n/8n MNNVL and IB |
| Fallback | Required external run | Automatic scan selection and correctness on a two-LSA-team group |

Repeated CUDA Graph replay is part of the contract, so its missing regression
is a correctness sign-off blocker rather than follow-up performance work.

TRAP acceptance must force a receiver over budget on the fused path, produce
the scan-equivalent diagnostic, and avoid peer-IPC out-of-bounds access while
the trap propagates. `--fused-meta-dispatch -R 4096` selects the relevant
benchmark path and budget, but the sign-off record must also capture the exact
topology, routing input, full command, and result.

PLC sign-off should attach the source SHA and build configuration, exact test
commands with pass/fail output, system topology and software versions, and raw
performance logs for the fused/unfused/scan comparison. Until those records
and the repeated-replay regression exist, this document does not establish PLC
sign-off.

## Source map

| Responsibility | Implementation anchor |
| --- | --- |
| Selection, allocation, UpdateHandle, dispatch wiring | `nccl_ep/nccl_ep.cc`: `dispatch_push_count_capable`, `ht_update_handle_count_mode`, `ncclEpDispatch` |
| Source histogram and private routing metadata | `nccl_ep/device/count_kernel.cuh`, launched from `device/jit/preprocess_jit.cuh` |
| Count delivery, reconstruction, readiness, and output publication | `nccl_ep/device/ht_ep.cuh`: `dispatch_push_map_warp` and its phase helpers |
| Host/device parameter marshaling and layout kernel | `nccl_ep/device/ht_ep_adapter.{cuh,cu}` |
| Dense-probability extraction after dispatch | `nccl_ep/device/ht_ep_adapter.cu`: `dense_to_sparse_prob` |
| FLAT-to-EM relocation and persistent maps | `nccl_ep/device/ht_ep.cuh`: `local_permute_dup`, launched through `device/jit/ht_dispatch_jit.cuh` |

## Open work

- **Multiple LSA teams.** Extend count exchange through the GIN metadata path.
  Peer-specific packed expert counts are the natural wire format for that work;
  they are not justified for the current single-LSA path by themselves.
- **Publication scaling.** Measure CTA 0 publication plus explicit readiness
  against the current duplicate rank-prefix writes and per-CTA prefix rebuild.
  Any replacement must preserve chunk-level MAP-to-payload ordering.
- **Large chunk counts.** The current chunk-base helper rescans earlier chunks.
  It has not been a bottleneck through 16K tokens/rank with a forced 32-token
  chunk, but an incremental prefix is available if larger domains expose the
  quadratic work.
- **Concurrent dispatch.** Supporting independent streams on one group would
  require handle-qualified landing/staging slots or an equivalent generation
  protocol; the current group-owned buffers intentionally assume ordered use.
