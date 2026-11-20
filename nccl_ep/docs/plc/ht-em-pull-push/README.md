# HT EM Pull Dispatch + Push Combine

## The constraint this design lifts

The existing HT EM modes (`kLocalPermute`, `kLocalDup`, `kNvlinkDup`) are all
sender-driven NVLink pushes, so the receive-side EM buffer must be peer-visible
on the sender GPU. That forces two costs on every call:

- `recv_x_em` (the dispatch output feeding the grouped GEMM) and the combine
  source must sit in IPC-shared memory, not caller memory.
- The EM layout is reached through a second kernel — dispatch writes a FLAT
  staging then `local_permute_dup` scatters into EM slots, and combine runs
  `local_permute_reduce` before the NVLink push. One extra HBM round trip per
  token on each leg.

Pull dispatch reverses the direction. Each receiver reads the rows it needs
from the sender's caller-owned input, so `recv_x_em` becomes local and the
local permute collapses into the same kernel. Push combine mirrors this: local
reduce first, then push one row per `(t, R)`, so the combine source is local
too.

## Requirements

- Keep dispatch output and combine source local; the caller hands `recv_x_em`
  straight to the grouped GEMM.
- Fold `local_permute_dup` into dispatch and `local_permute_reduce` into
  combine; no separate HBM round trip.
- Preserve the existing intra-LSA head/tail synchronization contract; add no
  persistent state.
- Preserve dispatch recipes `NONE` and `QUANT_FWD`; preserve forward and
  backward combine for the `NONE` recipe.
- Bound peer-visible allocations by the actual received token count, not by
  the per-EM-copy fan-out.

## Design

**Source:** `nccl_ep/device/ht_ep.cuh` (`PullTmaStage`, `dispatch_pull`,
`combine_push`, `combine_reduce`), `ht_ep_adapter.{cuh,cu}` (launchers),
`nccl_ep/nccl_ep.cc` (mode wiring).

**Scope:** HT + `EXPERT_MAJOR` layout + single LSA team (one NVLink domain).
Dispatch recipes `NONE` and `QUANT_FWD`; push-combine `NONE` fwd/bwd.

### Mode gate

`HtEmMode::kPullPush` is added alongside the three existing enum values
(`nccl_ep.cc:768-772`). `NCCL_EP_HT_PULL_PUSH` selects pull/push at group
creation for both `EXPERT_MAJOR` and `FLAT` layouts when `num_lsa_teams == 1`;
the old `NCCL_EP_HT_EM_PULL_PUSH` name is no longer recognized. This document
describes the expert-major implementation. The single-team requirement is
rejected otherwise (`nccl_ep.cc:1931-1955`). `em_pull_enabled()` and
`em_comb_push_enabled()` both key off the enum, so the two legs move together.

### Pull dispatch

`dispatch_pull` (`ht_ep.cuh:5533`) uses one warp per recv slot per wave. Each
pull warp keeps exactly one remote row in flight through a per-warp
`PullTmaStage<kRowBytes>` (`ht_ep.cuh:65`): one 16B-aligned smem row buffer and
one `mbarrier`. `issue()` starts `cp.async.bulk` from
`peer_input_ptrs[src_rank]`; the wait is deferred so the warp overlaps the
remote TMA with control-plane work — ballot-compacting
`flat2em_slot_map[slot,*]` and pulling the sparse `topk_weights` at
`srcpos_row[k]`. On completion the warp `cp.async.bulk`-stores the row into
every active EM slot. `bulk_wait_reads` (`cp_async_bulk_wait_group_read<0>`)
releases the smem row so the next iteration can issue while the stores drain;
one `bulk_drain` per warp before block exit completes them.

```text
Per pull warp (one recv slot per wave)

  +----------------------------------------------------------------------+
  | [1] issue TMA (non-blocking)                                         |
  |     cp.async.bulk  peer_input_ptrs[src] -> row_buf  (HiddenInt4*16B) |
  +----------------------------+-----------------------------------------+
                               |  in-flight
                               v
  +----------------------------------------------------------------------+
  | [2] control-plane, hides under [1]:                                  |
  |     - ballot-compact flat2em_slot_map[slot,*] -> s_active[0..cnt)    |
  |     - src_w[srcpos_row[k]] -> recv_topk_weights_em[es] (order-pres)  |
  |     - QUANT_FWD: issue TMA for the scale row (2nd smem stage)        |
  +----------------------------+-----------------------------------------+
                               |
                               v
  +----------------------------------------------------------------------+
  | [3] wait: mbarrier_try_wait_parity(mbar); flip phase                 |
  +----------------------------+-----------------------------------------+
                               |
                               v
  +----------------------------------------------------------------------+
  | [4] for a in [0, cnt):                                               |
  |       cp.async.bulk  row_buf   -> recv_x_em[s_active[a]]             |
  |       QUANT_FWD:     scale_buf -> recv_x_scale_em[s_active[a]]       |
  +----------------------------+-----------------------------------------+
                               |
                               v
  +----------------------------------------------------------------------+
  | [5] bulk_wait_reads: cp_async_bulk_wait_group_read<0> (release the   |
  |     smem rows; the stores keep draining while the next row is in     |
  |     flight). One counter covers both the token and scale stages.     |
  +----------------------------------------------------------------------+

One bulk_drain (cp_async_bulk_wait_group<0>) per warp before the block exits
completes the stores.
```

`QUANT_FWD` instantiates a second `PullTmaStage` at the scale-row width, so
the remote scale read rides the async proxy alongside the token row. The
scale stage is used when the scale row is a multiple of 16B and its staging
fits; otherwise the register path is kept.

Concurrently, one pad warp per block zero-fills each expert's trailing pad
range (past `per_expert_counts_active[e]`) via vectorized `st.global.cg int4`.
The pad range is output-disjoint from the pull warps, so the overlap is
covered by the block-exit `__syncthreads`.

**CTA layout.** `kPullDispatchThreadsPerSlot = 32`,
`kPullDispatchMaxPullWarps = 24`, `kPullDispatchPadWarps = 1`. The pull-warp
count is JIT-picked as `clamp(smem_optin / blocks_per_sm / smem_per_warp, 1,
24)`, so the kernel takes the largest count the device shared memory allows.
A row too large to stage even one warp is rejected in `ncclEpDispatch`
(`nccl_ep.cc:4385`).

**Transposed read schedule.** Slots are pre-sorted by source rank, so a
contiguous window per warp would cluster each wave's reads on one peer. Pull
uses `token = gwarp * num_waves + wave` so each wave stripes across all
source ranks, spreading NVLink read bandwidth.

**Dynamic chunk claim (fused count).** With the fused count exchange, the MAP
warps of each block build per-chunk row ranges (32 tokens, one source rank)
that the pull warps of the same block consume. A static chunk range per CTA
leaves CTAs finishing tens of us apart (rows per chunk and link rates vary),
which is a 30-37 us kernel tail at EP16/32 on Rubin. MAP warps claim chunks
dynamically instead:

- **Static first chunk.** Every MAP warp takes its first chunk statically
  (`blockIdx * 2 + map_idx`), so no claim atomic sits on its critical path.
- **Grid-wide claim.** A MAP warp takes its next chunk with
  `atomicAdd(chunk_claim, 1)`; a value at or past `nRanks * ceil(tokens / 32)`
  ends the warp. `chunk_claim` is the word after `layout_ready` in
  `pull_recv_cursor` (one int larger) and is zeroed with `layout_ready`: by the
  head leader before the head gate on the fused flow, by
  `lsa_map_publish_kernel` on the unfused flow. Row placement is unchanged
  (the layout scan fixes it); only which warp handles a chunk varies.
- **Per-block throttle.** A MAP warp claims only while its next unit is below
  `next / 32 + kPullDispatchChunkLookahead * kPullDispatchMapWarps`, where
  `next` is the block pull-warp ticket counter, so the last chunks go to the
  CTAs that drain first.
- **Fixed ring.** Chunk ranges live in a per-warp ring of
  `kPullDispatchChunkRing` entries instead of a buffer sized for the worst-case
  per-CTA chunk count; a `static_assert` keeps it above the lookahead.
- **Read acks.** Each ring slot has a read counter (`chunks_read`). A pull warp
  adds one (release) after it copies the slot's `first` and `count`; a chunk's
  `kPullDispatchChunkTokens / kPullDispatchSlotBatch` tickets are its readers.
  A MAP warp reusing a slot waits (acquire) until the counter shows every
  reader of the previous chunk, so a delayed pull warp never sees a newer chunk
  in its entry.
- **Done flags.** Each MAP warp sets a done flag (release) after its last
  `ready` store. `dispatch_pull_wait_chunk` returns `kReady`, `kSkip` (the unit
  is never produced; take a new ticket) or `kFinished` (every MAP warp is done
  and none produces this unit).

A pull warp blocks only on a unit `v <= next / 32`; the owner MAP warp of `v`
has its next unit `u <= v`, which the throttle always admits, so the owner is
never throttled while a pull warp waits on it. A CTA whose MAP warps claim
nothing sets all done flags and its pull warps leave on `kFinished`. The ring
wait cannot deadlock: the throttle admits a MAP warp's chunk only after the
tickets of the chunk that last used its slot are all claimed, and a claimed
ticket reads an already published entry without blocking. The ring size is a
performance knob, not a correctness bound. Ordering: the MAP warp waits for
the slot's reads, stores the range, then releases `ready` at block scope; the
pull warp acquires `ready`, reads the range, then releases a read ack; the grid
counter publishes no data. `test_ht_pull_chunk_ring` wraps the ring and delays
one pull ticket per block. Row contents
are identical; only the run-to-run row order changes, as it already does with
the `rank_cursor` and `expert_cursor` atomics.

`NCCL_EP_HT_DISPATCH_DYNAMIC_CLAIM_MIN_TOKENS` (default 2048, 0 always claims)
selects the mode per call from the handle's tokens per rank. Below it the MAP
warps keep the static chunk assignment (chunk `round * stride + blockIdx * 2 +
map_idx`), because claiming costs more than it balances there.

### Push combine

`combine_push` (`ht_ep.cuh:5943`) is one warp per recv slot, with slots
claimed dynamically (see "Row claiming" below). The warp
ballot-compacts `flat2em_slot_map[slot,*]` to get the `n <= top_k` valid EM
copies, decodes `(dst_rank, t)` from `recv_slot_to_src[slot]`, and reduces
the `n` rows into an FP32 accumulator tiled by `kHiddenVec int4` lanes. The
row can be split into chunks of `kChunkSize` int4 (a whole number of tiles of
`kHiddenVec * 32` int4, `combine_push_buf_int4`), each reduced into the *same*
per-warp smem buffer and pushed with its own TMA store, so the buffer only
holds one chunk, not the whole row (see "Chunking" below). In the
backward pass the pushed row carries a co-located pad: after the full row is
pushed, `prob_dst[srcpos[k]] = topk_weights_em[k]`.

```text
Per warp (one claimed recv slot per iteration, n <= top_k valid EM copies)

  +----------------------------------------------------------------------+
  | [1] ballot-compact flat2em_slot_map[slot,*] -> smem[slot_in_block]   |
  |     n = popcount(valid)                                              |
  +----------------------------+-----------------------------------------+
                               |
                               v
  +----------------------------------------------------------------------+
  | [2] (dst_rank, t) = decode_src(recv_slot_to_src[slot], ...)          |
  |     s_dst = peer_staging_ptrs[dst_rank]                              |
  |              + (t*team_size + my_lsa_rank) * stride                  |
  +----------------------------+-----------------------------------------+
                               |
                               v
  +----------------------------------------------------------------------+
  | [3] for tile in [0, kNTiles):    one chunk per tile (whole row if    |
  |                                   chunking is pinned off)            |
  |       tile>0: tma_store_wait<0> -- block until the prior tile's      |
  |             TMA store has drained row_buf, so it's safe to reuse     |
  |       reduce n copies -> FP32 acc for this tile (k = 0..n-1),        |
  |             copy loads issued in batches before they are summed      |
  |       acc -> row_buf                 (same smem buffer every tile)   |
  |       tma_store_fence; tma_store_1d row_buf -> s_dst[tile's bytes]   |
  +----------------------------+-----------------------------------------+
                               |
                               v  (BWD only)
  +----------------------------------------------------------------------+
  | [4] BWD: prob_dst = s_dst + row_bytes  (co-located pad)              |
  |          for each copy:  prob_dst[srcpos[k]] = topk_weights_em[k]    |
  +----------------------------------------------------------------------+
```

**Chunking.** The chunk width is set with `NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE`,
in hidden elements. Unset uses the auto default: one tile (128 int4, 2KB, i.e.
1024 BF16 elements, once a row has at least 97 int4) for EP < 16, and the whole
row for EP >= 16 (see the EP sweep below); on Rubin (SM107) it is one tile at
every EP size. On expert-major rows an explicit value wins on every GPU, and `0`
forces the whole row; flat and MXFP8 combines always push the whole row.
Other values are rounded up to whole tiles and clamped to the row,
and the chunk count is `ceil(hidden / chunk_size)`. A one-tile chunk gives a
2KB per-warp buffer, so the block keeps all `kCombinePushMaxSlots = 24` warps
(`combine_push_slots_per_block`) at any hidden size. The chunk body has a
compile-time copy bound and loads `8 / kHiddenVec` copies per batch before
summing them, and the next claimed slot's routing metadata is loaded one
iteration ahead and decoded with a multiply-high divide. The chunk size (`kChunkSize`) is
a compile-time template parameter of `combine_push`, so each value is its own
JIT binary, keyed by the `_c<int4>` suffix of the variant name. A whole-row
chunk that does not fit the shared memory cap is rejected with a message asking
for a smaller `NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE`; this only applies to an
explicit `0` or a very wide row. A fused MXFP8 row that does not fit is rejected
with its size and the per-warp budget instead, since it has no chunk setting.

Benchmarked (DSv3 k8, EP=4, combine total in us, `ep_bench -a ht -L em`), one
tile per chunk against the whole row. Intermediate sizes were uneven: at hidden
7168, 2 or 3 tiles per chunk were slower than the whole row.

| Hidden (elems) | Row  | Whole row (`CHUNK_SIZE=0`) | Default / `CHUNK_SIZE=1024` |
| -------------: | ---: | ----------------: | ----------------: |
|           2048 |  4KB |               255 |               220 |
|           4096 |  8KB |               495 |               366 |
|           7168 | 14KB |               719 |               600 |

One tile per chunk wins at every measured size, including rows that fit the
smem budget whole, because the 24-warp block and the batched gather no longer
depend on the row width.

Across EP sizes (GB300 MNNVL, 8 nodes, 32 comm SMs, 4096 tokens/rank) one tile
per chunk is clearly faster at EP 4 and 8 (DSv3 -24%/-3%, Nemotron -30%/-12%), a
wash at EP 16, and neutral-to-slower and run-to-run unstable at EP 32 (DSv3 up to
+37%) -- from EP 16 up the combine already runs at the NVLink bandwidth plateau
with no gather latency left to hide. The auto default therefore chunks one tile
for EP < 16 and pushes the whole row for EP >= 16.

The sweep above is GB300. On Rubin (SM107) the one-tile chunk was also faster at
EP 16 and 32, so there the auto default chunks one tile at every EP size.

**Row claiming.** Rows with several valid EM copies cost more than single-copy
rows, and peer NVLink links differ in speed, so static per-warp row ranges
finish unevenly (40-70 us cross-SM skew at EP 32). Warps instead claim rows
through `combine_push_claim_t`:

- Each block owns a contiguous region of `ceil(num_recv / grid)` rows and its
  warps claim rows from it one at a time through a shared-memory counter, so a
  block keeps streaming to one peer.
- The last 1/4 of every region is a grid-wide pool claimed by global ticket.
  Ticket `t` maps to region `t % grid`, pool position `t / grid`, so the pool
  drains evenly across regions and idle SMs pick up rows from slow ones.
- Rows up to 8KB skip the pool and use only the block-owned region: the pool
  atomic costs more than the balance gains on small rows.
- Region assignment is rotated by the rank's LSA index, so concurrent senders
  start on different destinations.
- Below `NCCL_EP_HT_COMBINE_DYNAMIC_CLAIM_MIN_TOKENS` tokens per rank
  (default 512, 0 always claims) claiming is skipped: each warp walks a
  contiguous static range. At 128-256 tokens/rank the claim atomics cost 1-3
  us more than the balance gains; from 512 tokens claiming is faster at every
  row size measured.

Which warp reduces which row changes, but each row is still reduced in the
fixed `k = 0..n-1` order, so the output is unchanged.

`combine_reduce` (`ht_ep.cuh:6292`) is one attn token per 128-thread sub-warp.
It derives which `(t, R)` partials exist from this rank's own `topk_idx` — no
`(t, R)` valid flag rides the NVLink push and no staging zero-init runs:

```
e_id             = topk_idx[t, wlane]        (wlane < num_topk)
rank_valid       = 0 <= e_id < experts_per_lsa_team
src_rank_of_lane = e_id / experts_per_rank
same_rank        = __match_any_sync(0xFFFFFFFF, src_rank_of_lane)   // dedup within topk
rank_leader      = rank_valid && (__ffs(same_rank) - 1 == wlane)
present_ranks    = __ballot_sync(0xFFFFFFFF, rank_leader)           // bitmask
```

Dedup covers the case where two of a token's experts live on the same rank
— `combine_push` already merged those into one pushed row. In the backward
pass, lane `lane < num_topk` reads
`staging[t*team_size + src_rank_of_lane[lane]][row_bytes + lane*4]` to
recover `combined_topk_weights[t, lane]`.

```text
Per attn token t (one 128-thread sub-warp)

  +----------------------------------------------------------------------+
  | [1] wlane in [0, num_topk):                                          |
  |       e_id = topk_idx[t, wlane]                                      |
  |       src_rank_of_lane = e_id / experts_per_rank                     |
  |       dedup via __match_any_sync  (2 experts on same rank -> 1)      |
  +----------------------------+-----------------------------------------+
                               |
                               v
  +----------------------------------------------------------------------+
  | [2] present_ranks = __ballot_sync(rank_leader)  -> bitmask           |
  +----------------------------+-----------------------------------------+
                               |
                               v
  +----------------------------------------------------------------------+
  | [3] for r in present_ranks (bl & (bl-1) peels one bit):              |
  |       acc += staging[t*team_size + r]                                |
  |     attn_output[t] = acc                                             |
  +----------------------------+-----------------------------------------+
                               |
                               v  (BWD only)
  +----------------------------------------------------------------------+
  | [4] BWD (lane < top_k):                                              |
  |       r = src_rank_of_lane[lane]                                     |
  |       combined_topk_weights[t, lane] =                               |
  |         staging[t*team_size + r][row_bytes + lane*4]  (co-loc pad)   |
  +----------------------------------------------------------------------+
```

**CTA layouts.** `combine_push`: `kCombinePushMaxSlots = 24` warps is the
`__launch_bounds__` upper bound; the actual block is sized down from that by
`combine_push_slots_per_block` to fit the per-warp smem buffer (see
one-tile chunking above), grid = `comm_num_sms`. `combine_reduce`:
`kThreadsPerSlot = 128`, `kSlotsPerBlock = 8`, `kBlockDim = 1024`, grid =
`shuffle_sms` (`NCCL_EP_SHUFFLE_SMS`).

**Determinism.** The intra-rank reduction is bitwise reproducible for a fixed
input. The ballot compact writes valid EM copies to
`smem_flat2em_slot_map[..][my_pos]` at ascending lane order, so the sum order
is fixed to `k = 0..n-1` regardless of load timing. Each thread accumulates
into its own FP32 register for its own tile lanes — no cross-thread shuffle
reduce — and each output int4 is written by a single thread to a slot-unique
`(dst_rank, t, R)` address, so there is no output collision. `combine_reduce`
sums peers in the ascending order that `bl & (bl-1)` peels from
`present_ranks`, again independent of arrival order.

**Anti-camping staging stride.** `comb_stage_stride_bytes(row_bytes,
reserve_prob)` (`ht_ep.cuh:5871`) pads row strides to miss the 768-byte HBM
camping period that concurrent NVLink writes collide on. Forward:
`row_bytes + 256` iff `row_bytes % 768 == 0`, else unchanged. Backward:
`row_bytes + 256` baseline (holds up to 64 prob floats), bumped to `+512` if
the `+256` still lands on the camping period. The same helper is used by
writer, reader, and staging allocation; allocation reserves
`kCombStagePadMaxBytes = 512` per row.

### Intra-LSA head/tail sync

Both kernels bracket the body with matched sync primitives
(`ht_ep.cuh:5424-5461`):

- `lsa_grid_head_gate`: block 0 warp 0 does an `ncclLsaBarrierSession`
  rendezvous, block 0 thread 0 releases `head_sync_flag`, other blocks spin
  on `ld.acquire`. Every rank's source is populated and visible before any
  peer read.
- `lsa_grid_tail_barrier`: `__threadfence_system`, `atomicInc` elects the
  last block which runs `ncclLsaBarrier` and clears the counter and flag.
  All peer accesses done before the source is reused next iteration.

Counter cross-recycling: dispatch's head gate uses the *combine* counter
(idle during dispatch) and its tail barrier uses the *dispatch* counter;
combine swaps them. No extra allocation.

`combine_push` additionally uses the second word of its counter block (the
`combine_head_sync_counter` slot, which pull-push combine does not otherwise
use) as the row-claim pool counter. `lsa_grid_tail_barrier` zeroes it together
with the barrier counter, so it starts at 0 on the next call.

### Buffer model

| Buffer                                 | `kLocalPermute`                                          | `kPullPush`                              | Role change                                        |
| :------------------------------------- | :------------------------------------------------------- | :--------------------------------------- | :------------------------------------------------- |
| `x` (attn input)                       | passed to sender                                         | peer-read source                         | Only peer-accessed dispatch buffer                 |
| `recv_x_em` (dispatch output)          | peer-visible                                             | local                                    | Main ergonomic win                                 |
| `expert_output_token` (IPC scratch)    | sender-push destination                                  | pull source (D2D-staged from `x`)        | Re-role; sized `per_rank_tokens * bytes`           |
| `expert_input_token` (combine staging) | peer-visible                                             | peer-visible push destination            | Row stride padded by up to 512B                    |
| `expert_output_prob` (IPC scratch)     | `max_output_slots * num_local_experts * lsa_ranks * f32` | `per_rank_tokens * MAX_NUM_TOPK * f32`   | Now sparse pull source for topk_weights            |
| `expert_input_prob`                    | allocated                                                | not allocated                            | Weights ride the token row pad instead             |
| `prob_buffer` (per-handle)             | `num_tokens * num_experts * f32`                         | not allocated                            | Wire-row conversion dropped                       |
| `global_routing_map` (bitmap)          | allocated                                                | not allocated                            | Replaced by `global_topk_idx` (uint16)             |
| `sparse_to_dense_map`                  | allocated + memset                                       | not allocated                            | Pull never reads it                                |

`peer_input_ptrs`, `peer_weight_ptrs`, `peer_scale_ptrs`, `peer_staging_ptrs`
ride the launch as inline `__grid_constant__` arrays of length
`kPullMaxLsaRanks = 128` (headroom past NVL72). Marshaled per call, no
persistent host staging.

### Shared-path changes outside the two kernels

- **`uint16` global topk replaces the routing bitmap**
  (`nccl_ep.cc:2011-2020, 3118-3141`). `pack_topk_idx` packs each rank's
  sparse `[max_tokens, num_topk]` topk row into `uint16` (sentinel
  `kTopkIdxInvalid = 0xFFFF`, asserts `num_experts <= 0xFFFF`) and
  AllGathers it as raw bytes via `ncclUint8`. The bitmap kernel, the
  bitmap AllGather, and the `sparse_to_dense_map` allocation + memset are
  all skipped; the scan feeds off `global_topk_idx`. This is the primary
  source of the `UpdateHandle` speedup below.
- **Sparse `topk_weights` end-to-end** (`nccl_ep.cc:2541-2559, 1037-1064`).
  `prob_buffer` is not allocated (`L.sz_prob = 0`); `expert_output_prob`
  is resized to `per_rank_tokens * MAX_NUM_TOPK * f32` (~`num_experts /
  num_topk`x smaller: 32x DS.v3, 51x Nemotron). `sparse_to_dense_prob_kernel`
  and its `cudaMemsetAsync` are skipped; dispatch pulls `[num_topk]` at the
  source position via `srcpos_map`. Backward combine scatters sparse-direct
  into the pushed row's pad, so `dense_to_sparse_prob_combine_kernel` is
  skipped too.
- **Token-indexed map size cap** (`nccl_ep.cc:2574-2611`). The baseline sizes
  `flat2em_slot_map` at `max_recv_tokens * num_topk * 4B`, where
  `max_recv_tokens` already includes the EM-copy fan-out; `kPullPush` caps
  the three token-indexed maps (`flat2em_slot_map`, `recv_slot_to_src`,
  `srcpos_map`) at `max_flat_recv_tokens = min(max_recv_tokens, max_tokens *
  lsa_team_size * rdma_team_size)` — one row per received *token*, not per
  EM copy.

## Impact

Lyris GB200, MNNVL, `comm_num_sms = 32`, shuffle SMs = all. `LP` =
`local_permute_dup`, `LR` = `local_permute_reduce`, `R` = `combine_reduce`.

### Forward, BF16

Nemotron 16N (EP=64, experts=512, topk=10, h=3072, t=8192/rank):

| Kernel                    | Baseline (us) | Pull-push (us) |   Speedup |
| :------------------------ | ------------: | -------------: | --------: |
| UpdateHandle              |           664 |            347 | **1.91x** |
| Dispatch (D)              |           985 |            775 |     1.27x |
| DispatchEpilogue (LP)     |           249 |          fused |         - |
| **D-side total (D+LP)**   |      **1234** |        **775** | **1.59x** |
| CombinePrologue (LR)      |           194 |          fused |         - |
| Combine (C)               |           785 |            822 |     0.96x |
| CombineEpilogue (R)       |             - |             62 |         - |
| **C-side total (LR+C+R)** |       **979** |        **884** | **1.11x** |
| **D+C total (excl. UH)**  |      **2213** |       **1659** | **1.33x** |

DS.v3 EP=32 (experts=256, topk=8, h=7168, t=4096/rank):

| Kernel                    | Baseline (us) | Pull-push (us) |   Speedup |
| :------------------------ | ------------: | -------------: | --------: |
| UpdateHandle              |           218 |            160 |     1.36x |
| Dispatch (D)              |           720 |            685 |     1.05x |
| DispatchEpilogue (LP)     |           190 |          fused |         - |
| **D-side total (D+LP)**   |       **910** |        **685** | **1.33x** |
| CombinePrologue (LR)      |           161 |          fused |         - |
| Combine (C)               |           662 |            687 |     0.96x |
| CombineEpilogue (R)       |             - |             88 |         - |
| **C-side total (LR+C+R)** |       **823** |        **775** | **1.06x** |
| **D+C total (excl. UH)**  |      **1733** |       **1460** | **1.19x** |

### Backward, BF16 (Nemotron 16N)

Same config as forward. `UpdateHandle` reuses the forward routing and is omitted.

| Kernel                  | Baseline (us) | Pull-push (us) | Speedup   |
| :---------------------- | ------------: | -------------: | :-------- |
| Dispatch (D)            |           737 |            767 |     -0.04 |
| DispatchEpilogue (LP)   |           225 |          fused |         - |
| **D-side total (D+LP)** |       **962** |        **767** | **1.25x** |

### Combine row claiming, BF16

GB300 MNNVL, 32 comm SMs, 4096 tokens/rank, BF16 combine. Combine total =
CP+C+CE (us), best of 3, static ranges (`main`) vs row claiming:

| Model   | EP | Static ranges | Row claiming |  Delta |
| :------ | -: | ------------: | -----------: | -----: |
| DS.v3   |  4 |           355 |          331 |  -6.6% |
| DS.v3   |  8 |           541 |          486 | -10.3% |
| DS.v3   | 16 |           659 |          636 |  -3.5% |
| DS.v3   | 32 |           741 |          739 |  -0.3% |
| Qwen3.6 |  4 |           134 |          123 |  -8.1% |
| Qwen3.6 |  8 |           176 |          163 |  -7.3% |
| Qwen3.6 | 16 |           214 |          201 |  -6.1% |
| Qwen3.6 | 32 |           232 |          227 |  -2.1% |

Gains shrink at high EP because the push is NVLink-bound there.

### Dispatch-only, MXFP8 vs BF16

Pull-push stacks with MXFP8 — row and scale are both pulled over NVLink:

| Variant             | Nemotron D+LP (us) | vs BF16 base | DS.v3 D+LP (us) | vs BF16 base |
| :------------------ | -----------------: | -----------: | --------------: | -----------: |
| Baseline BF16       |               1232 |        1.00x |             912 |        1.00x |
| Baseline MXFP8      |                793 |        1.55x |             575 |        1.59x |
| Pull-push BF16      |                784 |        1.57x |             680 |        1.34x |
| **Pull-push MXFP8** |            **482** |    **2.56x** |         **408** |    **2.24x** |

MXFP8 wins compound because the scale row is small enough to load once into
registers (or through the second `PullTmaStage`) and scatter alongside the
token, so the dispatch pipeline stays the same width.

### Dispatch chunk claim, BF16 (MR !378)

DS.v3 (experts=256, topk=8, h=7168, t=4096/rank), `comm_num_sms = 32`, full
mode, pull-push + fused count + unfused sync. Dispatch kernel (us), `main` vs
the dynamic chunk claim, run alternately on the same nodes in one job.

| Cluster | EP | main | chunk claim | delta |
| :------ | -: | ---: | ----------: | ----: |
| Rubin   |  4 | 299.2 | 264.9 | -11.5% |
| Rubin   |  8 | 315.9 | 295.8 |  -6.3% |
| Rubin   | 16 | 379.5 | 369.7 |  -2.6% |
| Rubin   | 32 | 433.8 | 421.1 |  -2.9% |
| GB300   |  4 | 282.6 | 281.1 |  -0.5% |
| GB300   |  8 | 436.4 | 428.4 |  -1.8% |
| GB300   | 16 | 722.1 | 710.0 |  -1.7% |
| GB300   | 32 | 647.3 | 637.6 |  -1.5% |

Combine is unchanged. The six guard lanes (other tokens, hidden and top-k)
improve by 2.5-7.4% on Rubin and 1.0-1.5% on GB300. Gains shrink at EP16/32
and on GB300 because the pull path is NVLink-bound there: the change removes
the tail, not the link-rate limit. The flat (non-fused) dispatch variant is
not changed.

Token sweep, GB300 EP4, h=7168, `Dispatch: total`, `main` vs the default
threshold (median paired delta, 6-8 alternating runs): 768 tokens/rank -0.3%,
1024 +0.3%, 1536 +1.3% (static path), 2048 -1.5%, 3072 -4.9%, 4096 -3.7% (claim
path). Always claiming costs +4% at 1024 tokens/rank and +2-6% at h=3584
between 1024 and 2048, hence the 2048 default.

## Validation

Correctness must be checked with the four supported configurations of
`NCCL_EP_HT_PULL_PUSH=1`: dispatch recipes `NONE` and `QUANT_FWD`, and
push-combine `NONE` forward and backward. `ep_bench.cu` covers forward
directly and the backward pass under `-B / --backward`
(`ep_bench.cu:4295, 4517, 5949-6071`): it reuses the forward routing,
dispatches without scales, combines with per-recv-token `topk_weights`, and
cross-checks `grad_topk_weights[t, k]` against the source weight the forward
pass delivered.

## Open items

- **Hidden ceiling.** The pull-warp count scales down to the device opt-in
  smem cap; only a hidden size whose single staged warp exceeds the cap is
  unsupported and `ncclEpDispatch` fails cleanly (`nccl_ep.cc:4379-4400`).
  No graceful fallback to the local-permute (`kLocalPermute`) path yet — the
  caller must disable `NCCL_EP_HT_PULL_PUSH`.
- **Multi-team.** `num_lsa_teams > 1` on the backlog.
- **`QUANT_FWD` scale-row registers.** `ScaleInt4PerLane` is JIT-sized to
  the actual row; very large rows should chunk through a register-tile loop
  (load tile -> scatter -> advance) to bound register pressure.
- **Sync mechanism.** `ncclLsaBarrierSession` today; TODO
  (`ht_ep.cuh:5422-5423`) to swap for a peer-ptr + atomic-flag sync (as EM
  local-permute uses) and measure.
- **Recipe coverage.** Dispatch: `NONE` and `QUANT_FWD` (`static_assert` at
  `ht_ep.cuh:5557`). Push-combine: `NONE` only, forward and backward.
