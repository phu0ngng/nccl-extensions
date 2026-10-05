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
(`nccl_ep.cc:768-772`). Selected at group creation when `layout ==
EXPERT_MAJOR`, `NCCL_EP_HT_EM_PULL_PUSH` is set, and `num_lsa_teams == 1`
(the last is rejected otherwise, `nccl_ep.cc:1931-1955`). `em_pull_enabled()`
and `em_comb_push_enabled()` both key off the enum, so the two legs move
together.

### Pull dispatch

`dispatch_pull` (`ht_ep.cuh:5497`) uses one warp per recv slot per wave. Each
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

### Push combine

`combine_push` (`ht_ep.cuh:5887`) is one warp per recv slot. The warp
ballot-compacts `flat2em_slot_map[slot,*]` to get the `n <= top_k` valid EM
copies, decodes `(dst_rank, t)` from `recv_slot_to_src[slot]`, and reduces
the `n` rows into an FP32 accumulator tiled by `kHiddenVec int4` lanes. The
tile pipeline software-prefetches copy 0 of the next tile while the current
tile's NVLink stores drain, then `int4`-stores the row to
`peer_staging_ptrs[dst_rank] + (t*team_size + my_lsa_rank) * stride`. In the
backward pass the pushed row carries a co-located pad: after the row store,
`prob_dst[srcpos[k]] = topk_weights_em[k]`.

```text
Per warp (one recv slot per wave, n <= top_k valid EM copies)

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
  | [3] reduce n rows -> FP32 acc, tiled by kHiddenVec int4 lanes        |
  |     software-pipelined: copy 0 prefetched into pf_nxt while the      |
  |     current tile's NVLink stores drain                               |
  +----------------------------+-----------------------------------------+
                               |
                               v
  +----------------------------------------------------------------------+
  | [4] push: int4 store  row_buf -> s_dst   (NVLink write to peer)      |
  +----------------------------+-----------------------------------------+
                               |
                               v  (BWD only)
  +----------------------------------------------------------------------+
  | [5] BWD: prob_dst = s_dst + row_bytes  (co-located pad)              |
  |          for each copy:  prob_dst[srcpos[k]] = topk_weights_em[k]    |
  +----------------------------------------------------------------------+
```

`combine_reduce` (`ht_ep.cuh:6100`) is one attn token per 128-thread sub-warp.
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

**CTA layouts.** `combine_push`: `kCombinePushThreads = 512` (16 warps),
`kSlotsPerBlock = 16`, grid = `comm_num_sms`. `combine_reduce`:
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
reserve_prob)` (`ht_ep.cuh:5838`) pads row strides to miss the 768-byte HBM
camping period that concurrent NVLink writes collide on. Forward:
`row_bytes + 256` iff `row_bytes % 768 == 0`, else unchanged. Backward:
`row_bytes + 256` baseline (holds up to 64 prob floats), bumped to `+512` if
the `+256` still lands on the camping period. The same helper is used by
writer, reader, and staging allocation; allocation reserves
`kCombStagePadMaxBytes = 512` per row.

### Intra-LSA head/tail sync

Both kernels bracket the body with matched sync primitives
(`ht_ep.cuh:5388-5410`):

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

## Validation

Correctness must be checked with the four supported configurations of
`NCCL_EP_HT_EM_PULL_PUSH=1`: dispatch recipes `NONE` and `QUANT_FWD`, and
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
  caller must disable `NCCL_EP_HT_EM_PULL_PUSH`.
- **Multi-team.** `num_lsa_teams > 1` on the backlog.
- **`QUANT_FWD` scale-row registers.** `ScaleInt4PerLane` is JIT-sized to
  the actual row; very large rows should chunk through a register-tile loop
  (load tile -> scatter -> advance) to bound register pressure.
- **Sync mechanism.** `ncclLsaBarrierSession` today; TODO
  (`ht_ep.cuh:5386-5387`) to swap for a peer-ptr + atomic-flag sync (as EM
  local-permute uses) and measure.
- **Recipe coverage.** Dispatch: `NONE` and `QUANT_FWD` (`static_assert` at
  `ht_ep.cuh:5521`). Push-combine: `NONE` only, forward and backward.
