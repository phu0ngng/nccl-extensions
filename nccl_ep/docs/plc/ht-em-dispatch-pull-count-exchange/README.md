# HT EM Pull Dispatch with Count Exchange

## Summary

This is the pull-push specialization of
[HT EM push dispatch with count exchange](../ht-em-dispatch-push-count-exchange/README.md).
It shares the count-row model, fused/unfused terminology, scan fallback,
layout contract, and DROP/TRAP behavior. This document focuses on the
pull-specific metadata requirement and dispatch schedule.

The original high-throughput (HT), expert-major (EM)
[pull-push](../ht-em-pull-push/README.md) path builds its receiver routing maps
in `ncclEpUpdateHandle`:

```text
UpdateHandle: pack top-k -> AllGather top-k -> global scan -> receiver maps
Dispatch:                                                maps -> pull payload
```

Pull-count moves map construction into forward dispatch:

```text
UpdateHandle: pack top-k + build count row
Dispatch: push metadata -> build receiver maps --+-> pull payload
                                                   ^
                                      chunk-level overlap
```

The fused path removes the pre-dispatch AllGather and global scan. It does not
remove token-level routing metadata: each receiver still needs every source's
packed top-k table to know which remote token rows to pull. The optimization
moves that exchange into dispatch, adds compact counts for layout construction,
and overlaps map construction with payload movement.

In fused forward dispatch, two directions operate at the same time:

- **Metadata is pushed.** CTA 0 on each source rank writes its count row and
  packed top-k table into every peer's metadata inbox.
- **Payload is pulled.** Payload warps on each destination rank issue remote
  reads for token, weight, and optional scale rows.

"Pull" therefore describes payload movement, not metadata movement.
Unfused mode AllGathers the count rows earlier and pushes only packed top-k in
dispatch; payload direction is unchanged.

## Why counts and packed top-k are both required

For source rank `s`, UpdateHandle builds

```text
count_row[s] = [ rank_counts[R] | expert_counts[E] ]
```

where `R` is the number of ranks in the LSA team and `E` is the number of
global experts.

- `rank_counts[d]` is the number of source tokens that reach destination rank
  `d`. Multiple expert hits on `d` count as one token.
- `expert_counts[e]` is the number of assignments to global expert `e`.

The counts determine sizes and starting offsets: one FLAT receive range per
source and one padded EM zone per local expert. They do not identify which
source tokens occupy those ranges. The packed top-k table provides that
token-level identity and the original top-k position needed to move the
matching weight.

The shared count-exchange idea leads to different map and payload paths:

|                            | Push count                                                                         | Pull count                                                               |
| -------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Token-level routing        | Remains private to the source                                                      | Packed top-k is pushed to every receiver                                 |
| Maps built during dispatch | Source builds sparse-to-dense (s2d) and sends LERM; local permute builds `flat2em` | Receiver builds `recv_slot_to_src`, `flat2em_slot_map`, and `srcpos_map` |
| Payload movement           | Source pushes one FLAT row per destination rank                                    | Receiver pulls one source row and scatters it directly to local EM slots |
| Local permute              | Required to move FLAT to EM                                                        | Absent                                                                   |

A push sender can keep token-level routing private because it chooses each
payload destination. A pull receiver must learn which remote rows to fetch, so
counts alone are insufficient.

Receiver MAP warps build three handle-owned maps during forward dispatch:

| Map                | Stored relation                                                   | Pull-dispatch use                                     | Combine use                                       |
| ------------------ | ----------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------- |
| `recv_slot_to_src` | `recv slot -> (source rank, source token)`                        | Selects the peer token, weight, and scale rows        | Selects the peer token row that receives output   |
| `flat2em_slot_map` | `(recv slot, compact local hit) -> EM slot`; `-1` means no output | Scatters one pulled row to its local-expert EM slots  | Gathers and reduces the corresponding EM outputs  |
| `srcpos_map`       | `(recv slot, compact local hit) -> original top-k position`       | Selects the matching weight from the source top-k row | Restores backward weights to their top-k position |

Here, `recv slot` is the logical rank-major FLAT slot that joins the three
maps; pull-count does not materialize a FLAT payload buffer. Forward constructs
the maps once, and backward dispatch and combine reuse them. Pull-count builds
no s2d or LERM because it has no sender-pushed FLAT payload or local-permute
stage.

## Scope and modes

Pull-count is selected only for:

```text
HT expert-major layout
&& ht_em_mode == kPullPush
&& one LSA team
&& non-eager mode
&& num_local_experts <= 65536 / MAX_NUM_TOPK
&& NCCL_EP_HT_EM_AG_SCAN_MODE is disabled
```

A pull-push handle that fails this gate uses the scan path. One LSA team may
span physical nodes; a group with multiple LSA teams uses the GIN path and is
outside the current design.

The capability gate and the caller's need for early layout outputs select one
of three paths:

| Mode          | UpdateHandle                                                                  | Forward dispatch                                                             |
| ------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Fused         | Builds a handle-private count row and packed top-k snapshot; no communication | Pushes count row and top-k, then builds layout and maps                      |
| Unfused       | AllGathers count rows and returns the requested layout outputs                | Pushes top-k only, then rebuilds the same layout and maps from cached counts |
| Scan fallback | AllGathers packed top-k and runs the global scan                              | Pulls payload using the prebuilt maps                                        |

Fused and unfused describe count-row delivery only. Both push packed top-k and
build receiver maps during dispatch. `NCCL_EP_HT_EM_AG_SCAN_MODE=1` forces the
scan fallback. The pull path does not read `NCCL_EP_HT_EM_COUNT_UNFUSED`; that
control applies only to local-permute push-count.

## Forward dispatch

The following is a schematic schedule, not a measured timeline. Every rank
executes both source and destination roles. Metadata is handle-owned before
publication, the peer inbox and synchronization state are group-owned, and the
maps remain with the handle for backward and combine.

```text
Source rank s                                      Destination rank d
-------------------------------                    -------------------------------
Handle-owned                                       Group-owned
  count_row[s]                                       peer inbox row [s]
  packed_topk[s]                                     layout bases + ready flag
                                                     rank/expert cursors

CTA 0, all threads:                                metadata producer
  count_row + packed_topk
    == peer stores ===============================> inbox row [s]
  system fence

CTA 0 head rendezvous <======= LSA barrier ======> CTA 0 head rendezvous
other CTAs wait on local head flag                  metadata is now readable

                                                    metadata consumers/producers
                                                    CTA 0 MAP warps:
                                                      count rows -> rank bases,
                                                                    expert bases
                                                      release(layout_ready)
                                                               |
                                                    both MAP warps in each CTA:
                                                      packed top-k chunk
                                                        -> recv_slot_to_src
                                                        -> flat2em_slot_map
                                                        -> srcpos_map
                                                      release(chunk_ready)
                                                               |
                                                               v
Source payload                                      payload warps:
  token[T,H]  ====================================> acquire ready map range
  weight[T,K]        receiver-issued remote reads    pull one source row
  scale[T,S]                                         scatter to local EM slots

                                                    PAD warp:
                                                      zero expert padding rows

source reuse waits <========== tail barrier ====== all peer reads complete
```

The head barrier publishes all source metadata before any receiver consumes
it. `layout_ready` publishes the common rank and expert bases to all CTAs.
Each MAP warp publishes a completed range through `chunk_ready`, allowing
payload warps to start before the remaining map chunks are built. The tail
barrier protects source memory from reuse while peers still read it.

### Map reconstruction

For destination rank `d`, prefix sums over rank and expert counts reserve a
disjoint FLAT range for every source and a padded EM zone for every local
expert. MAP warps walk the packed top-k rows, claim those reserved slots, and
record the source token, EM destinations, and original top-k positions. A
payload warp can then pull the source token once and scatter it to every local
expert hit.

### CTA and warp assignment

The launch uses `comm_num_sms` logical CTAs, multiplied by one or two CTAs per
SM according to the JIT's row-size policy. CUDA assigns those CTAs to physical
SMs.

Each pull-count CTA contains:

- two MAP warps;
- a JIT-selected set of payload warps; and
- one PAD warp.

The JIT permits at most 24 MAP-plus-payload warps, so the largest pull-count CTA
has 2 MAP warps, 22 payload warps, and 1 PAD warp. Shared-memory capacity may
select fewer payload warps.

MAP work is statically distributed. Let `B` be the launched CTA count. Each
unit covers 32 tokens from one source rank:

```text
u     = token_chunk * R + source_rank
owner = u % (2 * B)
CTA   = owner / 2
warp  = owner % 2
lane  = token within the 32-token chunk
```

Source rank is not bound to a CTA. The `(token chunk, source rank)` work space
is flattened first, then assigned by `owner`. For example, with `R=8` and
`B=8`:

```text
u=0..7:   (chunk 0, source rank 0..7) -> CTA 0/MAP 0 ... CTA 3/MAP 1
u=8..15:  (chunk 1, source rank 0..7) -> CTA 4/MAP 0 ... CTA 7/MAP 1
u=16:     (chunk 2, source rank 0)    -> CTA 0/MAP 0  (round-robin wrap)
```

Payload work is dynamic. Payload warps wait for a MAP range, then claim up to
four ready receive slots at a time from a CTA-shared cursor. They do not own a
fixed token and do not process tokens round-robin.

```text
Schematic CTA timeline (not measured)

time ----------------------------------------------------------------------->
MAP 0/1   [CTA 0 layout; others wait] [build chunk][publish range] ...
payload   [wait range] [claim <=4 slots][pull/scatter][claim next] ...
PAD       [wait layout] [zero disjoint padding rows] ----------------------->
```

## Correctness and lifetime

Pull-count preserves the scan path's routing and layout contract, but physical
slot order can differ. MAP warps use atomic cursors, so tokens within a source
range need not appear in scan order. The three maps and payload writes use the
same claimed slots, preserving their relationship.

Only forward dispatch constructs the maps. Backward dispatch reuses the
handle-owned `recv_slot_to_src`, `flat2em_slot_map`, and `srcpos_map` from
forward. Rebuilding them in backward could assign different physical slots and
disconnect the backward payload from the saved forward routing.

The main state lifetimes are:

| State                                             | Owner           | Lifetime                                                    |
| ------------------------------------------------- | --------------- | ----------------------------------------------------------- |
| Count row, packed top-k, optional gathered counts | Handle          | UpdateHandle through forward dispatch                       |
| Receiver maps and expert layout                   | Handle          | Built in forward; reused by backward and combine            |
| Peer inbox, cursors, layout flag, head/tail state | Group           | Reused by dispatches under the group's ordered-use contract |
| Source and destination payload buffers            | Caller or group | Protected by stream order and the dispatch barriers         |

Multiple handles may be prepared ahead because their metadata is private.
Their dispatches must remain ordered: concurrent dispatches on the same group
would alias the group-owned inbox, cursors, and barrier state.

FLAT capacity and padded EM capacity are checked separately. Under TRAP, an
overflow in either space reports the receive-budget error and traps. Under
DROP, out-of-capacity mappings are discarded and published counts are clamped;
the final CTA also removes any phantom active rows caused by different FLAT and
EM truncation boundaries. `recv_total_counter` retains the pre-drop padded
requirement.

Current limits are one LSA team, `MAX_NUM_TOPK == 32`, at most 128 LSA ranks,
`num_experts <= 65535`, and at most 2048 local experts. The local-expert bound
keeps `local_expert * MAX_NUM_TOPK + topk_position` representable in the MAP
warp's `uint16_t` hit encoding. Supported dispatch recipes are `NONE` and
`QUANT_FWD`.

## Cost and observed tradeoff

Let `T` be maximum tokens per source and `K` be top-k.

| Cost per receiver     | Pull scan                               | Pull-count                                                         |
| --------------------- | --------------------------------------- | ------------------------------------------------------------------ |
| Packed top-k metadata | `2 * R * T * K` bytes through AllGather | `2 * R * T * K` bytes through direct peer writes                   |
| Count metadata        | Derived by scan                         | `4 * R * (R + E)` bytes                                            |
| Map construction      | Standalone global scan in UpdateHandle  | Two MAP warps per dispatch CTA                                     |
| Payload start         | After AllGather and scan                | After metadata rendezvous, layout publication, and first map range |
| Map/payload overlap   | None                                    | Chunk-level overlap                                                |

Pull-count trades a shorter UpdateHandle critical path for more dispatch work:
direct metadata stores, count reduction, atomic slot assignment, and two MAP
warps reserved from the CTA.

A workspace run at source SHA
`4e178c2786fae24305990a7e68fd75309d513f28` used 32 ranks across 8 nodes in
one LSA team, BF16, `E=256`, `K=8`, `H=7168`, 32 communication SMs, 10 warmups,
and 50 measured iterations. Times are microseconds; Total is UpdateHandle plus
dispatch.

| Tokens/rank | Mode               | UpdateHandle | Dispatch |     Sum | Sum speedup vs scan |
| ----------: | ------------------ | -----------: | -------: | ------: | ------------------: |
|        4096 | Scan               |       211.24 |   693.24 |  904.48 |               1.00x |
|        4096 | Fused pull-count   |         9.43 |   732.45 |  741.88 |               1.22x |
|        4096 | Unfused pull-count |       158.69 |   732.91 |  891.60 |               1.01x |
|        8192 | Scan               |       304.70 |  1314.79 | 1619.49 |               1.00x |
|        8192 | Fused pull-count   |        17.03 |  1391.67 | 1408.70 |               1.15x |
|        8192 | Unfused pull-count |       136.13 |  1393.57 | 1529.70 |               1.06x |

In this run, fused dispatch was about 6% slower than scan dispatch, while the
UpdateHandle reduction produced a 1.15-1.22x speedup in their sum, equivalent
to 13-18% lower summed time. The result is directional: the workspace record
lacks a committed command, GPU and software inventory, and raw output needed
for PLC sign-off.

## Validation status

`run_tests.sh` configures pull-count and forced pull-scan passes for forward
routing, combine, backward map reuse, multiple live handles, quantized
dispatch, and DROP behavior. This is configured coverage, not a pass record for
this source revision. Historical logs contain successful fused `ep_bench -V`
runs at EP 8, 16, 32, and 64, but do not record the built source SHA.

Remaining sign-off gaps are:

- a pull-specific fused dispatch-time `layout_info` regression;
- the backward phantom-row overflow case in the pull-count test filter;
- TRAP and 2048-local-expert boundary coverage;
- repeated CUDA Graph replay; and
- a reproducible performance record with the exact build, commands, topology,
  software versions, and raw logs.

## Implementation map

| Responsibility                                            | Source                                                                                                    |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Mode selection, allocation, UpdateHandle, dispatch wiring | `nccl_ep/nccl_ep.cc`: `dispatch_pull_count_capable`, `ht_update_handle_pull_count_mode`, `ncclEpDispatch` |
| Packed top-k and count histogram                          | `nccl_ep/device/ht_ep_adapter.cu`: `pack_topk_and_count_row_kernel`                                       |
| Metadata publication, map construction, payload pull      | `nccl_ep/device/ht_ep.cuh`: `dispatch_push_map_publish`, `dispatch_pull_map_warp`, `dispatch_pull`        |
| Launch geometry and shared-memory fitting                 | `nccl_ep/device/jit/ht_dispatch_jit.cuh`: `launch_dispatch_pull`, `pull_dispatch_warps`                   |
| Host/device parameter marshaling                          | `nccl_ep/device/ht_ep_adapter.{cuh,cu}`: `launch_dispatch_pull`                                           |
| Scan fallback                                             | `nccl_ep/device/scan_kernel.cuh`                                                                          |
