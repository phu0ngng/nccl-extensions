# NCCL EP PLC documentation — non-public

This subtree contains internal NCCL EP PLC design material. It is explicitly
excluded from the public filter by `PUBLIC_PATHS.txt`; do not remove that
exclusion without an explicit review.

## PLCs

- [375: Hierarchical dispatch](375-ll-hierarchical-dispatch/README.md) — illustrated design, shared protocol, selection policy, kernel walkthrough, and performance evidence.

- [429: HT dispatch and combine readiness signals](429-ht-signals/README.md) - per-tile dispatch and per-chunk combine flags for expert-major pull-push, and the combine as one programmatic chain.
- [HT scan workspace ownership](ht-scan-workspace.md) — group-shared update-only metadata scan buffers.
- [Attention-FFN Disaggregation (AFD) for Low-Latency EP](attention-ffn-disaggregation/README.md) — partitions one LL communicator into token-owning attention ranks and expert-owning FFN ranks, with role-checked C/Python APIs.
- [271: Low latency LSA FP8 dispatch optimization](271-ll-lsa-fp8-dispatch/README.md) — shared quantization reuse, compact block geometry, vector stores, and capacity fallback.
- [285: LL TMA dispatch for LSA-only cases](285-ll-tma-dispatch/README.md) — warp-0 TMA stage-ring pipeline for the SEND phase, concurrent idle-warp sentinel/recv-count-buffer clear; default for LSA-eligible, unquantized, rank-major dispatch.
- [Fused NVFP4 combine global-scale derivation](nvfp4-combine-scale-fusion/README.md) — reverse TMA amax prepass with SMEM-stage reuse in the shared forward pipeline.
- [Low-latency epoch ticketing](ll-epoch-ticketing/README.md) — graph-safe LL two-bank resource selection.
- [HT EM pull dispatch + push combine](ht-em-pull-push/README.md) — local `recv_x_em`, fused local permute, sparse weights end-to-end.
- [Relative rank slot addressing for the LL dispatch staging buffer](ll-relative-rank-slot/README.md) — spreads per-rank write offsets to avoid receive-side camping.
- [MXFP8 combine](mxfp8-combine/README.md) — FP8 E4M3 + E8M0 block scales fused into the local permute reduce; packed NVLink row, BF16 RDMA hop.
- [MR343: HT EM push combine MXFP8](343-ht-em-combine-push-mxfp8/README.md) — MXFP8 quantize fused into the push combine reduce epilogue, dequantized in the final reduce; follow-up to the local-permute MXFP8 combine.
- [HT EM dispatch push count-exchange](ht-em-dispatch-push-count-exchange/README.md) — count-row exchange replaces per-token bitmap AllGather + global scan; receiver s2d/LERM rebuilt inside dispatch.
- [HT push dispatch tail redistribution (MR !347)](ht-push-dispatch-tail-redistribution/README.md) — coarse and redistributed work schedules, per-unit readiness, and receive-slot prefix reconstruction when multiple CTAs share a metadata chunk.
- [HT EM dispatch pull count-exchange](ht-em-dispatch-pull-count-exchange/README.md) — same count-exchange idea ported to pull-push: CTA 0 pushes count+top-k to peer inboxes, MAP warps rebuild receiver maps locally instead of a scan.
- [MR171: HT compact top-k weights and IDs](mr171-ht-compact-prob-wire/README.md) — automatic compact push dispatch with original-order weights and inline expert IDs, compact FP32 gradients for backward combine; pull-push keeps its existing sparse-direct path.
- [MR289: Gapless source top-k order](mr289-ht-compact-topk-order/README.md) — one-pass compact receiver with source top-k order by default; `NCCL_EP_HT_RECV_TOPK_ORDER=expert-ascend` restores ascending expert order through a JIT specialization.
- [Low-latency dispatch over CFT](ll-cft-dispatch/README.md) — single-warp counted-fabric-write payload path (`NCCL_EP_LL_CFT`); see [CFT API design](#cft-api-design) for the comparison with NCCL.
- [302: Low-latency 1-sided combine with automatic kernel selection](302-ll-combine-1sided/README.md) — dispatch-push/combine-pull protocol, and dispatch-side on-device auto-mode pick with static/dynamic disqualification.
- [MR298: LL CFT dispatch on the LSA rank-major kernel](298-ll-cft-lsa-dispatch/README.md) — staged and zero-copy LSA dispatch with per-source counted/ordinary completion and LSA combine.
- [Counting-warp scan of the top-k table (CFT path)](ll-count-scan/README.md) — pipelined, branch-free per-destination token counting, selected only under `NCCL_EP_LL_CFT`; the store path keeps its scan.
- [MR401: Low-latency Lamport combine](401-ll-combine-lamport/README.md) — opt-in `NCCL_EP_COMBINE_LSA=lamport`: sentinel-poisoned landing region polled by the receiver, own round counter, per-peer round words, poison restored by `ncclEpMaskClean` and at handle growth.

## CFT API design

We aim to keep the local CFT API as close to NCCL’s API as possible, so future
migration requires only small changes. The current NCCL dependency is 2.30.7,
which does not provide this device API, so this MR provides its own implementation in
[`device/cft.cuh`](../../device/cft.cuh).

The comparison below uses NCCL 2.32.1 at commit `c32740385`, dated September 15,
2026. It covers `nccl_device/cft.h`, `impl/cft__funcs.h`, and `impl/cft__types.h`. The LL
protocol is described in [LL CFT dispatch](ll-cft-dispatch/README.md).

### Device API

| Area | Gap / difference from NCCL | Expected migration |
|---|---|---|
| Types and shared memory | Same endpoint ID type, barrier layout, and enum values. | Replace local definitions with NCCL aliases. |
| Transport state | Ours uses a barrier pointer instead of a reference and adds `pendingTxCount`. | Use NCCL’s state with a small wrapper retaining `pendingTxCount`. |
| `put` / `putCounted` | Same fabric instructions and transaction accounting. NCCL also provides optional debug alignment checks. | Forward directly to NCCL. |
| `submit` | Ours additionally updates `pendingTxCount` to track submitted work since the last flush. | Retain `pendingTxCount` accounting while forwarding to NCCL. |
| `flushSmem` | Same operation and synchronization. | Forward directly to NCCL. |
| `flush` | Ours has a timeout and returns success or timeout. NCCL waits indefinitely and exposes fabric-report outputs; its barrier wait mode also differs. | Retain a bounded-flush wrapper using NCCL’s barrier state, and review report handling. This is the main behavior that needs custom code. |
| `waitCounted` | Same timeout API and polling/fence structure. Ours starts timing after the first unsuccessful poll; NCCL starts before the first load. | Use NCCL’s timeout overload. |
| `memFence` | Same fence instructions and release/acquire synchronization. | Forward directly to NCCL. |
| Initialization and lifetime | Explicit initialization rather than NCCL’s constructor/destructor; ours adds a proxy fence after initialization. | Construct the NCCL object once, keep a pointer in the send context, and preserve the required initialization ordering. |

LL dispatch uses `ncclCoopThread`. The extra `pendingTxCount` accounting keeps
outstanding work within the barrier’s transaction limit. A flush timeout is
fatal in the current protocol.

TMA staging and self-rank stores are separate from the fabric API and still
need their own helpers. NCCL’s multicast, reduction, and get operations are not
used by this dispatch path.

### Host API with NCCL 2.32.1

NCCL can replace most of the endpoint management in
[`nccl_ep_cft.cc`](../../nccl_ep_cft.cc):

| Area | Expected migration |
|---|---|
| Receive buffer and endpoints | Allocate with `ncclMemAlloc` and register with `ncclCommWindowRegister`, using `NCCL_WIN_COLL_SYMMETRIC` and `NCCL_WIN_CFT_COUNTED`. NCCL manages the endpoint lifecycle, handle exchange, and peer imports. Deregister the window and free the allocation through NCCL. |
| Device setup | Set `ncclDevCommRequirements::cftCaps` to `NCCL_CFT` when creating the device communicator. |
| Peer addressing | Replace `peer_le_dev` with `ncclGetPeerLeInfo`. Use the returned endpoint ID and full offset, which already includes the window's base offset. |
| Platform checks | Let NCCL check endpoint support. Keep the group-wide `NCCL_EP_LL_CFT` agreement, the JIT compiler check, and diagnostics that identify a failing rank. |

The LL protocol still owns the buffer layout, parity tracking, cumulative byte
counts, and count/flag cleanup. Its zero-copy and rank-masking restrictions also
remain.
