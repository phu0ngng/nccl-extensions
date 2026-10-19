# Attention-FFN Disaggregation (AFD) for Low-Latency EP

Status: Proposed (implementation in the accompanying MR)

Audience: NCCL EP PLC

## Summary

AFD is an MoE serving topology in which attention ranks and expert/FFN ranks are
physically separate groups of GPUs. Motivation, prior art, and the traffic-pattern
change are covered in the upstream RFC and are not repeated here:

- [NVIDIA/nccl#2111 — [RFC] nccl-ep: support Attention-MoE disaggregation](https://github.com/NVIDIA/nccl/issues/2111)

This document records the API and LL-path design chosen to implement that RFC.

**API.** One existing EP group spans all ranks. A V4 config field partitions it:

```c
ncclEpGroupConfig_t::num_attn_ranks   // N_a; 0 = classic homogeneous EP
```

Callers create the group and handle with the normal EP APIs
(`ncclEpCreateGroup` / `ncclEpCreateHandle`, Rank-Major). Optional
`ncclEpAfDispatch` / `ncclEpAfRecv` / `ncclEpAfCombine` / `ncclEpAfReturn`
wrappers validate A- vs F-side role at the call site; `ncclEpDispatch` /
`ncclEpCombine` remain valid if the framework branches itself.

**Runtime.** AFD is supported by the universal LL dispatch/combine kernels only;
the LSA-specialized SIMT/TMA kernels remain unchanged. The universal kernels use
index remapping plus role gating on the existing collective. Routing metadata
rides in the existing LL dispatch header (no sidecar). Transport setup uses the
same collective windows and QPs; role-only receive slabs may be omitted.

The first release is deliberately narrow: LL algorithm, Rank-Major layout,
square partition. See [Scope](#scope-and-limits).

## Core design: partition, do not fork

A single EP group still spans all `nRanks`. Forking a second asymmetric collective
would have meant a second dispatch/combine kernel family, a second transport
setup path, and a second thing to keep correct for every future LL change.
Partitioning instead:

```c
ncclEpGroupConfig_t::num_attn_ranks   // N_a; 0 = classic homogeneous EP
```

```text
A-side: world ranks [0, N_a)        own no experts; send tokens, receive results
F-side: world ranks [N_a, nRanks)   own all experts; receive tokens, send results
N_f = nRanks - N_a
```

The dispatch and combine kernels remain the symmetric all-to-all. AFD reduces to
three index-space edits plus role gating. `num_attn_ranks == 0` makes every edit
an algebraic no-op, so classic EP is byte-for-byte unchanged.

### Uniform local expert count

```text
numLocalExperts = num_experts / N_f
```

This is the single value on *every* rank, including A-ranks that own zero experts.
The alternative — zero local experts on the A-side — divides by zero throughout the
expert/rank arithmetic. Keeping it uniform means A-ranks compute the same
expert->rank mapping the F-ranks do, which is exactly what a sender needs.

`num_experts % N_f == 0` is validated at group creation.

### Two index spaces

The kernels carry both a **world rank** (`[0, nRanks)`, used for addressing peers,
signals, and masks) and an **F-index** (`[0, N_f)`, the rank's position among expert
owners, which is what `expert_id / numLocalExperts` naturally yields). AFD is the
translation between them:

```text
expert_id -> world rank :  expert_id / numLocalExperts + N_a
world rank -> expert base:  (currRank - N_a) * numLocalExperts
```

Everything else — per-rank counters, dedup scans, `rankDone` flags — stays in whichever
space it was already in. Blocks that are entirely F-index-relative (the
`countTokensPerRank_mask` scan and its `rankBegin/rankEnd` window) are left alone;
only the world-indexed sinks they feed gain `+ N_a`. Helpers that compare ranks
*relatively* rather than absolutely (`warpMarkFirstOccurrence`) are AFD-safe
unchanged, since a uniform offset does not change a first-occurrence test.

### Role gating

Four blocks would otherwise deadlock or publish unused traffic, because each waits
on or sends traffic its role never receives:

| Block | Gate | Without the gate |
|---|---|---|
| dispatch send/count publish | `afdRankOwnsTokens` | F-ranks publish zero counts into slots no receiver consumes |
| dispatch recv | `afdRankOwnsExperts` | A-ranks wait for tokens never sent to them |
| combine send | `afdRankOwnsExperts` | A-ranks compute a negative global expert id |
| combine recv | `afdRankOwnsTokens` | F-ranks wait for finish flags never raised |

The dispatch send/count-publish role is burned into the universal JIT variant, so
F-rank kernels compile out that work. The epoch-bank wipe and synchronization remain
unconditional on both roles.

## Routing metadata: no new channel

LL's dispatch message header already carries `token_id` plus one routing entry per
top-k position, because the symmetric path already needs the receiver to
demultiplex into per-expert buffers. The F-side reconstructs expert offsets from
exactly the same header, by exactly the same code — no AFD-specific sidecar.

A compact routing sidecar ahead of the payload remains an open question for the HT
path, whose preprocessing phase does compute a routing map the A-side would have to
ship. That is out of scope here.

## Transport

Because AFD is the same collective over the same group, every rank still participates
in transport setup and creates the required signals and GIN contexts. Role-only receive
slabs need not be allocated on the opposite role. For zero-copy dispatch, all ranks
participate in the window registration sequence, but A-ranks may register a minimal
placeholder rather than the full F-side receive slab. No QP or credit-path fork is
required. Validated inter-node in the source MR (see [Validation](#validation)).

## Public API

### Lifecycle

```c
ncclEpGroupConfig_t config = NCCL_EP_GROUP_CONFIG_INIT;
config.algorithm = NCCL_EP_ALGO_LOW_LATENCY;
config.num_experts = /* ... */;
config.num_attn_ranks = N_a;   // ranks [0, N_a) A-side; [N_a, nRanks) F-side
// ... other group config fields as for classic LL EP ...

ncclEpGroup_t group;
ncclEpCreateGroup(&group, nccl_comm, &config);

// A-ranks: topk_idx [num_tokens, top_k] over [0, num_experts)
// F-ranks: topk_idx [0, top_k] (no source tokens)
ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, group, NCCL_EP_LAYOUT_RANK_MAJOR,
                   topk_idx, /*layout_info=*/NULL, /*config=*/NULL, stream);

// ... dispatch / combine (see below) ...

ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(group);
```

`num_attn_ranks` is a prefix of the communicator order: there is no separate
create-AF API and no explicit attn/expert rank lists. An arbitrary rank set would
force an indirection table on every expert→rank mapping on the critical path, for
no expressiveness the caller cannot get by ordering its communicator.

Constraints enforced at group creation:

- `num_attn_ranks` in `[0, nRanks)`; `0` is classic homogeneous EP
- `num_attn_ranks > 0` requires the LL algorithm
- `num_attn_ranks == nRanks - num_attn_ranks` (square A:F partition)
- `num_experts > nRanks`
- `num_experts` divisible by `nRanks - num_attn_ranks`

Rank-Major layout is enforced when the handle is created or initialized.

At collective call time, AFD supports only `NCCL_EP_DISP_QUANT_NONE` and
`NCCL_EP_COMB_QUANT_NONE`; quantized AFD is not supported.

### Collectives

Every rank still participates in each collective call. Optional role-named
wrappers validate the caller's side and forward to the existing entry points:

| Role | API | Forwards to |
|---|---|---|
| A-side (rank `< N_a`) | `ncclEpAfDispatch` | `ncclEpDispatch` |
| A-side | `ncclEpAfCombine` | `ncclEpCombine` |
| F-side (rank `≥ N_a`) | `ncclEpAfRecv` | `ncclEpDispatch` |
| F-side | `ncclEpAfReturn` | `ncclEpCombine` |

The wrappers exist for call-site validation, not for distinct collective behavior.
Frameworks that prefer to branch themselves can call `ncclEpDispatch` /
`ncclEpCombine` directly with `num_attn_ranks` set. The Python facade exposes the same role checks as
`Handle.af_dispatch`, `Handle.af_recv`, `Handle.af_combine`, and
`Handle.af_return`; the generated low-level bindings use the corresponding
`af_dispatch`, `af_recv`, `af_combine`, and `af_return` names.

Tensor I/O matches classic Rank-Major LL EP: A-side supplies tokens and receives
combined results; F-side receives into a caller-allocated Rank-Major buffer and
returns expert outputs. A-ranks omit the dispatch receive tensors
(`outputs->tokens`, `outputs->topk_weights`, `outputs->topk_idx`, and
`src_rank_counters`) and omit combine `inputs->tokens`; those full Rank-Major
slabs are allocated only on F-ranks. Conversely, F-ranks omit dispatch
`inputs->topk_weights`. On F-ranks, `src_rank_counters` retains its `[nRanks]`
shape, but AFD writes only the active attention-source prefix
`[0, num_attn_ranks)`; the remaining entries are unspecified and must not be
consumed.

The exception is LL dispatch zero-copy, which retains the classic `AUTO`
contract that window attachment is consistent across ranks. Every rank enters
the same collective `ncclCommWindowRegister` call and attaches its handle as
`outputs->tokens`. F-ranks register the full receive slab; A-ranks use a minimal
placeholder allocation. NCCL supports asymmetric registration sizes, and no
dispatch payload targets the A-rank placeholder. With no attached windows, AFD
uses the ordinary staged path. A mixed windowed/non-windowed call is unsupported,
as it already is for classic LL EP.

## ABI

`num_attn_ranks` is appended as a **V4** field of `ncclEpGroupConfig_t`, with
`padding_v4` consuming the tail, and `NCCL_EP_API_VERSION` bumped to 4. V3 is
already occupied by the independent dispatch/combine SM-budget fields.

It deliberately does *not* claim the reserved `padding_v2`. That field is pinned as
a frozen boundary marker by `NCCL_EP_GROUP_CONFIG_V2_LAST_FIELD` and by an
`offsetof == 108` assertion in `test_public_struct_abi.cu`; repurposing it would
silently redefine what a V2 size means. Appending instead lets V1/V2 callers decode
`num_attn_ranks` as 0 — exactly the classic-EP default — through the existing
size-guarded `epDecodeStruct` path, with no version-specific handling anywhere.
`ncclEpCreateGroup` validates and decodes the config through that same path before
reading `num_attn_ranks`, so a short struct from an older caller cannot be read as
garbage.

## Scope and limits

Supported in this design:

- LL algorithm only. `num_attn_ranks > 0` with HT is rejected at group creation.
- Rank-Major layout only, enforced by `ncclEpCreateHandle` / `ncclEpInitHandle`
  when `num_attn_ranks > 0`.
- **Square partition**: `N_a == N_f`.
- Unquantized dispatch and combine recipes only. Quantized AFD is a follow-up;
  callers must use `NCCL_EP_DISP_QUANT_NONE` and `NCCL_EP_COMB_QUANT_NONE`.

The square-partition limit is enforced at group creation. In dispatch-recv and
combine-send, the peer rank is derived as `responsibleExpertIdx / numLocalExperts`,
which enumerates `[0, N_f)` — a *source* A-rank in dispatch-recv, a *destination*
A-rank in combine-send. That identification is only valid when `N_a == N_f`. A general
`N_a:N_f` ratio needs an inner loop fanning each expert group out over its share of
source ranks; that is a kernel change, not a parameter change.

Follow-ups, in expected order:

1. General `N_a:N_f` ratios (dispatch-recv / combine-send source-rank fan-out).
2. Expert-Major layout.
3. HT path, including a routing sidecar if preprocessing cannot reuse the LL header.
4. Add CFT support; AFD currently rejects the optional CFT transport.

## Implementation notes

`numAttnRanks` is threaded through `DispatchParams` / `CombineParams` and the LL
kernel arg structs as a runtime value. The two rank-ownership booleans derived from it
are burned into the JIT variant so role-inapplicable work compiles out; the numeric
partition itself does not multiply variants for every possible rank count.

Kernel selection forces AFD calls onto the universal LL implementation. The
LSA-specialized dispatch and combine entry points neither receive nor interpret AFD
state, which keeps their existing fast paths and ABI unchanged.

## Validation

The design was validated on EOS H100 in the source MR
([nccl/nccl!3078](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/3078)):
1-node and 2-node inter-node RDMA, 2 A-ranks + 2 F-ranks, 8 experts, `top_k=1`,
hidden 2048 — exact bit-for-bit round-trip.

`af_test.cu` is the corresponding in-tree test: A-ranks dispatch, F-ranks validate
payload and routing metadata, run an identity "expert" (echo `recv_x`), and return;
A-ranks validate the combined contribution from every distinct destination F-rank.
Each iteration uses a different sentinel so the test detects stale epoch-bank reads.
CI covers `top_k == 1`, `top_k == 8` including same-F-rank deduplication, and staged
plus asymmetric-window `AUTO` zero-copy dispatch.

The LL path in this repository has been restructured since the source MR (JIT
kernels, quantization recipes), so the kernel edits were re-applied by hand rather
than merged. **The port therefore needs its own inter-node run before it lands**;
building and the ABI assertions are not sufficient evidence for a kernel index-space
change.

Regression coverage for classic EP is the existing `ep_test` suite: every AFD edit is
guarded by `num_attn_ranks == 0` reducing to identity, so an unchanged `ep_test` is
the primary evidence that the default path is untouched.

## Risks

**Non-square partitions are unsupported.** Group creation rejects `N_a != N_f`.
Supporting general ratios requires the dispatch-recv/combine-send fan-out described
under [Scope and limits](#scope-and-limits).

**Index-space drift.** Future LL changes that introduce new expert<->rank arithmetic
will be written in whichever space is locally convenient and can miss the `+ N_a`
translation. The failure mode is a wrong peer under AFD and nothing at all under
classic EP, so `ep_test` will not catch it. Mitigation: keep the world-rank / F-index
distinction named and commented at each conversion site, and keep the multi-rank
`af_test` cases in CI.

**Optional role wrappers.** Four wrappers that forward to two existing entry points
are a small maintenance cost if AFD does not see uptake. Mitigation: they are pure
wrappers with no state, so they can be deprecated without touching the kernels.
