# Attention-FFN Disaggregation (AFD) for Low-Latency EP

Status: Proposed (implementation in the accompanying MR)

Audience: NCCL EP PLC

## Summary

AFD is an MoE serving topology in which attention ranks and expert/FFN ranks are
physically separate groups of GPUs. Motivation, prior art, and the traffic-pattern
change are covered in the upstream RFC and are not repeated here:

- [NVIDIA/nccl#2111 — [RFC] nccl-ep: support Attention-MoE disaggregation](https://github.com/NVIDIA/nccl/issues/2111)

This document records the design actually chosen to implement that RFC in the LL
path, and in particular the three places where the implementation diverges from
the RFC's sketch:

1. AFD is expressed as a **rank-index reinterpretation of the existing symmetric
   all-to-all**, not a new asymmetric collective.
2. Routing metadata needs **no new channel**; the existing LL dispatch header
   already carries it (RFC "Option A", obtained for free).
3. The public API is a set of **role-asserting wrappers** over the existing
   entry points, not a distinct AFD communicator type.

The first release is deliberately narrow: LL algorithm, Rank-Major layout,
square partition. See [Scope](#scope-and-limits).

## Core design: partition, do not fork

The RFC frames AFD as two disjoint groups joined by a new collective. Implementing
it that way would have meant a second dispatch/combine kernel family, a second
transport setup path, and a second thing to keep correct for every future LL
change.

Instead, a single EP group still spans all `nRanks`, and one new config field
partitions it:

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

Three blocks would otherwise deadlock, because each waits on traffic its role
never receives:

| Block | Gate | Without the gate |
|---|---|---|
| dispatch recv | `afdRankOwnsExperts` | A-ranks wait for tokens never sent to them |
| combine send | `afdRankOwnsExperts` | A-ranks compute a negative global expert id |
| combine recv | `afdRankOwnsTokens` | F-ranks wait for finish flags never raised |

The dispatch **send** phase needs no gate: an F-rank owns no source tokens, so its
handle reports `num_tokens == 0` and the send loop is empty. This falls out of the
existing token-count plumbing rather than being special-cased.

## Routing metadata: no new channel

The RFC lists this as an open question with two options — inline in the payload
header (A), or a compact sidecar sent ahead of the payload (B) — and leans toward B
as the better fit for the preprocessing/dispatch two-phase model.

For the LL path the question does not arise. LL's dispatch message header already
carries `token_id` plus one routing entry per top-k position, because the symmetric
path already needs the receiver to demultiplex into per-expert buffers. The F-side
reconstructs expert offsets from exactly the same header, by exactly the same code.
This is Option A, at zero cost.

Option B remains the open question for the HT path, whose preprocessing phase does
compute a routing map the A-side would have to ship. That is out of scope here.

## Transport

The RFC anticipates that pure-sender / pure-receiver roles "may require changes to
how nccl-ep sets up RDMA QPs and manages credit/flow control."

They do not, for this design. Because AFD is the same collective over the same
group, every rank still creates the same windows, signals, and GIN contexts. An
A-rank simply never has traffic on the dispatch-recv direction. Validated
inter-node in the source MR (see [Validation](#validation)).

## Public API

The RFC sketches `ncclEpCommCreateAF(comm, numAttnRanks, attnRanks, numExpertRanks,
expertRanks, ...)` with explicit rank lists. This design replaces the lists with the
`num_attn_ranks` prefix rule, which is what the kernel index arithmetic requires
anyway — an arbitrary rank set would force an indirection table on every
expert->rank mapping, on the critical path, for no expressiveness the caller cannot
get by ordering its communicator.

The AFD entry points are thin role-asserting wrappers over the existing tensor-based
API. `ncclEpAfComm_t` / `ncclEpAfHandle_t` are opaque aliases of `ncclEpGroup_t` /
`ncclEpHandle_t`:

| A-side | F-side | Forwards to |
|---|---|---|
| `ncclEpAfDispatch` | `ncclEpAfRecv` | `ncclEpDispatch` |
| `ncclEpAfCombine` | `ncclEpAfReturn` | `ncclEpCombine` |

plus `ncclEpAfCommCreate` / `ncclEpAfCommDestroy` and `ncclEpAfCreateHandle` /
`ncclEpAfDestroyHandle`.

The wrappers exist for the call-site assertion, not for behavior: they name the
caller's role and fail loudly when a rank calls the wrong side. Callers that prefer
to branch themselves can keep using `ncclEpDispatch` / `ncclEpCombine` directly with
`num_attn_ranks` set.

## ABI

`num_attn_ranks` is appended as a **V3** field of `ncclEpGroupConfig_t`, with
`padding_v3` consuming the tail, and `NCCL_EP_API_VERSION` bumped to 3.

It deliberately does *not* claim the reserved `padding_v2`. That field is pinned as
a frozen boundary marker by `NCCL_EP_GROUP_CONFIG_V2_LAST_FIELD` and by an
`offsetof == 108` assertion in `test_public_struct_abi.cu`; repurposing it would
silently redefine what a V2 size means. Appending instead lets V1/V2 callers decode
`num_attn_ranks` as 0 — exactly the classic-EP default — through the existing
size-guarded `epDecodeStruct` path, with no version-specific handling anywhere.

`ncclEpAfCommCreate` validates and decodes the config through that same path before
reading `num_attn_ranks`, so a short struct from an older caller cannot be read as
garbage.

## Scope and limits

Supported in this design:

- LL algorithm only. `num_attn_ranks > 0` with HT is rejected at group creation.
- Rank-Major layout only, enforced by `ncclEpAfCreateHandle`.
- **Square partition**: `N_a == N_f`.

The square-partition limit is the one worth being explicit about, because nothing
rejects a non-square config today beyond the shared assertions. In dispatch-recv and
combine-send, the peer rank is derived as `responsibleExpertIdx / numLocalExperts`,
which enumerates `[0, N_f)` — a *source* A-rank in dispatch-recv, a *destination*
A-rank in combine-send. That identification is only valid when `N_a == N_f`. A general
`N_a:N_f` ratio needs an inner loop fanning each expert group out over its share of
source ranks; that is a kernel change, not a parameter change.

Follow-ups, in expected order:

1. General `N_a:N_f` ratios (dispatch-recv / combine-send source-rank fan-out).
2. Expert-Major layout.
3. HT path, including the RFC's Option B routing sidecar.
4. Reject non-square partitions explicitly until (1) lands.

## Implementation notes

`numAttnRanks` is threaded through `DispatchParams` / `CombineParams`, the LL kernel
arg structs, and both JIT source emitters as a **runtime** argument, not a template
parameter. This is intentional: as a template parameter it would widen the JIT cache
key and compile a second variant of every hidden size / recipe / layout / index-width
combination for a value that costs one register.

## Validation

The design was validated on EOS H100 in the source MR
([nccl/nccl!3078](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/3078)):
1-node and 2-node inter-node RDMA, 2 A-ranks + 2 F-ranks, 8 experts, `top_k=1`,
hidden 2048 — exact bit-for-bit round-trip.

`af_test.cu` is the corresponding in-tree test: A-ranks dispatch, F-ranks run an
identity "expert" (echo `recv_x`) and return, A-ranks combine and compare. With
`top_k == 1` and unit weights the combine is a pure identity, so any deviation from
a bit-exact round-trip is a bug.

The LL path in this repository has been restructured since the source MR (JIT
kernels, quantization recipes), so the kernel edits were re-applied by hand rather
than merged. **The port therefore needs its own inter-node run before it lands**;
building and the ABI assertions are not sufficient evidence for a kernel index-space
change.

Regression coverage for classic EP is the existing `ep_test` suite: every AFD edit is
guarded by `num_attn_ranks == 0` reducing to identity, so an unchanged `ep_test` is
the primary evidence that the default path is untouched.

## Risks

**Silent non-square misconfiguration.** A caller setting `N_a != N_f` gets no error
today, only wrong results. Mitigation: add the explicit rejection (follow-up 4)
before advertising AFD beyond early adopters.

**Index-space drift.** Future LL changes that introduce new expert<->rank arithmetic
will be written in whichever space is locally convenient and can miss the `+ N_a`
translation. The failure mode is a wrong peer under AFD and nothing at all under
classic EP, so `ep_test` will not catch it. Mitigation: keep the world-rank / F-index
distinction named and commented at each conversion site, and run `af_test` in CI once
a multi-rank AFD job is available.

**API surface for a narrow feature.** Eight new entry points that forward to two
existing ones is a real maintenance cost if AFD does not see uptake. Mitigation: they
are pure wrappers with no state, so they can be deprecated without touching the
kernels.
