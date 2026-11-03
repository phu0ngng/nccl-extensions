# Low-Latency Lamport Combine (MR !401)

`NCCL_EP_COMBINE_LSA=lamport` selects an opt-in LL combine kernel
(`LlCombineKernel::k2SidedLamportRmLsa`) for the LSA (NVLink) path. It keeps the
send phase of the 2-sided LSA combine and replaces the finish-flag handshake and
the CTA barrier before the receive with a Lamport-style receive: the receiver
polls the payload itself. It is never selected automatically. This PLC records the
protocol, the design decisions and their consequences.

## The idea

A Lamport-style transfer needs no readiness flag, because the data is its own
flag:

1. The landing buffer is **poisoned** with a sentinel (`0xFF` bytes, i.e. the
   32-bit word `0xFFFFFFFF`).
2. The sender writes the payload straight into it. A word that is not the
   sentinel has arrived.
3. The receiver polls its own data until no word of what it reads is the
   sentinel, consumes it, and **re-arms** it (writes the sentinel back), so the
   buffer is poisoned again for the next use.
4. For this to be correct the payload must never contain the sentinel: the sender
   rewrites every payload word equal to `0xFFFFFFFF` to the canonical NaN of the
   token width before the store: `0x7FFF7FFF` for BF16 and FP16 (two 16-bit NaNs,
   `0x7FFF` each) and `0x7FFFFFFF` for FP32.

In the 2-sided kernel every rank finishes its sends, the last CTA publishes a
finish flag to every peer, and the receive starts only after all flags were
seen. With Lamport a warp starts consuming as soon as its own sends are done and
a token's receive overlaps with other warps' and ranks' sends.

## The implementation in this patch set

### Landing region

The region sits at the end of the group's RDMA buffer, behind every handle layout:

```text
|<----------- laid out per handle, handles overlap ----------->|<-------- Lamport region (per group) ------->|
| dispatch payload | counts/flags, banks 0/1 | combine slots .. | round words | segment 0     | segment 1     |
                                                                | 1 int/peer  | [token][j]    | [token][j]    |
                                                                |             | slot_bytes    | sentinel      |
                                                                |             | each          | until written |
```

A segment holds, for every token of the batch, `top-k` slots of one token row
each (`slot_bytes`), addressed `(token * top-k + j) * slot_bytes`. A slot is the
partial result of one rank for that token.

### Send side

The send phase of the 2-sided LSA kernel, plus the sentinel encoding and the gate.

- Each warp group owns one expert channel, so one destination rank (the rank
  that sent the token in dispatch). Its warps walk the tokens this rank received
  from that rank, one send per token (rank-major: one row per token and rank).
- A warp TMA-loads the token's partial result in 2 KB stages (three-stage ring,
  one stage prefetched) into shared memory and bulk-stores each stage straight
  into the owner's slot `(token, top-k position)` of segment `round & 1`, over
  NVLink.
- In the shadow of the first stage load the warp group checks the round gate once
  (see below). Between a stage's load and its store the warp rewrites payload
  words equal to `0xFFFFFFFF` to the canonical NaN and fences the proxy.
- After the sends the rank publishes `round + 1` into its word on every active
  peer with one relaxed system-scope global store per peer (`st.relaxed.sys.global`),
  from the highest CTAs (which own no receive vectors at small batches), and goes
  straight to the receive: no finish flag, no CTA barrier.

### Receive side

Every combined token is a row of `H` elements, split into 16-byte vectors (int4):
`V = H * elem_bytes / 16` per row (BF16, hidden 7168: 896 vectors of 8 elements).
The work is the flat list of `(token, vector)` pairs. Thread `g` takes pair `g`,
then `g + grid size`, and so on (grid-stride). A warp is 32 consecutive pairs.
The whole-tile hidden-size gate guarantees `V % 32 == 0`, so **a warp always
stays inside one token**, and the kernel asserts it.

A small example: hidden 512 in BF16 (the smallest eligible row, 1 KB = 64
vectors), 3 tokens, 2 CTAs of 2 warps, top-k 6 with six distinct ranks:

```text
(token, vector) pairs g = token * 64 + vector;  grid = 2 CTAs x 64 threads = 128;  192 pairs -> two passes

        pass 1 (g = 0..127)                        pass 2 (g = 128..191)
W0  CTA0 warp0   token 0, vectors  0..31           token 2, vectors  0..31
W1  CTA0 warp1   token 0, vectors 32..63           token 2, vectors 32..63
W2  CTA1 warp0   token 1, vectors  0..31           (idle)
W3  CTA1 warp1   token 1, vectors 32..63           (idle)       lane L of a warp -> vector (32 * half) + L
```

All lanes of a warp share the token, so the warp decodes its top-k list together,
once per pass of the grid-stride loop: lane `i` loads top-k entry `i`, `__match_any_sync` keeps the first
position of each destination rank (a rank sends one row per token), and a ballot
gives the warp-uniform mask of **live slots** (valid, not masked).

**A thread owns one vector position of its token across all live slots.** The
landing region holds one full row per live slot, so the thread reads the same
vector number from each of them:

```text
landing segment, token 1   (top-k 6: one row of 64 vectors per contributing rank)
                v0   v1   v2   v3   v4   v5   v6  ...  v63
slot 0 (rank A) [  ] [  ] [  ] [  ] [  ] [##] [  ] ...  [  ]
slot 1 (rank B) [  ] [  ] [  ] [  ] [  ] [##] [  ] ...  [  ]      thread (token 1, vector 5)
slot 2 (rank C) [  ] [  ] [  ] [  ] [  ] [##] [  ] ...  [  ]      = lane 5 of W2:
slot 3 (rank D) [  ] [  ] [  ] [  ] [  ] [##] [  ] ...  [  ]      reads column v5 of every live slot
slot 4 (rank E) [  ] [  ] [  ] [  ] [  ] [##] [  ] ...  [  ]
slot 5 (rank F) [  ] [  ] [  ] [  ] [  ] [##] [  ] ...  [  ]      neighbours (v4, v6, ...) belong to other lanes
```

```text
thread (token 1, vector 5):
  group 1  slots 0,1,2,3   issue the four 16 B loads together, spin until no word of them is 0xFFFFFFFF
                           re-arm the four vectors (write the sentinel); acc += slot0 + slot1 + slot2 + slot3
  group 2  slots 4,5       the same for the remaining slots;             acc += slot4 + slot5
  out[token 1][vector 5] = acc      one 16 B store; fp32 accumulation in top-k order, as in the 2-sided receive
```

- **Several slots are polled at once** (up to 4), so their global-memory load
  latencies overlap. The polls are local loads from this rank's own segment (the
  peers' NVLink stores land there); four independent loads in flight cost about
  one load latency per polling pass, not four. A vector counts as arrived only
  when none of its four 32-bit words is the sentinel: the receiver tests whole
  words, so the sender's per-word rewrite is sufficient (the same convention as
  flashinfer's Lamport MoE combine).
- **No synchronization between threads or warps**: a thread depends only on its
  own vectors, so a warp whose data is already there finishes at once, and a
  token's receive overlaps with other warps' and other ranks' sends.
- A slot still pending after the group timeout masks its peer, raises the error
  flag (one warning per CTA, a trap without a rank mask) and is left out of the
  sum. Unlike the 2-sided receive there is no agreement on which slots were left
  out: different column ranges of one token may include the late peer's
  contribution or not. After a timeout, treat the output rows of tokens routed to
  the timed-out rank as invalid.
- The last CTA to finish its receive advances the round counter.

## Design implications

**Separate epochs.** Dispatch and the non-Lamport combine kernels rely on the
signalling banks being reset before an operation starts: every launch wipes the
bank the next launch will use, and the all-peer handshake of the existing
kernels orders that wipe before any peer's next write. The Lamport combine has
no handshake, so it cannot wipe a bank safely. It therefore neither reads nor
advances the LL epoch, uses no signalling bank and no payload slot, and keeps
its own device-resident counter (`LamportEpochState::round`, whose parity selects
the segment). To the LL epoch a run of Lamport combines is a run of dispatches
with nothing in between, which dispatch already handles by wiping its next bank
itself; dispatch is unchanged. The one thing the Lamport combine shares with the LL epoch is its
`send_in_flight` mark, which pairs a send-only launch with the launch that completes it: one staged call per
group, whichever kernel runs it, so a second staged call (Lamport or not) before `ncclEpComplete` is caught like for
the other kernels. Without it, two staged Lamport combines would use the same segment, because the round advances
only in the receive. A call that falls back to a non-Lamport combine kernel
advances the LL epoch as usual, so Lamport and fallback combines interleave
freely.

**Separate Lamport buffer space.** All regions live in the group's RDMA buffer.
The offsets of the regions used by dispatch and the non-Lamport combine kernels
come from the layout of the handle in use, and every handle's layout starts at
offset 0, so handles reuse the same bytes. Their contents matter only within one
dispatch and combine: nothing in them has to survive from one call to the next.
The Lamport region has state that must survive: the poison and the round words.
Using a regular region would mean re-poisoning it before every combine, with a
synchronization the kernel exists to remove, and keeping that invariant against
every non-Lamport kernel that writes there. Its offset therefore cannot come
from a handle's layout. It is placed at the end of the buffer, behind the
largest layout any handle has needed, so no handle layout can reach it. It holds
two segments (double buffering) of `num_max_dispatch_tokens_per_rank × top-k`
combine slots, with the segment stride sized for the largest top-k of any
rank-major handle, so handles of different top-k share it.

**Relaxed, Lamport-only synchronization (the round gate).** The two Lamport
segments alternate by round, so a segment is reused every second combine and
nobody may write round `r + 2` into it before its owner R has re-armed round
`r`. Whether a writer P is held back without help depends on the traffic.

- *Dense:* R also sends P partial sums in round `r + 1`. P cannot finish round
  `r + 1` without them, and R sends them only after finishing round `r` (stream
  order), so P cannot reach round `r + 2` before R's segment is re-armed.
- *Sparse:* P sends R partial sums in every round but R sends P nothing, which is
  common at a small batch and a large EP size, where who sends to whom is a sparse,
  directed graph. P waits for nothing from R. With R late and no dispatch in
  between, P overwrites round-`r` data that R has not read:

```text
R (late):  ... busy ...   | Combine 1: consume seg 0 | Combine 2: seg 1
P (fast):  C1 -> R.seg 0  | C2 -> R.seg 1            | C3 -> R.seg 0   <- R's round-1 data, still unread
```

Every rank therefore keeps one word per peer, the number of rounds that peer has
started. After its sends, a rank publishes `round + 1` into its word on every
active peer using relaxed global stores (`st.relaxed.sys.global`, one per peer;
system scope because the word is in another GPU's memory). The store is relaxed
on purpose: the word only witnesses that the rank is in this round, so its
previous round has retired, not that any data has landed. Before its first store
into R's segment in round `r`, a sending warp group of P requires
`word[R] >= r`: R has started round `r - 1`, so its round `r - 2`, the previous
user of that segment, has retired. The check is lazy, per destination, by the
sending warps only, and overlaps the first stage load. The comparison is a
difference modulo 2^32, so the 32-bit round counter may wrap. A writer is never
more than one round ahead of the peer it writes to, and a rank far ahead of a
peer it does not write to is not delayed. A peer whose word does not arrive
within the group timeout is masked like a missing peer, and the sending warp
stores nothing into its segment for the rest of the kernel.

The gate cannot deadlock: a round's waits depend only on publishes of the
previous round, and every rank publishes unconditionally once its sends are done,
so by induction on rounds every wait is eventually satisfied. The word is read
only by the sending warps, after the first stage load is issued, so its latency
overlaps the load (one warp-coalesced load per sending warp). Reading it from
every thread at kernel entry made the single line a hot spot that cost about a
microsecond.

The same example with the gate, three combines on one group (round counter 0, 1,
2; the segment is the round's parity):

```text
R (late):  ... busy ...   | Combine 1: consume seg 0 | Combine 2: seg 1  | Combine 3: seg 0
P (fast):  C1 -> R.seg 0  | C2 -> R.seg 1            | C3: before the first store into R.seg 0, wait word[R] >= 2
                                                       (R publishes 2 after its sends in Combine 2, so its
                                                        Combine 1 has retired and seg 0 is re-armed)
```

**Rank-mask clearing poisons the region.** `ncclEpMaskClean` is the recovery path
after a rank was masked, so it cannot depend on that rank. The poison, the
round-word reset and the counter reset run inside the clean kernel, in its zero
step between its two mask-aware barriers (which skip a missing rank after the
timeout). A communicator-wide barrier would hang the survivors.
A masked rank is re-admitted only through `ncclEpMaskClean`: `ncclEpMaskUpdate`
alone leaves a late peer's payload and the old round state in the region, and the
next combine would consume them as data.

**Handle creation and top-k growth.** A handle with a larger top-k than the
region's capacity grows it. In an auto-sized group that reallocates the RDMA
buffer; with a user-specified size the region is created at the tail in place if
it fits, and the handle is rejected otherwise. Either way the region is
re-created: both segments poisoned, round words and counter zeroed, on every rank.
The API allows D(H1), create H2, C(H1), so a peer's first store can follow this
rank's handle creation with nothing in between. The install therefore fences
before and after the poison with a device synchronisation and a communicator-wide
barrier: the first drains every earlier store and round-word publish into the old
region, the second finishes every rank's poison before any rank's next combine. A
reallocation needs neither fence: its barrier is the first, and the poison is
written before the window registration, which keeps every peer out of the new
buffer until all ranks have poisoned theirs.
Handle creation is collective with no operation in flight, also with a user-specified
buffer size, where it is otherwise purely local; passing the largest top-k to the group
(`ncclEpGroupConfig_t::num_topk`) avoids it.

## Limitations

- **Hidden size.** The same whole-tile restriction as every LSA combine kernel:
  BF16/FP16 hidden a multiple of 512 elements (1 KB rows), FP32 a multiple of 256,
  and 1024 elements for tokens wider than 15872 BF16. Other sizes fall back to the
  default combine. The restriction comes from the shared send loop (fixed
  full-stage loads and stores) and from the shared host eligibility check, not from
  the Lamport receive, which needs only 512 B rows. It can be relaxed by a partial
  last stage in the sender plus logical padding of the receive; not done here.
- **Scope of the 2-sided LSA kernel.** Rank-major layout, unquantized tokens (no
  LogFMT), NVLink-only groups, no attention-FFN split, no Counted Fabric Transport (CFT) dispatch.
  Opt-in only. A call that is not LSA-eligible falls back to the general kernel, like for the
  other forced LSA kernels (`nccl_ep_env.h`), and `NCCL_EP_DEBUG` prints a line once.
- **Payload bits.** A payload word equal to `0xFFFFFFFF` is replaced by the
  canonical NaN, so the bit pattern of that NaN is not preserved.
- **Memory.** Two Lamport segments per group: `2 × num_max_dispatch_tokens_per_rank ×
  capacity top-k × slot_bytes` plus the round words (128 tokens, top-k 8, hidden
  7168 BF16: about 30 MB).
- **Creation and recovery stalls.** Creating or growing the region and
  `ncclEpMaskClean` are collective with no operation in flight. A reallocation
  adds only a device synchronization; an in-place install in a fixed-size group
  adds two device-and-rank fences (`cudaDeviceSynchronize` and `ncclBarrier`).
  Paid only in this mode, on growth.
- **Timeouts.** Like the 2-sided receive: without a rank mask a timeout traps.

## Validation

Pre-Mario (sm_107), 4 ranks on one node over NVLink; hidden 7168, top-k 8, 256
experts, BF16; `ep_bench`, event-timed combine p50 in µs, three repetitions;
no head-sync.

| tokens | 2-sided | Lamport |
| ---: | ---: | ---: |
| 1 | 23.8 – 24.0 | 19.1 – 20.5 |
| 32 | 24.2 – 24.9 | 20.5 – 21.6 |
| 128 | 26.9 – 27.2 | 22.7 – 23.7 |

`ep_bench --validate` passes for both at 1, 32 and 128 tokens. Resource usage of
the JIT-compiled combine kernel for the same configuration (`cuobjdump
-res-usage`; the base is `upstream/main`):

| kernel | registers | stack |
| --- | ---: | ---: |
| 2-sided, base | 48 | 48 |
| 2-sided, this change | 48 | 48 |
| Lamport | 64 | 72 |

None of them has local memory. The existing combine kernel keeps its register
and stack usage for this configuration (the launch bound now follows the block
size, so other block shapes get other budgets). The multi-rank
tests (`test_ll_lamport_combine`) cover back-to-back combines under dense and
sparse routing with a delayed rank, payload words equal to the sentinel,
`ncclEpMaskClean` followed by more rounds, Lamport and fallback combines mixed on
one group, handles of different top-k sharing the region, and a timed-out round
gate.

## Further work

- Split send and receive CTA roles. At small batches, with enough CTAs, receive
  CTAs can start consuming data as it arrives instead of after their own sends.
- Coarser polling granularity. The loads are already 16 B, but the protocol
  assumes only 32-bit atomicity, so the arrival test checks all four words of a
  vector and the sender rewrites every sentinel word. If 16 B single-copy
  atomicity of the bulk stores over NVLink can be established, both reduce to one
  word per vector.
