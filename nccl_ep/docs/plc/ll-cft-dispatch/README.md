# Low-Latency Dispatch Payloads over CFT Counted Fabric Writes

## Overview

The existing NVLink dispatch path uses warp-wide loads and stores to move each
token. Each destination's sending warp loads the same payload from `inData`
(gMem) into its own registers, then stores it into that destination's receive
buffer. A software counter and release/acquire ordering ensure that the
receiver does not read the payload before those stores are complete.

This design uses TMA to stage the payload once in shared memory, then CFT
counted puts to send it to the remote destination ranks. Lane 0 of warp 0 of
each CTA stages each token's payload and issues every transfer; the other lanes
of warp 0 prepare the routing, allocate the receive slots and write the header,
and no other warp moves payload. The last warp of each CTA counts tokens and
publishes the token counts. The hardware moves the bytes and updates a byte
counter on the receiver. The self-rank destination reuses the same staged
payload through a local TMA store.

The trade is less SM instruction and register work in exchange for shared-memory
staging and asynchronous-transfer setup. The path remains opt-in through
`NCCL_EP_LL_CFT=1`; disabling it keeps the existing store-based dispatch path.
This PLC covers dispatch within one NVLink domain. Combine keeps its transport
and its flags; its flag wait gains the zero store described in "Self-cleaning
count and flag words".

### Code layout

- `nccl_ep/device/cft.cuh`: the fabric transport, namespace `nccl_ep::cft`:
  `Smem`, `Cft<Coop>` with `init`, `put`, `putCounted`, `submit`, `flushSmem`,
  the bounded `flush`, `waitCounted` and `memFence`, plus the TMA staging
  helpers. Shaped after `nccl_device/cft.h` as of NCCL 2.32.1 (see "Current
  scope and remaining limitations").
- `nccl_ep/device/ll_ep_cft.cuh`: the dispatch protocol on top of it, one
  `__device__ __forceinline__` helper per phase of this document
  (`cftInitSender`, `cftStagePayload`, `cftSendToken`, `cftDrainSender`,
  `cftPublishCountEarly`, `cftAwaitPayload`) with their state in
  `CftTileGeometry`, `CftSendContext`, `CftSelfDestination`,
  `CftMessageLayout` and `CftRecvChannel`.
- `nccl_ep/device/ll_ep.cuh`: the LL dispatch kernel; the CFT arm calls the
  helpers above where the store path sends, publishes and waits.
  `nccl_ep/device/ll_ep_adapter.cuh/.cu`: `cft_dispatch_args_t` and the
  dynamic shared-memory sizing.
- `nccl_ep/nccl_ep_cft.h/.cc`: the host side: platform check
  (`nccl_ep_cft_platform_query`, `nccl_ep_cft_platform_judge`, collective
  verdict), fabric receive buffer and parity layout, logical endpoints and
  their exchange, strict init and teardown.

## The existing store path

### SM allocation and token assignment

The dispatch grid is sized from the total number of experts and the communication
SM budget, not from the token count. With the same number of experts on each rank:

```text
Total experts = numLocalExperts × numRanks

Warp groups per CTA = ceil(total experts / SM budget)
Number of CTAs      = ceil(total experts / warp groups per CTA)
```

When the expert count fits within the SM budget, each CTA has one warp group.
Larger expert counts require multiple warp groups per CTA. The SM budget comes
from the resolved communication configuration; CUDA schedules the CTAs on the
available SMs. Token assignment follows the CTA index, not a fixed physical SM.

A CTA has `G × ⌊32 / G⌋` warps for `G` warp groups (32 when `G` divides 32),
split evenly: each group is `⌊32 / G⌋` consecutive warps, with
`warpGroupId = warpId / warpsPerGroup` and `subWarpId = warpId % warpsPerGroup`.
"The first warp of a group" is its `subWarpId == 0` warp, "the second warp" its
`subWarpId == 1` warp; a group has at least two warps. With one group per CTA
the group is the whole CTA, so its first and second warps are warps 0 and 1.
A group's assigned expert is `responsibleExpertIdx = CTA index × G +
warpGroupId`; the group is valid when that index is below the total expert
count.

CTA `i` processes tokens `i`, `i + numCTAs`, `i + 2 × numCTAs`, and so on. For
example, with two CTAs:

```text
         Tokens processed in order →

CTA 0    [Token 0] → [Token 2] → [Token 4]
CTA 1    [Token 1] → [Token 3] → [Token 5]
```

Within each CTA, the first `topK` warps (warps `0 .. topK − 1`, whatever the
warp-group split) correspond to the token's top-k routing entries. If several
entries belong to the same destination rank, only the first one sends the token
to that rank. The last warp (`warpId == numWarps − 1`) is the counting warp: it
takes no part in sending and scans the routing tensor to determine the total
expected token count for its assigned destination ranks. It counts across all
input tokens, not just the tokens sent by its own CTA.

Each warp reads the routing entries of the token and checks whether an earlier
entry maps to the same rank as its own. The warp with the first entry for a rank
wins and sends; the others skip. For example:

```text
Token 0 selects three experts: A0, A1, B0

  Warp 0 → expert A0 → rank A → send
  Warp 1 → expert A1 → rank A → skip: warp 0 already sends to A
  Warp 2 → expert B0 → rank B → send
```

### Payload movement

Each winning warp's 32 lanes load the payload from `inData` (gMem) into registers
and store it into the destination's receive buffer. A token sent to three ranks
therefore goes through three separate warp-level load/store sequences:

```text
                         Same token in inData (gMem)
                                      │
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
             Warp A registers    Warp B registers    Warp C registers
                  │                   │                   │
                  ▼                   ▼                   ▼
             Rank A buffer       Rank B buffer       Rank C buffer
```

These are duplicate load instructions and register work. They do not necessarily
cause duplicate HBM reads, because the loads can hit in cache. The header carries
the token ID and the full top-k routing table, allowing each receiver to identify
the entries that belong to its local experts.

### How the receiver knows the stores are complete

The header and payload stores are weak stores by all 32 lanes of the winning
warp (`st.global.L1::no_allocate`, the header vectors first, then the payload).
Right after the warp has issued its stores, lane 0 of that warp adds one to
`rankDone[dst]` with a GPU-scope release atomic
(`atom.add.release.gpu.global.s32`). This release is part of the ordering chain
that makes the payload safe for the receiver to read.

Each source rank launches its own set of CTAs, with one counting warp per CTA.
All counting warps on that source rank scan the same complete local top-k table.
Each CTA counts destinations in its rank window: the ranks that host the experts
`[CTA index × G, CTA index × G + G)` assigned to its warp groups. For destination
`r`, `N_r` is the number of tokens this source sends to `r`, counting each token
once per destination. The counting warp's lanes count in registers, the warp
reduces, and lane 0 stores each `N_r` into the CTA's
`sharedNumTokensSentPerRank` in shared memory. CTAs with overlapping windows
compute the same full `N_r` for those destinations.

`rankDone` lives in the source rank's gMem workspace, so all of its CTAs share
the same per-destination completion counters. Let `F = FINISHED_SUM_TAG`.
Exactly one CTA contributes `F - N_r` to `rankDone[r]`: the owner CTA, the one
whose assigned expert range contains rank `r`'s first expert, global expert
`r × numLocalExperts`. In its counting warp, the lane whose index is `r` minus
the first rank of the window makes that addition with
`atom.add.release.gpu.global`. These updates can run in parallel across CTAs
for different destinations. The ownership rule ensures one counting
contribution per destination even when several CTAs compute its count.

For one source-to-destination pair, three contributions determine when the
destination is ready:

```text
CTA 0's counting warp, lane r:
  next bank cleared ───────────────────→ +F, release.gpu ─────────┐
                                                                  │
Owner CTA's counting warp, lane r − first rank of its window:     │
  N_r known ───────────────────────────→ +(F − N_r), release.gpu ─┤
                                                                  ├──→ rankDone[r]
Winning warp, lane 0:                                             │
  token 0 stores issued ───────────────→ +1, release.gpu ─────────┤
  token 1 stores issued ───────────────→ +1, release.gpu ─────────┤
                  ...                                             │
  token N_r−1 stores issued ───────────→ +1, release.gpu ─────────┘

Ready when: F + (F − N_r) + N_r = 2F
```

CTA 0's counting warp is also the cleaner: its lanes store zero over the whole
next signaling bank with plain stores, the warp synchronizes (`__syncwarp`),
and lane `r` adds `F` to `rankDone[r]` with `atom.add.release.gpu.global` for
every destination `r`. Winning warps from any CTA contribute `+1` for their
tokens sent to `r`; their payload destinations are independent of their CTA's
counting window. These additions can happen in any order. The counter reaches
`2F` only after the counting contribution, the next-bank cleanup, and all `N_r`
token contributions have arrived. The receive phase resets it to zero for the
next dispatch, after the grid barrier.

Count publication is a separate role from the single counting contribution.
After the CTA barrier (`__syncthreads`, which also publishes
`sharedNumTokensSentPerRank` to the whole CTA), lane 0 of the first warp
(`subWarpId == 0`) in **every valid warp group** (`responsibleExpertIdx <
numExperts`) reads the count from its CTA's shared memory and spins on
`rankDone[r]` with GPU-scope acquire loads (`ld.acquire.gpu.global.s32`) until
it reads `2F`. Its assigned expert determines the destination and channel:

```text
r       = responsibleExpertIdx / numLocalExperts
channel = responsibleExpertIdx % numLocalExperts
```

All publishers for `r`, including those in different CTAs, wait on the same
`rankDone[r]`. Each publishes `-N_r-1` into its own channel on the destination
with one `st.release.sys.global.s32` through the peer mapping. The whole chain,
with the instructions that carry the ordering:

```text
  Sender                                     Receiver
  ──────                                     ────────
  Winning warp, 32 lanes:
  header and payload stores ────────────────→ Receive buffer, split layout
  st.global.L1::no_allocate                   (source region, slot)
           │
           ▼
  Lane 0 of the winning warp:
  increment local rankDone[r]
  atom.add.release.gpu.global
           │
           ▼
  rankDone[r] reaches 2F
  (+F from CTA 0's counting warp after
   the next-bank clear,
   +(F−N_r) from the owner CTA's counting warp,
   +1 per token sent)
           │
           ▼
  Lane 0 of the first warp of each
  valid warp group observes 2F
  ld.acquire.gpu.global
           │
           ▼
  Publish count: -N_r-1 ────────────────────→ recvCntBuf[channel × numRanks + src]
  st.release.sys.global                           │
                                                  ▼
                                           Lane 0 of the second warp of the
                                           receive warp group observes nonzero
                                           ld.acquire.sys.global
                                           then st.relaxed.sys word = 0
                                                  │
                                                  ▼
                                           Read header and payload
```

There are `numLocalExperts` count channels per source-to-destination pair. Each
publisher writes the word at `recvCntBuf[channel * numRanks + currRank]` in the
destination's selected count bank, where `currRank` is the sender's rank. Each
valid warp group owns a unique assigned expert and therefore a unique
destination/channel pair. The publishers write the same source-to-destination
count into different words, once per channel. The local expert index identifies
the channel; the value is the full `N_r` for that source and destination.

Zero means the count has not arrived; `-1` represents a valid zero-token
dispatch. Waiting for the cleaner's contribution also prevents count
publication from allowing a peer to reuse a count bank that is still being
cleared: the peer's next write into that bank follows its acquire of this
count, and this count follows CTA 0's release after the clear.

### Receiver work assignment

Each receiver has `numLocalExperts` warp groups serving each source rank, for a
total of `numLocalExperts × numRanks` active groups. For example, with two local
experts and three ranks:

```text
Source rank 0 ──→ Receive groups 0 and 1
Source rank 1 ──→ Receive groups 2 and 3
Source rank 2 ──→ Receive groups 4 and 5
```

These groups divide the source's routing scan; a group does not exclusively own
one expert's output. In expert-major layout, the groups partition the
`(received token, top-k entry)` work. Each entry identifies its actual local
expert, and the processing warp copies the token to that expert's output. Thus,
all routing entries are covered even though expert ownership is not assigned
one-to-one to warp groups.

## CFT counted writes

### Stage once, send to each destination

Warp 0 owns the CFT sender in each CTA and runs it for every token of the CTA.
Its lanes load routing entries (lane `k` holds entry `k`), remove duplicate
destination ranks (the first lane for a rank wins), allocate receive slots (one
atomic per winning lane), write the header into the staging tile and, when the
self rank is a destination, into the local header slot. Lane 0 issues every
TMA, fabric, and mbarrier instruction for the sender. Lane 0 of warps
`0 .. topK − 1` still writes its routing entry into the `sendBuf` header slot as
on the store path; under CFT those warps move no payload.

One TMA load (`cp.async.bulk`, global to shared) stages the token's payload from
`inData` (gMem). Each distinct remote destination then receives one counted
`fabric.try_put` from that shared-memory tile. The header and payload travel
together in the current design.

```text
                         inData (gMem)                 sendBuf header (gMem)
                              │                                 │
                         One TMA load                  Ordinary lane loads
                         (payload only)                and shared-memory stores
                              │                        by the lanes of warp 0
                              ▼                                 │
                   Shared-memory staging tile ◀─────────────────┘
                       [header | payload]
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         Counted put     Counted put     Counted put
              │               │               │
              ▼               ▼               ▼
         Rank A buffer   Rank B buffer   Rank C buffer
```

Only the payload arrives by TMA. The header is still built in the `sendBuf`
staging slot exactly as in the store path, and the lanes of warp 0 copy it into
the front of the tile with ordinary loads and stores; at a few dozen bytes it is
too small to be worth an asynchronous copy. For a token sent to `D` distinct
remote ranks, the transfer sequence is one TMA load, `D` counted puts, and one
submit for the batch. Routing, slot allocation, header stores, and
synchronization still require SM work, but the payload no longer passes through
a separate set of registers for every destination.

The message size is `headerBytes + payloadBytes`. The header contains the token
ID and full top-k routing table, padded to a 16-byte boundary:

```text
headerBytes = align16(4 + topK × routingEntryBytes)
```

It can be as small as 16 bytes; for rank-major layout with top-k 8, it is 80
bytes. The same message is sent once to each destination rank, even when that
rank hosts several of the token's selected experts.

### Hardware delivery accounting

CFT updates a receiver-side byte counter only after the corresponding bytes
have been written into the receive buffer. All CTAs from one source rank
contribute to the same counter of the selected parity copy. The counter
records completed bytes, not the number of submitted puts. A put's contribution
can arrive in several increments rather than one update for the whole message.

```text
Source CTA 0: counted puts ──┐
Source CTA 1: counted puts ──┼──→ Receiver byte counter for this source
Source CTA 2: counted puts ──┘
```

This replaces the remote destination's per-token `rankDone` contributions, the
owner CTA's `F − N_r`, the cleaner's `+F` and the publisher's `2F` wait. The
receiver waits for the expected byte total and then issues an acquire fence
before reading the fabric-written header and payload.

### Publish the token count early

The token count still uses `recvCntBuf`. It is only one 4-byte word per receive
channel, or `numLocalExperts × 4` bytes per source-to-destination pair, so it is
not a useful bulk-transfer payload.

Under CFT, the counting warp of the owner CTA publishes `-N-1` into all
`numLocalExperts` count channels of a destination with relaxed stores
(`st.relaxed.sys.global.s32`, one per channel, through one peer-pointer
translation). The same ownership rule as the store path's `F − N_r` selects the
CTA, and the same lane, `r` minus the first rank of the window, issues the
stores. It does this for every destination it owns, the self rank included, as
soon as it knows `N`. It does not wait for the payload sends, for their
completion reports, or for any other CTA. Lane 0 of the first warp of each
valid warp group, the store path's publisher, publishes nothing under CFT.

```text
Counting warp (owner CTA): derive N → publish -N-1 (st.relaxed.sys), no wait

Warp 0 (every CTA): stage payloads → counted puts → completed bytes accumulate
                    self rank: TMA bulk store → drained before the grid barrier

Receiver:           count tells HOW MANY; the byte counter tells WHEN READY
                    self rank: the grid barrier tells WHEN READY
```

`rankDone` has no writer and no reader in the CFT send phase; the only access
left is the receive phase's reset to zero, which runs on both paths. Remote
destinations gate on the byte counter. The self destination gates on the grid
barrier, described in "The self-rank destination". The next-bank cleanup is
replaced by self-cleaning count words, described in "Self-cleaning count and
flag words".

The count scan deduplicates each token's destinations in the same way as the
sender. It batches routing loads and accumulates per-rank counts across the
warp, so count generation can overlap transmission. Early publication is safe
because the count no longer acts as the payload completion signal.

### Three synchronization purposes

The store path's ledger serves three purposes at once. CFT separates them.

Remote payload completion: how a receiver learns that every payload byte from a
source has landed.

- Store path: per-token `+1` release adds into `rankDone[r]`, the owner CTA's
  `F − N_r`, the cleaner's `+F`, the publisher's acquire of `2F`, the count's
  `st.release.sys`, the receiver's `ld.acquire.sys` of the count.
- CFT: the hardware byte counter, incremented after the bytes are written, the
  receiver's relaxed poll and one `fence.proxy.generic::fabric.alias.acquire.sys`.
- Removed: every `rankDone` access in the send phase and the count's release
  duty; the count store is relaxed.
- Still necessary: the count itself (it sizes the target), the poll, the fence,
  and the receiver's `rankArrivedCnt` fan-out to the other warps of its groups.

Cleanup of the next bank: who clears the count and flag words, and how the clear
is ordered against the next writer.

- Store path: CTA 0's counting warp clears the whole next bank, then releases
  `+F`; every count publish waits for `2F`, so a peer can write into that bank
  only after the clear. The combine kernel clears the dispatch's bank the same
  way from its CTA 0, gated by its `atomicCleanFlag`.
- CFT: the consumer of a word stores zero into it right after the acquire load
  that observed it. The dispatch's clear and its `+F` do not run. The combine
  kernel is unchanged and still clears the dispatch's count bank.
- Removed: the dispatch's bank clear, its `+F`, and the publish wait on it.
- Still necessary: one relaxed zero store per consumed word; the host zeroing of
  the buffer for the first use of each bank.

Self-rank completion: how the local receiver learns that its own rank's payload
is complete.

- Store path: the winning warp's `+1` release add covers the self token like any
  other; the self publisher waits for `2F` and stores the count.
- CFT: `cp.async.bulk.wait_group 0` and `fence.proxy.async.global` by lane 0 of
  warp 0 after its token loop, then the cooperative grid barrier (the kernel
  boundary in split-phase launches) between the send and receive phases.
- Removed: the per-CTA self release add, the self `F − N`, the self publisher's
  spin and release store.
- Still necessary: the two bulk waits, the proxy fence, and the barrier the
  kernel already has.

### Self-cleaning count and flag words

The dispatch count words and the combine finish-flag words share one signaling
region with two banks. The LL epoch advances once per dispatch and once per
combine and selects the bank (`epoch & 1`): dispatch `N` uses bank `b`, combine
`N` bank `b ^ 1`, dispatch `N + 1` bank `b` again. This LL epoch bank is
independent of the CFT parity copy that holds the fabric payload and the byte
counters. The store path clears the next bank from CTA 0 while the current
operation runs and gates every count publish on that clear through `rankDone`.
Without the gate, a peer released by this dispatch's count could write its next
flag into the bank while CTA 0 is still clearing it.

Under CFT, each consumer clears its own word. Lane 0 of the second warp of a
receive warp group polls its count word and stores zero into it
(`st.relaxed.sys.global.s32`) right after the acquire load that observed the
count. Lane 0 of the first warp of a combine receive group does the same for its
flag word. Exactly one thread consumes each word, and exactly one thread writes
it per operation: the owner CTA's counting warp under CFT, one publisher per
channel on the store path. CTA 0's clear and its `rankDone` contribution do not
run under CFT, and the count publish waits for nothing.

The next writer of a word is two LL operations away. A peer writes into it only
after finishing the operation in between. That operation needs this rank's next
kernel, which starts after the kernel that stored the zero has completed.

```text
  This rank                                    Peer
  ─────────                                    ────
  Op N-1 receive (bank b):
  ld.acquire.sys word (observes the value)
  st.relaxed.sys word = 0
           │
  Kernel N-1 completes: the zero is performed
           │
           ▼
  Op N send (bank b ^ 1):
   dispatch: st.relaxed.sys count ───────────→ ld.acquire.sys count, then the
             + counted puts                    byte counter reaches the target
   combine:  st.release.sys flag ────────────→ ld.acquire.sys flag
                                                      │
                                                      ▼
                                               Complete op N receive
                                               Start op N+1 (bank b)
                                                      │
                                                      ▼
                                               Write into the same word:
                                               st.release.sys flag (combine) or
                                               st.relaxed.sys count (CFT dispatch)
```

The zero store is complete before this rank issues anything in op N: a kernel
completes only after its stores are performed. The peer's store follows
something this rank issued in op N: a flag it acquired (a release/acquire pair
when op N is a combine), or a count it observed together with the bytes the
counter accounted for (when op N is a CFT dispatch; the count store is relaxed,
so this link is the observation of a performed store, not a release/acquire
pair). So the zero precedes the new value in the word's coherence order. A
zero-token round follows the same chain: the count alone releases the peer, and
no clear has to finish first. The zero store is same-address ordered after the
load that read the count, so it never erases a value the consumer has not read.
The first use of each bank relies on the host zeroing the buffer when it is
allocated.

The combine kernel is unchanged apart from the zero store: its CTA 0 still
clears the dispatch's count bank before its flag publishers run. In a
dispatch/combine alternation the count words are therefore cleared twice under
CFT; the consumer's zero is what clears the flag words, and the count words in
dispatch-only sequences.

The store path keeps its clear and its `rankDone` ledger. Its consumers also
zero their words, which duplicates the clear and changes nothing else.

### Receiver ordering and cumulative targets

The receiver first acquires the count word, as in the store path, and zeroes
it. For CFT, receiving the count establishes only the expected byte total; it
says nothing about payload visibility, and the payload may still be in flight.
The receiver turns the count into a target: `target = previous target + N ×
messageBytes`, where `messageBytes = headerBytes + payloadBytes` is the size of
one counted put and the previous target is the running total this receive
channel saved after its last round. It then polls the hardware byte counter with
relaxed loads until the counter is at or above the target. A single
`fence.proxy.generic::fabric.alias.acquire.sys` after a successful poll makes
the fabric-written bytes readable. The header is then read with ordinary loads
and the payload copied out with `ld.global.nc.L1::no_allocate` loads, exactly as
on the store path.

```text
  Sender                                     Receiver
  ──────                                     ────────
  Lane 0 of warp 0:
  counted put of [header | payload] ───────→ FABRIC receive buffer,
  fabric.try_put.async.counted::bytes         current parity copy, slot for
  + fabric.submit                             this source rank; the hardware
           │                                  adds the bytes to the source's
           │                                  byte counter after writing them
           │
  Counting warp of the owner CTA:
  publish count: -N-1 ──────────────────────→ recvCntBuf
  st.relaxed.sys.global (early)                   │
                                                  ▼
                                           Lane 0 of the second warp of the
                                           receive warp group:
                                           observe the count
                                           ld.acquire.sys.global
                                           st.relaxed.sys.global word = 0
                                                  │
                                                  ▼
                                           Target = previous target
                                                  + N × messageBytes
                                                  │
                                                  ▼
                                           Poll byte counter until ≥ target
                                           ld.relaxed.sys.global.u64
                                                  │
                                                  ▼
                                           One acquire fence
                                           fence.proxy.generic::fabric.alias.acquire.sys
                                                  │
                                                  ▼
                                           atom.add.release.gpu rankArrivedCnt[src];
                                           every warp of the groups for src:
                                           ld.acquire.sys until all have arrived
                                                  │
                                                  ▼
                                           Read the header (ordinary loads) and
                                           the payload (ld.global.nc) from the
                                           FABRIC receive copy; write the
                                           payload to the expert output
```

On the receiver, the count acquire, target computation, byte-counter poll, and
acquire fence all run on one thread per receive warp group: lane 0 of the
group's second warp (lane 0 of the first warp is the store path's count
publisher on the send side). After the fence, that thread adds one to
`rankArrivedCnt[src]` with a GPU-scope release add
(`atom.add.release.gpu.global`). Every warp of the `numLocalExperts` groups
serving that source spins on that word with `ld.acquire.sys` until all of them
have arrived; only then do all of their warps start the header scan and
copy-out.

The count only sizes the work; the payload may still be in flight when it
arrives. The copy-out therefore begins after the byte-counter poll and the
fence, not after the count. The bytes are read from the FABRIC receive region
of the current parity copy for that source rank (each slot holds header and
payload back to back) and written to the expert output exactly as in the store
path.

The hardware counters are 64-bit, one per source rank per parity copy, and are
never reset: they are cumulative over every dispatch that used that copy. Two
parity copies of the FABRIC buffer, flipped on the host once per dispatch, keep
consecutive dispatches in separate receive regions; each copy has its own
counters and cumulative targets. Each receive warp group keeps its own software
target, although the `numLocalExperts` groups serving the same source poll the
same hardware counter. After success, it saves the target it waited for, not a
possibly larger observed counter value. A zero-token round needs no payload
wait. The sender's 2^20 × 16 B tracking-mbarrier phase cap (next section) does
not enter this arithmetic: it bounds the completion reports one sender CTA has
outstanding, not the receiver's counter or target.

The count load is `ld.acquire.sys.global.s32` on both paths; the wait is one
function shared by both. On the store path the count is the payload release, so
the load must be an acquire. Under CFT the count store is relaxed, so the
acquire pairs with nothing: payload visibility comes from the counter poll and
the proxy fence for a remote source, and from the grid barrier or kernel
boundary for the self source; the zero store that follows the load is ordered
by same-address coherence; the flag word belongs to the combine, which keeps
its release/acquire pair. A relaxed count load would therefore be correct under
CFT. It is not done: it needs a CFT-gated variant of a shared wait, and no
measurement shows a benefit.

## TMA staging and sender completion

### Two staging tiles

Each CTA uses two shared-memory tiles. At most one global-to-shared TMA load is
in flight: the prologue loads the first token, and each iteration waits for its
current load before starting the next token's load into the other tile.

```text
  Lane 0 of warp 0                              Tile 0 (token A)      Tile 1 (token B)
  ────────────────                              ───────────────       ───────────────
  Wait for token A's G→S load                   gMem ──→ tile 0
  mbarrier.try_wait.parity (TMA mbarrier 0)     load complete
             │
             ▼
  Start token B's G→S load into the other tile                        gMem ──→ tile 1
  mbarrier.arrive.expect_tx + cp.async.bulk                           load in flight
  .shared::cluster.global.mbarrier::complete_tx
             │
             ▼
  fence.proxy.async.shared::cta
  (header stores visible to the async proxy)
             │
             ▼
  Send token A from tile 0, S→G:                tile 0 ──→ remote     (still loading)
   one fabric.try_put.async.counted::bytes      (sMem ──→ gMem)
   per remote destination, then fabric.submit   tile 0 ──→ self
                                                (sMem ──→ gMem)
   (self rank: cp.async.bulk.global.shared::cta
    + cp.async.bulk.commit_group)
             │
             ▼
  Wait until the transfers have read tile 0     tile 0 free
  fabric.wait.sync_restrict::reads
  cp.async.bulk.wait_group.read 0
             │
             ▼
  Repeat with token B: wait tile 1's load,
  start token C's load into tile 0, send B
```

The global-to-shared load of the next token and the shared-to-global transfers
of the current token are therefore always in flight on different tiles, and the
tile-release waits are for the reads of the tile only, not for delivery.
Routing, slot allocation, and header staging can run while the current payload
load is in flight: the header occupies a separate part of the tile. A warp
synchronization makes those stores visible to lane 0, and
`fence.proxy.async.shared::cta` makes the header visible to the asynchronous
transfers before they read the tile. Warp 0 manages tile reuse without handing
staging work to another warp.

### Tile reuse is not delivery completion

After submitting a token's puts, the sender waits for the fabric engine to
finish reading the tile. That makes the shared memory reusable; it does not
mean the remote receiver has received the whole message.

```text
Transfer has finished reading the tile ──→ Tile may be reused

Destination byte target is reached ──────→ Receiver may acquire and read

Every put's completion report has landed ──→ CTA may exit; its tracking
in the sender's tracking mbarrier            mbarrier in shared memory goes away
```

Each put registers expected completion units with the sender's tracking
mbarrier (one 16-byte unit per 16 bytes put). The sender normally drains those
reports once after its token loop, with one `mbarrier.arrive` and a wait on the
tracked parity, instead of waiting for full completion after every token. One
mbarrier phase can hold at most 2^20 units (16 MiB of puts), so a budget check
after each submitted batch drains earlier and flips the phase when the pending
units approach that cap. The CTA must not exit while a report can still target
its tracking mbarrier.

The TMA-load waits, fabric tracking-mbarrier waits, and receiver byte-counter
polls use the group timeout. Expiry prints a warning and traps; it does not
permit the sender to reuse an incomplete tile or the receiver to consume
incomplete data. `asyncErrorFlag` is set first when it exists, but under CFT
it never does: the flag is allocated only with `enable_mask`, which CFT
rejects (see "Masking and fault tolerance"). The local TMA bulk-group waits
described below have no timeout form.

## The self-rank destination

When a token also targets an expert on the sender's own rank, warp 0 reuses the
same staged payload. The lanes of warp 0 write the small header directly into
the local header slot with ordinary stores (`st.global.L1::no_allocate`, one
16-byte vector per lane) in the warp-wide phase, before the warp synchronizes
and before lane 0 issues anything for the token; lane 0 then issues one TMA
bulk store from the tile into the local payload slot. No fabric put or hardware
byte counter is needed for this destination.

```text
Header in registers ──→ Lanes of warp 0: st.global ──→ Local header slot

Payload in shared tile ──→ Lane 0 of warp 0: TMA store ──→ Local payload slot
```

This avoids having another warp reload the same payload from `inData` (gMem).
After each token, `cp.async.bulk.wait_group.read 0` ensures that the local TMA
store has finished reading the tile before it can be reused. Destination-write
completion is deferred until the end of the CTA's token loop.

```text
Within each token iteration:

      Issue TMA bulk store
      cp.async.bulk.global.shared::cta.bulk_group
                │
                ▼
      cp.async.bulk.commit_group
                │
                ▼
      cp.async.bulk.wait_group.read 0
                │
                ▼
      TMA has finished reading the tile
      Tile reusable once fabric reads are also complete


After the CTA's token loop:

      cp.async.bulk.wait_group 0
                │
                ▼
      All local TMA writes are complete
                │
                ▼
      fence.proxy.async.global
                │
                ▼
      Async-proxy writes ordered before this thread's later operations
                │
                ▼
      CTA barrier, then the cooperative grid barrier
```

Lane 0 of warp 0 waits for all local TMA writes with `cp.async.bulk.wait_group
0` and issues `fence.proxy.async.global`. It makes no release addition. The
release that publishes the self payload is the cooperative grid barrier before
the receive phase. The chain, link by link:

1. Lane 0 of warp 0 of every CTA issues the bulk stores and, after its token
   loop, `cp.async.bulk.wait_group 0`: every bulk store's writes are complete.
2. The same thread issues `fence.proxy.async.global`: the async-proxy writes
   are ordered before its later generic-proxy operations.
3. The same thread leaves the send branch and reaches the CTA barrier
   (`__syncthreads`), then the grid barrier. It is thread 0 of its CTA, so it is
   the thread that performs the grid barrier's release: `bar.sync 0`, then
   `atom.add.release.gpu.u32` on the grid barrier word.
4. Thread 0 of every CTA spins with `ld.acquire.gpu.u32` on that word until all
   CTAs have arrived, then `bar.sync 0` releases its CTA. A CTA can arrive only
   after its own drain, so the barrier completes only after every CTA's self
   payload is complete.
5. The receive groups run after the barrier. Lane 0 of the second warp of each
   group loads the self count with `ld.acquire.sys` (stored early by the
   counting warp, before its CTA reached the barrier, so it is already there),
   adds to `rankArrivedCnt[self]`, and every warp of the groups for the self
   source spins on it; then they read the header and payload with the same
   loads as for any store-path source.

The header stores are ordinary stores by the lanes of warp 0, ordered before
lane 0 by the warp synchronization at the end of the warp-wide phase, and the
lanes themselves pass the CTA barrier and `bar.sync 0` before thread 0's
release, so the same barrier covers them.

```text
  Sender (warp 0, every CTA)                    Local receiver
  ──────────────────────────                    ──────────────
  Header: st.global.L1::no_allocate by the
          lanes of warp 0, then __syncwarp
  Payload: lane 0, cp.async.bulk.global
           .shared::cta.bulk_group + commit_group
           │
           ▼
  Lane 0: cp.async.bulk.wait_group 0
          fence.proxy.async.global
           │
           ▼
  bar.sync (__syncthreads)
  cg::this_grid().sync():
    bar.sync 0
    thread 0: atom.add.release.gpu.u32 ──┐
    thread 0: ld.acquire.gpu.u32 ◀───────┤ every CTA's thread 0
    bar.sync 0                           │
           └─────────────────────────────┴───→ Lane 0 of the second warp of the
                                                receive group: ld.acquire.sys
                                                count -N-1 (stored early by the
                                                counting warp)
                                                       │
                                                       ▼
                                                Read header and payload
                                                from recvBuf, split layout
```

The self count is published early by the counting warp like every other count.
The receive groups read it only after the barrier, so the count never arrives
before the payload is complete. In split-phase launches the send kernel returns
before the barrier and the receive kernel runs later on the same stream; the
kernel boundary takes the barrier's place, and the drain in step 1 still runs
before the send kernel's CTA exits. The two local TMA bulk-group waits still
have no bounded/try form.

## Zero-copy extension

This is a proposed extension, not part of the currently enabled CFT path.
Currently, CFT and rank-major zero-copy are mutually exclusive; existing
store-path zero-copy behavior is unchanged.

For rank-major output, a CFT payload put could write directly into the
receiver's registered output buffer. This removes the receiver-side payload
copy while retaining the sender's shared-memory staging:

```text
Current CFT:

    Shared tile → Counted put → Internal receive buffer → Copy to output

With zero-copy:

    Shared tile → Counted put → Output buffer
```

The header would remain in a separate header buffer and use its own counted
put. Both puts contribute to the same receiver byte counter, so one completion
test covers the header and the payload. For a round containing `N` tokens, the
counter must advance by `N × (headerBytes + payloadBytes)` beyond its previous
target:

```text
Counted header put ──┐
                     ├──→ Byte counter advances by N × (header + payload bytes)
Counted payload put ─┘                          │
                                               ▼
                                         Acquire fence
                                               │
                                               ▼
                                   Header and payload readable
```

A separate header put is less efficient than a bulk payload put because the
header can be as small as 16 bytes. It adds another transfer command and
completion accounting per token and destination. Counting the header keeps it
under the same arrival check as the payload and preserves early count
publication. Ordinary remote header stores would require an additional
ordering mechanism before the receiver could safely read them.

The self-rank ordering stays the same in this extension: ordinary stores write
the header, and a TMA store writes the payload directly into the local output
row instead of the internal receive buffer. The bulk waits, the proxy fence and
the grid barrier (kernel boundary in split-phase launches) still cover both, so
the self-rank header does not need a counted put.

## Quantization challenge

Quantization remains a challenge for this design. Computing scales and
converting values adds SM computation, register work, and synchronization
between staging and transmission. The goal is to quantize each token once and
reuse it across destinations while preserving overlap between TMA loads and
outgoing transfers.

```text
Current CFT path:

    Stage payload ──→ Counted put


Additional work for quantization:

    Stage input
         │
         ▼
    Compute scales and convert values
    SM computation and register work
         │
         ▼
    Quantized payload and scales ready
    Synchronize before transmission
         │
         ▼
    Counted put
```

Quantization also changes the shape of the sender loop. Today every step after
the header copy runs in one `laneId == 0` region: wait for the TMA load, start
the next load, fence, puts, submit, tile-release waits. The TMA verdict is local
to lane 0 and nothing warp-wide depends on it. Quantizing in shared memory needs
all 32 lanes to read the staged bf16 tile after the load has landed, compute the
scales, convert the values, and write the quantized bytes and scales back into
the tile before lane 0 can put them. That splits the single lane-0 region into
two, with a warp-wide phase in between:

```text
    Lane 0: wait TMA load, start next load
         │
         ▼
    Broadcast the verdict to the warp (shuffle)
         │
         ▼
    All lanes: compute scales, convert, store into the tile
         │
         ▼
    __syncwarp, fence.proxy.async.shared::cta
         │
         ▼
    Lane 0: puts, submit, tile-release waits
```

Each token therefore pays an extra shuffle broadcast, a warp synchronization,
and a reconvergence point, and the warp-wide conversion sits between the load
wait and the puts instead of overlapping them.

## Masking and fault tolerance

CFT already provides operation error reporting; the missing piece is the EP
protocol that turns those reports—and software timeouts—into safe recovery.
This deserves a separate PLC and is outside the scope of this design.

The store path can skip a source after a receive timeout and mark it inactive
in the local rank mask. Supporting this behavior with CFT requires additional
recovery logic: the token count may already have arrived while some payload
bytes are still missing.

The CFT byte counter accumulates completed bytes across dispatches. It does not
identify which dispatch those bytes belong to, and a receiver timeout does not
cancel outstanding puts. Consequently, simply masking the source and adjusting
the counter is not enough to make its buffers safe to reuse.

```text
Round A expects 1,024 B; only 512 B arrive
                     │
                     ▼
          Receiver times out and masks source
                     │
                     ▼
       Unsafe recovery: reset counter and
           later reuse the same parity copy
                     │
                     ▼
       Round B expects 512 B; counter starts at 0
                     │
                     ▼
         The remaining 512 B from A arrive
                     │
                     ▼
       Counter reaches B's target—even though
               B's data has not arrived
```

The counter is not inherently unresettable—it is stored in gMem and initialized
to zero. The difficulty is knowing when it is safe to reset or reuse it. Late
puts can both increment the counter and overwrite the associated payload buffer.
Two alternating parity copies do not solve this if old traffic survives until
a copy is reused.

The main questions for a masking-aware CFT design are:

- Outstanding traffic: What guarantees that a masked source can no longer write
  into the old buffers? Can completion/error reports establish this, or is
  endpoint isolation needed?
- Continued operation: Can we leave that source's buffers unused while healthy
  ranks continue, with all receiver warp groups consistently skipping the
  source?
- Re-admission: Before accepting the source again, how do we synchronize its
  buffer parity, hardware counters, and every receive channel's cumulative byte
  target?

For now, CFT and `enable_mask` cannot be enabled together; a CFT receive
timeout prints a warning and traps instead of masking. The rest of this
section records what the code does on expiry today and what a timed-out round
leaves behind.

### One timeout budget for both paths

Every bounded device wait, on both paths, uses one group value,
`timeout_cycles`. The host resolves it once at group creation and prints the
result on rank 0:

```text
NCCL_EP_TIMEOUT_MS (positive)  >  config.timeout_ns (nanoseconds)
                               >  NUM_TIMEOUT_CYCLES (common.hpp default)
```

The CFT waits take this budget on purpose instead of private spin counts.
Their expiry is fatal, so a short private budget would turn a scheduling
hiccup into a lost context. The compile-time overrides
`NCCL_EP_CFT_PUT_WAIT_CYCLES`, `NCCL_EP_CFT_TMA_WAIT_CYCLES` and
`NCCL_EP_CFT_POLL_MAX_SPINS` exist only to force an expiry in a negative
control. The host converts `config.timeout_ns` directly to GPU cycles with
sub-millisecond precision, saturates oversized budgets, and clamps a positive
budget that rounds to zero to one cycle.

### The store path on expiry

Lane 0 of the receive group's second warp waits for the count word. In
combine, lane 0 of the receive warp waits for the finish flag. Both loops
compare the elapsed `clock64()` against the budget. On expiry:

```text
  Warning printed
        │
        ▼
  rankMask == nullptr ──yes──▶ trap (context lost)
        │ no
        ▼
  atomicExch rankMask[src] = 0            source marked inactive
  atomicExch.sys asyncErrorFlag = 1       host reads it (ncclEpGetAsyncError)
  source counted as zero tokens           this operation completes without it
        │
        ▼
  Later sends and receives skip the source (isRankMasked)
  ncclEpMaskClean: barrier over the unmasked peers, zero both signal
  banks, re-arm the mask
```

`rankMask` and `asyncErrorFlag` exist only when the group was created with
`enable_mask`. Without it the store path traps too.

### CFT on expiry

CFT and `enable_mask` are rejected together at `ncclEpCreateGroup`: the
collective vote returns `ncclInvalidUsage` on every rank. A CFT group
therefore has `rankMask == nullptr` and `asyncErrorFlag == nullptr`, and the
kernel's `cft_enabled` gate additionally requires `rankMask == nullptr`. Every
bounded wait on this path ends the same way when it expires:

```text
  Wait (owner)                                    Expiry
  ─────────────────────────────────────────────   ──────────────────────────────
  Count word, ld.acquire.sys loop                 warning, trap
  (lane 0, second warp of the receive group)      (store-path code, mask absent)

  Byte counter, ld.relaxed.sys.global.u64 poll    warning, trap;
  (same thread)                                   target not banked

  Staging load, mbarrier.try_wait.parity          warning, trap;
  (lane 0 of warp 0)                              tile never put

  Put drain, mbarrier.try_wait.parity             warning, trap
  .phase_type::conditional (lane 0 of warp 0)

  Combine flag, ld.acquire.sys loop               warning, trap
  (combine is unchanged)
```

A trap aborts every CTA of the kernel and leaves the CUDA context on that GPU
unusable; the host sees a launch failure at its next synchronization. Peers
waiting for this rank's count, bytes or flag expire in turn and trap as well.
The group does not continue and nothing is re-admitted. The code never reads
the fabric's error status, so an expiry does not distinguish a slow peer from
a dead one.

### What a timed-out round leaves behind

- The mask: there is none under CFT. A timeout means termination, not
  masking.
- The cumulative target: the receiver keeps the previous target;
  `cftCumulByteExpect[channel]` is not advanced by a round that did not
  arrive. No later operation reads it, because the kernel traps.
- The hardware counter: a plain `u64` at
  `parity_off + byte_cntr_base_off + src × 256` in the receiver's
  fabric-bound allocation. The host zeroes it once at group creation
  (`cudaMemset`) and never writes it again. Software can store to it, since
  it is ordinary mapped memory, but the fabric adds to it asynchronously as
  bytes land, so a software reset races with any put still in flight. A late
  put adds its byte count to the counter of the parity it was issued against,
  whenever its bytes land, and overwrites the slot bytes it was aimed at.
- The parity copy and the bank: the fabric buffer (slots, counters,
  cumulative targets) alternates per dispatch between its two parity copies.
  The host flips `dispatch_parity` in `ncclEpDispatch`; it is group-scoped. The count and flag words alternate
  per LL operation through the device epoch (`epoch & 1`), so consecutive
  dispatches share a count bank while using different parities. The count
  word of the failed round was already zeroed when its consumer observed
  `N`, before the byte poll.

```text
LL operation     dispatch A   combine A   dispatch A+1   combine A+1   dispatch A+2
count/flag bank      b           b^1           b             b^1            b
CFT parity           p            -            p^1            -             p
```

### Late bytes

Whether a receiver can be fooled depends on continuing after the expiry.
Today it cannot continue, so the hazard below is the reason the trap is
unconditional, not a live defect. Suppose round A on parity `p` expected
`N_A × m` bytes above the baseline `B`, received `a < N_A × m`, and the
kernel went on. Dispatch A+2 reuses parity `p`:

```text
Not banked (the current rule):
    target(A+2) = B + N_{A+2} × m
    counter after A+2's bytes = B + a + N_{A+2} × m  ≥  target
    → released early by `a` bytes; slots may hold A's late payload
      next to A+2's. A stale slot has a well-formed header (a token ID
      of round A), so the copy-out cannot tell.

Banked anyway:
    target(A+2) = B + N_A × m + N_{A+2} × m
    → reached only if every byte of A eventually lands; otherwise every
      later round on `p` expires.
```

Late bytes of round A that land after dispatch A+2 has passed its poll still
add to the counter and overwrite slots. The offset is permanent: counters are
never reset and targets only grow. Nothing detects it: the poll tests `>=`,
the count word carries only `N`, and neither the header nor the count word
carries an operation tag. The two-parity ring gives one dispatch of slack;
what keeps a fast rank out of a slow peer's parity copy is the combine round
trip, not the parity.

### What a longer timeout changes

A longer budget makes a slow but live peer, a descheduled kernel or a very
large round less likely to expire, and it delays the loud failure by the same
amount. It does not change what happens at expiry, does not make a dropped
put arrive, does not make late bytes distinguishable from the current round's,
and does not let the group continue. Recovery is orthogonal to the budget.

### Recovery options (not implemented)

None of the following is implemented. The trap stays unconditional until one
of them is.

1. Fatal with a host-visible reason. Allocate the async flag independently of
   `enable_mask` and let `ncclEpGetAsyncError` read it without the mask, so
   the host learns which wait expired on which pair before the context is
   lost. Smallest change; still a process restart.
2. Drain and resync. On expiry the receiver marks the source inactive, and
   every receive group skips it consistently (the mask read once per
   operation, not per thread). The sender must learn that every put to that
   destination has reported, complete or failed, through its tracking
   mbarrier and the fabric error count. Only then may the receiver reset that
   source's counter and targets on both parities and its count words, and
   re-admit it through a collective that includes the masked peer or a host
   handshake. Costs: a fabric error-status query, a per-destination tracking
   barrier instead of one per CTA, a quiesce protocol, parity and epoch
   re-agreement, and the rule that a timed-out tile is never re-put.
3. Per-operation tag. Carry a dispatch sequence number in the count word and
   in each slot header. The receiver checks every consumed slot against the
   current operation and rejects a mixed round instead of trusting the byte
   total. Costs: header bytes (the 16-byte padding absorbs a 4-byte tag for
   most top-k values), one compare per slot in the copy-out, and a count word
   that holds `N` and the tag. Detection only; continuing still needs
   option 2.
4. Deeper ring or per-operation counter reset. More parities widen the gap
   between two uses of a slot region; a counter reset per operation needs
   proof that the previous operation drained, which is option 2's quiesce.
   Costs: the fabric buffer grows with the number of parities, and the
   counter region grows with it.

The kernel-side TODO at the failure reporters lists the same prerequisites:
fault-versus-expiry discrimination, receive-region quiescence, parity across
re-admission, and tile quarantine.

## Current scope and remaining limitations

- The extension keeps its own driver-level CFT implementation
  (`cuLogicalEndpoint*` on the host, `fabric.try_put` PTX on the device)
  because the pinned NCCL submodule (2.30.7) predates NCCL's device CFT API
  and users below NCCL 2.32.1 need a working path. NCCL 2.32.1 added what this
  path depends on (`ncclCft::putCounted`, `ncclCft::waitCounted`, counted
  windows through `NCCL_WIN_CFT_COUNTED`). `device/cft.cuh` follows NCCL's
  API to keep future migration small. The differences and expected migration
  are described in [CFT API design](../README.md#cft-api-design).
- The enabled CFT path supports unquantized, 2-byte token elements
  (`QUANT_NONE`). Quantized recipes and fp32 tokens do not use it: the
  kernel's `cft_enabled` gate is false for those template variants, so they
  take the store path silently while the group still initializes CFT (see
  the table below).
- Remote CFT transfers require one NVLink domain. Cross-domain CFT dispatch is
  not supported; group-creation validation for that case still needs hardening.
- Both expert-major and rank-major dispatch are covered without zero-copy.
  The zero-copy extension above remains separate work.
- The CFT parity copy currently advances on the host. CUDA graph replay
  would reuse a captured parity, so replay-safe parity selection remains work
  to be completed.
- Qualification covers four ranks on one node and eight ranks on two nodes of
  one NVLink domain, both layouts, on NVL72 sm_107 nodes (CUDA 13.4, driver
  620) and GB200 sm_100 nodes (CUDA 13.4, driver 615).
- Under CFT, `rankDone` has no writer and no reader in the send phase (the
  receive phase's reset to zero remains on both paths) and the dispatch's
  next-bank clear does not run. Consumers zero their count and flag words on
  both paths; the store path also keeps its clear and ledger, and the combine
  kernel keeps its own clear on both paths.
- Enablement is explicit through `NCCL_EP_LL_CFT=1`; there is no automatic
  batch-size selection. All ranks must use the same setting.
- Group creation validates the platform before any endpoint work: the build
  toolkit, the CUDA runtime, the driver, the compute capability, the unicast
  endpoint attribute, the counted-ops endpoint limits, fabric-handle
  support and the release of the nvcc the JIT resolves (the LL kernels that
  carry the fabric PTX are compiled at runtime). The facts ride the wire-protocol vote, so every rank reaches the
  same verdict. When any rank falls short, group creation fails on every rank
  and the message names each failing rank and requirement. There is no
  fallback to the store path; unsetting the flag selects it.

### Support and limitations

Status under `NCCL_EP_LL_CFT=1` per feature. "Silent" means the request is
honoured by the store path with no message; "loud" means an error return with
a message naming the conflict. Coverage names the suites that exercised the
row; shapes are `experts/top-k/tokens`. Numbers live in the MR description.

| Feature | Under `NCCL_EP_LL_CFT=1` | Fallback or rejection | Correctness coverage | Performance coverage |
|---|---|---|---|---|
| Staged dispatch, bf16 or fp16 tokens, `QUANT_NONE`, expert-major | Implemented: CFT payload path | none needed (fp16 passes the gate but was not exercised) | `gate_layouts.sh` em cells (store and CFT arms); `campaign.sh` phase 0 (every timed shape validated in both arms, CFT proof on every rank); top-k 16/24/31 gate; zero-token pairs, ragged, non-uniform, dispatch-only, long-iteration and single-rank cells; `ep_test -a ll` `-s dispatch\|combine\|both` and `-g`; `validate_store.sh` as the store-path reference | Presented: top-k 8 at 64/128/256 experts, 32 to 4096 tokens; top-k 2/4/6 at 128 experts, 1024 and 4096 tokens. Not timed: tokens below 32 (1, 8 and 16 tokens validated at 4 ranks, both layouts, both arms) |
| Rank-major layout | Implemented: interleaved slots in the fabric buffer, same copy-out as the store path | none needed | `gate_layouts.sh` rm cells; top-k 16/24/31 gate; 8-rank rm cells; hardening cells rm | Presented: 128/8/2048 only; the matrix is expert-major |
| Zero-copy dispatch (rank-major, window-backed outputs) | Not implemented; suppressed group-wide while CFT is live | `zero_copy` AUTO or OFF: staged silently (a debug line under `NCCL_EP_DEBUG`); `zero_copy` ON: loud `ncclInvalidArgument` at `ncclEpDispatch` | Not exercised under CFT, by construction; store-path zero-copy is unchanged and not part of the MR 60 suites | none |
| Quantized dispatch, `NCCL_EP_DISP_QUANT_FWD` (scales forward, byte preserving) | Not implemented: gate term `kRecipe == QUANT_NONE` is false | Silent: CFT init succeeds and prints `[ll_cft] initialized`, the kernel variant carries no fabric code, no message names the fallback (from code; confirmed by a measured cell at 8 ranks) | `ep_bench --dispatch-quantization scales-forward` validates LL em and rm on the store path; not in the MR 60 suites | none |
| Quantized dispatch, `NCCL_EP_DISP_QUANT_DS_FP8E3M4` | Not implemented: same gate term | Silent, as above (from code; confirmed by a measured cell at 8 ranks) | `ep_bench` validates LL expert-major only; rank-major reports "skipped" as passed | none |
| fp32 tokens, `QUANT_NONE` | Not implemented: gate term `kTokenDtype != ncclFloat32` | Silent, as above (from code; confirmed by a measured cell at 8 ranks) | not in the MR 60 suites | none |
| Unsupported platform (build toolkit, runtime, driver, GPU generation, endpoint attributes, counted-ops limits, fabric handles, JIT nvcc release) | Rejected at `ncclEpCreateGroup` before any endpoint work; the facts travel with the collective vote | Loud `ncclInvalidUsage` on every rank; the message names each failing rank and requirement with the raw values | Negative cell through the debug-only driver-floor hook (`NCCL_EP_LL_CFT_DEBUG_MIN_DRIVER` under `NCCL_EP_DEBUG=1`) | n/a |
| Fault handling: `enable_mask` | Rejected at `ncclEpCreateGroup` after the collective vote | Loud `ncclInvalidUsage` on every rank, message names the ranks that asked | from code; negative cell (`ep_bench --mask-test` under CFT) rejected with the documented message at 4 and 8 ranks | n/a |
| Fault handling: timeouts | Every CFT wait bounded by the group budget; expiry prints a warning and traps; no mask, no async flag | Loud and fatal | No expiry cell in the suites; the count-word trap under CFT was observed during bring-up (a harness fault, cross-node); the byte-counter expiry has not been observed | n/a |
| Cross-node, 8 ranks on 2 nodes, one NVLink domain | Implemented: the pair predicate claims every non-self pair; endpoints imported over the fabric | none needed | MR 60 validation rows: 8-rank shapes both layouts, the two-node matrix with per-shape validation, ledger-removal 8-rank cells | Presented: 8-rank rows, top-k 8 at 64/128/256 experts, top-k 2 at 128 experts |
| Cross-domain (two NVLink domains) | Not supported | Loud: endpoint import fails, collective `ncclInvalidUsage` at `ncclEpCreateGroup`; the store path over two domains runs | Observed on two racks (learnings) | n/a |
| Split-phase (`send_only`) and CUDA graph | Split-phase implemented: the kernel boundary replaces the grid barrier. Graph capture works; replay reuses the captured parity | Silent for replay: no check | `ep_test -a ll -s ...` cells; `-g` launches the captured graph once, so replay is not exercised | n/a |
