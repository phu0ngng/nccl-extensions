# Relative Rank Slot Addressing for the LL Dispatch Staging Buffer

## The problem this design fixes

Each rank's LL dispatch staging region in a peer's `recvBuf` is placed at a
fixed offset equal to the sender's **absolute** rank index:
`recvBuf + srcRank * maxTokensPerRank * numBytesPerMsg`. Every receiver in the
group therefore has rank *k*'s data land at the identical physical offset,
for every receiver, at the same time:

```text
Absolute indexing (current) -- receiver's staging buffer, slots left to right:

  Receiver 0: [ R0 | R1 | R2 | R3 ]
  Receiver 1: [ R0 | R1 | R2 | R3 ]
  Receiver 2: [ R0 | R1 | R2 | R3 ]
  Receiver 3: [ R0 | R1 | R2 | R3 ]
                 ^
                 same physical slot, every receiver, always written by rank 0
```

Under concurrent NVLink writes to the increasing number of ranks, the destination
VAs are staying in the same range. The hypothesis is that it is causing camping
on the output NVL Ports (as port selection is based on VA subrange).
TODO: Confirm this with @Arnav Goel

| EP  | Dispatch avg (us) | Dispatch min (us) |
|----:|----------------------:|----------------------:|
|   4 |                  44.68 |                  40.80 |
|   8 |                  51.87 |                  47.23 |
|  16 |                  59.26 |                  52.96 |
|  32 |                  65.74 |                  55.78 |
|  64 |                  66.90 |                  57.15 |

**NOTE**: Not all the scaling issue is attributed to this fix.

## Design

The fix keeps the existing per-rank region size and total staging-buffer
footprint unchanged — only *where* each sender's region lands moves. Instead
of indexing by absolute rank, each sender computes a **relative** distance to
the receiver going forward around the rank ring:

```cpp
// Distance from senderRank to receiverRank going forward around the ring,
// in [0, numRanks). Branch-based (no modulo).
__device__ __forceinline__ int relativeRankSlot(int senderRank, int receiverRank, int numRanks) {
    return (senderRank >= receiverRank) ? (senderRank - receiverRank)
                                         : (senderRank - receiverRank + numRanks);
}
```

Applied on both ends of the wire protocol, same (sender, receiver) argument
order so both sides agree on the physical slot without any extra
coordination:

- **Sender** (`ll_ep.cuh`, forwarding-warp loop): region offset becomes
  `relativeRankSlot(currRank, dstRank, numRanks) * srcRankRegionBytes`
  (was `currRank * srcRankRegionBytes`).
- **Receiver** (`ll_ep.cuh`, recv-phase copy loop, both `EXPERT_MAJOR` and
  `RANK_MAJOR` layouts): `recvBuf + relativeRankSlot(srcRank, currRank, numRanks) * maxTokensPerRank * numBytesPerMsg`
  (was `srcRank * maxTokensPerRank * numBytesPerMsg`).

```text
Relative indexing (this change) -- same 4 receivers, slot = relativeRankSlot(sender, receiver):

  Receiver 0: [ R0 | R1 | R2 | R3 ]
  Receiver 1: [ R1 | R2 | R3 | R0 ]
  Receiver 2: [ R2 | R3 | R0 | R1 ]
  Receiver 3: [ R3 | R0 | R1 | R2 ]
                 ^
                 slot 0 is now written by a different sender at every receiver
```

Local bookkeeping indices (`outSrcInfo`/`outRecvTopkIdx` slot numbers) and the
separate zero-copy `recvDataWindow` direct-write path are unaffected; they
don't address into this staging buffer.

## Impact

Comparison against `upstream/main` (`32911ff`), 128 tokens/rank, `hidden=7168`,
`top-k=8`, `experts=256`, `bf16`, GB200:

| EP  | baseline avg (us) | relslot avg (us) | Δ%     | baseline min (us) | relslot min (us) | Δmin (us) |
|----:|----------------------:|----------------------:|--------:|----------------------:|----------------------:|-------------:|
|   4 |                  41.31 |                  38.43 |  -7.0% |                  37.31 |                  34.18 |        -3.13 |
|   8 |                  49.50 |                  46.40 |  -6.3% |                  44.10 |                  40.16 |        -3.94 |
|  16 |                  56.11 |                  52.75 |  -6.0% |                  50.66 |                  47.14 |        -3.52 |
|  32 |                  58.98 |                  54.12 |  -8.2% |                  53.06 |                  47.36 |        -5.70 |
|  64 |                  61.64 |                  55.84 |  -9.4% |                  54.53 |                  49.34 |        -5.19 |

Consistent improvement that grows with EP, matching the contention
hypothesis: more ranks means more concurrent writers hitting the old
scheme's shared absolute offsets.

## Validation

`--validate` (Dispatch + Combine) passes at every EP (4/8/16/32/64) on
`upstream/main`, with and without this change. No RDMA-path testing yet —
this cluster's MNNVL fabric keeps all tested EP scales on the NVLink
direct-write path.
