# HT Signaled Push Combine Direct Store (MR !437)

Follow-up to the chunk-major push of
[HT dispatch and combine readiness signals](../429-ht-signals/README.md) and to
[direct store in small LSA teams](../ht-em-pull-push/README.md) (its Store paragraph, MR425).

## Problem

The signaled push walked its chunks one warp at a time: each warp polled
`signal[s]` itself, then pushed chunk `s` of its rows through the warp's smem
chunk buffer and a TMA store, or stored directly by the unsignaled rule (a
multi-chunk row in an LSA team below 8 ranks). The buffer caps the warps per
block, each chunk waits for the warp's previous TMA store, and every warp spins
on the flag.

## Implementation

A signaled push is stored directly in an LSA team below 8 ranks, and a
single-chunk signaled push at every team size (`combine_push_direct_store`,
variant suffixes `_sig` and `_direct`): each lane stores its reduced int4
straight to the peer's staging row, so the block uses no dynamic shared memory
and keeps all `kCombinePushMaxSlots = 24` warps. From 8 ranks a multi-chunk
signaled push writes each chunk through the warp's smem chunk buffer and one
TMA store, which completes before the tail sync publishes the rows.
`NCCL_EP_TEST_COMBINE_PUSH_TMA_STORE` keeps the TMA store for a signaled push too,
so a small team can test it.

`combine_push_signaled_chunk` pushes one chunk with the whole block:

- One thread polls `signal[s]` and acquires it; the block barrier extends the
  acquire to every warp. A rank with no received rows skips the wait.
- The warps then claim the chunk's rows from the block's counter of that chunk
  and read the producer's data with coherent loads; a single-chunk push of
  rows over 8 KiB also claims from the grid-wide pool, as no other chunk shares
  its counter. From a grid of 24 blocks, at any team size, its pool is three
  quarters of every region, so each warp's rows spread over destinations. A row with one
  contribution is stored as loaded; others are summed and stored.
- The chunk geometry is a compile-time parameter, so one tile is processed at a
  time with up to 8 int4 loads in flight per lane (4 for top-k <= 2 and for rows
  that end in a partial chunk, which spill at 8).
- With chunk 0, the backward stores the weights (zero for a copy this rank dropped), and under
  `NCCL_EP_OVERFLOW_DROP` the push marks the row's pushed-row flag, which the reduce re-arms.

The row-major walk of a block that found every chunk ready at entry is gone:
a chunk's wait is one polling thread per block (a relaxed load, then an acquire
load) followed by a block barrier. The outputs still match
the unsignaled combine numerically for finite values, and a flag that is never
set traps after the device timeout.

In an LSA team of 8 or more ranks, a signaled combine whose row has two or more
chunks runs the chain of [429](../429-ht-signals/README.md) in two parts, unless
the group uses `NCCL_EP_OVERFLOW_DROP`: the push of every chunk but the last; then the
push of the last chunk on half the SMs, beside the reduce of the other chunks on
the rest; then the reduce of the last chunk. Only a TMA-store push can give half
the SMs to that reduce; a direct-store push issues every store from its own SMs,
so in smaller teams the combine stays one push and one reduce of the whole row.
Every rank passes `combine_signal` to a combine, or none does: a rank that
splits runs one more LSA barrier.

## Testing

`test_ht_signal` (`CombineSignalModeTest`) compares the signaled combine with the
plain one bitwise, forward and backward, for several chunks and the whole row, with
overflow drop and eager sizing, with the chunks published during the push or
before it. Below 8 ranks a push stores directly and the combine stays whole; from
8 ranks, and with `NCCL_EP_TEST_COMBINE_PUSH_TMA_STORE=1` at any team size, a
multi-chunk row without overflow drop runs the split chain. A 2880-element row
is split into three chunks, the last holding 832 elements, to cover the partial
last chunk.
