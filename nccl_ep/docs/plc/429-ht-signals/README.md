# HT dispatch and combine readiness signals (MR !429)

Two optional device-flag signals on the HT expert-major pull-push path let a GEMM on the same GPU overlap the
communication kernels:

| Signal | Producer | Consumer | Overlap |
|---|---|---|---|
| Dispatch signal | dispatch | FC1 GEMM | the GEMM starts on the first output tiles before the dispatch ends |
| Combine signal | FC2 GEMM | push combine | the first hidden-dim chunks are pushed before the GEMM ends |

Both are off by default (`nullptr`); a signal selects a separate JIT variant. Unsignaled calls also change: two pad
warps per pull block, an async-proxy fence after the pull head gate, the count row zeroed in the update kernel, the
top-k weights staged in the dispatch kernel, and a PDL forward dispatch once the group has signaled. Every push
combine, signaled or not, ends its push with the LSA tail barrier ("Combine launches" below).
The user-facing contracts are in [api_reference.md](../../documentation/api_reference.md), "Dispatch tile
completion" and "Combine readiness signaling".

## Background

Dispatch and combine are the two halves of the expert all-to-all. Dispatch sends each token to the ranks that hold
its experts; combine returns the expert outputs and sums them back on the token's rank. On the HT (high-throughput)
path the received tokens are laid out *expert-major*: grouped by local expert, so each expert's rows are contiguous
and ready for its GEMM. *Pull-push* is how the data moves within one NVLink domain (an *LSA team*): the dispatch
pulls peers' tokens into the local expert-major layout, and the combine pushes results into peer staging. FC1 and FC2
are the expert MLP's two GEMMs: FC1 consumes the dispatch output, FC2 produces the combine input.

The dispatch output is split into *tiles* (a fixed block of rows) and the combine's row data into *chunks* (a fixed
block of a row's hidden dimension). Each signal is a small int32 array that the producer fills in as it finishes a
tile or chunk, so the consumer can start on the finished ones instead of waiting for the whole kernel.

## API

One struct serves both signals:

```text
ncclEpSignal_t {
    size, magic,
    int                 granularity,  // dispatch: rows per tile; combine: hidden elements per chunk (0 = inherit env)
    ncclEpSignalUnit_t  unit,         // AUTO (default) = the op's natural unit; else TILE_ROWS / CHUNK_ELEMS
    ncclEpTensor_t*     signal        // int32 [>= number of units], reset to -1; producer sets signal[i] = i
}
// as ncclEpDispatchOutputs_t::dispatch_signal (V2) and ncclEpCombineInputs_t::combine_signal (V3)
```

A caller that passes an older, shorter outputs or inputs struct gets the signal disabled. The Python bindings do not
expose the new fields. The caller resets the flags to -1 before every call. Both signals need one NVLink domain: on a
group of several LSA teams a signaled call returns `ncclInvalidUsage` before any kernel launches.

## Dispatch signal

Illustrative timeline for an in-order walk. The ordered pull fills tiles expert by expert, so a later tile can finish
before an earlier one: acquiring `signal[t]` proves tile `t` alone.

```text
 stream  | dispatch (pull, in row order) ..............................|
 rows    | tile 0 | tile 1 | tile 2 | ...                      | tile T-1|
 flags   |   t=0  |   t=1  |   t=2  | ...                                |
                      |
                      v  signal[t] = t  (release, GPU scope)
 stream  |        FC1 GEMM (programmatic dependent): acquire signal[t], compute tile t
```

- **Row order.** The unsignaled pull assigns rows in claim order, so every expert's tiles fill in parallel and the
  first tile completes late. The signaled pull walks the output rows in order, one row per warp per step, so tile `t`
  completes as the walk passes it. The dispatch itself may finish later, but an in-order consumer gets its first
  tiles much sooner.
- **Publishing.** Each warp counts its rows (16-bit counters in group scratch) once their stores complete. The row
  that fills a tile release-stores `signal[t] = t`. Pad rows and rows dropped under `NCCL_EP_OVERFLOW_DROP` are counted
  where they are zeroed. Under overflow drop the output can end inside a tile; that tile is published once its rows
  below the budget are counted.
- **Two walks.** A call with a prebuilt routing map (the scan path, or a map reused by a backward or re-dispatch
  call) walks that map directly. A call that builds its maps on the fly, which is every forward call under fused
  count metadata, uses the *ordered pull*, which overlaps the
  dispatch kernel's two kinds of warp: the MAP warps, which build the routing maps, drop each mapped chunk into a
  block-local pool (a readiness bitmap with one bit per bucket, a bucket being a group of local experts); the payload
  warps, which copy token data, pull the lowest bucket first and claim their expert-major rows as they go. Tiles
  therefore fill expert by expert while the maps are still being built. The pool is capacity bounded; a call whose
  chunks do not fit returns `ncclInvalidArgument`.
- **Programmatic launch.** The kernel calls `griddepcontrol.launch_dependents` after the cross-rank barrier at its
  start, so a consumer behind it starts once every block is running. The consumer executes `griddepcontrol.wait`
  before it exits or triggers its own dependents, and writes nothing the dispatch reads or writes on any rank before
  that wait: peers still pull this rank's inputs.
- **Row map.** A signaled forward that builds its routing maps pulls from its chunk table and needs no row map. A
  signaled backward or re-dispatch rebuilds one per call in group scratch: the recv slot of the token whose lowest
  expert-major row each row is, taken from the handle's slot map. The handle keeps no map of its own, so the call
  follows whichever forward last filled the handle, also through an imported handle (`ncclEpImportHandle`) and a
  graph replay. The scratch holds the tile counters, then the map at a fixed offset after the counters of tile size 1,
  so changing the tile size between calls never aliases them.
  An eager group's handle can pad past the budget the scratch was reserved for; the group then grows the scratch
  before the dispatch, which cannot be captured.

## Combine signal

```text
 producer (FC2 GEMM)   | chunk 0 | chunk 1 | ... | chunk C-1 |
                            |         |               |
                            v signal[s] = s            v
 push combine          |   push 0  |  push 1  | ... | push C-1 |  (NVLink, peer staging)
                                                         |
 reduce                                                  | reduce whole rows -> output
```

- **Chunks.** A chunk is the push chunk of the unsignaled combine, set by `granularity` or (when 0)
  `NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE`: a whole number of push tiles, or the whole row. The signal has one flag per
  chunk, independent of the row count, so overflow drop and eager sizing work; a rank that received no token runs no
  pass.
- **Chunk-major push.** Each warp waits for `signal[s]`, then pushes chunk `s` of the rows it claims, reading the
  producer's data with coherent loads. Rows are claimed per chunk from the block's own region, regardless of
  `NCCL_EP_HT_COMBINE_DYNAMIC_CLAIM_MIN_TOKENS`. A block that finds every chunk ready at entry pushes row-major.
- **Programmatic launch.** The push starts when the kernel ahead of it triggers `griddepcontrol.launch_dependents`,
  with no grid-dependency wait at entry. That kernel therefore triggers only once every earlier kernel on the stream
  has completed, and is not a signaled dispatch. Every block waits for that kernel before it exits, so the combine
  completes after the producer.

## Combine launches

A combine is a push, which writes each row's reduced copies into the destination ranks' staging, and a reduce, which
sums the staging rows into the outputs. The push closes with the LSA tail barrier, so the reduce that follows it on
the stream reads rows every team member has finished writing. Signaled or not, a combine is one push of every chunk
and one reduce of the whole row.

- **Head gate.** The push runs it at its start: no peer pushes into this rank's staging before every rank is in the
  combine, so no rank still reduces the previous call's staging.
- **Tail barrier.** Every block fences its stores; the elected last block runs `lsa_grid_tail_barrier` and resets the
  grid counter, the head gate flag and the row claim pool counter.
- **Zero-token ranks.** A rank with no received tokens still runs its push launch, so no peer waits on it at the
  barrier. A rank with no source tokens skips an unsignaled reduce, which has no rows to read; a signaled one still
  launches, so its call count stays in step.
- **Signaled chain.** The push and the reduce of a signaled combine are one programmatic chain:

  ```text
   launch   SMs           entry                      exit
   push     combine SMs   trigger                    tail barrier; the elected block bumps the done word;
                                                     block 0 waits for the kernel before it, the others exit
   reduce   device        count the call, trigger    wait block: waits for the push;
                                                     token blocks: wait for the done word, then reduce
  ```

  After the push's tail barrier, every rank's rows of this push are in this rank's staging. The elected block's thread 0
  then adds a fixed quota to a per-group done word with a system-scope release, and the token blocks of the reduce
  acquire it before they read the staging. The reduce's blocks add one quota between them to a second word, the call
  word, so each block learns its call's index whatever order the blocks run in, and its target value of the done word.
  Both words only grow, so no reset is needed between calls or graph replays.
- **Completion.** Only block 0 of a signaled push waits for the kernel before it, so the other push blocks free their
  SMs as their rows are out and the reduce becomes resident on them. The reduce's wait block completes after the
  push, which completes after that kernel, so the combine still ends after the producer.
- **Next launch.** The reduce triggers at entry, so the next programmatic launch on the stream can overlap the tail of
  the kernel before the push, and the reduce.
- **Outputs.** The reduce writes the combine outputs, the backward weights included, before the kernel before the push
  completes, so that kernel must not access them, and memory it reads or writes must not be freed and reused as
  outputs.
- **Failed reduce.** A push that launched counts its call. If its reduce then fails to launch, the two words stay one
  call apart, so the group refuses every later signaled combine with `ncclInternalError`.

`NCCL_EP_HT_UNFUSED_SYNC` runs the tail sync as a separate kernel instead. Pull dispatch keeps its own barrier. The
barrier is the one described in [HT EM pull dispatch + push combine](../ht-em-pull-push/README.md).

## Implementation

- A signaled push is launched with programmatic stream serialization and triggers its reduce at entry. The reduce, in a
  separate JIT variant (`_pdl`), is one block larger: the last block is the wait block and takes no tokens
  (`kWaitBlocks`). `combine_push_signaled_tail` ends the push, `lsa_grid_tail_barrier` reports the elected block, and
  `combine_reduce_count_call` and `combine_reduce_wait_push` open the reduce. The two words live in the group's device
  counter block.
- The signaled push prefers the max-L1 carveout: it starts beside a max-shared-memory GEMM and would otherwise keep
  that GEMM's minimum L1 split.
- `lsa_grid_tail_barrier` ends the push and `lsa_grid_tail_reset` resets it, as in the unsignaled combine.

## Memory cost

The dispatch signal reserves buffers up front for every group that can carry it, whether or not a call requests one:
each group with expert-major pull-count dispatch and a single LSA team (not `NCCL_EP_HT_UNFUSED_SYNC`) reserves a chunk
table of 40 bytes per source chunk (32 tokens of a source rank), `lsa_team_size * ceil(max_dispatch_tokens_per_rank / 32)`
chunks: about 80 KB at 8 ranks and 8192 tokens. Handles reserve nothing.

A pull-push group with `NCCL_EP_OVERFLOW_DROP` keeps one int32 pushed-row flag per source token and LSA rank in the
combine prob region, `max_dispatch_tokens_per_rank * lsa_team_size` flags: the push marks a row's flag and the reduce
re-arms it. The combine signal itself reserves nothing. The per-call dispatch scratch is group-shared and sized at group
creation; an eager group, which cannot be captured, grows it to a handle's padded rows before a signaled dispatch.

## Testing

On 4 ranks (VR sm_107), container NCCL:

- `test_ht_signal`: the signaled combine bitwise equal to the unsignaled one, forward and backward, and with one
  rank's producer 20 ms late on the last chunk and then on chunk 1 (`LatePeerMatchesUnsignaledBitwise`); a zero-token rank followed by a call with tokens
  (`EmptyRankKeepsCallsInStep`); a programmatic kernel behind every signaled combine that checks the combine completes
  after the kernel before it; every dispatch tile published, with a programmatic consumer reading them as they are
  published; signaled backwards on imported handles of two layers, after plain forwards under new routing, bitwise
  equal to the plain backward. `test_ht_bwd` passes.
- `ep_bench --validate`: the signaled dispatch (tile flags, forward and backward) and the signaled combine, forward
  and backward, DeepSeek-V3 and Qwen3 shapes, and the plain push, forward and backward.
- `compute-sanitizer --tool synccheck` on the combine, microbatch and backward tests of `test_ht_signal`: no errors.
- `TwoMicrobatchSignalTest`: two handles with different routing and receive counts, queued in a 1F1B order with one
  sync (A signaled forward dispatch and combine, then B signaled forward dispatch, A signaled backward dispatch, B
  signaled forward combine, A signaled backward combine). Received rows, flags, combined tokens, gradients and weights
  equal values computed from the routing, and the sequence replays from a graph with fresh payloads. A reduce that
  does not wait for its push fails it.
- `DropStaleStagingTest`: after a plain combine that fills a rank's budget, a signaled two-chunk combine that drops most
  tokens returns zero or the token, never the earlier value, and exactly the budget's rows come back.
- Mutants, each caught by a test: a push without its tail barrier, and a row map rebuild without its row clear.
- Earlier revision, not rerun: `compute-sanitizer --tool racecheck` reported hazards only in `ht_dispatch_pull` (sites
  and disposition not recorded) and none in a combine kernel. The SASS of the BF16 reduces (forward and backward) on sm_100a, sm_103a and sm_107 has no `.CONSTANT` load.
- No test injects a failed launch: the library has no hook for it.

TE MoeOverlap layer on the NCCL EP integration stack, alternating runs on the same nodes:

- The unsplit chain makes the backward with BF16 weight gradients 5-35 us faster at VR EP4 (DeepSeek-V3 BF16 -1.6%,
  Qwen3 BF16 -1.9%) and 16-34 us at VR EP8, and is within 0.4% at GB300 EP4.
