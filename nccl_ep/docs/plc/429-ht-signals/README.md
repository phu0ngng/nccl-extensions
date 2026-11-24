# HT dispatch and combine readiness signals (MR !429)

On the high-throughput (HT) expert-major (EM) path, dispatch pulls tokens into contiguous rows for each local expert.
The first expert matrix multiplication (FC1) consumes those rows. The second (FC2) produces rows that combine
pushes back to the tokens' owners for reduction. With ordinary stream ordering, each consumer waits for its
producer to finish. This MR adds two optional device signals so each consumer can start as soon as part of its
input is ready.

MR: [nccl-ep: add HT dispatch/combine readiness signals](https://gitlab-master.nvidia.com/nccl/nccl-extensions/-/merge_requests/429)

| Signal   | Who publishes it? | What can the consumer do when one flag is ready?         |
| -------- | ----------------- | -------------------------------------------------------- |
| Dispatch | Pull dispatch     | FC1 can read one tile of consecutive expert-major rows.  |
| Combine  | FC2               | Push combine can read one column range across all rows.  |

Both signals use the same protocol. The caller resets a device array of 32-bit flags to `-1` before the producer starts.
When tile or chunk `i` is ready, the producer publishes `signal[i] = i` with a GPU-scope release store. The consumer
acquires that flag before reading the corresponding data. The array exposes no ready count, so each producer and
consumer tracks its own tile or chunk index. Combine consumes chunks in order; dispatch tiles can complete out of order.

## API

One struct serves both signals. It adds no new function: the signals ride on `ncclEpDispatch` and `ncclEpCombine`.

```text
ncclEpSignal_t {                       // NCCL_EP_SIGNAL_INIT
    size, magic,
    int                 granularity,   // dispatch: rows per tile; combine: hidden elements per chunk
    ncclEpSignalUnit_t  unit,          // AUTO (default) = the op's natural unit; ROWS or COLUMNS
    ncclEpTensor_t*     signal,        // int32 [>= number of flags], reset to -1; producer sets signal[i] = i
}
```

| Signal   | Field                                     | Struct version              | `granularity`                                                    |
| -------- | ----------------------------------------- | --------------------------- | ---------------------------------------------------------------- |
| Dispatch | `ncclEpDispatchOutputs_t::dispatch_signal` | V2 (struct grows 40 to 48 B) | EM rows per tile: a power of two up to 32768 that divides the per-expert alignment |
| Combine  | `ncclEpCombineInputs_t::combine_signal`    | V3 (V2 added `scales`; struct grows 32 to 40 B) | Hidden elements per chunk: a positive multiple of the push tile |

A null signal, or an older shorter outputs or inputs struct, selects the unsignaled path, and a signaled call
compiles a separate JIT variant. The Python bindings do not expose the new fields.

The kernel that consumes a signal can be launched with programmatic stream serialization and start before the op
ends. It must execute `griddepcontrol.wait` before it exits or triggers further work. A signaled forward dispatch
that builds its maps is itself a programmatic dependent of the kernel before it; other dispatches launch plainly.

```text
reset dispatch_signal and combine_signal to -1 (caller-owned)
ncclEpDispatch(..., outputs.dispatch_signal)   -> publishes tile i as signal[i] = i
FC1: for each tile, acquire signal[tile], then read its rows
FC2: write a column chunk for every row, fence, publish combine_signal[chunk]
ncclEpCombine(..., inputs.combine_signal)      -> pushes each chunk once published
```

Signaling supports forward and backward expert-major pull-push within one LSA team (one NVLink domain), with
synchronization inside the communication kernels. Multiple LSA teams return `ncclInvalidUsage` and
`NCCL_EP_HT_UNFUSED_SYNC=1` returns `ncclInvalidArgument`. Combine also requires `NCCL_EP_COMB_QUANT_NONE` and
`send_only == 0`.

## Dispatch tile readiness

Dispatch can trigger FC1 after every dispatch block passes the cross-rank entry barrier. Each warp counts its
rows after storing the tokens, any weights and scales, and any padding. The warp that completes a tile
publishes its flag. **Tiles can finish out of order.** FC1 acquires the flag for each tile it reads and, for
a Tensor Memory Accelerator (TMA) load, also executes `fence.proxy.async`.

```text
Schematic, not measured. One expert's output in tiles T0..T3; each cell is one tile time.

time          | 1   | 2   | 3   | 4   | 5   |
dispatch pull | T0  | T1  | T2  | T3  |     |
flag set      |  T0 |  T1 |  T2 |  T3 |     |   signal[t] = t when tile t is full
FC1 (PDL)     | wait| T0  | T1  | T2  | T3  |   launched at dispatch entry; only T3 runs after dispatch ends
```

The output tensor determines how many flags to **allocate**. The padded receive count determines how many to
**wait for**. Under overflow drop, that count is measured before the drop, so the receive budget also limits
what dispatch can publish.

```text
tile_rows = dispatch_signal.granularity
output_rows = outputs.tokens.sizes[0]
flag_capacity = ceil_div(output_rows, tile_rows)
published_rows = min(recv_total_counter, output_rows)
if overflow_policy == NCCL_EP_OVERFLOW_DROP:
    published_rows = min(published_rows, max_recv_tokens_per_rank)
published_tiles = ceil_div(published_rows, tile_rows)
```

For 128 output rows and 16 rows per tile, allocate eight flags. With a padded count of 96 and a drop budget
of 64, wait for four flags. In a partial final tile, read only rows below `published_rows`. A rank that
receives zero rows waits for none.

Peers may still be pulling from the dispatch buffers after a tile is ready. FC1 must preserve those buffers and
execute its grid-dependency wait before exiting or triggering further work.
[Dispatch signaling contract](../../documentation/api_reference.md#dispatch-tile-completion).

### Consuming the flags

A flag's position is its tile id, and the producer publishes no arrival order. If tile 0 and tile 2 are ready
but tile 1 is not, the array reads `[0, -1, 2, -1, ...]`. The consumer decides which ready tile to run, and keeps
its own progress.

A consumer that runs each expert's tiles in order does this:

```text
cursor[e]  = tiles of expert e already handed out        # consumer-owned
tile       = first_tile(e) + cursor[e]
ready_run  = consecutive tiles from `tile` whose flag is >= 0, within expert e
pick the expert with the largest ready_run (lowest expert on a tie)
run those tiles, then cursor[e] += ready_run
```

With `[0, -1, 2, 3 | 4, -1]` for experts 0 and 1, expert 0 has one ready tile (tile 1 blocks the run) and so does
expert 1, so the consumer takes tile 0 first. Tiles 2 and 3 wait for tile 1. The flags only say that a tile is
ready; the per-expert cursor lives in the consumer.

## Dispatch row order

**Row order** determines which tokens the payload warps pull first. Working on many experts at once can leave
several tiles partially filled, so the signaled pull gives lower expert-major rows priority. The order is built
inside `dispatch_pull` while rows are assigned, without a separate sort kernel.
[Dispatch kernel](../../../device/ht_ep.cuh).

Count mode is the default for eligible handles. Whether ranks exchange counts inside dispatch or in
`ncclEpUpdateHandle` does not change how rows are assigned. Rows are assigned on the fly, in three steps:

1. **Map.** Mapping warps take a prefix sum of the padded expert counts to find each expert's first row. They take
   chunks of 32 source tokens, put each token in the bucket of its lowest local expert, and mark that chunk in the
   block's bucket pool. Experts beyond 64 share a bucket, and so do more of them when the pool would exceed 16 KB of
   shared memory.
2. **Pull.** Payload warps take work from the lowest nonempty bucket. When a warp pulls a token, it claims the next
   row for each matching expert with an atomic increment, writes the token to those rows, and records them in the
   routing map. Pulling starts while later chunks are still being mapped, so tiles fill expert by expert.
3. **Publish.** After storing a row, the warp fences and counts it toward its tile. Each warp publishes its first
   few pulled tokens at once, so the first tiles do not wait for a full batch.

A signaled forward dispatch whose block-local chunk pool would need more than 16 KB of shared memory for one bucket
returns `ncclInvalidArgument`.

```text
slot 4 routes to experts 0 and 1
  -> pull it once, from expert 0's bucket (its lowest expert)
  -> claim row 2 in expert 0 and row 9 in expert 1
  -> write the token to rows 2 and 9 and record them in the routing map
```

One token has two destinations. It is pulled once, from the bucket of its lowest expert, and a row is claimed in
each expert it routes to.

## Combine chunk readiness and stream completion

FC2 publishes a chunk only after it has written and fenced those columns for **every received row**. For
example, with 4096 columns and a chunk width of 2048, FC2 can release columns 0-2047 while it computes columns
2048-4095. All ranks use the same width, a positive multiple of the push tile; any other `granularity` returns
`ncclInvalidArgument`.

```text
Schematic of a combine that does not split, not measured. Two chunks (columns 0-2047 and 2048-4095); each cell is
one chunk time.

time          | 1   | 2   | 3   | 4   |
FC2           | c0  | c1  |     |     |
flag set      |  c0 |  c1 |     |     |   combine_signal[c] = c once every row's columns are fenced
push          | wait| c0  | c1  | bar |   one push kernel; bar = tail barrier
reduce        | wait| wait| wait| rows|   resident from push entry (PDL)
```

Push of chunk 0 hides under FC2's chunk 1. The last chunk's push, the barrier and the reduce stay exposed.

Unless it splits (see **Split** below), signaled combine runs one push for all chunks and one reduction for whole rows.
A push warp acquires the chunk flag, reads those columns with coherent loads, and writes them to peer staging. A block
that finds all flags ready can push a whole row at a time. The push triggers the reduction at entry, so reduction blocks
can start while the push is active. The diagram below shows what they must wait for when FC2 precedes combine on the
same stream.

```text
FC2 chunk flags --> push warps --> tail barrier ---> token blocks: acquire completion,
                                    (all ranks)       reduce whole rows ----------------+
                                        |                                              |
FC2 finishes ------------------> push block 0 exits                                    |
                                        |                                              v
                                        +--> reduction wait block exits ---------> combine completes
```

Reduction can read staging as soon as all ranks have pushed their data. Its dedicated wait block
keeps combine ordered after FC2 even if the token blocks finish earlier. Arrows show dependencies.

Completion and staging reuse:

- Only push block 0 waits at exit for the preceding kernel on the same stream. Other push blocks free their SMs
  for reduction while that kernel is still running.
- The push entry barrier protects staging still in use by the previous call.
- After the tail barrier, the elected push block publishes completion for the current call with a system-scope
  release. Reduction blocks acquire that call's completion before reading.
- Two 32-bit group counters identify the call across launches and graph replays; a split uses a second pair for its
  head chunks. Empty ranks still participate in the barrier and advance the counters.
- If the reduction fails to launch after its push, the group rejects later signaled combines with
  `ncclInternalError`.

**Split.** The chain above serves a row of one chunk. A combine whose row has two or more chunks splits it into four
launches, unless the group uses `NCCL_EP_OVERFLOW_DROP` (only a whole-row reduce re-arms the pushed-row flags):

```text
 launch  work                   SMs                  waits for                   exit
 P1      push chunks 0..C-2     combine SMs          head gate                   tail barrier, then adds to head done
 P2      push chunk C-1         P1's, or env (*)     P1 to complete, at entry    tail barrier, then adds to done
 R1      reduce chunks 0..C-2   shuffle SMs          token blocks: head done     wait block: P2 to complete
 R2      reduce chunk C-1       R1's, or env (*)     token blocks: done          wait block: R1 to complete

 (*) NCCL_EP_HT_COMBINE_PUSH_LAST_CHUNK_SMS, NCCL_EP_HT_COMBINE_REDUCE_LAST_CHUNK_SMS
```

Only the last chunk's push and reduce then stay exposed after FC2. P2 skips the head gate, which P1 passed, and runs on
P1's SMs (`combine_num_sms` or `combine_bwd_num_sms`) unless `NCCL_EP_HT_COMBINE_PUSH_LAST_CHUNK_SMS` is set; R1 runs
beside it on the SMs of an unsplit reduce and also writes the backward weights. R2 runs on R1's SMs unless
`NCCL_EP_HT_COMBINE_REDUCE_LAST_CHUNK_SMS` is set. Both variables are read at group creation and may differ across
ranks. P1 and R1 count on their own pair of done and call words, P2 and R2 on the pair of the unsplit chain. Every rank
passes `combine_signal` to a combine, or none does: a rank that splits runs one more LSA barrier.

What the neighboring kernels must do
([combine signaling contract](../../documentation/api_reference.md#combine-readiness-signaling)):

- The kernel immediately before the push completes its own dependencies before triggering the push. A signaled
  dispatch triggers while it is still writing routing maps, so a kernel that waits for dispatch must sit between
  a signaled dispatch and a signaled combine.
- That predecessor keeps its live buffers disjoint from combine outputs, including backward weights.
- A successor waits before accessing combine buffers, exiting, or triggering more work.
- If FC2 runs on another stream, all its blocks must be resident before the push starts, and both must use the
  same CUDA memory synchronization domain. Buffer reuse then waits for FC2 on its own stream.

## Cost and effect on existing callers

Eligible fixed-budget groups reserve about **6 bytes per receive-budget row**, even if every call is unsignaled: 2
bytes for dispatch tile counters and 4 bytes for the row map that backward dispatch uses. The fixed reservation lets
a caller capture its first fixed-budget dispatch in a CUDA graph. Eager groups allocate the scratch at their first
signaled dispatch, sized to padded rows, and do not support dispatch capture. Pull-count groups also reserve 40 bytes per group of 32 source tokens per rank, or
**80 KiB** for eight ranks at 8192 tokens each. Handles need no signal scratch. Combine chunk flags are
caller-owned. Under overflow drop, the push marks one 32-bit flag per source token and LSA rank so reduction can
skip stale staging rows.

Plain (unsignaled) pull dispatch also changes:

- Handle update clears the count row.
- Forward pull stages unregistered top-k weights before its fused entry barrier.

## Validation

The signal tests ran on four VR ranks (SM 107) with the container's NCCL. They exercised three cases relevant
to early reads and staging reuse. [Signal tests](../../../tests/test_ht_signal.cu).

- **Late data.** FC1-style consumers read tiles as their flags appeared, bounded by the receive total they polled
  from -1, which with the expert offsets matched the plain forward's. Forward and backward combine matched
  unsignaled results bitwise, including cases where one rank delayed its final chunk by 20 ms (`MatchesPlain`).
- **Changing routes.** Signaled backward dispatch followed the maps of its handle's last forward: with two
  interleaved handles, after a routing change, and replayed from a CUDA graph after another handle's backward
  (`BackwardFollowsForwardMaps`). A CUDA graph of a signaled dispatch and combine replayed with matching rows,
  flags and combined tokens (`MatchesPlain`).
- **Staging reuse.** Tests covered a rank that sends no token and overflow drop after a call that populated
  staging. Dropped tokens returned zero. Removing the push tail barrier or the row-map clear made the
  corresponding test fail.

`ep_bench --validate` covered forward and backward DeepSeek-V3 and Qwen3 shapes with combine flags already
published. Compute Sanitizer `synccheck` reported no errors on the signal tests. `racecheck` reports shared-memory
hazards only in `dispatch_pull`, which `main` has as well. Failed reduction launches have no fault-injection coverage.

## Questions for review

- Which tile does the target FC1 schedule consume first? Measuring time to that tile alongside total dispatch
  time would show whether prioritizing low rows improves overlap on that workload.
- Can the target FC2 schedule publish a column range across all rows early enough to overlap the push?
  The whole-row backward benchmark leaves the benefit of finer chunks open.

## TODO

- Try counting rows per expert instead of per tile to reduce the signaling overhead of dispatch. The current
  producer keeps one fill counter per tile; a per-expert counter would need fewer atomic updates, with tile flags
  derived from it.
