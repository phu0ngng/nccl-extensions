# HT Push Dispatch Tail Redistribution (MR !347)

## Problem

HT push dispatch assigns token chunks to communication CTAs round-robin. If the last wave contains only a few chunks, only a few CTAs process that wave while the others have no more payload work. Adaptive chunk selection reduces this imbalance for some batches, but does not eliminate it above a complete wave.

This change redistributes the final partial wave into 16-token work units across the existing CTAs. Complete waves, metadata chunk sizes, buffer layouts and the adaptive chunk selector remain unchanged. The scope is HT push dispatch (`PUSH_PULL`, paired with pull combine), in both forward and backward directions. Pull dispatch (`PULL_PUSH`), combine and LL are unchanged.

## Decision

Keep two separate concepts:

| Term | Meaning |
| --- | --- |
| Metadata chunk, `C` tokens | The chunk used when UpdateHandle produces routing counts. Its identity and counts do not change when dispatch redistributes work. |
| Scheduling unit | A contiguous source-token interval assigned to one CTA and one payload pipeline. Normally one metadata chunk, but only 16 tokens in the redistributed tail. |
| CTA wave | One scheduling unit per CTA across a grid of `G` CTAs. This describes ownership, not a grid synchronization point. |
| FLAT slot | A row in a destination rank's rank-major receive staging buffer. A token gets one row per destination rank, even if it hits several experts on that rank. |

For `T` actual source tokens, preserve the complete coarse waves:

```text
bulk_chunks = floor(T / (G * C)) * G
bulk_tokens = bulk_chunks * C
tail_tokens = T - bulk_tokens

Coarse:        ceil(T / C) units, up to C tokens each
Redistributed: bulk_chunks units of C tokens
               followed by ceil(tail_tokens / 16) units of up to 16 tokens
```

Redistribution requires a complete wave followed by additional tokens. A batch smaller than or equal to one complete wave retains coarse scheduling.

### Example: 4608 tokens, 32 CTAs, chunk 128

```text
                         Complete wave                Partial wave
Source tokens:           [0, 4096)                    [4096, 4608)

Coarse scheduling:       CTA 0 ... CTA 31              CTA 0 ... CTA 3
                         128 tokens each              128 tokens each

Redistributed:           CTA 0 ... CTA 31              CTA 0 ... CTA 31
                         128 tokens each               16 tokens each
```

4608 is divisible by 128, so all 36 metadata chunks are complete. Nevertheless, 36 chunks do not fill two waves of 32 CTAs. The scheduling tail is the last four metadata chunks, not necessarily an incomplete metadata chunk.

The change reduces the maximum assigned token count from 256 to 144 per CTA in this example. It does not guarantee equal elapsed time: routing density, peer progress and transfer backpressure can still differ.

## Scope and specialization

The enum is `HtDispatchSchedule::{Coarse, Redistributed}`. These are work-distribution policies, not different chunk-size selectors.

`ht_dispatch_schedule(T, C, G, lsa_teams)` selects `Redistributed` when all conditions hold:

```text
one LSA team
16 <= G <= 64
C is one of 32, 64, 128, 256
T > G * C
T % (G * C) != 0
```

Otherwise it selects `Coarse`, including zero tokens, batches with no complete wave, exact complete waves, unsupported grids/chunks and multiple-LSA-team configurations. One LSA team can span physical nodes connected through NVLink. It does not mean one physical node.

The dispatch launcher resolves the schedule after the chunk has been selected, emits the corresponding template argument into the JIT source, and caches distinct `_coarse` and `_tail` variants. The template parameter propagates to the work iterator, MAP builders and payload producer/consumer warps. It removes redistribution indexing from the coarse specialization. This is a measured performance consideration, not only a code-organization choice.

`dispatch_work_t` precomputes the bulk boundary, unit count and final-unit size in its constructor. The payload loop body remains shared rather than duplicated into separate bulk and tail loops.

For single-LSA dispatch, the work bound is the handle's actual local source-token count, including under `Coarse`. Capacity still determines allocation sizes, buffer strides and address bounds. Ranks with different batches can therefore have different unit counts without changing one another's buffer layout. Multiple-LSA-team dispatch retains its existing capacity-based domain and coarse scheduling.

## Design

### Shared unit numbering for mapping and payload

The global unit index determines CTA ownership. Its index within that CTA determines payload-pipeline and MAP-warp ownership:

```text
global unit u -> CTA (u % G)
CTA-local unit q = floor(u / G)

payload pipeline = q % number_of_pipelines
s2d MAP warp     = q % number_of_MAP_warps
```

Every participant uses the same `dispatch_work_t` interpretation:

| Query | Coarse | Redistributed |
| --- | --- | --- |
| `first(u)` | `u * C` | Unchanged for bulk, then `bulk_tokens + (u - bulk_chunks) * 16` |
| `size(u)` | Up to `C` valid tokens | `C` for bulk, up to 16 for tail |
| `metadata_chunk(u)` | `u` | `floor(first(u) / C)` |
| Scan-map prefetch address | Chunk index with stride `C` | First-token index with stride 1 |

For the 4608-token example, the two-pipeline configuration gives each CTA its bulk chunk as local unit 0 on pipeline 0 and its 16-token tail as local unit 1 on pipeline 1. Each pipeline has a separate G2S producer warp and S2G consumer warp. Pipeline 1 need not wait for pipeline 0 to finish. There is no new barrier between the logical bulk and tail waves.

Forward count mode also has two MAP warps that build the routing addresses. With two payload pipelines, that is six warps per CTA. Backward reuses the forward maps and does not launch those count-map builders. Scan mode consumes maps prepared before dispatch instead.

### Existing metadata and receive-slot assignment

The [push count-exchange PLC](../ht-em-dispatch-push-count-exchange/README.md) describes the underlying protocol. The relevant metadata is:

| Metadata | Purpose |
| --- | --- |
| Count rows, `rank_counts[source][destination]` | Reserve disjoint FLAT receive ranges for source ranks. |
| Per-chunk rank counts, `chunk_counts[chunk][destination]` | Count source tokens routed to each rank in each original metadata chunk. |
| `token_dst_rank_bitmap[token]` | Identify the destination ranks hit by an individual source token. Multiple experts on the same rank are deduplicated. |
| `s2d[token][destination]` | Store the destination FLAT slot for that token, or `-1` when no row should be written. |
| LERM | Identify the destination-local experts for each received FLAT row. Used by the receiver's expert-major relocation. |

The MAP warps build `s2d` before the corresponding payload can use it. This is address construction, not token transfer. S2G later uses the resulting slot to push the token into the destination's FLAT staging buffer.

For source rank `s`, destination rank `r`, metadata chunk `m` and source token `t`, the slot is:

```text
slot(s, t, r) = earlier-source prefix
              + earlier-metadata-chunk prefix
              + routed-token prefix within metadata chunk m

earlier-source prefix:
    sum of rank_counts[source][r] for source < s

earlier-metadata-chunk prefix:
    sum of chunk_counts[chunk][r] for chunk < m

within-chunk prefix:
    number of tokens routed to r in [m * C, t)
```

These prefixes are per destination rank. A token routed to ranks 0 and 2 can have a different slot number at each destination. MAP needs offsets for every destination rank the source can write, not only the source's local rank.

### New mapping step: start a unit inside a metadata chunk

Previously one CTA owned a metadata chunk. `dispatch_push_map_chunk_slot_base` initialized `run[r]` from the first two prefixes. The MAP warp then walked the whole chunk in source-token order, advancing the cursor after each warp-sized group of tokens.

Redistribution lets multiple CTAs own different parts of that same metadata chunk:

```text
Metadata chunk 32: source tokens [4096, 4224)

Coarse:
  CTA 0 owns [4096, 4224)

Redistributed, 16-token units:
  CTA 0 owns [4096, 4112)
  CTA 1 owns [4112, 4128)
  ...
  CTA 7 owns [4208, 4224)

Metadata chunk 33 starts at 4224 and is split across CTAs 8 ... 15.
```

All eight CTAs start with the same earlier-source and earlier-chunk prefixes for metadata chunk 32. They cannot all write at that base. Each must additionally skip the routed tokens before its own unit start.

`dispatch_push_map_build_s2d` now does the following for each owned unit:

1. Compute `t0 = work.first(u)` and `m = work.metadata_chunk(u)`.
2. Initialize `run[r]` with the existing earlier-source and earlier-chunk prefixes.
3. For the redistributed specialization, inspect destination-rank bitmap entries in `[m * C, t0)` and add their per-rank hit counts to `run[r]`.
4. Assign slots for the unit's own tokens using that corrected cursor.
5. Publish readiness for this scheduling unit, not for the original metadata chunk as a whole.

Step 3 is the new prefix correction. It is empty for a bulk unit or the first subunit of a metadata chunk because `t0 == m * C`. It is needed for later tail subunits, even if the original metadata chunk itself is full.

### Four-rank example: what `run[r]` contains

Consider source rank 2 sending to four destination ranks. Keep `C = 128` and examine CTA 1's unit starting at token 4112, inside metadata chunk 32. The numbers below are illustrative routing counts.

```text
Destination rank r:                      0    1    2    3

Slots reserved by earlier sources:       5    4    6    3
Hits in this source's chunks 0 ... 31:   7    8    4    5
                                      ------------------
run[r] at metadata-chunk start:         12   12   10    8

Hits in tokens [4096, 4112):             8   12    4    4
                                      ------------------
run[r] at CTA 1's unit start:           20   24   14   12
```

Suppose the first four tokens of this unit route as follows:

| Source token | Destination ranks | Resulting `s2d` slots at ranks `[0, 1, 2, 3]` |
| --- | --- | --- |
| 4112 | 0, 2 | `[20, -1, 14, -1]` |
| 4113 | 1 | `[-1, 24, -1, -1]` |
| 4114 | 0, 1, 3 | `[21, 25, -1, 12]` |
| 4115 | 2 | `[-1, -1, 15, -1]` |

Without the new prefix correction, token 4112 would start at slots 12 and 10 and overlap slots belonging to CTA 0's preceding subunit.

`run[r]` is local cursor state in each MAP warp, not a shared counter updated by earlier CTAs. CTA 1 derives its offset from immutable counts and routing bits. It does not wait for CTA 0 or observe how far CTA 0 has progressed. Ownership follows token order, not CTA execution order.

### How ballot and popcount implement the prefixes

Each lane examines one token in a group of `kWarpSize` tokens. For a particular destination rank:

```cpp
const uint32_t rank_hit_mask = __ballot_sync(0xffffffffu, rank_slice.hit(hits, r));
run[r] += __popc(rank_hit_mask);
```

The ballot collects one destination-hit bit per lane. Popcount counts how many tokens hit `r`. During the prefix correction, lanes outside `[m * C, t0)` contribute zero, so a 16-token prefix counts only its 16 valid tokens even though all 32 lanes participate.

For the unit's own tokens, the same ballot pattern produces a lane-local prefix:

```text
slot for a hit in lane L = run[r] + popcount(hit bits in lanes before L)
run[r] for the next token group += popcount(all hit bits)
```

Together, the unit-start correction and the lane-local prefix reproduce the coarse algorithm's FLAT slots. The metadata chunks and their stored counts have not been repartitioned. Only the work ownership changes.

### Readiness, scan mode and backward

Forward count mode stripes each CTA's units across its MAP warps and publishes per-warp progress counters. G2S and S2G use the same CTA-local unit numbering when checking readiness. MAP orders the `s2d` stores with the existing GPU/proxy fences before releasing its counter. Consumers acquire readiness before consuming the protected mapping. Renumbering only the payload units would therefore be incorrect.

The LERM builder also walks the shared work schedule and uses the saved `s2d` slot for each token. Expert membership, DROP/TRAP overflow checks, receive capacities and the completion protocol remain unchanged.

Scan mode does not rebuild `s2d` inside dispatch. Its S2G map prefetch must nevertheless read the selected subrange, not assume every unit starts at `u * C`. `map_index()` and `kMapStride` retain chunk addressing for coarse scheduling and use token addressing for redistribution. The existing staging footprint remains capacity-based.

Backward dispatch uses the forward-populated maps with the same token-to-slot meaning. It can redistribute its payload work without rebuilding those maps. Thus the new prefix correction runs in forward count-mode MAP construction, not in backward payload writing.

## Cost and performance

Redistribution adds unit-index arithmetic and, in count mode, a bitmap prefix recount for each subunit that starts inside a metadata chunk. That recount examines fewer than `C` preceding tokens per unit. It buys more balanced tail ownership, not less total payload traffic. It can also increase simultaneously active senders and alter the split between payload pipelines. Small or already balanced batches need not improve.

Published MR measurements compare baseline `1ce344e` against candidate replay `c7b6697`: eager, staged HT push dispatch, EP8 in one NVLink domain, BF16, H7168/E512/K8, expert-major local permutation, 10 warmups and 50 timed iterations. The following are rank-mean full dispatch latencies in microseconds, not isolated main-kernel timings or model E2E. Positive reduction means faster.

| GPU / CTAs / tokens per rank | Forward baseline → candidate | Backward baseline → candidate |
| --- | ---: | ---: |
| VR200 / 32 / 4608 | 622.71 → 536.22 (+13.89%) | 575.38 → 486.04 (+15.53%) |
| GB200 / 32 / 4608 | 771.16 → 703.28 (+8.80%) | 718.36 → 648.04 (+9.79%) |
| VR200 / 32 / 8192, complete waves | 867.54 → 868.65 (−0.13%) | 802.25 → 808.65 (−0.80%) |
| GB200 / 32 / 8192, complete waves | 1176.12 → 1177.75 (−0.14%) | 1110.97 → 1111.27 (−0.03%) |

The MR retains the wider matrix, negative results and measurement records. These results motivate preserving the coarse specialization and complete-wave path rather than claiming a uniform improvement for every batch.

## Validation

Existing tests are extended:

| Test / area | Coverage relevant to this change |
| --- | --- |
| `test_ht_chunk_config` | Coarse versus redistributed selection, actual work versus capacity, supported grid/chunk bounds and safe arithmetic. |
| `test_ht_chunk_handle` | FLAT, fused/unfused expert-major count paths, unequal/zero batches, oversized capacities, two prepared handles, forward/backward and imported/relocated handles. Capacities include 193, 2304, 3072, 4096, 4608, 6144, 8191, 8192, 9216 and 16384. |
| Broader regressions recorded in the MR | Count/scan, overflow, zero-copy/scales, pull-dispatch fallback and forced two-LSA backward checks on VR200 and GB200, with existing skips retained. |

The MR records the tested revisions and separates broad regression coverage from later performance-only measurements.
