# HT EM Push Combine MXFP8 (MR !343)

Follow-up to [MXFP8 combine](../mxfp8-combine/README.md) for the push combine of
[HT EM pull dispatch + push combine](../ht-em-pull-push/README.md).

## Problem

The MXFP8 combine quantizes in `local_permute_reduce`, which only exists on the
expert-major local-permute path. The pull-push mode (`HtEmMode::kPullPush`) uses
`combine_push` and `combine_reduce` instead, and rejected MXFP8, so it always moved BF16
rows over NVLink. The push is NVLink bound, so halving the row should speed it up.

## Goals and non-goals

Goals:
- Support `quant_recipe = NCCL_EP_COMB_QUANT_MXFP8` in pull-push, forward and backward, with
  the same wire format and numerics as the local-permute path (shared codec in
  `mxfp8_quant.cuh`).
- Work in both push sync modes (fused head/tail sync, and `NCCL_EP_HT_UNFUSED_SYNC=1`).
- Leave the `NONE` push path unchanged.

Non-goals:
- A standalone quantize kernel. Quantization stays fused in the push.
- Chunked rows, FLAT layout, multiple LSA teams.

## Background

Terms used below (see the pull-push PLC for the full flow):

- **Slot**: one received token on this rank. One warp handles one slot.
- **Copy**: the token's row in one local expert's expert-major buffer. A slot has
  `n <= top_k` copies, and the combine sums them into one row.
- **Staging row**: the row on the token's owner rank at `t * team_size + R`. `combine_reduce`
  sums the `team_size` rows of a token.
- **Scale block**: 32 BF16 elements (four int4) sharing one E8M0 scale byte.
- **Packed row**: `[H E4M3 | H/32 E8M0]`, 7392 B at H = 7168 against 14336 B for BF16.

```
 copy 0 ..                      sum (FP32)                one TMA store
 copy n-1  ---------------->  per-warp smem row  ------------------------>  staging row
 (recv_x_em)                  (quantize here)                (NVLink)       (owner rank)
```

## Design

### Quantize in the push

`combine_push` already holds the FP32 sum before it writes the row, so quantization needs no
extra pass. Copy 0 is TMA-loaded into the per-warp smem row and the other copies are read
from global memory and summed in. Each lane owns one scale block per round (a warp covers 32
blocks, so `ceil(H/32/32)` rounds): it sums its block over the copies, takes the amax, derives
the E8M0 scale and writes the E4M3 bytes back into the smem row. No cross-lane amax is needed.

The FP8 bytes overwrite the BF16 bytes in place. Block `b` is BF16 at `[64b, 64b+64)` and FP8
at `[32b, 32b+32)`, which lies inside BF16 block `b/2`. That block is always read in the
same round or an earlier one, and a `__syncwarp` separates the reads from the writes of a
round, so no unread data is overwritten.

```
 smem row   | bf16 b0 | bf16 b1 | bf16 b2 | bf16 b3 | ...        | scale scratch |
 quantized  | f8 b0 | f8 b1 | f8 b2 | f8 b3 | ... | f8 b223 | scales |   (H=7168)
            '-------------- H bytes -------------------------'  H/32 B
```

E8M0 bytes stage in a scratch behind the BF16 row and are copied to `row + H` at the end, so
the packed row is contiguous and one TMA store pushes it. The existing tail drain covers that
store, so both sync modes order the scales before the tail barrier.

### Staging stride

The packed row is pushed at `comb_stage_push_stride_bytes(row_bytes, mxfp8, reserve_prob)`:

| Part | Rule | Origin |
| :--- | :--- | :--- |
| 128 B alignment | round the packed row up to 128 B so each remote store starts on a cache line (7392 B to 7424 B) | this MR |
| Anti-camping | add 256 B if the stride is a multiple of 768 B (see "Anti-camping staging stride" in the [pull-push PLC](../ht-em-pull-push/README.md#push-combine)) | existing |
| Backward weights | add 256 B for the co-located FP32 top-k weights, and another 256 B if that lands on a multiple of 768 B | existing |

Example, H = 7168: forward 7424 B. Backward 7424 + 256 = 7680, a multiple of 768, so 7936 B.
Writer, reader and allocation all use this helper. The allocation is sized for the BF16 row
and the packed row is smaller, so it is unchanged.

### Dequantize in the reduce

`combine_reduce` loads 8 payload bytes and one E8M0 byte per lane (four lanes share a block's
scale) and dequantizes into the existing FP32 accumulate. The output and the weight gather
use the packed stride.

### Integration

`combine_push` and `combine_reduce` take the recipe as a template parameter and the JIT
variant name carries `combine_recipe_name_tag()`. The `ncclEpCombine` gate that rejected
MXFP8 for pull-push is removed. The launcher sizes the block by the fused buffer
(`combine_push_mxfp8_fused_int4`: BF16 row plus scale scratch) and forces the whole-row path.
`local_permute_reduce` keeps its own quantizer (`mxfp8_quantize_tile`) because its data layout
needs a cross-lane amax.

## Alternatives considered

- **Chunked push** (per-chunk load, quantize, store, scales after the last chunk): +12% to
  +29% slower at EP 4, hidden 7168, and +12% at hidden 2048, since each chunk repeats the load,
  store and drain. Rejected, so MXFP8 push ignores `NCCL_EP_HT_COMBINE_PUSH_CHUNK_SIZE`.
- **Shuffle-based amax across four lanes**: regressed at EP >= 8 against the in-lane block.
  In-lane quantization plus the 128 B stride alignment removed the regression.
- **Standalone quantize kernel**: extra read and write of the row. Not provided. To quantize
  in a separate kernel use the local-permute combine.

## Validation

- `test_mxfp8_combine`: push forward and backward against a reference, at hidden 1536 (48
  scale blocks, a full and a partial 32-lane round) over the whole row.
- `test_mxfp8_combine_conversion`: exact-golden suite, also run on the push path.
- `ep_bench -a ht -L em -V -B --ht-flow pull-push --combine-quantization mxfp8`, hidden 1536,
  2048 and 7168, fused and unfused sync: forward combine, backward tokens and backward weights
  pass. Forward uses the same `2.5e-5` threshold as an unquantized combine, weights are
  bit-exact.

CI runs the unit tests and `ep_bench` in the default and unfused-sync modes.

## Performance

GB300, one LSA team per 4 GPUs, `-t 4096`, 32 comm SMs, all shuffle SMs, `-w 200 -i 100`,
best of 3. Total of `CombinePrologue + Combine + CombineEpilogue` in us, top-k 8, 256 experts.

| EP | DS.v3 (H 7168) BF16 | MXFP8 | speedup | Qwen3.6 (H 2048) BF16 | MXFP8 | speedup |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 355 | 342 | 1.04x | 131 | 130 | 1.01x |
| 8 | 557 | 383 | 1.45x | 179 | 150 | 1.19x |
| 16 | 653 | 396 | 1.65x | 213 | 157 | 1.36x |
| 32 | 742 | 478 | 1.55x | 234 | 164 | 1.43x |

The push is NVLink bound at 32 SMs, so MXFP8 helps once a row crosses the fabric more than
once per token (EP >= 8) and is neutral at EP 4. Without the 128 B stride alignment EP 16
loses about 3%.

## Limitations

- Whole-row push only, BF16 tokens, `H % 512 == 0`, single LSA team.
- No unfused quantization in pull-push.
