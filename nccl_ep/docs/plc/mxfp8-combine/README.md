# MXFP8 Combine — FP8 E4M3 + E8M0 Block Scales

## The constraint this design lifts

Dispatch already has a quantized leg: `NCCL_EP_DISP_QUANT_FWD` (scales-forward)
puts FP8 E4M3 rows plus a block-scale row on the wire. Combine has no
equivalent. Its only non-`NONE` recipe is the experimental NVFP4 path, which
requires the caller to compute an FP32 per-token global scale and hand it in
through `inputs->scales`. So on every step the combine leg moves BF16 — `2H`
bytes per `(token, contributor)` — while the dispatch leg moves roughly `H`.

Two properties of the existing combine path make a fused quantized recipe
cheap rather than an extra pass:

- **A kernel already touches every element on the way in.**
  `local_permute_reduce` reduces a token's `top_k` expert-major copies into one
  FLAT staging row before the combine kernel runs. It reads all the bytes and
  writes a full row. Quantization folded into its epilogue costs no HBM
  round trip, and *shrinks* the row it writes.
- **A warp already works in FP32 on the way out.** The RED warp group widens
  the staged row to `float2` accumulators for the weighted reduction.
  Dequantization is a multiply on a value that was going to be widened anyway.

The remaining design question is where *not* to quantize. The inter-node RDMA
hop is a second transport with its own staging and wire format; rather than
carry a second (and untested) quantized path through it, this recipe is
NVLink-only and a multi-LSA-team group is rejected at the API.

## Requirements

- No API-visible scale tensor. The recipe is opt-in per call and generates its
  own E8M0 scales; `inputs->scales` must stay `NULL`.
- No extra kernel and no extra HBM round trip: quantize in the
  `local_permute_reduce` epilogue, dequantize in the RED accumulator loop.
- Preserve the `NONE` value contract otherwise — FP32 weighted reduction,
  BF16 output.
- One transport only: single LSA team (NVLink). Multi-LSA-team groups are refused, not silently downgraded.
- Do not grow the combine SMEM budget; G2S stage count is what pays for it.
- Every configuration without a quantizing kernel must fail loudly, never
  silently transport BF16 under an MXFP8 recipe.

## Design

**Source:** `nccl_ep/device/quant_recipe.cuh` (recipe traits),
`nccl_ep/device/mxfp8_quant.cuh` (encode/decode),
`nccl_ep/include/cuda_fp_features.h` (toolkit gate, geometry constants),
`nccl_ep/device/ht_ep.cuh` (`local_permute_reduce` epilogue, `mxfp8_red_map`,
RED/G2S/N2N warps), `ht_ep_adapter.{cuh,cu}` + `device/jit/ht_combine_jit.cuh`
(recipe plumbing, JIT variant), `nccl_ep/nccl_ep.cc` (validation, gating,
buffers).

**Scope:** HT only, **single LSA team (NVLink) only**, `EXPERT_MAJOR` layout on
the EM local-permute path (`kLocalPermute`), forward and backward combine.
`hidden % 512 == 0`. CUDA 12.8+. LL, multi-LSA-team (inter-node) groups, the
`kPullPush` push-combine path, and all other EM modes are rejected, not
silently downgraded.

### Recipe traits: geometry as compile-time policy

`combine_recipe_traits<kRecipe>` (`quant_recipe.cuh:28-98`) is the single
place that answers "how many bytes is a row, on which leg". The primary
template is left undefined, so an unhandled recipe is a compile error rather
than a silent default. Kernels take the recipe as a template parameter and ask
the traits; there are no per-recipe branches in hot loops, SMEM layout, or
transport sizing.

| Trait | `NONE` | `MXFP8` |
| :--- | :--- | :--- |
| `kScaleBlock` | 0 | 32 |
| `wire_token_bytes(H)` | `H * sizeof(dtype)` | `H` (1 B/elem) |
| `scale_bytes_per_token(H)` | 0 | `H / 32` |
| `packed_bytes(H)` — NVLink / G2S | `H * sizeof(dtype)` | `H + H/32` |
| `rdma_bytes(H)` — inter-node | same as packed | *not defined* |
| `rdma_scale_bytes(H)` | 0 | *not defined* |

For BF16 in, the intra-node row goes from `2H` to `1.03125H` bytes — a 48.4%
reduction on the NVLink leg. MXFP8 deliberately defines no `rdma_bytes` /
`rdma_scale_bytes`: the RDMA warps are the only code that reads them, so
omitting them makes an inter-node MXFP8 kernel fail to *compile* as well as
being refused at the API.

### Wire format

The scales are not a separate buffer. They ride in the tail of the same row,
which is what keeps the transport, the SMEM staging and the TMA descriptors
single-region for both recipes:

```text
Packed intra-node row (one token, one contributor), H = hidden:

  byte 0                                H              H + H/32
    +-----------------------------------+----------------+
    |  FP8 E4M3 payload, 1 byte/element |  E8M0 scales   |
    |            H bytes                |   H/32 bytes   |
    +-----------------------------------+----------------+
     ^                                   ^
     elements 0..H-1                     one byte per 32-element block

  16-byte aligned  <=>  H % 512 == 0     (H + H/32 = 33H/32; 33H/32 % 16 == 0
                                          iff H % 512 == 0, since gcd(33,32)=1)
```

`H % 512` is therefore not a tuning choice but the alignment condition for the
packed row, enforced at `nccl_ep.cc:579`. It also makes the scale row itself
`>= 16 B`, which the TMA path needs. `NCCL_EP_MXFP8_SCALE_BLOCK` and the
derived `NCCL_EP_MXFP8_HIDDEN_ALIGN = 16 * block` live in
`cuda_fp_features.h:25-30` so host validation and device kernels cannot drift.

### Quantize: fused into the local permute reduce

`local_permute_reduce` (`ht_ep.cuh:6052`) runs 128 threads per token slot.
Thread `t` owns `int4` indices `j = t + nn*128`; one `int4` is 8 BF16 elements
= 4 `float2` accumulators. An MXFP8 block is 32 elements = **4 consecutive
`int4`s = 4 consecutive lanes**, so the block amax is two butterfly shuffles,
with no smem and no `__syncthreads` (`ht_ep.cuh:6169-6195`):

```text
One MXFP8 block = 32 elements = 4 consecutive int4s = lanes L..L+3
b = block id = j >> 2

  lane        L          L+1         L+2         L+3
  j          4b         4b+1        4b+2        4b+3
  elements [0..7]     [8..15]     [16..23]    [24..31]
             |           |           |           |
             +--- __shfl_xor_sync(amax, 1) ------+
             +--- __shfl_xor_sync(amax, 2) ------+
                            |
                  e = float_to_e8m0(amax)      ceil-log2 of amax/448
                            |
        lane with (j & 3) == 0  ->  dst_scale[b] = e      (one byte)
                            |
        every lane: scale_inv = e8m0_to_scale_inv(e)
                    4 x pack_e4m3x2_scaled(acc[p], scale_inv)  ->  uint2
                    store 8 B at out_row + j*8
```

All lanes run the shuffles whether or not their `int4` is in range, so the warp
stays converged. The `u` loop is quantized one iteration at a time rather than
vectorized over `kHiddenVec`, so the `e8m0`/`out` temporaries do not multiply
by the unroll factor — the kernel is budgeted at 64 registers per thread at
`blocks_per_sm == 1`.

The output stride switches with the recipe
(`kOutRowBytes = (kCombineRecipe == NCCL_EP_COMB_QUANT_MXFP8) ? kPackedRowBytes : kInRowBytes`,
`ht_ep.cuh:6093`), with four `static_assert`s pinning the packed geometry,
its 16-byte alignment, and the `4 BF16 pairs per int4` assumption the `uint2`
epilogue makes.

Because the input read is unchanged and the output write drops from `2H` to
`1.03H`, adding quantization makes this kernel slightly *faster*, not slower —
see the `comb_pro` column under Impact.

### Dequantize: the RED accumulator map

The `NONE` RED loop assigns element `n*THRDS + tid` to accumulator `n`. Under
that strided map each accumulator is an isolated pair in a different E8M0
block, so MXFP8 would pay one scale load and one broadcast per accumulator.
`mxfp8_red_map<NUM_ACC>` (`ht_ep.cuh:2499`) switches to contiguous ownership so
one wide load shares one scale read:

```text
kPairs = widest power of two dividing NUM_ACC, capped at 8 (= one 16 B load)
vec_base(n) = (n / kPairs) * (kPairs * THRDS) + tid * kPairs

Strided (NONE):        acc n  <- pair  n*THRDS + tid
   tid0 [p0]      [p128]     [p256]   ...      one scale read per acc

Contiguous (MXFP8):    acc n  <- pair  vec_base(n) + (n % kPairs)
   tid0 [p0 p1 p2 p3 p4 p5 p6 p7]  tid1 [p8 .. p15]  tid2 [p16 .. p23] ...
        \_______ one 16 B ld ______/                  \___ next block ___/
        \_____ scale_data[pair0/16] ______/  16 pairs = 32 elements = 1 block
```

`kLanesPerScale = 16 / kPairs` (2 at `kPairs == 8`) is how many threads share a
block scale. The `static_assert(16 % kPairs == 0)` is what guarantees a vector
never straddles two scale blocks — otherwise a single wide load would need two
scales.

Two consequences that are easy to get wrong and are pinned in the code:

- **The store must mirror the load.** `combine_store_reduced_token`
  (`ht_ep.cuh:2748`) selects `Map::pair_id(...)` under MXFP8 and the strided
  index otherwise. A strided store over contiguously-filled accumulators would
  permute the token — a silent, plausible-looking corruption.
- **The tail must not wide-load into the scale bytes.** The FP8 payload and the
  E8M0 tail are one contiguous region, so a full-width load on the last vector
  would read scales as data. The loop takes a per-pair path whenever
  `pair0 + kPairs` exceeds the payload (`ht_ep.cuh:2648-2662`).

### Transport: NVLink only

```text
   expert output (BF16, EM layout)
             |
             |  local_permute_reduce: reduce top_k EM copies, quantize
             v
   FLAT staging  [ FP8 H | E8M0 H/32 ]      <-- 1.03H bytes
             |
             |  intra-LSA (NVLink), G2S stage = packed_bytes
             v
   RED warp: dequant fp8*2^(E-127), FP32 weighted accumulate
             |
             v
   S2G: BF16 attn_output
```

There is no second leg. `validateCombineRecipe` refuses the recipe when
`is_internode_available(group)` (i.e. `rdma_team_size > 1`), so the cross-LSA
G2S ring, the `gin_RED`/`gin_G2S` regions and the N2N put are never reached
with an MXFP8 row. The traits reinforce that structurally: the RDMA warps are
the only consumers of `rdma_bytes` / `rdma_scale_bytes`, they are instantiated
solely under `if constexpr (LSA_TEAMS != 1)`, and MXFP8 defines neither — so
wiring MXFP8 into the inter-node path is a compile error, not a silent BF16
send.

### SMEM staging

With a single LSA team only the cross-LSA ring is allocated, and it carries the
packed width (`ht_ep.cuh:770-786`, `938-951`). The multi-team column is what a
`NONE` group gets; MXFP8 never reaches it:

| Ring | single LSA team | multi LSA team |
| :--- | :--- | :--- |
| `cross_lsa_token_G2S` | `packed` (`1.03H`) | `max(packed, 2H)` = `2H` |
| `lsa_token_G2S` | not allocated | `packed` (`1.03H`) |
| `token_S2G` | output dtype (`2H`) | output dtype (`2H`) |

So per G2S stage the token SMEM goes `2H -> 1.03H`. Strides are rounded up to 128 B for TMA;
`H + H/32` is not naturally 128 B-aligned (`H=7168 -> 7392`), which is why the
rounding is explicit in both the cost model and the layout builder rather than
assumed.

`NCCL_EP_COMBINE_MAX_ACC` (default 64, `ht_ep.cuh:25`) is added as a build
knob for the RED register budget, propagated to the four flag sets and into
the JIT `-D` flags (`Makefile:325-338`, `nvcc_compiler.cc:209-211`).

### Host contract and gating

`validateCombineRecipe` (`nccl_ep.cc:554-582`) rejects, in order: LL algorithm,
a pre-12.8 toolkit, non-2D tokens, `hidden % 512 != 0`, non-BF16 output, a
non-NULL `inputs->scales`, and non-BF16 tokens. Layout is not known there, so
the path check lives in `ncclEpCombine` (`nccl_ep.cc:5098-5112`) and covers
both branches of the push/pull split:

```cpp
if (mxfp8_combine && (!em_permute_combine || comb_push)) { ... return ncclInvalidArgument; }
```

`kPullPush` is the case worth calling out: it satisfies `em_permute_combine`,
but routes through `combine_push` / `combine_reduce_stage`, neither of which
quantizes. Without the `|| comb_push` term the call would set
`params.combine_recipe = MXFP8` and then transport BF16 — wrong numbers, no
error.

Both JIT variant names carry `_mxfp8` from the shared
`combine_recipe_name_tag()` (`jit_source_literals.hpp:128-139`), so an MXFP8 kernel can never
alias a cached `NONE` kernel. `launch_combine_reduce` takes `hidden` and the
input row width and derives the packed width forwards via
`combine_recipe_packed_row_bytes()` (`ht_ep_adapter.cu:1894-1896`), so no
launcher has to invert the packed geometry.

### The E8M0 reciprocal at the top of the range

`e8m0_to_scale_inv` (`mxfp8_quant.cuh:72-81`) is worth its comment block. The
obvious implementation bit-casts the exponent field:

```cpp
return __uint_as_float((254u - e) << 23);   // wrong at e = 0xFE
```

`float_to_e8m0` caps at `0xFE` to avoid the `0xFF` E8M0 NaN code, and at that
cap the exact value is `2^-127` — not a normal FP32 (the smallest normal is
`2^-126`) but the subnormal `0x00400000`. The bit-cast yields
`biased_exp = 0, mantissa = 0`, which reads as `+0.0`. On an Inf-carrying block
that produced:

```text
amax = Inf  ->  e = 0xFE  ->  scale_inv = 0.0  ->  Inf * 0 = NaN
            ->  cvt.rn.satfinite.e4m3x2.f32(NaN) = 0x7F (E4M3 NaN)
            ->  dequant(0x7F, 2^127) = NaN          (instead of Inf)
```

The fix defers to the runtime `__nv_fp8_e8m0 -> float` cast, the same pattern
as TransformerEngine's `reciprocal_scale`. That type is CUDA 12.8+, which is
where `host_build_supports_mxfp8()` (`common.hpp:132`) comes from; the
software branch under `#else` exists only so the header still compiles on
older toolkits, and the API rejects MXFP8 on those builds so it is never
reached.

### Buffer model

| Buffer | `NONE` | `MXFP8` | Change |
| :--- | :--- | :--- | :--- |
| `expert_input_token` (FLAT staging) | `2H` per slot | `H + H/32` per slot | Holds the packed record; allocation unchanged (sized for the `2H` worst case) |
| G2S token stage (intra) | `2H` | `H + H/32` | 48% smaller per stage |
| `gin_RED` / `gin_G2S` token regions | `2H` | n/a | Inter-node only; MXFP8 groups are single-team |

No separate scale buffer is allocated: the E8M0 bytes live in the tail of the
packed row, so the recipe adds no IPC region to the group.

## Impact

Lyris GB200, `NCCL_EP_COMM_SMS=32`, NCCL `2.31.0a13+cuda13.2`, `--segment=nodes`,
`--exclusive`. Node counts 2/4/8/16, EP = nodes x 4. Configs: `nemotron`
(experts 512, top-k 10, `H=3072`, 8192 tokens/rank) and `dsv3` (experts 256,
top-k 8, `H=7168`, 4096 tokens/rank).

**Attribution caveat.** `--mxfp8` is an end-to-end switch: it turns on
MXFP8-shaped dispatch `SCALES_FORWARD` *and* the MXFP8 combine recipe, so the
deltas below mix this branch's work with the pre-existing FP8 dispatch path.
Use `--combine-quantization mxfp8` on its own to isolate the combine leg. The table below isolates the combine side: `comb_k` is the combine
kernel, `comb_pro` is `local_permute_reduce`. Dispatch columns are omitted
because nothing in this branch touches them.

| Config | Nodes | `comb_pro` base -> mxfp8 | `comb_k` base -> mxfp8 | Combine total | Δ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| nemotron | 2 | 172.3 -> 169.1 | 450.1 -> 395.5 | 622.4 -> 564.6 | **-9.3%** |
| nemotron | 4 | 188.2 -> 183.3 | 611.4 -> 512.5 | 799.6 -> 695.8 | **-13.0%** |
| nemotron | 8 | 193.6 -> 187.8 | 735.8 -> 581.6 | 929.4 -> 769.4 | **-17.2%** |
| nemotron | 16 | 194.8 -> 190.9 | 780.5 -> 636.6 | 975.3 -> 827.5 | **-15.2%** |
| dsv3 | 2 | 151.0 -> 149.4 | 437.9 -> 251.7 | 588.9 -> 401.1 | **-31.9%** |
| dsv3 | 4 | 158.2 -> 154.7 | 574.6 -> 318.9 | 732.8 -> 473.6 | **-35.4%** |
| dsv3 | 8 | 159.2 -> 155.7 | 662.7 -> 370.7 | 821.9 -> 526.4 | **-36.0%** |
| dsv3 | 16 | 157.4 -> 153.9 | 725.3 -> 404.3 | 882.7 -> 558.2 | **-36.8%** |

All figures are microseconds, kernel means.

Two things the numbers say about the design:

- **`dsv3` gains roughly twice what `nemotron` does** (-36% vs -15% at 16
  nodes). The recipe removes bytes, not instructions, so the win tracks how
  much of the kernel is row transport: `H=7168` moves 2.3x the bytes per row
  of `H=3072`, against per-token overheads that do not scale with `H`.
- **`comb_pro` gets marginally faster (-1% to -3%) despite doing the
  quantization.** `local_permute_reduce` reads the same `top_k` BF16 EM copies
  either way and writes `1.03H` instead of `2H`. The shuffles and the E4M3
  conversion are absorbed by the shorter store. This is the payoff of fusing
  into a kernel that was already there rather than adding a quantize pass.

Combine total does not fall by the full 48% of the wire reduction because the
S2G output leg stays BF16 in every recipe, and per-token overheads do not scale
with the row.

### Combine prologue micro-sweep

Isolating the `local_permute_reduce` epilogue change of "hoist the E8M0 decode
out of the per-element quantize" (`v10` -> `v11`), against the BF16 baseline:

| Config | Nodes | baseline | v10 | v11 | v11 vs v10 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| nemotron | 2 | 171.9 | 163.1 | 161.5 | -0.9% |
| nemotron | 4 | 190.6 | 177.9 | 173.4 | -2.5% |
| nemotron | 8 | 192.0 | 180.5 | 179.2 | -0.7% |
| nemotron | 16 | 192.9 | 183.8 | 177.4 | -3.5% |
| dsv3 | 2 | 150.5 | 143.7 | 138.5 | -3.7% |
| dsv3 | 4 | 156.2 | 147.4 | 142.3 | -3.4% |
| dsv3 | 8 | 158.6 | 149.7 | 144.5 | -3.4% |
| dsv3 | 16 | 156.9 | 149.6 | 141.9 | -5.1% |

Mean `v11` vs `v10`: -2.9%.

## Validation

Two suites, both HT, `EXPERT_MAJOR`, run on 1 x GB200 (4 GPUs, `sm_100`):

- **`tests/test_mxfp8_combine.cu`** (5 tests) — API contract and two
  end-to-end smoke cases: expert-major combine matches the BF16 reference
  forward and backward (the backward case also pins that the FP32 weight
  gradients round-trip bit-exactly through a quantized token path),
  flat layout rejected, non-NULL `inputs->scales` rejected, and the wire
  geometry check that packed bytes are `H` and scale bytes are `H/32`.
- **`tests/test_mxfp8_combine_conversion.cu`** (16 tests) — the quantizer itself. The
  first suite cannot catch a quantizer bug: its data is piecewise constant and
  lands on the E4M3 grid, so it round-trips bit-exactly. `ep_bench --validate`
  does compare against a predicted round trip (below), but only over whatever
  blocks its pattern happens to produce. The conversion suite builds an independent host model of
  MXFP8 *from the format definition* (using `frexp` to keep exact powers of two
  exact and `nearbyint` for round-half-to-even), pins that model against
  hand-computed goldens, and only then uses it to judge device round trips.
  Coverage: the 17..48 rounding table, scale invariance, the underflow floor,
  in-block dynamic range, sign symmetry, distinct scales for adjacent blocks,
  single/multi-contributor accumulation local and remote, cross-rank vs
  same-rank error contract, and non-finite propagation.

`ep_bench --combine-quantization mxfp8 --validate` checks the combine primitive on its own, the way
the NVFP4 dispatch check does: it predicts the quantization instead of widening
the tolerance, so MXFP8 is held to the same `kCombineHTThreshold = 2.5e-5` as an
unquantized combine (`ep_bench.cu:1055`).

The combine input is the dispatch output, as it is for an unquantized run, so
every slot holds the source token's row and the expectation stays the upstream
one. A quantized dispatch recipe breaks the staging copy -- those rows are not
BF16 -- but not the identification: the bench writes `(src_rank, token)` into
bytes 0..2 of every scales-forward row (`scalesForwardTokenByte`,
`ep_bench.cu:987`) and the dispatch recipes move them untouched. For a quantized
dispatch the bench therefore reads back those three bytes per slot and rebuilds
that token's BF16 row, which keeps the combine check end-to-end -- a token
delivered to the wrong slot still shows up as a wrong sum.

The reference reproduces the data path rather than the arithmetic result: for
each token, one FP32 row per contributing source rank is built as `orig x
(experts of this token on that rank)` -- what `local_permute_reduce` produces --
each row is round-tripped through `mxfp8ValidationRoundTripKernel`
(`ep_bench.cu:2970`) -- written from the format definition, not by calling
`nccl_ep::mxfp8`, so a bug in the production quantizer cannot hide behind it --
and the dequantized rows are summed.

Backward combine takes the same recipe -- nothing in the host gating or the
kernel templates ties MXFP8 to the forward pass -- and reduces gradients through
the same per-source-rank grouping, so `ep_bench --backward
--combine-quantization mxfp8 --validate` checks it against the same predicted
round trip. Whether gradients *should* be carried at E4M3 is a modelling
question this recipe does not answer.

The harness is 4 GPUs in one NVLink domain, which is the only topology the
recipe supports, so "remote" contributions are carried over NVLink by design
rather than as a gap. Not covered: blocks whose
amax is small enough to drive the E8M0 byte to 0 with a non-zero payload
(`amax <= 448 * 2^-127`), unreachable from realistic expert outputs.

## Open items

- **`combine_recipe_literal` now defaults instead of failing.**
  `jit_source_literals.hpp:107-117` changed the `default:` arm from `nullptr`
  to `"NCCL_EP_COMB_QUANT_NONE"`. That makes the `recipe_literal == nullptr`
  rejection in `launch_ll_combine` (`ll_combine_jit.cuh:88-91`) dead code: a
  future unrecognized recipe now compiles a `NONE` kernel rather than being
  refused. MXFP8 itself is safe (it has an explicit case, and LL rejects it in
  `validateCombineRecipe`), but the safety net for the next recipe is gone.
- **Inter-node unsupported.** A multi-LSA-team group is refused at the API.
  Extending the recipe across the RDMA hop means choosing between a second
  lossy conversion (quantize the hop too) and a widened cross-LSA G2S ring
  (send BF16); neither is implemented or measured, and this design forecloses
  neither.
- **LL unsupported.** No `combine_recipe_traits` specialization for the LL
  kernels; rejected at validation.
- **`kPullPush` unsupported.** Push combine has no quantizing kernel; rejected
  explicitly. Folding the quantize epilogue into `combine_push` is the natural
  follow-up, since that path already does the local reduce.
- **`hidden % 512`.** Follows from 16-byte alignment of `H + H/32`. Both focus
  configs satisfy it (`3072 = 6*512`, `7168 = 14*512`); a hidden size that does
  not is refused rather than padded.
