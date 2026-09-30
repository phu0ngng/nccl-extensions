# Low Latency LSA FP8 Dispatch Optimization

## Summary

This design optimizes the existing fused DS-FP8 dispatch operation for
NVLink-reachable ranks using the low-latency (LL), rank-major LSA kernel.
Quantize each token once into CTA shared memory, reuse that result for every
destination rank, and reduce the block size when the routing and receive
geometry permit. Copy the staged FP8 payload with 128-bit vectors.

The recommendation is to integrate this specialization into the existing LSA
kernel rather than maintain a standalone kernel. Routing, peer addressing,
completion, epoch handling, and output formats retain their existing contracts.
There is no new public API, quantization recipe, or user configuration option.

## Problem and scope

Main already contains an optimized LSA dispatch family for unquantized tokens.
The branch adds the existing DS-FP8 recipe to that family. Its initial
implementation quantizes the same BF16 token separately for each distinct
destination rank. Deduplication already prevents duplicate sends to multiple
experts on the same rank, but does not eliminate repeated quantization across
ranks.

The new staging and compact-block policy applies only to:

```text
nvlinkOnly
&& layout == RANK_MAJOR
&& recipe == DS_FP8E3M4
&& hidden <= 16384
&& shared staging fits the capacity budget
```

`nvlinkOnly` means every destination is LSA/NVLink reachable; it is not a new
physical-node-count check. Tuning used GB200 GPUs within one NVLink domain.
BF16 dispatch, other recipes, expert-major dispatch,
RDMA/general dispatch, and combine retain their existing selection policies.

## Design

### Quantization and payload movement

The numerical recipe is unchanged: BF16 input becomes FP8 E4M3 payload with
one FP32 inverse scale per 128 elements. Both general and LSA dispatch use
`ll_dispatch_recipe.cuh` for the quantization arithmetic.

For a token owned by a CTA, all forwarding warps cooperatively quantize the
row into shared memory. The existing worker barrier publishes both the routing
header and the staged payload. Elected destination warps then copy the same
payload and scales into each peer's receive region.

```text
BF16 input -> cooperative quantization -> CTA shared payload and scales
                                              |
                                    worker publication barrier
                                              |
                         +--------------------+--------------------+
                         v                    v                    v
                     peer rank A          peer rank B          peer rank C
                         +--------------------+--------------------+
                                              |
                                     worker reuse barrier
                                              |
                                          next token
```

The extra worker-only barrier prevents the next token from overwriting staging
while another forwarding warp still reads it. The control warp does not join
these worker barriers; their participant count is `(numWarps - 1) * 32`.
All forwarding warps in a CTA follow the same token loop, including warps that
have no elected destination. A CTA with no token skips that loop as a unit and
still participates in the existing completion protocol.

Quantization reads eight BF16 values through an `int4` and produces eight FP8
bytes in an `int2`. The staged copy groups two such results into an `int4`:
16 FP8 bytes per lane-level copy. Shared staging is explicitly 16-byte aligned;
the existing receive-slot layout supplies aligned destinations. FP32 scales
are copied separately. `int4` here is a vector of four 32-bit integers, not
four-bit quantization. Disassembly confirms `LDS.128` and `ST.E.128` for this
copy, versus `LDS.64` and `ST.E.64` previously. The wire layout and number of
payload bytes do not change.

### Block geometry and register pressure

Keep the existing number of warp groups and CTAs. Attempt to allocate at most
16 warps per block instead of the original budget of 32:

```text
compactWarpsPerGroup = floor(16 / numWarpGroups)
compactNumWarps      = compactWarpsPerGroup * numWarpGroups
```

Use this geometry only when there are at least two receive warps per expert
group and at least `topK + 1` total warps, including the control warp. Otherwise,
retain the original geometry. Integer division can produce fewer than 512
threads, such as the validated 480-thread configuration. Expert ownership and
the number of CTAs do not change; receive work is distributed over fewer warps.

Compact variants use `__launch_bounds__(512, 1)` and leave the outer receive-copy
loop's unrolling to the compiler. Repeated tuning favored this over `(512, 2)`;
the latter's lower register allocation did not translate into a consistent
latency benefit. Explicit outer-loop unrolling did not consistently improve
multi-node latency.

### Shared memory capacity and fallback

Dispatch reuses the group's cached `max_dynamic_smem` capacity and the existing
`ncclEpGroup_test_setMaxDynamicSmem` test override. Because this allocation is
static shared memory, the chooser caps the usable budget at 48 KiB instead of
treating the larger opt-in dynamic capacity as available to static arrays.

The conservative staging requirement is:

```text
requiredBytes = hidden + (hidden / 128) * sizeof(float) + 32
usableBytes   = min(cachedCapacity, 48 KiB)
```

The final 32 bytes reserve space for epoch scratch and allocation alignment;
this is a conservative selection bound, not a report of exact compiler usage.
The payload allocation is per token and reused, so it does not grow with token
count, rank count, or expert count.

| Condition | Quantization | Block geometry |
| --- | --- | --- |
| Eligible, fits, compact geometry valid | Shared once per token | Up to 16 warps |
| Eligible, fits, compact geometry invalid | Shared once per token | Original geometry |
| Insufficient staging capacity | Direct per destination | Original geometry |
| H greater than 16384 | Direct per destination | Original geometry |
| Other dispatch paths | Existing implementation | Existing geometry |

The hidden-size cutoff is tuning-derived, not a correctness or hardware limit.
A follow-up sweep on four GB200 GPUs validated staging at H=8192, 12288,
and 16384 with 128, 256, and 512 tokens per rank, top-k 8, and 256 experts.
Five rotated repeats per configuration favored staging through H=16384;
the capacity check remains independent of this tuning cutoff.

The host passes the staging choice through JIT generation as a template
parameter. Disabling staging therefore removes the row-sized allocation and
reuse barrier at compile time, rather than merely skipping their execution.
Both staging and compact geometry participate in JIT cache identity. This is
necessary when the same handle switches between capacity budgets in a test.
The fallback still requires the original kernel's basic resources; it is not
a promise to run with arbitrarily little shared memory.

## Alternatives and performance

Nine alternatives to direct per-destination quantization were explored:
rolled loops, partial unrolling, shorter FP32 temporary lifetimes, a different
per-lane quantization width, shared quantization, and four combinations of
compact geometry, shared staging, and register-constraining launch bounds.
Shared quantization with constrained compact geometry won the initial study. Changes
to quantization unrolling and temporary lifetime did not produce a consistent
gain; the alternate per-lane width was slower.

The initial repeated comparisons below used one node, four GB200 ranks, BF16 input,
LL rank-major DS-FP8 dispatch, H=7168, top-k=8, and 256 experts. Each variant
had five runs, each with 30 warmups and 200 measured iterations. Values are
medians of host-observed run-average dispatch times, not per-iteration medians
or GPU-only kernel times. Variant order alternated or rotated; no outliers
were discarded.

| Comparison | Tokens per rank | Baseline in microseconds | Candidate in microseconds | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Initial LSA direct quantization versus staged compact kernel | 128 | 32.87 | 26.45 | 19.5% |
| Staged compact `int2` copy versus `int4` copy | 128 | 26.99 | 26.98 | Essentially unchanged |
| Staged compact `int2` copy versus `int4` copy | 512 | 41.80 | 40.59 | 2.9% |

The first comparison used `8f6e6e00` versus the pre-`int4` candidate
`bfd82cec`; it is not a comparison against main. The vector-copy comparison
produced the implementation folded into `90b34c84`. These are separate paired
experiments: their percentages must not be multiplied into a claimed final
speedup against main.

At H=7168, the initial direct kernel and staged compact kernel both used 64
registers per thread, 48 bytes of stack, and zero spill loads/stores in the
ptxas check. The then-capped compact `int4` variant retained 64 registers and
48 bytes of local allocation; its measured static shared memory was 7,408 bytes per block.
The corresponding `int2` variant used 7,400 bytes. Theoretical occupancy was
50%, with one 1024-thread block or two 512-thread blocks per SM by resource
limits. These figures do not measure achieved occupancy or account for cluster
placement. Nsight Compute stall and achieved-occupancy measurements were not
available in this study.

Follow-up launch-bound tuning covered H=1024–8192 on four GB200 ranks and
H=7168 on 8, 16, and 32 ranks within one NVLink block. It used 128 tokens per
rank, top-k 8, 256 total experts, five rotated repetitions, 100 warmups, and
2,000 measured iterations. All 248 runs passed validation. These measurements
support the selected launch bound, not a universal speedup across workloads.

### Expected scaling

The quantization saving depends on distinct destination ranks per token,
bounded by `min(topK, ranks)`, rather than expert count alone. More distinct
destinations increase potential reuse; NVLink traffic remains proportional to
the number of destinations. More experts can change launch geometry and routing
distribution without increasing quantization reuse. More tokens amortize fixed
overheads, while tiny workloads can expose staging and barrier costs. These are
design expectations, not measured rank/expert scaling results. One-token tests
establish correctness, not a performance win at one token.

## Validation

The capacity revision passed the following on four GB200 ranks:

- `LlDispatchSmem.StagingRespectsCapacityAndHiddenLimit`: exact capacity
  boundary, one byte below, invalid budgets, and hidden-size limits.
- `QuantizationRecipeTest.DsFp8OneTokenCapacityFallbackMatchesStaging`: one
  token per rank, staged to mocked-small-capacity fallback to staged on the same
  handle; receive counts and bitwise equality of FP8 bytes and FP32 scales.
- Existing `LlCombineSmem` tests and
  `LifecycleTest.LlCombineFitsMockedSmallSharedMemoryAndPreservesResults`.
- Existing expert-major `DsFp8E3M4DispatchCompletes` regression.
- Validated `ep_bench` with one token per rank for BF16 dispatch/combine, FP8,
  split dispatch with int32 indices and nonuniform tokens, top-k 16,
  H=16384 direct fallback under the former H=8192 cutoff, and a 384-expert
  configuration with 480-thread CTAs.

Earlier optimization checks also covered H=512/8192, top-k 1, 32 experts,
512 tokens per rank, and BF16 split dispatch with zero-copy output. These
earlier checks are not represented as fresh runs of the capacity revision.

## Implementation and packaging

| Responsibility | Source |
| --- | --- |
| Shared quantization, vector copy, worker barriers, direct fallback | `nccl_ep/device/ll/ll_dispatch_lsa.cuh` |
| Common quantization arithmetic | `nccl_ep/device/ll/ll_dispatch_recipe.cuh` |
| Capacity chooser and internal parameter | `nccl_ep/device/ll_ep_adapter.cuh` |
| Geometry and staging selection | `nccl_ep/device/ll_ep_adapter.cu` |
| Cached group capacity plumbing | `nccl_ep/nccl_ep.cc` |
| Template selection, launch bounds, cache identity | `nccl_ep/device/jit/ll_dispatch_jit.cuh` |
| Regression tests | `nccl_ep/tests/test_lifecycle.cu`, `test_quantization_recipe.cu` |

Make and CMake explicitly stage the new recipe header. Python native staging
and wheel package-data both include `nccl_ep/device/ll/*.cuh`, preserving the
header's relative path for runtime JIT. The installed header was checked
against source; a fresh Python wheel build was not part of this validation.

No public struct or ABI changes are needed. The existing capacity override is
test-only, not a new user knob. Rollback is to revert the optimization and its
selection wiring while retaining the initial LSA DS-FP8 implementation; the
direct quantization path remains available in the same kernel family.
