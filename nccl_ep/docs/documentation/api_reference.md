# API Reference

Complete reference for the NCCL EP public C API. Every entry point exported by
`nccl_ep.h` is listed here.

Behavioral topics are covered in dedicated guides rather than repeated per
function: [Quantization](quantization.md), [Zero-Copy](zero_copy.md),
[Recv Overflow Policy](overflow_policy.md), and [HT Eager Mode](eager_mode.md).

## Contents

| Group | Functions |
|---|---|
| [Library](#library) | `ncclEpGetVersion` |
| [Group Management](#group-management) | `ncclEpCreateGroup`, `ncclEpGroupDestroy` |
| [Tensor Descriptors](#tensor-descriptors) | `ncclEpTensorAlloc`, `ncclEpTensorDestroy` |
| [Handle Management](#handle-management) | `ncclEpCreateHandle`, `ncclEpInitHandle`, `ncclEpUpdateHandle`, `ncclEpHandleMemSize`, `ncclEpImportHandle`, `ncclEpHandleDestroy` |
| [Communication Operations](#communication-operations) | `ncclEpDispatch`, `ncclEpCombine`, `ncclEpAfDispatch`, `ncclEpAfRecv`, `ncclEpAfCombine`, `ncclEpAfReturn`, `ncclEpComplete` |
| [Fault Tolerance](#fault-tolerance) | `ncclEpMaskQuery`, `ncclEpMaskUpdate`, `ncclEpMaskClean`, `ncclEpGetAsyncError`, `ncclEpErrorClear` |

## Library

### `ncclEpGetVersion()`

```c
// Return the NCCL_EP_VERSION_CODE of the NCCL EP library in the supplied integer.
// The value encodes MAJOR, MINOR and PATCH as MAJOR*10000 + MINOR*100 + PATCH.
//
// Arguments:
//   version - [OUT] Pointer to receive the library's NCCL_EP_VERSION_CODE value
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpGetVersion(int* version);
```

Compare against `NCCL_EP_API_VERSION` before relying on a struct field
introduced in a later release; see the ABI notes in the main
[README](../../README.md#core-concepts).

## Group Management

### `ncclEpCreateGroup()`

```c
// Create an EP group from an NCCL communicator
//   This call is collective and must be invoked by all ranks in the group.
//   Any custom device-memory allocator is supplied via config->alloc
//   (see ncclEpAllocConfig_t); zero-init falls back to cudaMalloc/cudaFree.
//
// Arguments:
//   ep_group   - [OUT] Pointer to newly created EP group
//   comm       - [IN]  Existing NCCL communicator
//   config     - [IN]  Pointer to EP configuration structure
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCreateGroup(
    ncclEpGroup_t* ep_group,
    ncclComm_t comm,
    const ncclEpGroupConfig_t* config
);
```

In HT mode, `dispatch_num_sms` / `combine_num_sms` set the forward budgets and
`dispatch_bwd_num_sms` / `combine_bwd_num_sms` the backward ones
(`pass_direction = NCCL_EP_BWD_PASS`); a backward field left AUTO inherits the
resolved forward budget. Each budget resolves, highest first, from its
operation environment variable, `NCCL_EP_COMM_SMS`, its field, `max_num_sms`,
then the algorithm default. LL groups ignore the backward fields. All ranks must
resolve the same budgets for every layout and pass, otherwise the call returns
`ncclInvalidUsage`. See the
[SM-budget controls](../../README.md#high-throughput-tuning) for environment overrides.

### `ncclEpGroupDestroy()`

```c
// Destroy an EP group and release associated resources.
//
// Arguments:
//   ep_group     - [IN]  EP group to destroy
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpGroupDestroy(
    ncclEpGroup_t ep_group
);
```

## Handle Management

An NCCL EP handle represents the state that connects related dispatch and combine
operations — a dispatch/combine pair on the LL path, or both the forward and
backward passes in HT mode. The handle holds only the metadata-related state; the
staging buffers are owned by the EP group.

### Multiple handles per group

Both HT and LL support several handles being active on the same EP group, as a
1F1B / pipeline-parallel schedule requires. Updating one handle does not disturb
another handle's pending dispatch or combine. LL handle creation requires no
communication in flight on the group; see `ncclEpInitHandle` below.

### `ncclEpCreateHandle()`

```c
// Create and initialize an EP handle.
// Performs dispatch setup and (in HT mode only) metadata exchange.
// This call is collective and must be invoked by all ranks in the group.
// The routing carried by `topk_idx` is cached on the handle; subsequent
// ncclEpDispatch / ncclEpCombine calls reuse it until ncclEpUpdateHandle
// replaces it with new routing.
//
// Arguments:
//   handle              - [OUT] Pointer to newly created and initialized EP handle
//   ep_group            - [IN]  A valid EP group
//   layout              - [IN]  Receive buffer layout. Required; must not be NCCL_EP_LAYOUT_UNSET.
//                                HT supports FLAT / EXPERT_MAJOR; LL supports EXPERT_MAJOR / RANK_MAJOR.
//   topk_idx            - [IN]  Pointer to a caller-owned tensor descriptor holding
//                               top-K expert indices (2D [num_tokens, top_k]; int32 or int64).
//   layout_info         - [IN/OUT, optional] Named-struct pointer carrying device-side
//                         metadata tensor pointers. See ncclEpLayoutInfo_t for the fields
//                         populated at handle time and the layouts each applies to.
//                         NULL = no metadata.
//   config              - [IN]  Optional handle configuration (e.g. expert-major
//                               alignment via dispatch_output_per_expert_alignment);
//                               NULL = defaults.
//   stream              - [IN]  CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpTensor_t* topk_idx,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpHandleConfig_t* config,
    cudaStream_t stream
);
```

### `ncclEpHandleDestroy()`

```c
// Destroy an EP handle and release all associated resources.
//
// Arguments:
//   handle         - [IN]  EP handle to destroy
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpHandleDestroy(
    ncclEpHandle_t handle
);
```

### `ncclEpInitHandle()`

`ncclEpCreateHandle` is the fused form of `ncclEpInitHandle` +
`ncclEpUpdateHandle`. Use the split form when handle allocation and per-step
routing happen at different points — for example to allocate all handles before
a CUDA graph capture begins.

```c
// Allocate handle buffers; LL initialization may be collective (see below).
// Call ncclEpUpdateHandle before the first ncclEpDispatch/ncclEpCombine.
//
// handle_mem == NULL:  NCCL EP allocates via alloc_fn; handle owns the memory.
// handle_mem != NULL:  wraps a caller-owned 1D ncclUint8 tensor
//                      (>= ncclEpHandleMemSize); handle owns no memory and
//                      ncclEpHandleDestroy frees only the struct.
//
// Arguments:
//   handle     - [OUT] Newly created handle
//   ep_group   - [IN]  A valid EP group
//   layout     - [IN]  Receive buffer layout. Required; must not be NCCL_EP_LAYOUT_UNSET.
//   config     - [IN]  Handle configuration (see ncclEpHandleConfig_t). NULL = defaults.
//   num_topk   - [IN]  Required for LL (> 0); pass -1 for HT
//   handle_mem - [IN]  NULL = internal alloc; non-NULL = caller-owned device buffer
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpInitHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpHandleConfig_t* config,
    int num_topk,
    const ncclEpTensor_t* handle_mem);
```

**Collective under AUTO sizing or hierarchical LL bank layout changes.** With auto-sized RDMA
buffers this call *may* be collective: if the requested `(layout, num_topk)`
needs more memory than the current allocation, it reallocates. Hierarchical LL
can also collectively change shared bank offsets without reallocating. All ranks must
then call it in lockstep with the same `(layout, num_topk)`, with no other
communication in flight, and it must not appear inside a CUDA graph capture.
Complete all pending staged operations with `ncclEpComplete` and synchronize
their streams on all ranks before LL handle initialization. An explicit
`rdma_buffer_size` prevents reallocation but not collective bank layout changes.
With the Lamport combine (`NCCL_EP_COMBINE_LSA=lamport`), a handle whose
`num_topk` exceeds the capacity of the Lamport landing region also re-creates
the region collectively, with the same constraints; setting
`ncclEpGroupConfig_t::num_topk` to the largest `num_topk` of any handle avoids it.
Re-capture EP graphs after reallocation or shared bank layout changes. See the
[`rdma_buffer_size` notes](../../README.md#algorithm-related-configurations).

### `ncclEpUpdateHandle()`

```c
// Per-step collective: prepare the handle for the given top-k routing decisions.
// Must be called after ncclEpInitHandle and before ncclEpDispatch. The routing is
// cached on the handle and reused by every dispatch until the next update.
//
// Arguments:
//   handle      - [IN]  Handle from ncclEpInitHandle
//   topk_idx    - [IN]  [num_tokens, top_k]; ncclInt32 or ncclInt64. The caller
//                       ensures expert ids fit in the chosen width.
//   layout_info - [IN/OUT, optional] Named local tensors (NULL = none provided).
//                       See ncclEpLayoutInfo_t for the fields populated at handle
//                       time and the layouts each applies to. LL mode: must be NULL.
//   stream      - [IN]  CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpUpdateHandle(
    ncclEpHandle_t handle,
    const ncclEpTensor_t* topk_idx,
    const ncclEpLayoutInfo_t* layout_info,
    cudaStream_t stream);
```

In HT this is where the metadata scan runs, so it is also where a recv overflow
is detected and where `recv_total_counter` is written. See
[Recv Overflow Policy](overflow_policy.md).

### `ncclEpHandleMemSize()`

```c
// Query the device bytes required for a handle's persistent routing state, for
// callers that want to supply handle memory themselves via the handle_mem argument
// of ncclEpInitHandle / ncclEpCreateHandle. Group-owned update-only scan workspace
// is allocated lazily by ncclEpInitHandle and is not included in size_out.
//
// Arguments:
//   ep_group  - [IN]  A valid EP group
//   layout    - [IN]  Receive buffer layout. Required; must not be NCCL_EP_LAYOUT_UNSET.
//   config    - [IN]  Handle configuration (see ncclEpHandleConfig_t). NULL = defaults.
//   size_out  - [OUT] Required bytes for handle_mem
//   num_topk  - [IN]  Required for LL (> 0); optional for HT
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpHandleMemSize(
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpHandleConfig_t* config,
    size_t* size_out,
    int num_topk);
```

### `ncclEpImportHandle()` (HT only)

Binds a handle to a `handle_mem` buffer already prepared by `ncclEpUpdateHandle`
on a *different* handle, without rerunning the collective. Intended for callers
that snapshot a prepared `handle_mem` and later rebind it at a different address
— `ncclEpHandleState_t` carries the host metadata needed to reinterpret that
snapshot.

```c
// Import an HT handle from a caller-owned handle buffer already prepared by
// ncclEpUpdateHandle with matching group, layout, config, and state metadata.
// Local only: no device-state copy, CUDA launch, or metadata scan. Dispatch
// and Combine can use the returned handle without another UpdateHandle.
//
// For forward dispatch: handle_mem may be snapshotted any time after
// ncclEpUpdateHandle completes on the source stream.
//
// For backward dispatch or combine under fused metadata (push-count: expert-major
// local-permute and Flat; pull-count: the pull/push flow; see
// ncclEpDispatch's layout_info notes): handle_mem must be snapshotted after the
// matching forward ncclEpDispatch has completed, with the copy ordered after that
// work. ncclEpUpdateHandle alone does not write the per-token maps that backward
// dispatch and combine read. Otherwise the per-token maps are complete after
// ncclEpUpdateHandle.
//
// Arguments:
//   handle     - [OUT] Newly imported handle
//   ep_group   - [IN]  The EP group used to prepare handle_mem
//   layout     - [IN]  The layout used to prepare handle_mem
//   config     - [IN]  Handle configuration used to prepare handle_mem. NULL = defaults.
//   state      - [IN]  Host metadata describing the prepared state
//   handle_mem - [IN]  Caller-owned prepared 1D ncclUint8 device buffer
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpImportHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpHandleConfig_t* config,
    const ncclEpHandleState_t* state,
    const ncclEpTensor_t* handle_mem);
```

`ncclEpHandleState_t` (`num_tokens`, `num_topk`, `topk_idx_datatype`) is
caller-constructed from the values used to prepare the snapshot; initialize it
with `NCCL_EP_HANDLE_STATE_INIT`.

## Tensor Descriptors

`ncclEpTensor_t` is a plain value struct.  The common path is to allocate on the
stack (or as a struct member), zero-initialise it with `NCCL_EP_TENSOR_INIT`, and
fill fields directly:

```c
int N = 128;
int H = 2048;
size_t dims[2] = { N, H };
ncclEpTensor_t t = NCCL_EP_TENSOR_INIT;
t.ndim = 2; t.datatype = ncclFloat16; t.data = data_ptr;
t.sizes = dims;   // caller owns; must outlive the descriptor's use
```

For window-backed tensors set `win_hdl` / `win_offset` instead of `data`.

When a heap-allocated descriptor with a library-owned `sizes` copy is more
convenient, use `ncclEpTensorAlloc` to obtain one and `ncclEpTensorDestroy`
to release it (the backing data buffer remains caller-owned).

All user-facing tensor fields (`data`, `ndim`, `datatype`, `sizes`, `win_hdl`,
`win_offset`) are directly accessible as struct members.  Public structs
(`ncclEpDispatchInputs_t`, `ncclEpLayoutInfo_t`, …) hold `ncclEpTensor_t*`
pointers, so callers can mix stack-, static-, and heap-allocated descriptors.

### `ncclEpTensorAlloc()`

```c
// Allocate a tensor descriptor sufficient to represent the requested shape.
// The library copies `sizes` into its own storage, so the caller's array need
// not outlive the call. The backing data buffer remains caller-owned.
//
// Arguments:
//   tensor   - [OUT] On success, receives a pointer to the new descriptor.
//   ndim     - [IN]  Number of dimensions (> 0).
//   datatype - [IN]  Element type.
//   sizes    - [IN]  Array of `ndim` dimension sizes.
//   config   - [IN]  Optional allocation configuration. NULL = defaults.
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpTensorAlloc(
    ncclEpTensor_t** tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    const size_t* sizes,
    const ncclEpTensorAllocConfig_t* config);
```

### `ncclEpTensorDestroy()`

```c
// Release a descriptor previously returned by ncclEpTensorAlloc.
// Does not free the descriptor's data buffer, which is caller-owned.
//
// Arguments:
//   tensor - [IN] Pointer returned by ncclEpTensorAlloc. NULL is accepted.
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpTensorDestroy(ncclEpTensor_t* tensor);
```

## Communication Operations

### `ncclEpDispatch()`

Perform EP dispatch: send tokens to experts according to routing decisions.

LL rank-major dispatch accepts FP32 or BF16 routing weights in
`inputs->topk_weights` with shape `[num_tokens, top_k]`. The corresponding
`outputs->topk_weights` tensor must use the same dtype and shape
`[num_ranks, max_dispatch_tokens_per_rank, top_k]`. The JIT specializes weight
loads and stores by dtype; the internal routing wire header remains FP32.
LL expert-major combine also accepts FP32 or BF16 routing weights in
`outputs->topk_weights`; BF16 weights are widened in registers before the
FP32 weighted reduction. HT routing weights remain FP32.

```c
// Perform EP dispatch
//   * Sends tokens and metadata to the experts according to routing decisions.
//   * This call is collective and must be invoked by all ranks in the group.
//   * Routing (topk_idx) is taken from the handle — supply it once via
//     ncclEpCreateHandle / ncclEpUpdateHandle.
//   * Cross-boundary tensors are carried in named-struct fields
//     (ncclEpDispatchInputs_t / ncclEpDispatchOutputs_t / ncclEpLayoutInfo_t),
//     each field a `ncclEpTensor_t*` to a caller-owned descriptor.
//
// Arguments:
//   handle        - [IN,OUT] EP handle
//   inputs        - [IN]     Named preallocated input tensors. `inputs->tokens` is required;
//                            other fields (topk_weights, scales) are optional and depend on
//                            algorithm/layout.
//   outputs       - [IN,OUT] Named preallocated output tensors. `outputs->tokens` is required;
//                            other fields (topk_weights, topk_idx, scales) are optional and
//                            depend on algorithm/layout. For HT, outputs are 2D [N(r), data_size].
//                            For LL expert-major, outputs are 3D [num_local_experts, N(r), data_size].
//                            For LL rank-major,   outputs->tokens is 3D
//                            [num_ranks, max_dispatch_tokens_per_rank, data_size].
//   layout_info   - [IN,OUT, optional] Named-struct pointer for device-side metadata tensors.
//                            LL mode: optional `expert_counters` (1D ncclInt32 / ncclInt64,
//                            size = num_local_experts) receives per-expert recv counts.
//                            NULL = no metadata.
//   config        - [IN]     Dispatch configuration.
//   stream        - [IN]     CUDA stream. If ncclEpDispatch is called on a different stream than
//                            the stream used in `ncclEpCreateHandle`, it is the responsibility
//                            of the user to synchronize between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream
);
```

### Dispatch tile completion

Set `ncclEpDispatchOutputs_t.dispatch_signal` to let a consumer on the same GPU (for example the FC1 GEMM) start on
output tiles before the dispatch ends. `nullptr` keeps the unsignaled path. Both signal fields need
`NCCL_EP_API_VERSION` >= 5.

**Requirements**

- HT expert-major pull dispatch: the pull/push flow (`ncclEpGroupConfig_t::ht_flow`, or AUTO or
  `NCCL_EP_HT_PULL_PUSH=1` selecting it) within one LSA team, in either pass direction. A group of several LSA teams
  returns `ncclInvalidUsage`; any other unsupported group returns `ncclInvalidArgument`.
- A handle created with a per-expert alignment that `granularity` divides.
- Fused sync: `NCCL_EP_HT_UNFUSED_SYNC` is not supported; a signaled dispatch there returns `ncclInvalidArgument`.
- A forward dispatch that builds its maps (pull-count): `ceil(lsa_team_size * ceil(max_dispatch_tokens_per_rank / 32) /
  32) * 4 <= 16384`, else `ncclInvalidArgument`.
- The `ncclEpSignal_t` is read at call time. The `signal` memory stays valid until the dispatch and its consumer
  complete on `stream`.

**`ncclEpSignal_t`** (shared with the combine signal)

| Field | Contract |
|---|---|
| `granularity` | Rows per tile: a power of two up to 32768 that divides the handle's per-expert alignment |
| `unit` | `NCCL_EP_SIGNAL_UNIT_AUTO` (default) or `NCCL_EP_SIGNAL_UNIT_ROWS`; any other value returns `ncclInvalidArgument` |
| `signal` | Bound, 4-byte aligned device int32 tensor of at least `ceil(outputs.tokens.sizes[0] / granularity)` entries: sized for the declared output rows, not the rows received. A rank whose `outputs.tokens` has no rows may pass any tensor, empty or unbound |

**Protocol**

1. Reset `signal` to -1 before each dispatch.
2. Tile `t` is rows `[t * granularity, (t + 1) * granularity)` of the padded output. Once a same-GPU acquire load reads
   `signal[t] == t`, its tokens, weights and scales, padding included, are visible (a TMA reader adds
   `fence.proxy.async`).
3. The published rows are `min(recv_total_counter, outputs.tokens.sizes[0])`, further clamped to
   `max_recv_tokens_per_rank` under `NCCL_EP_OVERFLOW_DROP` (`ncclEpLayoutInfo_t::recv_total_counter` is the padded
   total, pre-drop). Tiles `[0, ceil(published rows / granularity))` are published, approximately in row order;
   under drop the last one can be partial. Later entries stay -1. Take `recv_total_counter` of this routing from:
   - `ncclEpUpdateHandle`: complete before the dispatch starts.
   - This forward dispatch's `layout_info` (HT count mode): the dispatch writes it and `expert_offsets` with relaxed
     stores while the consumer may already run, and until then they keep their old values. Reset both to -1 before
     the dispatch and poll them with GPU-scope relaxed or acquire loads until they are not -1, or read them after
     `griddepcontrol.wait`. `expert_counters` is valid only after `griddepcontrol.wait`.
   - A backward dispatch writes no `layout_info`: its consumer uses the count of the forward whose maps it follows.
4. Wait only for those tiles. A rank that receives no token publishes none; waiting for every allocated flag can
   hang.

**Consumer launch.** A kernel launched next on `stream` with `cudaLaunchAttributeProgrammaticStreamSerialization`
starts once every dispatch block has passed the cross-rank barrier at its start. The consumer:

1. reads only tiles whose flag it has acquired;
2. executes `griddepcontrol.wait` before it exits or calls `griddepcontrol.launch_dependents`, so later work on
   `stream` also follows the dispatch;
3. is not a signaled combine (see "Combine readiness signaling");
4. writes no memory the dispatch reads or writes on any rank before its `griddepcontrol.wait`. Peers still pull
   `inputs` (tokens, weights, scales), so keep them alive and unmodified until then.

A consumer on another stream leaves room for every dispatch block to be resident, or the ranks can hang.

**Dispatch launch.** A signaled forward dispatch that builds its maps is a programmatic dependent of the kernel
before it on `stream` and runs `griddepcontrol.wait` at entry. That kernel completes only after every earlier kernel
on `stream`: it is launched without programmatic stream serialization, or runs its own `griddepcontrol.wait`. Every
other dispatch launches plainly.

**Limits**

- A signaled dispatch of a fixed-budget group allocates nothing, so every one of them can be captured, including the
  first. An eager group (`max_recv_tokens_per_rank` = `NCCL_EP_AUTO`) cannot be captured; it allocates or grows the
  scratch before a signaled dispatch whose handle pads each expert zone past it, which waits for `stream` to drain.

**Memory.** Groups that can carry a dispatch signal reserve its buffers at creation, whether or not a call
requests one: a chunk table of 40 bytes per source chunk (`lsa_team_size * ceil(max_dispatch_tokens_per_rank / 32)`
chunks) per group. A fixed-budget group also reserves the group-shared per-call scratch at creation, at about 6
bytes per `max_recv_tokens_per_rank` row; an eager group allocates it at its first signaled dispatch, sized to its
handles' padded rows.

### `ncclEpCombine()`

Perform EP combine: gather expert outputs and return in original token order.

```c
// Perform EP combine
//   * Gathers outputs from experts and returns them to their source in original token order.
//   * This call is collective and must be invoked by all ranks in the group.
//   * Cross-boundary tensors are carried in named-struct fields
//     (ncclEpCombineInputs_t / ncclEpCombineOutputs_t).
//
// Arguments:
//   handle           - [IN,OUT] EP handle that was used for `ncclEpDispatch()` operation.
//   inputs           - [IN]     Named preallocated input tensors. `inputs->tokens` is required;
//                               LL expert-major: 3D [num_local_experts, N(r), data_size].
//                               LL rank-major:   3D [num_ranks, max_dispatch_tokens_per_rank, data_size].
//                               HT:              2D [N(r), data_size].
//                               Backward combine: also set `inputs->topk_weights` (HT only):
//                               expert-major 1D [N(r)], FLAT 2D [N(r), top_k].
//   outputs          - [IN,OUT] Named preallocated output tensors. `outputs->tokens` is required;
//                               2D [num_tokens, data_size] in original token order.
//                               LL expert-major: `outputs->topk_weights` [num_tokens, top_k]
//                               is used by the receive-side reduction.
//                               HT backward combine: `outputs->topk_weights` [num_tokens, top_k]
//                               receives per-token routing weights.
//   config           - [IN]     Combine configuration (e.g. `send_only` for LL staged mode).
//   stream           - [IN]     CUDA stream. If `ncclEpCombine()` is called on a different stream than
//                               the stream used in `ncclEpCreateHandle()`, it is the responsibility
//                               of the user to synchronize between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream
);
```

### Combine readiness signaling

Set `ncclEpCombineInputs_t.combine_signal` to let the push combine overlap the producer of `inputs.tokens` (for
example the FC2 GEMM). `nullptr` keeps the unsignaled path.

**Requirements**

- HT expert-major push combine: the pull/push flow, as for the dispatch signal, within one LSA team. A group of
  several LSA teams returns `ncclInvalidUsage`; any other unsupported call returns `ncclInvalidArgument`.
- `NCCL_EP_COMB_QUANT_NONE`, `send_only == 0`, and no `NCCL_EP_HT_UNFUSED_SYNC`.
- A 2D `inputs.tokens` whose rows are a multiple of 16 bytes.
- Fixed or eager receive sizing, `NCCL_EP_OVERFLOW_DROP` and backward combine are supported.
- The `ncclEpSignal_t` is read at call time. The `signal` memory stays valid until the combine completes on `stream`.

**`ncclEpSignal_t`** (shared with the dispatch signal)

| Field | Contract |
|---|---|
| `granularity` | Push chunk width in hidden elements: a positive multiple of the push tile, else `ncclInvalidArgument`. A push tile is `min(ceil(row bytes / 512), 4) * 512` bytes (1024 BF16 elements for rows of at least 2048 bytes). Chunk `s` is columns `[s * granularity, min((s + 1) * granularity, hidden))`, so the signal has `ceil(hidden / granularity)` flags. Match it on every rank |
| `unit` | `NCCL_EP_SIGNAL_UNIT_AUTO` (default) or `NCCL_EP_SIGNAL_UNIT_COLUMNS`; any other value returns `ncclInvalidArgument` |
| `signal` | Bound, 4-byte aligned device int32 tensor of at least one entry per chunk |

For example, with BF16 hidden 4096 and `granularity` 2048, the signal has two flags: publish `signal[0] = 0` once
columns [0, 2048) of every received row are written, then `signal[1] = 1` once [2048, 4096) are.

**Protocol**

1. Reset `signal` to -1 before the producer starts.
2. Once columns of chunk `s` of every received expert-major row are written and fenced, the producer release-stores
   `signal[s] = s` at GPU scope.
3. The combine pushes chunk by chunk as the flags arrive. The output matches the unsignaled combine for finite
   values. A flag that is never set traps after the device timeout.

**Launch.** The push is the programmatic dependent of the kernel before it on `stream`, with no
`griddepcontrol.wait` at entry: it starts when that kernel calls `griddepcontrol.launch_dependents`, or ends.

1. The kernel directly before the combine triggers only once every earlier kernel on `stream` has completed: it is
   launched without programmatic stream serialization, or it triggers after its own `griddepcontrol.wait`.
2. That kernel is not a signaled dispatch, whose dependent starts while the dispatch still writes the maps the push
   reads.
3. The reset of `signal`, and `inputs.topk_weights`, are complete before the producer starts.

The push and the reduce are a programmatic chain: the reduce starts beside the push, reads the staging once the push
has passed its LSA tail barrier, and completes after the push, which completes after the producer. A producer on
another stream must have every block resident before the push starts, or the push can fill the GPU and the ranks
hang.

In an LSA team of 8 or more ranks, where the push stores by TMA, a row of two or more chunks runs the chain in two
parts, unless the group uses `NCCL_EP_OVERFLOW_DROP`: the push of every chunk but the last on the combine SM budget;
then the push of the last chunk, beside the reduce of the other chunks on the shuffle SMs (as an unsplit reduce); then
the reduce of the last chunk. The last chunk's push and reduce run on the SMs of the other chunks' push and reduce
unless `NCCL_EP_HT_COMBINE_PUSH_LAST_CHUNK_SMS` or `NCCL_EP_HT_COMBINE_REDUCE_LAST_CHUNK_SMS` sets them (1 to the
device SM count, read at group creation; another value is ignored with a warning). These two may differ across ranks.
Each push ends on the LSA tail barrier. Every rank therefore passes `combine_signal` to a combine, or none does: a rank
that splits runs one more LSA barrier than a rank that does not, and the ranks hang.

The reduce lets the next kernel on `stream` start at its entry, as a signaled dispatch does for its consumer. A
kernel launched next with programmatic stream serialization runs `griddepcontrol.wait` before it exits or triggers,
and until then neither reads nor writes `outputs` or any memory the combine reads or writes on any rank. A kernel
therefore stays between a signaled combine and the next one. The reduce also writes `outputs`, `outputs.topk_weights`
included, while the kernel before the combine may still run. That kernel does not access `outputs`, and no memory it
reads or writes is freed and handed back as `outputs` (as a caching allocator may do on the same stream).

**Streams and memory.** The push waits for its predecessor on the combine stream. When the producer runs on another
stream, `signal[s] == s` establishes only that chunk `s` is ready: the producer may still run after it publishes the
last flag, so order the producer's stream independently (an event or stream wait) before freeing or reusing
`inputs.tokens`. The flags are GPU-scope, so the producer and the combine must be in the same CUDA memory
synchronization domain.

If the reduce launch of a signaled combine fails, every later signaled combine on the group returns
`ncclInternalError`: recreate the group.

The combine signal reserves no memory. See the [PLC document](../plc/429-ht-signals/README.md) for the design.

### AFD role wrappers

Attention-FFN Disaggregation (AFD) partitions an LL rank-major group into a
leading prefix of attention ranks and a trailing set of expert ranks, selected
by `ncclEpGroupConfig_t::num_attn_ranks`. The four role-named APIs validate the
calling rank and then forward to the ordinary dispatch or combine operation.
They do not create separate collectives: every rank must call its matching side
of each row below in the same order.

| Collective | Attention ranks | Expert ranks |
|---|---|---|
| Dispatch | `ncclEpAfDispatch` | `ncclEpAfRecv` |
| Combine | `ncclEpAfCombine` | `ncclEpAfReturn` |

```c
ncclResult_t ncclEpAfDispatch(
    ncclEpHandle_t handle,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream);

ncclResult_t ncclEpAfRecv(
    ncclEpHandle_t handle,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,
    const ncclEpDispatchConfig_t* config,
    cudaStream_t stream);

ncclResult_t ncclEpAfCombine(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream);

ncclResult_t ncclEpAfReturn(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,
    cudaStream_t stream);
```

On attention ranks, dispatch receive tensors and combine `inputs->tokens` are
omitted. On expert ranks, dispatch `inputs->topk_weights` is omitted, while the
rank-major dispatch receive tensors and combine input tensor are required. A
wrong-role call returns `ncclInvalidUsage`; an invalid handle returns
`ncclInvalidArgument`.

AFD follows the ordinary LL `AUTO` zero-copy contract: window use must be
consistent across the collective. When dispatch output tokens are window-backed,
every rank participates in the same `ncclCommWindowRegister` call and attaches
the resulting handle to `outputs->tokens`. F-ranks register the full receive
allocation. A-ranks may register a minimal placeholder allocation because NCCL
supports asymmetric window sizes; their local allocation is not written. If no
rank attaches a window, dispatch uses library staging as usual. Mixing windowed
and non-windowed descriptors across ranks is unsupported, as in classic LL EP.

Python exposes the same role checks as `Handle.af_dispatch`, `Handle.af_recv`,
`Handle.af_combine`, and `Handle.af_return`.

AFD currently supports unquantized communication only. Both sides of dispatch
must use `NCCL_EP_DISP_QUANT_NONE`, and both sides of combine must use
`NCCL_EP_COMB_QUANT_NONE`. Other quantization recipes are not supported with
`num_attn_ranks > 0`.

### `ncclEpComplete()` (LL mode only)

Must be the first NCCL EP operation to be executed after dispatch or combine,
no other op is allowed on handle or group in between.

```c
// Continues a staged EP operation to completion.
//   * This should be called after a prior `ncclEpDispatch()` or `ncclEpCombine()` call with `send_only` flag set.
//
// Arguments:
//   handle     - [IN,OUT] EP handle used in the preceding staged operation
//   config     - [IN]     Completion configuration (see ncclEpCompleteConfig_t). NULL = defaults.
//   stream     - [IN]     CUDA stream
//
// Notes:
//   - If `ncclEpComplete()` is called on a different stream than the operation initiation call
//     (i.e., `ncclEpDispatch()` or `ncclEpCombine()`), it is the responsibility of the user to
//     synchronize between streams to ensure correctness.
//   - Only LL mode is supported.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpComplete(
    ncclEpHandle_t handle,
    const ncclEpCompleteConfig_t* config,
    cudaStream_t stream
);
```

## Fault Tolerance

These APIs require `ncclEpGroupConfig_t::enable_mask = 1` and are **LL mode
only**. With masking enabled, a remote rank that times out during dispatch or
combine is masked (skipped) rather than trapping the GPU, and a host-visible
error flag is raised.

The timeout is `ncclEpGroupConfig_t::timeout_ns` (0 = default, ~100 s), which
`NCCL_EP_TIMEOUT_MS` overrides. Setting it too low risks masking merely slow
ranks.

With `NCCL_EP_COMBINE_LSA=lamport` the combine has no barrier, so after a timeout
different column ranges of one output row may include the timed-out rank's
contribution or not. Treat the rows of tokens routed to a timed-out rank as
invalid.

A typical recovery loop polls `ncclEpGetAsyncError`, inspects which ranks were
masked with `ncclEpMaskQuery`, continues in degraded mode or re-admits ranks with
`ncclEpMaskClean`, and then re-arms detection with `ncclEpErrorClear`.

### `ncclEpMaskQuery()`

```c
// Query the active-mask status of all ranks.
//   Copies the mask buffer to a user-provided device tensor.
//
// Arguments:
//   ep_group     - [IN]  EP group with masking enabled
//   mask_status  - [OUT] Device pointer to int[nRanks]. 1 = active, 0 = masked (failed).
//   stream       - [IN]  CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskQuery(ncclEpGroup_t ep_group, int* mask_status, cudaStream_t stream);
```

### `ncclEpMaskUpdate()`

```c
// Set the mask for all ranks at once.
//   With NCCL_EP_COMBINE_LSA=lamport, re-activating a masked rank requires
//   ncclEpMaskClean: Update alone does not reset the Lamport landing region.
//
// Arguments:
//   ep_group   - [IN] EP group with masking enabled
//   mask       - [IN] Host pointer to int[nRanks]. 1 = active, 0 = masked (failed).
//   stream     - [IN] CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskUpdate(ncclEpGroup_t ep_group, const int* mask, cudaStream_t stream);
```

### `ncclEpMaskClean()`

```c
// Reset masks and RDMA buffers so previously masked ranks can re-join.
//   Collective: all surviving ranks must call simultaneously.
//   Resets RDMA buffers via a cross-rank barrier and sets all masks to active.
//   Does NOT reset the async error flag - call ncclEpErrorClear() separately.
//
// Arguments:
//   ep_group - [IN] EP group with masking enabled
//   stream   - [IN] CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskClean(ncclEpGroup_t ep_group, cudaStream_t stream);
```

This re-admits a *delayed* rank within the same communicator. Replacing a rank
requires a new communicator (e.g. `ncclCommGrow`) and a new EP group.

### `ncclEpGetAsyncError()`

```c
// Poll for asynchronous errors (e.g., rank timeout).
//   Lightweight host-side check - reads a pinned CPU flag, no GPU sync required.
//   The flag is set by the kernel when a timeout masks a rank; clear it
//   explicitly via ncclEpErrorClear().
//
// Arguments:
//   ep_group  - [IN]  EP group with masking enabled
//   error_out - [OUT] 0 = no error, 1 = timeout occurred (one or more ranks masked)
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpGetAsyncError(ncclEpGroup_t ep_group, int* error_out);
```

### `ncclEpErrorClear()`

```c
// Clear the async error flag.
//   Lightweight host-side reset - writes zero to the pinned CPU flag.
//   Use after detecting an error (via ncclEpGetAsyncError) to re-arm the flag
//   for detecting new failures. Call after ncclEpMaskClean (full recovery) or
//   standalone when surviving ranks continue in degraded mode.
//
// Arguments:
//   ep_group - [IN] EP group with masking enabled
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpErrorClear(ncclEpGroup_t ep_group);
```
