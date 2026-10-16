# HT flat pull dispatch and push combine

MR: !310

## Purpose and selection

Extend the expert-major pull/push transport to flat receive layouts without
changing the default transport. `NCCL_EP_HT_PULL_PUSH=1` selects it for both
layouts and replaces `NCCL_EP_HT_EM_PULL_PUSH`; the old name is not an alias.
Transport and synchronization settings must be consistent across ranks.

The path supports one LSA team of at most 128 ranks, top-k up to 32, and at
most 65535 experts. Dispatch accepts NONE and FWD scale transport; combine
accepts NONE. Local-dup and NVLink-dup overrides are incompatible with PP.
Each token's nonnegative expert IDs must be unique.

## Metadata and dispatch

UpdateHandle exchanges uint16 top-k routing and runs the existing scan.
Flat uses this scan regardless of the EM scan-mode override. Fused metadata
is a separate change.

Each source token occupies one receive slot per destination rank, even when
several local experts are selected. The scan stores a receive-slot-to-source
map and packed local-expert/original-top-k-position entries. Expert entries
are ordered by local expert ID. DROP also preserves destination slots so
combine can skip contributions that were not delivered.

`dispatch_pull_flat` in `device/ht_ep.cuh` assigns receive rows to warps.
Existing TMA stages load peer token rows while sparse weights and expert IDs
are decoded. The caller receives flat rows directly, including forwarded
scales when requested. Non-windowed inputs first use the existing staging
fallback; input windows avoid that copy. Both paths remain supported.

Metadata is handle-owned, including backward source positions. No LERM is
allocated for flat PP because these kernels do not consume it.

## Combine and synchronization

The caller reduces its local expert outputs into one flat row. The
`combine_push_flat` kernel pushes each row into source-rank staging indexed
by source token and contributing rank. A separate reduction sums the retained
rank contributions. Backward combine also returns weights to their original
top-k columns. Fully dropped contributions produce zero, and empty source
ranks still participate in communication before skipping local reduction.

The reduction specializes handle dimensions only for flat. Expert-major
retains runtime dimensions and its existing variant granularity.

Payload kernels use the existing head/tail synchronization helpers.
`NCCL_EP_HT_UNFUSED_SYNC=1` selects standalone synchronization kernels.
The kernel algorithms and shared-memory warp budget are unchanged by the
review fixes.

## API checks and validation

Host validation rejects invalid flat output dtypes/shapes, backward weight
widths, flat expert-counter descriptors, and oversized pull/push teams in
release builds. Internal device bounds assertions remain safeguards.

Focused tests cover forward/backward values, rank-deduplicated receive order,
DROP, empty ranks, interleaved and reused handles, relocated handle snapshots,
scale forwarding, and both synchronization choices. Invalid dtype/width
cases verify error returns followed by a successful operation on the handle.
The CI configuration reuses these tests and includes a flat PP benchmark.

Performance depends on the workload: combine still uses peer staging and a
separate reduction. Compare matching input-window and zero-copy settings;
changing input windowing changes which staging costs are included.
