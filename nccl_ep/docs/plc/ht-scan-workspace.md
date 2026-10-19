# HT scan workspace ownership

The high-throughput metadata scan uses three temporary device buffers while
`ncclEpUpdateHandle` prepares a handle:

- a per-token destination-rank mask;
- decoupled-lookback scan state;
- for expert-major permutation, a temporary global-token-to-receive-slot map.

These buffers are update-only. Dispatch and combine consume the persistent maps
and counts stored in each handle. They are therefore allocated lazily once per EP
group and shared by its scan handles. This matches the existing lifetime of the
group-owned routing map used by the same scan and preserves the rule that handle
updates on one group are ordered.

Count-mode metadata remains handle-owned because it survives from update into
dispatch and permits several prepared handles to be live at once. Count-only
groups do not allocate the scan workspace.

For `max_dispatch_tokens_per_rank = 16384`, 256 ranks, and an eight-rank LSA
team, sharing removes about 48 MiB from every additional expert-major scan
handle: 32 MiB of rank masks and 16 MiB of token-to-slot entries, plus the small
scan-state allocation. The exact saving scales with the configured token and
rank counts.
