# NCCL EP PLC documentation — non-public

This subtree contains internal NCCL EP PLC design material. It is explicitly
excluded from the public filter by `PUBLIC_PATHS.txt`; do not remove that
exclusion without an explicit review.

## PLCs

- [Low-latency epoch ticketing](ll-epoch-ticketing/README.md) — graph-safe LL two-bank resource selection.
- [HT EM pull dispatch + push combine](ht-em-pull-push/README.md) — local `recv_x_em`, fused local permute, sparse weights end-to-end.
