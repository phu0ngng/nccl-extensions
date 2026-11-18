# Standalone NCCL-EP QA

`run_ep_qa.sh` runs NCCL-EP validation on machines that do not provide Slurm
or `nvsubmit`. It uses `mpirun`/`mpiexec` for the C++ and Python integration
tests and the repository's existing self-spawning launcher for unit tests.

The runner never builds NCCL or NCCL-EP. Point `--install` at a prepared tree
with this layout:

```text
<install>/
  include/
  lib/
  test/nccl_ep/
    ep_test
    ep_bench
    af_test
    test_*
```

## Prerequisites

- Hopper or Blackwell GPUs
- CUDA, NCCL, and NCCL-EP artifacts compatible with the target machine
- MPI (`mpirun` or `mpiexec`) for smoke, bench, and Python suites
- The NCCL-EP Python package and its dependencies for the Python suite
- GNU `timeout`, Bash 4+, and standard command-line utilities

Set `CUDA_HOME` when CUDA is not discoverable through `nvcc` in `PATH`.
For multi-node testing, configure passwordless MPI launch and use `--hostfile`.

## Suites

| Suite | Content |
| --- | --- |
| `smoke` | Five canonical `ep_test` LL/HT correctness cases |
| `unit` | Complete `tests/run_tests.sh` matrix on one machine |
| `bench` | LL/HT `ep_bench` matrix plus end-to-end AFD correctness |
| `hier` | Opt-in multi-node LL BF16 rank-major hierarchical dispatch qualification |
| `python` | Complete `ep_test.py` matrix plus AFD API export verification |
| `full` | Unit, smoke, bench, and Python suites |

## Release qualification

Use the following test plan for an EP release:

1. Run `full` with all ranks contained within one NVLink domain.
2. Run `bench` additionally with ranks distributed across nodes to exercise
   RDMA.

```bash
# 1. Full qualification within one NVLink domain
bash nccl_ep/qa/run_ep_qa.sh --install /path/to/nccl-ep-install --np 8 --unit-gpus 8 --suite full

# 2. Repeat EP bench across two nodes over RDMA
bash nccl_ep/qa/run_ep_qa.sh --install /path/to/nccl-ep-install --np 16 --hostfile /path/to/hosts --suite bench
```

This covers `ep_bench` in both NVLink and RDMA configurations without repeating
the smoke, unit, or Python suites on the multi-node run. Preserve the result
directory from each configuration as separate qualification evidence.

### Hierarchical dispatch qualification

Run this separate suite on two and four nodes with FULL GIN and one LSA domain
per node. Use 8, 16, or 32 total ranks, distributed evenly across nodes. Confirm
the actual LSA and rail teams in initialization logs; a hostfile alone does not
establish the topology. The forced candidate should fail on unsupported setups.
This suite is deliberately separate from `full`, which also supports a single
NVLink domain.

```bash
bash nccl_ep/qa/run_ep_qa.sh --install /path/to/nccl-ep-install \
  --np 16 --hostfile /path/to/hosts --suite hier
```

The 25 cases use `ep_bench --validate` with LL, RM, BF16 payloads, hidden 7168, top-k 8,
and 256 experts:

- Baseline, forced candidate, and AUTO at matched capacity for 1, 16, 256, and
  4096 tokens. The candidate selects direct or hierarchical sending locally.
- Forced candidate at capacity 512 with 1, 32, 64, and 256 active tokens.
- Nonuniform counts including an empty sender, a 513-token tail, and fixed-count
  CUDA graph replay.
- A 16-QP forced candidate to exercise queue reuse and a 16-QP AUTO case to
  exercise restricted-budget fallback.
- BF16 routing weights in forced-candidate Direct and relay sends, int32 routing
  indices, and fixed-count CUDA graph replay. Other cases use FP32 routing weights.

Inspect the AUTO selection logs as well as validation results. Graph coverage
here replays fixed captured counts; it does not test replay-varying device token
counts. The extended mixed-mode and replay-varying experiments remain recorded
in the [archived engineering records](https://gitlab-master.nvidia.com/nccl/nccl-extensions/-/blob/backup/hier-dispatch-pre-squash-20261105/nccl_ep/docs/plc/375-ll-hierarchical-dispatch/README.md), with their frozen
benchmark sources. They are not flags in the production benchmark.

For performance qualification, repeat on the same node pair with baseline and
candidate cases interleaved, recording source commits, installed artifact
hashes, topology, capacity, and timing mode. Retain distributions and investigate
repeatable regressions; a single noisy timing is not a CI failure. Use the
CollectiveX ladder for the broader performance study. Sanitizer and longer
multi-node reuse stress runs also belong to QA, not the bounded unit tier.

### EP 0.3 feature coverage

The default matrix keeps the release plan compact while exercising the public
EP 0.3 controls:

- independent dispatch and combine SM budgets in LL and HT;
- explicit HT `push-pull` and `pull-push` flow selection;
- forced-scan metadata exchange in both NVLink and RDMA configurations;
- required fused metadata exchange in the single-LSA NVLink configuration;
- LL rank-major DSFP8 zero-copy on NVLink;
- staged, AUTO zero-copy, and larger top-k AFD round trips; and
- the Python AFD, `ht_flow`, and `meta_exchange_policy` bindings.

`require-fused` and LL DSFP8 zero-copy are intentionally omitted when
`--hostfile` is present: the former requires a single LSA team and the latter is
an NVLink-only path. Import-handle relocation, dispatch chunk/copy modes, CUDA
graph replay, compact top-k ordering, and configuration precedence remain in
the `unit` suite, where they can be validated without duplicating the full
benchmark matrix.

The `bench` suite also carries the QA-requested boundary scenarios:

- 4100 tokens with HT Flat/Expert-Major and BF16/FP16/FP32 (six cases);
- four ranks with 96 local experts per rank and HT Flat/Expert-Major; and
- multi-node 8- or 16-rank runs with ten experts per rank, pairing
  `(hidden=8192, top-k=8)` and `(hidden=16384, top-k=22)` across both HT
  layouts.

The four-rank cases use their own MPI launch and require at least four ranks in
the selected configuration. The non-power-of-two expert cases run only when a
`--hostfile` is supplied with `--np 8` or `--np 16`. Explicit Expert-Major
receive capacities keep these boundary cases within practical GPU memory while
retaining ample headroom for their deterministic uniform routing.

AFD coverage follows the repository integration matrix: staged and AUTO
zero-copy round trips run with every even `--np` of at least four, while the
larger top-k case uses the supported 32-expert geometry on 4, 8, or 16 ranks.
The `unit` suite additionally inherits `test_ll_afd_selection` from
`tests/run_tests.sh`, covering AFD universal-kernel selection rules.

### EP 0.3 compatibility gates

Treat platform and dependency compatibility as small release gates rather than
repeating `full` everywhere:

- Use the VR/NVL72 target for one normal NVLink `full` run.
- Run `bench` once for each required RDMA or MNNVL transport topology.
- Run the dedicated `build:cft-arm-cu134` and `integration:cft-polyphe` CI jobs. The
  supported job requires the CUDA 13.4 build to execute CFT successfully with
  a matching driver. These jobs use the Slurm-specific
  `nccl_ep/ci/run_cft_ci_tests.sh` flow and are not part of this standalone
  runner.
- Against the lowest supported NCCL version, run `smoke` plus one LL and one HT
  benchmark case on each required architecture instead of repeating `full`.

Run a four-GPU smoke test:

```bash
bash nccl_ep/qa/run_ep_qa.sh \
  --install /path/to/nccl-ep-install \
  --np 4 \
  --suite smoke
```

Run the complete suite on one eight-GPU NVLink domain:

```bash
bash nccl_ep/qa/run_ep_qa.sh \
  --install /path/to/nccl-ep-install \
  --np 8 \
  --unit-gpus 8 \
  --suite full
```

The unit suite is single-node, ignores `--hostfile`, and uses `--unit-gpus`
rather than the multi-node `--np` value. When qualifying multiple hosts, run it
once on each host, then run the MPI suites with a hostfile. A `full` run
performs its unit phase on the launch host before using the hostfile for its
MPI phases.

List or dry-run cases before reserving machines:

```bash
bash nccl_ep/qa/run_ep_qa.sh --suite bench --list
bash nccl_ep/qa/run_ep_qa.sh --suite bench --case 'bench-ll-*' --dry-run
```

Blackwell NVFP4 coverage is opt-in with `--enable-nvfp4`. HT benchmark cases
are enabled by default and can be omitted with `--disable-ht` on unsupported
configurations.

Each executed case receives its own log. The result directory also contains
`environment.txt`, `results.tsv`, and `junit.xml`. The runner continues after
failures unless `--fail-fast` is set, and returns nonzero if any case fails.

## Scope and maintenance

The standalone matrix is derived from the repository's CI suites under
`nccl_ep/ci/tests/` and the pre-nvsubmit runners removed by MR 239. Changes to
CI coverage should update this runner in the same change. `--list` is intended
to make textual coverage comparisons straightforward.

Do not add an implicit build step to this runner. In particular, do not invoke
the top-level `nccl-submodule` target: dependency builds must use explicitly
bounded parallelism and should be prepared before QA execution.
