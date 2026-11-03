# CI test suites

The EP child pipeline runs all test jobs through nvsubmit 0.3.7 and publishes
JUnit results. Each nvsubmit job uses `--failed-testsuite failed-testsuite.yml`
to write a suite only when tests fail, and archives the file when present.
Log paths are configured separately for each cluster through
`EOS_NVSUBMIT_LOG_DIR`, `PRE_TYCHE_NVSUBMIT_LOG_DIR`, and `PRE_NYX_NVSUBMIT_LOG_DIR`;
all three store job logs under
`/lustre/fsw/coreai_libraries_nccl/gitlab-runner/logs/nccl_ep/${CI_PIPELINE_ID}/${CI_JOB_ID}`.
Each cluster's setup job installs nvsubmit once in `nvsubmit-venv/` alongside
the copied `build/` directory and publishes its path to the test jobs.
Cluster cleanup jobs remove the pipeline's install directory only after that
cluster's setup, unit, and integration jobs succeed, including the Pre-Tyche CFT test.
Failed runs retain their installs, and logs remain in the separate log directories.
The suite definitions are in `tests/`:

- `unit.yml`: GoogleTest binaries and the host-only LL dispatch policy test,
  including HT duplication, scan, unfused-count, pull-push, and filtered
  overflow/drop coverage, and the Lamport combine.
- `integration.yml`: the C++ correctness and benchmark matrix.
- `python.yml`: all 28 Python sanity cases, using the wheel and interpreter
  prepared by `python/ci/prepare_ep_test_env.sh`.

Unit cases use `run_unit_test.sh` with four Slurm tasks sharing a bootstrap
UID file. The wrapper passes Slurm rank and world size to the existing
`--rank` and `--nranks` arguments, and supplies `--uid-file` with a path
containing the binary name and Slurm job and step IDs to isolate concurrent
runs. Test arguments, including GoogleTest filters, are forwarded unchanged. The LL epoch protocol uses one rank and the LL group
epoch uses exactly four. Cluster configuration is in `systems/`; the
Pre-Nyx test jobs are disabled.

For local unit runs, build with `make -C nccl_ep/tests`, then run the complete
matrix with the existing launcher:

```bash
bash nccl_ep/tests/run_tests.sh 4
```

## Hierarchical dispatch: CI versus QA

CI checks deterministic invariants without multi-node performance gates:

- `test_ll_dispatch_policy` runs once per unit job without a communicator. It
  covers fitted choices, extrapolation, input bounds, and restricted-QP fallback.
  The current GPU-oriented runner still reserves one GPU for this host-only case.
- `test_ll_kernel_selection` includes the host policy/protocol checks: capacity
  geometry, receive selection, local send selection, delivery masks, epochs,
  completion counts, and peer/context indexing.
- `test_ll_epoch_protocol` uses one GPU for epoch rollover and bounded shared
  control reuse stress. It makes no network or timing assertion.

Multi-node transport correctness, mixed sender modes, QP reuse, graph replay,
and paired performance comparisons belong to the opt-in
[`hier` QA suite](../qa/README.md#hierarchical-dispatch-qualification).
Existing integration CI remains unchanged; the new suite is not added to every
merge-request pipeline.
