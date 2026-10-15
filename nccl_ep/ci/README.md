# CI test suites

The EP child pipeline runs all test jobs through nvsubmit 0.3.5 and publishes
JUnit results. Log paths are configured separately for each cluster through
`EOS_NVSUBMIT_LOG_DIR`, `PRE_TYCHE_NVSUBMIT_LOG_DIR`, `PRE_NYX_NVSUBMIT_LOG_DIR`,
and `IPP6_NVSUBMIT_LOG_DIR`. EOS, Pre-Tyche, and Pre-Nyx store job logs under
`/lustre/fsw/coreai_libraries_nccl/gitlab-runner/logs/nccl_ep/${CI_PIPELINE_ID}/${CI_JOB_ID}`.
IPP6 uses `/storage/gitlab-runner/logs/nccl_ep/${CI_PIPELINE_ID}/${CI_JOB_ID}`.
Each cluster's setup job installs nvsubmit once in `nvsubmit-venv/` alongside
the copied `build/` directory and publishes its path to the test jobs.
Cluster cleanup jobs remove the pipeline's install directory only after that
cluster's build, setup, and test jobs succeed, including the Pre-Tyche CFT test.
Failed runs retain their installs, and logs remain in the separate log directories.
The suite definitions are in `tests/`:

- `unit.yml`: all 19 GoogleTest binaries, expanded into 70 binary/mode cases,
  including HT duplication, scan, unfused-count, pull-push, and filtered
  overflow/drop coverage.
- `integration.yml`: the C++ correctness and benchmark matrix.
- `python.yml`: all 28 Python sanity cases, using the wheel and interpreter
  prepared by `python/ci/prepare_ep_test_env.sh`.

Unit cases use `run_unit_test.sh` with four Slurm tasks sharing a bootstrap
UID file. The wrapper passes Slurm rank and world size to the existing
`--rank` and `--nranks` arguments, and supplies `--uid-file` with a path
containing the binary name and Slurm job and step IDs to isolate concurrent
runs. Test arguments, including GoogleTest filters, are forwarded unchanged. The LL epoch protocol uses one rank and the LL group
epoch uses exactly four. Cluster configuration is in `systems/`; IPP6 and
Pre-Nyx test jobs remain disabled.

For local unit runs, build with `make -C nccl_ep/tests`, then run the complete
matrix with the existing launcher:

```bash
bash nccl_ep/tests/run_tests.sh 4
```
