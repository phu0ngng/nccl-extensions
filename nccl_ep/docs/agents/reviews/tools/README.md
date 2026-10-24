# Review evidence pack

Scripts that gather mechanical evidence for a review once per change, so the
critics read artifacts instead of reasoning about what a compiler or a test run
would show. They need a CUDA toolkit; the GPU steps are optional. They are run
only through `BUILD_EVIDENCE_SKILL.md`, which asks the user first: nothing here
is started on an agent's own initiative.

    tools/review_prep.sh <base-rev> <head-rev> [out-dir]

writes `out-dir/` (default `review-evidence/` next to the repository) with an
`INDEX.md` that triage and every critic read first. Build and run steps that
cannot work on the machine are skipped and listed as skipped in the index.

| Artifact | Produced by | Settles |
|---|---|---|
| `diff.patch`, `hunks.md` | `hunks.py` | what changed, per file and hunk, with line counts; the short form triage reads to rule areas in or out |
| `build_<side>.log`, `warnings_new.md` | `review_prep.sh`, `warnings_diff.py` | the change compiles with the project flags in `DEBUG=0` and `DEBUG=1`; new compiler warnings |
| `res_usage_diff.md` | `res_usage_diff.py` on `cuobjdump -res-usage` of both `libnccl_ep.so` | registers, spills, shared and stack of every statically compiled kernel, base against head |
| `jit_<side>/variants.tsv`, `jit_diff.md` | `jit_variants.py` on the JIT cache a test run filled, or on sources `jit_enumerate.sh` generates for a fixed configuration matrix when no test ran (the generators' arguments and the kernel enumerators are parsed from the tree's headers by `jit_enumerate_gen.py`, so only a new generator parameter needs a binding there; kernels the matrix does not build are printed as `NOT COVERED`) | which JIT variants compile; `ptxas -v` numbers and SASS instruction counts per variant, base against head. Only a cache filled by a test run says which variants the tests reach; `INDEX.md` states which source the pack used |
| `tests_host_<side>.log` | touched tests that do not include `test_common.h`, run without a GPU | host-only tests (kernel selection, parsing) pass before and after, on any node |
| `tests_<side>.log` | the touched test binaries, run on the GPUs present | the tests pass before and after |
| `sanitizer_<tool>.log` | `run_tests_sanitized.sh` | `memcheck`, `initcheck`, `racecheck`, `synccheck` on the touched tests |
| `mutation_<name>.log` | `mutate_and_run.sh` | whether a test still passes with a given defect applied (the trigger of a test-validity finding) |

What the pack does not do:

- `clang-format`: `.clang-format-ignore` excludes `nccl_ep/`, so formatting is not
  CI-enforced here and the pack does not run it. Formatting findings stay with
  the code-organization critic under `CODING_SKILL.md`.
- Latency or bandwidth: `ptxas -v` and SASS counts are static. A speedup claim
  still needs a measurement; `ep_bench` runs need MPI and are left to the author
  or to a documented manual command.

## Requirements

- `nvcc`, `cuobjdump`, `c++filt` on `PATH` or under `CUDA_HOME`.
- A built NCCL at `NCCL_HOME` (default: the vendored submodule build).
- For the GPU steps: GPUs visible to `nvidia-smi` (the unit tests need 4), and
  `compute-sanitizer` for the sanitizer step.
- Python 3.8+; the scripts use only the standard library.

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `NCCL_HOME` | `third_party/nccl/build` | NCCL build to compile and link against |
| `CUDA_HOME` | `/usr/local/cuda` | toolkit |
| `REVIEW_GPUS` | `nvidia-smi -L | wc -l` | ranks to run tests with; `0` skips every GPU step |
| `REVIEW_TESTS` | touched `tests/test_*.cu` | space-separated test binaries to run (basename) |
| `REVIEW_SANITIZERS` | `initcheck memcheck` | `compute-sanitizer` tools to run; empty skips |
| `REVIEW_ARCH` | from `nvidia-smi` compute capability, else `sm_90` | `-arch` for offline JIT compiles and for `jit_enumerate.sh`, whose CFT variants need `sm_100` or newer with CUDA 13.3+ |
| `REVIEW_SKIP_BUILD` | unset | reuse `out-dir/build_<side>` from an earlier run |
| `REVIEW_JOBS` | `nproc` | `make -j` |

## How critics use it

`REVIEW_METHOD.md` ("Mechanical evidence") says which claims each artifact
settles. A critic cites the artifact and the row (`jit_diff.md`, variant
`ll_dispatch_..._weights7`), not a recomputation. A critic that needs a number
the pack does not hold runs the single script that produces it, with the same
flags, and says so.
