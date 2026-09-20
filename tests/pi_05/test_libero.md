# PI0.5 LIBERO closed-loop benchmark

## Contract

- `test_libero.py` validates suite identity, Boolean success values, unique
  `(task_id, episode_idx)` coverage, and the required success count.
- `test_libero.sh` runs CPU regression, EGL preflight, Carrot policy server, and the official
  OpenPI evaluator. Policy and renderer must occupy different physical GPUs; H20 permits at most
  one renderer per GPU.
- Reproducibility settings: `seed=7`, `replan_steps=5`, `50 trials/task`, `10 tasks/suite`.
  A full four-suite run contains 2,000 episodes.

## Prerequisites and execution

In the Gemini shell, resolve `MY_DFS` with `wx-detect-my-dfs`. The runner asserts that the
downloaded checkpoint, tokenizer, OpenPI checkout, LIBERO environment, and dguard exist; it does
not download or install them.

```bash
# Single-task preflight, 10 episodes.
bash tests/pi_05/test_libero.sh

# Full suite, 500 episodes and the 97% threshold.
TASK_ID=all EPISODES=50 MIN_SUCCESSES=485 \
OUTPUT_ROOT="${MY_DFS}/benchmarks/carrot-pi05-libero-spatial" \
bash tests/pi_05/test_libero.sh

# Four suites, each with exactly 500 unique episode records.
for suite in libero_spatial libero_object libero_goal libero_10; do
  TASK_SUITE="$suite" TASK_ID=all EPISODES=50 MIN_SUCCESSES=0 \
  DGUARD_STOP_MINUTES=180 \
  OUTPUT_ROOT="${MY_DFS}/benchmarks/carrot-pi05-libero-four-suites/${suite}" \
  bash tests/pi_05/test_libero.sh || break
done
```

## Provenance and progress

- Carrot checkout at reorganization: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Official OpenPI commit: `215abfb217dbac7d5f1273282331b9b1866c0479`.
- LIBERO commit: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
- Available image: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8`; build, smoke,
  and registry push passed (digest `sha256:9cb9a54d20c113b967bd0d464f63eb7446693bf5c56363d8f2417a47c67f71ae`).
  The image actually used by the four-suite run has not been verified.
- Historical Carrot task-0 gate: `10/10`; historical spatial run: `495/500`.
  Their Docker image and exact Carrot commits were not recorded in the source notes.
- Official JAX checkpoint baseline: spatial `489/500`, object `497/500`, goal `487/500`,
  LIBERO-10 `465/500`; total `1938/2000`.
- The four-suite run completed on 2026-09-19; final results and artifacts are recorded below.
  Its actual runtime image tag and remote Carrot commit remain unverified.

Historical details below are retained as compact provenance for the benchmark and its related
smoke/health runs; the private `carrot_pi05_libero_sft_plan.md` memory is not a test result.

## 2026-09-19 local reorganization check

- `PASS`: `test_libero.py` synthetic JSONL validation, `3 passed` on local devcloud.
- `NOT RUN`: simulator rollout and GPU health; those require the Gemini runner.
- Carrot commit at check: `72d8168540d74e890358003cbbb4886f5c7ca0e4`;
  Docker image: not applicable (local devcloud).

## Migrated related benchmark records

- The official JAX baseline is not a Carrot result: OpenPI
  `215abfb217dbac7d5f1273282331b9b1866c0479`, LIBERO
  `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`, `1938/2000` overall (`489/500`, `497/500`,
  `487/500`, `465/500` by suite), with metrics under
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/openpi-libero/pi05-libero-official-215abfb-20260916/`.
- The JAX and eager PyTorch quick smoke each completed `10/10`; the default `torch.compile`
  attempt timed out and is retained as a warning, not the result. Evidence is under
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/openpi-libero/pi05-libero-jax-pytorch-smoke-20260917`.
- An external OpenPI renderer-health run, not a Carrot unit test, completed eight serial task-0
  runs at `1/1` with no new Xid. Evidence is under
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/openpi-libero/libero_spatial-task0-1x-20260917-12*`
  (eight directories spanning `120554`–`121527`); no Carrot commit or Docker image was recorded.
- The OpenPI PyTorch spatial 100x experiment was a `FAIL`: task 0–7 targeted 100 episodes each,
  but only 15/800 completed before all clients exited 134, with new H20 Xid 31. Evidence is under
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/openpi-libero/pi05-libero-pytorch-spatial-task0-7-100x-20260917`.
  The proposed safe topology was four policy/renderer GPU pairs in two waves; this is not a
  Carrot benchmark pass.
- The archived four-suite plan was marked `NOT RUN` before the later run documented below.

## 2026-09-19 four-suite progress snapshot (superseded)

The runner contract specifies `seed=7`, `replan_steps=5`, and 50 trials/task. The read-only
Gemini check found policy on physical GPU 0 and the sole EGL renderer on physical GPU 1.
Completed suites each had 500 unique `(task_id, episode_idx)` records and 500 videos:

| Suite | Success | Status |
|---|---:|---|
| `libero_spatial` | 493/500 | complete |
| `libero_object` | 490/500 | complete |
| `libero_goal` | 487/500 | complete |
| `libero_10` | 373/385 | running; 385 unique records and videos |

At this snapshot the policy server, evaluator, and orchestrator were alive; Xid count was 0.
The `libero_10` score is partial and must not be reported as a `/500` result. Final exit codes,
full-run aggregate, actual runtime image, and remote Carrot commit remain unverified; the
remote source copy has no `.git` metadata. The build digest above is not proof of the image
used by this benchmark. This snapshot is superseded by the final check below.

## 2026-09-19 four-suite final result

The read-only final check found exactly 500 unique `(task_id, episode_idx)` records and 500
videos in each suite, with `E2E_OK` and `PASS`. The main orchestrator exited with code 0;
evaluator and policy processes had exited. Xid count remained 0, and dguard had recovered
to `DGUARD_WATCH=1`.

| Suite | Carrot | Official JAX | Carrot minus official |
|---|---:|---:|---:|
| `libero_spatial` | 493/500 | 489/500 | +4 |
| `libero_object` | 490/500 | 497/500 | -7 |
| `libero_goal` | 487/500 | 487/500 | 0 |
| `libero_10` | 461/500 | 465/500 | -4 |
| Total | 1931/2000 (96.55%) | 1938/2000 (96.90%) | -7 (-0.35 pp) |

Per-task successes, task IDs 0–9 in order:

- Spatial: `50, 50, 50, 48, 49, 49, 49, 50, 49, 49`.
- Object: `49, 50, 49, 46, 48, 49, 49, 50, 50, 50`.
- Goal: `48, 49, 50, 46, 48, 48, 50, 50, 50, 48`.
- LIBERO-10: `47, 50, 47, 48, 50, 49, 47, 50, 30, 43`.

Evidence root: `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/carrot-pi05-libero-four-suites/`.
The run directories are `libero_spatial/20260919-024341`,
`libero_object/20260919-040444`, `libero_goal/20260919-053557`, and
`libero_10/20260919-065422`. Each contains `episodes.jsonl`, `videos/`, `eval.log`,
`server.log`, and `render-smoke.log`.

Spatial exceeds the 485/500 acceptance threshold. The aggregate is seven successes below
the official JAX result; Object and LIBERO-10 account for the deficits, with LIBERO-10 task 8
at 30/50. This establishes close closed-loop behavior, not strict numerical parity. The
runtime image tag and Carrot commit could not be recovered from the container or its `.git`-less
source copy; do not attribute this run to the available v1.10 image without separate evidence.

## 2026-09-19 Spatial SFT checkpoint comparison

On Gemini `mpi-1760058051-launcher`, compare SFT steps 100, 200, and 300, then the
official PyTorch base checkpoint. The unchanged runner uses `TASK_SUITE=libero_spatial`,
`TASK_ID=all`, `EPISODES=2`, and `MIN_SUCCESSES=0`: two official initial states for each
of the ten tasks, 20 episodes per checkpoint. Seed 7 and `replan_steps=5` are fixed by
the runner. Policy GPU 0 and EGL renderer GPU 1 are isolated; runs are serial.
The local Carrot source synced before this run was clean at
`7ea4ad8cc25784b24dc7f2cccab9fb3943760f98`. The actual remote image tag and OpenPI
commit could not be recovered from the remote `.git`-less checkout. The evaluator
`examples/libero/main.py` was byte-identical to the local OpenPI `89d9220` file
(SHA-256 `6dfddcdb25087f9714afe5ce39d03d2710be9cdb359fffd3a4eddf9c32460a50`).

```bash
for step in 00000100 00000200 00000300; do
  MY_DFS="$MY_DFS" TASK_SUITE=libero_spatial TASK_ID=all EPISODES=2 MIN_SUCCESSES=0 \
  DGUARD_STOP_MINUTES=180 \
  CHECKPOINT_DIR="$MY_DFS/experiments/carrot/pi05_libero_sft_2k_20260919/checkpoints/step-$step" \
  OUTPUT_ROOT="$MY_DFS/benchmarks/carrot-pi05-libero-sft-spatial-20/step-$step" \
  bash tests/pi_05/test_libero.sh
done

# Attempted only after the three SFT evaluations finished.
MY_DFS="$MY_DFS" TASK_SUITE=libero_spatial TASK_ID=all EPISODES=2 MIN_SUCCESSES=0 \
DGUARD_STOP_MINUTES=180 \
CHECKPOINT_DIR="$MY_DFS/hf-hub/Physical-Intelligence/pi05_base_pytorch" \
OUTPUT_ROOT="$MY_DFS/benchmarks/carrot-pi05-libero-sft-spatial-20/pi05_base_pytorch" \
bash tests/pi_05/test_libero.sh
```

The base model has `action_horizon=50` versus SFT `action_horizon=10`; all four use the
same LIBERO normalization stats (SHA-256
`b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`).
Only the three SFT checkpoints produced closed-loop scores; the attempted base comparison
is not a weight-only ablation.

| Checkpoint | Status | Successes | Evidence |
|---|---|---:|---|
| SFT step 100 | PASS | 5/20 | `benchmarks/carrot-pi05-libero-sft-spatial-20/step-00000100/20260919-132628/episodes.jsonl` |
| SFT step 200 | PASS | 15/20 | `benchmarks/carrot-pi05-libero-sft-spatial-20/step-00000200/20260919-133430/episodes.jsonl` |
| SFT step 300 | PASS | 19/20 | `benchmarks/carrot-pi05-libero-sft-spatial-20/step-00000300/20260919-134053/episodes.jsonl` |
| `pi05_base_pytorch` | FAIL before rollout | — | `benchmarks/carrot-pi05-libero-sft-spatial-20/pi05_base_pytorch/20260919-134745/server.log` |

Evidence paths in the table are relative to the current `MY_DFS` on Gemini. Each SFT runner
exited 0; independent `test_libero.py` checks confirmed exactly 20 unique
`(task_id, episode_idx)` records, the suite name, and Boolean success values. Per-task
successes for task IDs 0–9 were:

| Checkpoint | Per-task successes, each out of 2 |
|---|---|
| SFT step 100 | 0, 0, 2, 0, 1, 1, 0, 0, 0, 1 |
| SFT step 200 | 2, 1, 2, 2, 2, 1, 2, 0, 1, 2 |
| SFT step 300 | 2, 2, 2, 2, 1, 2, 2, 2, 2, 2 |

The base runner exited 1 before creating `episodes.jsonl`: Carrot's LIBERO policy factory
raises `ValueError: LIBERO requires action_horizon=10 and model action_dim >= 7`, while the
base checkpoint `config.json` has horizon 50. Thus base has **no valid /20 score** from this
entry point. After all attempts, dguard was back at `DGUARD_WATCH=1`, no evaluator/server
remained, and no new Xid was observed. The user chose not to pursue another base entry point
or modify the Carrot policy factory, so no base success rate is reported.

## 2026-09-19 SFT checkpoints versus official JAX (running)

Evaluate the new SFT checkpoints at steps 100, 300, and 1000 on two Gemini
launchers with the existing Carrot server and OpenPI LIBERO evaluator.
Training was stopped after log step 1470; the last completed checkpoint is step 1400.
The three selected checkpoints have nonempty model weights, config, trainer state,
training YAML, and LIBERO norm stats. The source synced before this benchmark matches
local Carrot `ab55aa343f87996d73d47e63dc7186bb5348be85`; remote checkouts have no
Git metadata, so their full commit identity is not independently recoverable.

Run one suite at a time on each launcher, but use the two launchers concurrently;
their physical GPU UUIDs differ. For each suite, `TASK_ID=all`
and `EPISODES=50` cover all ten tasks and 500 official init states. The runner
fixes seed 7, `replan_steps=5`, policy GPU 0, and EGL renderer GPU 1.
`mpi-1759754893-launcher` takes step 100 (all four suites) plus step 1000
Spatial/Object; `mpi-1760058051-launcher` takes step 300 (all four suites)
plus step 1000 Goal/LIBERO-10. Thus each H20 has at most one renderer.
`MIN_SUCCESSES=0` checks result completeness without imposing a success-rate
threshold on early SFT checkpoints. Each successful run must contain 500 unique
`(task_id, episode_idx)` records and 500 videos, exit 0, and have no new Xid.
The per-host job lists and fail-fast preflight are in `run_libero_sft_benchmark.sh`;
each suite pauses dguard for at most 360 minutes and restores it on exit.

`mpi-1759754893-launcher` (`__HOST_IP__=29.209.160.111`):

```bash
MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
MY_DFS="$MY_DFS" bash "$MY_DFS/work/carrot/tests/pi_05/run_libero_sft_benchmark.sh" \
  old 29.209.160.111
```

`mpi-1760058051-launcher` (`__HOST_IP__=29.209.163.23`):

```bash
MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
MY_DFS="$MY_DFS" bash "$MY_DFS/work/carrot/tests/pi_05/run_libero_sft_benchmark.sh" \
  new 29.209.163.23
```

Official JAX reference (same four suites, 50 episodes/task):

| Suite | Official JAX |
|---|---:|
| `libero_spatial` | 489/500 |
| `libero_object` | 497/500 |
| `libero_goal` | 487/500 |
| `libero_10` | 465/500 |
| Total | 1938/2000 (96.90%) |

For each SFT checkpoint, report its score by suite and total, then subtract
the corresponding Official JAX score. Do not substitute the earlier Carrot
1931/2000 result for the official reference. Current status: **RUNNING**.
The two remote background tasks are `4d9b96d8-0245` (old launcher) and
`587999ab-0111` (new launcher). Both passed the 29-passed/1-skipped CPU checks
and H20 EGL smoke check, and reached the first Spatial suite's policy-server
startup. Persistent output root:
`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/carrot-pi05-libero-sft-four-suites-20260919/`.
The result table will be filled only from validated persistent JSONL and exit codes.

### 2026-09-20 read-only progress snapshot

The benchmark remains **RUNNING**.

| Suite | step 100 | step 300 | step 1000 | Official JAX |
|---|---:|---:|---:|---:|
| Spatial | 167/500 | 429/500 | running | 489/500 |
| Object | 330/500 | 485/500 | running | 497/500 |
| Goal | 199/500 | 370/500 | 453/500 | 487/500 |
| LIBERO-10 | running | 229/500 | running | 465/500 |
| Total | running | 1513/2000 (75.65%) | running | 1938/2000 (96.90%) |

Step 300 has completed all four suites and totals `1513/2000 (75.65%)`, versus
Official JAX `1938/2000 (96.90%)`: `-425` successes (`-21.25 pp`). The overall
three-checkpoint benchmark remains incomplete. The actual container image tag and
remote OpenPI checkout commit were not recorded for this run.

Both remote tasks were still running at this snapshot: old launcher task
`4d9b96d8-0245`, new launcher task `587999ab-0111`. No OOM or Xid was seen in
the checked logs; this is a log observation, not a hardware-wide health claim.
Completed suites had 500 JSONL records and matching videos in the persistent
evidence root:
`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/carrot-pi05-libero-sft-four-suites-20260919/`.

### 2026-09-20 relocation to the new launcher

At the user's request, the old launcher's background task `4d9b96d8-0245` was
stopped to free that host. Its step-100 LIBERO-10 run stopped at 434/500 records;
the partial JSONL and videos remain under
`step-00000100/libero_10/20260919-225758/` and are **not a benchmark score**.
No benchmark policy/evaluator processes remained on the old host, and dguard
reported `DGUARD_WATCH=1` after the stop.

The new launcher continues its original task `587999ab-0111` (step-1000
LIBERO-10 at the time of relocation). A separate background task
`e81633a2-0125` runs `run_libero_sft_relocated.sh 125759`: it waits for the
original new-host runner to exit, then serially reruns step-100 LIBERO-10 and
runs step-1000 Spatial/Object. The same `test_libero.sh` writes each rerun
under a new timestamp directory within the existing suite output root, so the
partial run is preserved. The queue is **RUNNING/WAITING**, not completed; it
does not share an H20 renderer or policy port with the original task.
