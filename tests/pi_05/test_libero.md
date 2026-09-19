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
