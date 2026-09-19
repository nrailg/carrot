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
- Planned image for the next run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8`
  (build pending; do not attribute earlier results to this image).
- Historical Carrot task-0 gate: `10/10`; historical spatial run: `495/500`.
  Their Docker image and exact Carrot commits were not recorded in the source notes.
- Official JAX checkpoint baseline: spatial `489/500`, object `497/500`, goal `487/500`,
  LIBERO-10 `465/500`; total `1938/2000`.
- A later Carrot four-suite run was initiated separately; its final per-task and per-suite
  results, actual image tag, git commit, artifacts, and Xid check have not been verified here.

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
- The archived four-suite plan was marked `NOT RUN`; a later background run was started, but
  its final status is unverified here. No result, image, or commit is attributed to it.
