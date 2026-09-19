# PI0.5 inference and embodiment contract

- Python: `test_pi05_inference.py` checks RoboTwin regression, tokenization, artifact validation,
  LIBERO transforms, normalization, masks, and action decoding.
- Run: `bash tests/pi_05/test_pi05_inference.sh` after resolving `MY_DFS`.
- Historical combined result: `24 passed, 1 skipped`; that count includes other files, and the
  image tag and Carrot commit were not recorded.
- Status after relocation: local devcloud `PASS` (22 tests) on 2026-09-19; Docker image not applicable.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## Migrated historical records

- The earlier CPU regression was `20 passed, 1 skipped in 18.82s`; the skip was the
  real-checkpoint GPU smoke. The fixed-seed NumPy/Torch `state (4,14)` and `actions (4,50,14)`
  comparison was elementwise equal.
- `test_pi05_stats.py` could not collect because `datasets` was missing; this is a blocked
  dependency, not a pass. The real RoboTwin checkpoint smoke was not run because no
  checkpoint/session was available.
- The later CPU transform regression was `24 passed, 1 skipped in 18.40s`; the Gemini task-0
  gate was `10/10` with evidence at
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/carrot-pi05-libero-smoke/20260918-153519`.
  Historical Docker image and Carrot commit were not recorded.
