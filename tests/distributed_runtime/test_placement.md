# Distributed placement contract

- Python: `test_placement.py` checks world-size calculation, invalid counts, and shared role bundles.
- Run: `bash tests/distributed_runtime/test_placement.sh` after resolving `MY_DFS`.
- Historical image tag and Carrot commit: not recorded. Status after relocation:
  local devcloud `PASS` (4 tests) on 2026-09-19; Docker image not applicable.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).
