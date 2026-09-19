# Ray runtime contract

- Python: `test_ray_runtime.py` checks worker state, error propagation, GPU visibility, channel transport, and role placement.
- Run: start Ray according to `carrot-test`, then `bash tests/distributed_runtime/test_ray_runtime.sh` with `MY_DFS` set.
- A historical Isaac/Ray coexistence check listed this test but marked remote execution
  `NOT RUN`; no historical image tag or Carrot commit was recorded.
- Status after relocation: `NOT RUN`.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## Migrated historical record

- The planned serial coexistence check never ran this Ray test remotely. No Docker image,
  Carrot commit, or result was recorded; simulator examples are not represented by this unit test.
