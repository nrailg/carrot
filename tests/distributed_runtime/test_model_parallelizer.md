# Model parallelizer contract

- Python: `test_model_parallelizer.py` checks disabled FSDP, missing process group, mixed precision, and PI0.5 layer wrapping.
- Run: `bash tests/distributed_runtime/test_model_parallelizer.sh` after resolving `MY_DFS`.
- Historical result: `5 passed` together with PI0.5 modeling tests; the historical run did not
  give a separate per-file count, image tag, or Carrot commit.
- Status after relocation: local devcloud `PASS` (4 tests) on 2026-09-19; Docker image not applicable.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## Migrated historical records

- The official-model FSDP unit path passed together with modeling coverage; strict checkpoint
  loading had no missing/unexpected keys. The exact per-file image and Carrot commit were not
  recorded.
- The combined modeling/parallelizer regression was `5 passed`; it does not justify a separate
  count for this file.
