# Distributed SFT checkpoint

- Python: `test_sft_checkpoint_distributed.py` checks toy-model two-GPU FSDP optimizer state round trip.
- Run: set `MY_DFS` on a two-GPU Gemini node and
  `bash tests/sft/test_sft_checkpoint_distributed.sh`.
- Fewer than two CUDA devices now fails the pytest case.
- Historical result: `1 passed in 19.37s` after a paired-barrier adjustment. Its image tag and
  exact Carrot commit were not recorded.
- Status after relocation: `NOT RUN`.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## Migrated historical record

- The toy two-GPU FSDP2 round trip passed as `1 passed in 19.37s` after the paired-barrier
  adjustment. The earlier canonical optimizer state-dict probe failed with `KeyError: 0` and was
  replaced by raw optimizer DCP; that debugging attempt is not counted as a final result.
  Historical image and Carrot commit were not recorded.

## 2026-09-26 H20 check

The test passed within the current tracked-file pytest run (`73 passed,
2 intentionally skipped, 6 asset-dependent tests deselected` overall) on
`mpi-launcher@mpi-1759754893-launcher`, merged Carrot source `eddfffb`.
Missing two-GPU hardware was separately verified to produce `FAILED`, not
`SKIPPED`. Run output:
`${MY_DFS}/experiments/carrot/test-runs/pi05-pr32-20260926T043636Z/tracked_pytest_failfast.log`.
