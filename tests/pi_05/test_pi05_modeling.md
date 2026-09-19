# PI0.5 model contract

- Python: `test_pi05_modeling.py` checks batch masks, loss shapes, and OpenPI-format checkpoint export/reload.
- Run: `bash tests/pi_05/test_pi05_modeling.sh` after resolving `MY_DFS`.
- Historical combined result: `5 passed` with parallelizer tests; separate checkpoint-layout
  evidence covers export. Exact per-file image tag and Carrot commit were not recorded.
- Status after relocation: local devcloud `PASS` (2 tests) on 2026-09-19; Docker image not applicable.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## Migrated historical records

- Strict official checkpoint loading had no missing/unexpected keys, one-step output `(1, 50, 32)`
  was finite, and BF16 gradient-checkpointing forward/backward had finite loss/gradients; the
  complete Ray+FSDP2 one-step SFT smoke passed with loss `0.218750` and grad norm `3.904898`.
- The modeling/parallelizer combined regression was `5 passed`; no per-file image tag or Carrot
  commit was recorded.
- OpenPI root export/strict reload coverage and the real two-GPU details are recorded in the
  checkpoint documents, without inventing an image or commit.
