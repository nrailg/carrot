# PI0.5 distributed checkpoint round trip

- Python: `test_pi05_checkpoint_distributed.py` loads a real PI0.5 checkpoint and verifies a two-GPU FSDP save/reload path.
- Run: set `MY_DFS` and `CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT` to an existing checkpoint, then
  `bash tests/pi_05/test_pi05_checkpoint_distributed.sh`.
- Historical result: real two-GPU test `PASS`. Its image tag and exact Carrot commit were not
  recorded.
- Status after relocation: `NOT RUN`.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## Migrated historical record

- The real `pi05_base_pytorch` two-GPU FSDP2 run passed in `199.32s` and covered PI0 collective
  DTensor gather, paired barriers, 812-key OpenPI strict load, FP32 master-weight export, AdamW
  DCP probe/restore, and a second full save. Historical image and Carrot commit were not recorded.
