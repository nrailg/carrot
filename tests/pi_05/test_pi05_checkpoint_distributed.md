# PI0.5 distributed checkpoint round trip

- Python: `test_pi05_checkpoint_distributed.py` loads a real PI0.5 checkpoint and verifies a two-GPU FSDP save/reload path.
- Run: set `MY_DFS` and `CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT` to an existing checkpoint, then
  `bash tests/pi_05/test_pi05_checkpoint_distributed.sh`.
- Missing checkpoint or fewer than two CUDA devices now fails the pytest case.
- Historical result: real two-GPU test `PASS`. Its image tag and exact Carrot commit were not
  recorded.
- Status after relocation: `NOT RUN`.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## Migrated historical record

- The real `pi05_base_pytorch` two-GPU FSDP2 run passed in `199.32s` and covered PI0 collective
  DTensor gather, paired barriers, 812-key OpenPI strict load, FP32 master-weight export, AdamW
  DCP probe/restore, and a second full save. Historical image and Carrot commit were not recorded.

## 2026-09-26 H20 rerun: PASS

On `mpi-launcher@mpi-1759754893-launcher`, with Carrot commit
`eddfffbda2c2980bf1563266da8275093d137bc2` synced to the personal DFS,
`/opt/venvs/carrot`, and the local `pi05_base_pytorch` checkpoint:

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
export CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT="$MY_DFS/hf-hub/Physical-Intelligence/pi05_base_pytorch"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash tests/pi_05/test_pi05_checkpoint_distributed.sh
```

Result: `1 passed in 197.74s`, exit 0, on two H20 GPUs. Output:
`${MY_DFS}/experiments/carrot/test-runs/pi05-pr32-20260926T043636Z/pi05_checkpoint_distributed.log`.
