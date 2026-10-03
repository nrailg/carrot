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

## 2026-10-03 BF16 export / FP32 master real PI05 resume: PASS

- 两卡真实 PI05 导出 `precision=bfloat16`，strict reload 后 attention Q 参数为 BF16。
- 向 FP32 master 写入 BF16 无法精确表达的 0.123456 探针，验证 DCP 恢复原值而非舍入值。
- 同时检查 AdamW、step、再次保存；不执行额外拟合训练。
- 使用现有 `.sh`，源码基线 `a9dd5c55` 加未提交改动，实际镜像 tag 待记录。

- 实际结果：真实 PI05 两卡 FSDP：1 passed in 228.84s，exit 0。
- 环境：Gemini mpi-launcher@mpi-1784764303-launcher，H20 两卡；Python3.12.13 / torch2.11.0+cu128 / transformers5.5.4，实际 Docker image tag 未记录。
- Carrot 实际源码：a9dd5c55f04026be313a55f2215346f02784377c 加未提交改动，Mac/GPU 七文件 SHA256 匹配；相关 OpenPI git 215abfb217dbac7d5f1273282331b9b1866c0479（本轮运行 Carrot 源码）。
- 证据：MY_DFS/test-runs/so101_bf16_export_20261003/{regressions_50.log,real_pi05_fsdp.log,source_hashes.json,environment.json}；源码快照同目录。Ruff/compile/bash/diff PASS，worker 原有 I001 未改，单独检查忽略该既有项。

## 2026-10-03 optimizer-only resume: NOT RUN

- 用户接受 BF16 舍入损失；删除额外 model DCP 保存/恢复，续训模型只从 BF16 safetensors 加载。
- 期望模型参数/输出为 BF16 舍入后的值，optimizer/scheduler/step 仍需完整恢复。
- 基线 a9dd5c55 加未提交修改；使用现有 runner，镜像 tag 未记录。

- 本轮真实3B两卡测试未重跑（NOT RUN）；相关保存/恢复逻辑已通过两卡toy FSDP回归。此前完整master恢复的PASS不代表当前optimizer-only方案。
