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

## 2026-10-03 BF16 export / FP32 master FSDP resume: PASS

- 两卡 toy FSDP 导出 BF16 safetensors；通过 DCP 恢复原始 master 后输出与保存前逐位相同。
- AdamW shard、scheduler/step、再次保存继续校验。
- 使用现有 `.sh`，源码基线 `a9dd5c55` 加未提交改动，实际镜像 tag 待记录。

- 实际结果：50项合并回归 PASS，41.45s，包含该文件的两卡 toy FSDP 测试。
- 环境：Gemini mpi-launcher@mpi-1784764303-launcher，H20 两卡；Python3.12.13 / torch2.11.0+cu128 / transformers5.5.4，实际 Docker image tag 未记录。
- Carrot 实际源码：a9dd5c55f04026be313a55f2215346f02784377c 加未提交改动，Mac/GPU 七文件 SHA256 匹配；相关 OpenPI git 215abfb217dbac7d5f1273282331b9b1866c0479（本轮运行 Carrot 源码）。
- 证据：MY_DFS/test-runs/so101_bf16_export_20261003/{regressions_50.log,real_pi05_fsdp.log,source_hashes.json,environment.json}；源码快照同目录。Ruff/compile/bash/diff PASS，worker 原有 I001 未改，单独检查忽略该既有项。

## 2026-10-03 two-GPU FSDP simplify: PASS

- 按用户要求移除 checkpoint_version、可选 model 参数及旧格式兼容分支；model 为必填参数。
- 保留 BF16 推理导出和 FP32 model/optimizer DCP，无损 round trip 仍需通过；删除两项兼容专用测试。
- 使用现有 runner 验证，基线 a9dd5c55 加未提交修改，Docker tag 未记录。

- 实际定向回归：7 passed, 4 warnings in34.70s，exit0；包含该文件，覆盖单进程无损恢复、两卡toy FSDP与BF16 strict reload。
- H20两卡、torch2.11.0+cu128/transformers5.5.4，镜像tag未记录；源基线a9dd5c55加未提交修改。两修改源码Mac/GPU SHA256一致，Ruff/compile/diff PASS。证据MY_DFS/test-runs/so101_bf16_export_20261003/simplified/regressions_7.log及两源码快照。

## 2026-10-03 optimizer-only resume: PASS

- 用户接受 BF16 舍入损失；删除额外 model DCP 保存/恢复，续训模型只从 BF16 safetensors 加载。
- 期望模型参数/输出为 BF16 舍入后的值，optimizer/scheduler/step 仍需完整恢复。
- 基线 a9dd5c55 加未提交修改；使用现有 runner，镜像 tag 未记录。

- 实际结果：7 passed, 4 warnings in34.78s，exit0；涵盖单进程checkpoint、两卡toy FSDP及BF16 strict reload。
- 环境：H20两卡/torch2.11.0+cu128/transformers5.5.4，镜像tag未记录，Carrot a9dd5c55加未提交改动，Mac/GPU五源码SHA256一致。Ruff/compile/diff PASS；证据MY_DFS/test-runs/so101_bf16_export_20261003/optimizer_only/regressions_7.log及源码快照。
