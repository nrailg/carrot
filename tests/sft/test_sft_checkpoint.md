# SFT checkpoint format

- Python: `test_sft_checkpoint.py` checks model/optimizer/scheduler/step round trip and export artifacts.
- Run: `bash tests/sft/test_sft_checkpoint.sh` after resolving `MY_DFS`.
- Historical checkpoint work reports a combined Gemini regression `PASS`; it does not identify
  this file's image tag or exact Carrot commit.
- Status after relocation: `NOT RUN`.
- Carrot commit at relocation: `72d8168540d74e890358003cbbb4886f5c7ca0e4`.
- Docker image for next GPU run: `wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8` (planned, not historical).

## Migrated historical record

- The combined checkpoint regression was `5 passed, 3 warnings in 31.61s`; warnings were
  single-process DCP distributed-initialization notices. It also verified OpenPI root artifacts,
  optimizer/scheduler/step restore, and strict PI0.5 reload. No per-file Docker image or Carrot
  commit was recorded, so neither is inferred here.

## 2026-10-03 BF16 export / FP32 master resume: PASS

- 新版本 checkpoint 的 `optimizer/` DCP 同时保存 model master 与 AdamW；根目录是推理导出。
- 检查 BF16 推理文件、逐位 FP32 master 恢复、moments/scheduler/step 恢复。
- 旧无版本字段的 optimizer-only DCP 保持可读；新版本漏传 model 必须失败。
- 使用现有 `.sh`，源码基线 `a9dd5c55` 加未提交改动，实际镜像 tag 待记录。

- 实际结果：50项合并回归 PASS，41.45s，包含该文件的 6 项测试。
- 环境：Gemini mpi-launcher@mpi-1784764303-launcher，H20 两卡；Python3.12.13 / torch2.11.0+cu128 / transformers5.5.4，实际 Docker image tag 未记录。
- Carrot 实际源码：a9dd5c55f04026be313a55f2215346f02784377c 加未提交改动，Mac/GPU 七文件 SHA256 匹配；相关 OpenPI git 215abfb217dbac7d5f1273282331b9b1866c0479（本轮运行 Carrot 源码）。
- 证据：MY_DFS/test-runs/so101_bf16_export_20261003/{regressions_50.log,real_pi05_fsdp.log,source_hashes.json,environment.json}；源码快照同目录。Ruff/compile/bash/diff PASS，worker 原有 I001 未改，单独检查忽略该既有项。

## 2026-10-03 SFT checkpoint simplify: PASS

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
