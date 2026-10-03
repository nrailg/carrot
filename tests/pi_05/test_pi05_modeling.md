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

## 2026-10-03 BF16 inference export: PASS

- 新导出将浮点权重舍入为 BF16，配置 `precision=bfloat16`，strict reload 后 attention 使用 BF16。
- 保存不能改动内存中的 FP32 master；动作投影/tied embedding 的重载值应与 BF16 舍入值严格相同。
- 使用现有 `.sh`，待 Gemini 验证；源码基线 `a9dd5c55` 加未提交改动，实际镜像 tag 待记录。

- 实际结果：50项合并回归 PASS，41.45s，包含该文件的 2 项测试。
- 环境：Gemini mpi-launcher@mpi-1784764303-launcher，H20 两卡；Python3.12.13 / torch2.11.0+cu128 / transformers5.5.4，实际 Docker image tag 未记录。
- Carrot 实际源码：a9dd5c55f04026be313a55f2215346f02784377c 加未提交改动，Mac/GPU 七文件 SHA256 匹配；相关 OpenPI git 215abfb217dbac7d5f1273282331b9b1866c0479（本轮运行 Carrot 源码）。
- 证据：MY_DFS/test-runs/so101_bf16_export_20261003/{regressions_50.log,real_pi05_fsdp.log,source_hashes.json,environment.json}；源码快照同目录。Ruff/compile/bash/diff PASS，worker 原有 I001 未改，单独检查忽略该既有项。

## 2026-10-03 BF16 modeling simplify: PASS

- 按用户要求移除 checkpoint_version、可选 model 参数及旧格式兼容分支；model 为必填参数。
- 保留 BF16 推理导出和 FP32 model/optimizer DCP，无损 round trip 仍需通过；删除两项兼容专用测试。
- 使用现有 runner 验证，基线 a9dd5c55 加未提交修改，Docker tag 未记录。

- 实际定向回归：7 passed, 4 warnings in34.70s，exit0；包含该文件，覆盖单进程无损恢复、两卡toy FSDP与BF16 strict reload。
- H20两卡、torch2.11.0+cu128/transformers5.5.4，镜像tag未记录；源基线a9dd5c55加未提交修改。两修改源码Mac/GPU SHA256一致，Ruff/compile/diff PASS。证据MY_DFS/test-runs/so101_bf16_export_20261003/simplified/regressions_7.log及两源码快照。
