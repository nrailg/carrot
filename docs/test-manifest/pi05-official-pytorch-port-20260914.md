# PI0.5 官方 PyTorch 移植测试记录

## 测试思路

- 验证 Carrot 可直接导入并加载官方 PyTorch checkpoint，不修改 site-packages。
- 验证模型结构、单步推理与 SFT loss/backward 的关键路径。

## 测试代码

- `tests/test_pi05_modeling.py`：验证 SFT batch 与官方模型调用契约。
- `tests/test_model_parallelizer.py`：验证官方模型层级下的 FSDP 单元。
- `tests/test_pi05_openpi_parity.py`：验证 Carrot 与官方 PyTorch 输出。

## 执行步骤

```bash
cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot
PYTHONPATH=src:tests python -m pytest -v -s --timeout=1800 <target tests>
```

## 结果

- 本地静态与契约测试：
  - `ruff check`：通过。
  - `tests/test_pi05_modeling.py tests/test_model_parallelizer.py`：`5 passed`。
  - parity 测试在非 CUDA 本地环境按预期跳过。
- Gemini H20，Transformers 5.5.4：
  - 官方 checkpoint 严格加载通过，missing/unexpected 均为空。
  - 单步采样通过，输出 shape `(1, 50, 32)` 且全部 finite。
  - `tests/test_pi05_openpi_parity.py`：`1 passed in 57.62s`，Carrot 与官方
    PyTorch golden 的数值差异为 0；JAX 仅保留诊断。
  - BF16 图像、gradient checkpointing 下 forward/backward 通过，loss 与梯度
    均 finite，峰值分配显存约 14.59 GB。
- 完整 Carrot Ray + FSDP2 一步 SFT smoke：通过，loss `0.218750`、grad norm
  `3.904898`；测试禁用保存且未产生输出目录。
