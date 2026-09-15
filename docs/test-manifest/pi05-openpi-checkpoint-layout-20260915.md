# PI0.5 OpenPI checkpoint layout 测试记录

## 测试思路

- 验证 checkpoint 根目录直接产出 OpenPI PyTorch 的 `model.safetensors` 与 `config.json`，
  不再生成模型 DCP 或 `pretrained_model/`。
- 验证 optimizer 独立保存为 DCP，并能连同 scheduler 和 step 完整恢复。
- 验证两卡 FSDP2 下 optimizer shard 能并行保存并恢复，模型根目录导出保持完整权重。
- 验证 PI0.5 导出的 OpenPI 配置与权重可以通过 `PI0Policy.from_pretrained` 严格回读。
- 使用真实 `pi05_base_pytorch` 模型执行两卡 FSDP2 save、根目录 strict load、optimizer
  resume 和二次 save，覆盖实际 PI0.5 参数规模、dtype 与分片拓扑。
- 验证通用 checkpoint 层不再 materialize DTensor；所有 rank 将原始 state dict 交给
  PI0，由 PI0 collective gather，且只有 rank 0 写 OpenPI 文件；模型文件写完前所有
  rank 保持在 PI0 内部 barrier，避免提前进入后续 NCCL collective。

## 测试代码

- `tests/test_sft_checkpoint.py`：断言新目录布局、模型权重、optimizer、scheduler、step
  及附属文件的单进程 round trip。
- `tests/test_sft_checkpoint_distributed.py`：断言 toy 模型在两卡 FSDP2 下的 optimizer
  shard 恢复、模型导出 collective 契约和恢复后二次保存。
- `tests/test_pi05_modeling.py`：断言 PI0.5 OpenPI PyTorch 导出的配置字段、权重文件名
  和严格回读结果。
- `tests/test_pi05_checkpoint_distributed.py`：由环境变量显式启用真实 PI0.5 两卡
  checkpoint round trip；测试产物写入 pytest 临时目录。

## 执行步骤

```bash
export PYTHONPATH="/work/carrot/src:/work/carrot/tests:$PYTHONPATH"
cd /work/carrot
pytest -v -s --timeout=1800 tests/test_sft_checkpoint.py tests/test_pi05_modeling.py
pytest -v -s --timeout=1800 tests/test_sft_checkpoint_distributed.py
CARROT_PI05_OPENPI_PYTORCH_CHECKPOINT=/path/to/pi05_base_pytorch \
  pytest -v -s --timeout=1800 tests/test_pi05_checkpoint_distributed.py
```

## 结果

- PASS：DTensor 导出职责调整后，Gemini H20 组合回归为
  `5 passed, 3 warnings in 31.61s`；warnings 仅为
  单进程 DCP 未初始化 distributed 的提示。
- PASS：真实 `pi05_base_pytorch` 两卡 FSDP2 测试为
  `1 passed in 199.32s (0:03:19)`，覆盖 PI0 内部 DTensor gather 和配对 barrier、
  812-key OpenPI root strict load、FP32 master 权重保存、AdamW DCP probe 恢复和第二次
  完整保存；未出现 NCCL timeout。
- PASS：配对 barrier 调整后的 toy 两卡 FSDP2 测试为 `1 passed in 19.37s`。
- 调试记录：早期 toy 方案使用 PyTorch canonical optimizer state-dict API，第二次保存
  出现 `KeyError: 0`，改为 raw optimizer DCP 后通过。真实 PI0.5 首轮发现 FSDP gather
  展开 tied embedding，strict load 报 unexpected key；显式保留 OpenPI 的 `lm_head`
  canonical key 后通过。
- `git diff --check`、目标文件 `ruff check`、`compileall` 均通过；代码与 CephFS 测试副本
  逐文件比较一致。真实测试生成的 81 GiB pytest 临时产物已删除。
- 本地 pytest 在 collection 阶段因开发机 Python 未安装 `torch` 而停止；相同目标用例已在
  Gemini 的 PyTorch 环境完成上述正式验证。
