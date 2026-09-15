# PI0.5 旧 parity 测试清理记录

## 测试思路

- 删除只服务于 OpenGiga/LeRobot 兼容的测试和转换工具。
- 确认官方 OpenPI PyTorch parity 测试仍可收集，且仓库没有悬空引用。

## 测试代码

- `tests/test_pi05_openpi_parity.py`：保留的官方 PyTorch 严格 parity gate。

## 执行步骤

```bash
rg -n "pi05_checkpoint_utils|test_pi05_checkpoint_parity|test_pi05_inference_parity|convert_open_giga_pi05" .
python -m ruff check tests/test_pi05_openpi_parity.py
PYTHONPATH=src python -m pytest -q tests/test_pi05_openpi_parity.py
```

## 结果

- `PASS`：旧测试、mapping helper 和转换脚本已删除，仓库无活动代码悬空引用。
- `PASS`：保留的官方 parity 测试 Ruff 通过且收集成功；本地因未设置 GPU golden
  环境变量按预期跳过。
