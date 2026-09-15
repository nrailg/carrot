# PI0.5 测试注释补充记录

## 测试思路

- 确认只增加测试意图和逻辑块说明，不改变测试行为。

## 测试代码

- `tests/test_pi05_modeling.py`：说明 SFT batch contract、fixture 特殊取值和布局断言。
- `tests/test_model_parallelizer.py`：说明 FSDP mock 边界及 wrapping contract。
- `tests/test_pi05_openpi_parity.py`：说明两套 golden 的角色、严格 gate 和诊断范围。

## 执行步骤

```bash
python -m ruff check tests/test_pi05_openpi_parity.py tests/test_pi05_modeling.py tests/test_model_parallelizer.py
PYTHONPATH=src python -m pytest -q tests/test_pi05_modeling.py tests/test_model_parallelizer.py
PYTHONPATH=src python -m pytest -q tests/test_pi05_openpi_parity.py
```

## 结果

- `PASS`：Ruff 通过；模型与 parallelizer 测试 `5 passed`。
- `NOT RUN`：GPU parity 在本地因未设置 golden 环境变量按预期跳过；本次未改测试逻辑。
