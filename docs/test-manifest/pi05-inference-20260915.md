# PI0.5 本地推理测试记录

## 测试思路

验证 train/infer observation 一致、动作正反变换、输入不被修改和 checkpoint 加载契约。

## 测试代码

- tests/test_pi05_inference.py：固定动作 oracle、分桶边界、三相机映射、缺失 artifact。
- tests/test_pi05_modeling.py、tests/test_pi05_stats.py：训练预处理和导出回归。
- tests/test_pi05_inference_checkpoint.py：真实 checkpoint 完整采样、有限输出及固定 noise 可重复性。

## 执行步骤

本地 CPU：

```bash
PYTHONPATH=src python -m pytest -q tests/test_pi05_inference.py tests/test_pi05_inference_checkpoint.py tests/test_pi05_modeling.py
```

单独对比 HEAD 的 robotwin_preprocess 与新实现：固定 seed=7，NumPy/Torch 输入
state (4,14)、actions (4,50,14)，逐元素严格相等。

GPU（尚未执行；同步代码到 Gemini 后指定真实 RoboTwin SFT export）：

```bash
CARROT_PI05_INFERENCE_CHECKPOINT=/path/to/sft/export PYTHONPATH=src python -m pytest -q tests/test_pi05_inference_checkpoint.py
```

## 结果

- PASS：20 passed, 1 skipped in 18.82s；skip 是真实 checkpoint GPU smoke。
- PASS：HEAD/new 的 NumPy/Torch state、actions 均逐元素相等。
- PASS：改动的 Python 文件 ruff check；git diff --check。
- BLOCKED：原 test_pi05_stats.py 收集时缺 datasets，未执行，不算通过。
- NOT RUN：真实 checkpoint GPU smoke；Gemini remote_status 没有活动会话，未提供目标 checkpoint。
- 此版仅支持 robotwin_preprocess 训练的三相机、14 维 joint checkpoint；
  smoke 不代表 RoboTwin rollout success rate。WebSocket/evaluator 属于后续阶段。
