# PI0.5 LIBERO transform policy 测试记录

## 测试思路

- 验证通用 `Pi05Policy` 严格执行注入的 input/output transforms，不包含 embodiment 分支。
- 验证 RoboTwin 重构前后的 observation、动作解码和错误检查 contract 不变。
- 验证 LIBERO 两路图像、缺失右腕 mask、8D/7D 数据、mean/std normalization、prompt-only tokenization 和 `[10,7]` 输出。
- 验证带 actions 的 LIBERO 样本可直接执行共享 input transforms，为后续训练复用锁定接口。

## 测试代码

- `tests/test_pi05_inference.py`：RoboTwin 零回归、transform 顺序、LIBERO input/output、官方 norm stats 布局和 loader contract。
- `tests/test_pi05_inference_checkpoint.py`：真实 RoboTwin checkpoint 固定 noise sampling；未提供环境变量时明确 skip。
- `tests/test_pi05_modeling.py`：PI0.5 batch、loss 和 checkpoint 基础回归。

## 执行步骤

```bash
python -m ruff check \
  src/carrot/models/pi05/transforms.py \
  src/carrot/models/pi05/preprocessing.py \
  src/carrot/models/pi05/embodiments \
  src/carrot/models/pi05/inference/policy.py \
  src/carrot/models/pi05/inference/policy_config.py \
  src/carrot/models/pi05/inference/__init__.py \
  src/carrot/cli/serve_pi05_policy.py \
  tests/test_pi05_inference.py \
  tests/test_pi05_inference_checkpoint.py \
  tests/test_pi05_modeling.py

export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH}"
python -m pytest -q \
  tests/test_pi05_modeling.py \
  tests/test_pi05_inference.py \
  tests/test_pi05_inference_checkpoint.py

# Gemini H20：先跑单 renderer EGL smoke，再跑 Carrot server + 1 episode LIBERO 闭环。
bash examples/libero/test_pi05_libero_inference.sh
```

## 结果

- `PASS`：ruff 检查通过。
- `PASS`：`24 passed, 1 skipped in 18.40s`。
- `PASS`：官方 LIBERO `norm_stats.json` 与本地 PaliGemma tokenizer 的真实资产 transform smoke；得到 state `(32,)`、actions `(10,32)`、三路 `(3,224,224)` 图像、right-wrist mask=false、tokens `(200,)`。
- `SKIP`：真实 RoboTwin checkpoint smoke 未设置 `CARROT_PI05_INFERENCE_CHECKPOINT`。
- `NOT RUN`：官方 LIBERO PyTorch checkpoint 单次 GPU sampling；当前机器无 GPU，且本机可见权重为 root-only。
- `NOT RUN`：Gemini H20 单 renderer smoke 和 Carrot server + LIBERO 单 episode 闭环，等待用户审阅脚本。
