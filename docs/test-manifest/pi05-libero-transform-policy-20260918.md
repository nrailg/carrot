# PI0.5 LIBERO transform policy 测试记录

## 测试思路

- 验证通用 `Pi05Policy` 严格执行注入的 input/output transforms，不包含 embodiment 分支。
- 验证 RoboTwin 重构前后的 observation、动作解码和错误检查 contract 不变。
- 验证 LIBERO 两路图像、缺失右腕 mask、8D/7D 数据、quantile normalization、prompt-only tokenization 和 `[10,7]` 输出。
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

# Gemini H20：先跑单 renderer EGL smoke，再跑 Carrot server + task 0 的 10 episode gate。
bash examples/libero/test_pi05_libero_inference.sh

# 完整 spatial benchmark：10 tasks x 50 trials，目标至少 485/500。
TASK_ID=all EPISODES=50 MIN_SUCCESSES=485 DGUARD_STOP_MINUTES=120 \
OUTPUT_ROOT=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/carrot-pi05-libero-spatial \
bash examples/libero/test_pi05_libero_inference.sh
```

## 结果

- `PASS`：ruff 检查通过。
- `PASS`：`24 passed, 1 skipped in 18.40s`。
- `PASS`：官方 LIBERO `norm_stats.json` 与本地 PaliGemma tokenizer 的真实资产 transform smoke；得到 state `(32,)`、actions `(10,32)`、三路 `(3,224,224)` 图像、right-wrist mask=false、tokens `(200,)`。
- `SKIP`：真实 RoboTwin checkpoint smoke 未设置 `CARROT_PI05_INFERENCE_CHECKPOINT`。
- `NOT RUN`：官方 LIBERO PyTorch checkpoint 单次 GPU sampling；当前机器无 GPU，且本机可见权重为 root-only。
- `PASS`：quantile 修复后的 CPU regression，`24 passed, 1 skipped`。
- `PASS`：Gemini task 0 10-episode gate，`10/10` 成功、10 条唯一 JSONL 记录，无新增 Xid；证据目录为 `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/carrot-pi05-libero-smoke/20260918-153519`。
- `NOT RUN`：quantile 修复后的完整 spatial 500-episode benchmark。
