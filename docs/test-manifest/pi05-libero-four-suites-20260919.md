# Carrot PI0.5 LIBERO 四套 benchmark 测试记录

## 测试思路

- 使用已通过 spatial 495/500 的 Carrot PyTorch policy，按官方 evaluator 分别运行 spatial、object、goal、LIBERO-10。
- 固定 seed 7、replan steps 5、每 task 50 trials；每套 10 tasks，合计 2,000 episodes。
- 顺序使用物理 GPU 0 作为 policy、GPU 1 作为唯一 EGL renderer，逐套保留 JSONL、日志和视频。
- 每套核对 500 个唯一 `(task_id, episode_idx)`，汇总逐 task、逐 suite 和 2,000 条总成功率，与官方已记录的 1938/2000 对照。

## 测试代码

- `examples/libero/test_pi05_libero_inference.sh`：选择 suite、运行闭环并核对 500 个唯一记录。
- `tests/test_pi05_inference.py`：LIBERO quantile 和动作输出的 CPU contract 回归。

## 执行步骤

```bash
for suite in libero_spatial libero_object libero_goal libero_10; do
  TASK_SUITE="$suite" TASK_ID=all EPISODES=50 MIN_SUCCESSES=0 \
  DGUARD_STOP_MINUTES=180 \
  OUTPUT_ROOT="${MY_DFS}/benchmarks/carrot-pi05-libero-four-suites/${suite}" \
  bash examples/libero/test_pi05_libero_inference.sh || break
done
```

## 结果

- `NOT RUN`：等待 Gemini 会话连接及 Luna xHigh 运行。
