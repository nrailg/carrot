# OpenPI LIBERO 官方 Benchmark 测试记录

## 测试思路

- 使用官方 JAX `pi05_libero` checkpoint 和 checkpoint 自带的 LIBERO norm stats，复现官方闭环 success rate。
- 先验证 checkpoint 加载、单次推理和单 episode EGL rollout，再运行四个官方 suite。
- 每张 H20 最多绑定一个 LIBERO rendering 进程；各进程使用独立端口、EGL device 和输出目录。
- checkpoint、逐 episode 结果、聚合 metrics、日志和视频全部写入 Distributed FS。

## 测试代码

- `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/openpi/examples/libero/main.py`：官方 observation、action replanning、初始状态和成功判定基线。
- 仅增加逐 episode JSONL 输出和包含 task/episode id 的视频文件名，不修改 policy 输入、环境步进、初始状态或 success contract。

## 执行步骤

```bash
# 1. 严格加载 DFS 上的官方 checkpoint，并验证一次 policy inference。
# 2. 每张 H20 启动不超过一个 EGL renderer，运行单 episode preflight。
# 3. 运行 libero_spatial、libero_object、libero_goal、libero_10；每 task 50 episodes。
# 4. 从 DFS JSONL 聚合各 suite success rate 和四项平均值。
```

## 结果

- `PASS`：四个 suite 均完成 500 episodes，四个 evaluator 均 exit code 0。
- checkpoint：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/pi05_libero`
- OpenPI commit：`215abfb217dbac7d5f1273282331b9b1866c0479`
- LIBERO commit：`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- 参数：seed 7，50 trials/task，10 tasks/suite，replan steps 5。
- `libero_spatial`：489/500，97.80%。
- `libero_object`：497/500，99.40%。
- `libero_goal`：487/500，97.40%。
- `libero_10`：465/500，93.00%。
- 四套宏平均 / 2,000 episode 微平均：1938/2000，96.90%。
- 完整性：每套均有 500 个唯一 `(task_id, episode_idx)`，每个 task 恰好 50 episodes。
- H20 隔离：四个 renderer 分别独占 GPU 4、5、6、7；运行期间未发现共享、CUDA/XLA 错误或 kernel panic。
- robosuite 在正常完成后打印 EGL destructor 清理 traceback；发生在 `Total episodes: 500` 之后，不影响退出码或结果。
- DFS 汇总：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/openpi-libero/pi05-libero-official-215abfb-20260916/metrics.md`
- 机器可读明细：同目录 `metrics.json`；逐 episode JSONL、完整日志和视频分别位于 `results/`、`logs/`、`videos/`。
