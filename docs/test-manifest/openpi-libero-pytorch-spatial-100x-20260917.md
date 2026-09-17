# OpenPI PyTorch LIBERO Spatial 100x 测试记录

## 测试思路

- 使用由官方 `pi05_libero` JAX checkpoint 转换得到的 PyTorch checkpoint 和原始 normalization stats。
- `libero_spatial` 的 task 0–7 各运行 100 episodes，共 800 episodes；task 8–9 暂不运行。
- GPU 0–7 各固定运行一个对应 task。
- 每张 H20 同时最多一个 LIBERO EGL renderer；policy 使用 eager 路径，避开已确认的首次
  `torch.compile(mode="max-autotune")` websocket timeout。
- 每个 task 使用独立 server 端口、JSONL、日志和视频目录，避免并发写冲突。

## 测试代码

- `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/openpi/examples/libero/main.py`：增加可选
  `task_id` 参数，用于让每个 evaluator 只运行一个 task；success contract 不变。

## 执行步骤

```bash
# 每个 task：独立 GPU、独立 policy server/client、seed 7、replan 5、100 episodes。
# task 0–7 并发；task 8–9 不启动。
```

## 结果

- `FAIL`：8 路同卡 policy server + EGL renderer 在 231 秒后中止。task 0–6 各完成
  2 条、task 7 完成 1 条，共 15/800，15 条均 success；所有 client exit code 134。
- 内核在多张 H20 上同步记录 NVIDIA `Xid 31`：`ENGINE GRAPHICS` 的 MMU fault。
  说明每张 GPU 同时承载 PyTorch policy 和 EGL renderer 的拓扑不安全；未自动重试。
- server/client 均已由调度脚本清理，失败证据保存在：
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/openpi-libero/pi05-libero-pytorch-spatial-task0-7-100x-20260917`。
- 建议改为已验证的 4 对隔离拓扑：GPU 0–3 分别运行 policy server，GPU 4–7
  分别运行唯一 renderer；task 0–3 与 task 4–7 分两波完成。
