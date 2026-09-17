# OpenPI LIBERO 八卡逐卡健康测试记录

## 测试思路

- 使用 PyTorch eager policy 和 `libero_spatial` task 0 的单个 episode，验证每张物理 GPU
  的 LIBERO EGL renderer 路径。
- GPU 0 作为 renderer 时 policy 使用 GPU 1；GPU 1–7 作为 renderer 时 policy 使用 GPU 0。
- 八次测试严格串行，任何时刻只有一个 renderer，policy 与 renderer 永不共卡。
- 每次由安全 runner 独立校验 JSONL、温度和新增 Xid；出现 Xid 时该次返回失败并保持
  dguard 关闭，不并发或自动扩大负载。

## 测试代码

- `examples/libero/run_openpi_libero_eval.sh`：单 task/单 renderer、GPU 隔离、结果完整性和
  Xid 检测。
- OpenPI `examples/libero/main.py`：task 0 选择、LIBERO rollout 与逐 episode JSONL。

## 执行步骤

```bash
# renderer GPU 0
./examples/libero/run_openpi_libero_eval.sh --episodes 1 --server-gpu 1 --render-gpu 0

# renderer GPU 1–7，逐条串行执行
./examples/libero/run_openpi_libero_eval.sh --episodes 1 --server-gpu 0 --render-gpu <1-7>
```

## 结果

- `PASS`：8 张物理 GPU 均分别完成 task 0 的 1/1 episode，八次 runner exit code 均为 0；
  整组耗时约 652 秒。
- renderer GPU 0–7 的最高核心温度依次为 39、37、33、36、37、33、33、36°C；最高显存
  温度依次为 44、42、37、41、40、39、41、42°C，无过热。
- 测试前后内核 Xid 计数均为 0；八个运行目录均未生成 `new-xid.log`。
- GPU 0 作为 renderer 时 policy 使用 GPU 1；其余七次 policy 使用 GPU 0。测试严格串行，
  没有 server/renderer 共卡。
- 测试结束后无 policy/evaluator/runner/telemetry 残留，dguard 已正常恢复。
- 证据目录为
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/openpi-libero/libero_spatial-task0-1x-20260917-12*`，
  对应运行时间从 `120554` 到 `121527`，共 8 个目录。
- runner 后续默认使用持久化 tokenizer
  `${MY_DFS}/hf-hub/big_vision/paligemma_tokenizer.model`，以及从本次已验证临时副本迁移得到的
  `${MY_DFS}/hf-hub/Physical-Intelligence/pi05_libero_pytorch`；不再依赖节点 `/root`。
- `PASS`（持久化路径回归）：迁移后使用 runner 默认路径在 GPU 0/2 再运行 1 个 episode，
  1/1 success、exit code 0、无新增 Xid；server 日志确认从持久化 checkpoint assets 加载
  LIBERO norm stats，且没有触发 tokenizer 下载。
