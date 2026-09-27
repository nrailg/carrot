# SO101 真实数据与 WebSocket GPU 服务端到端测试

## 目的

不连接真机，以真实 orange_cube_merged 观测通过正式客户端访问独立进程中的
`carrot.cli.serve_pi05_policy`，加载 step-00005000 checkpoint 执行 GPU 推理。
使用 `deployment.run()`，覆盖 dataset+log 的预热、两轮请求、动作日志和报告。

- 服务绑定 localhost，测试动态选择端口，启动失败或超过 180 秒即失败。
- 每轮消费 2 帧，共 2 个 chunk；确认请求帧为 0、2，总计记录 4 帧动作。
- 返回并保存 float32[50,6] 有限动作，记录的目标与预测前两帧一致。
- 禁止调用机器人连接入口；所有动作均标记 executed=false。
- 图像、summary、comparison、NPZ、events 和 server 日志保存到 DFS。
- 成功或异常退出均结束本测试创建的 server 进程。
- 请求超时设为 120 秒以验证功能；不代表满足默认 5 秒超时或实时控制要求。
- 不比较随机采样的两轮输出是否相同；MAE 仅供调试，不代表任务成功率。

## 运行

需要既定 `/opt/venvs/carrot` 环境中的 serving/client 依赖及空闲 GPU。
确认 dguard 后，仅需提供个人 DFS 根目录，runner 推导数据与 checkpoint 路径。

```bash
export MY_DFS=/absolute/path/to/personal/dfs
CUDA_VISIBLE_DEVICES=0 bash "${MY_DFS}/work/carrot/tests/pi_05/test_so101_websocket.sh"
```

产物目录由 runner 输出，为 `${MY_DFS}/test-runs/so101-websocket-<UTC时间>-<PID>`。

## 2026-09-27

状态：NOT RUN，等待 Gemini 执行。镜像、依赖版本、源码和结果将在执行后回填。
