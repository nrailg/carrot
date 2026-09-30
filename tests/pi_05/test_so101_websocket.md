# SO101 真实数据与 WebSocket GPU 服务端到端测试

## 目的

不连接真机，以真实 orange_cube_merged 观测通过正式客户端访问独立进程中的
`carrot.cli.serve_pi05_policy`，加载 step-00005000 checkpoint 执行 GPU 推理。
使用 `runner.run()`，覆盖 dataset+log 的预热、两轮请求、动作日志和报告。

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

状态：**PASS**，`1 passed in 72.90s`，runner exit=0。
同次回归 client `3 passed in 0.28s`、deployment `8 passed in 6.72s`，exit 均为 0。

- 完整测试输出：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101-websocket-20260927T101647-2701786/pytest.log`。
- 客户端产物：该目录 `client/` 下的 `summary.json`、`events.jsonl`、两份 NPZ、
  `versions.json`、首帧双相机 PNG、`comparison.json` 和 `actions.png`。
- `summary`：completed=true、reason=max_chunks、chunks=2、steps=4。
  观测帧 0、2，四帧动作全部 executed=false；两次稳态请求约 285.1 / 264.7 ms。
  仅两次采样，不作为延迟基准；报告比较了 4 帧示教动作，不代表任务成功率。
- 服务端通过正式 CLI 启动，命令留在 `server_command.json`；`server.log` 记录 localhost 监听、
  HTTP 健康检查返回 200 及 WebSocket 连接建立。
- 测试后检查服务进程已结束，GPU 显存归零；dguard 暂停 10 分钟并已安排自动恢复。
- 镜像：`wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8`；GPU：H20 0。
- Carrot HEAD：`3a3468e61c2b9ba1b64df3c798ae16c24096c0c8` （包含本次测试三件套）。
  远端无 Git metadata，已比对源码；客户端源码 SHA256 见 `versions.json`。
- Python 3.12.13，pytest 9.1.1，Torch 2.11.0+cu128，LeRobot 0.6.1，NumPy 2.3.1，
  websockets 16.1.1，openpi-client 0.1.0。LeRobot wheel Git commit 未记录。

### 补齐客户端依赖

从已缓存的真实 wheel 安装固定 OpenPI commit
`215abfb217dbac7d5f1273282331b9b1866c0479` 的 openpi-client；wheel 内 13 个 Python 文件
与该 commit 内容逐一匹配。wheel SHA256：
`585cb23bcfa6366a53bc3f01951bf29682be6ff7fe8a3897e099dfa5124c2548`。

```bash
uv pip install --python /opt/venvs/carrot/bin/python --no-deps --no-index \
  /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/so101-test-deps-20260927T101542/wheels/openpi_client-0.1.0-py3-none-any.whl
```

未下载或升级其他依赖；沿用项目对 openpi-client 的 NumPy 2 兼容 override。
`dm-tree` 仍未安装，本次协议/部署路径不导入它；不据此声称整个 openpi-client 包的依赖完整。


## 2026-09-30：服务单位接口更新

旧orange-cube测试服务显式加 `--joint-units normalized`，服务将夹爪统计量转换为[0,1]并声明握手单位，client采用相同转换。该真实GPU重载/网络测试本轮NOT RUN；CPU mock及共享推理回归另见本轮SFT/客户端测试结果。源码d16d29a加未提交改动，Docker tag未记录。


## 2026-09-30：统一degree/夹爪百分点

当前契约取代此前单位方案：所有SO101样本、stats、网络接口、日志和驱动使用degree与[0,100]夹爪。
删除单位换算及额外裁剪，保留模型已有q01/q99 Normalize/Unnormalize。
CPU验证范围：样本/stats原值、训练/推理一致性、导出/加载、握手、mock驱动与报告。
真实数据入口：`bash tests/pi_05/test_so101_dataset.sh`，逐帧核对单腕录制与client数据。
状态：NOT RUN（本轮未执行GPU checkpoint/WebSocket测试）；相关CPU契约回归已通过。
未操作真机、启动训练或重启policy server。
源码：c4593f2加工作区改动；Docker image tag未记录，Python 3.12.13，LeRobot 0.6.1。

证据：`$MY_DFS/test-runs/so101_degrees_20260930/final_tests.log`、`source_hashes.json`。
