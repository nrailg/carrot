# SO101 部署组装与资源清理

## 目的与关键断言

验证 dataset/robot 观测与 log/so101 执行四种组合，在正常完成和推理超时两种情况下的行为。

- 仅机器人观测或执行模式调用硬件工厂；log 执行端不发送动作。
- 正常与异常退出均关闭客户端及已连接的机器人，并保存准确的 summary。
- 串口、模型客户端和数据源使用替身；不加载 checkpoint 或连接真实硬件。

## 运行

只需提供个人 DFS 根目录 `MY_DFS`；公共 runner 从其下的 `work/carrot` 加载源码，
激活 `/opt/venvs/carrot` 并设置源码 `PYTHONPATH`。

```bash
export MY_DFS=/absolute/path/to/personal/dfs
bash "${MY_DFS}/work/carrot/tests/so101_real/test_so101_deployment.sh"
```

## 结果

### 2026-09-30：度数与归一化配置回归

状态：**PASS**。四种模式与正常/超时退出扩展到两种单位配置，使用未连接的真实
Feetech总线计算标定限位，硬件连接和动作下发由替身处理。16项包含在本轮完整
SO101 CPU合跑中：`53 passed in 7.83s`。命令、环境与源码版本见
[本轮共同记录](test_so101_profiles.md#2026-09-30)，未记录本文件独立耗时。

### 2026-09-26：PASS（历史合跑）

本文件包含在当日 30 项通过的合跑中，完整命令、环境版本、Carrot/OpenPI commit
及证据见 [原始合跑记录](test_so101_runtime.md#结果)。未记录本文件的独立耗时。
实际环境为 devcloud CPU 临时 venv；Docker image tag 不适用，LeRobot wheel Git commit 未记录。

### 2026-09-27：补齐独立 runner 与档案

运行状态：**NOT RUN**，本次未重新执行 pytest 或远端测试。
静态检查：同名 runner 的 `bash -n` 及 `git diff --check` 通过。


## 2026-09-27：Gemini 验证

状态：**BLOCKED**；`收集阶段 ModuleNotFoundError: No module named 'openpi_client'`，runner exit=2。

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
cd "${MY_DFS}/work/carrot"
bash tests/so101_real/test_so101_deployment.sh
```

证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101-20260927T/test_so101_deployment.log`。
测试断言尚未执行；本轮未安装缺失依赖或改写共享环境。

本轮镜像、源码和环境记录见 [共同环境](test_so101_runtime.md#本轮共同环境与源码)。


## 2026-09-27：补齐依赖后回归

**PASS**，`8 passed in 6.72s`，runner exit=0。使用同名 runner，MY_DFS 和源码路径同上。
仅离线安装固定 commit 的 openpi-client 0.1.0，未升级其他依赖。
安装来源、镜像、源码版本和同次端到端测试见
[WebSocket 测试记录](../pi_05/test_so101_websocket.md#2026-09-27)。
完整日志：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101-websocket-20260927T101647-2701786/pytest.log`。
