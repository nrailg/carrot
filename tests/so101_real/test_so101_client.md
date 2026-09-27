# SO101 WebSocket 客户端协议

## 目的与关键断言

通过本地真实 TCP/WebSocket 验证 MessagePack NumPy 往返、有限接收超时和服务端文本异常传播。

- 动作保持 float32，状态和动作逐元素一致。
- 服务端不响应时抛出 TimeoutError；文本异常保留原始错误信息。
- 使用本地测试服务器，不连接远端策略服务或机械臂。

## 运行

只需提供个人 DFS 根目录 `MY_DFS`；公共 runner 从其下的 `work/carrot` 加载源码，
激活 `/opt/venvs/carrot` 并设置源码 `PYTHONPATH`。

```bash
export MY_DFS=/absolute/path/to/personal/dfs
bash "${MY_DFS}/work/carrot/tests/so101_real/test_so101_client.sh"
```

## 结果

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
bash tests/so101_real/test_so101_client.sh
```

证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101-20260927T/test_so101_client.log`。
测试断言尚未执行；本轮未安装缺失依赖或改写共享环境。

本轮镜像、源码和环境记录见 [共同环境](test_so101_runtime.md#本轮共同环境与源码)。


## 2026-09-27：补齐依赖后回归

**PASS**，`3 passed in 0.28s`，runner exit=0。使用同名 runner，MY_DFS 和源码路径同上。
仅离线安装固定 commit 的 openpi-client 0.1.0，未升级其他依赖。
安装来源、镜像、源码版本和同次端到端测试见
[WebSocket 测试记录](../pi_05/test_so101_websocket.md#2026-09-27)。
完整日志：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101-websocket-20260927T101647-2701786/pytest.log`。
