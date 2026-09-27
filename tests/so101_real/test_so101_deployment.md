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

### 2026-09-26：PASS（历史合跑）

本文件包含在当日 30 项通过的合跑中，完整命令、环境版本、Carrot/OpenPI commit
及证据见 [原始合跑记录](test_so101_runtime.md#结果)。未记录本文件的独立耗时。
实际环境为 devcloud CPU 临时 venv；Docker image tag 不适用，LeRobot wheel Git commit 未记录。

### 2026-09-27：补齐独立 runner 与档案

运行状态：**NOT RUN**，本次未重新执行 pytest 或远端测试。
静态检查：同名 runner 的 `bash -n` 及 `git diff --check` 通过。
