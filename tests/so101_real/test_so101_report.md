# SO101 动作报告

## 目的与关键断言

验证报告只比较成功完成的有效动作，排除已保存但未执行的预测。

- 两步预测仅记录一步成功事件时，compared_frames 必须为 1。
- 六个关节 MAE 必须均为 1，不计入未执行帧的巨大误差。
- 从持久化日志生成非空 actions.png；不需要模型、数据集或机器人。

## 运行

只需提供个人 DFS 根目录 `MY_DFS`；公共 runner 从其下的 `work/carrot` 加载源码，
激活 `/opt/venvs/carrot` 并设置源码 `PYTHONPATH`。

```bash
export MY_DFS=/absolute/path/to/personal/dfs
bash "${MY_DFS}/work/carrot/tests/so101_real/test_so101_report.sh"
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

状态：**PASS**；`1 passed in 6.69s`，runner exit=0。

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
cd "${MY_DFS}/work/carrot"
bash tests/so101_real/test_so101_report.sh
```

证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101-20260927T/test_so101_report.log`。

本轮镜像、源码和环境记录见 [共同环境](test_so101_runtime.md#本轮共同环境与源码)。
