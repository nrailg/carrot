# SO101 客户端控制与协议测试

## 目的

验证数据集/真机观测、日志/机器人执行、关节与相机映射、示教动作隔离、
episode 边界与 padding、初始姿态检查、驱动限幅、超时/中断停止行为。
协议、部署组装和报告测试分别见同目录 `test_so101_client`、
`test_so101_deployment`、`test_so101_report` 三件套。

## 运行

机侧环境安装 examples/so101_real/README.md 的依赖后，从仓库根目录运行：

```bash
python -m pytest -v tests/so101_real/test_so101_runtime.py
```

Gemini 使用既定 `/opt/venvs/carrot` 和当前 `MY_DFS`：

```bash
export MY_DFS=/absolute/path/to/personal/dfs
bash "${MY_DFS}/work/carrot/tests/so101_real/test_so101_runtime.sh"
```

## 结果

### 2026-09-27：runner 按同名测试拆分

runtime runner 现在仅运行 `test_so101_runtime.py`；其他三个测试各有独立 runner 和档案。
本次 pytest **NOT RUN**；四个 runner 的 `bash -n` 和 `git diff --check` 通过。
以下保留原始合跑证据。

### 2026-09-26：PASS

最终命令（从 Carrot 仓库根目录运行）：

```bash
CARROT_SO101_DATASET=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/felixmayor/orange_cube_merged \
PYTHONPATH="$PWD:$PWD/src" \
  /tmp/carrot-so101-integration-venv/bin/python -m pytest -q \
  tests/so101_real tests/pi_05/test_so101_dataset.py
```

结果：**30 passed in 6.42s**，其中客户端/协议/报告/组装测试 29 项，真实数据读取 1 项。
四种运行模式的工厂选择与正常/超时退出均已覆盖；默认 dataset+log 未连接任何硬件，
log 执行端零 send_action。网络测试使用本地真实 WebSocket，硬件调用由 fake driver 替代。
报告测试验证未完成的动作不计入 MAE；真实数据测试验证视频解码与末尾 padding。

- 环境：devcloud，本地独立临时 venv，Python 3.12.13、pytest 9.0.3、NumPy 2.2.6、
  PyTorch 2.11.0+cpu、LeRobot 0.6.1 wheel、websockets 16.1.1、matplotlib 3.11.2。
- 源码：Carrot HEAD d8348c8da9048c1382a312f28cbd5ab84fcff998 加本次未提交改动。
  OpenPI client 固定 215abfb217dbac7d5f1273282331b9b1866c0479；LeRobot wheel Git commit 未记录。
- Docker image tag：不适用，本次在 devcloud 运行，没有使用 GPU 容器。
- 最初精简依赖环境为 NumPy 2.4.4，先通过 21 个测试；上述 30 项最终结果使用 README
  安装方式配置的完整 CPU 客户端依赖，以 NumPy 2.2.6 为准。
- `python -m examples.so101_real.main --help` 成功；ruff、bash -n、git diff --check、
  `uv lock --check --offline` 全通过。独立机侧依赖解析和安装均成功。
- 无真实机器人动作，未运行完整 GPU checkpoint 推理；相应验收单独记录。


## 2026-09-27：Gemini 验证

状态：**PASS**；`17 passed in 0.33s`，runner exit=0。

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
cd "${MY_DFS}/work/carrot"
bash tests/so101_real/test_so101_runtime.sh
```

证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101-20260927T/test_so101_runtime.log`。

### 本轮共同环境与源码

- 镜像：`wepsdl/carrot:v1.10-driver-575.57.08-cuda-12.8`。
- Python 3.12.13，pytest 9.1.1；解释器 `/opt/venvs/carrot/bin/python`。
- PyTorch 2.11.0+cu128，LeRobot 0.6.1，NumPy 2.3.1，websockets 16.1.1，
  matplotlib 3.10.8，transformers 5.5.4，safetensors 0.8.0；openpi-client 未安装。
- LeRobot wheel Git commit 未记录；本轮未使用 OpenPI client 源码替代缺失安装。
- Carrot 本地 HEAD：`3c7238c7ec8cbf2c0de9b864f6c0a493ccbd5115` 加新增 SFT runner/档案。
- 远端源码无 `.git`；同步与依赖版本证据见同一 run 目录的 `verification.log`。
- 主代理独立对比 82 个实现、测试和 runner 文件的 SHA256，均与远端一致。
- 本轮五个文件共 24 项通过；client/deployment 因缺 `openpi_client` 收集失败。
- 未连接机器人，未安装依赖。历史 devcloud PASS 不替代本轮结果。


## 2026-09-27：合并 runner 模块回归

`deployment.py` 和 `runtime.py` 合并为 `examples/so101_real/runner.py`；
现有循环与组装测试保留各自文件名，导入及 monkeypatch 改为 runner。

在 devcloud 已有 `/tmp/carrot-so101-integration-venv/bin/python` 下执行：

```bash
PYTHONPATH="$PWD:$PWD/src" /tmp/carrot-so101-integration-venv/bin/python -m pytest -q tests/so101_real
PYTHONPATH="$PWD:$PWD/src" /tmp/carrot-so101-integration-venv/bin/python -m pytest -q tests/so101_real/test_so101_runtime.py::test_default_profile_needs_no_robot_configuration
```

首轮 28 passed、1 failed（配置文件名被误改）；修复该路径后单项重跑 1 passed。
29 个用例均已有通过结果，未声称一次完整重跑。CLI `--help` 和 WebSocket 测试
`--collect-only` 均成功；本轮未重新运行 GPU 端到端。
日志：`/tmp/carrot-so101-runner-regression.D1uHnS/`（本地临时目录）。
环境：Python 3.12.13、pytest 9.0.3、LeRobot 0.6.1、websockets 16.1.1；
镜像不适用。runner.py SHA256 为
`b68c387bb67758e1ee69313b505f41ad4260fc6d88d254481e90599f69b00328`。
源码为当前工作区，Ruff、diff 检查通过，迁移函数 AST 逐项对比一致
（仅 RemotePolicy 类型标注替换为 PolicyClient）。
