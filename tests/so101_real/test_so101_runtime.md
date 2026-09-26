# SO101 客户端控制与协议测试

## 目的

验证数据集/真机观测、日志/机器人执行、关节与相机映射、示教动作隔离、
episode 边界与 padding、初始姿态检查、驱动限幅、超时/中断停止行为。
本地 WebSocket 测试使用真实 TCP 连接验证 MessagePack NumPy 往返、超时和服务端异常。

## 运行

机侧环境安装 examples/so101_real/README.md 的依赖后，从仓库根目录运行：

```bash
python -m pytest -v tests/so101_real
```

Gemini 使用既定 `/opt/venvs/carrot` 和当前 `MY_DFS`：

```bash
bash tests/so101_real/test_so101_runtime.sh
```

## 结果

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
