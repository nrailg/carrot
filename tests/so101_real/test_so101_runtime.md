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

### 2026-09-30：单腕与度数适配回归

状态：**PASS**。原双相机/30FPS/归一化用例保持回归；动作执行端现在显式接收限位。
17项包含在完整SO101 CPU合跑中：`53 passed in 7.83s`。单腕、15FPS、度数与标定
换算用例、命令、环境及源码版本见 [本轮共同记录](test_so101_profiles.md#2026-09-30)。
未记录本文件独立耗时，未连接真实设备。

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


## 2026-09-30：review清理与模型单位边界回归（准备）

- 范围：显式YAML相机字段、config传递、degree→radian及夹爪百分点→[0,1]，样本/stats同时换算；反向驱动换算、起点容差、导出与握手单位契约。
- 命令：既定Gemini venv中从当前同步源码运行 `python -m pytest -q tests/pi_05/test_so101_sft.py tests/so101_real`。均使用CPU及fake设备，不接触真实串口或相机，不启动训练/模型评估。
- 预期：新单位回归与已有客户端/日志/调度测试通过，旧服务缺单位声明时在硬件连接前拒绝。
- 状态：NOT RUN；源码d16d29a加本轮未提交改动，Docker image tag未记录；实际结果运行后补充。


### 2026-09-30 20:38北京时间：实际结果 PASS

- Gemini task56253f33-0052 exit0：`tests/pi_05/test_so101_sft.py tests/so101_real tests/pi_05/test_pi05_inference.py tests/sft/test_sft_checkpoint.py` 合计92 passed；真实数据同名shell另2 passed，共94项。CPU-only，CUDA_VISIBLE_DEVICES为空；未接触机器人或新模型推理。
- Ruff通过；31个改动源码/配置/shell的Mac与hz1 SHA256一致，shell语法、py_compile和git diff --check通过。实际源码d16d29a加未提交改动，Docker tag未记录。
- 初轮夹爪缩放后固定归一化epsilon造成约1e-5偏差，测试原1e-6零点容差过严；改成有解释的2e-5。随后wrapper子shell因MY_DFS未export失败，已修正执行环境；风格问题已修正。最终测试正常完成，未跳过失败项。
- 完整最终日志及初轮失败日志在 `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101_review_cleanup_20260930/`，含final_tests.log和source_hashes.json。
- 新YAML配置/新单位接口准备完成；未启动新训练或真机任务，未commit。旧服务没有单位metadata，后续联调须按当前代码重启。


## 2026-09-30：显式数据来源与转换后的接口单位

状态：**NOT RUN**；本轮没有可用Gemini会话，未执行pytest或连接硬件。

- 通用 `dataset_repo` 默认为空，缺失时配置校验失败；测试中显式提供来源。
- 新增两种单位模式的握手校验：degree输入转换后应为radians，旧位置模式应为normalized；degree或另一模式的服务声明均被拒绝。
- runner错误消息显示实际服务单位和输入转换后预期单位，缺少单位metadata时单独报错。
- 静态检查PASS：修改文件Python AST语法与100字符行长、通用YAML空repo、runner `bash -n`、`git diff --check`。

复现：`bash tests/so101_real/test_so101_runtime.sh`；硬件组装mock回归入口为 `test_so101_deployment.sh`。
相关三件套：`test_so101_runtime.py`、`test_so101_runtime.sh`、本文件。
源码：`d16d29a7245c56f17d9eec07e42234c11b831200` 加当前未提交改动；Docker image tag及远端依赖commit未记录，本轮未连接远端。


## 2026-09-30：统一degree/夹爪百分点

当前契约取代此前单位方案：所有SO101样本、stats、网络接口、日志和驱动使用degree与[0,100]夹爪。
删除单位换算及额外裁剪，保留模型已有q01/q99 Normalize/Unnormalize。
CPU验证范围：样本/stats原值、训练/推理一致性、导出/加载、握手、mock驱动与报告。
真实数据入口：`bash tests/pi_05/test_so101_dataset.sh`，逐帧核对单腕录制与client数据。
状态：PASS，包含本文件的CPU回归共89项通过，耗时9.88s；Ruff通过。
未操作真机、启动训练或重启policy server。
源码：c4593f2加工作区改动；Docker image tag未记录，Python 3.12.13，LeRobot 0.6.1。

证据：`$MY_DFS/test-runs/so101_degrees_20260930/final_tests.log`、`source_hashes.json`。


## 2026-09-30：删除数据语义字段

范围：删除dataset spec、训练导出、policy与握手、报告中的额外声明和校验。
样本与stats保留源数值，已有Normalize/Unnormalize与标定限幅保留。
预期：无额外metadata的统计文件可以加载、日志可以绘图；握手仍核对动作维度等推理结构。
命令：合跑SO101 SFT、全部so101_real、共享PI05 inference与SFT checkpoint回归，
另执行 `bash tests/pi_05/test_so101_dataset.sh` 验证真实录制数据。
状态：PASS，含本文件的CPU回归共88项通过（10.44s），Ruff通过；未进行GPU推理、训练或真机操作。
源码：bf11592加工作区改动；Docker tag未记录。

环境：Python 3.12.13、LeRobot 0.6.1；证据：
`$MY_DFS/test-runs/so101_source_values_20260930/final_tests.log` 和 `source_hashes.json`。
