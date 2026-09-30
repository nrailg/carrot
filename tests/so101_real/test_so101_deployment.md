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


## 2026-09-30：wipe step100 真机单步

单步 **PASS**；短块 **BLOCKED**（起始姿态超过10°阈值），默认启动路径仍需旧目标防护。
本次结果不代表连续闭环、任务效果或默认启动行为已经验收。

- 源码 commit `9e2369b`；Mac独立client Python3.12.14、LeRobot0.6.1、
  Torch2.11.0、NumPy2.2.6、websockets16.1.1、固定commit OpenPI client0.1.0。
  Mac无Docker镜像；服务端本轮Docker tag未核实，LeRobot wheel commit未记录。
- 本地数据 `~/.cache/huggingface/lerobot/local/so101_32_episodes_20260929_163347`，
  metadata与动作parquet SHA256均与GPU训练副本一致；2 episodes/354帧、单wrist、15FPS、度数。
- follower端口 `/dev/cu.usbmodem5C821078421`、ID `my_awesome_follower_arm`；
  现有标定与寄存器匹配。1–6号扭矩初始为0、Status为0、当前位置在标定范围内。
- 发现初始Goal_Position为零刻度，和部分当前位置相差约180°；LeRobot的configure会启用扭矩。
  本次先只读确认Operating_Mode=0、Phase多圈位已清除，再在扭矩关闭时将Goal_Position
  原始值写为当前Present_Position并读回校验；然后关闭串口、调用正式客户端入口。
  **该准备是本次外部操作，未自动包含在commit的connect_robot中。**
- 正式入口：`run(load_config(Path("/Users/wujunyu/.cache/carrot/so101_step4_20260930/deployment.yaml"), {}))`。
  dataset + robot、episode0/frame0、execute_steps=1、max_chunks=1、相对限幅5。
  先预热后正式请求；summary completed=true、chunks=1、steps=1；动作日志仅一次executed=true。
- 下发相对量（1–5号度数，6号百分点）：`[+2.425,-0.791,+1.100,-5.000,-2.880,-0.758]`。
  2号目标经绝对下界限幅，4号经相对5°限幅。原始目标均在标定范围内。
- 约0.3秒后读回：1/3/4/5号编码器变化 `+18/+17/-40/-29`；2/6号未观察到位置变化，
  因此不宣称所有关节都到达目标。1–6号Status均0，标定JSON未变；退出后扭矩保留为1。
- 单帧日志预览耗时约306ms；正式推理使用新的随机噪声，不能把预览目标等同于本次下发值。
- 2号肩关节起始位置与录制首帧差12.48°，超过短块阈值；未重复下发或自动对齐。
- 证据：`/Users/wujunyu/.cache/carrot/so101_step4_20260930/`，包含preview、deployment.yaml、
  startup_before.json、single_step/events.jsonl及summary.json、after_single_step.json、verification.json。


## 2026-09-30：用户现场观察后的3帧与约3秒测试

3帧与45帧动作块执行 **PASS**；跟踪精度/任务效果未验收。

- 用户明确环境安全并在现场观察，先要求3帧，再要求3秒；起始容差临时设为15°，
  容纳2号肩与录制首帧12.48°偏差。相对目标限幅仍5°/5百分点；每次仅1个response，
  分别消费前3行、前45行，15FPS、max_chunks=1。源码未改变。
- 运行前只读复核标定、Status、原始当前位置/目标范围及旧目标差；旧目标已在当前姿态附近。
  不再复用最初零刻度旧目标，也没有扩大绝对限位。
- 录制episode0前三行动作相同：`[2.021978,-105.142860,96.131866,54.285713,-4.527472,0.407166]`；
  单位为前五轴度数、夹爪百分比。用户对角度的疑问已用原始parquet核对。
- 3帧：summary completed=true、steps=3；命令区间0.212秒，4号实际下发64.099/63.571/61.462°，
  退出读回所有Status为0。没有继续消费该response后续行。
- 45帧：summary completed=true、steps=45；首条command至complete为3.191秒，
  包含日志与节拍开销；不能表述成精确3.000秒。动作块结束后额外写当前位置保持目标，
  读回确认Goal一致，扭矩保留为1，Status1–6均0。
- 45帧前后反馈：2号肩 -104.044→-69.582°，4号腕俯仰63.121→23.912°，
  3号肘95.604→95.429°。肘最后模型目标52.023°、限幅下发89.989°，实际未跟上；
  不能把状态码0或执行完成等同于轨迹/任务通过。本轮未放宽相对限幅。
- 日志present来自sink读数，驱动限幅使用其后更新的读数：最大sent与前一读数差5.088°。
  一次机械验证误以该旧读数检验严格5°阈值而失败；复核调用顺序及驱动日志后明确其并非
  驱动限幅所用位置。不声称日志旧读数能独立证明每次驱动限幅，需进一步记录同步位置验证。
- 运行入口仍为`run(load_config(Path(...), {}))`；配置与证据在
  `/Users/wujunyu/.cache/carrot/so101_step4_20260930/` 下deployment_three_step.yaml、
  deployment_three_seconds.yaml、three_step/、three_seconds/、before/after_three_seconds.json、
  verification_three_step.json、verification_three_seconds.json。
- 用户观察3帧时“动了一点点”；45帧效果尚未由用户描述。未切换实时相机/关节观测闭环。


## 2026-09-30：reset后30秒实时观测试跑与3号响应排查

实时链路与定时停止 **PASS**；动作跟踪/任务效果未验收，3号肘响应不足。

- 用户确认先前运动正常、自行reset，并授权30秒运行且现场观察；本次切换robot+robot，
  每块执行5步（15FPS）后读取真实关节与wrist图，不循环录制episode。相对限幅保持5°/5百分点。
- 标定匹配、Status初始全0；reset后扭矩部分0/部分1，旧目标尚在附近。
  启动前外部将raw Goal_Position设为当前raw Present_Position并读回，默认入口仍未自动加入该防护。
  OpenCV index0外接相机640×480@15实测出图；实景为桌面，未断言任务物体配置正确。
- 临时launch脚本`~/.cache/carrot/so101_step4_20260930/live_thirty_seconds/run_live.py`
  在首次command日志后以SIGALRM计时30秒，超时抛DurationLimit，经现有runner异常清理停止。
  未改产品代码。正式命令使用Mac独立venv Python、源码PYTHONPATH、HF离线环境运行该脚本。
- 完成45块/225条executed=true动作；46次正式请求（不含预热），最后一次在推理等待中到时中断，
  未产生该轮prediction/command。45份NPZ均为有限float32[50,6]，无示教reference。
- 首条command到stopped为30.004秒，网络/相机清理后30.270秒；随后外部写当前位置保持目标，
  读回Goal一致。终止时Status1–6全0、Torque_Enable全1。原summary completed=false、
  reason=DurationLimit，属于计划停止，不能冒充runner自然完成；trial_summary单独记录结果。
- 3号运动期反馈95.165..96.220°，模型目标87.399..98.752°，下发目标90.165..97.275°；
  15条指令相对sink此前读数超过3°。此前3秒测试也仅95.604→95.429°。
- 停止后只读3号Torque_Enable=1、Status=0、P/I/D=16/0/32、CW/CCW Dead Zone=1/1、
  Max_Torque_Limit=1000、温度读数59；保持目标等于反馈，电流raw1/负载raw0。
  这些是停后读数，不代表运动时电流/负载，更不能据此排除机械卡滞/舵机驱动问题。
- 实际算法把目标误差截在当前位置±5°，当前位置不动时目标不会按上一指令累加；
  “误差限幅+带载控制响应不足”为候选解释，未定根因、未改PID、未追加点动。
  上游旧版本也报告小max_relative_target可能不动（issue1527），只能作线索，不证明本机原因。
- 证据目录`~/.cache/carrot/so101_step4_20260930/live_thirty_seconds/`：
  before/after.json、trial_summary.json、verification.json、camera_preflight.jpg、run/events.jsonl、
  run/chunk_*.npz、servo_readonly_after.json；源码沿用此前提交，Mac环境版本同前。


## 2026-09-30：3号肘单关节直接目标诊断

直接命令与反馈采集 **PASS**；减小角度方向跟踪不足，根因尚未完全分离。

- 用户明确授权通过代码直接下发角度测试。仅写3号elbow_flex的Goal_Position，
  使用带应答write并读回原始目标；不调用configure、不改PID/标定、不写其他关节目标。
  扭矩原已开启，初始标定匹配、Status全0。
- 初始反馈96.220°/raw3002；依次请求91.220°（raw2945）、86.220°（raw2888）、
  返回96.220°（raw3002），每段保持2秒、采35组位置/目标/扭矩/状态/电流/负载/温度/Moving。
  此诊断绕过SOFollower的5°目标差限幅，目标仍在已确认标定范围内。
- -5°：96.220→94.813°，仅移动-1.407°；电流最大raw1、负载绝对值最大raw218。
- -10°：94.813→90.154°，该段移动-4.659°（相对最初-6.066°）；电流最大raw108、
  负载绝对值最大raw283。返回：90.154→95.956°，电流最大raw1、负载绝对值最大raw174。
  电流/负载未换算物理单位，不把负载PWM估计值当成测得扭矩。
- 105样本原始Goal均读回匹配、扭矩1/Status0；所有关节退出Status0。
  结束把3号目标设为实测raw2999保持；未继续试更大角度或更改控制参数。
- 最后0.5秒反馈基本停住、减小方向仍有约3.6–3.9°误差，反向回起点误差约0.26°。
  证据支持5°误差限幅与带载/摩擦/PID响应组合导致欠跟踪，而非完全不响应直接命令；
  单次对照仍不能独立确定PID或机械原因，也未测试更长保持时间。
- 证据 `~/.cache/carrot/so101_step4_20260930/elbow_direct_probe/`：before.json、
  samples.jsonl、results.json、after.json、verification.json；数据采集命令在工具记录中。
