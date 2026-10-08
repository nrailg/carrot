# SO101：数据集输入调试与真机部署

GPU 上运行 Carrot PI0.5，Mac 通过 WebSocket 获取动作，LeRobot 0.6.1 控制 SO101。
默认读取数据集并记录预测，不连接机械臂。

## 环境与策略服务

控制电脑使用独立 Python 3.12 环境，从本仓库根目录运行，并加载当前源码：

```bash
uv venv --python 3.12 examples/so101_real/.venv
uv --no-config pip install --python examples/so101_real/.venv/bin/python \
  --torch-backend cpu \
  --overrides examples/so101_real/overrides.txt \
  -r examples/so101_real/requirements.txt
source examples/so101_real/.venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

OpenPI 客户端的历史依赖声明为 NumPy <2，而 LeRobot 0.6.1 要求 NumPy >=2。
`overrides.txt` 显式统一到 NumPy >=2,<2.3；协议使用 NumPy 数组与 MessagePack，
对应兼容性由 `tests/so101_real/test_so101_client.py` 验证。`--no-config` 避免继承
仓库 GPU 环境的依赖覆盖与索引；机侧选用 CPU PyTorch，无需安装完整 Carrot 或 CUDA。

仓库 `client` extra 与上述机侧 requirements 一致，声明 LeRobot 的 `dataset,feetech` extras，
包含 `pyserial`、`deepdiff` 和 Feetech SDK。root `uv.lock` 仍面向 Linux x86_64/CUDA；
Mac 继续使用上面的独立 CPU 环境安装命令。

GPU 使用已有 `/opt/venvs/carrot`，确认 `MY_DFS` 后运行：

```bash
source /opt/venvs/carrot/bin/activate
cd "$MY_DFS/work/carrot"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
bash recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/serve.sh
```

`serve.sh` 仍加载历史 step100 checkpoint；不会启动训练或控制机械臂。
服务加载 checkpoint 自带 tokenizer 和 `norm_stats.json`。
服务只允许一个控制客户端串行使用。

## 数据与统计量

数据读取、WebSocket请求/响应、示教reference、日志和下发直接使用源数值。
训练配置使用 `norm_stats_source: dataset`，由LeRobot从数据集 `meta/stats.json`
读取 `observation.state` 和 `action` 的统计量；Carrot不重新计算或缩放统计量。
模型已有Normalize使用每轴q01/q99：

```text
z = 2 * (x - q01) / (q99 - q01 + 1e-6) - 1
x = (z + 1) / 2 * (q99 - q01 + 1e-6) + q01
```

训练与推理共用这层Normalize，输出由action stats反归一化还原。
分位数外的数据可以超出 `[-1,1]`，不额外裁剪或按机器人范围归一化。
训练导出实际统计量到 `output_dir/checkpoints/step-XXXXXXXX/norm_stats.json`，
内容仅包含 `state` 和 `action`；服务默认读取此文件，不增加数据语义假设。
服务握手包含embodiment、action_dim、action_horizon、num_steps；客户端核对推理结构。
实际下发的范围仍由机器人现有标定和相对目标限幅决定。

## 数据集观测 → 日志

`knock_down_the_cylinder.yaml` 对应推倒圆柱体的单腕数据：15FPS、
`base_camera: null`、`wrist_camera: wrist`。数据字段 `observation.images.wrist`
和实时相机 `wrist` 映射到 `observation/wrist_image`，不伪造其他视角。
本地目录须包含 `meta/`、`data/`、`videos/`；FPS 与配置一致。

```bash
python -m examples.so101_real.main \
  --config examples/so101_real/knock_down_the_cylinder.yaml \
  --server-uri ws://GPU_HOST:8080 \
  --dataset-root /absolute/path/to/so101_knock_down_the_cylinder \
  --output-dir outputs/so101_knock_down_the_cylinder_one_frame
```

默认 episode0/frame0，只记录第一步，另有一次无下发的预热；日志保存完整动作窗口。
可增加 `--episode`、`--start-frame`、`--prompt`；默认 prompt 来自录制样本。
输出目录须不存在。通用 `deployment.yaml` 的 `dataset_repo` 留空，需显式填写。
完整episode调试可将 `max_chunks: null`、`execute_steps: 10`；末块按剩余帧数截断。
示教动作仅供比较，不发送给模型或机器人。

## 数据集观测 → 机器人开环

复制单腕模板，填写串口、机器人ID、标定目录。数据集观测模式不连接相机：

```bash
python -m examples.so101_real.main --config /absolute/path/to/my_so101.yaml \
  --action-sink robot --execute-steps 1 --max-chunks 1 \
  --output-dir outputs/so101_single_action
```

该命令会控制机械臂。动作先按标定绝对范围裁剪，再由驱动限制相对当前反馈的目标变化。
`max_relative_target`、`initial_state_tolerance` 直接与反馈值比较。
多步执行前核对实机与录制起点，不自动对齐姿态。日志中的 `sent` 是驱动返回的发送目标，
不代表运动后的反馈位置。

## 真机观测 → 日志或闭环

模板中填写非空 prompt、腕相机实际编号、朝向和15FPS采集参数，再运行：

```bash
# 打开机器人与相机，记录预测，不调用send_action。
python -m examples.so101_real.main --config /absolute/path/to/my_so101.yaml \
  --observation-source robot --action-sink log --max-chunks 5 \
  --output-dir outputs/so101_live_observations

# 真机观测与执行。
python -m examples.so101_real.main --config /absolute/path/to/my_so101.yaml \
  --observation-source robot --action-sink robot --execute-steps 10 --max-chunks 30 \
  --output-dir outputs/so101_closed_loop
```

这两条命令使用配置中的 server_uri。采集视角、朝向和标定需与录制一致。
`robot + log` 仍会执行 LeRobot 的正常连接和舵机配置过程。

## 时序与结果

- 模型返回 `[50,6]` 的绝对位置目标，顺序为 shoulder_pan、shoulder_lift、elbow_flex、
  wrist_flex、wrist_roll、gripper；`execute_steps` 决定消费几行。
- `num_steps` 是服务端去噪次数；块内按配置FPS执行，块间等待推理。
- Ctrl+C、断连、超时或错误时停止新增命令，断开后保留最后位置目标。
- 输出包括配置、版本、事件、首帧图像、`chunk_*.npz`；正常结束生成比较指标与轨迹图。
  MAE只比较消费的有效帧，不含padding，也不代表任务成功率。

CPU测试及档案位于 `tests/so101_real/`；模型重载测试见
`tests/pi_05/test_so101_inference_checkpoint.md`。

## 执行端动作兜底

限幅仅在 `SO101Sink.send()` 下发机器人前执行；policy 返回原始反归一化预测，无需额外
bounds 配置或服务参数。连接时先核对 LeRobot 文件与设备标定一致，再读取
`robot.bus.calibration` 的 `range_min/range_max`，经 SDK `_normalize()` 换算为各轴动作单位。
五个角度轴遵循 `use_degrees`，夹爪为 [0,100]；float32 命令边界向区间内取整。
LeRobot 的 DEGREES 写入分支本身没有绝对限幅，故执行端先 clamp，再交给驱动处理
`max_relative_target`。

`action_limits` 事件记录本次实际采用的范围；NPZ `predicted` 和 `command.target` 保留原始
模型输出。`action` 事件中的 `bounded_target` 是绝对限幅结果，`absolute_clipped` 为逐轴
标记，`clip_delta = bounded_target - 原始目标`；`sent` 是驱动相对限幅后的发送目标，
不是机器人运动后的实测反馈。原有 `clipped` 表示最终发送目标是否改变。
日志模式保持原始预测，报告 MAE 仍衡量原始模型拟合。此轮仅验证 fake robot/离线 SDK，
尚未实际驱动机器人。

训练 jitter recipe 也复用 `actions.py` 的限位换算，通过 `state_jitter_calibration_path`
读取同步到 GPU 的同一份 LeRobot 标定 JSON，无需在 YAML 重复五轴范围。
远端保留相同目录布局：`$MY_DFS/.cache/huggingface/lerobot/calibration/`。
Mac 标定文件变更后需更新 GPU 的共享副本；离线读取不连接硬件。

## 单步到位后再推理

同步单步真机执行可在YAML显式开启：

```yaml
execute_steps: 1
wait_for_target: true
target_tolerance: 1.0
target_timeout_s: 3.0
```

每次只发送一个目标，等待实际反馈连续3次落入容差后再采集下一观测、请求模型。
等待期间只读取反馈，不重发或修正目标。
容差作用于五轴degrees和夹爪原始单位；等待对象是绝对/相对限幅后实际发送的`sent`。
默认关闭，保持录制动作回放的原时序；只允许`action_sink: robot`、`use_degrees: true`。
`action`事件记录`target_reached`、`target_error`、`target_wait_s`和全部`target_samples`。
若超过时限仍未到位，记录`target_reached: false`和实际残差，然后读取最新实测状态继续推理。
等待本身不修改PID、不补偿Goal，也不能消除静态追踪偏差；超时继续不代表已经到位。

真机执行连接完成后，客户端将`elbow_flex`和`wrist_flex`的`P_Coefficient`设为32，
并读回校验；写入失败或读回不符即停止初始化。覆盖发生在LeRobot默认配置之后，
避免每次重连被重设为16。I/D沿用SDK默认的0/32，其余关节P保持默认16；
日志模式不额外写入P。此设置来自固定目标对照，完整轨迹的改善程度尚待验证。
