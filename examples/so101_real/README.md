# SO101：数据集输入调试与真机部署

GPU 上运行 Carrot PI0.5，控制电脑通过 OpenPI WebSocket 协议获取动作。
客户端使用 LeRobot 0.6.1 控制 SO101；默认读取数据集、记录预测，不连接机械臂。

## 1. 环境与策略服务

控制电脑使用独立 Python 3.12 环境，从本仓库根目录运行：

```bash
uv venv --python 3.12 examples/so101_real/.venv
uv --no-config pip install --python examples/so101_real/.venv/bin/python \
  --torch-backend cpu \
  --overrides examples/so101_real/overrides.txt \
  -r examples/so101_real/requirements.txt
source examples/so101_real/.venv/bin/activate
```

OpenPI 客户端的历史依赖声明为 NumPy <2，而 LeRobot 0.6.1 要求 NumPy >=2。
`overrides.txt` 显式统一到 NumPy >=2,<2.3；协议使用 NumPy 数组与 MessagePack，
对应兼容性由 `tests/so101_real/test_so101_client.py` 验证。`--no-config` 避免继承
仓库 GPU 环境的依赖覆盖与索引；机侧选用 CPU PyTorch，无需安装完整 Carrot 或 CUDA。

仓库 `client` extra 与上述机侧 requirements 一致，声明 LeRobot 的 `dataset,feetech` extras，
包含 `pyserial`、`deepdiff` 和 Feetech SDK。root `uv.lock` 仍面向 Linux x86_64/CUDA；
Mac 继续使用上面的独立 CPU 环境安装命令。

GPU 机器沿用已有 `/opt/venvs/carrot` 环境，确认包含 OpenPI 客户端及 websockets。
仓库 `serving` extra 声明服务依赖；root uv 配置也明确覆盖 OpenPI 的旧 NumPy 上限。
已完成环境配置并确认 `MY_DFS` 后：

```bash
source /opt/venvs/carrot/bin/activate
cd "$MY_DFS/work/carrot"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -m carrot.cli.serve_pi05_policy \
  --embodiment so101 \
  --checkpoint "$MY_DFS/experiments/carrot/pi05_sft_so101_orange_cube/checkpoints/step-00005000" \
  --device cuda:0 --num-steps 10 --port 8000
```

使用 checkpoint 自带 tokenizer、统计量和精度配置。现有 step 5000 保存的是 FP32
配置；先在有足够显存的 GPU 上验证加载及推理延迟。服务只允许一个控制客户端串行使用。

### 单腕 wipe：15 FPS、度数模式

`wipe.yaml` 对应 `nrailg/so101_wipe_down_the_cylinder` 与其 step100 checkpoint。
配置使用 `base_camera: null`、`wrist_camera: wrist`、`fps: 15`、`use_degrees: true`。
数据集字段 `observation.images.wrist` 和实时相机名 `wrist` 都映射为
`observation/wrist_image`；不发送缺失的 base 视角，也不复制腕图填充它。

本地自采数据没有 Hub revision 时显式设置 `dataset_revision: null`。
完整数据目录必须包含 `meta/`、`data/`、`videos/`；数据原始 FPS 必须与配置一致。
从仓库根目录先运行无硬件的单帧日志模式：

```bash
python -m examples.so101_real.main \
  --config examples/so101_real/wipe.yaml \
  --server-uri ws://GPU_HOST:8080 \
  --dataset-root /absolute/path/to/so101_wipe_down_the_cylinder \
  --output-dir outputs/so101_wipe_one_frame
```

真机模式需复制该模板，填写 follower 串口、ID、标定目录，以及腕相机的实际编号与朝向。
前五轴使用度数，第六轴夹爪仍为开合百分比；动作绝对限位从驱动现有标定换算，
不得使用旧归一化位置的 `[-100,100]` 作为角度限位。`max_relative_target` 和
`initial_state_tolerance` 与该模式的单位一致：前五轴为度数，夹爪为百分点。
实际关节已越过标定范围时停止下发；标定限位不替代摄像头线缆的活动范围检查。

块内按15Hz执行，块间等待推理；模型仍预测50步、去噪10步，`execute_steps` 控制消费
几行动作。新增配置仍默认 `dataset + log`、单块单步。下方双相机示例使用旧公共数据的
30FPS归一化配置；四种运行模式也适用于wipe，但必须使用wipe模板与对应checkpoint。

## 2. 数据集观测 → 日志

将已下载的 `felixmayor/orange_cube_merged` 完整本地目录复制到控制电脑，包含
`meta/`、`data/` 和 `videos/`。默认 revision 为
`c021b3c22a3de4e70e81010e54fb250a5dde348b`。
客户端禁止 Hub 联网；配置中的 revision 不会自动校验本地文件来自哪个提交，
复制时应保留下载记录。缺少必需文件会失败，不会改用其他数据集。

```bash
python -m examples.so101_real.main \
  --config examples/so101_real/deployment.yaml \
  --server-uri ws://GPU_HOST:8000 \
  --dataset-root /absolute/path/to/orange_cube_merged \
  --output-dir outputs/so101_one_frame
```

默认 episode 0、frame 0、一次正式请求，只记录第一步；另有一次无动作下发的预热。
日志会保存完整 50 步预测。可增加 `--episode`、`--start-frame` 或 `--prompt`；默认任务
文本来自数据集样本。输出目录必须不存在，避免覆盖结果。

完整 episode 调试：复制模板并将 `max_chunks: null`、`execute_steps: 10`，再指定该配置。
每轮读取位置前进实际消费的 N 帧，最后一轮按 episode 剩余帧数截断，遇到边界即结束。
示教动作仅作比较，不会发送给 policy 或作为机器人命令。

## 3. 数据集观测 → SO101 开环动作测试

复制 `deployment.yaml`，填写以下硬件字段：

```yaml
robot_port: /dev/ttyACM0          # 替换为实际串口
robot_id: my_follower           # 必须匹配已有标定文件名
calibration_dir: /absolute/path/to/calibration
use_degrees: false
max_relative_target: 5.0
initial_state_tolerance: 10.0
```

然后显式选择机器人执行端（本例为 SO101），先只发一步：

```bash
python -m examples.so101_real.main \
  --config /absolute/path/to/my_so101.yaml \
  --server-uri ws://GPU_HOST:8000 \
  --dataset-root /absolute/path/to/orange_cube_merged \
  --action-sink robot --execute-steps 1 --max-chunks 1 \
  --output-dir outputs/so101_single_action
```

这条命令会实际控制机械臂。数据集观测模式不连接相机；真实关节读数只用于起始状态核对、
限幅和记录，不混入 policy 的录制观测。预测来自旧图像，因此此模式验证开环动作链路。

公共数据元信息未注明单位和标定。该配置根据数据范围采用 LeRobot 归一化位置假设：
五个关节 [-100,100]，夹爪 [0,100]；下发前核对你的机器人约定。
LeRobot 0.6.1 的默认角度制在这里被显式关闭。

动作目标先按合法位置范围裁剪，再由驱动按当前读数限制每个关节的相对目标变化，
日志记录预测、裁剪目标和 `send_action()` 返回的实际命令。多步或多块执行前，
真实起始状态与录制状态逐维差必须不超过 10；不满足时退出，不自动对齐姿态。
单步验证后，可用 `--execute-steps 10 --max-chunks 3` 做短序列测试。

## 4. 真机观测 → 日志／SO101 闭环

在同一配置中设置非空 `prompt`，并填写真实相机参数，例如：

```yaml
prompt: pick up the orange cube
cameras:
  top:
    index_or_path: 0
    width: 640
    height: 480
    fps: 30
    color_mode: rgb
    rotation: 0
  fpv:
    index_or_path: 1
    width: 640
    height: 480
    fps: 30
    color_mode: rgb
    rotation: 0
```

设备编号、分辨率和旋转填写你的实际采集配置；先检查输出目录中的首帧图像。
本机自采与部署应使用同一相机朝向、标定和归一化位置配置。

```bash
# 真机观测，只记录预测；会打开机器人和相机，但不调用 send_action。
python -m examples.so101_real.main --config /absolute/path/to/my_so101.yaml \
  --observation-source robot --action-sink log --max-chunks 5 \
  --output-dir outputs/so101_live_observations

# 真机观测和真机执行。
python -m examples.so101_real.main --config /absolute/path/to/my_so101.yaml \
  --observation-source robot --action-sink robot --execute-steps 10 --max-chunks 30 \
  --output-dir outputs/so101_closed_loop
```

两条命令使用配置中的 server_uri。首次启动从单步开始；`robot + log` 也会执行 LeRobot
正常的连接与舵机配置过程，不代表硬件完全不受影响。

## 5. 时序、退出与结果

- 模型返回 `[50,6]`；`execute_steps` 控制消费几行，`num_steps` 是服务端去噪次数。
- 每行是绝对目标位置，顺序为 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、
  wrist_roll、gripper，各带 `.pos` 后缀。
- 块内目标频率由 `fps` 配置决定，块间等待推理，实际全程频率会降低；记录实际耗时。
- 连接超时 10 秒、预热 60 秒、后续响应 5 秒，可在配置中调整。
- Ctrl+C、断连、超时或错误时停止新增命令，退出后不自动续跑；断开保留最后位置目标。
  舵机仍可能完成已收到的动作，此行为不等同于硬件急停。
- 每次运行保存 `config.json`、`versions.json`、`events.jsonl`、`summary.json`、
  首帧图像、各块 `chunk_*.npz`；正常结束还生成 `comparison.json` 与 `actions.png`。
- MAE/轨迹仅比较本轮消费的有效帧，不统计 padding；这些指标用于调试，不代表任务成功率。

CPU 测试：`python -m pytest -v tests/so101_real`。
GPU 重载测试及状态见 `tests/pi_05/test_so101_inference_checkpoint.md`。
