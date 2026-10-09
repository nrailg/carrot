# Knock down the cylinder：frame-index wrist image + clamped state jitter

状态：**NOT RUN / PENDING**。本轮仅编写代码与静态检查；未启动训练、推理服务或机器人。
比较对象为旧cylinder no-vision + ±3° state jitter；不修改其配置、源码或历史记录。

目的：用确定的 episode 内 zero-based frame index 图片提供外部进度条件。
这是带外部进度的轨迹重放，不能据此声称策略已自主识别阶段。
共用 `examples.so101_real.frame_index.render_frame_index` 在黑底224×224 RGB uint8图上画固定四位
十进制七段数字（0000..9999），无随机性、字体、文件I/O或真实相机。
图中仅含当前index，不含state/action/未来帧或任务答案。

训练数据：`nrailg/knock_down_the_cylinder_1_20260930_222251`，264帧、15 FPS、单episode；
prompt由数据读取 `Knock down the cylinder`。factory验证metadata total_episodes==1，且
metadata total_frames等于实际长度；其他单episode数据也可复用同一factory。
旧loader仍会读取视频再丢弃，这轮不优化loader。先复用现有state_jitter wrapper，五轴
U(-5°,5°)按共享标定clamp；夹爪、action、padding和norm_stats保持原契约，随后只替换腕图。
仅wrist mask=true，base/right=false；fit.vision=true，不使用DropVision。

相对旧cylinder no-vision + ±3° jitter baseline，本recipe有两项实验条件变化：引入frame-index腕图条件，以及
将五轴state jitter从±3°提高到±5°；共享SDK标定clamp仍保留。因此不能将效果差异仅归因于vision。

其余对照设置：官方 `pi05_base_pytorch_h10` step0，h10，2000steps，warmup100后constant1e-6；
8 GPU BF16 FSDP、micro4/global64/GAS2、seed1000，每500step保存。noise/t保持随机。
独立输出：`$MY_DFS/experiments/carrot/pi05_so101_knock_down_the_cylinder_frame_index/knock_down_the_cylinder_1_20260930_222251/training`。
运行前按carrot-recipe核对空闲GPU、当前MY_DFS、已有Ray、资源、配置固定路径与源码版本。
未来获准运行训练时：

```bash
export SOURCE_COMMIT=<verified-source-commit>
bash recipes/pi05_sft_so101_knock_down_the_cylinder_frame_index/run.sh
```

runner复用现有fit_validation训练入口，不复制trainer，不修改模型结构。预期step500/1000/
1500/2000各有checkpoint、tokenizer、norm_stats和optimizer shards。验收需分别证明训练退出、
checkpoint完整/可加载与离线效果；当前均PENDING；旧no-vision评估需适配相同帧号图片条件，不能直接沿用。

## 离线 dataset -> log（未来执行）

服务必须使用正常vision路径 `carrot.cli.serve_pi05_policy`，例如在未来空闲GPU运行：

```bash
python -m carrot.cli.serve_pi05_policy --embodiment so101 \
  --checkpoint "$MY_DFS/experiments/carrot/pi05_so101_knock_down_the_cylinder_frame_index/knock_down_the_cylinder_1_20260930_222251/training/checkpoints/step-00002000" \
  --device cuda:0 --host 0.0.0.0 --port 8080
```

不要使用no-vision服务或DropVision包装。checkpoint应自带tokenizer与norm_stats。
将下面配置另存为本地绝对路径的 `frame_index_offline.yaml`，替换数据、服务和新输出路径：

```yaml
observation_source: dataset
action_sink: log
server_uri: ws://GPU_HOST:8080
dataset_root: /absolute/local/knock_down_the_cylinder_1_20260930_222251
dataset_repo: nrailg/knock_down_the_cylinder_1_20260930_222251
output_dir: /absolute/new/frame_index_offline
base_camera: null
cameras: {}
frame_index_image_frames: 264
episode: 0
start_frame: 0
fps: 15
execute_steps: 1
max_chunks: null
```

```bash
PYTHONPATH="$PWD/src:$PWD" python -m examples.so101_real.main \
  --config /absolute/local/frame_index_offline.yaml
```

指定总帧数必须等于所选episode长度；请求使用其局部frame_index，不用跨episode全局索引。
未指定prompt时读数据原task。检查config.json中的frame_index_image_frames、events.jsonl的
image_mode/总帧数/request.frame、NPZ.frame与planned_steps；每次请求只含实测或录制state、
当前index合成腕图和prompt。默认旧配置frame_index_image_frames=null，普通图像模式不变。

## 未来真机 opt-in 示例（默认不得执行硬件）

仅在另行授权真机执行且标定、起点与服务已核对后，将离线配置改为：

```yaml
observation_source: robot
action_sink: robot
base_camera: null
cameras: {}
frame_index_image_frames: 264
start_frame: 0
prompt: Knock down the cylinder
robot_port: /dev/REPLACE_WITH_VERIFIED_PORT
robot_id: REPLACE_WITH_VERIFIED_ID
calibration_dir: /absolute/existing/calibration
use_degrees: true
wait_for_target: true
execute_steps: 1
max_chunks: null
```

该模式保留实测state与现有prompt，SDK收到空camera配置，不打开相机。预热、重复read、
到位等待与反馈采集均不推进index；只有实际采用动作后的source.advance(count)推进。
h10/K1每发送1个推进1，K5推进5；最后不足K只消费剩余帧，到N停止，不循环或夹在末帧。
index表示已发送动作进度，**不是已到位进度**。现有等待超时记录残差后继续，因此pose/index
可能不同步；限位、PID和网络行为沿用现有客户端。

## 2026-10-09 准备记录

正式CPU契约pytest：NOT RUN/PENDING，见 `tests/so101_real/test_so101_frame_index.md`。
训练/服务/模型推理/真机：NOT RUN。实际Docker image tag、Carrot commit、上游commit、
运行任务ID、训练日志与checkpoint：未记录（尚未运行）。无实验效果结论。
