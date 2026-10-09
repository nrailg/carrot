# Knock down the cylinder：frame prompt + no vision + clamped state jitter

状态：**NOT RUN / PENDING**。本轮仅编写代码与本地静态检查；未启动训练、推理服务或机器人。
比较对象为旧cylinder no-vision + ±3° state jitter；不修改其配置、源码或历史记录。

目的：将episode内zero-based frame index拼入原始任务prompt，提供外部进度条件。
这是frame-conditioned trajectory replay，不能据此声称模型已自主识别phase。
唯一轻量formatter为 `examples.so101_real.frame_prompt.format_frame_prompt`，训练与推理共用：

```python
f"{prompt.rstrip()} Frame: {index:04d}."
```

例如 `Knock down the cylinder Frame: 0263.`。帧号接受非负整数，拒绝bool、负数与非整数；
`:04d`仅指定最小宽度，10000等更大索引也合法。每次从原始任务文本拼接，不在上一请求prompt上重复追加。
帧号条件不编码state/action/未来帧或任务答案，无字体或相机依赖。

训练数据：`nrailg/knock_down_the_cylinder_1_20260930_222251`，264帧、15FPS、单episode；
原始prompt由数据读取 `Knock down the cylinder`。factory验证metadata total_episodes==1，
且total_frames等于实际长度。其他单episode数据也可复用此factory。
先复用现有state_jitter wrapper，五轴U(-5°,5°)按共享SDK标定clamp；随后只修改prompt。
夹爪、action、padding、原图和norm_stats保持原契约。旧loader仍读取视频，本轮不优化loader。
`fit.vision=false`复用现有DropVision，训练最终所有image pixels=0、image_mask=false。

相对旧cylinder no-vision + ±3° jitter baseline，有两项实验条件变化：增加prompt帧号条件，
以及五轴state jitter从±3°提高到±5°；SDK clamp保留。不能将效果差异仅归因于帧号条件。
其余对照设置：官方 `pi05_base_pytorch_h10` step0，h10，2000steps，warmup100后constant1e-6；
8 GPU BF16 FSDP、micro4/global64/GAS2、seed1000，每500step保存。noise/t保持随机。
独立输出：`$MY_DFS/experiments/carrot/pi05_so101_knock_down_the_cylinder_frame_prompt/knock_down_the_cylinder_1_20260930_222251/training`。

## 未来训练（本轮不执行）

运行前按carrot-recipe核对空闲GPU、当前MY_DFS、已有Ray、资源、配置固定路径与源码版本。
未来获准运行时：

```bash
export SOURCE_COMMIT=<verified-source-commit>
bash recipes/pi05_sft_so101_knock_down_the_cylinder_frame_prompt/run.sh
```

runner复用fit_validation训练入口，不复制trainer，不修改模型结构。预期step500/1000/1500/2000
各有checkpoint、tokenizer、norm_stats和optimizer shards。训练退出、checkpoint完整/可加载与
效果评估需分别提供证据；当前均PENDING。旧评估若不拼相同frame prompt，不能作为本recipe评估。

## 离线 dataset -> log（未来执行）

必须使用本recipe的no-vision服务入口。它复用生产policy factory/WebsocketPolicyServer，并按
现有state_only_rollout评估机制追加共享DropVision、重建input transform；不加载新模型结构。
服务本地适配在SO101Inputs之前补固定全零uint8[224,224,3] wrist输入，仅用于满足现有输入契约；
末端DropVision仍将所有图像pixels与mask清零。生产policy与SO101Inputs未改。
未来空闲GPU且获准启动服务时：

```bash
python -m recipes.pi05_sft_so101_knock_down_the_cylinder_frame_prompt.serve \
  --checkpoint "$MY_DFS/experiments/carrot/pi05_so101_knock_down_the_cylinder_frame_prompt/knock_down_the_cylinder_1_20260930_222251/training/checkpoints/step-00002000" \
  --device cuda:0 --host 0.0.0.0 --port 8080
```

checkpoint应自带tokenizer与norm_stats。服务握手发布 `frame_index_prompt=true`、`no_vision=true`，
客户端启用帧prompt时必须验证两标志，拒绝普通vision服务。服务端DropVision确保所有mask=false。

将下面配置另存为本地绝对路径的 `frame_prompt_offline.yaml`，替换数据、服务和新输出路径：

```yaml
observation_source: dataset
action_sink: log
server_uri: ws://GPU_HOST:8080
dataset_root: /absolute/local/knock_down_the_cylinder_1_20260930_222251
dataset_repo: nrailg/knock_down_the_cylinder_1_20260930_222251
output_dir: /absolute/new/frame_prompt_offline
base_camera: null
wrist_camera: null
cameras: {}
frame_index_prompt_frames: 264
episode: 0
start_frame: 0
fps: 15
execute_steps: 1
max_chunks: null
```

```bash
PYTHONPATH="$PWD/src:$PWD" python -m examples.so101_real.main \
  --config /absolute/local/frame_prompt_offline.yaml
```

总帧数必须等于所选episode长度，使用episode局部frame_index。未覆盖prompt时读数据原task。
本recipe显式设置base_camera/wrist_camera均为null且cameras为空，客户端请求只有state与prompt，
不生成图像、不打开相机。帧prompt开关与相机选择正交，image_keys仅由非null视角配置决定；
如显式选了相机，source照常读取所选图片，robot配置必须精确匹配所选视角。
检查config.json中的frame_index_prompt_frames、events.jsonl的image_mode=no_vision、
frame_index_prompt/总帧数/request.frame/request.prompt，以及NPZ.frame/planned_steps。
image_mode由是否配置视角决定，不由帧prompt开关决定。
默认 `frame_index_prompt_frames=null` 保持普通模式。

## 未来真机 opt-in 示例（默认不得执行硬件）

仅在另行授权真机执行且标定、起点与服务已核对后，将离线配置改为：

```yaml
observation_source: robot
action_sink: robot
base_camera: null
wrist_camera: null
cameras: {}
frame_index_prompt_frames: 264
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

保留实测state，始终从配置原始prompt拼帧号；SDK收到空camera配置，不打开相机。预热、重复read、
到位等待与反馈采集均不推进index；只有实际采用动作后的source.advance(count)推进。
h10/K1每发送1个推进1，K5推进5；末尾不足K只消费剩余帧，到N停止，不循环或夹在末帧。
index表示已发送动作进度，**不是已到位进度**。现有等待超时记录残差后继续，因此pose/index
可能不同步；限位、PID和网络行为沿用现有客户端。

## 2026-10-09 准备记录

正式CPU契约pytest：NOT RUN/PENDING，见 `tests/so101_real/test_so101_frame_prompt.md`。
仅本地compileall、bash -n及diff检查；未验证实际tokenizer、checkpoint加载或模型效果。
训练/服务/模型推理/真机：NOT RUN。实际Docker image tag、Carrot commit、上游commit、
运行任务ID、训练日志与checkpoint：未记录（尚未运行）。无实验效果结论。
