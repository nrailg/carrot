# Knock down the cylinder：frame prompt + no vision + clamped state jitter

状态（2026-10-09整理）：正式CPU契约39/39 PASS，真实数据/tokenizer预检PASS；2000-step训练已启动。本档案最后留存的检查为17:44 Asia/Shanghai、1545/2000 steps，15分钟后台watchdog当时已启用。**最终训练验收和效果评估待补**，不把该历史检查当作当前进程状态。
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


## 2026-10-09 最终方案与调用链留档

开发版本：`8d474105e173cd7f62efe61a515fdf2bf4fd4d23`，已推送`mygh/testRealRobo3`；这是准备代码版本，
不是已运行实验版本。当前事项是最早cylinder单episode264帧，不是另一AI负责的Arm Dance。
用户最终选择prompt帧号替代帧号图片，保留no vision、±5° state jitter/SDK clamp、h10、
官方base step0与2000step预算；未启动本实验。

| 位置 | 职责 |
| --- | --- |
| `examples/so101_real/main.py` | CLI接收`--frame-index-prompt-frames 264`，表示启用条件并设定总帧数，不是当前帧号 |
| `examples/so101_real/frame_prompt.py::format_frame_prompt` | 唯一格式`Knock down the cylinder Frame: 0053.`；04d为最小宽度，不限制到9999 |
| `augmentation.py::FramePromptDataset.__getitem__` | 从原始任务文本和样本index生成训练prompt，不改监督action/stats |
| `examples/so101_real/observations.py::DatasetSource.read / RobotSource.read` | 每次从原task/config.prompt与self.frame构建请求，重复读取不累计后缀 |
| `examples/so101_real/runner.py::run_loop` | 单进程循环复用模型服务；发送count个动作后advance(count)，不是每帧启动进程 |
| `serve.py::enable_no_vision` | 服务端补零腕图满足SO101Inputs，末端DropVision清零所有pixels和mask |

相机配置与帧号条件独立；本recipe客户端base/wrist均null、cameras为空，只发送state和prompt。
等待、预热不推进；K1每次预测10条只执行首条，K5执行前5条，到264停止。等待超时继续时，
帧号代表已发送动作数，并不证明机械臂已到位。

| 验证事项 | 本轮状态 |
| --- | --- |
| Python语法、bash -n、git diff --check | PASS，本地静态检查 |
| 主模型独立fake RobotSource：K1/K5完整264、重复read不推进、无图请求、相机配置独立、index10000合法 | PASS，纯CPU轻量检查，非pytest |
| 正式pytest、完整服务输入变换与实际tokenizer契约 | NOT RUN / PENDING |
| 2000step训练、checkpoint可加载、动作拟合、末段抖动/真机 | NOT RUN / PENDING |

下一步先在空闲且授权的远端环境运行`tests/so101_real/test_so101_frame_prompt.sh`，
核对实际tokenizer保留帧号与state条件、训练/推理最终pixels/mask一致，再按recipe启动新训练。
本次只记录，不连接远端、不启动服务或机器人。效果尚未知；即使有效也仅证明外部进度条件下的行为，
不能据此确认原抖动只由阶段歧义造成，或宣称自主识别阶段/泛化。


## 2026-10-09 16:56：正式启动2000-step训练

目标是验证在原cylinder demo的状态附近，显式进度条件能否减少阶段歧义并支持稳定推进。
先确认帧prompt条件下的动作拟合，再测试同条件下连续state反馈，最后另行授权真机h10/K1 replay。
拟合、离线反馈与真机是不同验收；效果改善也不能单独证明原抖动因果（jitter同时±3→±5）。

| 项目 | 本轮证据/状态 |
| --- | --- |
| 正式CPU契约 | 39 passed in 9.05s，task4df664ae-0634 exit0，已dump归档 |
| 真实数据/tokenizer预检 | task4df664ae-0636 exit0；264帧/单episode/15FPS；264个帧条件token序列互异 |
| 训推条件 | 同state下token/mask相等，视觉pixels0/maskfalse，action/padding与原Parquet一致，24次jitter范围检查PASS |
| 启动 | 2026-10-09 16:56:32 Asia/Shanghai，session4df664ae，后台task4df664ae-0637 |
| 资源 | 8×H20，复用空闲Ray29.209.160.111:6379/job1a000000；dguard暂停120min且run.py实际停止 |
| 初始进展 | 16:58:35亲核step1–3连续有限，step3 loss0.094971/grad_norm2.06549/LR3.960e-8，warmup正确 |
| 版本 | Mac HEAD db756f9cf6ff3643f9db2a635b86265c8486bf53，核心方案8d47410；87运行文件Mac/GPU SHA一致 |
| 初始化/预算 | 官方h10 base从step0新训，2000step/save500，不resume、不追加预算 |

训练输出：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_so101_knock_down_the_cylinder_frame_prompt/knock_down_the_cylinder_1_20260930_222251/training`。

证据根：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_so101_knock_down_the_cylinder_frame_prompt_monitor/20261009T085419Z`，含source_snapshot/expected_source_hashes、environment、pytest_backend.log、preflight.py/json/backend、launch.json、monitor.py、ray_logs、training_metrics.csv及latest_training_status.json。

实际Python3.12.13/torch2.11.0+cu128/LeRobot0.6.1/transformers5.5.4/Ray2.58.0/NumPy2.3.1；
Docker image tag及上游commit未记录。Mutagen当前唯一冲突是无关旧wipe recipe远端pycache，
未清理或覆盖；本轮必要文件逐个SHA已通过。Ray dashboard HTTP不可用，监控改为读取本job原worker日志。

后续检查后台task及rank0所有新增指标，记录连续性/有限性/LR；发生NaN/Inf或实际崩溃停止本job，
不自动重训、不影响其他任务。完成后核对exit0/step2000/四checkpoint参数更新、BF16/config/stats，
归档task、确认dguard恢复，再做带相同帧prompt的离线拟合/连续反馈评估。本次未驱动机械臂。


2026-10-09 17:26:24 Asia/Shanghai进展：task4df664ae-0637仍running，亲核step1–950连续、loss/grad/LR全部有限、warmup/constant正确。step950 loss0.005233/grad_norm0.198066/LR1e-6，最近100step平均loss0.00429964。step500已保存：trainer.step500，812 tensors全部BF16，config BF16+h10，norm_stats/tokenizer存在、optimizer8shards；未做checkpoint推理或效果评估。dguard watch0/run.py停止/restore计划仍在，暂停时间足够当前剩余预算。monitor更新ray_logs/CSV/latest_status和progress_checkpoint500.json。预计剩余约30–35min（据当前吞吐估算）。未重复启动、追加预算或操作机器人。


### 2026-10-09 17:44：启用15分钟后台监控

此前是用户询问后手动检查，没有自动检查调度。现已启动独立watchdog task4df664ae-0648（PID636988），每900秒检查本训练driver626482及job1a000000全部rank0指标，调用既有monitor.py持久化日志/CSV/status。首次检查17:44:12完成：step1–1545连续有限、LR正确；step1545 loss0.001918/grad_norm0.149079/LR1e-6。watchdog_status为watching，下次17:59:13 Asia/Shanghai。dguard watch0、run.py停止、restore计划在。

watchdog.py、watchdog_status.json、watchdog_checks位于既有证据根20261009T085419Z，不入Git。发现NaN/Inf只对核验命令行与启动身份的本训练driver发送SIGTERM并归档；检测失败退出待人工核验；driver退出时标记待最终审计，不据2000行日志宣称成功。无自动重训、无机器人操作。此后台脚本不能主动向Codex对话发消息；训练完成仍须核对退出码/checkpoint/资源释放和dguard恢复。
