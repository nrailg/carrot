# LeRobot PI0.5 / Arm Dance 官方实现对照

2026-10-09：使用独立 venv 和官方 LeRobot 0.6.1，复查之前 Carrot Arm Dance 的训练与
离线拟合。全部新增代码局限于本 recipe；不 import Carrot，不修改已有代码、共享模型或数据。

截至2026-10-10：2000step训练、144帧离线play和144步真机闭环均已完成验收。
离线五轴首动作MAE2.81291°；真机执行跟踪MAE0.664102°，模型对同序号示教MAE6.583984°。
真机全程Status0、无等待超时，但肩/肘轨迹仍有明显波动，尚未稳定复现示教。
详细协议、失败重试、执行频率和归档证据见下方完成记录。

数据 `nrailg/arm_dance_20261008_225311_20261008_225313`：单 episode、144帧、15FPS，
任务 `Dance with the arm`；五运动轴为角度，gripper 保持录制单位。
初始化 `$MY_DFS/hf-hub/lerobot/pi05_base`，tokenizer 使用已有本地
`$MY_DFS/hf-hub/google/paligemma-3b-pt-224`，全程离线。

## 对照条件与边界

- 无视觉：三个 image tensor 置零，所有 image mask 置 false。
- 训练 raw state 的五轴独立 uniform ±3°，按原 SO101 标定范围 clamp；gripper、action、统计量不变。
- h10、每次执行一步、seed1000、global batch64、2000step、每500step保存。
- AdamW LR1e-6、WD1e-10、betas[0.9,0.95]、eps1e-8、grad clip1；warmup100后LR恒定1e-6。
- 离线 play 用录制 state，不加 jitter，同样屏蔽视觉。

官方 PI0.5 没有关闭全部 image mask 的配置，因此通过官方 policy/processor 注册接口加载
本 recipe 的最小适配：`PI05ControlPolicy` 只覆盖图像输入预处理，`StateJitterStep` 只处理
归一化前 state。官方 transformer、attention、flow matching、loss、optimizer、训练循环及
动作推理方法未修改；不是不加适配的原生 CLI。每次预检核验 wheel 中490个Python文件的SHA。

仍有实现差异：原 Carrot 是8卡FSDP/micro4/GAS2，本次为官方8卡DDP/batch8；原版本有FP32
优化器主参数和FP32归约，本次遵循官方混合精度/优化器行为。原loss按32维并屏蔽尾部padding，
官方loss按真实6维、保留尾部重复动作；原state prompt的padding与官方可能不同。
本次初始化用LeRobot官方base，原用OpenPI h10转换版，尚未证明两份权重逐项一致。
上述差异均保留，不能仅凭loss数值判断Carrot是否有BUG。

## 环境、命令与验收

独立 venv `/opt/venvs/lerobot-official-arm-dance`，无 system-site-packages，未安装Carrot。
官方PyPI wheel `lerobot-0.6.1-py3-none-any.whl` SHA256：
`1894516040c65f80a45bd9741f8174aae90ed5d93da0627ab4f1a85fd8d75e90`。
依赖从已有环境按METADATA闭包独立复制，缺少/版本不符的包用官方wheel补齐；106包依赖检查通过。
包来源及安装记录在 `$MY_DFS/experiments/carrot/lerobot_official_arm_dance_packages/`。
Docker image tag及上游源码commit未记录；PyPI版本和wheel SHA可溯源。

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
export RUN_ID=<unique_run_id>
export SOURCE_COMMIT=<verified_commit>
bash "$MY_DFS/work/carrot/recipes/lerobot_pi05_arm_dance/run.sh"
```

独立输出 `$MY_DFS/experiments/carrot/lerobot_pi05_arm_dance/$RUN_ID/`：保存源码快照、
预检环境/数据/权重SHA、训练命令、exit code、官方checkpoint及离线预测/逐轴指标。
开始前核对GPU仅有dguard，runner暂停dguard并在退出时恢复；不覆盖任何旧run。
预检先检查真实数据、视频、quantiles、tokenizer与输入适配，然后用3帧base离线推理验证
812个权重完整加载与全false视觉mask，再进入2000step训练，最后对144帧做离线play。

训练成功需exit0/2000step/4个完整checkpoint/有限loss及gradient；checkpoint可加载和
离线拟合另行验收。离线报告首动作和有效chunk的逐轴MAE/P50/P90/max，排除尾部padding，
gripper独立统计。离线拟合不代表真机闭环成功；本runner不操作机器人。

## 历史与运行记录

原Carrot训练重新核验：exit0/2000step/四个完整checkpoint/8rank有限非零更新。
原tokenizer本地五文件无需模型config.json，两venv词表SHA与示例token一致。
之前使用Hub ID+空cache引发配置请求失败，属于本轮路径偏移，不能据此说旧tokenizer缺失。
之前本recipe草稿还错误使用了视觉与官方默认LR/WD，现已对齐上述条件。

- `smoke_20261009T122700Z`：预检失败（Hub tokenizer路径）。
- `smoke_20261009T123900Z`：预检失败（numpy统计量类型检查），已修正。
- `smoke_20261009T124300Z` / task816a3192-0695：CPU预检通过，base加载中按用户要求停止，
  exit-15；未进入训练、未操作机器人，dguard已恢复。
- `controls_cpu_20261009T134100Z` / task816a3192-0711：CPU预检exit0；490源码SHA、
  真实144帧数据、tokenizer、3路图像全零/全false mask、jitter/gripper不变性、106包依赖通过；五轴标定范围与原SDK逐项一致。
- 正式运行计划 `matched_20261009T123800Z`：branch/commit及同步SHA核验后启动2000step；
  本runner从run目录中的源码快照执行，避免运行中工作区变化影响实验。

参考：[官方训练/推理](https://huggingface.co/docs/lerobot/il_robots)、
[PI0.5](https://huggingface.co/docs/lerobot/pi05)。

- `matched_20261009T123800Z` / task816a3192-0714：12:37:00Z启动，CPU预检通过，
  官方加载全部key成功；base-play权重检查错误要求812归档张量=813个state-dict键，exit1。
  官方loader从lm_head恢复共享embedding别名，修正检查为逐张量相等+别名data_ptr相同，
  未进入训练，dguard恢复。补充完整CPU processor加载/归一化逆变换检查与显式PI0.5注册。
- `processor_cpu_20261009T124100Z` / task816a3192-0718：exit0；完整官方processor加载、
  state tokenization、action归一化逆变换通过。
- `matched_20261009T124200Z` / task816a3192-0720：144s后exit1，官方812张量加载成功，
  检查进一步发现embedding恢复为独立clone，非共享存储；改为缺省embedding从源lm_head逐值精确核验。
  CPU预检增加独立副本可通过、篡改副本必失败检查。仍未进入训练，dguard恢复。
- `weight_cpu_20261009T124500Z` / task816a3192-0721：exit0，embedding独立副本及故意篡改
  检查通过，完整processor/数据/源码预检仍通过。

## 正式训练已启动：2026-10-09 20:50 +08

- 分支 `codex/lerobot-pi05-arm-dance`；实际训练源码提交
  `6b9e809557fdf38e8bcc16e9eacc779778975b0a`，远端10文件SHA已核验。
- run `matched_20261009T124600Z` / Gemini task `816a3192-0723`。
- 持久化目录：
  `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/lerobot_pi05_arm_dance/matched_20261009T124600Z/`。
- CPU预检通过；base的812归档张量及恢复的embedding逐值核验通过，3帧no-vision离线推理通过。
- 12:50:23Z正式进入训练；已亲核8个rank、8GPU均100%利用率、约49.5GiB占用。
  前20步loss/gradient/LR全部有限；step1 loss0.841，step20 loss0.837、grad norm7.030、
  warmup LR2.1e-7。官方确认batch8×8=64、2000step。
- 证据：`preflight.json`、`base_play/metrics.json`、`train_command.txt`、
  `startup_backend_snapshot.log`、`startup_audit.json`；完整后台日志在任务结束后另行dump。
- 状态：训练运行中；尚未验收2000step/checkpoint/完整144帧离线play。runner在训练成功后
  自动执行离线play，退出时恢复dguard；本次未操作机器人。

## 完成核验：2026-10-10

任务 `816a3192-0723` 训练和离线play均exit0；最终评测文件生成于2026-10-09
22:06:04 +08。完整后台输出已保存至该run的`backend.log`并释放MCP任务条目。
`completion_audit.json`记录实际训练源码commit、四份checkpoint、最终指标和离线结果。

- 日志精确包含2000条优化更新，loss/gradient/LR全部有限；末步loss0.313、grad norm2.531、
  LR1e-6，末100步平均loss0.26842。
- `000500`/`001000`/`001500`/`002000`四份checkpoint的保存步数分别核对一致，
  每份都有813张量model、processor/statistics、optimizer、scheduler、RNG和training_step。
  `last`指向`002000`；最终checkpoint813张量已由离线play逐值核验并成功加载。
- 离线预测shape(144,10,6)，144帧、1395个有效chunk动作；NPZ全部有限，
  已从预测/target重新计算首动作和有效chunk MAE，与metrics.json一致。
- dguard已恢复，watch1/guard/run.py运行，无待恢复计划；未操作机器人。

| 首动作轴 | MAE |
| --- | ---: |
| shoulder_pan | 0.175° |
| shoulder_lift | 4.207° |
| elbow_flex | 8.989° |
| wrist_flex | 0.535° |
| wrist_roll | 0.158° |
| gripper | 0.0218（录制单位） |

五轴首动作平均MAE **2.81291°**，肩抬/肘误差明显；肘首动作最大误差32.58°。
这是无视觉、录制state/no-jitter的teacher-forced离线评测，尚未做真机闭环。
官方训练和推理链路可运行，但本配置尚未完全拟合；前述优化精度、loss、初始化等实现差异
仍存在，不能用本轮loss直接判定Carrot代码BUG。

## 2026-10-10 官方远程推理与真机验证

目的：加载上述 002000 checkpoint，以官方 `async_inference.PolicyServer`、官方
`predict_action_chunk`、前后处理器和 gRPC transport 执行 Arm Dance 无视觉闭环。
模型位于 Gemini，SO101 连接本机 Mac；不使用 Carrot 模型或客户端。
`serve.py` 只注册已有 `pi05_control`、校验权重/输入、添加 BF16 autocast 和运行记录。
推理导出目录链接原始权重，仅将 processor 的 state jitter 关闭。
`live.py` 使用官方 RPC stub 和 SOFollower SDK，以同步 K1 保留原实验的等待语义；
这不是直接调用 `lerobot-rollout` CLI，也不使用官方异步客户端的动作队列合并。

协议：h10/K1、NFE10、随机 seed1000、任务 `Dance with the arm`；零图像占位且三路 mask
全部为 false；先做三帧网络 smoke，再缓慢对齐录制首帧，执行 144 次真实 state 闭环。
每步最多等待 3 秒至五轴误差 ≤3°，超时记录并继续，目标相对限幅20°并限制标定范围。
直接连接 bus 以保留肘/腕 P32、其余 P16、I0/D32，不调用会重写 PID 的 `robot.connect()`。
退出写入当前 raw 位置保持扭矩，记录 Status 和前后寄存器。
验收：官方权重813张量逐值一致、三帧 smoke 有限动作、真机144条记录/退出0、Status0、
轨迹与指标完整；程序完成与动作效果分别报告。
所有源码、配置与验证记录限定本 recipe；本机独立 client venv 位于
`~/.cache/carrot/lerobot_official_arm_dance/client_venv`。

启动前硬件预读：原 follower USB 已恢复，六轴 Status0；当时尚未发送本轮运动指令。

本次运行：Gemini task `bb81765e-0741`，实际启动约 2026-10-10 06:20Z，目录
`$MY_DFS/experiments/carrot/lerobot_pi05_arm_dance/live_20261010T062100Z`。
启动命令：`MY_DFS=<本会话已确认DFS> LIVE_RUN_NAME=live_20261010T062100Z bash recipes/lerobot_pi05_arm_dance/run_live_server.sh`。
GPU grpcio1.84.0来自已有依赖的RECORD文件复制；107包uv check通过。
Mac grpcio1.78.0/protobuf7.36.1，独立环境pip check与两个客户端契约测试通过。
本机输出：`~/.cache/carrot/lerobot_official_arm_dance/live_20261010T062100Z`。
上电Torque/Lock0，先运行`prepare_robot.py --output <本机run>/prepare_robot.json`，
逐值写入当前raw Goal后启用保持，PID不变。模型加载与真机执行的后续结果另行追加。

实际完成客户端命令（第一次类型校验失败后，复用同一已加载且核验813张量的服务）：

```bash
LOCAL_RUN="$HOME/.cache/carrot/lerobot_official_arm_dance/live_20261010T062100Z"
CLIENT_PYTHON="$HOME/.cache/carrot/lerobot_official_arm_dance/client_venv/bin/python"
REMOTE_RUN="$MY_DFS/experiments/carrot/lerobot_pi05_arm_dance/live_20261010T062100Z"
"$CLIENT_PYTHON" -u "$LOCAL_RUN/source_retry/live.py" \
  --server "$GPU_ADDRESS:8080" \
  --checkpoint "$REMOTE_RUN/server/inference_model" \
  --demonstration "$LOCAL_RUN/demonstration.json" \
  --output "$LOCAL_RUN/client_retry" --play --reuse-loaded-server
```

`MY_DFS`、`GPU_ADDRESS`来自本轮已确认的Gemini会话。
`demonstration.json`由原数据集parquet按录制顺序导出，包含原始`states`/`actions`各144×6，
以及`action_names`六轴顺序，已随客户端产物归档。上述命令记录的是历史执行，重复运行需使用
新的输出目录。新服务首次连接不传`--reuse-loaded-server`；该选项仅用于复用已核验的同一服务。

完成结果（2026-10-10 06:23:49Z / 北京时间14:23:49）：

- 官方服务813权重张量逐值核对PASS、每次输入三路零图像/false mask，state jitter关闭。
- 第一客户端三帧smoke通过，因官方相对限幅拒绝int20，在第一次对齐发送前exit1；
  raw保持成功。修为float20.0后复用同一已核验服务（Ready清队列、不重载），
  再做三帧smoke并完成144步；本轮seed1000后共六次smoke消耗随机采样。
- `client_retry`进程exit0、144条唯一step0..143且输入/输出/实际位置全部finite；
  每步Status0、144次都在五轴3°阈值内、超时0。
- 五轴执行跟踪MAE0.664102°；逐轴[0.170568,1.471661,1.240067,0.305530,0.132685]°。
- 模型输出对同序号示教五轴MAE6.583984°；逐轴[0.166412,13.526941,18.289799,
  0.746718,0.190051]°。夹爪模型/示教MAE0.027324、执行MAE0.416671，单独的录制单位。
- 5步发生相对限幅，1步发生绝对限幅；有效闭环约58.55s/2.46Hz，平均RPC0.28377s。
  示教录制15FPS，因此同序号MAE不等于按时间对齐的轨迹误差或成功率。
- 退出raw保持读回PASS；前后PID一致、Torque/Lock1、Status0。
  服务在客户端完成后主动SIGINT收尾，服务exit130是预期人工停止，不能写成服务exit0；
  dguard恢复watch1/guard/run.py、无restore计划，GPU仅其占用，8080监听已消失。

验收结论：官方远程推理与真机执行链路已跑通，当前checkpoint的肩/肘轨迹仍有明显波动，
没有复现示教轨迹；不能据此单独归因Carrot代码或排除其他实现差异。
持久化产物位于该run的`server.log`、`server/loaded.json`、`server/predictions.jsonl`和
`client_artifacts/`（完整日志、两次客户端源码快照、`completion_audit.json`、
`client_retry/trajectory.jsonl`、`metrics.json`、前后寄存器、raw保持、`trajectory.png`）。
本机同名cache目录保留可直接查看的轨迹图和JSON。

最终审计：运行后再次核对官方490个Python源文件与固定PyPI wheel逐字节一致；
服务端150条预测（六次smoke＋144真机）与客户端全部144条正式输入/输出逐项精确匹配，
结果写入`server_audit.json`。本轮新增脚本ruff全通过，两个客户端契约单测通过。
验证完成时工作区HEAD为b98e2f3；后续归档提交不代表实际执行源码。
训练实际源码commit保持6b9e809；真机实际源码以run/source与client_artifacts/source_retry快照为准。
