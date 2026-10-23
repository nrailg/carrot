# 多色换笔临摹 RL → LIBERO

状态：正式pilot已启动，当前准备/示范采集阶段；无迁移效果结论。

起点是既有 LIBERO SFT `step-00000100`。历史 745/2000（37.25%）只用于说明模型来源，
不能替代本实验新跑的配对 init 基线。三个对照为 init、临摹示范 SFT、临摹 PPO 最终 checkpoint；
初始化后不使用 LIBERO 训练数据，不按 LIBERO 评测选 checkpoint。

单 Panda 真实抓取红、绿、蓝三支独立刚体笔，在画布上临摹后归还。笔尖与笔杆之间是固定关节，
笔与手之间没有固定关节；只有 reset 可以设置笔的位姿。只有笔尖接触画布才出墨，离纸断线。
oracle 只按生成的笔画程序给出 pick/draw/place 子任务 prompt，并在场景参考板显示当前画稿；模型看不到物体位姿、
接触力或示范动作。目标端直接执行 LIBERO 原始 prompt，不加 planner。

源任务和目标 LIBERO 复用完全相同的输入变换：base/wrist 两张 RGB 与任务文字，第三图像槽保持 mask。
目标画稿仅显示在仿真场景里的参考板上，通过相机观察，不额外传入原始参考图。
接口保留 8D EEF 位姿/双指状态，但与 OpenPI 官方 `pi05_libero` 配置一致，
`discrete_state_input=False`，π0.5 的连续状态分支也关闭，因此状态不进入策略网络。
这描述的是该模型配置；LIBERO 官方基线本身提供并使用本体状态。核对依据：
[LIBERO默认观测](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/configs/data/default.yaml)、
[OpenPI配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)、
[π0.5状态分支](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)。
输出为 7D EEF delta/gripper：位置尺度 0.05m，
旋转尺度 0.5rad，正夹爪动作闭合；240Hz 物理、20Hz 控制。模型 horizon=10，最多执行 5 步；
子任务或 episode 边界提前结束 chunk，按实际长度计算概率 mask、折扣和 SMDP GAE。
EEF 使用手掌下方0.097m抓握中心和LIBERO夹爪旋转约定，双指状态一正一负；位置加固定
MountedPanda基座偏移[-0.66,0,0.912]，不含物体真值。沿用Workbench机械臂重力补偿，笔仍受重力。

奖励为分色 IoU 增量 + 单次抓取/归还各 0.1 + 最终成功 1。成功要求 IoU≥0.72、任务笔均归位、
夹爪松开且移开。颜色错误、额外涂画降低 IoU。正式任务至少两色，默认三色；训练种子低于 1,000,000，
评测从 1,000,000 开始。示范课程从一色过渡到三色，收集 64 个成功 episode，最多启动 128 次。

VLM 冻结；动作专家、action projection、time MLP 使用 FP32 master 和 BF16 forward。
PPO 每个 chunk 随机选择一个 Flow-SDE 去噪转移，其余走 ODE；5 denoise、noise=0.5，
clip=0.2、γ=.99、λ=.95、1 epoch，策略 lr=5e-6、价值头 lr=1e-4。
SFT 最多 500 updates，batch16，lr=1e-5；监督目标不跨子任务，参考颜色不做随机增强。
每次 PPO 更新前重算全部 old logprob，并检查骨干没有梯度。
每10次PPO更新保存一次可恢复的learner checkpoint；最终对照仍固定采用最后一个RL checkpoint。

24h pilot：准备/示范/SFT 最多 6h，RL 最多 10h 或 200k 实际控制转移，余下时间评测和归档。
截止检查在动作块/优化步边界，正在执行的一次调用和最终保存可能产生少量超时。
源码冻结和真实物理验收通过后才启动该预算。没有成功示范或 SFT 更新时直接失败，禁止空跑 RL。

目标评测用固定原始统计量、10 denoise、execute5；Spatial/Object/Goal 为主指标，Long 单列。
三组权重均先通过相同 BF16 导出路径再加载评测，避免原始 FP32 与新 BF16 导出的精度差异。
按 episode 轮次优先完成 40 tasks ×3 checkpoints，再添加下一轮，目标共 600 episodes。
只汇总三组都完成且初态 SHA/seed 一致的 case，未配对结果单独保留。源任务另记 IoU、抓放次数、
掉笔、成功率和耗时。代表性视频和逐 episode JSON 随结果持久化。

## 资源与运行

渲染限定 RTX PRO 5000；H20 用于模型。H20 同卡多个独立 renderer 存在已知故障风险。
两个池分别检测 `MY_DFS`、同步源码、核对 SHA 和已有 GPU 进程；不得停止他人的 Isaac/Ray 任务。
本次 Pro5000 会话为 72GB 卡，DFS 为 zjk1；H20 DFS 为 hz1。双向 TCP 与两节点 Ray 加入已通过。
不假设两池共享目录：renderer 数据经 Ray 分块传输到 learner DFS，每个文件校验 SHA-256。

1. 先运行 `tests/multicolor_transfer/test_transfer.sh`、`test_physics.sh`、`test_policy.sh`。
2. 各节点配置已确认的 `MY_DFS`、`RAY_HEAD_IP`、空闲 `CARROT_VISIBLE_GPUS` 和 dguard 时长，
   分别执行 `start_ray.sh head`（H20）及 `start_ray.sh worker`（Pro5000）。只加入需要的两个节点。
3. 在 H20 配置 `LEARNER_DFS`、`SIMULATOR_DFS`、两边 node IP、`RUN_NAME`、`RAY_ADDRESS`，
   执行 `bash recipes/pi05_multicolor_transfer/run.sh`。YAML 内所有环境变量必须解析成功。
4. 续训使用新的 `RUN_NAME` 并追加 `--resume <可信 checkpoint>`。恢复 FP32 权重、critic、optimizer、
   RNG 和计数；不恢复物理轨迹，从新 episode 开始，不能称为逐位仿真恢复。

持久化目录为各自 DFS 的 `experiments/carrot/pi05_multicolor_transfer/<RUN_NAME>`。
learner 目录含 config、源码 SHA、events、SFT 日志、demos、checkpoint、源/目标逐 episode 指标和 result。
renderer 目录含 simulator.log、原始 demos 和 source snapshots。`learner.pt` 只允许加载可信本实验文件。

## 验证和运行记录

2026-10-23：19 项 CPU 契约测试通过；H20 真实模型 PPO/SFT 反传、冻结骨干、
checkpoint 导出/FP32 optimizer/RNG 恢复通过，old logprob 最大误差 3.81e-6，峰值19.97GiB。
两池真实 Isaac/PPO/LIBERO 闭环两次通过；第二次 old logprob 最大误差4.77e-7。
用户改为场景参考板后，双相机模型和闭环均重新通过；改变8D状态不改变固定seed动作输出。
第三轮模型old logprob误差7.45e-8，第三轮跨池闭环误差0。
第八轮物理预检两个环境均两色成功，IoU0.803491/0.795135，均2次抓取/归还、零掉笔，
hold/partial reset与相机检查通过。三色64个示范仍需在正式采集中验收。
Carrot 基线 `ff9e882` 加未提交改动；RLinf 参考 commit 见 NOTICE；运行镜像 tag 未记录。
实际开始时间、MCP task ID、最终步数、退出码、checkpoint 完整性和配对指标待运行后补充。

### pilot-20261023-01

2026-10-23 23:21:09（Asia/Shanghai）启动，H20 MCP task `8ec98b1f-0034`。
24h预算从driver的events.jsonl中start时间开始计算；当前运行中，尚无SFT/RL完成结论。
23:22:44已保存init checkpoint（model.safetensors、FP32 learner.pt、manifest/config/norm齐备），
4env partial reset通过（3个未重置slot保持不变），已进入64成功示范采集。
渲染Pro5000 launcher GPU1，模型H20 launcher GPU0；沿用已验证Ray `29.209.162.214:6379`。
两池dguard暂停2880min，未停止Pro5000 GPU0已有用户仿真。

两池均从本地已同步源码按88文件SHA清单复制到独立source目录，脱离后续Mutagen更新；
清单SHA-256为 `b9196d520bd12010503e6c919cd41c7d859da2b83618252c116497e8842ec3c5`。
H20同步目录另有旧模块，此次snapshot只包含本地清单，未删除或覆盖那些文件。

H20输出：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_multicolor_transfer/pilot-20261023-01`。
Pro5000输出：`/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_multicolor_transfer/pilot-20261023-01/renderer`。
两个池的源码分别位于同一experiment根目录的`sources/pilot-20261023-01`。
H20的`resolved-pilot.yaml`固定Pro5000 snapshot的simulator_source、Isaac和LIBERO入口路径。
实际入口：设置上面的两池DFS/node IP、RUN_NAME及RAY_ADDRESS后，
`CARROT_SOURCE_DIR=<H20 snapshot> bash <H20 snapshot>/recipes/pi05_multicolor_transfer/run.sh --config <H20 snapshot>/resolved-pilot.yaml`。
完整实参由输出config.yaml保存；driver事件在events.jsonl，渲染/示范进度在Pro5000的simulator.log与demos/collection.json。

首批已完成3个成功示范（seed15/12/13），IoU0.8042/0.8348/0.8018、均零掉笔，已入库3/64；
这是单色课程阶段，尚未完成三色示范采集，尚未开始SFT/RL更新。
