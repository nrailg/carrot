# 多色换笔临摹 RL → LIBERO

状态：SFT500步已完成，每50步权重保留。2026-10-24按用户要求停止旧RL；step100/500各两条训练内单色闭环均0成功。当前先诊断SFT和推理链路，验证后再改完整trajectory及每轮1～2次更新的RL；尚无收敛或迁移效果结论。

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
SFT 最多 500 updates，global batch 256，lr=1e-5；监督目标不跨子任务，参考颜色不做随机增强。
SFT 与 PPO 共用 `micro_batch_size=8`、`global_batch_size=256`；单 learner 自动累积32次梯度。
`global_rollout_batch_size=4096` 按环境/action chunk计数，`num_envs=32` 时自动采集每环境128 chunks；
每轮 PPO 1 epoch、16次optimizer step。渲染worker不计入DP；当前DP=1，未声称使用全部H20训练。
micro 同时限制采样、推理、价值计算和反传批量；变长SFT目标先按样本平均再累积，避免分块改变权重。
不足4096的预算尾部只记账并丢弃，不进行缩小batch的更新；一次完整PPO更新都没有则报错。
示范仍用 `demo_num_envs=4` 维持一色到三色课程，SFT后切换32环境。
`demonstration_source` 可显式指向前次renderer demos目录，只复制有result.json提交标记的成功episode，
校验必需数组并保留来源，新目录继续采集；旧目录不修改，旧失败不计入新run的attempt预算。
每次 PPO 更新前重算全部 old logprob，并检查骨干没有梯度。
每10次PPO更新保存一次可恢复的learner checkpoint；最终对照仍固定采用最后一个RL checkpoint。

24h pilot：准备/示范/SFT 最多 6h，RL 最多 10h 或 1,024,000 实际控制转移，余下时间评测和归档。
转移预算足以覆盖至少50轮完整rollout（满长chunk时每轮20,480转移），但10h时限可能先到；不承诺收敛。
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
24h预算从driver的events.jsonl中start时间开始计算。本轮已为全局batch配置调整而停止，SFT/RL更新均为0。
停止时已提交20个成功示范，供新run复用；旧配置为SFT GBS16、PPO GBS8、4环境×64chunks。
原始driver日志保存在driver-stopped-for-batch-reconfiguration.log，旧产物与源码快照保留。
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

### 全局批量配置验收

基础实现已提交23b8ba8；新增micro/global batch配置后，22项CPU检查通过。
H20 policy-20261023-04真实完成GBS256/micro8的PPO与SFT各一次，梯度累积32次，
old logprob最大误差3.8147e-6；精确恢复、状态不变性、拒绝小rollout与丢弃尾部均通过。
跨池cross-pool-20261023-04通过32环境/64相机、31slot partial reset、97文件SHA、
160实际转移和4个LIBERO suite接口检查。测试不构成收敛或迁移效果证据。

### pilot-20261023-02

运行源码提交 `71ff4b9`，2026-10-23 23:57:26（Asia/Shanghai）启动。
原始 H20 MCP task `8ec98b1f-0047`；MCP会话重连后以持久化事件及进程验证状态。
截至2026-10-24 00:27，driver PID22325与renderer PID1248346均在运行，init checkpoint已保存，
4env partial reset通过。已复用pilot-01的20个成功示范，并新增12个，共32/64；
最近四个双色任务IoU0.7911–0.8004，均两次抓取/归还、零掉笔。SFT/RL正式更新仍为0。
初始化曾有carb.launcher辅助子进程超时错误，主仿真随后完成初始化并持续采集，未因此重启。

配置：micro8、SFT/PPO global batch256、global rollout4096；RL32环境×128chunks，
单learner梯度累积32次、每轮16个optimizer steps、1 PPO epoch；SFT最多500步。
准备6h/RL10h/总24h硬预算，RL转移上限1,024,000；示范课程仍4环境，SFT后切换32环境。
两池88文件冻结清单SHA256：`208722772efb8e34c9d7ce856a545f24dc895f7595fdc0c8307f75401f583592`。
源码分别位于各池个人DFS的`experiments/carrot/pi05_multicolor_transfer/sources/pilot-20261023-02`，
H20的resolved-pilot.yaml固定Pro5000源码与旧示范来源，运行不受后续Mutagen修改影响。

H20输出：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_multicolor_transfer/pilot-20261023-02`。
Pro5000输出：`/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_multicolor_transfer/pilot-20261023-02/renderer`。
观察H20的events.jsonl/sft.jsonl，以及Pro5000的demos/collection.json与reuse.json。

### 2026-10-24 09:26 中断核查（Asia/Shanghai）

Pro5000的collection.json确认64/64成功示范（started67，达到目标后interrupted3）；
数据保留在上述renderer/demos。H20在02:31:04停止写入，demos有435个文件、
37,814,804,793字节；无传输完成事件，无sft.jsonl/result.json，仅有init checkpoint。
Pro5000的Ray日志（UTC 18:32:06，即北京时间02:32:06）报告GCS失联超过60秒并退出，
本实验renderer worker随之退出；不能据此确定H20容器为何失联。
09:24用原连接重新登录H20仍鉴权后立即断开；通过DevCloud只读访问H20持久化目录完成上述核查。
正式SFT/RL更新为0，未达到前两步验收；未新建或重启实验，未修改已有数据或停止用户GPU0仿真。
已请用户确认H20资源状态或提供新连接；恢复后应保留并校验现有产物，再继续传输和训练。

用户随后报告节点已挂；本轮归档为节点故障导致的运行中断，故障根因未查明，
不能据此判断SFT/RL算法或迁移效果失败。已核实heartbeat `sft`为PAUSED；
等待恢复资源后复用64条示范，保留旧输出并校验传输完整性。

### 恢复运行 pilot-20261023-02-recovery-01

用户提供新H20节点mpi-1832456670-launcher，并保留原Pro5000。两池DFS及IP实测不变；
Pro5000原64条成功示范通过Demonstrations数据契约检查，共132,393帧、39,895,233,082字节。
复用原71ff4b9冻结源码（H20核验88文件SHA一致）与全部64条示范；
新输出目录采用pilot-20261023-02-recovery-01，保留原中断目录，不重复采集。
SFT micro8/GBS256/GAS32/最多500步、rollout4096、RL32环境及原预算不变。
新H20 dguard暂停2880分钟；Pro5000原暂停仍有效，用户GPU0进程630356保持运行。
计划恢复两池Ray、按原runner执行复用→SHA校验传输→SFT→RL→评测。

2026-10-24 09:59:22（Asia/Shanghai）恢复driver已后台启动，H20 MCP task `e7ede9f3-0027`。
实际执行原snapshot的run.sh，--config为`sources/pilot-20261023-02/resolved-recovery-01.yaml`；
该配置仅更改输出目录与demonstration_source（指向pilot-02完整64示范），不改算法参数。
两节点Ray共32CPU/2GPU，H20仅GPU0训练、Pro仅GPU1渲染，dguard/用户仿真边界同上。
自动审批曾拒绝强制停止Ray；只读确认两端均无raylet/GCS且端口空闲后直接启动成功，未执行强制停止。
自动检查sft已更新为恢复run并启用，等待正式两步验收；当前仅确认driver start，尚无SFT step。

10:03核查：driver PID5087运行，init checkpoint与partial reset通过；
10:02:52确认复用64/64（无新增采集），正在全量跨池传输39.9GB，已收约0.67GB。
当前copy_artifacts仅支持新目录传输，没有续接旧pilot-02的37.8GB部分副本；正式SFT尚0步。
用户先要求停止轮询，旧heartbeat已删除；随后明确授权每10分钟单次检查，已按新频率创建，
禁止触发内循环等待，健康传输静默，正式两步验收后结束检查。

### H20 teacher 副本与 dguard（用户追加要求）

恢复run的`demos/`本身位于H20 CephFS，传输完成后保留，不能当临时目录清理。
完整副本地址：
`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_multicolor_transfer/pilot-20261023-02-recovery-01/demos`。
必须等`transfer-manifest.json`生成才能视为完整副本；截至此次检查尚在传输，已约7.9GB。
新增`learner_demonstrations`配置：未来SFT重跑填上述完整目录，且`demonstration_source: null`；
校验SHA、文件集合和64条数据契约后直接调用SFT，跳过collect/copy_artifacts。
修改已同步到H20 work/carrot，CPU 22项通过；当前运行的71ff4b9冻结源码保持不变。
用户要求开启H20 dguard：两节点已核验WATCH=1；worker run.py运行，launcher因模型进程存在跳过。
现有dguard按整节点是否有GPU进程判断，不按利用率逐卡补负载；不能保证训练节点空闲卡被占用，
也不能据此前节点故障推断平台回收原因。本次未修改dguard脚本或干扰当前learner。

用户进一步明确授权临时占用H20空闲卡：launcher GPU1–7启动原dguard/run.py，
7进程，1800秒自动退出，timeout1860秒并附15秒强制退出上限；task e7ede9f3-0037。
不使用训练GPU0，不操作Pro5000；worker沿用已开启的dguard。此占用不改变正式训练配置。

### 2026-10-24 10:25 正式SFT前两步验收通过

恢复run于10:19:49完成跨池传输：450文件、39,895,247,315字节，
`demos/transfer-manifest.json`已生成，完整SHA校验通过；H20 CephFS副本保留供后续复用。
正式SFT step1（10:20:05）loss=0.13483943650498986，grad_norm=0.5854445099830627；
step2（10:20:19）loss=0.10529550747014582，grad_norm=0.5547556281089783。
两步均GBS256/micro8/GAS32，指标有限；累计samples_seen分别256和512。
本次单次检查最新step23，loss=0.025886536095640622，grad_norm=0.1950606107711792，
samples_seen=5888，driver PID5087仍在运行；未完成500步，尚不能声称收敛或迁移成功。
日志：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_multicolor_transfer/pilot-20261023-02-recovery-01/sft.jsonl`。
按用户要求结束十分钟heartbeat，训练/RL/评测/Ray保持运行；临时空闲卡占用仍按自身30分钟时限退出。

两步验收后用户更新要求：既然训练已开始，继续每30分钟单次检查；已创建对应heartbeat。
正常进展静默，阶段完成/故障/需要用户操作时通知，不做触发内循环等待。

### 用户要求从头重训，每50步保存

旧recovery-01在正式SFT step371（loss0.00774819）时按用户要求停止driver task e7ede9f3-0027，
保留全部数据、日志和init checkpoint；它没有中间SFT权重，不能当作step100模型。
新运行计划pilot-20261024-sft50-01，仍从原LIBERO step-00000100初始化；
SFT micro8/GBS256/GAS32/最多500步，每50步保存sft-step-000050/000100等目录，
包括BF16导出、FP32权重、SFT AdamW与RNG状态；当前CLI resume仍只用于RL，勿误称已支持SFT续训。
learner_demonstrations明确指向recovery-01 H20 CephFS完整demos，demonstration_source=null；
不重新采集或跨池传数据。新运行前验证完整manifest与SHA，旧输出不覆盖。

重训前验证：22项CPU契约测试PASS；H20真模型周期checkpoint、优化器状态和重载测试PASS（task e7ede9f3-0047，221s）。
H20完整teacher预检PASS（task e7ede9f3-0051，82s），64episodes/132393frames，未跨池传输。
两池冻结源码快照sources/pilot-20261024-sft50-01共88文件SHA一致。

11:51:16启动新driver task e7ede9f3-0054，源码commit ba45957，已连接既有Ray，正式SFT已验收前两步。
既有heartbeat automation已切换至此run，每30分钟单次检查；正常进展静默。

11:54核实新run正式SFT step3，driver PID21305仍运行。
前两步loss=0.1348394365/0.1052955075，grad_norm=0.5854445100/0.5547556281，均有限；
两步GBS256/micro8/GAS32。events含dataset_reuse(64episodes/132393frames)，不含dataset_transfer，
新输出没有demos目录，直接使用旧HZ持久teacher。日志为新输出sft.jsonl；每50步的实际产物由后续检查核实。


### 2026-10-24 12:10 首个周期checkpoint与后续验收

新run driver PID21305存活，最新SFT step71，loss0.0279224114、grad_norm0.0983796343。
已检查checkpoints/sft-step-000050/manifest.json：sft_updates=50；model.safetensors为7,233,650,296字节，
learner.pt为6,216,845,069字节。正式周期产物存在，尚未单独加载此正式checkpoint评测。
用户目标更新为SFT完成后继续RL并验证两者收敛；现有driver自动衔接，无需重复启动。
验收分别检查SFT全程loss/梯度及早晚窗口趋势，RL奖励/成功率趋势及KL/clip/梯度稳定性，
最后结合独立种子的源任务成功率/IoU、init/SFT/RL配对LIBERO评测。仅loss下降、完成预算或文件存在不构成收敛证据。
保留每30分钟单次检查；证据不足则标记未验证，效果不佳则继续依据结果诊断，不预先承诺收敛。


### 2026-10-24 14:03 SFT完成，切换RL环境

13:52:46 events.jsonl记录sft_updates=500；500条SFT记录loss/grad_norm全部有限，GBS256/micro8/GAS32全程正确。
step500 loss=0.0103970514、grad_norm=0.1208265051；前20/末20步平均loss=0.0597583965/0.00770341884。
每50步的10个checkpoint（50至500）及最终checkpoints/sft均已核实manifest、model.safetensors和learner.pt文件。
最终sft的learner.pt为2775877329字节（RL状态），周期checkpoint包含额外SFT AdamW；model均为7233650296字节。
13:53:47记录rollout_environments num_envs=32、partial_reset=PASS；driver PID21305仍运行。
当前未出现ppo或source_episode事件，尚不能确认首个完整RL更新，更不能判断SFT/RL任务效果或收敛。
未再次跨池传数据；数据仍复用HZ完整teacher。继续30分钟检查，下一步核对首个PPO更新及奖励/成功率。
证据：HZ experiments/carrot/pi05_multicolor_transfer/pilot-20261024-sft50-01下events.jsonl、sft.jsonl和checkpoints。


### 2026-10-24 SFT独立闭环诊断与rollout口径修正

用户要求rollout按完整trajectory计数，展开后的实际sample数sum(num_seg)/train GBS控制在1至2次更新；
当前4096chunks/256=16次更新不符合该要求。旧run保留作诊断，不能作为新口径收敛证据。
SFT诊断先在见过的单色任务seeds15、12上，分别加载step100和500，用原生10步ODE/每次执行最多5步，
完整运行到success、failure或4000控制步上限，禁止用低loss或离线MAE替代成功率。
H20独立推理服务GPU2/3，Pro独立renderer GPU2/3；不占正式RL GPU0/1，不停止Pro用户GPU0。
使用既有serve_pi05_policy与eval_sft.py，模型留在HZ，通过WebSocket交换观测和动作，不复制teacher。
评测持久化progress.jsonl（抓放/IoU/分段耗时）、episodes.json、result.json和视频。
先比较训练内任务来定位SFT/推理问题；这些结果不得作为泛化率或选择LIBERO模型的依据。


### 2026-10-24 用户纠正rollout、训练更新次数及SFT验收

用户要求rollout GBS按完整trajectory计数；segment/sample由每条变长轨迹实际展开，训练更新次数按sum(num_seg)/train GBS控制在1至2，micro仅控制显存。旧实现4096chunks/256=16不符合。不能用PPO理论支持多minibatch来代替实际配置验收。用户需要不好不坏、有改善空间的SFT起点，不是先fit到满成功；0%须先查SFT/推理。当前算法未改、旧RL仍运行，不声称满足新语义。
已读GiGPO master TrajectoryCollector的完整收集与gather_rollout_data展开、verl v0.5.0 actor.update_policy的mini/micro边界、Dressage training.md的segment归属与prompt等权、版本控制；这些参考不代表已改代码。
本地MCP重启使旧session丢失；已用本线程凭据恢复H20 session7739aeea、Pro a6ef4c1b，driver PID21305存活，资源并未因此重启。后续monitor已更新会话。
离线真SFT推理诊断task7739aeea-0073 exit0/452s：72示范观测、step50/100/250/500，数据直接HZ读取。结果在HZ tests/carrot/multicolor_transfer/sft-inference-20261024-01；roundtrip=1.19e-7，step500 ODE xyz动作MAE [.007858,.005631,.013805]，SDE [.018108,.015573,.023245]，夹爪符号均100%。native/封装ODE最大差step500=.008787，step50=.041386，未严格parity。离线不能替代闭环成功率。
独立闭环诊断已启动：H20GPU2 step100 ODE10 server7739aeea-0078（port28100），GPU3 step500 server7739aeea-0079（port28500），均timeout5400；ProGPU2 client a6ef4c1b-0326、GPU3 client a6ef4c1b-0327，均2个训练内单色seeds15、12，完整episode上限4000controls。Pro输出为实验根diagnostics/sft-100-seen-onecolor-01及sft-500-seen-onecolor-01，有progress.jsonl/episodes.json/result.json/video；服务只交换观测动作，不传teacher或模型文件。任务完成后只停止对应0078/0079，保持旧RL、Ray及用户ProGPU0。新版heartbeat仍30分钟单次检查。当前只确认服务连接和client启动，未确认首动作或结果。

### 2026-10-24 停止旧RL与SFT闭环诊断

按用户最新指令停止driver task e7ede9f3-0054，确认PID21305消失；Ray与旧权重保留。
H20两节点dguard WATCH=1；补起仅占空闲卡的7200秒巡检task7739aeea-0088：
launcher卡0/1/2/4/5/6/7，worker八卡。launcher GPU3留作推理诊断；实测其余七卡利用率100%。
未触碰Pro5000 GPU0用户进程630356。

独立原生ODE10闭环step100和500均完成seeds15/12，每条4000controls，均0成功、IoU0。
step100两条均未抓笔；step500仅seed12抓到笔，停在draw、contact_samples=0。
输出在Pro实验根diagnostics/sft-{100,500}-seen-onecolor-01，result.json、episodes.json和视频齐备。
对应推理server task7739aeea-0078/0079已停止。

下一诊断复用原teacher，不采集新数据、不启动训练：eval_sft.py增加expert/replay模式及逐步state/action记录；
在同一冻结仿真器回放seeds15/12完整teacher，另跑在线expert对照。输出分别diagnostics/teacher-replay-01和teacher-expert-01。
离线复测将ODE时间步对齐原生累计方式，并单独比较实际部署BF16与训练侧FP32/autocast路径。

teacher-replay-01完成2/2成功，IoU=.79921037/.83395028、1056/1068controls；
teacher-expert-01完成2/2成功，IoU=.80419976/.83438486、1057/1068controls。可复现有效动作和判定。
接下来通过serve_sft.py恢复step500的FP32 learner权重，统一ODE，seeds15/12分别从reset、
在线expert执行500controls后接管做闭环trace；输出diagnostics/sft-500-trace-01、sft-500-draw-prefix-01。
该诊断不更新模型、不作为新正式训练，也不传teacher。

### 2026-10-24 目标讨论与随后纠正

曾将用户关于SFT迁移的讨论误解为调整主线；用户随后明确RL仍需做，该解释已撤回。
当前优先排查SFT闭环/推理BUG及reward-advantage信号，再修正完整trajectory及每轮1～2次更新；
SFT的LIBERO迁移评测后置，不以LIBERO测试结果挑选checkpoint。
advantage监控要求保留：归一化前后符号/零占比、均值/方差/分位数与reward/return/value一起检查，
不能把标准化产生正负号当成有效奖励信号。23项CPU回归通过，未启动RL。
FP32恢复诊断server task7739aeea-0096已监听28500；client a6ef4c1b-0345/0346分别从reset及expert500步后接管，
输出sft-500-trace-01/sft-500-draw-prefix-01。沿用半小时单次检查，不主动密集轮询。

用户随后澄清：RL仍要做；当前优先排查SFT闭环失败与reward/advantage信号，SFT迁移评测后置。
旧RL继续保持停止，修复验证后按完整trajectory及每轮1～2次更新推进，不取消RL目标。

在训练集原始关键观测上，step500已出现明显动作偏差：seed15首帧teacher xyz[-.3,0,-.0993]，
FP32模型预测均值约[-.3022,.2529,-.3078]；等待帧则接近teacher。补齐维度反事实未改善误差，
不能据此认定根因。新增20步固定18观测拟合诊断task7739aeea-0105，仅验证梯度/推理权重链路，
micro8/GBS256/lr1e-5，HZ输出tests/carrot/multicolor_transfer/sft-critical-20261024-02。

reward-only对照中critic置零（不是旧RL实际advantage）：成功teacher累计reward1.9992/2.034，
非零奖励126/142帧；当前每控制步gamma=.99/lambda=.95将首动作advantage衰减至约8e-15，
约29%优势绝对值<=1e-8。失败策略前2090controls奖励全0；teacher抓笔后接管只获得一次.1抓笔奖励。
后续必须记录真实critic下的分布，不能把critic误差或标准化制造的正负号当成有效奖励。

### 2026-10-24 FP32闭环与训练/推理契约核对

两组FP32闭环已结束：sft-500-trace-01两条均4000controls、picks0、IoU0；
sft-500-draw-prefix-01在expert500步后接管，两条均IoU0，seed12在950controls失败终止，
seed15在4000截断，均有画布接触。H20 server7739aeea-0096已停止，未启动旧RL。
仅缩小了导出精度和推理封装的疑点，尚不能确认完整根因。

SFT关键帧20步拟合task0105 exit0/370s；xyz MAE从[.01719,.04863,.05305]下降到
[.01443,.00967,.03978]。原生联合attention与生产缓存路径对照task0112 exit0/191s，
速度相对L2=.00420178，三组动作参数梯度cosine=.999385/.999807/.999351；BF16容差内通过。
未发现此次测试覆盖范围内的梯度断路；这些离线结果不能替代闭环修复验收。

teacher存在隐藏等待计数导致的动作标签冲突：两条单色示范第253/254帧状态逐元素相同、prompt相同；
双图MAE约.17/.16（0～255尺度），xyz动作z却由-.00045变成-.3。代码PaintExpert先等待220次
调用再下降，计数不在策略输入中。此事实支持teacher时序设计存在可学习性问题，但尚未通过干预
证明它能解释全部闭环失败，不直接重采集/修改旧teacher。

advantage监控真GPU回归task7739aeea-0108 PASS/333s，GBS256/micro8/GAS32，
旧概率误差3.81e-6，有限梯度；原始/标准化分布、FP32恢复、周期SFT保存均通过。该回归使用
已知正负奖励fixture，不是实际RL成功或信号验收。独立真实critic离线诊断task7739aeea-0114
读取HZ的完整teacher，Pro仅传4865字节奖励元数据；检查奖励项与critic项对GAE的分别贡献。
输出HZ tests/carrot/multicolor_transfer/reward-signal-20261024-01；这是SFT时保存的未训练critic，
不是旧RL更新四轮后的critic，不能把它写成旧在线advantage。

真实初始critic分解task0114已exit0/262s：teacher共1056/1068controls，按phase/最多5步展开
213/215chunks。reward总和1.99921/2.03395；前100控制步reward-only优势最大仅约2.95e-12，
该窗口实际critic优势正15%/负85%，去掉全部reward仍逐元素相同（FP32）。当前gamma/lambda
每控制步折扣使早期抓笔动作几乎收不到任务信号；标准化正负号并不解决这个问题。
诊断不是旧RL实际优势复原。全部本轮诊断已完成，旧RL未恢复；完整trajectory与1～2次更新语义尚未实施。

2026-10-24 CPU离线credit尺度核验PASS：输出HZ tests/carrot/multicolor_transfer/credit-scale-20261024-01/result.json。
逐控制步初始reward-only优势，seeds15/12：原gamma=.99/lambda=.95为8.404e-15/8.260e-15；
仅lambda=1为.00313239/.00306365；gamma=lambda=1为1.99921048/2.03394985，等于未折扣总奖励，断言通过。
lambda逐控制步衰减是早期信号消失的主要来源之一；这只是离线反事实，不表示lambda=1已获在线验收。
没有新GPU任务、模型推理或配置变更；零奖励失败轨迹也不能靠增大lambda获得任务奖励。
