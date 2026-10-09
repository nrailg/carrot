# SO101 arm dance：no vision + state jitter

## 2026-10-09：训练完成，首轮真机 K=1 验证完成

| 项目 | 设置 / 状态 |
|---|---|
| 目标 | 在新 arm dance 示教上验证无视觉、输入 state 增强的拟合能力 |
| 数据 | `nrailg/arm_dance_20261008_225311_20261008_225313`，1 episode / 144 frames / 15 FPS |
| 任务文本 | `Dance with the arm`，训练从数据集读取 |
| 初始化 | 官方 `pi05_base_pytorch_h10` 独立副本，从 step0 新训，不 resume |
| 输入增强 | 五运动轴 raw state 独立 U(-3°, 3°)，按共享 LeRobot 标定限幅；gripper/action 不改 |
| 视觉 | 所有图像 pixels=0、attention mask=false；复用 `DropVision` |
| 动作 / 随机性 | action horizon 10；每个 batch 随机 noise 和 t |
| 训练 | 2000 steps；warmup 100 后 constant LR 1e-6；seed 1000 |
| 分布式 | 8 GPU BF16 FSDP，micro batch 4 / global batch 64 / GAS 2 |
| 统计量 | 从这份 arm dance 原始数据计算；不复用 cylinder 的 norm_stats |
| 保存 | step500/1000/1500/2000 已保存并核验 |
| 输出 | `$MY_DFS/experiments/carrot/pi05_so101_arm_dance/arm_dance_20261008_225311_20261008_225313/` |
| 状态 | 训练2000步验收通过；新模型真机K1完成144/144，五轴平均跟踪MAE0.574°，现场反馈平稳；尚未精确复现示教节奏 |
| 当前判断 | 已学到部分动作趋势，但学习与轨迹复现效果一般；尚未达到精确重放 |
| 实际任务 / 环境 | 源码 bb3374b0560f141cf65a57fe21932bd62ff7ba89；Gemini session224605fe / task224605fe-0613 / Ray job17000000；Docker image tag / 上游 commit 未记录 |

Mac 原始数据：
`~/.cache/huggingface/lerobot/local/arm_dance_20261008_225311_20261008_225313`。
远端预期数据根：
`$MY_DFS/hf-hub/nrailg/arm_dance_20261008_225311_20261008_225313`。
共享标定：
`$MY_DFS/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json`。

2026-10-08 本地核验：Parquet 共 144 行，frame_index 连续 0–143、episode_index 全为 0，
state/action 均为有限 float32[6]；关节顺序与 SO101 一致，腕部视频文件存在。
原始 `data/chunk-000/file-000.parquet` SHA256：
`752fb901eab1ddb0892ceee2ed4b68afe36fabdb971a805db40a28c3424f8629`。

配置核验 PASS：与旧 no-vision+jitter2000 的 YAML 仅数据 repo/root 和 output_dir 不同；
新数据元信息与相机字段匹配，`bash -n run.sh` 通过。启动前进一步通过远端数据/源码SHA、视频解码及增强检查，见下方实际运行记录。

确认当前 MY_DFS、同步源码/完整数据/标定并核对 GPU、Ray、base 与 tokenizer 后运行：

```bash
export SOURCE_COMMIT=<已核验的本轮Mac源码commit>
bash recipes/pi05_sft_so101_arm_dance/run.sh
```

`MY_DFS` 和 `RAY_ADDRESS` 必须由运行环境明确设置。train.yaml 使用当前已有资源布局的
固定路径；若实际 MY_DFS 不同，先更新配置和 runner 的路径检查。runner 拒绝覆盖已有输出，
保存实际 train.yaml、calibration.json 和源码提交标识；暂停 dguard 120 分钟，退出恢复。
训练复用现有 fit-validation runner 与 state-jitter 数据封装，不新增 train.py 或模型改动。

验收以实际退出码、`SFT finished`、2000 条连续有限 loss/grad/LR、warmup/constant 曲线、
四个 checkpoint 的 step/horizon/权重/统计量/tokenizer/optimizer 完整性与参数更新为准。
效果评估另行执行；旧 cylinder rollout 评估写死 264 帧，不能直接用于本数据。
训练完成不等同于已拟合或真机成功。训练 runner 不操作机器人；下方真机记录来自独立试跑。
生成的图像、数据和权重不入 Git。


## 2026-10-09：实际启动与首批指标

提交 `bb3374b` 后启动本轮；本节为启动后的记录，不属于运行源码提交。用户只要求commit和启动，未push。
当前重新检测的唯一Ceph团队为`/mnt/ceph-hz1-csp/mm-base-plt2`、远端用户nrwu，
`MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu`。Mac经DevCloud上传完整6文件数据，
与81个必要源码/配置逐项SHA256核对PASS；没有重启Ray或操作机器人。

CPU-only preflight通过：144帧/任务文本正确；AV1视频首/中/末可解码；24次增强采样均在±3°
与标定范围内，gripper/action/padding/原统计量不变；实际共享transform后所有pixels0/maskfalse。
官方h10副本配置BF16/action_dim32；训练从step0新训，未resume或追加预算。

实际命令：确认环境并设置`MY_DFS`、`RAY_ADDRESS=29.209.160.111:6379`、
`SOURCE_COMMIT=bb3374b0560f141cf65a57fe21932bd62ff7ba89`，执行本recipe的`run.sh`。
后台task `224605fe-0613`，Ray job `17000000`；已进入8-rank SFT loop（micro4/global64/GAS2）。
2026-10-09 14:07:36核验rank0日志step1–38连续、loss/grad_norm/LR均有限，warmup符合公式；
step38 loss=0.158447、LR=3.861e-7、clip前grad_norm=3.443800。
之后driver输出已见step54 loss0.146179/LR5.446e-7/grad_norm1.957084；不以初始loss判断已拟合。

持久化启动证据：`$MY_DFS/experiments/carrot/pi05_so101_arm_dance/preflight_20261009/`，
含数据/源码SHA、source_snapshot、preflight.py/json、launch.json、monitor.py、ray_logs、
training_metrics.csv/latest_training_status.json。Mac轻量镜像在`~/.cache/carrot/arm_dance_20261009_launch/`。
训练checkpoint另在上表输出根的training目录，每500保存；尚未达到首个保存点。
rank0原日志：
`/tmp/ray/session_2026-10-02_18-30-48_059614_194136/logs/worker-2ba06a5f84f83754d66bbf2a937cfae4bc0dd901762a8eae8e445a33-17000000-610681.out`。
独立shell激活`/opt/venvs/carrot`、设置MY_DFS/源码PYTHONPATH/offline后可运行证据根monitor.py，
复制本job日志并检查新增step的连续性、有限性及LR；后台任务状态另用remote_task_output检查。

dguard已暂停120分钟，亲核watch0/run.py不运行且有恢复计划；run.sh EXIT trap结束时恢复。
若任务超过暂停期限需延长；本次不在训练中恢复dguard。训练完成状态、四checkpoint完整性与
最终拟合效果仍待核验，不能将“已启动”记作完成。图片/数据/权重不进Git。


### 2026-10-09 14:43:40 Asia/Shanghai：1239/2000，正常运行

本任务224605fe-0613仍running。独立读取rank0完整记录step1–1239连续、loss/grad/LR全部有限、warmup/constant正确。最新loss0.016942、clip前grad_norm0.212322、LR1e-6；最近100步平均loss0.01807016。step500和step1000的trainer_state.step分别对应，model.safetensors均812参数/BF16、config BF16/h10、norm_stats及optimizer目录存在。该检查确认保存产物的基本结构，尚未完成最终checkpoint/参数更新/拟合效果验收。日志/CSV/latest_training_status已持久化到证据根；dguard实际watch0/run.py停止/恢复计划仍在，暂停期限足够当前预计剩余约25分钟。没有新训练、评估或机器人动作。


### 2026-10-09 15:03:44 Asia/Shanghai：1907/2000，正常运行

后台task224605fe-0613仍running；亲核step1–1907连续、loss/grad/LR全部有限，warmup/constant正确。step1907 loss0.015837、LR1e-6、clip前grad_norm0.247251；末100步平均loss0.01654453。step500/1000/1500已有正式checkpoint，trainer.step对应且每组812参数/BF16。预计剩余约3–5分钟，最终保存与退出尚待确认；未评估拟合效果。monitor已更新持久化日志/CSV/status；dguard watch0/run.py停止/restore仍计划中。


### 2026-10-09：训练完成，独立验收通过

后台task224605fe-0613正常退出0，driver明确记录`SFT finished` step2000；
rank0共2000条连续有限loss/clip前grad_norm/LR，warmup100和constant1e-6符合配置。
最终loss=0.014139175415039062，日志中的grad_norm=0.200457，LR=1e-6。
逐一读取8个rank原日志，均有有限且非零的FP32训练参数更新。

四个checkpoint的trainer.step分别为500/1000/1500/2000，config为BF16/h10，
每组812个权重张量均BF16；tokenizer/train_config/norm_stats及8个非空optimizer分片和metadata齐全。
state/action统计量与本轮新数据preflight逐元素相同；action_out_proj.weight有限且相对base有更新，
最终head最大绝对变化0.001220703125（参数数值，不是角度误差）。未进行optimizer恢复测试。
最终模型在上表输出根`training/checkpoints/step-00002000`。

完成证据归档到同一preflight_20261009根：completion_audit.sh/json/log、
training_backend.log、完整ray_logs与training_metrics.csv；latest_training_status标记completed。
后台task已dump归档并释放，Ray可用GPU恢复8；dguard实际watch1、guard.sh/run.py运行且无restore计划。
轻量证据回拉Mac缓存，不下载权重、不提交图片/数据。未操作机器人。

本轮只确认训练和保存正常，尚未验证arm_dance动作MAE、连续反馈或真机效果，不能据loss宣称已拟合。


### 2026-10-09：新模型真机K=1试跑（准备）

用户明确要求驱动试跑；USB重新接入后确认follower为/dev/cu.usbmodem5C821078421，六轴Status0/标定一致、电压12.2–12.4V，当前状态接近arm_dance frame0。只操作follower，leader/相机不使用。选择本轮step2000，模型SHA84636381b984478bab138648864bb663d267b889ea008f1eea73ae279d6e37cc。当前执行源码HEAD00b084f3964b818bf585742ae9796ab2b9cb3f79加既有未提交frame_prompt改动；70必要源码Mac/GPU SHA一致，没有修改这些改动。

方案沿用之前已执行的包装和生产run_loop/SO101Sink：no vision pixels0/maskfalse、prompt Dance with the arm、h10/NFE10/eager cache/BF16（含FP32岛）、每次随机noise，K1同步读取真实state；缓慢对齐录制起点，再log10→live10→live144。等待3°/3s，超时留残差继续，无重发；绝对标定clamp+SDK相对20°，elbow/wrist P32，其他P16/I0/D32。核验有限输出/index0发送/Goal读回/Status/退出hold，并分开统计sent-vs-real和同序号示教差异，后者不是时间对齐任务成功指标。

独立产物Mac ~/.cache/carrot/so101_arm_dance2000_act1_20261009T081422Z、Ceph $MY_DFS/experiments/carrot/同名根，含plan、脚本、源码快照/SHA。实际命令client.py --mode log|live --count 10|144 --output 独立case；服务serve.sh仅本step2000，dguard暂停20分钟/EXIT恢复。未重复训练，不提交图片数据/权重，不commit/push；Docker tag和上游commit未记录。本节记录准备，真实执行结果另追加。


### 2026-10-09：首轮真机完成，现场平稳

服务224605fe-0626/PID623598加载正确arm_dance step2000，运行时690 BF16+122 FP32参数（FP32岛），eager cache/NFE10，pixels0/maskfalse。训练源码bb3374b、执行源码00b084f加既有未提交frame_prompt修改已分别留档。log10、live10、live144分别exit0；不驱动的log10只检查10个录制state，首动作五轴平均MAE0.476081°，样本小不能代表完整episode拟合。

短段10/10和主轮144/144均在3°/3s等待协议下到位，主轮0超时、2次肩抬绝对限幅、无相对限幅；所有Status0。主轮正式请求到完成46.605秒（含模型与等待延迟），每次预测10仅发送index0，再读实际state；不是一次性播放原始动作。退出写当前位置hold/Goal读回通过，扭矩保持、串口关闭。用户现场反馈：“平稳，没有明显异常”。

主轮每动作等待结束的位置与已发送目标比较（五运动轴degrees）：

| 轴 | MAE | P95 | max |
|---|---:|---:|---:|
| shoulder_pan | 0.073191 | 0.284115 | 0.469488 |
| shoulder_lift | 0.903944 | 1.933482 | 2.392105 |
| elbow_flex | 1.245599 | 2.458231 | 2.801712 |
| wrist_flex | 0.400897 | 0.810847 | 1.117947 |
| wrist_roll | 0.246180 | 0.391499 | 0.444304 |

五轴平均MAE0.573962°；未限幅的raw模型目标对同一实测位置为0.575225°。gripper原单位另列：MAE0.422262/P950.429215/max0.430578，不能与角度混合平均。最终实测[-3.780220,-75.208791,37.846154,62.417582,-5.230769,0.754976]。

亲读164个NPZ（log10/live10/live144）、原Parquet和事件：state/prompt来源、frame推进、h10有限输出、首动作发送、Goal读回、Status和hold核验通过。真实state模式在预热前读一次frame0、预热后重新读正式frame0；审计已排除预热观测。主轮raw目标对同动作序号示教的MAE=[0.088328,7.094868,11.135056,0.583805,0.126677]°，五轴均值3.805747°。图中肩抬/肘部推进更早、末态大致接近；同步执行时长与原15FPS不同，这组差异是序号参考，不能当作时间对齐跟踪误差。无视觉/state-only且单轮随机noise，也不能证明任务泛化或旧cylinder末段腕屈抖动问题已解决。

产物根同上，新增analyze.py/independent_results.json、逐动作tracking.csv、tracking PNG/SVG/PDF、completion.json/hardware_after.json和server_evidence。实际调用生产run_loop/SO101Sink，未修改生产代码；对齐包装源自前轮并只更换数据/prompt/144帧/服务metadata。现场无再次驱动或温度诊断。

核对归属后只SIGTERM本服务PID623598，task exit0，backend日志dump归档释放；8080关闭/进程消失、源码和模型SHA不变。dguard亲核watch1/guard.sh与run.py运行/无restore计划。Mac全部轻量结果归档Ceph同名根mac_results，图片/数据不进Git，未commit/push。


### 2026-10-09：144动作逐关节与原数据对照

按同一动作序号0–143，将原Parquet action作为参考；实机取每次等待结束的最后实测位置。模型列使用本轮真实state输入产生的raw index0，非录制state输入的离线拟合测试。无新增推理或机器人动作。

| 关节 | 模型对示教MAE | 实机对示教MAE | 实机对示教P95 | 实机对示教max | 实机对发送目标MAE |
|---|---:|---:|---:|---:|---:|
| shoulder_pan | 0.088328 | 0.096459 | 0.351649 | 0.967033 | 0.073191 |
| shoulder_lift | 7.094868 | 6.730769 | 21.657147 | 26.153847 | 0.903944 |
| elbow_flex | 11.135056 | 10.463370 | 23.973624 | 25.582413 | 1.245599 |
| wrist_flex | 0.583805 | 0.539072 | 1.098904 | 1.274727 | 0.400897 |
| wrist_roll | 0.126677 | 0.215507 | 0.439560 | 0.439560 | 0.246180 |
| gripper（原单位） | 0.016767 | 0.408885 | 0.429243 | 0.429243 | 0.422262 |

五运动轴均值：模型对示教3.805747°，实机对示教3.609035°，实机对发送目标0.573962°。gripper原单位另列。主要同序号差异在肩抬/肘部，模型目标本身已有明显变化；机器人跟踪误差较小。曲线显示在较小动作序号进入后段姿态，包含路径推进节奏差异：闭环正式过程46.605秒，源记录15FPS，不是时间对齐GT，也不能将该指标当作同一录制state输入的拟合误差。未使用DTW或其他重新对齐掩盖差异。

新增end_wait_vs_same_index_demo_action和live_input_vs_same_index_demo_state至independent_results.json，原始事件+Parquet独立复算MAE通过；analyze.py、analysis_stdout.json与CSV仍在同一产物根。补充结果同步Ceph mac_results，不commit/push。


2026-10-09补充逐frame误差图：同一144帧原Parquet action，与真实state闭环产生的raw模型index0和每次等待结束实测位置分别相减。plot_frame_errors.py直接读NPZ/事件/Parquet并核对原Parquet SHA，输出frame_errors.png/svg/pdf与144行CSV；每个关节独立子图，保留正负，无平均/平滑/DTW。横轴0–143为动作序号，不是物理时间；五轴degrees、gripper原单位。Mac与Ceph原产物根mac_results归档，图不入Git。无新推理或机器人动作。


2026-10-09起点/早期偏移复核：live144已缓慢发送原frame0 state并读取precise_start_ready；五轴实测减源state=[-0.615385,0.087909,-0.087915,0.615384,0.087912]°，首次正式request为同一实测值，prompt正确。不是20°级起点错位，但不能排除小起点差异影响闭环。前8个动作肘部保持约95°；frame8输入95.4286°，raw目标87.8506°，实际89.3626°，同序号示教action93.7582°；frame9输入89.2747°、raw目标78.1294°、实际79.4286°，示教action仍93.7582°。此后真实state反馈沿更靠后的姿态继续推进，frame20目标67.5986°/实测68.2637°/示教action92.0879°。这定位早期偏移发生在模型目标及其闭环反馈，不能凭这些证据区分随机noise、局部state敏感性与state-only阶段歧义。

采样/执行动力学也是待验证因素：源15FPS，state是录制时实测、action为目标，二者并不一致（frame30肘state90.7692°、action83.8242°）；本轮wait3°后再读取state，约0.32s/动作，下一输入不能假定等于同序号录制state。“序号提前”也不等于物理时间更快，完整本轮46.6s。未执行固定noise/录制state完整离线对照，也未重新驱动。


2026-10-09按用户指出的gripper/shoulder_pan第0帧差异复核比较口径。录制frame0 gripper state0.754976/action0.407166，正式实测输入0.754976、raw模型0.406060、执行后仍0.754976；所以原actual-minus-action图的夹爪0.348并非模型目标同量误差，而且数据原始state-action已有相同间距。shoulder_pan录制state-3.076923、实测输入-3.692308（起点残差-0.615385°）、示教action-4.043956、raw模型-4.001627（输出差+0.042329°）、执行后-3.692308（对raw残差+0.309320°）。起点确实已设置/缓慢对齐但非逐轴完全重合；代码小于0.8°即停止起点校正、正式assert不超过1°。运行wait3°/gripper3原单位允许上述执行残差，不证明物理上无法进一步到位。

新增plot_state_action_frame_errors.py与state_action_frame_errors PNG/SVG/PDF/CSV：蓝线改为每次正式request的实测输入state减录制同序号observation.state，橙线仍模型raw首action减录制同序号action。共144点，无平均/平滑/DTW，不混淆行动前state与行动后目标；蓝线gripper接近0，shoulder_pan初始-0.615°，肩抬/肘部中段偏移仍大。两个指标参考不同，已明确标注；旧actual-vs-action图保留为执行结果对目标参考。新增文件归档原Mac/Ceph mac_results，不新驱动或推理。

2026-10-09按用户要求增加state_action_frame_errors_180deg PNG/SVG/PDF：五角度轴统一Y范围[-180,180]°，gripper保留原单位/原刻度，数据与state/action对照口径不变。脚本/CSV/图片保存原产物根并同步Ceph mac_results，不入Git。

2026-10-09增加three_trajectories PNG/SVG/PDF/CSV和plot_three_trajectories.py：按用户要求每关节3条原值线，dataset=原Parquet observation.state（与前述frame66取值一致）、model=真实state输入产生的raw action[0]、robot=同次正式request的输入实测state，144帧逐点验证NPZ/request一致。各轴按实际数据范围自动显示，不用±180°、不作平均/平滑。夹爪dataset/robot重合、model为接近源action的目标，三者阶段不同已注明；同序号对照不是物理时间对齐。原产物根归档，图不入Git，无新驱动或推理。


## 2026-10-09：本轮阶段结论——学进去了一部分，效果一般

用户看过逐frame原始角度三线图后的判断：“感觉学得非常一般。好歹是学进去了。”
本轮保守记录为：模型学到了部分动作趋势，新模型能驱动真机平稳完成有限长度闭环；
学习与录制轨迹复现质量仍一般，未达到精确重放。现场平稳与轨迹精确复现分别记录，
不将正常执行、低跟踪误差或训练正常退出写成充分拟合成功。

证据来自本轮arm_dance单episode144帧、no vision+五轴±3°jitter/clamp、官方base新训2000steps；
step2000以h10/NFE10/K1/真实state反馈/随机noise执行，wait容差3°/3s（夹爪原单位），
144/144正常完成、0超时、Status0，用户现场反馈“平稳，没有明显异常”。
机器人等待结束相对发送目标的五轴平均MAE0.574°，反映执行跟踪，不能代表相对示教的拟合误差。

以three_trajectories为主要直观证据：dataset=录制observation.state，
model=同次推理raw action[0]，robot state=推理前实测输入state；横轴为同动作/推理序号0–143，
各轴按实际范围显示，无平均/平滑/DTW，非物理时间对齐。shoulder_lift/elbow_flex的中段
推进与原录制明显不同，其他轴总体变化较小。frame66实测state相对录制state分别
+27.341°/-30.066°；±180°显示会压缩视觉差异，不据此改变原始数值判断。

起点设置过，最大五轴实测残差约0.615°，并非完全重合。夹爪的录制state与action本来不同，
原actual-minus-action图包含该差距；state-vs-state图中夹爪基本一致。
目前只有10个录制state的离线首动作检查，完整144帧同输入离线拟合评估未执行；
早期模型跳变、闭环反馈、随机noise、局部状态敏感性、阶段歧义与执行节奏的贡献尚未隔离。
不将本轮效果一般归因于某一个已确定BUG，也不将新arm_dance表现当作旧cylinder腕屈抖动已解决。

逐帧三线图/脚本/CSV以及先前原始事件、NPZ和独立复算结果位于
Mac ~/.cache/carrot/so101_arm_dance2000_act1_20261009T081422Z，
并已归档Ceph $MY_DFS/experiments/carrot/同名根/mac_results。
服务0626已退出0并归档释放，dguard已恢复。此次仅补充实验结论和长期记忆，
不新训、不再驱动机器人、不commit/push；图片、数据和权重不入Git。
