# LIBERO + 多色 draw 全参数 SFT

状态：2026-10-24相机/latency及SFT实现已提交975fc31。用户随后明确要求参考屏全程显示静态完整三线图，由机器人完成整图规划。动态单线参考与子任务提示的数据链已停止，保留20条相机修正版数据；正在验证静态整图版本。正式SFT尚未执行optimizer step，下方任务链为历史记录。

## 实验协议

两条训练线各跑一次，使用相同pi05_base。纯LIBERO训练200步，在100/200步分别保存L100/L200；Mix200每个global batch精确128个LIBERO与128个draw样本。

| checkpoint | 更新步数 | LIBERO样本次数 | draw样本次数 |
| --- | ---: | ---: | ---: |
| L100 | 100 | 25600 | 0 |
| L200 | 200 | 51200 | 0 |
| Mix200 | 200 | 25600 | 25600 |

VLM、视觉编码器、动作专家全部训练，使用正式trainer/sft，不使用RL的冻结prefix路径。GBS256、micro4、单节点8GPU/FSDP2、GAS8、恒定LR1e-5；AdamW等配置见train.yaml。每50步保存；固定比较100/200/200，不按目标评测挑checkpoint。

源数据为真实三色抓笔—画线—归还。参考板全程显示完整目标图，actor只收到固定的整图临摹指令，不接收teacher阶段、当前颜色或下一步提示。示范teacher保留内部步骤，但奖励/成功按颜色独立统计，允许任意绘画顺序；动作执行与训练的10步horizon都可跨teacher阶段，仅在episode末尾截断。源/目标共用LIBERO动作变换与norm_stats。此任务训练端到端行为策略，不额外加入外部planner；是否学会整图规划须由闭环评测验证。

精确采样manifest在启动前冻结，训练设置shuffle=false。两路LIBERO样本流来自相同独立RNG，Mix200中的25600次LIBERO抽取等于L100曝光。每域loss按整个global batch有效时间步归一化，混合目标为0.5*L_LIBERO+0.5*L_DRAW；保存每域加权loss贡献与有效监督量。

## 执行顺序

1. 当前会话重新检测DFS，检查GPU/dguard/已有进程。源码在Mac开发并同步，运行用独立冻结副本；不覆盖旧demos/checkpoints，不停止用户其他任务。
2. Pro5000运行collect.py，使用camera.json的右侧高机位。静态整图版本验证固定16条三色任务后，从seed200000起收集64条三色成功示范、最多128次尝试。--reuse-physics-gate仅允许同一static-reference-v1任务契约、相同scene/teacher SHA及16条完整成功记录，旧动态单线gate不能复用。失败则保留证据并停止，不降成功阈值。最长采集窗口默认8h，不代表收敛承诺。
3. 将完整demos、manifest.json、teacher-gate.json复制到H20新持久化目录，校验SHA；两个池不共享CephFS，不重复传输旧约40GB数据。
4. H20准备8张空闲GPU与专用Ray资源，设置下面环境变量，运行run.sh。prepare.py核验teacher证据与数据，生成两份采样清单、训练配置和provenance。
5. run.sh首先执行8GPU两步梯度预检，再串行训练libero和mixed。预检临时权重丢弃；每条正式路线只训练一次。出现异常停止，不能跳过预检。需以metrics.jsonl实际正数step确认启动，以退出状态和完整checkpoint确认完成。

```bash
# Pro5000；MY_DFS/GPU已经在当前会话确认。
source /opt/venvs/carrot/bin/activate
cd "$CARROT_SOURCE_DIR"
export PYTHONPATH="$PWD/src:$PWD/tests:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="$SIMULATOR_GPU" python recipes/pi05_libero_draw_sft_transfer/collect.py \
  --output "$COLLECTION_OUTPUT" \
  --camera-config recipes/pi05_libero_draw_sft_transfer/camera.json \
  --robot-usd "$MY_DFS/isaacsim_assets/Assets/Isaac/6.0/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"

# H20；CARROT_SOURCE_DIR指向已校验冻结源码；CARROT_COMMIT记录其基线提交。
# RUN_DIR是新实验目录；DRAW_ROOT指向新数据/demos；TEACHER_GATE指向teacher-gate.json。
# RAY_ADDRESS必须指向已核验的8GPU训练资源，不能挤占其他实验。
bash recipes/pi05_libero_draw_sft_transfer/run.sh
```

## 评测协议

启动阶段先完成数据与全参数训练验证。三组LIBERO正式配对评测后续接入：40tasks×10初态×3组=1200episodes，主指标Spatial/Object/Goal宏平均，Long单列。统一原生ODE10、execute5、相同初态SHA和逐调用noise seed。Mix200与base另测32条三色未见seed（1000000–1000031），成功率、IoU和抓放阶段分别记录。

源任务至少8/32成功且高于base后，可将同checkpoint的LIBERO评测扩到50初态，共6000episodes；不重复训练。不满足时先报告源任务未确认学会，不据此否定能力迁移。三组仅验证辅助SFT，不能称未见LIBERO任务的零样本迁移，也不能独立归因于精细绘画能力。

## 验证记录

2026-10-24本地：test_schedule.py 4项PASS（含变长loss独立oracle），无GPU依赖。

H20远端：test_schedule.py与test_data.py共7项PASS，任务9f2afe6d-0122退出0。实际LIBERO数据检查任务9f2afe6d-0120退出0：273465帧、1693个episode，核对首帧及episode末帧的10步action mask符合采样清单算法。

2026-10-24 19:58北京时间，PRO GPU1启动16条事件驱动teacher预检，通过后自动采集64条三色示范；任务f7822684-0364。PRO个人DFS为`/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu`；输出在其`experiments/carrot/pi05_libero_draw_sft_transfer/sft-mix-20261024-01/collection`，仿真日志在`renderer/simulator.log`。GPU0已有用户任务，未停止。

H20任务9f2afe6d-0125运行8卡两步梯度检查，先用真实LIBERO验证VLM/视觉编码器/action expert可更新；此检查临时权重丢弃。输出在H20个人DFS的`experiments/carrot/pi05_libero_draw_sft_transfer/sft-mix-20261024-01/h20-preflight`。正式混合数据到位后仍需验证其完整路径。

两池冻结相同101文件源码到各自个人DFS的`experiments/carrot/pi05_libero_draw_sft_transfer/sources/sft-mix-20261024-01`，逐文件SHA见source-manifest.json；运行中的快照不热改。

分支codex/libero-draw-sft-transfer，基线6be3d5a；修改尚未提交。实际Docker image与上游commit未记录。不得把进程启动写成teacher验证或正式训练成功。

## 旧任务链（已停止，由相机修正版替代）

- PRO首次预检0364在511秒时由Agent中断以补充阶段进度日志；旧输出与视频保留，没有形成teacher通过结果。当前任务**f7822684-0371**使用冻结源码`sources/sft-mix-20261024-02`，输出`sft-mix-20261024-02/collection`；每100controls记录teacher-progress.jsonl。完成全部16条门槛后才自动采64条示范。
- PRO任务**f7822684-0373**等待当前采集进程1339058退出，原子写task-status.json；缺少完整manifest/gate则标记失败。
- DevCloud一次性复制进程PID**2479911**已脱离SSH会话，等待采集状态成功后复制新数据并逐文件验证SHA，再原子发布H20的`sft-mix-20261024-02/collection`。日志在H20同一run目录的copy-detached.log，进程记录copy-pid.json。只复制新数据，不复制旧64条。此前SSH前台复制等待已主动结束，无数据复制；其失败标记已在确认无产物后清除。
- H20训练队列任务**9f2afe6d-0140**使用冻结源码`sources/sft-mix-20261024-04`（105文件SHA清单），等待上述原子目录。数据校验→8GPU实际混合梯度检查→纯LIBERO200→Mix200串行执行；输出`sft-mix-20261024-02/training`。教师/采集/复制失败即停止，不用旧数据替代。此前0135/0139仅等待数据时被替换，均未训练。
- 已确认旧Ray全部job结束、actor DEAD，再停止旧H20/PRO Ray，H20重建单节点48CPU/8GPU，地址29.209.162.214:6379。PRO物理进程与GPU0用户任务独立，不受Ray更换影响。H20等待期间保留dguard；启动训练前暂停本地120分钟，并检查无其他GPU进程，结束恢复。
- 当前未配置W&B凭据，已关闭本recipe的W&B上传；训练指标写metrics.jsonl，不改变任何训练超参数。源码04与02的teacher和模型训练代码一致，仅新增队列/复制入口、档案及关闭W&B。
- 当前检查证据：H20两步全参数PASS见tests/libero_draw_transfer/check_model.md；PRO首批4/4三色任务成功，均picks=returns=3、drops=0，controls2221–2345、IoU0.7970–0.8104；已进入第二批。证据collection/teacher-episodes.json。这是4/16预检进展，不是完整16条门槛通过。

上述任务链已停止，不得恢复它来覆盖相机修正版。配对LIBERO评测入口仍待接入。

## 相机与latency修正

- H20原数据同级`demos_backup`已完成`cp -a --reflink=auto`，451个文件、39,895,314,594字节，源与备份逐文件SHA一致。证据为原run目录的`demos_backup-verification.json`和`demos_backup-task.log`。原数据未覆盖。
- 旧机位的事件teacher采集已保留24条；旧PRO采集PID1339058、DevCloud复制等待PID2479911、H20训练等待PID100410均停止。16条物理门槛已全部成功，证据`.../sft-mix-20261024-02/collection/teacher-gate.json`。
- 新camera.json使用位置[0.90,-0.40,1.00]、朝向[0.47,0.10,0.32]、focal_length=0.025；主/腕相机仍为224×224，不增加真值图输入。正前方机位静态清楚，但运动时遮挡参考板下部，因此选择右侧高机位。
- `camera_scout.py`在相同物理状态渲染六个角度，每100controls及phase切换保存截图；切换相机后渲染8帧冲刷异步渲染器。初版仅2帧会出现上一机位图像，初版图片不作为最终角度证据。
- 重新生成的观测在action之前保存；teacher_mode=events按到位、手指开合和双指接触推进，不执行legacy的固定220ticks。必要的物理抓放过程保留，不剪切轨迹中间帧。
- 完整运动检查`camera-motion-20261024-03`：seed300000成功，2345controls、IoU0.79881、picks=returns=3、drops=0；六个视角在同一物理状态取样，右侧高机位三色绘画截图见`old-new-drawing.png`。按旧回放相同阈值检查，新轨迹开头没有旧约11秒空等；归还阶段仍有8.7–8.8秒低平移段，手指持续从约7mm张开到35mm，不应宣称所有停顿已消失。证据`latency-summary.json`。
- 两池相同110文件冻结到`sources/camera-20261024-04`，逐文件SHA核验。PRO GPU2采集任务**6b56755c-0413**，输出`camera-fixed-20261024-01/collection`；复用16条physics gate，另保存新机位运动验证路径。完整manifest生成后才写task-status.json complete=true。
- DevCloud复制进程PID**2519882**，日志H20 `camera-fixed-20261024-01/copy-detached.log`；按新manifest复制并校验后原子发布collection，包含camera.json。失败写collection.failed。
- H20队列任务**b434d471-0151**：使用相同冻结源码和新collection，等待复制后自动执行原定全参数SFT两条路线，输出`camera-fixed-20261024-01/training`。当前无正式optimizer step，禁止把队列说成训练完成或开跑。启动前Ray确认单节点48CPU/8GPU、资源使用0；等待期间dguard保留。
- 最新新数据进度4/64，seeds200002/200000/200001/200003全部三色成功、picks=returns=3、drops=0、IoU0.7984–0.8087。首条实际训练数据已导出`camera-fixed-20261024-01/first-demo.mp4`：2239帧、20fps、111.95秒，首帧/绘画帧确认采用新机位，数组帧数一致；本地副本SHA与远端相同，ffprobe帧数/帧率通过。preview任务6b56755c-0421退出0并已归档日志。

## 静态整图更正（覆盖上述相机修正版任务链）

- 975fc31保存更正前代码。用户要求参考图静态显示三条线，由机器人完成整图规划；此前逐条显示参考线并更换pick/draw/place提示不符合该定义。
- 已停止PRO采集0413（SIGINT，保留20条成功数据，task-status complete=false）、DevCloud复制PID2519882和H20等待0151，旧版本不再自动进入训练。
- 新版本在初始化/reset时显示完整目标图，整个episode不更新参考屏；prompt固定。teacher schedule只用于示范生成和诊断，不能控制actor输入或动作chunk的执行边界。
- PaintProgress按颜色统计抓取、画完和归还，保留物理接触、分色IoU0.72、笔归位、夹爪打开等要求；六种颜色顺序都允许。dataset拒绝动态prompt，动作监督跨teacher阶段；gate记录static-reference-v1，prepare拒绝旧任务定义。
- H20 CPU验证15项PASS，任务b434d471-0153，见tests/libero_draw_transfer/test_static_task.md。PRO完整三色画面验证0428退出0/309秒：2345controls、IoU0.7988097、抓取/归还各3次、掉笔0，prompt每步恒定，参考屏始终完整三线图。新113文件冻结源码`sources/static-reference-20261024-01`；这是teacher验证，不是模型自主规划效果。
