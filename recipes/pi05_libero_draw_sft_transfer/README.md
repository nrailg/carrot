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
- 静态整图更正已提交**ae6d4bd**，此前实现提交975fc31；未push。PRO新任务**6b56755c-0429**使用GPU2，输出`static-reference-20261024-01/collection`，重新验证此任务定义的16条三色示范后采64条，不复用旧动态图gate。
- 新复制PID**2540932**，H20同run目录copy-detached.log；H20队列**b434d471-0155**，同113文件冻结源码、CARROT_COMMIT=ae6d4bd，输出同run/training。等待正确任务数据原子发布后执行既定SFT，当前正式optimizer step仍为0。任务链只运行一次，没有创建自动监控。

## 数据重复检查

用户要求确认数据不重复。已检查全部128个候选训练seed：线条几何与笔架布局组合128种唯一；训练200000–200127、验证300000–300015、评测1000000–1000031互不重叠。旧版本目录不会合并进新训练集。

copy_collection.py发布前要求64条成功episode的seed、完整actions.npy及state.npy SHA分别唯一，报告写入发布目录deduplication.json；重复或缺条即失败，不发布到H20训练目录。只做episode级检查，不删除相邻相似帧。4项测试PASS，见test_deduplication.md。

已替换复制等待进程2540932为**2543905**，使用独立冻结源码`sources/deduplication-20261024-01`，日志改为同run/copy-dedup.log；旧copy-pid.json保留为copy-pid-before-dedup.json。采集0429和H20等待0155没有重启。最新仍在首批4条静态任务验证，control2201、drops全0；正式训练数据0/64，实际数据唯一性须生成后检查，不能把空目录说成去重通过。

2026-10-25更新：静态整图16/16验证全部通过，正式采集2/64，seeds200002/200000均三色成功、drops0，正式采集已运行约610秒。PRO0429、去重复制2543905及H20等待0155均存活；当前正式optimizer step仍0。用户再次授权完成后立即训练，沿用现有自动衔接链，未重启采集或新增验证要求。

## 2026-10-25 数据完成与训练启动修复

静态整图64/64全部成功，148165帧，drops总计0，平均IoU0.8058094、最低0.7854384。64个seed、64个actions SHA与64个state SHA均唯一。449文件、44,677,762,270字节已逐文件校验并发布到H20；deduplication.json passed=true。PRO采集在北京时间03:07:25完成，日志已归档collection-task.log。

H20旧任务0155在真实混合两步检查的DataLoader collate中失败：LIBERO mask为Tensor，draw mask为NumPy，首个正式训练step尚未执行；完整错误日志failed-mixed-mask-preflight.log。Pi05TransformedDataset统一mask为bool Tensor，四项类型混合回归与其余数据/任务测试合计19PASS。

新任务**b434d471-0164**使用SHA冻结113文件`sources/train-20261025-01`。复用原training目录的已验证配置和两份采样清单，不重采、不重抽样、不覆盖旧日志；启动记录training/launch-maskfix.json保留新源SHA与原配置/清单SHA。执行混合两步全参数检查后串行LIBERO200和Mix200；GPU检查与正式训练是否开始，须以后续result.json与metrics.jsonl更新。

真实混合8GPU两步检查已通过：`training/gpu-preflight/result.json` passed=true，loss为0.07393486/0.07620291，grad_norm为0.66191465/0.52057791，vision/VLM/expert代表参数均有有限非零梯度与约1e-5更新。临时检查权重丢弃。

0164随后进入正式Ray训练，但forkserver DataLoader在首个batch前序列化内存映射数据，actor RSS约83GB且未产生正式step；已定向终止该任务的8个actor及子进程，保留日志`failed-forkserver-launch.log`。改为与通过检查相同的`dataset.num_workers=0`，原配置保留，另写`libero-loader0.yaml`和`mixed-loader0.yaml`；其他训练参数、采样清单和输出路径不变，记录`training/launch-loader0.json`。recipe默认也改为0，避免重复序列化45GB示范。

北京时间2026-10-25 09:44启动任务**b434d471-0176**，使用同一SHA冻结源码train-20261025-01，顺序执行两份loader0配置。复用本轮已通过的GPU检查；不重复采集、采样或临时训练。正式step以metrics.jsonl为准。

0176正式训练已确认启动：`training/libero/metrics.jsonl`连续step1–8，step8 loss=0.0610054508、grad_norm=0.248627275、lr=1e-5，全程有限；8张H20显存各约62GB、利用率89–100%。该记录是正式optimizer step，不是临时检查。纯LIBERO200结束后同一后台任务自动执行Mix200，当前尚未完成训练或评测。

用户于2026-10-25授权30分钟后检查、训练完成后自动开始测试，并根据结果检查错漏和分析能力迁移。当前聊天heartbeat `sft` 每30分钟推进，沿用本档案配对评测协议；未恢复旧RL监控。新探索性算法方案按AGENTS.md先检索并确认，既定评测和常规故障修复无需再次批准。

## 2026-10-25 10:33 半小时检查

H20 task b434d471-0176 仍健康运行。直接核对持久化 metrics：LIBERO 已完成连续200步，末步loss0.0241624434、grad_norm0.0637181103；Mix最新连续163步，loss0.0165391248、grad_norm0.0568414815（loss_libero0.0156105664、loss_draw0.0009285602）。两路现有全部指标有限、LR1e-5；8卡98–100%利用率。LIBERO50/100/150/200和Mix50/100/150 checkpoint目录包含model.safetensors、trainer_state/config/tokenizer/stats/optimizer；本轮只查文件存在/大小，尚未验证最终checkpoint可加载，不宣称完整训练完成。后台task日志只转发到Mix93，实际metrics为163，不能按日志滞后判卡住。

尚未启动正式评测，等待当前Mix200及任务收尾；不重启训练、不重复检查、不重采。dguard启动暂停120分钟，当前运行约49分钟，尚未临近到期。评测接线仍待完成：旧recipes/pi05_multicolor_transfer/eval_sft.py未传新camera/固定逐调用noise，不能原样作为新协议正式评测；复用Pi05Policy.infer(noise=...)及已有LIBERO设施完成配对入口。下次确认训练完成后即实施并启动。heartbeat保持有效，本轮健康等待静默。

## 2026-10-25 训练完成与正式评测

0176退出0/4699秒，训练日志已归档training-task.log。两路均连续200步、所有loss/grad有限；LIBERO末步loss0.0241624434/grad0.0637181103，Mix末步loss0.0161379930/grad0.0639157817。各50/100/150/200的trainer_state步数正确、812个safetensors键可读、optimizer文件非空；证据training/completion.json。推理加载另由评测server严格加载全部权重确认，不把optimizer文件存在称为续训验证。原收尾trap错误使用dguard start（实际仅支持on），检查发现未恢复；本次评测前已显式暂停两池head360分钟，评测结束应使用on --local恢复，不能再用start。

新增配对eval_common/serve_eval/eval_libero/eval_draw与run_eval_lane，复用Pi05Policy的显式noise接口与原生采样。noise由episode身份+调用号SHA确定，不含模型名称；每条记录初态SHA/noise键/调用数、checkpoint与stats SHA，异常直接失败。LIBERO复用标准图像旋转、224输入、10步物体稳定、execute5；绘画单环境避免其他slot终止截断动作块，固定整图prompt、训练同camera、无teacher接管。两项noise契约回归PASS，ruff通过。

两池123文件SHA冻结`sources/eval-20261025-01`。H20并行LIBERO后台任务L100=0191（policy0/render1/port8101）、L200=0192（2/3/8102）、Mix200=0193（4/5/8103），每组40tasks×10初态。render1/3/5已实际EGL映射核验，每卡一个renderer。产物run/evaluation/libero/{L100,L200,Mix200}，server.log/client.log/status.json及episodes/episodes.jsonl；当前任务启动，实际episode须后续核验。

绘画H20服务base=0194（GPU6/8104）、Mix200=0195（GPU7/8105），产物evaluation/draw-servers/*.json；服务SHA记录在模型加载后写入。PRO客户端待服务就绪后启动各32条未见seed；用户GPU0进程630356未动。evaluation/launch.json保存源SHA/任务/协议，源src代码与训练快照逐文件相同。不要重启0176、重跑训练或复用旧oracle评测。

三组LIBERO已实际产出episode：L1007、L2008、Mix20011，首条init SHA一致，模型严格加载成功。PRO首次0441/0442在openpi_client import失败（未进入仿真），日志已归档；复用本机已安装包的13文件隔离副本evaluation/dependencies并保存来源/SHA，修正PYTHONPATH后base=6b56755c-0446、Mix200=6b56755c-0447，分别PRO GPU1/2。不把启动或少量成功当提升结论。

PRO0447已开始实际Isaac闭环：seed1000000、controls5，H20策略服务与新相机仿真链连通；待完整episode与32条汇总。heartbeat已更新为监控这些评测任务，不再等待已完成训练。

## 2026-10-25 11:33 评测进度检查

当前7个后台任务0191–0195、0446/0447均存活，无进程错误。LIBERO持久化唯一episode计数：L10087/400（39成功），L20093/400（62成功），Mix200105/400（70成功）；执行进度/任务构成不同，不能直接比较这些未配对累计成功率。两池head dguard仍按计划暂停、自动恢复已排程，距17:10尚远。

绘画base和Mix200各完成1/32，首条seed1000000均4000controls截断、0成功、picks0/IoU0；init/camera/noise SHA一致且prompt固定。两路正推进seed1000001至3005controls，未抓笔。单条结果不足以判定整体是否学会；保留现有完整32条协议，不重启、不提前改训练或评测。绘画速度约每条11分钟，后续需关注是否接近guard恢复时间。健康等待静默，下一轮仍只读当前任务一次，有异常再处理。

## 2026-10-25 12:03 评测进度检查

7个当前任务0191–0195、0446/0447均健康运行。LIBERO唯一episode：L100204/400（109成功），L200221/400（160成功），Mix200235/400（167成功），均进入Goal；仍不可直接比较不同任务构成的累计成功率。绘画base/Mix各4/32，已完成seeds1000000–1000003全部0成功、0抓笔、IoU0；第五条分别推进至1005/1505controls。尚无完整源任务或LIBERO结论，不重启/不改变协议。两池guard仍暂停且自动恢复计划存在，距离17:10足够；继续下轮检查，健康等待静默。

## 2026-10-25 12:33 评测进度与用户范围确认

7个评测/服务任务仍健康。LIBERO三组唯一episode为L100302/400（145成功）、L200313/400（213成功）、Mix200324/400（212成功），均已进入Long，尚未完成400条。绘画base6/32、Mix7/32，已完成样本全部0成功、0抓笔、IoU0，继续原定32条。两池guard暂停及恢复排程正常，未重启或新增作业；等待完整结果再汇报。

用户询问训练set，已核实实际YAML与meta：本地lerobot/libero整个train split，1693ep/273465帧/10FPS，共Spatial/Object/Goal/Long四套40任务，未用LIBERO90。评测同40任务闭环，每task前10初态；没有任务级holdout，也没对训练示范初态去重，因此不是未见任务零样本迁移或严格held-out测试。已明确告知用户。用户认为数据规模不平衡、可能需要refine，但最新明确要求“先测试出结果”；保持本轮配置和协议，不提前改数据/配比/训练。原主指标Spatial/Object/Goal、Long单列；对照Mix-L200控制总步数、Mix-L100控制LIBERO曝光，1:1抽样不等于任务多样性或有效监督量相当。

## 2026-10-25 13:03 评测进度检查

7个当前任务仍运行，无进程故障。LIBERO唯一episode计数L100354/400（149成功）、L200360/400（224成功）、Mix200383/400（231成功），Long尚未全部结束。绘画base/Mix各9/32，已完成seeds1000000–1000008均0成功、0抓笔、IoU0；第10条分别到2005/3005controls。遵循用户“先测试出结果”，保持完整协议，不提前重训或refine。两池dguard暂停/恢复排程正常，17:10到期；继续半小时检查，健康等待静默。

## 2026-10-25 13:33 LIBERO全量10初态结果（最新）

三组LIBERO各400条已全部完成、任务0191/0192/0193退出0，status.complete=true。任务输出已dump到各组task.log并释放，后续不要再查询这些task。逐条核验400个预期唯一键/同初态SHA/同BDDL/env_seed，重新计算全部noise流匹配记录SHA，stats SHA一致；原始episodes.jsonl保留，汇总H20 run/evaluation/libero-summary-10.json。

|模型|Spatial/100|Object/100|Goal/100|Long/100|前三套/300|全部/400|
|---|---:|---:|---:|---:|---:|---:|
|L100|44|65|36|5|145（48.33%）|150（37.50%）|
|L200|66|81|60|24|207（69.00%）|231（57.75%）|
|Mix200|67|85|51|28|203（67.67%）|231（57.75%）|

Mix-L200主指标-1.33pp，配对44胜/48负/208平，按task聚类bootstrap95%CI[-10.00,+7.67]pp；全部400条63胜/63负、差0。Mix-L100主指标+19.33pp（78胜/20负/202平），task聚类CI[12.00,27.33]pp；总分+20.25pp。CI使用20000次配对task重采样seed20261025，只反映本训练run与这些任务/初态的不确定性，不含训练种子随机性。相同总步数未观察到整体提升；相同LIBERO曝光下Mix较好，但多了100更新步和辅助数据，不能独立归因为绘画能力迁移。Goal相对L200下降9pp，Long上升4pp，总分抵消。

PRO绘画base0446/Mix0447各12/32均0成功、0抓笔、IoU0；正在第13条，仍继续原协议。H20绘画服务0194/0195继续运行，待PRO完成再停服务并恢复guard（17:10到期前留意）。用户最新说“每task抽50条即可”，已解释当前10/task、50会扩大到2000/model并询问；尚无回答，不自行扩到50，不因旧条件规则扩大评测。当前10/task的LIBERO结果应及时报告，完整绘画结果与refine分析待后续。

## 2026-10-25 14:03 绘画评测检查

仅检查仍活跃的H20服务0194/0195与PRO客户端0446/0447，四者健康。base14/32、Mix15/32完成，所有已完成episode均0成功、0抓笔、IoU0；base第15条到3005controls、Mix第16条到505controls。LIBERO已完成，不重复查询已释放任务、不扩50、不重训。两池head guard仍暂停且恢复排程正常（17:10）；按当前约11–12分钟/条预计绘画接近17:20–17:40完成，16:30左右应检查剩余时长并延长暂停，避免定时恢复冲突。健康等待静默，保持当前协议。

## 2026-10-25 14:24 用户追问抓笔失败：进度与初步诊断

北京时间14:20独立读PRO持久化episode：base/Mix200各16/32，均0成功、0抓笔、IoU0；两客户端仍运行，第17条。LIBERO既有1200条完成结果不变。MCP本地服务重启导致旧session登记丢失，使用既有凭据重连成功；新H20 session fd08c725，PRO d27e7c61，个人DFS重新检测确认同原路径。原task仍存活且可以用新session读取旧task_id（已实测0447），不要重启作业；0194/0195与0446/0447的task ID保持。两池guard仍暂停，17:10恢复，16:30应检查延期。

已取回并检查Mix200 seed1000000视频，机械臂开始抬升离开笔架，随后离开主要视野，三支笔仍在架上。视频与接触/抓取计数支持“没有完成接近抓取”，不支持归因为夹住后滑脱。本地视频memory/_multicolor_transfer_diagnostics/Mix200-seed1000000.mp4；contact-sheet包含视频帧0/4/20/80/240/799。此观察仅一条视频，不能替代全部32条。

只读核对实际采样清单：绘画25600个窗口起点，23488唯一，占148165帧的15.85%；每段还有10帧动作监督，不可将此比例当完整监督覆盖率。64条示范开头前10帧合计被选作窗口起点117次。第一抓笔phase0共14335帧、抽到2478次；全部抓笔phase0/3/6合计抽7372次，并非完全漏采抓笔。

用当前Mix200推理服务对训练示范episode-0000的真实观测做三点离线动作检查，未开新训练/仿真、不改运行源码。noise为default_rng(123+frame).standard_normal((10,32)) float32；frame0预测第一步xyz[-.17259,-.34866,-.34289]，teacher[-.3,+.3,-.09932]，y方向相反；10步xyz MAE=.27422。frame100/200对应MAE=.01041/.01809，三处夹爪符号10/10一致。该单示范/单noise诊断证明局部中途拟合可好而起始接近动作仍错误，不能以整体loss很低证明闭环掌握，也不足以锁定欠训练、采样或动作时序为唯一根因。下一步应跨示范/多个noise验证起始误差及训练-推理输入一致性，保留当前正式评测。已有model/normSHA匹配、训练推理共享transform，但尚未排除所有契约问题。


## 2026-10-25 14:33 绘画评测检查

使用新session fd08c725/d27e7c61读取原任务0194/0195/0446/0447，各仅一次，四者仍健康。PRO持久化结果base/Mix200各17/32，17个唯一seed1000000–1000016，均0成功/0抓笔/IoU0；当前第18条，base1505、Mix3505controls。未新增或重启作业。guard14:19已确认暂停且恢复计划正常，16:30再按剩余时长延期。自动化sft已改为最新session与14:33状态，去除旧训练等待指令，保留完成后诊断及清理要求。本轮无须重复通知。

用户追问L200/Mix总分相同，已重读原始400条文件及loaded.json：不同模型SHA、不同结果SHA；逐条同初态配对168同成功/106同失败/各63独有成功，净差恰好0。分套+1/+4/-9/+4抵消，不能解释为行为相同或能力等价；已告知用户。


## 2026-10-25 14:45 数据混合、任务差异与失败解释校正

用户问绘画数据量/重放比例、混合方式、train/eval差异及失败原因。已读取实际mixed-samples.npz和冻结eval生成器，审计H20 run/evaluation/data-mixture-audit.json；本地副本memory/_multicolor_transfer_diagnostics/data-mixture-audit.json及draw-task-differences.csv逐条列64训练+32评测的seed、线条颜色/三点UV坐标、笔架槽位颜色。

LIBERO1693轨迹/273465帧/40任务/10FPS，绘画64轨迹/148165帧/20Hz控制，同一三线任务族。每step256样本精确128+128，域内有放回均匀抽帧起点，每个窗口10动作，边界padding mask；按每域实际有效步数归一化后各0.5权重，并非梯度范数或原始loss值各一半。200步每域25600窗口；LIBERO24422唯一、重复起点4.6016%，draw23488唯一、重复起点8.25%。有效动作目标LIBERO248755、draw255540。draw每轨迹抽358–451窗口、平均400；不等于每条完整轨迹重放400遍。draw每帧期望曝光约为LIBERO1.846倍；真正差距是独立场景/任务多样性，相邻帧和重叠窗口高度相关，帧数不代表独立样本。逐窗口有效动作数已纳入loss权重。

实际train seed200000–200063（episode保存顺序异步），eval1000000–1000031，生成器/连续参数范围相同，参数级无完全相同任务。每条三条两段折线，颜色上下排列随机，角度±0.3rad、中点弯折±.06、尺度.75–1.1、UV平移±.035，笔架三个固定槽位的颜色排列随机。机器人初态、相机、固定prompt、桌面/笔架槽位/画布物理位置、材质光照固定。训练覆盖6种笔架排列和6种线条色序，但36组合只覆盖28；eval19组合中5组合训练未见，涉及8/32episode。因此是同生成分布的新几何与部分新组合，不是新机器人/新相机/全新任务。训练使用teacher成功轨迹，评测policy闭环，不是读取另一份动作标签。

将离线动作复现检查扩大为8条训练demo（排序每8条取1）×frames0/100/200×noise seeds123/456/789，共72次。当前Mix服务固定ODE10、原checkpoint。起始帧24次xyzMAE均值.29713，第100帧.01035、第200帧.02862；夹爪符号一致率全部1。起始第一步y符号12/24与teacher不同。证据H20 run/evaluation/train-observation-action-audit.json与同名本地副本。首次执行在JSON序列化np.int64时失败，未写结果；相同请求重跑后成功归档；这属于离线诊断，不计入正式评测，也未新增训练或仿真。

重要校正：任务允许任意颜色顺序，teacher y方向相反不自动构成错误（可能选择另一支笔）。之前“方向反了所以失败”的表述过强，已向用户明确更正。离线MAE仅证明未复现所抽示范动作，不能单独证明动作不合法或唯一根因；闭环视频与零抓取才确认失败。训练观测也存在较大初始动作差异，不能把失败仅归为eval未见seed。尚未做训练seed闭环policy评测，teacher64/64成功不等于policy在train64/64成功。欠拟合/监督分布/闭环偏离/训练推理契约等需进一步实证区分，不开始新训练。

MCP再次重启清空session登记，诊断后用原凭据重连H20新session98b2ba23并重新检测个人DFS同原路径；PRO旧session可能也需重连，未动后台作业或恢复guard。正式评测继续原任务，16:30留意guard。


## 2026-10-25 用户更正：无放回遍历后再重新洗牌

用户明确反对域内有放回随机抽样，要求先遍历完再放回。已修正src/carrot/data/mixture_schedule.py：每个来源独立生成随机排列并消费，用尽该来源所有起点才生成下一轮排列；每step128/128、有效动作归一化、L100/Mix200的LIBERO前缀配对保持。注意1:1与来源大小不同决定两域各自耗尽，不能误称两域同步完成一轮。当前预算纯LIBERO51200、混合每域25600均小于来源总数，因此新清单不会重复任何帧起点；相邻窗口的动作目标仍可能重叠，这是10步监督契约。

本次是未来清单生成逻辑的常规修正，不覆盖已完成实验原有有放回清单，不修改正在评测的冻结sources，不自动重训或重新评测。旧实验重复起点4.60%/8.25%的历史统计仍成立。Mutagen已同步开发树，本次源/测试SHA远端一致；与本修改无关的旧so101目录冲突未动。

H20 session98b2ba23 CPU验证：test_schedule7PASS（0.60s），test_data8PASS（9.21s），ruff通过。新增测试覆盖独立来源不同大小、完整轮次/尾部、多轮重新洗牌、确定性、实际数据规模无重复及曝光配对；档案tests/libero_draw_transfer/test_schedule.md。源码SHA ae7f7e5ff9eab7ff04abb0868cd7f3601b9a5ad090178f2ba339dbeeb533f7cc，测试SHA c68965c5b4cb40282560a1b85960ae64f5ff1d06265ea39f3953629087b43956。改动未提交，运行作业保持。
