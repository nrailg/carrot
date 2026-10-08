# SO101 原始示教动作真机 replay（2026-10-08）

用户休假归来，明确要求先检查机器人，再开始走SFT数据。此轮执行step1：Mac直接读取单episode原始Parquet action，绕过policy/GPU/视觉，逐步5帧→30帧→264帧；没有修改标定、PID、加速度或生产runner，没有模型训练/推理，也没有leader/camera访问。

## 数据与环境

- 单episode：`/Users/wujunyu/.cache/huggingface/lerobot/local/knock_down_the_cylinder_1_20260930_222251`，264帧/15FPS，时间戳0..17.5333秒，完整发送窗口约17.6秒。
- 多episode：`/Users/wujunyu/.cache/huggingface/lerobot/local/knock_down_the_cylinder_20260930_222351`，30episode/3213帧/15FPS，本轮未执行。
- follower白臂端口`/dev/cu.usbmodem5C821078421`、id `my_awesome_follower_arm`、标定目录`~/.cache/huggingface/lerobot/calibration/robots/so_follower`，use_degrees=True，无相机。1肩旋/2肩抬/3肘/4腕屈/5腕转为degrees，6夹爪保留源数据单位。
- 使用本机`examples/so101_real/.venv/bin/python`，packages={"lerobot": "0.6.1", "numpy": "2.2.6", "pyarrow": "25.0.1", "torch": "2.11.0", "matplotlib": "3.11.2"}。Carrot本轮HEAD `7650c4acf7cbff76b887a7f2cfc096645043f83a`，本机运行无Docker；上游Git commit未记录，以安装包版本与实际源码为据。
- 原Parquet SHA256：`cd69202c4c1a09436dd4278a7e8b68584b6e998885716f7f49f817fc884baf6b`。

## 启动与执行协议

先只调用bus.connect/read校验六舵机与标定，不调用会配置/启扭矩的Robot.connect。初次读数六轴Status0、Torque0、OperatingMode0、Phase12；电压寄存器122..124（约12.2–12.4V）、温度41–42。旧Goal寄存器全0，绝不直接启扭矩追旧目标。

启扭矩前将六轴raw Goal预置当前raw Present并读回确认。每轮将当前姿态分60个插值目标、每50ms逐步对齐录制首state，再等2秒。独立对齐阶段与录制trajectory分开记录；回放开始要求五轴首state误差≤5°、预对齐位移初轮≤10°。完成30帧后姿态离起点约23°，完整轮允许慢速对齐范围≤30°，真正开始回放的5°容差没有放宽；完整轮实际首state五轴最大误差仅0.7033°。首5/30轮肘初始仍+4.571°，不能假定各轮精确同起点。

原action与当前标定范围比较：第2轴最小原值−105.14286°，标定下界−104.83516°；出现最大0.307696°绝对裁剪，原target及bounded均保存，未改源数据/标定。其余轴不裁剪；回放不用旧反馈中心5°相对限幅，发送前任一运动轴target与反馈差>20°则中止，不隐式裁剪。新轨迹不是逐行等待到位，而按绝对deadline15FPS发送，记录发送前和约33ms后反馈及每帧Status。最后目标保持约1秒后采集settled反馈，退出时写当前raw位置保持并读回，保留扭矩，再关闭通信。

本机诊断脚本与全部产物位于`/Users/wujunyu/.cache/carrot/so101_action_replay_20261008/`；不是生产runner新增模式。脚本先用真实Parquet+模拟bus冒烟验证预置Goal先于启扭矩、5行执行与退出hold，无串口访问，随后才实际执行。首5/30脚本退出后保存对应版本快照；完整轮有运行前快照。真实命令示例：

```bash
examples/so101_real/.venv/bin/python \
  /Users/wujunyu/.cache/carrot/so101_action_replay_20261008/replay.py \
  --dataset /Users/wujunyu/.cache/huggingface/lerobot/local/knock_down_the_cylinder_1_20260930_222251 \
  --output /Users/wujunyu/.cache/carrot/so101_action_replay_20261008/full264 \
  --count 264 --align
```

输出目录必须不存在；以上目录已有结果，不可覆盖重跑。各轮开始/结束估算时间由event单调时钟映射当前墙钟保存summary.json（明确为估算），原始单调时间留events.jsonl。不是远端Gemini任务，stdout会话first5=95089、first30=35182、full264=59020，全部REPLAY_COMPLETE且exit0。

## 实际结果

| 回放 | 已发送 | 实际FPS | 肘最后保持后相对Goal误差 | 故障/退出 |
|---|---:|---:|---:|---|
| 前5帧 | 5/5 | 14.7465 | +3.341° | 六轴Status0、exit0、hold读回通过 |
| 前30帧 | 30/30 | 14.9658 | +4.132° | 六轴Status0、exit0、hold读回通过 |
| 完整episode | 264/264 | 14.9956 | −0.176° | 六轴Status0、exit0、hold读回通过 |

完整发送间隔median=66.683ms，min=61.454ms，max=72.138ms；无模型推理等待，不能把本次15Hz推广到policy实时闭环。逐条target与原Parquet逐元素相同、frame连续0..263、264个Goal读回与bounded差<0.1源单位；全部Status与complete/hold事件独立核对通过。

更适合判断“是否重放原轨迹”的是实测关节位置对录制observation.state：以下直接用每帧发送前反馈与同frame录制state比较，不插值、不优化对齐延迟。

| 五轴 | 同帧state MAE° | P95° | max° | 约33ms后反馈对当前Goal MAE° |
|---|---:|---:|---:|---:|
| shoulder_pan | 0.2581 | 0.6154 | 1.1429 | 1.0559 |
| shoulder_lift | 0.3347 | 1.1429 | 1.8462 | 1.8808 |
| elbow_flex | 0.1895 | 0.5275 | 0.9670 | 2.0423 |
| wrist_flex | 0.2784 | 0.8659 | 2.1099 | 1.6510 |
| wrist_roll | 0.1968 | 0.4396 | 0.5275 | 0.3173 |

最后一列含舵机动态延迟与目标偏差，不能当成模型MAE。原录制state对示教action本身也有偏差，肘若干负向目标平台的约3–4°欠跟踪在本次与原录制中都可见；不能全部算成本轮新增的replay错误。最后1秒settled五轴Goal误差[+0.176,+0.440,−0.176,−0.176,+0.352]°；夹爪实测0.754976始终不动，源单位目标误差约0.393平均，夹爪不计入角度统计。

当前结论：在这一次完整episode、当前配置下，通信/标定/状态正常，整段原action命令及15FPS时序完成，实测关节轨迹较好复现录制state（五轴MAE约0.19–0.33°）。不能宣称舵机完全精确跟踪所有Goal、所有方向或负载健康，也不能据此认定圆柱任务成功；本次没有相机记录；用户现场反馈“我看着还行”，随后请求原始MP4回忆和对照，尚未确认圆柱任务成功。本次未执行模型预测、实时视觉闭环或泛化。

产物：每轮events.jsonl/results.json/commands.csv，summary.json/独立分析analyze.py，完整full264_tracking.png/svg/pdf。图中灰虚线原action、橙色Goal读回、蓝色本次实测state、绿色原录制state，时间轴为从回放开始的秒数。图片只在Mac cache，不入Git。初步执行后get_joint再次读六轴Status全部0。

下一步为数据集观测→正式policy→日志，先复现已验证checkpoint拟合，再做dataset-policy真机执行；不是本轮自动启动服务或控制下一段的授权。

原始腕部视频：`/Users/wujunyu/.cache/huggingface/lerobot/local/knock_down_the_cylinder_1_20260930_222251/videos/observation.images.wrist/chunk-000/file-000.mp4`，已应用户请求用Mac默认播放器打开；这是原录制视频，不是本次回放视频。

原视频AV1编码，已另生成并打开播放器兼容的H.264副本`/Users/wujunyu/.cache/carrot/so101_action_replay_20261008/original_episode_h264.mp4`；核验264帧/17.6秒，原视频不变，转码副本非逐像素无损。


## 2026-10-08：原始action horizon10/execute5分块回放（准备）

用户明确先验证原始数据分块，不启动模型服务。每次取10条原action、只执行前5条、stride5，统一15FPS绝对deadline，不人为加入块间等待。264帧对应53块，最后一块只有4条有效action，padding只补窗口不执行。先10帧/2块验证边界，再完整264帧；验收发送frame连续0..263、与原Parquet逐元素相同、无padding执行/无额外等待，比较块内/块间时序及实测关节与原state/上一轮回放。沿用bus-only标定/Status检查、启扭矩前raw Goal预置、慢速首state对齐、原绝对裁剪/20°差断言及退出当前位置保持；不修改PID/标定/生产runner，无camera/leader访问。

本轮脚本/plan/结果：`/Users/wujunyu/.cache/carrot/so101_action_chunk_replay_20261008T141608/`，独立输出first10与full264，不覆盖原结果。执行命令为本机venv python该目录replay.py，传`--dataset <同一单episode目录> --output <新目录>/first10或full264 --count 10或264 --align`。执行前真实Parquet纯序列校验10/264通过，尚未实际分块运动。此前checkpoint核对未启动服务/推理，不能混作本轮动作来源。


### 2026-10-08：原始action分块实测完成

本机先10帧/2块task27327 exit0，再264帧/53块task23304 exit0，均CHUNK_REPLAY_COMPLETE。独立analyze.py task73465 exit0，逐个读取55个chunk NPZ、原Parquet、command/feedback/Status及原连续回放结果，核验窗口h10/consume5/stride5；末块consume4、padding未执行，frame连续0..263、target与源action逐元素相同，新旧264条Goal寄存器也逐元素相同。未调用policy/模型/相机，也没有加入模拟推理延迟。

完整轮实际14.99899FPS；块内211个间隔median66.7659ms，跨块52个间隔median66.0586ms/max71.7266ms，全部间隔median66.6969ms。新实测before_command对同frame原录制state的五轴MAE=[0.275058,0.333000,0.208459,0.242091,0.206793]°，P95=[0.690110,1.142858,0.690107,0.865934,0.527473]°，max=[1.230770,2.021976,1.670326,2.109892,0.527473]°。原连续回放MAE=[0.258075,0.334665,0.189477,0.278388,0.196803]°；五轴均值0.253080°对0.251482°。新旧实测位置同frame差MAE=[0.064935,0.074259,0.151515,0.052281,0.013320]°。gripper单列源单位、两轮实测相同。

完整轮起点误差=[−0.439560,−1.230768,+1.406590,−0.439559,−0.439560]°，与原连续轮最大0.7033°并非完全相同；短10轮肘起点+4.6593°且短轮state MAE约2.4703°，不能用短轮充当完整轮同起点对照。源肩抬0.307696°绝对裁剪沿用、未改标定/PID。最后保持后五轴Goal误差=[+0.263736,+0.351648,−0.087912,−0.175824,+0.351648]°，退出raw当前位置hold读回通过、扭矩保留、串口关闭；额外get_joint六轴Status0。

结论限于这次单次对照：原始动作每次取10/执行前5、推进5且无额外块间等待，发送目标相同、时序连续、实测轨迹与原录制和此前连续回放接近，未观察到明显分块退化。尚未验证推理等待/模型预测/实时观测反馈/圆柱任务成功，不据此宣称这些gap消失。本轮现场视觉反馈未另行确认。

产物根`/Users/wujunyu/.cache/carrot/so101_action_chunk_replay_20261008T141608`，保存运行前replay.py/plan.json/sequence_smoke.json、first10/full264 events/results/chunk NPZ/commands.csv、analyze.py/summary.json、chunk_replay_comparison PNG/SVG/PDF。脚本SHA82677eb8e3d7d587c38b0535e80e92f0eaf5d59caf2e62ac6f62cdee0d7b12fc，Carrot HEAD7650c4acf7cbff76b887a7f2cfc096645043f83a，Parquet SHA同前，Mac LeRobot0.6.1，无Docker、上游commit未记录。代码和图均在cache，不新增生产实现、训练或GPU实验，未commit/push。


## 2026-10-08：无视觉step500，预测10/执行1/录制state输入（运行准备）

用户进一步允许延迟，将协议改为每次预测10帧只执行第1帧，然后再请求模型；优先使用屏蔽视觉模型。本轮选择已有无增强state-only500 checkpoint：`$MY_DFS/experiments/carrot/pi05_so101_state_only_rollout/20261004T023935Z/training/checkpoints/step-00000500`，完整264帧/随机训练noise+t，不重新训练。70个必要源码/配置Mac/GPU SHA一致，checkpoint三个hash与原评估归档相同/812张量全BF16/step500已核验。

采用原factory+既有DropVision，所有视觉pixels0/maskfalse；BF16/eager缓存/NFE10，推理noise保持随机。输入仍按frame0..263使用录制state与Knock down the cylinder prompt，图像仅零占位且在服务端明确mask，无相机或leader访问，真机反馈只记录不作为模型condition。复用生产run_loop同步调度：infer→send首1行→补足单动作15fps时段→advance1→下一infer，允许请求延迟且统计实际控制频率。先log-only10，再real5、full264；均用独立输出，不覆盖旧回放。安全sink沿用bus-only现有标定检查、启扭矩前raw Goal预置、慢速对齐/起点5°、目标反馈差20°中止、退出当前位置hold；预测绝对范围裁剪>5°中止，小裁剪明确记录，不使用旧反馈中心5°裁剪，不改PID/标定。

Mac脚本/结果`/Users/wujunyu/.cache/carrot/so101_no_vision_act1_20261008T143126`，源码快照/plan/source_hashes已保存。Gemini当前session26a05933、container mpi-launcher@mpi-1784764303-launcher；MY_DFS经当前唯一CephFS与__SYS_USER_NAME__核验仍为`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu`。服务任务26a05933-0467持久化到`$MY_DFS/experiments/carrot/so101_no_vision_act1_20261008T143126`，serve.sh暂停dguard30分钟/EXIT恢复，无其它GPU实验运行，不动Ray/Isaac。此阶段尚无模型真机结果。结束核验退出/Status/hold、264个NPZ/state/ref/首1动作映射、请求与发送时序，分开统计预测对源action、实测对目标和录制state，退出服务、归档日志并确认dguard恢复。


### 无视觉500-step act1实际结果：真机完整轮提前停止，log-only264完成

| 阶段 | 样本/执行 | 状态 |
|---|---|---|
| 首次log连接 | 0次推理、0个真机动作 | 服务未就绪HTTP502，exit1；未连接串口，独立保留不混入统计 |
| ready后log-only | 10次预测、0个真机动作 | exit0 |
| 真机短段 | 5次预测、5个动作 | exit0/Status0/退出hold通过 |
| 真机完整轮尝试 | 8次预测、仅7个动作 | exit1；frame7目标裁剪>5°，发送前中止/退出hold通过 |
| 后续log-only | 完整264次预测、0个真机动作 | exit0；只分析录制state预测与目标范围，未重跑机器人 |

真机完整尝试的frame7肩抬预测−110.431305°，当前标定下界−104.835165°，越界5.596140°。触发的是**本轮诊断sink设置的绝对裁剪>5°断言**，不是舵机故障、模型异常值、网络失败或旧反馈中心5°相对限幅。frame7的command日志表示尝试发送意图，没有对应action事件，真实Goal写入前已中止；仅frame0..6共7个预测index0被下发。全部Status0，finally预置当前raw Goal并读回通过，保留扭矩/关闭串口，随后get_joint六轴状态仍0。未放宽断言或自动重跑。

完整尝试7个动作的间隔median304.272ms、min295.867/max425.843ms，实际约3.093个动作/秒；8次正式请求median230.601ms。它不是连续15Hz。完整尝试起点差五轴[−0.264,−1.231,+1.934,−0.440,−0.440]°；短5轮肘起点+4.132°，起点并非逐次相同。已执行7个动作的约33ms后实测对Goal MAE=[0.389,1.381,2.989,2.110,0.264]°，样本少且含动态延迟，不能当整episode或模型拟合指标。

后续独立log-only264帧（每帧新随机推理、与中断轮noise未配对）的**每次预测第1动作对原action**五轴MAE=[0.841334,2.075742,1.192955,1.666428,0.158643]°，五轴均值1.187020°；P95=[3.089703,5.362343,3.662127,4.212074,0.411797]°，max=[5.946434,9.410095,5.366211,6.790050,1.652525]°。gripper MAE0.020845源单位另列。全10步2595有效行的五轴平均MAE1.682330°，与本轮采用首1指标分开，不能用之前采用5/8noise平均1.346750°当同协议对照。

该独立log-only轮首动作肩抬29帧越出标定范围、最大需裁4.927141°，肘11帧越界、最大0.784631°；本轮未出现需裁>5°的首动作。它不能推翻实际中断轮frame7需裁5.596140°，两轮noise不同；10次preflight的frame5也曾需裁>5°但未驱动。所以上述“整体MAE小”不代表每次随机输出都能在标定范围内执行。这里只观测到范围约束触发，不凭此宣称训练/推理BUG或模型完全不能执行。

独立读取全部287个实际请求NPZ（10+5+8+264），核对每个原state/reference、frame推进1、planned1、模型shape10×6/有限值、command与预测index0一致、真实action只对应实际成功行、末段reference不跨episode；逐项复算MAE/P95/max、裁剪和时序，保存independent_results.json。loop_smoke.json验证264 frame/265 calls含warmup，无网络/串口；sink_smoke.json验证发送前拒绝极端target和退出raw hold，同样无真实硬件访问。服务预热1次、每个成功client连接另预热1次，不混入正式预测指标。

实际命令：Mac `PYTHONPATH=/Users/wujunyu/work/carrot examples/so101_real/.venv/bin/python <产物根>/client.py --mode log|robot --count 10|5|264 --output <独立case目录>`，uri=ws://29.209.160.111:8080。会话分别preflight失败17459、ready61878 exit0、short38957 exit0、full62454 exit1、log26481203 exit0、analysis19919 exit0。Gemini服务26a05933-0467主动只终止自身PID532888，exit0/414s，后台日志dump释放为service_backend.log；EXIT已恢复dguard，亲核watch1/guard.sh/run.py运行、无restore计划，服务PID释放，源代码与checkpoint hash不变。

Mac产物根同上，含脚本、运行前70文件snapshot/hash/plan、四case和失败连接、commands/predictions CSV、independent_results与final_completion、partial_robot_trial与full_log_predictions PNG/SVG/PDF。Ceph对应产物根保存服务preflight/loaded/serve脚本/源码/后台日志/completion；已拉回Mac server_evidence。原始Mac结果也归档到Ceph mac_results。实际Carrot HEAD7650c4acf7cbff76b887a7f2cfc096645043f83a；GPU包版本由preflight.json记录、Docker image tag/上游commit未记录；未安装下载、未改生产代码/标定/PID、未commit/push、图片不入Git，用户Isaac未触碰。

本轮不是实时closed-loop：全部policy输入为录制state、视觉屏蔽，反馈仅记录；未把预测action或机器人实测反馈作为下一次condition。真机完整回放未完成、圆柱任务成功未验证。下一步需先明确如何处理模型预测超出标定范围；不自动更换模型、训练或继续机器人动作。
