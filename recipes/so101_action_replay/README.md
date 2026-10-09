# SO101 原始示教动作真机 replay（2026-10-08）

最新进展（2026-10-08）：no-vision+jitter step2000真机K1/P32肘腕、3°等待/3s超时后继续，264动作完整执行，263到位/1超时。等待结束时五轴对sent平均MAE0.650°、对原模型目标0.674°；重发已撤销，肩抬一次残差3.351°。当前硬件肘腕P32/其余P16。完整表与曲线见文末。


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


## 2026-10-08：no vision+jitter2000，预测10执行1真机验证

用户要求测试新模型效果，沿用此前每次预测10/执行1/允许同步延迟的方案。选择本轮
`pi05_so101_state_jitter/20261008T093114Z/training/checkpoints/step-00002000`，
BF16/eager cache/NFE10，训练五轴±3°jitter后clamp。新模型SHA605eecdff3df591a6a3249e4b9a6b9070c6f6d656989b185d16e54596f1d4b5c。
先log-only检查→录制state短段5动作验证执行链路→真实state短段10动作，正常后再有限长度闭环。
视觉始终pixels0/maskfalse，不打开相机/leader；下一请求实际读取机器人状态，不将预测action假装成反馈。
生产run_loop同步请求及SO101Sink绝对限幅+LeRobot相对限幅20，诊断脚本仅负责bus-only连接、
预置Goal/缓慢对齐原起点、Status/Goal实测记录及finally当前位置hold；不调用会重新配置的Robot.connect。
真实state模式不将同动作序号的示教当时间对齐GT，单独评估是否移动/裁剪/跟踪/漂移/任务结果。
执行源码7135c722ab0813fa8a08cebe162ff127b8035260，75文件Mac/GPU SHA一致，Docker tag未记录。
Mac脚本与产物`~/.cache/carrot/so101_jitter2000_act1_20261008T114951Z/`，
Ceph `$MY_DFS/experiments/carrot/so101_jitter2000_act1_20261008T114951Z`，
client.py --mode log|robot|live --count N --output CASE。输出为独立目录，不覆盖旧结果，无图片入Git。
启动前bus-only读取六轴Status0、当前位置接近原首state；尚未下发本轮模型动作。
服务启用时dguard暂停30分钟、EXIT恢复，不重启Ray、不训练。结束核验实际执行计数/有限值/
真实输入新鲜度/index0映射/裁剪与Goal读回/Status及退出hold、归档服务日志和核对dguard恢复。


### 实际结果与逐轴轨迹对照（完成归档，真机轮部分执行）

使用上述step2000 checkpoint，生产推理h10/NFE10/eager cache，执行K1；每次重新读取真实机械臂state作为下一请求condition。视觉pixels0/maskfalse，训练和本轮推理noise均随机。运行时沿用当前成功offline eval的690 BF16+122 FP32参数dtype图（norm等有FP32岛），不误写为所有运行时参数均BF16。

| 阶段 | 预测/模型动作 | 结果 |
|---|---:|---|
| log10 | 10/0 | 完成 |
| 录制state robot5首次 | 1/0 | 诊断相对限幅传int导致TypeError；修为float，失败保留 |
| 录制state robot5重试 | 5/5 | 完成、Status0、退出保持读回通过 |
| live10 | 10/10 | 完成；起点最大误差3.516°，与后续精确起点分开 |
| log264 | 264/0 | 完成；首动作对示教action五轴平均MAE1.070083° |
| precise首次 | 0/0 | 诊断sink缺config属性，准备阶段失败，退出保持通过 |
| precise重试 | 158/158 | 第159次请求被操作者SIGINT打断；264动作预算未完成，非硬件故障 |

精确起点五轴最大差0.791206°；158次动作实际3.11325Hz（录制15Hz），median请求240.172ms、发送间隔315.033ms，允许同步等待。16/158个目标触发绝对clamp，肩抬/肘最大2.340103°/0.739243°，相对20°限幅未触发。所有servo Status0，退出当前位置raw hold写入/读回相同、保留扭矩/关闭串口。约33ms反馈对sent goal五轴MAE=[0.886786,2.202572,1.841696,1.862571,0.334069]°，包含动态延迟，不能当模型拟合误差。

**修正现场即时判断**：最初只看腕部边际范围就说“明显偏离”并提前停止，证据不足；原单episode腕屈最低−10.901099°，本轮−14.769231°，超出3.868132°。完整逐轴对照显示大部分路径接近，但约原frame105–135的中间转折被走浅：肩pan示教最低−23.032967°（frame120），实测最低−13.626373°，少转约9.406593°；肩lift这一段也走浅。不能仅用整体均值盖过局部缺失。

为避免把实测动作序号当示教frame，先做五轴最佳单调位置匹配（起点0/可跳过源帧/终点自由），五轴MAE0.746029°，但最大跳30源帧、只匹配87个不同源frame，**会跳过困难段，不能当主要完整轨迹指标**。随后使用只允许相邻步进的DTW覆盖源frame0..223，按每个源frame等权、在其DTW对应实测样本中选最近者：

| 角度轴 | 部分路径对齐MAE ° | P95 ° | max ° |
|---|---:|---:|---:|
| shoulder pan | 1.211538 | 5.362638 | 11.428571 |
| shoulder lift | 2.098116 | 9.542856 | 9.846153 |
| elbow flex | 1.415621 | 4.395607 | 4.483513 |
| wrist flex | 1.268446 | 3.942857 | 9.230770 |
| wrist roll | 0.239011 | 0.527473 | 3.164835 |

五轴平均1.246546°。这是使用实测state与原observation.state的**允许速度调整的局部几何对齐**，不是同时间误差、模型action对GT误差、末端笛卡尔误差或完整重放成功率；终点223来自前一最佳匹配，后续224..263尚未评估，DTW重复匹配不增加真实采样数。图trajectory_comparison蓝线原state、橙线DTW对齐实测、灰区为停止后未覆盖尾部，横轴原数据frame，不是真机动作序号。用户确认现场没有圆柱，只观察动作，因此未测试推倒任务。

独立检查每case NPZ、request live实测输入/recorded原state、预测index0与command及实际send映射、clamp/Goal量化/Status/hold；所有结果保留，未把失败准备与短轮混入158轮。源Parquet SHA cd69202c4c1a09436dd4278a7e8b68584b6e998885716f7f49f817fc884baf6b。服务前两次诊断启动分别因读取config.precision和过严全BF16断言失败，保留service_initial_backend.log/service_dtype_probe_backend.log；修为config.dtype和成功offline dtype图一致后服务task b1c4b5e1-0544正常完成/exit0并dump释放。仅终止本服务PID570649，亲核PID/8080已释放、dguard watch1/guard+run运行/无restore计划；未重启Ray或其他实验。

Mac产物根同准备记录，包含全部client版本、source.json/hash、case日志/NPZ、analyze.py/independent_results.json、align_dtw.py/trajectory_alignment与trajectory_dtw JSON/NPZ、plot_alignment.py、PNG/SVG/PDF、final_completion.json；Ceph同根mac_results归档Mac结果、原根保留服务与源码证据。未修改生产代码/标定/PID，不commit/push，图片不入Git，Isaac未触碰。

结论：本轮真实反馈能沿示教的大部分路径运动，局部转折未充分重现、尾段未完成；尚不能判定完整closed-loop replay成功或失败。下一次如用户要求继续，保持同一协议并先完成全段记录，单独观察中间肩pan/lift转折；不因几度边际超出就将整条轨迹认定为发散。


### 2026-10-08：同模型完整264次真实反馈重放（准备）

用户认可部分路径效果并明确要求完整放完。保持同step2000/no vision/h10执行K1/NFE10/eager/随机noise、真实状态反馈、绝对clamp+相对20°，先精确对齐原frame0。新独立根`so101_jitter2000_full_act1_20261008T121258Z`（Mac cache与Ceph experiments/carrot同名），复用刚验证的client.py/serve.py，源码7135c72，无生产改动。计划执行264个模型动作，不因越过示教边际范围几度提前停止；硬件Status/非有限/通信与Goal检查保持。结束核对264实际发送与实测输入、退出保持，比较全轨迹/最终位置；无圆柱，不判任务成功，不训练或commit/push。


### 同模型完整264次实际执行结果

用户要求“完整放完”，按准备协议重新起点对齐后实际完成264/264模型动作，client exit0，完整264 NPZ/264 request/264 send，`reason=max_chunks`。起点五轴最大差0.439560°。第1到264动作83.104893秒，实际3.164675Hz，median发送间隔307.665459ms。使用本次独立新随机noise，不与上一轮158次noise配对。

53/264动作触发绝对clamp（肩lift最大3.158226°/肘0.895569°），相对20°未触发；全部六轴Status0，Goal读回量化检查通过；finally当前位置raw hold写入/读回相同，保持扭矩、串口关闭，独立get_joint六轴仍Status0。结束后不追加动作或手工回位。

**完成的是264次执行预算，尚未完整重现示教终点**。本次后半段腕部约−6°附近徘徊，到约第220次后才继续抬升，最终反馈37.978022°，原最终state50.373627°，差−12.395605°。最终另外四运动轴对原最终state差[+0.351648,−0.967033,−0.087915,−0.351648]°。随后独立静止读腕38.066°（不同采样时刻）。中间肩pan仍走浅：示教−23.032967°、本次实测最低−12.747252°，少约10.285714°。不把总体均值当作完整replay成功；现场无圆柱，不判任务成功。

对全部原state0..263/实测0..263做起终点锚定相邻DTW，每个源frame等权选其DTW配对中最近实测（允许时间拉伸/重复位置，不是同时间误差）：

| 角度轴 | 全源轨迹几何MAE ° | P95 ° | max ° |
|---|---:|---:|---:|
| shoulder pan | 1.161838 | 4.219781 | 10.285714 |
| shoulder lift | 2.461539 | 12.342858 | 13.978022 |
| elbow flex | 1.625708 | 6.329674 | 6.417580 |
| wrist flex | 2.362970 | 12.483517 | 12.483517 |
| wrist roll | 0.218115 | 0.351648 | 1.494506 |

五轴平均1.566034°。与上轮部分终点223的1.246546°覆盖范围/终点/随机noise不同，不直接作变好变坏对照。全段图蓝线原示教state/橙线完整DTW实测；另保存未经对齐的真实时间/目标/实测图，显示后半段驻留。

亲读264 NPZ逐个核验frame/planned1、无reference输入、实际新鲜读取state与request一致、前一次发送结束后才读取下一state、预测10×6有限、预测index0→command→SO101Sink absolute clamp→SDK实际send/Goal读回一致；核对264完整状态、原Parquet SHA、Status/退出hold，保存analyze.py/independent_results.json/trajectory_dtw.npz。client完全复用上一轮已核验版本，无生产改动/训练/PID或标定修改。

Gemini b1c4b5e1/MY_DFS仍经本次唯一CephFS+__SYS_USER_NAME__核验；Mutagen flush，75源码Mac/GPU hash及checkpoint hash PASS。唯一Mutagen conflict为无关旧wipe_overfit目录未跟踪__pycache__，未改动；本轮必要源码一致。服务task b1c4b5e1-0553、PID571989，确认cmdline归属后SIGTERM，仅关闭本服务，exit0/220s，service_backend.log dump释放；亲核PID消失/8080关闭/dguard watch1/guard+run运行/无restore计划。Docker tag/上游commit未记录。

Mac产物`~/.cache/carrot/so101_jitter2000_full_act1_20261008T121258Z`，Ceph`$MY_DFS/experiments/carrot/so101_jitter2000_full_act1_20261008T121258Z`（mac_results保存Mac所有结果），含原协议脚本/源码快照/hash/plan/launch、live264日志/NPZ、analysis/最终核验、完整轨迹与真实时间PNG/SVG/PDF；图片/数据不入Git。源码7135c72，recipe记录工作区未提交，未commit/push。当前无服务/新机器人动作；后续需分别排查中段转折走浅与后半段推进不稳定，不能仅凭一轮noise确定原因。


### 用户现场反馈与末段抖动线索

用户确认“没卡住、碰撞；不过最后waist一直在抽搐”。waist具体指底座还是wrist已另询问，暂不替用户定轴。按原日志核对最后20次，底座pan raw/sent目标范围2.397917..4.497095°、相邻平均变动0.431056°/最大1.315176°；实测范围2.989011..4.395605°。腕flex目标34.755722..38.219124°、相邻平均0.841494°/最大1.882313°；实测34.285713..38.065933°。这两轴该段raw=sent，绝对/相对限幅未制造跳变。腕roll实测末20次固定−4°，目标相邻最大0.113188°。肩lift目标相邻最大2.848305°，也存在请求间变化。不能仅凭主观词确定哪个关节振荡。

使用实际checkpoint norm_stats，按源码numpy quantile normalize/补零32dim/torch.bucketize逐项复算末60请求状态condition，29对离散state完全相同（prompt相同/视觉全部maskfalse）。例如zero-based请求208和212同完整state桶，pan目标6.012762 vs5.237297°（差0.775465°），wristflex目标−7.678396 vs−4.898486°（差2.779910°）。pi05不使用连续state_proj，连续state差仅桶化后进入prefix；因此此对模型有效condition相同，但输出不同。sample_actions无noise传入时每次sample_noise，模型eval无dropout。这支持随机采样造成目标不一致的线索；未运行同固定noise对照，不能将全部抽搐归因于noise或排除servo追踪/死区影响。

诊断只读已有日志/CPU，无新服务/推理/机械臂动作。tail_jitter_analysis.json、check_condition_jitter.py、same_condition_jitter.json保存证据；Mac直接import carrot做纯normalize时因缺ray失败，随后使用已检查源码的等价numpy算式/torch bucket表达式复算，无安装或静默切换生产实现。下一直接判别实验可固定观测，随机noise vs同noise重复推理，log-only检查目标稳定性，不改PID或用平滑掩盖原因。


用户随后确认waist指手腕弯曲或旋转。日志更指向腕flex：末20次目标相邻最大1.8823°、实测随目标改变；roll实测均−4°、目标相邻最大0.1132°。约3.16Hz逐轮采样不能排除未采到的高频硬件振荡，暂将首要排查定为腕flex目标的跨请求随机不一致；不直接改PID/生产noise或启动新真机试验。


#### 已确认末段模型原始腕屈目标往复

按用户“确认模型输出末端是否往复”要求，重新逐个读取最后15次原始NPZ（zero-based249..263），与command和send核对，腕flex原始预测index0逐项等于实际sent，限幅未改写。14个相邻目标变化中7次上升/7次下降、11次方向反转；目标35.834877..38.219124°，总绝对变动12.148567°但净变化仅+0.290672°。例如末7次36.545→37.573→37.507→38.219→37.309→38.210→36.877°。因此明确确认**模型原始首动作目标在末段往复跳动**，执行端将其下发；该事实无需靠硬件振荡假设。尚未证明随机noise是唯一原因，也未证明预测10动作在单chunk内周期振荡。verify_reciprocation.py/verified_reciprocation.json及tail_reciprocation PNG/SVG/PDF在本轮cache/Ceph，末20次真实请求序号图无DTW。没有新增模型推理或机器人动作。


#### 中段转折走浅：已采用模型目标与机械臂追踪的区分

按用户要求检查“模型输出差异还是转动不足”，读取原始command/send/实测。肩pan原示教state最低−23.032967°；本轮全部264次**实际采用的预测index0**最小−13.125168°，raw=sent、未clamp，实测最小−12.747252°。zero-based请求61（第62次）下发−13.125168°，下一请求62实测−12.747252°，已追至差0.377915°；但该下一模型首动作改发−5.572178°左右，随即折返。因此约10°路径缩水的直接证据指向模型所给/采用目标提前折返，不能解释为“曾发−23°但机器人只到−13°”。肩lift同处目标−33.969°/下一读−31.165°，确有约2.804°跟踪滞后，不能概括所有硬件误差都为0；硬件动态是否通过feedback促成模型折返未做反事实对照。h10/K1，后9步不执行；请求61完整pan未来最深约−15.997°，仍未到示教低点，不能拿未执行的未来预测当下发动作。该比较不将机器人请求序号当原示教frame，source最低是路径极值对照。

保存turnaround_command_vs_feedback.json；只读既有结果，没有新推理或机器人动作。


### 2026-10-08：单步到位后再推理（准备）

用户要求每次动作走到位以减少追踪滞后。新增执行端wait_for_target选项，实际sent目标误差≤1（五轴degrees/夹爪原单位）连续3次才返回，20ms轮询，3秒超时；超时action/全部反馈先记录再停止下一推理，不重发目标/补偿或改PID。runtime新增测试25 passed/6.84s，全部so101_real 57 passed/7.83s，修正import顺序后Ruff PASS。源码7135c72加未提交执行端修改。

新根`so101_jitter2000_wait_act1_20261008T124321Z`，Mac cache/Ceph experiments/carrot同名；沿用同jitter2000模型/无视觉/h10 K1/NFE10/eager/随机noise，首先精确起点live5，五目标均到位才完整264。超时不自动重试，记录静态残差与退出hold。所有原限幅/Status/Goal检查保留，无圆柱，不判任务成功。


### 单步到位等待：CPU通过，真机首动作静态残差导致停止

源码7135c72+未提交wait选项，Mac/GPU90必要文件SHA一致，checkpoint仍605eecdf...；新Gemini session25dad1a8，重新核验当前唯一CephFS/mm-base-plt2/nrwu。live5实际仅发送1个模型动作，未开始第二次请求或full264。起点最大误差0.351645°。每个目标等待误差≤1（五轴degrees/夹爪原单位）连续3次，20ms轮询/3s超时；仅等待、没有重发/反馈补偿/PID修改。

首动作120次反馈/3.004633s，误差[+0.529620,+0.904816,+1.645111,+1.683010,+0.385261,+0.430990]，肘/腕flex超1°。最后整整1秒六轴读数span全0，残差保持不变；目标无绝对/相对clamp，Goal读回通过。因此观察到固定命令下的静态追踪残差，不能认为仅延长等待即可保证到位。首动作已真实发送并记录action（target_reached=false/target_error/target_wait_s/120 target_samples），之后run_loop断言停止，client exit1是预定超时保护，无后续推理。六轴Status0，退出raw当前位置hold读回通过、保留扭矩/关闭串口；独立get_joint仍Status0。hold后肘读91.736°与等待最后91.209°不同，不冒充其为原目标的到位值。

实际sent target=[−1.496653,−103.278442,89.563683,52.470837,−0.165481,0.323986]；等待最后反馈=[−0.967033,−102.373627,91.208794,54.153847,0.219780,0.754976]。模型首目标未改，等待逻辑没有消除静态误差。后续如需保证更小残差，需针对静态误差决定反馈目标修正或servo控制设置；不静默放宽容差/宣称已完成到位，也不自动重跑。

独立核对NPZ/request/action/120samples、全部Status与hold，保存independent_results.json/completion.json。模型服务25dad1a8-0564/PID574692归属检查后仅SIGTERM本服务，exit0/187s，dump service_backend.log释放；亲核PID/8080关闭，dguard watch1/guard+run运行/无restore计划。Mac/Ceph新根同准备记录，mac_results归档全部Mac轻量产物，source_snapshot/hash、plan和服务证据齐全；没有图片/权重入Git、没有commit/push、用户已stage的recipe记录保持index不动。


### 2026-10-08：到位容差2°，K1完整轨迹（准备）

用户接受此前1.68°残差，要求保持wait_for_target/K1跑完。明确以2°（夹爪2原单位）连续3读数到位、3s超时，未到位不自动放宽/重试。新根so101_jitter2000_wait2deg_act1_20261008T125241Z，同jitter2000/no vision/h10 K1/NFE10/eager/random noise，精确起点后计划264实际动作。无Goal补偿/PID/标定改动，原限幅/Status与退出hold保留。


### 2026-10-08：2°到位等待/K1重跑，第二动作静态残差而停止

用户接受上一轮1.68°残差并要求完整轨迹；本轮明确2°（夹爪2原单位）连续3次/3s超时，未修改模型/PID/标定。根`so101_jitter2000_wait2deg_act1_20261008T125241Z`（Mac ~/.cache/carrot、Ceph $MY_DFS/experiments/carrot同名）。同jitter2000/no vision/h10 K1/NFE10/eager/随机noise，源码7135c72加未提交wait实现，90必要文件Mac/GPU SHA和checkpoint hash核验通过。

实际发送2/264动作，client exit1/超时停止，未请求第三次。起点五轴最大差0.703294°。第一动作12次反馈/0.266409s连续3次≤2达标；第二动作121次反馈/3.003357s未达标。第二步肘目标85.266632°，舵机Goal读回85.230769°，最后实测88.747253°，对模型目标残差+3.480621°（对量化Goal约3.516484°）；最后1秒六轴读数span均0。第二动作六轴残差[+0.525252,+0.374023,+3.480621,−0.143879,+0.583692,+0.419673]；两个动作均无绝对/相对clamp。模型index0→sent逐项一致、Goal量化误差<0.1°，下一观测在前次等待结束后读取，NPZ/request/action/反馈重算通过。没有完整轨迹结果。早先commentary误称首动作失败，已向用户更正为第二动作。

这项残差是固定下发目标与实测之间的执行端跟踪误差，不能算成模型对示教的预测误差。只读检查当前servo P16/D32/I0、CW/CCW dead zone1、Status全0；SDK角度换算为互逆线性式，量化约0.088°/tick，不能解释3.48°。PID/负载/摩擦/其他执行细节原因尚未隔离，I0是线索而非定论。寄存器在退出hold后读取，不冒充超时瞬间电流/负载；独立get_joint肘89.275°是hold后的不同目标状态。建议后续绕开模型，用固定关节目标定位跟踪；没有自行改PID或再启动运动。

所有Status0，退出raw hold写入读回一致、扭矩保持/串口关闭，独立读仍Status0。仅归属检查后停止本服务PID575741，task25dad1a8-0570 exit0/217s，dump service_backend.log释放；亲核PID消失/8080关闭/dguard watch1、guard.sh/run.py运行、无restore计划。independent_results.json/completion.json/post_hold_registers.json以及client/serve/hash/source快照保存，Mac轻量结果归档Ceph mac_results；图片/数据不入Git，没有commit/push，保留用户已stage记录。Docker image tag/上游commit未记录。


### 固定小目标跟踪诊断（准备）

绕开模型，bus-only连接/原PID不变/不改标定，初始raw目标预置后保持其他轴，肘±4°与腕±3°每相位4s、9相位，记录raw Goal/Present/Status及被测轴速度/电流/负载/Moving/Goal_Position_2，最后当前位置raw hold。数据为当前实际硬件读数，初始化为当前姿态，无checkpoint或GPU；限校准内、最大偏离起点170ticks、温度<65°C/Status0。脚本diagnose_tracking.py --output CASE先只读检查计划，--execute才运动。新独立Mac根`/Users/wujunyu/.cache/carrot/so101_fixed_tracking_20261008T130136Z`，预期9target/完整反馈与exit_hold，读回目标一致后比较方向性静态残差；不把单次定位当故障根因。


固定目标baseline已完成9相位/exit0，肘负向约3.34–3.43°、正向约0.44–0.53°；腕负向约1.49–1.58°、正向约0.18°，最后1秒位置均静止。开始同一raw anchor/相同9目标的临时P32对照，仅3/4号P16→32、I0/D32保持，目标限小幅范围；使用Lock解锁/写入/读回确认/重新锁定，finally恢复原P/I并读回，其他轴参数不变。命令diagnose_tracking.py --execute --p 32 --reference <根>/baseline/events.jsonl --output <根>/p32，本机session7898，无模型/GPU服务。

P32轮9相位1009样本exit0，肘负向1.5824°、腕负向0.6154°，finally两个轴P16/I0恢复读回且Status0。随后同一9固定raw目标临时I1、P16/D32保持（仅肘/腕），--i 1 --reference baseline/events.jsonl --output i1；结束恢复原参数，之后原P16/I0复测验证可逆性。参数写入均有持久日志与读回，不改变生产默认配置。


### 固定目标诊断完成：方向相关稳态误差，P增益有可逆因果影响

2026-10-08本机直接驱动，不使用模型/网络/视觉/数据集轨迹。6轮各9个相同raw目标、每目标4秒，54段/6295采样；六轮client均exit0/summary completed、所有Goal全段恒定且与下发raw逐项一致，Status全0/退出当前位置hold读回一致。按原始日志独立复算，不用模型输出充当GT。肘目标梯度±4°、腕±3°，其他轴锁存固定目标，未连续重发当前位置。两轴参数实验及单轴控制变量实验均finally恢复P16/I0，D32未改；最终独立硬件读回全六轴P16/I0/D32/Lock1/Status0、扭矩保持/串口关闭。

以下同一目标从较大角度方向接近，记录最后1秒实测−Goal均值，均为degrees，非模型预测MAE：

| 条件（D32） | 肘同目标负向残差 ° | 腕屈同目标负向残差 ° |
|---|---:|---:|
| 原P16/I0 | 3.428571 | 1.582418 |
| 两轴P32/I0 | 1.582418 | 0.615385 |
| 两轴P16/I1 | 3.428571 | 1.406593 |
| 恢复P16/I0 | 3.516484 | 1.582418 |
| 仅肘P32/I0 | 1.582418 | 1.582418 |
| 仅腕P32/I0 | 3.516484 | 0.615385 |

肘同目标raw2923=89.274725°，腕raw2700=54.593407°；所有轮完整9目标raw序列完全相同。原P16从下方返回同目标时，肘仅+0.527473°/腕−0.175824°，从上方时+3.428571°/+1.582418°，显示方向依赖。同相位最后1秒位置span均0，延长等待不是当前解决办法。原P16→临时P32→恢复P16时残差降后再现；单独改肘P时只改善肘、单独改腕P时只改善腕，排除两个关节同时改P的主要混淆。当前能明确归因的是本轴增益影响静态跟踪残差，而不是模型/SDPA/noise/网络延迟/度数换算。原命令到Goal量化<0.1°；raw误差也存在，标定零点不能解释Goal与Present的40tick差。没有绝对/相对clamp参与本固定raw目标试验。

最符合证据的机理是小P下，误差减小时纠偏驱动力不足以继续克服方向相关的负载/静摩擦，留下稳态误差；机械重力与摩擦各自贡献未独立测量，不能断言齿轮损坏或唯一物理原因。I1在4s内未消除残差（写入读回均通过），故不能把问题简单等同I0，也不能宣称任意增大I会解决。直接同类一手报告：https://github.com/huggingface/lerobot/issues/3400，报告LeRobot P16和方向相关不到位/小误差静摩擦；其P32修复结论不直接移植成本机“已完全修复”。LeRobot上游common.py的HoldLatch也明确讨论P控制重力稳态误差：https://github.com/huggingface/lerobot/blob/main/examples/isaac_teleop_to_so101/common.py 。厂商教程确认PID地址21–23：https://www.feetechrc.com/Data/feetechrc/upload/file/20201127/start%20%20tutorial201015.pdf 。

P32在本姿态将肘约3.4→1.58°、腕1.58→0.62°，只是候选执行参数，不代表任意姿态/完整轨迹或震荡、峰值电流已经验证。P32肘试验出现更大的瞬态电流raw峰值（单肘P32减小目标阶段184 vs baseline23），不把改善误差理解为无代价。全部采样实际位置仍在标定内，所有相位结束温度最高56°C，未触发故障。没有改生产默认PID，没有追加模型轨迹/训练/服务，后续若采用P32应先验证整段跟踪和末端稳定性。

主产物Mac `/Users/wujunyu/.cache/carrot/so101_fixed_tracking_20261008T130136Z`、Ceph `$MY_DFS/experiments/carrot/so101_fixed_tracking_20261008T130136Z`：6case events/summary、精确运行版本快照、安装SDK源码/hash、firmware_registers.json、final_register_check.json、independent_results.json/completion.json、samples.csv/phase_metrics.csv及PNG/SVG/PDF。第一4轮中间分析以initial4_前缀保留；最终图只比较本轴P32与P16/I1/恢复轮。Goal_Position_2全部读0，仅作为原始记录，不据此推断内部profile或到位；固件主3/次15、型号777。运行脚本在recipe/diagnose_tracking.py，离线py_compile通过、实际54段完整日志核验通过，未宣称pytest或Ruff运行（本机ruff不可用）。

Carrot HEAD7135c722ab0813fa8a08cebe162ff127b8035260+未提交wait及诊断脚本；Mac LeRobot0.6.1/NumPy2.2.6/Matplotlib3.11.2/pyserial3.5，本机无Docker，上游commit未记录。CLI stdout会话baseline34680/p327898/i152664/restored_p1660392/elbow_only_p329121/wrist_only_p3268187全部exit0；只读preflight48124 exit0/未发目标。无GPU任务/dguard变更，归档前远端确认watch1/guard.sh+run.py运行/无restore计划。图片数据不入Git，未commit/push，用户已stage的recipe记录未动index。


## 2026-10-08：P32 真机 K1 到位等待复测（准备）

状态：本轮已运行并因到位超时提前停止，未完成264。源码 `a4b905351c2e19ef1443d0655b510eb969497217`。同一 step2000/no vision/jitter±3°模型，h10/NFE10/eager/随机noise、K1、2°连续3次/3s超时、相对限幅20°，对齐同录制起点后最多264动作。通过生产 `connect_robot` 在SDK默认配置后设置肘/腕P32，I0/D32及其余P16；读回核对。超时不放宽、不补偿模型目标。比较sent与实测反馈、完成步数；随机推理目标并非旧轮逐项配对，不将变化全归因于P。

Mac产物 `/Users/wujunyu/.cache/carrot/so101_jitter2000_p32_wait2deg_act1_20261008T140224Z`；Ceph `$MY_DFS/experiments/carrot/so101_jitter2000_p32_wait2deg_act1_20261008T140224Z`。实际命令见产物plan.json/client.py/serve.sh；client调用生产run_loop/SO101Sink，起点对齐与退出当前位置hold沿用前轮外部包装。模型权重SHA605eecdff3df591a6a3249e4b9a6b9070c6f6d656989b185d16e54596f1d4b5c。预期NPZ/逐动作等待反馈/PID与Goal读回/summary、最后Status0且保持当前位置。Docker tag/上游commit未记录；图数据不入Git。


### P32/2° 本轮提前停止；用户要求3°重跑

2°轮实际发送3动作，前2动作到位，第三动作肩抬残差2.021973°、肘1.638077°/腕屈−0.050121°，3.003848s超时而停止，client exit1。上一P16轮在第二动作肘3.480621°停止；本轮第二动作肘1.722015°，但模型随机noise/反馈不同，目标85.266632° vs87.904358°，不作为同目标因果估计。第三目标肩抬已被绝对限幅为−106.241753°。Status全0、退出raw hold读回通过、P32保留，服务0585 exit0已释放/8080关闭/dguard恢复。

新3°轮准备：`/Users/wujunyu/.cache/carrot/so101_jitter2000_p32_wait3deg_act1_20261008T140606Z`，用户明确仅放宽到位容差至3°，K1/P32/3s超时/264预算不变；重新对齐起点。此改动放宽执行验收，不代表伺服物理残差自动减小；实测结果待记录。


### P32/3° 实测结果：27动作到位，第28动作肩抬静态误差而停止

同step2000/no vision+jitter/h10/NFE10/eager/随机noise，K1；仅容差2→3°（夹爪3原单位），3s超时与其他参数不变。起点五轴最大差0.263737°，真实新鲜state反馈。client session7356 exit1是超时保护，实际发送28/264动作、前27动作达到连续3次≤3的条件，没有第29次请求；**未完成整段轨迹**。前一条现场进度误报为第25步，已按完整日志更正28。

| 运行 | 实际发送 | 达标动作 | 停止轴与残差（实测−sent，°） |
|---|---:|---:|---|
| 旧P16/2° | 2 | 1 | 肘 +3.480621 |
| 本轮肘/腕P32，2° | 3 | 2 | 肩抬 +2.021973 |
| 本轮肘/腕P32，3° | 28 | 27 | 肩抬 +3.507830 |

3°轮第28动作sent=[−1.248312,−17.529808,59.210396,−12.566795,−3.967663,0.398405]，最后残差=[+0.369191,+3.507830,+0.350044,+0.171191,−0.032337,+0.356571]；108次反馈/3.003770s，最后1秒所有轴位置span0。该步无clamp；完整28步有2步绝对限幅、无相对限幅。28个动作停止等待时五运动轴平均绝对残差分别[0.866409,1.335503,1.443760,0.767571,0.415230]°；夹爪0.342030原单位。该均值是停止等待时残差，受容差/动作目标/反馈路径影响，不能当模型拟合MAE或与旧两步直接做因果比较。

判断：3°放宽使本次能走过更多动作；未改变肩抬的物理控制能力，仍因该轴P16静态残差停止。肘/腕P32在固定目标实验有独立改善证据，此次跟踪也符合该迹象；不能据随机闭环目标变化量化其整段收益。没有进一步放宽、修改肩部P或补偿模型目标。对示教轨迹/任务成功尚未完成验收。

独立读取28份NPZ/request/command/action/全部等待反馈，与sent/Goal逐项核对，复算error与连续3次条件，数据Parquet SHA与原录制一致，PASS；Goal读回量化差均<0.1°、全部Status0/退出raw当前位置hold读回一致。独立停后读回肘/腕P32，其余P16，全轴I0/D32/Lock1/Torque1/Status0、最高温56°C，串口关闭。硬件P32保留，未恢复P16。

模型服务25dad1a8-0589 PID579946 exit0/123s，归属确认后仅SIGTERM本服务，日志dump释放；亲核进程消失/8080关闭/dguard watch1、guard.sh/run.py运行/无restore计划。上轮0585同样已清理。两轮产物各自Mac ~/.cache/carrot 与Ceph $MY_DFS/experiments/carrot同名目录，含client/serve/plan、原始反馈、CSV、verify_run.py/independent_results.json、hardware_after与服务证据；90个必要源码Mac/GPU hash一致，源码a4b9053，模型权重SHA同前。实际Docker tag/上游commit未记录；无图数据入Git，未commit/push。


## 2026-10-08：超差时重发固定sent目标的真机对照（准备）

NOT RUN。用户要求mismatch时重发同一目标。源码a4b9053加未提交actions.py改动；26项CPU模拟测试/Ruff通过。同step2000 no-vision+jitter模型、h10/NFE10/eager/随机noise、K1/P32肘腕/其他P16、3°连续3次/3s超时，最多264动作；不补偿目标、不重置截止时间。先慢速倒序已执行目标回起点，然后原精确对齐；记录target_resends与逐反馈resent、Goal读回和退出hold。对照是上一轮3°/28动作（27达标，第28肩抬3.507830°超时），随机模型闭环非精确配对。

产物Mac `/Users/wujunyu/.cache/carrot/so101_jitter2000_p32_resend_wait3deg_act1_20261008T141851Z` /Ceph `$MY_DFS/experiments/carrot/so101_jitter2000_p32_resend_wait3deg_act1_20261008T141851Z`；命令见plan.json/client.py/serve.sh。预期核对重发目标恒定、发送次数、到位/超时不推进、反馈与状态、停止后hold；没有因重发成功就宣称解决所有跟踪问题。镜像tag/上游commit未记录，图数据不入Git。


### 重发版本实测：32动作/31到位，同目标重发未消除肩抬静态残差

client58503 exit1按3s超时停止：实际32/264动作、前31连续3次≤3达标，无第33请求。全程200次重发，第32动作115次反馈/114次同sent重发，3.006554s后肩抬仍+3.351120°；最后1秒所有轴读数span0。该步sent肩抬−17.724747°、Goal读回−17.802198°、实际−14.373627°，肘+1.420452°/腕+0.363421°。这是执行跟踪残差，非模型对示教误差，也不证明机械上该角度不可达。没有自行更改肩抬P或继续放宽容差。

| 相同K1/P32肘腕/3°/3s条件 | 发送动作 | 达标动作 | 最后肩抬残差 ° | 最后动作重发次数 |
|---|---:|---:|---:|---:|
| 前轮只等待 | 28 | 27 | 3.507830 | 0 |
| 本轮超差重发 | 32 | 31 | 3.351120 | 114 |

模型noise和反馈轨迹未配对，28→32不能归因于重发收益；能直接确认的是本轮同目标重发114次仍未消除静态误差。32份NPZ/request/index0/Goal/反馈及target_resends=sample resent计数独立复算PASS，已在容差时不重发、截止最后一条不重发，驱动重发返回目标与首次sent严格一致（生产assert）。五轴停止等待时MAE[0.825144,1.270191,1.469306,0.850263,0.427472]°，夹爪0.347422原单位；非模型MAE，受停止容差影响。

重发实现已保留于actions.py，runtime CPU26项/Ruff PASS；未commit/push。起点前慢速倒序上一轮28目标回退exit0，再原精确对齐（最大差0.615382°）。全部Status0、末尾raw当前位置hold读回一致；独立硬件读回肘腕P32/其余P16/I0/D32/Lock1/Torque1，最高温56°C/串口关闭。服务25dad1a8-0594 PID581578 exit0/132s，归属确认SIGTERM后dump释放，进程消失/8080关闭，dguard watch1/guard.sh/run.py运行/无restore计划。

产物Mac ~/.cache/carrot/so101_jitter2000_p32_resend_wait3deg_act1_20261008T141851Z，Ceph $MY_DFS/experiments/carrot同名根（含mac_results）。90必要源码Mac/GPU SHA一致；a4b9053加未提交重发代码，模型权重同前，无新训练；镜像tag/上游commit未记录。若继续定位，建议固定同一肩抬raw目标对比P16/P32，不将换随机模型目标当控制变量验证；尚未执行。


## 2026-10-08：撤销retry，到位等待超时后继续完整轨迹（准备）

NOT RUN。用户明确撤销刚才的重发逻辑，允许到位超时后继续下一模型动作，完整记录目标与反馈差距。actions.py已恢复HEAD单次发送，只改runner去掉target_reached失败即终止的断言；其余故障正常终止。runtime25项/Ruff PASS。完整264动作，同step2000/no-vision+jitter模型，P32肘/腕、其他P16，h10/K1/NFE10/eager/random noise、3°连续3读数/3s等待、相对限幅20°。超时后下一次使用最新真实state，超时仍记false，不冒充到位。

先慢速倒序上一轮32个已执行目标回起点，再精确对齐。每动作一次记录raw模型目标、absolute/relative限幅sent、最后等待反馈和servoGoal读回；完成后各轴MAE/P95/max按264个动作各一次统计，模型target-vs-real与sent-vs-real分列。保存CSV与全程曲线（不进Git）。产物Mac `/Users/wujunyu/.cache/carrot/so101_jitter2000_p32_wait3deg_continue_act1_20261008T142755Z` /Ceph `$MY_DFS/experiments/carrot/so101_jitter2000_p32_wait3deg_continue_act1_20261008T142755Z`，脚本与命令见plan/client/serve；最大等待总量792s，dguard暂停20分钟后结束恢复。源码a4b9053加未提交改动，镜像tag/上游commit未记录。


### 完整执行 PASS；到位263/264，单次超时已记录并继续

client22706 exit0、summary completed=true/264chunks/264steps，264份有限float32[10,6] NPZ、一对一request/command/action，1265次等待反馈。起点五轴最大差0.351648°；动作前只发送一次，等待期间不重发，达到3次≤3或最多3s后继续使用实测state推理。独立从原始日志重算NPZ/index0/最新live_observation配对、Goal读回<0.1°、等待条件、误差与结束hold全部PASS。不会把不达标动作记成到位。

**主指标为每个动作等待结束时的实测−实际发送目标sent，各264样本且包含超时：**

| 关节 | MAE ° | P95绝对误差 ° | 最大绝对误差 ° |
|---|---:|---:|---:|
| 肩旋shoulder_pan | 0.507350 | 1.567314 | 2.456977 |
| 肩抬shoulder_lift | 1.106184 | 2.503593 | 3.351345 |
| 肘elbow_flex | 0.546800 | 1.790310 | 2.768364 |
| 腕屈wrist_flex | 0.684801 | 1.509824 | 2.443221 |
| 腕转wrist_roll | 0.403468 | 0.598189 | 0.961319 |

五轴均值0.649721°；夹爪另计MAE0.352490/P95 0.433713/max0.443431原始单位。相对**原模型输出**的五轴MAE=[0.507350,1.188565,0.583423,0.684801,0.403468]°，均值0.673522°；raw目标vs实测绝对max肩抬4.989655°。本轮28动作有绝对clamp，无相对clamp，所以raw-vs-real与sent-vs-real不完全相同。这里均为执行端跟踪差距，**不是模型对示教的拟合误差**。

唯一超时为第34动作，肩抬sent−21.241455°、最后误差+3.351345°，等待3.005916s后已记录并继续第35动作。263动作满足连续3次≤3，整体等待均值0.096672s/累计25.521511s；第一至末动作日志窗口95.176855s，不等于原录制17.6s。初始运动尚未到位的瞬态sample最大差另列[7.024497,12.679335,7.712952,15.415575,2.895385]°，不与等待结束残差混用。x轴为执行的模型动作编号1..264，非原录制等时间帧。

结果：本次完整闭环的大多数目标可在3°内跟上，少数姿态仍有肩抬静态残差；忽略一次到位超时后，轨迹能够继续完成。单次随机noise闭环没有精确配对P16对照，不能量化P改动的整段因果收益，亦不能宣称任务成功或示教轨迹已经复现。状态全0，退出当前位置raw hold读回一致/扭矩保留/串口关闭；当前肘腕P32、其余P16/I0/D32/Lock1。

温度说明：本轮`Present_Temperature`寄存器最大原始值57，来自夹爪；用户手摸外壳感觉凉。停后六轴原始读数54/56/55/53/54/57，型号全777，LeRobot sync_read、逐轴read与直接packet_handler.read1ByteTxRx(address63)三种方式逐轴一致，厂商官方SDK同样定义温度地址63并直接返回字节：https://github.com/ftservo/FTServo_Arduino/blob/main/src/SMS_STS.h 、https://github.com/ftservo/FTServo_Arduino/blob/main/src/SMS_STS.cpp 。不存在当前可见的软件归一化/地址错配证据，但未用外部温度计校准，不能将上报57直接称为外壳实测57°C，内外温差与温感偏差尚未分离。历史“最高57°C”表述据此澄清为寄存器上报值。

曲线与数据在Mac产物根：target_feedback.png/svg/pdf（raw模型/sent/等待结束实测）、tracking_residual.png/svg/pdf（实测−sent），target_feedback.csv（264动作全部6轴目标/反馈/两类误差）、wait_samples.csv（1265反馈）与analyze_run.py/independent_results.json。图片均不入Git。

服务25dad1a8-0599 PID583154 exit0/247s，归属确认后SIGTERM/dump归档释放；亲核进程消失/8080关闭/dguard watch1/guard.sh/run.py运行/无restore计划。Mac ~/.cache/carrot/so101_jitter2000_p32_wait3deg_continue_act1_20261008T142755Z 与当前确认DFS /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot同名根/mac_results。90必要源码SHA一致，模型SHA仍605eecdf...；Carrot a4b9053加未提交超时继续代码，CPU runtime25项/Ruff PASS；镜像tag/上游commit未记录。未commit/push，没有修改标定或添加训练预算。


### 2026-10-08：逐轴曲线复核与用户反馈

已在对话展示目标/实测轨迹与结束等待残差两张六关节图。横轴为执行动作编号1–264；蓝线raw模型目标、橙线限幅后sent、绿线等待结束实测，残差图为实测−sent，红虚线±3，红点标记第34动作超时。夹爪为原始单位，排除于五轴平均值。各轴MAE/P95/max见上表，肩抬误差最大；五轴平均0.649721°并非每轴都为0.65°。

用户查看后反馈“看着还行”，接受本轮目标跟踪表现；这不是模型轨迹对示教轨迹或推倒圆柱任务的验收。温度问题按用户要求暂时搁置，不继续排查或据未校准读数推断真实外壳温度。保留K1、只发送一次、等待最多3s且超时记录后继续、下一次使用实测state的执行方式。

Ceph mac_results归档共383项文件，按archive_hashes.json逐文件SHA256复核PASS；图像/CSV/NPZ/脚本与原始日志留在缓存及Ceph，不提交Git。代码及文档提交包含超时继续行为和对应回归测试；本轮测试结果仍为25 passed/Ruff PASS，无新增训练或机器人动作。


### 2026-10-08：下一阶段末段手腕抖动，原始NPZ重新核验

本阶段目标区分模型输出跳动与机器人执行振荡。只读两轮既有264动作原始NPZ/command/action/反馈，无新推理或机器人运动。复算预测index0与command一致，末20次两腕轴raw=sent，限幅未制造跳动。

| 末20动作 | 原P16/无等待 | 最新肘腕P32/等待后继续 |
|---|---:|---:|
| 腕屈模型目标范围 ° | 34.755722–38.219124 | 36.310707–39.087227 |
| 腕屈相邻目标平均/最大变化 ° | 0.841494 / 1.882313 | 0.737707 / 1.843342 |
| 腕屈19次变化中的方向反转次数 | 14 | 13 |
| 腕转相邻目标最大变化 ° | 0.113188 | 0.094159 |

最新轮末20次腕屈实测范围37.186813–38.769230°，结束等待对sent的MAE0.656732°；腕转实测固定−3.824176°。旧轮末20次腕转实测固定−4°。旧轮腕屈最后15次目标11次反转、总绝对变化12.148567°/净变化0.290672°，复现历史记录；原始末7次36.544617→37.572815→37.507275→38.219124→37.308605→38.209980→36.876751°。

结论：跨请求模型首动作腕屈目标本身已往复，执行端发送该变化，低MAE并不意味着目标平滑；不能只归因于硬件自行抖动。随机diffusion noise、状态桶切换/反馈耦合尚未分离，不能声称noise是唯一原因；也不能排除观测间未采到的高频servo振荡。两轮不同noise/state/P/wait，数字变化不能当作P32抑抖因果收益。这里的“末段”为最后20个实际模型动作，不等同原录制最后20frame或完整示教终点。

图已展示：上排旧P16/无等待，下排最新P32/等待，左腕屈/右腕转，横轴动作245–264。旧轮实测采用下一次推理前观测（最后一个无后续观测故留空），最新轮用结束等待反馈，采样协议差异已明确。分析脚本/results.json/PNG/SVG/PDF在Mac `/Users/wujunyu/.cache/carrot/so101_wrist_jitter_review_20261008/`，图片数据不入Git。本阶段尚未做固定观测随机noise/固定noise的模型重复推理对照；该对照能进一步区分采样随机性与state反馈变化，不用机器人。


### 2026-10-08：仅离线进一步分析，相同输入输出变化与K1推进停滞

用户要求先分析、不得驱动机械臂。本轮只读Mac既有日志/NPZ与原Parquet，未打开串口/相机、未启动模型服务或新GPU推理、未改生产代码。检查生产SO101 quantile normalize→pad32→torch.bucketize生成state文本；pi05=true不走连续state_proj，输出为绝对动作反归一化、无state加回；prompt统一、DropVision全0/maskfalse。桶重建是已核对源码的float32公式，而非当时直接记录的token ids；另以完整raw state完全相等的请求对提供更强证据。

最新P32/wait轮最后20次共17对同桶输入（11个不同condition），其中2对raw state逐元素完全相同。第258和260次（1-based）raw六维state完全相同，腕屈首动作却37.987293° vs39.076000°，差1.088707°。同桶但raw略异的相邻第252/253次首动作37.243885° vs39.087227°，差1.843342°。末60有86对同桶/15对完全相同raw state，完全相同raw state的最大首动作差1.508904°（第234/235次）。旧P16/no-wait末20完全相同state第258/259次差1.028198°，末60第212/213次差2.642297°。因此跨请求变化不能全部归因于机器人state变动或桶边界跳变。

源码sample_actions(noise=None)每次抽torch.normal(0,1)，同一process后续10次Euler更新不再抽noise，policy模型eval。随机noise是首要可隔离嫌疑；历史NPZ均未存noise，本轮没有固定noise重跑，不能宣称已完成noise因果证明或绝对排除其他数值不确定性。

最新最后20个chunk内部腕屈相邻动作平均变化0.457472°，跨请求首动作平均变化0.737707°。20/20个chunk的第10个动作大于首动作，平均A9−A0=+4.117244°，A9均值41.750053°；但实际采用的A0−输入state均值−0.310046°，20次中14次向低角度发目标，范围−1.191830..+1.724590°；末20输入state净推进仅+1.054947°。这与K1每次只执行首动作、反复重算后留在38°附近相符，不能据此声称K5必能解决或机器人完全无贡献。3°到位容差允许这类小目标差立即通过，wait不保证有净进展；固定目标静态误差/死区仍可能共同影响闭环。

原Parquet SHA通过。录制最后20帧腕屈action50.505493..50.593407°，相邻平均变化0.009254°/最大0.087914°、净变化0；state49.846153..50.373627°。真实最后20次输入腕屈37.186813..38.769230°，所以运行预算末段不是已达到示教最终姿态。两者未做时间配对，不将差距当同步误差。

五轴raw±3°增强支持集合只做离线几何检查：最新末20输入各可由原frame223/224加增强得到，对应腕屈action38.021976..38.109890°（原state腕屈35.868/36.835附近）；并不是这些输入存在多度腕屈冲突标签的证据。无视觉/无历史的状态条件与K1反馈可能让模型停留在某个较早局部状态；增强是否造成该现象仍未隔离，不能直接归咎于jitter。

分析脚本condition_analysis.py、condition_results.json和运行log保存在Mac ~/.cache/carrot/so101_wrist_jitter_review_20261008/，旧/新两轮数据路径见前节；不入Git。下一明确诊断为相同记录state下随机noise多次与固定noise重复的模型-only对照，保存输入token/noise/full chunk；尚未执行。温度继续搁置，不修改PID或执行协议。
