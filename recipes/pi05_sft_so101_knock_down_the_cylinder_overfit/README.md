# SO101 knock_down_the_cylinder：拟合实验

## 2026-10-03：完整单 episode 延长拟合（训练与评估完成）

用户取消后续无视觉、单frame、固定noise等简化实验，继续已验证能拟合的完整episode条件。
目的：在同一训练数据上增加训练预算，比较误差能进一步降低到什么程度；真机测试等用户休假结束。
以下历史配置与结果保留，本轮以本节和当前`train.yaml`为准。

| 项目 | 本轮配置 |
|---|---|
| 数据 | `nrailg/knock_down_the_cylinder_1_20260930_222251`，1 episode / 264帧 / 15 FPS |
| 输入 | 全部帧、腕部RGB、原prompt `Knock down the cylinder`、原metadata统计量 |
| 随机性 | 普通`Pi05SFTLossFn`，每batch随机noise与t，不使用简化实验worker |
| 初始化 | 官方`Physical-Intelligence/pi05_base_pytorch`的独立副本`pi05_base_pytorch_h10`；权重不变，不resume，新optimizer/scheduler，从step0开始 |
| action horizon | 副本`config.json`中只将`action_horizon`从50改为10，匹配dataset h10 |
| 预算 | 全新训练5000step，未加载先前SFT模型或训练状态 |
| LR | 前100step warmup到1e-6，之后constant1e-6；decay_steps=5000且floor=peak |
| 并行 / batch | 8 GPU BF16 FSDP，micro4 / global64 / GAS2，seed1000 |
| checkpoint | 每1000step保存；预期新step1000/2000/3000/4000/5000，模型导出BF16 |
| 输出 | `$MY_DFS/experiments/carrot/pi05_sft_so101_knock_down_the_cylinder_overfit_h10_5000`，禁止覆盖旧结果 |
| 状态 | 训练exit0/5000step/9371s，5000条有限指标；五checkpoint评估exit0/779s，1460NPZ远端/Mac核验PASS，dguard恢复运行 |
| 源码 | 提交`27bedad2148018df38aa3d0c6ba87f6bd1b21664`，分支`whyRecipeFailed2`；训练源码5个关键文件Mac/GPU SHA一致；镜像tag未记录 |
| 监控 | 每30分钟检查全量新增loss/grad norm/LR、进程和checkpoint；NaN/Inf或崩溃提前停止；训练器本身在非有限loss时立即失败 |

本轮五个checkpoint全10步主指标（反归一化角度）：

| step | 肩旋 MAE° | 肩抬 MAE° | 肘 MAE° | 腕屈 MAE° | 腕转 MAE° | 验证 |
|---:|---:|---:|---:|---:|---:|---|
| 1000 | 0.724 | 1.947 | 1.075 | 1.099 | 0.120 | 292预测/远端与Mac核验PASS |
| 2000 | 0.662 | 1.903 | 1.035 | 0.961 | 0.107 | 292预测/远端与Mac核验PASS |
| 3000 | 0.593 | 1.489 | 0.840 | 0.852 | 0.096 | 292预测/远端与Mac核验PASS |
| 4000 | 0.502 | 1.264 | 0.764 | 0.775 | 0.087 | 292预测/远端与Mac核验PASS |
| 5000 | 0.546 | 1.363 | 0.641 | 0.729 | 0.079 | 292预测/远端与Mac核验PASS |

### 本轮主指标与完整统计

以下均为本次官方base新训练产生的checkpoint，五运动轴按肩旋/肩抬/肘/腕屈/腕转排序，
角度均已反归一化。264个唯一主观测各一次，不混入28个跨noise探测；episode padding排除。

**首1/5/10步：五运动轴 MAE / P95 / max（°）**

| step | 窗口 | MAE 五轴 | P95 五轴 | max 五轴 |
|---:|---:|---|---|---|
| 1000 | 1 | [0.583, 1.472, 0.853, 1.022, 0.119] | [1.761, 3.490, 2.404, 2.858, 0.277] | [3.317, 5.748, 4.144, 4.719, 2.110] |
| 1000 | 5 | [0.624, 1.693, 0.926, 1.011, 0.114] | [2.173, 4.617, 2.763, 2.836, 0.279] | [3.859, 8.039, 5.262, 6.361, 2.785] |
| 1000 | 10 | [0.724, 1.947, 1.075, 1.099, 0.120] | [2.580, 5.586, 3.209, 2.917, 0.314] | [7.108, 10.694, 5.530, 7.877, 2.785] |
| 2000 | 1 | [0.520, 1.391, 0.813, 0.996, 0.098] | [1.709, 3.396, 2.518, 2.521, 0.242] | [2.557, 6.177, 4.825, 4.395, 1.771] |
| 2000 | 5 | [0.595, 1.692, 0.946, 0.940, 0.101] | [2.101, 4.750, 3.112, 2.381, 0.267] | [3.693, 9.655, 6.025, 4.842, 2.560] |
| 2000 | 10 | [0.662, 1.903, 1.035, 0.961, 0.107] | [2.394, 5.351, 3.163, 2.402, 0.298] | [5.819, 12.418, 6.861, 7.798, 2.560] |
| 3000 | 1 | [0.458, 1.156, 0.718, 0.846, 0.093] | [1.485, 2.844, 2.275, 2.153, 0.221] | [2.516, 5.912, 5.144, 3.581, 1.365] |
| 3000 | 5 | [0.522, 1.363, 0.795, 0.815, 0.092] | [1.916, 3.927, 2.468, 1.992, 0.234] | [3.535, 7.452, 5.841, 4.935, 1.620] |
| 3000 | 10 | [0.593, 1.489, 0.840, 0.852, 0.096] | [2.058, 4.475, 2.593, 2.195, 0.258] | [4.602, 9.979, 5.841, 7.388, 1.620] |
| 4000 | 1 | [0.426, 1.097, 0.645, 0.755, 0.079] | [1.444, 2.549, 2.014, 2.111, 0.219] | [2.367, 5.291, 3.870, 3.006, 1.048] |
| 4000 | 5 | [0.460, 1.193, 0.713, 0.763, 0.082] | [1.600, 3.412, 2.163, 1.819, 0.221] | [2.505, 6.593, 4.676, 3.762, 1.408] |
| 4000 | 10 | [0.502, 1.264, 0.764, 0.775, 0.087] | [1.701, 3.473, 2.293, 1.922, 0.240] | [4.928, 8.631, 4.676, 4.553, 1.408] |
| 5000 | 1 | [0.448, 1.146, 0.582, 0.788, 0.081] | [1.521, 2.562, 1.860, 1.994, 0.211] | [2.687, 5.390, 3.339, 3.042, 1.137] |
| 5000 | 5 | [0.506, 1.285, 0.627, 0.744, 0.079] | [1.773, 3.033, 2.024, 1.920, 0.216] | [3.548, 5.879, 4.242, 3.458, 1.499] |
| 5000 | 10 | [0.546, 1.363, 0.641, 0.729, 0.079] | [1.873, 3.244, 2.092, 1.887, 0.223] | [4.915, 8.103, 4.242, 5.024, 1.499] |

**额外28个noise探测：全10步，280有效action行，单列（°）**

| step | MAE 五轴 | P95 五轴 | max 五轴 |
|---:|---|---|---|
| 1000 | [0.610, 2.412, 1.288, 2.456, 0.169] | [3.463, 6.438, 4.321, 7.052, 0.367] | [6.268, 8.717, 5.854, 10.204, 0.480] |
| 2000 | [0.630, 2.485, 1.532, 2.372, 0.229] | [3.161, 5.322, 5.953, 6.434, 0.469] | [4.960, 7.262, 6.773, 8.299, 0.535] |
| 3000 | [0.556, 1.597, 1.342, 2.665, 0.178] | [3.085, 4.087, 5.623, 6.689, 0.369] | [4.787, 5.472, 6.331, 8.541, 0.413] |
| 4000 | [0.457, 1.315, 1.150, 1.542, 0.166] | [2.505, 3.822, 4.585, 3.984, 0.336] | [3.211, 7.124, 5.233, 5.761, 0.478] |
| 5000 | [0.550, 1.682, 1.035, 1.608, 0.144] | [2.266, 3.554, 4.200, 4.021, 0.278] | [3.320, 4.219, 4.864, 4.878, 0.367] |

**gripper：源数据单位，全10步单列**

| step | 主MAE / P95 / max | probe MAE / P95 / max |
|---:|---|---|
| 1000 | 0.009742 / 0.034843 / 0.084761 | 0.014214 / 0.045484 / 0.094728 |
| 2000 | 0.007319 / 0.022624 / 0.085167 | 0.011960 / 0.036227 / 0.073891 |
| 3000 | 0.006786 / 0.018966 / 0.085781 | 0.010834 / 0.032636 / 0.080580 |
| 4000 | 0.005454 / 0.014302 / 0.078622 | 0.009610 / 0.032719 / 0.074729 |
| 5000 | 0.004698 / 0.011757 / 0.086013 | 0.007323 / 0.021609 / 0.086068 |

**最差主观测：全10步五运动轴的最大单点误差**

| step | frame | max（°） |
|---:|---:|---:|
| 1000 | 10 | 10.694 |
| 2000 | 10 | 12.418 |
| 3000 | 10 | 9.979 |
| 4000 | 79 | 8.631 |
| 5000 | 9 | 8.103 |

本轮五轴平均MAE由step1000的0.992852°降到step5000的0.671664°，下降32.4%。
4000→5000的均值仅改善1.0%，肩旋/肩抬回升，肘/腕两轴继续改善；按该平均指标step5000最好，肩两轴step4000更好。
step5000肩抬P95=3.244°、max=8.103°；平均误差较小仍保留局部尾部误差。
先前独立1000-step BF16 run均值0.626369°，不属于本次训练。新5000-step未全面超过该旧run，不能据此声称增加步数必然更好；本轮配对趋势只比较本次五份checkpoint。

**训练曲线与证据**

[逐step CSV](artifacts/h10_5000_20261003/training_metrics.csv)。曲线PNG/SVG/PDF保存在下述MONITOR目录与Mac缓存`~/.cache/carrot/so101_h10_5000_20261003T123204Z/`中，不提交图片到Git。
对应MONITOR同名文件，PNG已检查并在Codex请求打开。5000个原始log点保留，深色线为trailing100-step算术均值，loss/grad用log轴；
grad为clip前全局L2范数，clip阈值1。LR为worker在scheduler.step后打印的值，不另作时间对齐。
训练前100step loss均值0.07382257/grad均值1.21558188，末100step loss均值0.0011378/grad均值0.08534488；
末段有随机波动，训练loss并非严格单调。训练task32434904-0284 exit0/9371s、评估task32434904-0302 exit0/779s，
均已dump归档释放，日志training_backend.log/evaluations_backend.log。五组共1460 NPZ、每组264主样本/2595主有效行+28probe/280行；
原Parquet/reference/padding/noise逐元素核验、MAE/P95/max远端与Mac复算通过，292个noise彼此不同且跨checkpoint配对。
checkpoint/source hashes核验通过，五组source与加载dtype规则一致；存储812个BF16 tensors，生产加载保留690 BF16/122 FP32参数规则。
optimizer_metadata_validation.json记录五组scheduler step与2428项optimizer-only metadata，参数更新证据见training_completion.json。
Mac首次FP64复算step4000 probe P95与报告FP32统计差1.45e-7；远端同NumPy2.3.1复现实验确认是统计插值dtype差。
按报告的FP32统计定义复算全部通过，FP64差值在各mac_validation.json单列，预测与参考数据未修改。
result_summary.json、final_remote_checks.json、各step的metrics/provenance/loaded_dtypes/independent_validation/mac_validation及脚本快照已留档。
训练源码提交为`27bedad`；评估源码扩展、本轮记录及曲线/CSV随结果另行提交，执行版本以source_snapshot和SHA为准。Docker tag未记录；包版本见preflight/provenance。
评估结束dguard DGUARD_WATCH=1、guard.sh PID224、run.py实际运行、restore not scheduled，8GPU均1185MiB/100%为dguard；
没有操作机器人或启动服务。用户休假后继续原始action replay→dataset-input policy执行→同步简单closed-loop→泛化。
本轮必需检查已完成，so101自动检查已暂停，不追加训练或简化消融。

以下保留启动前评估计划与准备记录，实际完成结果以本轮主指标节为准。

计划完成后保存逐step CSV与完整loss、clip前grad_norm、LR曲线；原始曲线保留，若加平滑明确窗口。
五组使用同一264主观测、noise及10NFE，额外28个noise探测单列，首1/5/10的MAE/P95/max分别统计。
所有角度指标为反归一化后的degrees，gripper按源单位单列。

Gemini session `32434904`，container `mpi-launcher@mpi-1784764303-launcher`；复用已核验空闲Ray
`29.209.160.111:6379`，本次job `05000000`。8张H20，训练前dguard暂停360分钟，driver EXIT trap恢复。
持久化监控目录：
`$MY_DFS/experiments/carrot/pi05_sft_so101_knock_down_the_cylinder_overfit_h10_5000_monitor/20261003T123204Z`。
其中`preflight.json`/`launch.json`/`source_snapshot`记录配置、数据hash、源码、环境与初始化证据；
`ray_logs`保存本job日志副本。指标来自`launch.json`所记rank0 worker stdout，driver日志只负责状态，
不能因driver没有step日志误判没有进展。后台结束后归档`training_backend.log`。
本次环境：torch2.11.0+cu128、transformers5.5.4、lerobot0.6.1、NumPy2.3.1、Ray2.58.0。

21:10检查：task仍running，累计1205条连续有限指标，step1–99 warmup与step100后constant LR均核验通过
（日志LR取值按打印舍入容差核对）。step1000为812个BF16 tensors、h10、trainer_state step1000，
state/action stats文件可解析、8个optimizer shard与metadata均非空；对比官方base的BF16数值，
action_in/out projection及out bias均有实际更新。只确认保存与参数更新，不声称checkpoint GPU加载或拟合效果通过。
当前8GPU训练持续，dguard暂停且run.py不运行、恢复已排程，无需延长暂停。
证据：MONITOR下`progress.json`、`progress_20261003T131019Z.json`、`checkpoint1000_header_parameter_check.json`、
`training_metrics.csv`及刷新后的`ray_logs`。未启动额外训练或并发GPU评估。

21:40检查：task仍running，step1–2224连续有限；较上次新增1019条指标全部有限，
最近100step loss均值0.00145545；warmup与constant LR逐条核验PASS。step1000/2000均为
812个BF16 tensors、h10、对应trainer_state.step、8份非空optimizer shard和metadata，
action_in/out projection与out bias相对官方base均有实际变化。5个关键训练源码/config文件
与启动快照hash一致；dguard仍暂停、run.py停止、恢复已排程，不需要延长。
本次证据：`progress_20261003T134032Z.json`及更新后的`progress.json`、`launch.json`、
`training_metrics.csv`、`ray_logs`；没有追加训练、并发评估或修改已加载代码。

22:10检查：task仍running，累计3243条连续有限指标，新增1019条全部有限，
最近100step平均loss0.00119657；warmup/constant LR核验PASS。新step3000 checkpoint
为812个BF16 tensors、h10、对应trainer_state.step、8份非空optimizer shard/metadata；
统计量与step1000逐字节相同，action projection参数相对官方base均有实际变化。
5个关键训练源码/config与启动快照hash一致；dguard持续暂停且run.py停止，恢复已排程。
本次证据`progress_20261003T141044Z.json`及更新后的progress/launch/CSV/ray_logs，
checkpoint仅完成文件/header/参数核验，GPU加载及拟合误差仍待统一评估。

22:40检查：task仍running，累计4255条连续有限指标，新增1012条全部有限；
最近100step平均loss0.00091627，LR warmup与constant阶段核验PASS。新step4000
为812个BF16 tensors/h10/trainer_state step4000，8份optimizer shard/metadata非空，
stats与step1000相同，action projection参数相对官方base有实际变化。
5个关键源码/config与启动快照hash一致，dguard暂停/run.py stopped/恢复已排程，
无需延长暂停；没有追加训练或并发评估。证据`progress_20261003T144031Z.json`，
以及更新的progress/launch/CSV/ray_logs。训练完成与GPU拟合评估仍未验收。

启动时先重新核对MY_DFS、源码同步、GPU/Ray和dguard，再运行：

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
export RAY_ADDRESS=<verified-ray-address>
bash "$MY_DFS/work/carrot/recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/run.sh"
```

用户明确要求从官方base重新训练，已撤销先前step1000续训方案。
按用户要求，复制官方base到`$MY_DFS/hf-hub/Physical-Intelligence/pi05_base_pytorch_h10`，
仅修改副本`config.json`的`action_horizon: 10`。原base及复制后的权重不修改。
已删除新增recipe `train.py`；`run.sh`恢复普通`python -m carrot.cli.train_sft`入口，
不传`--resume`，训练loop/loss/随机noise和t保持生产实现。新checkpoint按已合入fix导出BF16。

评估计划：用正式`create_so101_policy → infer → sample_actions`，eager cache / 10NFE，
每个checkpoint在264唯一训练观测各测一次，固定每帧各自的noise以与step1000配对，
额外跨noise探测单列；首1/5/10步分别报告MAE/P95/max及最差帧，排除episode padding。
先前独立1000-step训练的BF16基线（非本轮新step1000）全h10五轴MAE为`[0.486372,1.191866,0.562415,0.813097,0.078097]°`。
本目录原`evaluate.py`/`diagnose.py`/`serve.sh`保留历史两episode/h50与旧模型设置，
不作为本轮评估/服务入口；本轮完整episode评估协议见
[已完成的BF16重评](../pi05_sft_so101_fit_validation/README.md)。

现有`reevaluate.py`增加`--step`/`--paired-eval`，支持本轮五个checkpoint；已导出BF16时直接加载，
不重新导出。脚本SHA256 `50280f21e8b4ca2c9d24cf42c071f633b2c777e3beb5ac42f85263346a86ca22`，
这项评估扩展及启动记录是提交后的修改；准备时同步、Ruff/compile/CLI检查PASS并保存快照，本轮新checkpoint评估已完成，实际证据见上文。
训练完成并确认GPU空闲后，逐step执行，输出分别用`$MONITOR/evaluations/step1000`等独立目录：

```bash
bash recipes/pi05_sft_so101_fit_validation/reevaluate.sh \
  --case-dir "$MY_DFS/work/carrot/recipes/pi05_sft_so101_knock_down_the_cylinder_overfit" \
  --step 1000 \
  --paired-eval "$MY_DFS/experiments/carrot/pi05_so101_fit_validation/20261002T183105Z/episode_h10/evaluation/step1000" \
  --output "$MONITOR/evaluations/step1000" \
  --source-commit 27bedad2148018df38aa3d0c6ba87f6bd1b21664
```

之后逐个运行既有`verify_reevaluation.py`，核对原Parquet、padding、配对noise及复算MAE/P95/max。
1000/2000/3000/4000/5000沿用相同协议，旧1000-step基线单列，不作为本轮新step1000结果。

验收分别核对官方base初始化/step0、新optimizer/scheduler、100step warmup、
训练结束step5000/exit0、5000条有限训练指标/LR、五个BF16 checkpoint和stats，再独立复算评估预测。
本轮新训练与五checkpoint评估均已完成，独立核验通过；`so101`自动检查已暂停。
不进行机器人动作，不恢复已取消的简化实验。

此前已撤销的resume方案准备检查（不作为当前base初始化证据）：Gemini32434904实际检测当前MY_DFS，同步后train.yaml/run.sh SHA256与Mac一致；
`/opt/venvs/carrot`下`SFTConfig`解析通过，源checkpoint架构h10/dim32、step与scheduler均1000，
optimizer DCP metadata、tokenizer和dataset完整，新输出目录未存在。
CPU scheduler恢复原状态后逐步检查1001–5000，LR始终1e-6；bash语法和git diff检查通过。
本次没有执行真实DCP恢复、训练、GPU评估或机器人操作，不能把配置检查当续训通过。

此前已撤销的自定义worker方案准备检查：Gemini32434904重新检测MY_DFS，train.yaml/run.sh/train.py
三文件Mac/GPU SHA256一致，Ruff/compile/bash/diff检查PASS。
CPU契约检查用生产`Pi05SFTLossFn.prepare_inputs`确认未对齐h50会拒绝h10 batch，
隔离GPU/FSDP初始化后执行真实recipe setup hook，h10 batch通过，权重、loss对象和step0不变。
新scheduler逐步检查step1–99为warmup，100–5000保持1e-6；官方base资源/config存在。
这是CPU配置/契约检查，未完整重载3B base或执行真实训练，不能当成GPU启动通过。

当前简化方案已落实：Gemini task`32434904-0278`复制官方checkpoint到上述h10副本，exit0/30s，
只修改根`config.json`的horizon。权重SHA256两份一致：
`df7352a32146db5dc72708cc70932cf7ff79f222769dce5a80dc4b23616cb974`，文件为独立副本；
原config仍h50，其它checkpoint文件内容核对一致，普通SFT配置解析和bash/diff检查PASS。
已确认新增`train.py`在Mac/GPU均删除，train.yaml/run.sh两端hash一致，模型与数据均h10。
复制日志：`MY_DFS/experiments/carrot/pi05_so101_fit_validation/base_h10_copy_20261003.log`，
任务已归档释放；没有启动训练或GPU评估。

## 2026-09-30：数据直接传递（尚未启动新训练）

- 任务为knock_down_the_cylinder，录制prompt `Move an object` 保留。
- 数据、统计量、策略请求/响应、日志和下发直接使用源数值，不添加语义字段或校验。
  模型原有q01/q99 Normalize/Unnormalize保留，不额外缩放或裁剪训练目标。
- `train.yaml` 显式单腕相机字段，factory相机key默认None，直接读取dataset metadata stats。
- 新输出目录为 `pi05_sft_so101_knock_down_the_cylinder_overfit_100`，尚未运行。
  历史step100、诊断路径和指标保留原名，不覆盖旧模型或数据。
- `serve.sh` 加载历史step100；`diagnose.py`、`evaluate.py` 直接比较源数值。
  本轮未执行GPU推理、重启服务或操作真机。
- 本地数据目录为 `$MY_DFS/hf-hub/nrailg/so101_knock_down_the_cylinder`；Mac缓存同名。
- 以下已完成实验和测试保留当时的记录，当前实现以本节为准。

## 2026-09-30：训练输入与前 5 帧动作诊断（完成）

- 目的：确认同一录制观测能否复现示教动作，区分预测误差与部署输入处理问题。
- 模型：已有 step100 policy server，`ws://127.0.0.1:8080`；不启动训练、不操作机器人。
- 数据：原始推倒圆柱两条 episode、354 帧、单腕 RGB、15 FPS、六维绝对目标。
- 审计：checkpoint/dataset 全部统计量、关节顺序、帧时间戳、50步动作对齐及
  episode padding；逐帧比较训练与客户端入口处理后的 state、图像、mask、tokens。
- 推理：354个训练观测每个请求一次，服务使用随机噪声；统计首1/5/50帧的六轴
  MAE、P95和bias，并重算原有32个固定噪声观测的相同窗口指标。
- 入口：`diagnose.py`、`diagnose.sh`；独立输出，不覆盖此前评估。
- 命令：确认 `MY_DFS` 后设置
  `SO101_DIAG_OUTPUT="$MY_DFS/experiments/carrot/pi05_sft_so101_wipe_overfit_100/diagnostics/training_inputs_20260930T111750Z"`，
  运行 `bash recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/diagnose.sh`。
- 验收：退出0，354个唯一观测与预测、两条episode、有效padding统计正确；input_audit.json、
  metrics.json、all_predictions.npz与逐帧NPZ齐全。这里只评估训练观测，不验收闭环任务成功率。
- 本地源码 d16d29a 加未提交诊断脚本；实际镜像tag尚未记录。运行结果在本节续写。

### 实际结果

- 成功任务 `56253f33-0034`，退出0、耗时166秒；补充重复抽样任务 `56253f33-0037`，
  退出0、耗时56秒。共354个唯一训练观测（198/156）、首5步1750个有效目标、
  全50步15250个有效目标；另选两条episode首帧和四个分散异常帧，各重复8次，共48次推理。
- 输出在以上目录加 `_retry1`。首轮 `56253f33-0032` 因诊断脚本误用
  `pyarrow.parquet.concat_tables` 在推理前退出，已修正为 `pyarrow.concat_tables`；
  重复抽样初轮 `56253f33-0036` 因直接序列化Tensor在请求发送前失败，改为显式NumPy请求。
  两轮失败日志保留，不计入成功样本。
- checkpoint/dataset 20组统计量逐项最大差为0；全部354帧训练与客户端处理后的
  图像、state、mask、tokens完全一致；动作时间窗口/padding/关节顺序通过，
  timestamp最大偏差4.45e-7秒。OpenPI loader明确pi05=True、离散state输入开启；
  导出config只存架构字段，不直接读取不存在的discrete_state_input配置项。

| 关节 | 首1步MAE | 首5步MAE | 首5步P95绝对误差 |
|---|---:|---:|---:|
| shoulder_pan | 1.926° | 2.036° | 9.408° |
| shoulder_lift | 3.761° | 3.970° | 11.942° |
| elbow_flex | 2.795° | 2.990° | 9.133° |
| wrist_flex | 1.607° | 1.717° | 6.580° |
| wrist_roll | 0.445° | 0.465° | 2.134° |
| gripper | 0.00994百分点 | 0.01000百分点 | 0.06390百分点 |

- 首5步最大单点肩/肘误差43.310°/28.001°；超过10°的比例分别8.743%/4.114%。
  总体轨迹接近示教，但转折处预测仍明显滞后且存在反向首目标。
- 各重复8次：episode0 frame46肩/肘首5步MAE均值17.308°/10.731°；
  episode1 frame32为15.674°/15.119°；两条episode首帧肩约2.0°/2.3°。
  episode1 frame103示教要求肩-59.824→-69.099°、肘46.549→52.264°，
  重复首目标肩7/8、肘8/8方向相反。随机性影响误差，但异常并非仅一个随机样本。
- 结论范围：同训练输入的统计/预处理错位未发现；模型在训练数据转折处仍不能可靠复现，
  100step尚不能验收充分拟合。不是闭环评测，也不能由此确定唯一根因。
  建议固定失败帧/噪声作为回归集，诊断拟合与条件歧义后再决定训练预算；节拍问题另行修复。
- 独立读回核对354份逐帧NPZ、汇总shape、唯一键、有效目标数，重算MAE/P95一致；
  11份本地/远端相关源码SHA256一致。Ruff、bash语法、py_compile、diff检查通过，
  另有窗口/padding及Parquet API轻量检查通过；没有重跑无关测试。
- provenance.json记录Python3.12.13、Torch2.11.0+cu128、NumPy2.3.1、LeRobot0.6.1、
  PyArrow23.0.1、Transformers5.5.4、OpenPI client0.1.0；实际Docker tag未记录。
- 本地结果/曲线副本：`/Users/wujunyu/.cache/carrot/so101_model_diagnosis_20260930T111750Z/`。
  包含metrics/input_audit、all_predictions/repeat_probes及肩肘/六轴/异常块曲线。
- 本轮无串口/相机访问、无运动、无追加训练、无commit。GPU0 policy服务健康，
  GPU1–7原有matmul保持；dguard短暂停10分钟后已自动恢复，收尾DGUARD_WATCH=1。

## 目的与配置

在用户录制的两条 episode 上做小数据拟合。100 step 是本轮预算，不预先宣称足够过拟合。

- 当前本地数据 `${MY_DFS}/hf-hub/nrailg/so101_knock_down_the_cylinder`，2 episodes，354 帧（198/156），15 FPS。
- 唯一相机 `observation.images.wrist`，映射模型左腕槽；缺失 base/right_wrist 的 mask=false。
- robot_type=so_follower；沿用数据任务文本 `Move an object`，不改标注或单位。
- 用户确认录制日志 robot.use_degrees=True、teleop.use_degrees=True：前五轴度数，第六轴夹爪开合百分比。样本、统计量和部署均保留这些单位。下方历史100step训练仍使用原单位。
- 数据夹爪state全程0.549073，action约0.244..0.407；记录此数据限制，不改数值。
- 初始化 `${MY_DFS}/hf-hub/Physical-Intelligence/pi05_base_pytorch`，不使用旧 orange-cube SFT 权重。
- horizon=50，15FPS目标时距；8 GPU BF16 FSDP，micro batch4，global batch64，GAS2。
- AdamW LR=1e-4 恒定，100 step，log每步，save每25步，W&B关闭。
- 约6400个样本窗口，相当于354帧约18遍抽样；分布式drop_last使实际epoch计数略有不同。
- 数据无固定Hub revision，使用用户指定本地副本，记录元信息和parquet哈希，不隐式下载。

## 运行

确认GPU/Ray空闲和dguard后，在既定venv及源码环境中：

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
export RAY_ADDRESS=<verified-ray-address>
bash "$MY_DFS/work/carrot/recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/run.sh"
```

当前新输出目录 `${MY_DFS}/experiments/carrot/pi05_sft_so101_knock_down_the_cylinder_overfit_100`；不覆盖已有目录。
预期 checkpoints/step-00000025、50、75、100，保存权重、统计量、tokenizer与训练状态。

## 验收

- 退出0、有SFT finished/step100、loss及梯度有限、四个checkpoint文件完整。
- 训练前后对每episode均匀抽取16帧（共32个固定训练观测）、相同噪声做50步动作预测，排除episode padding，记录六关节MAE、归一化RMSE及state保持基线。
- 样本拟合、训练完成、模型可重载分别记录；无留出集，不据此宣称泛化或真机任务成功。

## 2026-09-29

状态：100-step训练及固定训练样本推理评估已有完整日志和产物；详见下方结果。

### 启动记录

- 2026-09-29 11:30:38 UTC（北京时间19:30:38）通过现有空闲Ray启动；后台task `2f6f5af6-0264`。
- 预留40 CPU/8 H20；dguard暂停30分钟，未重启Ray或停止其他实验。
- 源码基于 `d0af86db8e93c400d1170b5a30828eaa6971d246` 加本次单相机/robot_type兼容与recipe改动，启动前同步文件SHA256一致。
- 前置SO101 CPU测试5 passed；真实factory首尾样本和视角mask验证通过。
- 持久监控目录 `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_sft_so101_wipe_overfit_monitor/20260929T1129Z`，CPU日志 `cpu_tests.log`。
- 此记录仅证明任务启动，训练完成和拟合效果待下节实际结果。


### 2026-09-29 20:31 北京时间结果核查

- 持久训练日志 `train_backend.log` 有 `SFT finished: step=100`；rank0记录step1..100共100条。
- 首步loss=0.137848，末步=0.009677887；前10步均值0.0777427，后10步均值0.0115624。
- step25/50/75/100四个checkpoint均有权重、config、norm_stats、tokenizer_config和8份optimizer shard；step100已由独立评估成功加载。
- 训练前后使用同一32个训练观测、固定噪声，排除padding后1345行有效动作。

| 指标 | base | step100 |
|---|---:|---:|
| 动作范围归一化RMSE | 0.4661695 | 0.1795024 |
| shoulder_pan MAE（度） | 7.8626 | 2.2565 |
| shoulder_lift MAE（度） | 30.6453 | 8.4070 |
| elbow_flex MAE（度） | 22.9190 | 5.8599 |
| wrist_flex MAE（度） | 9.8191 | 2.6059 |
| wrist_roll MAE（度） | 2.2817 | 0.7456 |
| gripper MAE（数据中的百分比单位） | 0.10930 | 0.01785 |

- RMSE下降约61.5%；保持当前state基线RMSE=0.6059362。拟合明显改善，但未达到可宣称充分过拟合的证据；未延长超过用户指定的100step。
- 这是训练样本子集评估，不是留出集或真机任务成功率。
- 评估证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_sft_so101_wipe_overfit_100/evaluation/{base,step100}/metrics.json`；对应NPZ在相同目录。
- 完整训练、评估stdout见上述monitor目录的 `train_backend.log`、`rank0_worker_stdout.log`、`eval_base.log`、`eval_step100.log`。

### 2026-09-29 MacBook 联调服务

- 启动入口 `serve.sh`；`MY_DFS=<个人DFS> CUDA_VISIBLE_DEVICES=0 bash recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/serve.sh`。
- 2026-09-29约20:45北京时间最初启动8000端口task `1d7252fc-0307`；用户确认Mac仅可访问8080/8081后，该任务已停止。
- 20:49切换到8080，新后台task `1d7252fc-0313`，GPU0、host=0.0.0.0；serve.sh默认端口已改8080。
- launcher地址 `29.209.160.111`；WebSocket `ws://29.209.160.111:8080`，健康检查 `/healthz`。
- dguard暂停120分钟，计划22:45:48北京时间自动恢复；服务持续运行，恢复后可能竞争GPU资源。
- 模型step100；请求字段 observation/state（6维，前5轴度数、夹爪百分比）、observation/wrist_image（RGB uint8）、prompt（训练文本 Move an object）。省略不存在的外部相机，不伪造双相机。
- 输出绝对目标float32[50,6]，对应15FPS训练时间尺度。
- 当前部署使用单腕15FPS、use_degrees=true配置；历史诊断过程保留原结果。
- 主代理从devcloud访问该IP被IDC网关拒绝：HTTP403、acl.14001/acl_denied，说明需要igate权限；此结果不能判定MacBook的办公网路由是否可达。未调整网络权限。

- 20:51北京时间新服务加载完成，远端localhost和launcher IP的8080 `/healthz` 均返回200 OK；8000无监听。
- 20:51:50北京时间WebSocket真实单腕数据推理通过：metadata horizon50/dim6/num_steps10/so101，返回float32[50,6]全部有限，首次客户端往返534ms。证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_sft_so101_wipe_overfit_100/serving/20260929T1249Z/health_websocket_infer.log`。主代理已独立读取日志；服务保持运行。


## 2026-09-30：review清理验收

94项CPU/真实数据契约回归通过，Ruff/语法/diff检查通过，31个同步文件hash一致。部署字段dataset_revision移除；相机配置通过config对象传递；camera key默认None并在recipe/test YAML显式指定。单位转换包含状态、动作、统计量、起点比较和发送日志，旧checkpoint必须显式声明统计量单位，新checkpoint导出自带单位。数据目录两端真实重命名，文件内容未改。未commit、未新增训练/模型评估/真机动作，旧服务未重启。测试日志在个人DFS `test-runs/so101_review_cleanup_20260930/final_tests.log`。


## 2026-10-03：单帧数据选择器归档

- `single_frame.py`提供recipe专用dataset factory：
  `recipes.pi05_sft_so101_knock_down_the_cylinder_overfit.single_frame.build_dataset`。
  它调用原SO101 factory后用Subset重复指定源帧，保留原action窗口、metadata与统计量。
- kwargs中的`frame_index=53`、`repeat_count=64`可为8rank/micro4/GAS2提供完整batch；
  其他数据/模型参数应来自对应实验的实际配置，不能将本README历史两episode配置当作新实验配置。
- 越界frame和非正repeat_count直接失败，不回退到全量训练。
- 测试三件套`tests/pi_05/test_so101_single_frame_recipe.{py,sh,md}`，
  2026-10-03提交前真实回归4/4 PASS；运行记录见测试md。
- 最新264帧/horizon10/去视觉/固定noise的完整实验及精度审计在
  [SO101逐步拟合验证](../pi05_sft_so101_fit_validation/README.md)，本目录历史产物保留。
