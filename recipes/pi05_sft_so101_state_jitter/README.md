# SO101无视觉：输入state ±3°增强

## 2026-10-08：no vision + clamped jitter，2000 steps（完成）

| 项目 | 本轮设置/状态 |
|---|---|
| 目标 | 修复输入越界并统一标定来源后，重新训练无视觉模型；结果与旧2000step严格复现 |
| 初始化 | 官方 pi05_base_pytorch_h10，从 step0 新训，不 resume |
| 数据 | 原始单 episode 264 帧；horizon10；五轴每次独立 U(-3°,3°)，按 LeRobot 标定 clip |
| 视觉/noise | 所有视觉 pixels0/maskfalse；训练 noise/t 始终随机 |
| 预算 | 2000 steps、warmup100 后 constant1e-6；8GPU BF16 FSDP、micro4/global64/GAS2、seed1000 |
| 保存/评估 | 每500保存；训练完成后自动用最终step2000跑既有h10/K5、8noise×53chunk离线评估及Parquet核验 |
| 输出 | $MY_DFS/experiments/carrot/pi05_so101_state_jitter/20261008T093114Z |
| 状态 | 2026-10-08 17:36:22 Asia/Shanghai 启动；19:14起核验task exit0/SFT finished2000；2000条连续有限指标，warmup/constant正确；step2000 loss0.006175/grad_norm0.132241，末20步平均loss0.007045；最终评估/独立核验PASS |
| 源码/任务 | Mac提交133357115cc0820abf24c96a5b0dba87f0956a49；96文件SHA一致；Gemini b1c4b5e1 / task b1c4b5e1-0526 / Ray job13000000；Docker image tag 未记录 |

运行：确认当前MY_DFS与RAY_ADDRESS，设置SOURCE_COMMIT为本轮Mac实际提交，然后执行
`bash recipes/pi05_sft_so101_state_jitter/run.sh`。runner拒绝覆盖已有训练/评估目录，
保存实际train.yaml/calibration.json；dguard暂停120分钟，退出恢复。当前训练只对输入state
做增强clamp；policy输出保持原样，机器人执行限幅不参与该离线评估。验收训练2000连续有限
指标、step2000/checkpoint完整与参数更新；评估需核对NPZ/reference/padding/noise及复算指标。
新结果不能替代下方历史500step结果，也不能单次证明输入clamp的独立因果收益；不操作机器人。

本轮h10权重SHA与官方base相同；已归档preflight.json、expected_source_hashes.json、source_snapshot、launch.json及ray_logs。
rank0日志为`/tmp/ray/session_latest/logs/worker-604ba4b757cf8bfdda37a465a0a4ad6c50529e3e2ace9605a666d29b-13000000-555994.out`，
输出根monitor.py可持久化training_metrics.csv/latest_training_status.json并检查连续性、有限性和LR。
结束已亲核dguard watch1/guard.sh与run.py运行，无恢复计划，Ray空闲224CPU/8GPU。Ray dashboard未开放；状态以后台task及worker日志核对，不重启Ray。
本轮运行源码为1333571；以下记录在训练结束后归档，保持训练源码与结果记录可区分。

### 1000→2000 steps的进展判断

| 五运动轴平均MAE（degrees） | 历史增强1000step | 本轮增强2000step | 降幅 |
|---|---:|---:|---:|
| 真实state输入 | 1.391723 | 1.191509 | 14.39% |
| 连续action→state反馈 | 14.209092 | 10.313210 | 27.42% |

1000step来自[历史匹配拟合实验](../pi05_sft_so101_state_jitter_matched_fit/README.md)的step1000，
2000step来自本轮实际训练/评估；本轮已独立证明与同一历史实验的step2000权重及预测严格一致。
两者均为no vision+±3° jitter、完整264帧、h10/K5/NFE10和配对noise评估。
在这组实验中，延长训练同时改善了基础拟合和连续反馈，反馈MAE降幅更大；
这一观察不等同于已验证反馈稳定性，反馈仍约10.31°，且存在尾部大误差和预测越界。
改善不能归因于新增input clamp，因为当前离散state condition及新旧2000step结果没有改变；
也不能单凭这一条episode、一个训练seed证明jitter的独立收益或泛化能力。

### 本轮最终结果与重复性核验

| step2000评估模式 | 五运动轴平均MAE（degrees） | 有效action行 | 原始预测越界行 |
|---|---:|---:|---:|
| 每块输入数据集真实state | 1.191509 | 2112 | 80（3.79%） |
| 连续action→state反馈 | 10.313210 | 2112 | 109（5.16%） |

共同协议：无视觉，horizon10/NFE10/K5；8条noise序列×53块，424NPZ/848次推理。
下一state精确等于上一块采用index4；只统计有效采用动作，gripper不计入五轴degrees平均。
真实state五轴MAE=[1.068920,1.975627,1.198843,1.537592,0.176563]°；
反馈五轴MAE=[6.203267,21.274952,8.119689,15.537370,0.430771]°。
反馈肩抬P95/max=67.460030/90.218208°，仍有明显轨迹偏离；不是每块单调增大，也不是真机结果。
原始预测最大越界幅度为真实state肩抬2.616570°/肘0.707642°，反馈肩抬1.690331°/肘1.078781°；
统计使用本轮标定。离线输出/反馈没有执行限幅，不能当作启用robot clamp后的实际表现。

本轮step2000与20261004_2000step旧未clamp增强实验的model.safetensors/config/norm_stats SHA一致，
424NPZ的全部key数组逐元素相同。因此这些结果是本次重新训练/评估的真实复现，没有观察到输入clamp带来的效果变化。
检查64×264=16896个增强state：1906行raw角度因clamp改变，state离散桶差异为0。
当前五轴标定下界经quantile归一化均<-1，上界均>最后桶边界0.9921875；
所以被clamp的边界外数值原本已落入相同端点桶。PI0.5的state经prompt离散输入，
embed_suffix仅非pi05分支使用连续state投影。这解释了为什么原始角度限幅有效、模型condition仍相同；
它不约束预测action，也不能据此断言jitter普遍无效。旧无增强500step基线1.346750°/12.450586°
训练预算不同，不能据新2000结果把差异单独归因于增强或clamp。

完成证据：四checkpoint（500/1000/1500/2000）actualstep/h10/812参数全BF16，stats/tokenizer及
8份optimizer shard完整；四组权重hash不同、8rank action head有限非零更新。
原Parquet逐元素核对reference/state/padding/424不同noise/index4反馈，MAE/P95/max复算PASS；
Mac回拉424NPZ再次NumPy复算PASS。详见输出根completion_audit.py/json、evaluation/independent_validation.json、
training_metrics.csv和training_evaluation_backend.log（后台task已dump释放）。
Mac缓存：`~/.cache/carrot/so101_state_jitter_20261008T093114Z/`，包含轻量结果与预测，无checkpoint权重。
本节为提交1333571之后的实验记录；无机器人操作，生成物不入Git。


## 2026-10-08：训练增强改读 LeRobot 标定文件（未重训）

已移除五轴 bounds 数值，三个 jitter recipe 统一通过 state_jitter_calibration_path
读取同一份 LeRobot 标定 JSON，与 robot 执行共用 SDK 换算。路径与同步约定见
[标定来源说明](../pi05_sft_so101_state_only_rollout/README.md)。历史训练结果不变，未重跑实验。


2026-10-04已完成。这轮训练state增强没有降低整体连续反馈误差：五轴平均MAE由12.450586°增至14.705301°（+18.109%）；真实state输入也由1.346750°增至1.656161°（+22.975%）。中段局部改善、后段偏离更大；不能用放大倍率9.245→8.879的下降宣称改善，因为基础误差也增加了。

| 检查 | 完成证据 |
|---|---|
| 契约/preflight | 2测试PASS；75文件初始SHA同步；base/stats/视觉屏蔽/32次真实扰动检查PASS |
| 训练 | 500条连续有限loss/grad/LR；warmup/constant核验；step500 loss0.010122、grad0.187971 |
| checkpoint | step500/h10/812参数全BF16、norm_stats/tokenizer/8optimizer shards完整；8rank参数有更新 |
| 推理 | 8noise序列×53段=424NPZ/848次推理，eval+verify task0387 exit0/469s |
| 独立核验 | 原Parquet/reference/state/padding/noise/精确index4反馈及MAE/P95/max复算PASS；Mac再次核验PASS |
| 配对比较 | 两轮原数据、424个noise、有效位置、干净输入逐元素一致；生产代码SHA与包版本一致 |
| 收尾 | 后台日志dump释放，dguard实际watch1/guard+run运行/无恢复计划；Ray空闲224CPU/8GPU |

## 实验条件

| 条件 | 本轮设置 |
|---|---|
| 初始化/数据 | 同官方h10 base，step0，不resume；同完整1episode/264帧 |
| 训练 | 500step/warmup100/constant1e-6，8GPU BF16 FSDP，micro4/global64/GAS2，seed1000 |
| 唯一训练变化 | 每次读取每个运动轴独立U(-3°,3°)，原始degrees；Normalize/TokenizePrompt之前 |
| 不扰动的字段 | gripper、示教action、padding、图像、任务；norm_stats沿用原数据 |
| 视觉/noise | 训练和推理全部视觉mask=false/pixels0；训练noise/t仍随机 |
| 入口 | 既有fit_validation/train.py、FitTrainWorker；增强仅recipe dataset wrapper，不新增train.py |
| 评估 | 复用state_only_rollout/evaluate.py与verify.py；干净输入，生产BF16/eager缓存/NFE10/h10，只采用前5帧 |
| 反馈 | 下一state精确取上一段采用的index4反归一化动作；起点真实state，后续不重置 |
| 对照与配对 | 每段真实state vs连续预测state，8noise序列×53段；与无增强旧轮同seed/noise/reference/padding |
| 统计 | 每模式2112有效采用action行；五轴反归一化degrees的MAE/P95/max，gripper原始单位另列 |
| 输出 | $MY_DFS/experiments/carrot/pi05_so101_state_jitter/20261004T033515Z |

这是理想位置跟踪下的离线state反馈实验，没有视觉condition、舵机动力学或真实环境反馈。
原数据下一观测与上一示教动作本身存在跟踪差异；直接action当state也改变条件分布。
两组各只有一次500step训练，8条noise序列属于推理重复，不是8次训练；本轮仅说明这一预算/数据/增强幅度下未见整体收益，不能证明state增强普遍无效，也不能单独归因于diffusion数值误差或BUG。
±3°附近局部扰动、state-only条件歧义及较大偏离后的行为，都没有通过本实验单独区分。用户休假期间真机待执行。

## 为什么误差会变大：证据与解释

这里的比较是两次各500step的无视觉新训练：相同官方base、h10、264帧episode、优化器/LR/warmup/seed配置；唯一有意改变的是训练输入state是否加五轴独立U(-3°,3°)，评估noise/ref/state完全配对。它不是增强训练多跑了500step，也不是拿它和含视觉5000step模型比。

**直接观察到的证据：**增强轮的训练目标更难拟合。末20步训练loss均值为0.009779，对照为0.004589；全500步均值0.028136，对照0.020394。干净真实state评估MAE同时由1.346750°增至1.656161°。所以反馈MAE增加的一个直接组成因素是基础策略本身在同一训练预算下拟合得更差。图中的loss/grad只是训练指标，不能单独证明因果，但与干净state评估方向一致。

**与反馈结果一致、但尚未被单独验证的解释：**

1. 训练把输入state挪动，却保留对应的示教action target。对小扰动，这鼓励模型对state变化输出不变；但如果实际偏差要求策略执行纠正动作，同一个target就没有教模型怎样恢复。因此这种增强可能学到局部不敏感，却没有学到闭环恢复。
2. 注入的是逐轴、逐样本独立、零均值的±3°扰动。反馈偏离可能沿时间相关，并由延迟、伺服跟踪误差或前一段动作造成；随着chunk推进，图中的肩/肘/腕偏差也可远大于3°。这两种扰动分布并不匹配，局部噪声增强不保证能覆盖较远的反馈状态。
3. 无视觉且只有一个episode时，关节state未必能唯一表示当前任务阶段或期望后续动作。改变state到示教轨迹外，还可能暴露这种条件歧义；单episode示教target也没有为偏离后的状态提供纠正标签。
4. 离线测试把模型输出的action直接当下一时刻的state，是理想位置跟踪假设。示教action与下一观测本就不完全一致，所以该递推不等同于真实机械臂转移。结果证明这个指定的离线反馈协议误差较大，不能据此断定真机误差幅度相同。

图中的反馈误差随chunk并非单调增加：例如无增强在frame100附近较高、之后下降再上升；增强轮在一些中段chunk更好、末段更差。因此“误差变大”指全episode汇总和末端更差，不是每一段都比上一段更差。五轴反馈MAE分别为[5.624,31.883,8.966,15.273,0.506]°和[7.300,33.350,14.646,17.751,0.478]°；主要退化在肩旋、肩抬、肘、腕屈，腕转略好。8条推理noise序列只有一次训练的重复采样，不是8个独立模型，不能估计训练seed方差。

**结论强度：**实测支持“这次±3°输入抖动在500step预算下提高了训练难度，且没有降低本协议的整段反馈MAE”。它不证明抖动增强总是无效，也未定位偏差由监督target不一致、转移分布不匹配、state-only歧义或理想state转移中的哪一项主导。尚不能把反馈放大归结为diffusion数值误差或生产代码BUG。

## 配对结果

以下平均MAE先对每个运动轴的全部有效采用动作求绝对误差平均，再对五轴等权平均；padding与未采用的后5步排除。

| 输入模式 | 无增强 | 训练±3°增强 | 变化 |
|---|---:|---:|---:|
| 每段真实state | 1.346750° | 1.656161° | +0.309412° / +22.975% |
| 连续预测state反馈 | 12.450586° | 14.705301° | +2.254715° / +18.109% |
| 反馈/真实state倍率 | 9.244915 | 8.879148 | 基础MAE增加，不能单看倍率 |

五轴顺序：肩旋、肩抬、肘、腕屈、腕转。gripper不是degrees。

| 模型/输入 | 五轴MAE (°) | 五轴P95 (°) | 五轴max (°) | gripper MAE / P95 / max |
|---|---|---|---|---|
| 无增强/真实state | [1.089999, 2.495712, 1.408912, 1.590769, 0.148356] | [4.294232, 6.723961, 4.241116, 3.985218, 0.423907] | [9.359035, 15.519402, 12.439972, 8.867929, 2.238647] | 0.018696 / 0.057847 / 0.143405 |
| 无增强/反馈state | [5.624303, 31.883308, 8.966387, 15.273020, 0.505911] | [25.749317, 79.218979, 22.914099, 57.191273, 2.363657] | [37.776077, 93.270340, 35.977585, 64.129059, 4.072813] | 0.058257 / 0.169359 / 0.208991 |
| ±3°增强/真实state | [1.483450, 2.782825, 1.677512, 2.118152, 0.218868] | [5.005030, 7.666819, 4.900407, 5.538103, 0.596311] | [17.886923, 17.307716, 17.996334, 21.663740, 3.305909] | 0.018323 / 0.067592 / 0.124711 |
| ±3°增强/反馈state | [7.300444, 33.350147, 14.646380, 17.751417, 0.478116] | [22.445984, 85.781090, 36.322098, 62.304501, 1.211482] | [34.263714, 94.382690, 41.664818, 67.078415, 4.620028] | 0.074887 / 0.213680 / 0.261867 |

逐段示例是8条noise序列的五轴平均MAE；frame是示教帧编号，每次采用5帧，末段只有4帧。

| 采用区间 | 无增强真实state | 增强真实state | 无增强反馈 | 增强反馈 |
|---|---:|---:|---:|---:|
| [0,5) | 1.487713° | 2.380179° | 1.487713° | 2.380179° |
| [5,10) | 1.317577° | 1.514876° | 3.599570° | 5.500283° |
| [10,15) | 1.114768° | 1.313459° | 5.581170° | 8.775607° |
| [100,105) | 3.624696° | 3.651296° | 20.433855° | 11.393012° |
| [260,264) | 1.170744° | 1.139190° | 14.630211° | 25.472925° |

增强模型8条反馈序列整体MAE分别约[15.604,15.436,17.531,15.643,15.859,16.475,15.407,5.687]°，存在明显noise敏感性。曲线阴影是8序列P10–P90，不是置信区间；x轴为采用chunk起点的示教frame。部分中段改善不能替代全episode统计。

## 运行与产物

实际MY_DFS由当前Gemini挂载重新确认：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu`。
11:40 UTC+8启动，session cb79ec05/Ray job07000000；训练任务0367完成500step后，eval源码快照对相对case路径使用relative_to时报错，组合wrapper exit1，尚未生成任何预测。修复只涉及case路径resolve；未重训、未resume或增加训练预算。保留evaluation_initial_path_failure与training_and_initial_evaluation_backend.log。
随后绝对--case-dir单独启动0387，完成全部评估和validator exit0。checkpoint audit0382 exit0，全部训练日志、CSV与证据归档。

代码在用户要求后先commit，当前重写后的Git为6a893b8（增强），02d36da（共享评估/连续反馈实验）；只提交文本，无图片，未push。启动快照来自原71b1144加本轮working tree，原654a74b/c668ff3等历史已被共享Git整理；运行期间源码/配置hash与快照核对，数学代码没变。evaluation的source-commit标签f53155d与当前Git历史不同，实际执行以provenance/source_snapshot/SHA为准，路径修复另存evaluation_repair.json。镜像tag未记录；包版本在provenance。

输出根保存training/checkpoints/step-00000500、training_metrics.csv、ray_logs、preflight、源码快照、checkpoint_validation.json；evaluation保存424NPZ、metrics/provenance/independent_validation.json、paired_comparison.json/paired_chunk_metrics.csv，以及配对误差/动作轨迹PNG/SVG/PDF。paired_training_curves含两轮loss、clip前grad与LR，淡线逐step、粗线20step尾随均值。
Mac副本：`~/.cache/carrot/so101_state_jitter_20261004T033515Z/`，不含大模型权重，含mac_validation.json；全部NPZ/reference/noise/反馈再验PASS，独立FP64均值与报告FP32归约最大差2.1222e-5°，源数据和反馈仍逐元素精确。图片只留Ceph/Mac缓存，不入Git。

参考：robomimic官方观测noise randomizer；DART在采集时加扰动并记录纠正动作，与本轮离线仅抖state/保持target不同，不能用其收益推定本实验有效。结论以本轮配对结果为准。

## 2026-10-04 后续记录：保守解读

后续增强2000step及500/1000/1500/2000评估已完成，见 [匹配拟合实验记录](../pi05_sft_so101_state_jitter_matched_fit/README.md)。增强模型较晚checkpoint的真实state和反馈平均MAE均下降，1500/2000的反馈指标也低于旧无增强500对照。因此，本文500step下的退化现象只能作为这次固定预算的观察，不能推广为增强的一般效果。

按用户要求，整个增强比较统一记为探索性实验，尚不能判断增强对模型鲁棒性的独立收益。本文关于基础拟合、监督与状态转移的解释仍是未隔离验证的可能原因；不同输入分布下的训练loss，以及相近的真实state平均MAE，都不足以完成因果归因。保留原始结果和解释的历史记录，不据此决定采用或放弃增强。

> 2026-10-08：当前共享augmentation/config已增加五轴标定bounds，raw state加噪后clip；
> action/gripper/stats不变。CPU契约测试8 passed。本文历史运行使用未clamp的旧版本，未重训。
