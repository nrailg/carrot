# SO101无视觉模型：连续action chunk反馈测试

## 2026-10-08：训练增强改读 LeRobot 标定文件（未重训）

三个 jitter recipe 已删除重复的五轴 `state_jitter_bounds` 数值，统一使用：

```yaml
state_jitter_calibration_path: /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json
```

远端目录统一为 `$MY_DFS/.cache/huggingface/lerobot/calibration/`，对应 Mac 的
`~/.cache/huggingface/lerobot/calibration/`，包含 robots/teleoperators/backups。
该文件是 Mac 现有 LeRobot 标定 JSON 的共享副本，SHA256
`68ba463158a5bfde211bb2633d552ba13790c17be378d2e657338990e3bf1243`。
GPU 无法直接读取 Mac 缓存，标定变更后需同步这一文件；不再逐个修改 YAML 的角度表。
`examples/so101_real/actions.py` 的 `load_action_limits()` 实例化 SDK 但不 connect，
与机器人执行共用 `action_limits()`；五轴按 degrees 换算、float32 边界向内取整，
增强后 clip，action/gripper/stats 保持原值。缺文件立即失败，不生成默认标定。
评估读取干净 state，记录配置中的路径、解析后的范围及标定 SHA，并保存 calibration.json；
这些是当前配置来源，不能据此声称下方旧 checkpoint 曾使用输入 clamp。
验证：增强契约10项、相关回归86项（合计96）通过，Ruff通过；实际264帧数据工厂加载及
Mac/GPU标定SHA核对通过，证据见`tests/pi_05/test_so101_state_jitter.md`。
本轮没有新训练、模型评估或真机动作；历史结果和训练快照保持原值。


## 2026-10-08：保留原始数据，按实测范围小幅扩肩抬标定

用户改变方案：不采用清洗数据，检查实际state/action后小幅扩机器人range_min/max。
原始单episode264帧的肩抬action最小−105.142860°，原范围外8帧；30episode3213帧最小−105.758240°，原范围外521帧。
两份数据的state及其余四个运动轴action均无越界。只将follower肩抬raw范围[932,3317]对称扩为[916,3333]，
degrees范围±104.835165°→±106.241758°，每侧增加1.406593°，覆盖最极端原action并留约0.4835°。
原始数据、stats和旧checkpoint不改；当前recipe已恢复指向原始单episode，不使用生成的`_clamped`副本。
三个jitter recipe与契约测试中的肩抬bounds同步更新，其余关节/gripper/homing_offset均不改。

已通过bus-only连接，只解锁肩抬EEPROM、写Min_Position_Limit/Max_Position_Limit并恢复Lock；
未调用Robot.connect/configure、未写Goal或Torque。六轴读回证实Goal/Torque/Lock/homing/其余limits和Status均与写前相同，
设备范围与保存文件一致；Status全0。备份/脚本/前后寄存器与SHA在Mac `~/.cache/carrot/so101_range_expand_20261008/`。
原标定SHA9bb12feb1f18ff03e4f6298e6655fd0a0f5f68d68409f0a5c79f06178d0ee65b，
新SHA68ba463158a5bfde211bb2633d552ba13790c17be378d2e657338990e3bf1243。
离线SDK枚举全部4096个肩抬raw值确认degrees不变；两份数据全部state/action的degrees→raw目标也逐项不变。
新范围下两份原数据state/action均无越界，原单episode6个文件SHA保持原值。没有动作轨迹或机械极限实测，
旧评估的133/41行越界等仍以旧标定为基准，不能视作新范围下统计。
新bounds版本Gemini CPU回归8 passed in7.20s，Ruff/7文件Mac-GPU同步SHA通过；Ceph原Parquet SHA未变。

## 2026-10-08：修复state jitter越出标定范围（未重训）

当时新增五轴`state_jitter_bounds`（现已改为标定文件路径），源自`my_awesome_follower_arm`现有标定：
±[112,104.835164835,97.274725275,96.615384615,180]°，顺序pan/lift/elbow/wrist flex/wrist roll。
共享augmentation在raw state加U(-3°,3°)后、Normalize/Tokenize之前clip五轴；gripper/action/padding/stats不变。
evaluate移除增强专用bounds再读取干净数据，并将bounds记入评估metadata。
Gemini CPU契约测试8 passed，Ruff/语法/7文件同步SHA通过，见`tests/pi_05/test_so101_state_jitter.md`。
**下方所有已完成训练及checkpoint使用未clamp的旧增强，本次只改代码/config，没有重训或重新评估模型。**

复查旧step1000的h10/K5真实state组：133/2112采用action行越界，其中108行对应五轴均合法的reference。
肩抬95个越界预测的reference全部距离下界≤2.8572°（部分reference本身越出0.3077°），
这95个预测对reference的MAE2.4918°；肘42个越界预测的reference距上界0.3516–1.0549°，MAE0.9932°。
说明边界附近的小幅拟合误差即可触发越界；不能仅由“有越界”推断feedback或增强是主因。
分析证据：Mac缓存本轮输出根`bounds_cause_observations.json`，使用旧424NPZ，无新推理。

现有flow-matching loss、NFE10 Euler和quantile反归一化没有关节范围硬约束；输入clamp不能保证输出合法。
原action略超当前标定的原因仍需核对录制/当前标定一致性，不能直接认定标签错误。
相关研究与方法取舍记录在canonical memory同名状态文档：DAgger/DART针对反馈分布偏移及纠正示教；
SafeDiffuser在生成过程加入约束。后续可考虑raw degrees终点投影并单列原始越界统计，或收集纠正示教；
这些输出/采样/数据采集改动均未执行，不能把clamp当作轨迹拟合或闭环恢复成功。

## 2026-10-08：无视觉 + state ±3°，重新训练1000步

按用户要求原地修改本recipe的`train.yaml`，复用既有state_jitter dataset wrapper，不增加训练实现。
本节是新运行；下方2026-10-04的无增强500步记录及其产物保持为历史结果。

| 项目 | 新运行设置/状态 |
|---|---|
| 目的 | 重新训练带state jitter的无视觉模型，供后续模型输出与真机执行排查；尚无效果结论 |
| 初始化/数据 | 官方pi05_base_pytorch_h10从step0开始，不resume；完整单episode/264帧 |
| 增强 | Normalize/TokenizePrompt之前，五运动轴各自独立U(-3°,3°)；action、gripper、padding、原统计量保持不变 |
| 训练 | 1000step，warmup100后constant1e-6，8×H20 BF16 FSDP，micro4/global64/GAS2，seed1000；每500保存 |
| 视觉/随机性 | 视觉pixels=0、mask=false；训练noise和t随机 |
| 输出根 | `$MY_DFS/experiments/carrot/pi05_so101_state_only_rollout/20261008T074452Z`，独立目录 |
| 进度 | 已完成1000step及424NPZ离线评估；训练/评估wrapper exit0，1000条指标连续有限，独立修正后收尾核验PASS；16:27已确认dguard实际恢复、GPU/Ray资源释放 |
| 后续离线评估 | 最终step1000复用evaluate.py/verify.py，8条noise序列×53chunk；真实state对照与预测末action反馈分别统计，NFE10/h10/采用5 |

实际MY_DFS已由本会话唯一CephFS挂载与`__SYS_USER_NAME__`核验。
命令：`MY_DFS=<已核验DFS> RAY_ADDRESS=29.209.160.111:6379 SOURCE_COMMIT=7650c4acf7cbff76b887a7f2cfc096645043f83a bash recipes/pi05_sft_so101_state_only_rollout/run.sh`。
源码基于该Mac提交加本轮未提交recipe改动，执行前归档源码快照和SHA；Docker image tag未记录。
run.sh从当前配置解析输出目录和最终step，避免误写历史evaluation；不操作机器人或启动推理服务。
验收以1000条连续有限loss/grad/LR、训练退出码、checkpoint实际step/BF16/h10/stats/参数更新及原Parquet独立评估核验为准。
预计纯训练约30分钟、含加载/保存/离线评估约40分钟，以实际日志为准；NaN/Inf或实际崩溃停止本轮，不自动重训。
历史matched-fit实验已有相同增强1000步结果；本次是相同seed配置重跑，不是独立训练seed的因果对照。
图片、模型、日志和临时诊断脚本留Ceph/Mac缓存，不入Git；启动时recipe改动尚未提交，执行版本以启动快照和SHA为准。

预检task `26a05933-0479` exit0/29s，已归档`preflight_backend.log`。
72文件Mac/GPU SHA一致，官方base和h10副本权重SHA相同，Parquet SHA与原录制相同；32次实际读取验证五轴扰动范围、action/gripper/padding不变、原统计量一致、视觉全屏蔽和action反归一化roundtrip通过。
dguard已暂停60分钟，EXIT恢复；后台supervise每1800秒归档本job日志、核验连续/有限指标与LR，异常停止本轮，不自动重训。
15:59按用户要求将当前对话`so101` heartbeat启用为每15分钟；运行输出根`monitor.py`亲核并归档本job指标，原后台30分钟保护检查保留。执行源码/config/script共71文件SHA仍与启动一致（进度README除外）。本轮recipe改动已提交`56b9b17`，回放记录另提交`eca46af`；启动版本仍以当时快照为准，未push。
启动时加载的收尾脚本误把checkpoint的根目录tokenizer文件当成`tokenizer/`目录；已保留`supervise_loaded_snapshot.py`并修正磁盘脚本，训练代码/已加载配置未变。
独立收尾task `26a05933-0483`等待训练/离线评估结束后使用正确文件位置复核；若0480因该旧检查exit1，须以failure.json中的训练评估wrapper退出码和独立completion.json区分，不能报告训练崩溃或重新训练。
输出根含preflight/source_snapshot/expected_source_hashes/launch/ray_logs/latest_training_status；原始执行快照与后续README进度更新分开保留。

## 2026-10-04：训练、连续反馈评估与独立核验完成

| 阶段 | 真实进展 |
|---|---|
| 上机检查 | PASS；71个源码文件Mac/GPU SHA一致，官方base与h10副本权重SHA一致；264帧、所有视觉mask=false/pixels=0 |
| 500-step训练 | 2026-10-04 10:49 UTC+8启动，session `cb79ec05` / task `cb79ec05-0334`，Ray job `06000000`；11:05亲核step1–500连续有限、warmup/LR一致；step500 loss0.005086/grad0.152102，最后20步loss均值0.0045889；SFT finished step500，checkpoint完整/812张量全BF16/参数更新PASS |
| 连续chunk评估 | 8/8条noise序列、424NPZ/848次完整推理完成；NFE10/horizon10/采用5帧，两组配对 |
| 原Parquet独立核验 | PASS；原state/action/padding/424个不同noise、上一段index4反馈、首段一致，全部统计复算 |

启动命令：`MY_DFS=<当前实测DFS> RAY_ADDRESS=29.209.160.111:6379 bash recipes/pi05_sft_so101_state_only_rollout/run.sh`。
启动前dguard暂停60分钟，driver EXIT恢复；源码来源 `df46c0a87a5cafacfb82ed03c2a1ea8f3a163c62`，
本recipe尚未commit，实际源码快照/SHA单独存证；Docker image tag未记录。
`preflight_backend.log`/`preflight.json`/`launch.json`及`source_snapshot`已保存到本轮输出根，
训练worker日志归档到`ray_logs/`；最终后台日志完成后归档。

用户澄清要验证连续diffusion process之间的误差传播：预测A[0,10)，保留[0,5)，
将A[4]直接当下一次state，再预测A[5,15)，保留[5,10)，依此类推。
此前NFE扫描只测试了单次diffusion内部步数，不回答该问题，结果保留在原recipe。
用户进一步指定训练无视觉模型约500step，从而无需构造随预测动作变化的视觉输入。
含视觉的state/跟踪偏差反馈方案没有启动，已撤销草稿，采用本recipe。

| 项目 | 本轮条件 |
|---|---|
| 初始化 | 官方pi05_base_pytorch独立h10副本，权重相同，step0开始，不resume |
| 数据 | nrailg/knock_down_the_cylinder_1_20260930_222251，完整1episode/264帧/15FPS |
| 视觉 | 训练和推理均使用既有DropVision：所有图像mask=false、pixel=0；模型架构不改 |
| 训练 | 500step，warmup100后constant1e-6，8H20 BF16 FSDP，micro4/global64/GAS2，seed1000 |
| 随机性 | 每batch随机noise/t，不使用single frame或fixed训练noise |
| 入口 | 复用fit_validation/train.py与FitTrainWorker，不新增train.py、不修改生产训练代码 |
| 输出 | $MY_DFS/experiments/carrot/pi05_so101_state_only_rollout/20261004T023935Z/training |
| 推理 | checkpoint step500，BF16生产infer/sample_actions，NFE10、预测horizon10、采用5帧 |
| 连续反馈 | 初始使用真实state；后续state=上一段采用的第5帧预测位置，丢弃后5帧，跨段不重置state |
| 对照 | 每段state均用对应数据集真实state，其它条件/noise与反馈组配对 |
| noise重复 | 8组独立noise序列；seed=1000+起始frame+chain*100000，每段新noise、两组同noise |
| 长度 | 每条53段，frame0/5/.../260；最后4帧有效，排除padding；共424NPZ/848次推理 |
| 统计 | 只统计实际采用的前5帧：各段/全episode/末状态五轴MAE/P95/max与对照差值/倍率，gripper单列 |

`run.sh`先训练，再评估与独立核验；运行前核对当前DFS、GPU/Ray、源码同步、资源与视觉屏蔽。
训练完成不代表充分拟合，基础拟合误差与连续反馈误差分别报告；若loss/grad/LR非有限则停止本次实验。
独立核验将直接读原Parquet，检查state/action/padding/noise、反馈恰好取上一段index4、每条初始输出一致、复算指标。
暂不使用真机或环境动力学，状态转移假设执行目标即成为下一state；这是state-only理想位置跟踪测试。
不模拟视觉，不加入recorded-state跟踪偏差补偿，不增加训练预算，不恢复旧消融suite。
完整结果与退出状态见下文；图片留Ceph/Mac缓存，不提交Git。


## 结果：直接反馈会明显偏离示教动作

训练从官方base新起，不resume，500步结束；step500 loss=0.0050857067，grad_norm=0.152102，最后20步平均loss=0.0045889。500条指标连续有限，warmup100/constant1e-6逐步核验，8个rank的action head参数均有有限非零更新。checkpoint step500/config horizon10/precision bfloat16、812个参数张量全BF16、norm_stats/tokenizer/8份optimizer shard完整；checkpoint对比base的action head最大变化0.00048828125。

以下只统计每段实际采用的前5帧；最后一段4帧有效，排除padding。每模式8条noise序列×264帧=2112有效action行，同一个episode的8次noise重复，不是8个独立episode。五轴按肩旋/肩抬/肘/腕屈/腕转，均为反归一化degrees；gripper原始单位另列。

| 输入状态方式 | 五轴MAE（°） | 五轴均值（°） | 五轴P95（°） | 五轴max（°） | gripper MAE（原始单位） |
|---|---|---:|---|---|---:|
| 每段真实state对照 | [1.090, 2.496, 1.409, 1.591, 0.148] | 1.346750 | [4.294, 6.724, 4.241, 3.985, 0.424] | [9.359, 15.519, 12.440, 8.868, 2.239] | 0.018696 |
| 上一段采用末action直接反馈 | [5.624, 31.883, 8.966, 15.273, 0.506] | 12.450586 | [25.749, 79.219, 22.914, 57.191, 2.364] | [37.776, 93.270, 35.978, 64.129, 4.073] | 0.058257 |

全episode五轴平均MAE由1.346750°增至12.450586°，增加11.103836°，约9.245倍。8条对照序列均值范围1.296–1.389°，反馈序列11.129–13.506°，并非只有一条noise异常。

| 实际采用区间 | 真实state对照：五轴平均MAE（°） | 连续预测state：五轴平均MAE（°） |
|---|---:|---:|
| [0,5) | 1.488 | 1.488 |
| [5,10) | 1.318 | 3.600 |
| [10,15) | 1.115 | 5.581 |
| [15,20) | 2.296 | 6.704 |
| [25,30) | 0.722 | 7.579 |
| [50,55) | 1.991 | 9.983 |
| [100,105) | 3.625 | 20.434 |
| [150,155) | 1.532 | 14.345 |
| [200,205) | 1.121 | 3.283 |
| [250,255) | 1.147 | 14.572 |
| [260,264) | 1.171 | 14.630 |

首段两组逐元素相同；后续直接反馈明显偏离示教轨迹，但误差不单调增长。动作轨迹图显示反馈组部分状态变化没有重放出来，不能概括为每段都按固定倍数线性累加。

本轮回答的是无视觉模型、h10/采用5帧/NFE10、理想位置跟踪下的离线state反馈。500step的真实state对照也仍有1.347°平均误差，不能称充分拟合。没有隔离出纯数值误差：状态分布偏离、模型条件响应、noise以及录制动作与实际state的跟踪差都会影响反馈；不凭此宣称生产代码BUG或含视觉5000step模型/真机必然同样失败。原数据各chunk边界的真实next state与前一示教endpoint action本来就不完全相同，五轴MAE约[1.072,1.809,1.961,1.637,0.434]°；本实验按用户要求直接把预测action作为state，没有加入lag补偿。未追加训练、未更改生产默认NFE10。

### 退出、证据与产物

- task `cb79ec05-0334` exit0，elapsed1276s；`SFT finished step500`、`CHUNK_ROLLOUT_COMPLETE NPZ424`、`INDEPENDENT_CHUNK_ROLLOUT_PASS`均在归档日志；task已dump释放。
- `checkpoint_validation.json`/`checkpoint_validation_backend.log`确认checkpoint完整、BF16和参数更新；核验task0354 exit0并dump释放。
- `evaluation/independent_validation.json`逐元素核对原Parquet/state/action/padding/noise、索引4反馈和首段一致，复算逐段/全episode/末动作/input-state的MAE/P95/max。Mac再次读取全部424NPZ、独立BF16 noise位舍入及FP64误差复算PASS；FP64与原FP32 mean最大差4.2655e-5°（归约顺序/精度），源数据/反馈仍严格相等。最初Mac FP32汇总因归约顺序相差1.8e-6°略超1e-6相对容差，改为独立FP64均值并明确容差，不修改模型结果。
- 初始71个Mac/GPU文件SHA一致；运行期间仅README进度变化，源码/配置/脚本不变，已存`running_source_check.json`；源码commit `df46c0a`加未提交recipe，实际快照/包版本/SHA见preflight/provenance，镜像tag未记录。
- 完整Ceph输出：`$MY_DFS/experiments/carrot/pi05_so101_state_only_rollout/20261004T023935Z`，实际本次MY_DFS为`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu`；含训练checkpoint、500行原始CSV/曲线、worker日志、424NPZ、metrics/comparison/核验/脚本快照。
- Mac结果与曲线：`/Users/wujunyu/.cache/carrot/so101_state_only_rollout_20261004T023935Z/`，不含模型/optimizer巨量权重。`evaluation/chunk_feedback_error.*`、`chunk_feedback_trajectory.*`、`chunk_metrics.csv`、`training_curves.*`已生成；曲线阴影表示8序列10–90百分位，训练平滑窗口20步。图片在Codex显示，未加入Git。
- 结束亲核dguard watch1、guard.sh/run.py实际运行、无restore计划；没有服务/机器人操作。原so101周期检查保持暂停；训练预算未追加，未commit/push。下一步真机执行与简单同步closed-loop replay仍等用户休假结束，之后再泛化。


### 2026-10-08 16:18 UTC+8进度核验

本轮训练959/1000，所有已记录step连续且loss/clip前grad/LR有限、warmup/constant核验通过，无alarm；71个执行源码/config/script hash保持不变。step500已亲读trainer_state/config/safetensors头：actualstep500/h10/precision BF16/812张量全部BF16，norm_stats与根tokenizer文件存在，optimizer8shards完整；不是只看目录名。当前没有evaluation NPZ，不提前下拟合/鲁棒性结论。dguard watch0/run.py未运行/restore scheduled（启动60分钟后约16:48恢复），此时无需延长；15分钟heartbeat仍ACTIVE。checkpoint_progress.json和最新CSV/status/ray_logs已持久化。


### 2026-10-08 16:27 UTC+8：1000步训练与离线评估完成

SFT finished实际step1000，最终loss0.0095510483、clip前grad0.200166，末20步loss均值0.00793345；1000条连续有限、warmup100/constant1e-6逐步核验通过。8rank action head均有有限非零更新。step500/1000均actualstep正确/h10/precision BF16/812张量全BF16/norm_stats/root tokenizer文件/8optimizer shards完整。训练和既有离线评估wrapper退出0，未重训或追加预算，启动至wrapper完成约36分钟。

424NPZ/848infer及原Parquet validator通过，8noise序列×53chunk、每模式2112个有效采用action行；首5步采用、index4直接反馈、视觉全屏蔽/NFE10。独立逐元素核对原state/reference/padding、与旧baseline的noise配对及实际反馈输入，FP64重算MAE/P95/max通过；Mac拉回全部424NPZ再次重算通过。下面是五轴反归一化degrees等权平均，gripper另列源单位。

| 模型 | 每段真实state MAE | 连续预测action反馈 MAE |
|---|---:|---:|
| 原无增强500step对照 | 1.346750° | 12.450578° |
| 本轮±3°增强1000step | 1.391723° | 14.209092° |

同协议配对下分别高3.34%/14.12%。两者训练预算不同、只有一个训练seed，不能隔离增强的独立影响；连续反馈误差仍大，未解决偏离。该离线采用5步协议不同于刚才真机预测10取1、录制state输入协议，不能直接拿1.187°首动作MAE当同条件对照。本轮全部424NPZ所有数组与历史matched-fit增强1000step逐元素相同，是相同seed配置复现，不是额外训练seed证据。新模型尚未执行真机。

| 本轮输入 | 五轴MAE°（肩旋/肩抬/肘/腕屈/腕转） | 五轴P95° | 五轴max° | gripper MAE / P95 / max |
|---|---|---|---|---|
| 真实state | [1.243971, 2.335813, 1.446337, 1.744409, 0.188084] | [4.273483, 6.964398, 4.399042, 4.917789, 0.480837] | [17.647377, 19.134695, 20.410488, 24.369333, 3.299152] | 0.015534 / 0.061571 / 0.118214 |
| 预测反馈state | [7.346681, 33.046529, 14.704263, 15.504951, 0.443034] | [22.114888, 84.127824, 35.795469, 62.207777, 1.023831] | [36.802103, 91.045082, 40.682957, 65.194218, 4.248776] | 0.079941 / 0.236089 / 0.261443 |

收尾状态分别保留：0480因启动时已加载的tokenizer目录断言exit1，failure.json明确训练/评估wrapper_exit_code=0；0483完成checkpoint/评估核验后因GPU尚未完全空闲时检查dguard run.py而exit1。两者均是收尾核验错误，非训练失败。释放后触发dguard并再次执行正确finish_audit，INDEPENDENT_COMPLETION_AUDIT_PASS；实际watch1/guard.sh与run.py运行/无restore计划，Ray空闲224CPU/8GPU、run.sh PID释放。final_remote_checks.json记录最终状态，初始失败日志和快照保留，无需恢复或重训。

0480/0483后台已dump归档释放为training_evaluation_backend.log和initial_finish_audit_backend.log。输出根保存completion/completion_audit_notes/independent_final_comparison/final_remote_checks/逐stepCSV/原始worker日志；Mac副本~/.cache/carrot/so101_no_vision_jitter_20261008T074452Z，排除训练大权重/optimizer，含mac_validation.json。15分钟so101 heartbeat在完成核验后暂停。只更新记录，未额外commit/push、机器人动作或改动用户Isaac。


### 2026-10-08：预测角度与当前标定范围核验

只读全部424NPZ，对当前follower保存的标定端点使用LeRobot bus._normalize转换成degrees；bus未connect、无串口打开。五轴范围分别为±[112,104.835165,97.274725,96.615385,180]°，这是当前标定软件范围，不是经过验证的机械硬限位。calibration SHA9bb12feb1f18ff03e4f6298e6655fd0a0f5f68d68409f0a5c79f06178d0ee65b，具体结果保存Mac缓存angle_bounds_validation.json。

| 输入模式 | 采用前5步越界action行/总行 | 比例 | 肩抬/肘越界行数 | 肩抬/肘最大越出边界 |
|---|---:|---:|---:|---:|
| 真实state | 133/2112 | 6.30% | 95 / 42 | 4.929850° / 0.935709° |
| 连续预测反馈 | 41/2112 | 1.94% | 30 / 15 | 3.629404° / 0.926431° |

同一action可多个轴越界，逐轴计数不能直接相加；2112行是同一episode×8noise序列，不是2112个唯一时刻。仅肩抬/肘越界，其余三运动轴没有。采用前5动作里，两模式都没有越界幅度>5°的动作。反馈组最严重肩抬chain3/frame255/offset0预测−108.464569°，比标定下界−104.835165°低3.629404°。

若计算全10步有效预测（包含丢弃后5步、排除padding，4184行/模式），真实state组213行越界，其中5行越出>5°、最大肩抬7.533076°；反馈组62行越界，最大仍3.629404°，无>5°。原示教action本身肩抬最大越出0.307696°；8条重复序列共64/2112源action行越界。

此离线递推直接使用原始未限幅预测，反馈condition有3/424个chunk输入state越界（肩抬2、肘1）；这些state并未模拟机器人限幅后的状态。14.209°指相对示教action的MAE，不是越出标定范围14°；越界比例更低也不意味着轨迹更准确。新模型未执行机器人，无任何实际超范围目标发送。
