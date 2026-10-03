# SO101逐步拟合验证

## 2026-10-03：当前进度与后续顺序

目标顺序保持：**拟合能力 → 简单条件真机 closed-loop replay → 泛化**。
只采用主代理独立核验的新 suite/audit 证据；旧代理结论不作为验收。

| 验证条件 | 当前进度 | 已确认的结论 / 尚缺的证据 |
|---|---|---|
| 单 episode | 新数据264帧，含视觉h10组1000step完成 | 原FP32评估已核验；完整episode需在BF16生产采样条件下重评，未验收充分拟合 |
| horizon 降到10 | 四个新消融组均使用h10 | 这是新suite统一条件，未新做h50与h10的严格配对对照 |
| 去掉视觉 | 整episode无视觉组1000step完成 | 原FP32主指标核验通过；BF16重评待执行 |
| 只用一个frame | frame53随机noise组1000step完成 | 原FP32八noise评估已核验；BF16同条件重评待执行 |
| 同frame固定noise | frame53/noise1053组1000step完成 | BF16/eager全计算与条件KV cache均能很好拟合，五运动轴MAE约0.02–0.09°；不扩展为跨noise或整episode成功 |
| 推理路径与导出 | BF16导出、optimizer-only续训、SDPA dtype assert已实现 | 最后两轮定向测试7项与cache测试6项PASS；旧37.55°来自audit错误的SDPA/mask组合，并非生产cache代表 |

下一步先复用现有checkpoint，在独立输出目录保存BF16推理导出，
通过正常policy/`sample_actions()`重评同frame、同noise，再覆盖其余三组。
不追加训练、不覆盖历史checkpoint；episode主指标仍为264唯一观测各一次，
固定noise验收使用1053本身，重复/其他noise另列。首1/5步指标与全10步指标分开报告。
当前尚未执行这轮完整生产入口重评，不能把代码回归PASS当成新的拟合结果。

真机阶段等用户结束休假、有设备后执行：
1. 绕过模型，按录制FPS replay原始action，核对关节顺序、单位、限幅后的实际目标和反馈。
2. 录制observation输入已离线验收的policy，执行预测动作，检查正式部署链路与离线结果一致。
3. 使用真实state/图像，在接近示教的简单场景中闭环重放；**同步infer→执行→重新观测作为基线**。
4. 同步闭环通过后再比较异步，再进行物体位置/初始姿态等泛化验证。

当前无新policy/机器人操作，自动检查保持暂停；新日志均已归档，dguard恢复。

## 2026-10-03：BF16 保存与续训

用户确认推理导出应保存 BF16。`PI0Pytorch.save_pretrained()` 现在将浮点权重副本
转换为 BF16，并写入 `precision=bfloat16`；不原地改动训练参数，不覆盖旧实验 checkpoint。
strict reload 后 attention 使用 BF16，保留模型原有的 FP32 normalization/projection 计算规则。

续训直接加载根目录 BF16 模型，建立 FSDP 后恢复 optimizer、scheduler 和 step。
`optimizer/` DCP 只保存 optimizer，不额外保存 FP32 master；接受模型参数的 BF16 舍入损失。
无版本字段、历史格式兼容分支或额外 model 恢复参数。

BF16 导出初次验证：50项回归 **PASS**（41.45s），含当时的两项兼容测试；
用户随后要求移除兼容逻辑，这两项测试已删除；简化后的定向回归 **7项 PASS**（34.70s）。
简化前真实 PI05 两卡保存/重载/再次保存：**PASS**（228.84s）；
这是此前完整master恢复方案的历史结果；当前改回 optimizer-only DCP，定向回归 **7项 PASS**（34.78s）；
本轮未重跑真实3B两卡测试。证据位于上述目录的 `optimizer_only/`。
证据：`MY_DFS/test-runs/so101_bf16_export_20261003/`；测试三件套见
`tests/pi_05/test_pi05_modeling.*`、`tests/pi_05/test_pi05_checkpoint_distributed.*`、
`tests/sft/test_sft_checkpoint.*`、`tests/sft/test_sft_checkpoint_distributed.*`。

SDPA mask dtype BUG 是独立问题：旧 audit 的条件 prefill 使用 SDPA；当前训练的双流
full forward 和正常 `sample_actions()` 都使用 eager，未经过出错的 cuDNN SDPA 分支。
本次没有重新跑四组训练或宣称新增拟合指标；真机验证仍待执行。

## 2026-10-03：缓存差异定位与修复

**旧 audit 的 BF16 缓存 37.55° 肩部误差来自错误的 backend/mask dtype 组合，
不能代表生产 `sample_actions()` 的缓存路径。** 生产入口将条件/action 设为 eager；
旧 audit 直接 prefill，条件分支默认 SDPA，却传入 FP32 additive mask。

当前 H20 / torch2.11.0+cu128 / transformers5.5.4，BF16 SDPA 实际使用 cuDNN。
最小复现中，修改被 block mask 禁止读取的动作 token，条件输出仍改变，最大差异8.8125。
独立数学参考验证：BF16 Q/K/V + FP32 mask 时 cuDNN 有效 query max error0.857032；
同一 mask 转成 BF16 或 bool 后降到0.003536，与 math backend 同量级。
最初在 `GemmaAttention` 内转换 mask dtype 以验证修复；按用户要求现改为 assert，
浮点 mask 必须与 query dtype 一致，bool 合法，禁止底层隐式转换。
audit 在 prefix mask 构造入口显式指定计算 dtype；默认 eager，并记录实际 SDPA 内核。
当前 mask contract/cache GPU 回归 **6项 PASS**（17.29s），未重跑真实 checkpoint audit；
证据位于上述 cache diagnosis 证据根的 `dtype_assert/`。

同一已训练 checkpoint/frame53/noise1053，10 NFE，无新增训练：

| 条件 | 采样路径 | pan MAE° | lift MAE° | 五运动轴 MAE° |
|---|---|---:|---:|---|
| BF16参数/输入，原FP32 buffers | full forward/eager | 0.05496 | 0.07471 | [0.05496,0.07471,0.06648,0.08541,0.02029] |
| 同精度，修复前正确设置eager | KV cache/eager | 0.05853 | 0.05841 | [0.05853,0.05841,0.07138,0.06983,0.01971] |
| 同精度，旧SDPA+FP32 mask复现 | KV cache/错误mask dtype | 6.38102 | 37.55038 | [6.38102,37.55038,4.69201,12.24285,1.75125] |
| 同精度，修复mask dtype | KV cache/SDPA-cuDNN | 0.02739 | 0.06163 | [0.02739,0.06163,0.10193,0.07087,0.01166] |
| 原FP32 master参数/FP32计算 | full forward/eager | 1.44334 | 1.68998 | [1.44334,1.68998,0.39601,0.36083,0.17860] |
| 参数先BF16舍入，再转回FP32计算 | full forward/eager | 0.02116 | 0.06865 | [0.02116,0.06865,0.09663,0.04767,0.02208] |

最后一个对照保留 FP32 运算，仅把参数变成训练BF16计算所使用的可表示值；
证据将原FP32重载残差主要定位到 master 参数值与 BF16 有效参数值的差异，
不能解释为 FP32 运算本身错误。FSDP配置param_dtype=BF16，导出保存FP32 master且precision=FP32；
本节诊断时默认推理加载该FP32配置；后续BF16导出修复见顶部小节，旧checkpoint仍保留原样。
缓存结构本身没有发现跨层/位置错误；eager full/cache teacher velocity max gap0.015625，
修复后的SDPA/full max gap0.078125，两个backend数值不保证逐bit相同。

6项新GPU回归及相关推理/SFT/recipe测试共 **48/48 PASS**，17.96s；
修复前新增block隔离测试FAIL，修复后PASS；Ruff/bash/compile检查PASS。
主代理在Mac从原Parquet frame53..62复算3次审计/30份rollout的MAE/P95/max，全部通过。
证据：`MY_DFS/test-runs/so101_cache_diagnosis_20261003/`，包含修复前源码、3次audit结果、
SDPA mask探测、独立复算、日志与源码hash；详细测试档案见
`tests/pi_05/test_pi05_cache_parity.md`。未追加训练或操作机器人，自动检查仍暂停。

## 2026-10-03：主代理独立重跑（训练、评估与推理审计已完成）

### 本轮实验进度表

共 **4组重新训练 + 1项推理链路检查**，依次执行。更新于北京时间 **2026-10-03 14:43**，
仅引用新输出 `20261002T183105Z` 的实际日志；初始task `af3ae39e-0178` 中断后，
续跑task `d9b1a9fa-0188` 已exit0；审计task `d9b1a9fa-0197` / `d9b1a9fa-0200` 均exit0。
四组训练各1000步及必需核验已完成；后台日志归档，dguard已恢复，30分钟自动检查收尾暂停。

| 编号 | 实验 | 训练数据 | action horizon | 视觉 | 训练noise | 当前进度 | 拟合结论 |
|---|---|---|---:|---|---|---|---|
| 1 | `episode_h10`：缩短预测窗口 | 1 episode / 264帧 | 10 | 有 | 每次随机 | **训练完成：1000/1000、exit0**，loss 0.003049；预测校验通过 | 尚未充分拟合；264唯一观测主指标见下 |
| 2 | `episode_h10_no_vision`：移除视觉条件 | 1 episode / 264帧 | 10 | 无，image mask关闭 | 每次随机 | **训练完成：1000/1000、exit0**，loss 0.003369；主代理复算通过 | 尚未充分拟合；与含视觉组相近 |
| 3 | `frame53_h10_no_vision`：固定一个观测 | frame53，重复为64个batch样本 | 10 | 无 | 每次随机 | **训练完成：1000/1000、exit0**，loss 0.000278；主代理复算通过 | 尚未充分拟合；wrist_flex MAE约2.91° |
| 4 | `frame53_h10_no_vision_fixed_noise`：再固定noise | 同一frame53 | 10 | 无 | 固定seed1053；t仍随机 | **训练完成：1000/1000、exit0**，loss 0.00004362；主代理复算通过 | 原FP32评估仍有残差；BF16无缓存同noise诊断通过 |
| 5 | `audit.py`：训练/推理一致性检查，不追加训练 | 实验4的新checkpoint及同一frame/noise | 10 | 无 | 同seed1053 | **完成：两次audit均exit0**，输入/stats/往返核对通过 | BF16无缓存拟合很好；FP32及BF16缓存路径不一致，需定位 |

- 4组均从同一 `pi05_base_pytorch` **重新初始化**，1000step、warmup100、constant LR1e-6，
  global batch64/micro4/GAS2、8GPU BF16 FSDP；只保存最终checkpoint，原264帧stats保留。
- 每组验证实际参数更新、完整训练步数/有限指标、模型独立重载，以及源值六轴MAE/P95/max。
  loss只是训练进度，不能当作拟合通过证据。
- 新输出完整路径：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_so101_fit_validation/20261002T183105Z/`。
- 控制契约测试已完成：**3/3 PASS**；真实数据代表帧的10步action/padding与原始Parquet逐元素一致，63个源码/recipe hash与Mac一致。
- 既有horizon50和旧suite产物保留为历史，不混入本轮结果。真机replay、closed loop及泛化保持待执行。

### 2026-10-03：提交归档与指标单位说明

- 本文所有六轴动作MAE/P95/max均在原`Unnormalize`后对原Parquet action计算；
  前五轴单位degrees，gripper保留源单位。这些值不是normalized action误差，也不是flow loss。
- 提交包括四组累积消融recipe、checkpoint/NPZ校验、显式产物续跑和精度/full-cache审计，
  以及单帧factory与两套测试三件套。生产模型/推理源码未修改，根因仍待定位。
- 提交前仅整理recipe imports/行宽及补type hints/override；已完成实验的版本由
  Ceph中`source_snapshot`、`resume_source_snapshot_*`和两次audit快照固定，不把整理后源码冒充运行版本。
- 本次CPU回归共**7/7 PASS**（fit controls 3项8.62s、single-frame 4项6.57s），
  Ruff、shell语法及Mac/GPU 8文件hash检查PASS。
  记录见`tests/pi_05/test_so101_fit_controls.md`及`test_so101_single_frame_recipe.md`。
- 提交前检查证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101_fit_validation_commit_20261003_d9b1a9fa_0209/`；日志、版本及hash均已留档。
- 本轮提交到当前`whyRecipeFailed`分支，并按用户授权推送`mygh`；提交SHA由Git历史及canonical memory记录。

### 本轮最终结果：最简条件具备拟合能力，推理数值与缓存路径需排查

四组均从base初始化完成1000步。主代理独立确认各组train exit0、1000条连续有限loss/LR/grad、
100..1000步恒定LR1e-6、action head非零更新、checkpoint及原始stats；所有预测reference、
padding、noise与源数据契约一致，MAE/P95/max复算通过。suite完成时间UTC21:06:40（北京时间05:06:40）。

| 新训练case | 主指标范围 | step1000六轴MAE（前五度数、gripper源值） |
|---|---|---|
| episode_h10 | 264唯一观测各一次，2595有效action行 | [0.64922, 1.87276, 0.92951, 1.53690, 0.14697, 0.01065] |
| episode_h10_no_vision | 同上；28重复noise探测另列 | [0.82175, 1.69622, 0.79436, 1.47174, 0.15033, 0.00787] |
| frame53_h10_no_vision | frame53，8个noise / 80有效action行 | [1.04001, 1.75012, 0.36551, 2.90931, 0.18680, 0.00749] |
| frame53_h10_no_vision_fixed_noise | frame53、训练noise1053自身 / 10有效action行 | [1.44368, 1.68479, 0.39592, 0.35951, 0.17853, 0.00231] |

上表使用原checkpoint独立加载的FP32采样路径，四组仍未全部达到严格诊断参考。
固定noise组50-NFE同noise MAE为[1.35370,1.49524,0.39491,0.37841,0.17141,0.00267]，
没有消除肩部残差。其他7个noise的10-NFE MAE为
[1.71813,4.38974,1.29964,5.93712,0.45000,0.04390]，只作跨noise诊断，不能当同条件拟合指标。

#### 同checkpoint / 同frame53 / 同noise1053的推理审计

原审计task `d9b1a9fa-0197`，补充精度控制task `d9b1a9fa-0200`，两次exit0、无反向/训练。
第二次结果文件生成于北京时间05:21:11，最终读回复核于14:41之后。
train/eval state及tokens、stats精确相同；action归一化往返最大源值误差3.8147e-6。
增加保留原FP32 buffers的BF16参数对照，以排除直接model.to(BF16)连同buffers转换的影响。

| 计算条件 | 采样路径 / 10 NFE | 六轴MAE |
|---|---|---|
| checkpoint原生FP32 | KV cache | [1.44196, 1.68638, 0.39699, 0.35831, 0.17843, 0.00231] |
| 同FP32 | full forward，无cache | [1.44334, 1.68998, 0.39601, 0.36083, 0.17860, 0.00230] |
| BF16参数与输入，原FP32 buffers | KV cache | [6.38102, 37.55038, 4.69201, 12.24285, 1.75125, 0.10598] |
| 同BF16，原FP32 buffers | **full forward，无cache** | **[0.05496, 0.07471, 0.06648, 0.08541, 0.02029, 0.000142]** |
| 参数、输入、buffers全BF16 | full forward，无cache | [0.04409, 0.07550, 0.07834, 0.06898, 0.01894, 0.000246] |

保留原buffers的BF16无cache采样，P95为[0.09163,0.15836,0.11650,0.18538,0.03203,0.000304]，
max为[0.09919,0.16534,0.12956,0.21725,0.03300,0.000360]；已低于本recipe严格诊断参考。
主代理将审计输出动作与已核对源Parquet的reference再次独立复算，结果保存在
`inference_audit/independent_validation.json`。

**更新判断：最简同frame/同noise条件下，模型能够很好拟合10步动作。**
之前原生FP32采样的残差不能单独作为训练能力不足的证据。训练使用BF16 FSDP，checkpoint保存FP32
master weights；审计显示改变计算精度会显著改变本checkpoint输出。BF16的full forward与缓存路径
也不一致：7个teacher点的32维velocity最大差异，FP32最大0.00506，保留原buffers的BF16最大1.80664。
这给出训练/推理数值配置和缓存实现需要进一步定位的证据，尚未锁定具体代码行或认定根因。

边界：BF16单卡诊断近似训练计算配置，但未精确复现FSDP wrapping/分布式计算；该项仅验证一个frame、
一个noise、10步动作，不证明整episode、跨noise、视觉条件、真机replay/closed loop或泛化已成功。
没有将BF16缓存诊断的较大误差混入原FP32评估，没有修改生产代码或追加训练预算。
下一步应定位精度差异与full/cache不一致，再恢复按训练条件一致的评估；真机因休假无设备仍待执行。

#### 归档与收尾

- 输出根沿用新`20261002T183105Z`；initial失败与resume日志分别为
  `suite_initial_backend.log`、`suite_resume_backend.log`；`completion.json`包含4个新case。
- 审计源码快照`inference_audit/audit_initial.py`及`audit_precision_control.py`，结果分别为
  `result.json`、`result_precision_control.json`，后台日志`initial_backend.log`、
  `precision_control_backend.log`；独立验证JSON记录两个源码SHA256与退出码。
- 本地新实验副本：`/Users/wujunyu/.cache/carrot/so101_fit_validation_independent_20261003/`，
  保留configs/metrics/NPZ/logs/audit，checkpoint留Ceph；与旧suite cache严格区分。
- Carrot main d4999bbe6557c2b91273dbe3abbd229532882d66加未提交recipe/test变更，
  实际依赖版本和源码hash见environment/resume_environment；Docker tag、上游源码commit未记录。
- dguard已核对DGUARD_WATCH=1、run.py运行、无待恢复计划。后台task均归档释放；
  30分钟自动检查在必需核验完成后暂停。未commit/push、未操作机器人。

### 04:44：单帧组完成，固定noise组进行中

- 第三组train exit0/1000step，final loss0.000277549；主代理亲自确认1000条连续有限指标、
  100..1000步LR1e-6、参数实际更新及checkpoint/stats。base/step1000各8份NPZ均与
  原Parquet frame53..62的action、有效mask和生成noise逐元素一致，MAE/P95/max独立复算通过。
- 主指标为同frame的8个noise、80个有效action位置；前五轴度数，gripper源值：

| 单帧组主指标 | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---:|---:|---:|---:|---:|---:|
| base MAE | 5.45112 | 16.51410 | 1.68020 | 11.31727 | 0.44095 | 0.22114 |
| step1000 MAE | 1.04001 | 1.75012 | 0.36551 | 2.90931 | 0.18680 | 0.00749 |
| step1000 P95 | 1.32380 | 2.68966 | 0.69984 | 3.89075 | 0.25312 | 0.01050 |
| step1000 max | 1.42093 | 3.07909 | 0.98995 | 4.33800 | 0.27056 | 0.01243 |

单帧训练明显改善同观测预测，但pan/lift/wrist_flex仍超诊断参考，未充分overfit；
该观测的指标不能与覆盖264观测的episode均值直接比较来判断条件优劣。
第四组step353指标有限，主要验收仍待training noise1053自身预测，audit尚未执行。

04:14定时检查：当前task仍运行，单帧组到560/1000，已记录loss/LR/grad全部有限；固定noise组仍排队，尚无新增拟合结论。

### 04:08：去视觉组完成，单帧组进行中

- 当前续跑task `d9b1a9fa-0188` 仍运行。第二组完整1000条step/loss/LR/grad有限，
  100..1000步LR1e-6、train exit0、action head实际更新；checkpoint及原state/action stats核对通过。
- 主代理逐一读取base/step1000各292份NPZ，与原始Parquet核对reference、padding、noise，
  独立复算MAE/P95/max一致；主指标仍为264唯一观测/2595有效action行，重复28份探测另列。

| 去视觉组主指标 | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---:|---:|---:|---:|---:|---:|
| base MAE | 6.23543 | 8.52162 | 5.83834 | 6.14811 | 0.52501 | 0.07355 |
| step1000 MAE | 0.82175 | 1.69622 | 0.79436 | 1.47174 | 0.15033 | 0.00787 |
| step1000 P95 | 2.66643 | 4.38014 | 2.26997 | 3.64168 | 0.40345 | 0.02812 |
| step1000 max | 6.93249 | 8.91853 | 7.45623 | 9.70310 | 3.15881 | 0.08830 |

前五轴为度，gripper为源单位。去视觉后部分轴略好、部分略差，整体与含视觉组相近，
不能据此判定视觉就是拟合障碍；本组仍未充分overfit。第三组step336指标有限，第四组排队。
最后case完成后再运行推理audit；本次未追加训练或改变生产代码。

### 03:21真实进展与续跑记录

- 初始suite退出1：第一组训练/两次评估已完成，随后校验调用了不存在的
  `pyarrow.parquet.concat_tables`。这是新增recipe校验代码的错误，已改为
  `pyarrow.concat_tables`，不涉及生产训练/推理模型。失败后台日志保留为
  `suite_initial_backend.log`。
- `run.sh --resume-root .../20261002T183105Z` 只复用第一组完整训练/评估产物，并继续剩余三组；
  配置逐项一致性、exit0和step1000验证后才复用。没有重复第一组训练、覆盖checkpoint或追加预算。
  当前task `d9b1a9fa-0188`；续跑命令、源快照及环境记录见同目录
  `resume_launch.json`、`resume_source_snapshot_*`、`resume_environment_*`。
  源码hash比较仅suite.py/run.sh变化、增加audit.py，生产src和fit/train/evaluate未变。
- 第一组：亲自读取1000条连续且有限的loss/LR/grad记录，100..1000步LR均1e-6；
  action head实际参数更新非零，checkpoint/model/stats存在且已独立重载完成评估。
  所有292份NPZ的reference/padding/noise与原Parquet/生成器逐元素一致，MAE/P95/max复算一致。
- 主指标使用**264唯一观测各一次、2595有效action行**；另外28个重复noise探测单列。
  五个运动轴单位为度，gripper保留数据源单位，顺序如下：

| 主指标 | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---:|---:|---:|---:|---:|---:|
| base MAE | 4.23578 | 7.66720 | 5.10150 | 5.83303 | 0.52736 | 0.07465 |
| step1000 MAE | 0.64922 | 1.87276 | 0.92951 | 1.53690 | 0.14697 | 0.01065 |
| step1000 P95 | 1.83710 | 4.87370 | 2.58866 | 3.86300 | 0.35371 | 0.03262 |
| step1000 max | 6.60309 | 11.04094 | 6.47925 | 8.70530 | 2.07155 | 0.09958 |

这些新结果显示误差明显下降，但多个运动轴仍超出recipe的严格诊断参考，不能宣称充分overfit。
第二组训练已启动。初始失败后dguard已恢复并核对；续跑时再次暂停360分钟，退出trap负责恢复。
30分钟自动检查已更新为当前task，完成四组后仍需执行独立推理audit。

- 用户明确不采信旧GPT-6 Luna结论；下方旧结果仅保留为历史档案，本轮不据此判定。
- 新输出时间戳目录，从同一base重新跑四组，各1000step/warmup100/constant1e-6。
- recipe控制契约3项CPU检查通过：随机objective与生产一致、固定noise仍随机t、去视觉mask关闭。
- 训练保存1000条有限loss/LR/grad校验及实际action head更新；评估保存latent，逐元素核对原Parquet目标、padding和noise，并独立复算MAE/P95/max。
- 固定noise组以训练noise1053自身为主要拟合验收，其他noise单列诊断，不能将二者混为一种结论。
- 配置和recipe快照/实际版本/hash与日志保存在新输出，不覆盖历史；真机仍待休假结束。

## 2026-10-02/03：执行用户确认的plan

- 目标：拟合能力→简单条件closed-loop replay→泛化。用户休假无机器人，本轮仅离线验证。
- 保留既有单episode50步和单frame53/50步结果；新cases.yaml累积简化：
  单episode/horizon10→无视觉→固定frame53→固定noise1053。
- 每轮base重新初始化，1000step、warmup100、constant LR1e-6、8GPU BF16 FSDP、
  micro4/global64/GAS2，原264帧stats/源数值保持，只保存最终checkpoint减少重复存储。
- 使用recipe专用worker复用生产SFT setup/训练循环/checkpoint；第一批前将模型horizon设为10。
  无视觉同时将canonical image和image_mask清零；训练/推理使用同一DropVision。
  随机noise分支与生产loss等价，固定noise为BF16可表示的同一latent，所有rank/调用一致。
  t仍逐次随机，图像增强逻辑保留（无视觉时所有image token被mask）。
- 每轮检查实际action_out_proj参数更新；额外记录六个有效轴和26个padding轴的loss，
  保持原生产32维均值objective，不改变权重。训练结束、重载、拟合分别验收。
- episode评估覆盖全部264帧（每帧1noise），frame53/82/201/202各8noise，共292个预测；
  单frame各8noise。base/最终checkpoint共享相同latent，排除episode padding。
  固定训练noise额外单列该noise误差；最后case补50次去噪诊断，与默认10次分开记录。
- 检查MAE/P95/max、reference与源数据一致，不能用loss下降替代拟合结论。
  暂以五个运动轴MAE≤0.5°且P95≤1°、gripper源值MAE≤0.02作为严格诊断参考，
  不视为真机成功/正式用户验收阈值，不根据阈值自动启动硬件。
- 命令：确认MY_DFS/RAY_ADDRESS后bash recipes/pi05_sft_so101_fit_validation/run.sh。
  独立输出MY_DFS/experiments/carrot/pi05_so101_fit_validation/<UTC启动时间>/<case>/。
  后台suite每900秒查询一次训练，训练退出后保存rank0日志并评估，再开始下一轮。
- 当前版本main d4999bb加未提交变更；Docker tag未记录。任务ID和逐轮结果见下方。

### 启动记录

- 重新核对的MY_DFS为/mnt/ceph-hz1-csp/mm-base-plt2/nrwu，Ray 29.209.160.111:6379，
  8张H20可用、Ray无live actor，Ceph可用空间480T。Python3.12.13、torch2.11.0+cu128、lerobot0.6.1。
- 真实数据预检：264帧；6个代表帧均返回float32[10,6]；episode末帧9步padding。四种配置均通过SFTConfig解析。
- 静态Python compile、bash -n及git diff --check通过；没有加跑单元测试。
- suite task 375a9103-0152于UTC 2026-10-02 15:56:16启动，输出根
  $MY_DFS/experiments/carrot/pi05_so101_fit_validation/20261002T155609Z/。独立日志在每个case的driver.log。
  dguard暂停360分钟并由run.sh退出trap恢复；训练每900秒报告进度。
- 首个episode_h10 rank0首forward通过，初始loss 0.094238；有效六轴单项误差均有记录。

### 旧suite历史结果（用户要求不采信，不能作为本轮结论）

suite `375a9103-0152` 于UTC 2026-10-02 15:56:16启动，输出根
`$MY_DFS/experiments/carrot/pi05_so101_fit_validation/20261002T155609Z/`。
四组的训练和10-NFE评估均已完成；第四组额外的50-NFE评估也已完成。
所有case均为1000步、从同一base初始化，动作评估采用同一组noise seed对比base与step1000。
NRMSE为跨有效动作元素的源action q01–q99范围归一化RMSE；MAE向量按
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` 排列。
前五轴以度计，第六轴是数据源单位。诊断参考为运动轴MAE≤0.5°且P95≤1°，gripper MAE≤0.02；这不是硬件成功阈值。

| 组 | 输入/训练noise | 最终训练loss | NRMSE base → step1000 | NRMSE下降 | step1000六轴MAE |
|---|---|---:|---:|---:|---|
| `episode_h10` | 264帧；视觉；每步随机noise | 0.00304866 | 0.223544 → 0.040632 | 81.82% | [0.670°, 1.920°, 0.925°, 1.693°, 0.152°, 0.0107] |
| `episode_h10_no_vision` | 264帧；无视觉；每步随机noise | 0.00336862 | 0.227609 → 0.038659 | 83.02% | [0.811°, 1.698°, 0.798°, 1.521°, 0.156°, 0.0080] |
| `frame53_h10_no_vision` | 固定frame53；无视觉；每步随机noise | 0.00027755 | 0.396075 → 0.032080 | 91.90% | [1.040°, 1.750°, 0.366°, 2.909°, 0.187°, 0.0075] |
| `frame53_h10_no_vision_fixed_noise` | 固定frame53；无视觉；固定训练noise seed1053 | 0.00004362 | 0.396075 → 0.101706 | 74.32% | [1.684°, 4.052°, 1.187°, 5.240°, 0.416°, 0.0387] |

第四组step1000在**训练noise seed1053本身**上的chunk MAE为：10-NFE
`[1.444°, 1.685°, 0.396°, 0.360°, 0.179°, 0.0023]`；50-NFE
`[1.354°, 1.495°, 0.395°, 0.378°, 0.171°, 0.0027]`。50-NFE跨8个noise聚合MAE为
`[1.754°, 4.526°, 1.463°, 5.815°, 0.451°, 0.0417]`，NRMSE 0.110900，相比10-NFE的0.101706略差。
增加推理步数没有显著改善总体结果。固定noise降低训练loss，并改善训练noise本身的预测，但仍未让全部动作轴达到诊断线。
因此四组目前都不能判为充分overfit；误差随简化条件总体下降，但只固定训练noise并未让跨noise预测变好。
这仍不足以判定代码有bug。下一步按plan进一步隔离训练目标、输入对齐、归一化和推理链路。
真机原始action replay、closed-loop replay与同步执行仍待用户回到机器旁后开展；本轮没有操作机器人。

每组的`result.json`、base/step1000 `metrics.json`、逐样本NPZ和checkpoint位于上述Ceph输出根各case目录；`driver.log`保留训练step、学习率和梯度日志。`completion.json`列出四组完成；四个step1000 checkpoint及结果文件均存在，训练任务退出码0。dguard退出trap已恢复巡检（`run.py`进程运行）。
