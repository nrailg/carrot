# 机器人研究问题地图：摘要级综述

本报告把已收集条目整理成研究问题地图，以机器人学习为主，兼顾感知、规划、控制、硬件与系统。它是摘要级研究版图；少量摘要未交代机制的条目补读了官方正文方法概述。
检索主窗口为 2026-07-12 至 2026-10-12，但语料也包含会议补充的更早预印本和未来会议候选。下述统计针对当前收集语料，不能解读为这三个月整个机器人领域的完整发表统计。

原始来源记录 **9,659** 条，保守去重论文条目 **9,073** 个。

| 条目状态 | 数量 |
|---|---:|
| 摘要支持 | 7,551 |
| 与机器人不相关 | 641 |
| 信息不足 | 881 |
| 仅标题暂判 | 0 |

主问题每篇只计一次；次问题计数允许一篇涉及多个类别。以下数量描述本次收集语料，不代表整个领域的发表占比。

| 研究问题 | 主问题篇数 | 作为次问题篇数 |
|---|---:|---:|
| 数据与监督成本 | 467 | 571 |
| 泛化迁移 | 493 | 750 |
| 感知状态 | 1,165 | 1,167 |
| 指令理解与任务推理 | 262 | 390 |
| 长任务与记忆 | 244 | 427 |
| 动作与物理交互 | 1,000 | 1,303 |
| 自主学习与持续改进 | 357 | 343 |
| 安全、鲁棒与恢复 | 797 | 810 |
| 人机与多机协作 | 510 | 460 |
| 效率与部署 | 832 | 882 |
| 本体与传感设计 | 328 | 106 |
| 评测与机制理解 | 1,096 | 442 |

## 分类阅读

子问题与解决路线是主代理对本轮摘要的归纳，并非互斥的正式二级标签。代表论文用于说明路线，不表示质量排名；每篇代表论文均重新核读证据。

### 数据与监督成本

**子问题：**示教与标注昂贵；人类视频缺少机器人动作标签；海量外部数据中哪些样本有用；合成数据如何满足任务需求。

**主要解决路线：**筛选与目标示教相似的人类数据；视角变换和逆动力学补动作；仿真及受任务约束的场景生成；弱监督、自监督与少量本地数据适配。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [DINS-IO: Learned Inertial Odometry via Differentiable INS Consistency](https://arxiv.org/abs/2607.20232) | 学习惯性里程计依赖昂贵稠密位置真值，难规模化获取监督。 | 将捷联惯导速度递推写为滑窗共享偏置最小二乘闭式一致损失自监督预训练，再以少量轨迹LoRA校准度量速度。 | 感知、控制或系统支撑技术，摘要未明确机器人实验。 |
| [DISEIL: Demonstration Distillation for Sample-Efficient Imitation Learning](https://arxiv.org/abs/2609.08123) | 交互模仿虽能中断失败，却未主动选择需纠正的失败类型及示范起点。 | 定位首次不可靠步骤以几何描述聚类失败模式，由视觉语言及语言模型提出示范请求并经任务约束核验。 | 单轮仿真且专家多为脚本；语言模型仅产生示范请求，不产生机器人动作。 |
| [Mulligan: Performance-Guided Data Collection for Efficient On-Robot Learning](https://arxiv.org/abs/2610.05882) | 监督部署失败集中少数初始状态，均匀采集浪费操作员时间，丢弃失败又降低价值学习的数据利用。 | Mulligan每轮优先从已失败和未试初态启动人辅操作，把全部成败数据训练价值并联合交互模仿及动作选择，在相同监督回合预算比较提升。 | 人机收集完成98%含操作员干预，不能写为独立策略成功率；2550留出盲测与收集预算分开。 |
| [From Sample to Subset Construction: Coverage-Aware Curation of Robot Demonstrations](https://neurips.cc/virtual/2026/poster/150964) | 示教逐条评分忽视保留集合覆盖依赖，冗余掩盖稀少关键转移。 | PROSE预算内顺序选轨迹或片段，合闭环回报影响运动学可靠及相对已选集合的新覆盖边际收益，避免独立好坏筛选。 | 稀少不完美片段可有价值，不将运动可靠性等同简单删除全部失败轨迹 |
| [Splat2Act: From Captured Scenes to Personalized VLA Manipulation](https://arxiv.org/abs/2610.13945) | 真实建筑中的操作策略缺少低成本场景专属数据，通用模型难以直接适应大范围空间与个性化物体。 | 从手机视频重建可碰撞高斯场景并发现放置位置，由特权仿真教师生成演示，再分阶段微调通用及个性物体操作策略。 | 62.5%为聚合评测；12项任务子集实机63.9%与仿真66.7%分别报告，手机采集不代表无需特权教师。 |

### 泛化迁移

**子问题：**环境外观、物体与任务变化；跨机器人形态和控制接口；仿真与真实动力学差异；跨传感器或数据域适配。 已见单臂技能能否重组成未见双臂配对。

**主要解决路线：**分离任务相关因素与背景；学习共享或形态无关表示；测试时适配与分布校准；动力学翻译及仿真RL对齐；按适用范围转移知识。 将可复用技能解耦，并在需要合作时门控通信。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [World Translation: Minimizing Sim-to-Real Gap with Backward Dynamics Extraction and Unpaired Domain Translation](https://arxiv.org/abs/2607.18154) | 突然接触等隐藏动力学无法从历史恢复，前向真实模型难迁移仿真策略。 | World Translation从已观察转移反向提取隐藏动力学，以无配对域翻译保留动力学内容并转换仿真实际风格。 | 摘要级材料；作者报告，未独立复现。 |
| [One Demonstration, Many Objects: Generalizing Manipulation via Local Contact Geometry](https://arxiv.org/abs/2609.01938) | 人类示教灵巧操控缺少精准接触激励，且易对新物体和实机迁移失效。 | 以接触点局部几何定义操控策略和接触中心奖励，使局部结构保留时跨形状尺寸质量摩擦迁移。 | 泛化依赖局部接触结构保留。 |
| [Impedance Cloning: Learning Equilibrium Point Parameters for Contact-Rich Manipulation](https://arxiv.org/abs/2609.30842) | 接触表面几何偏移时，复制示教轨迹难维持接触力并被动吸收不确定性。 | Impedance Cloning用粒子滤波从无需力/力矩传感器的双边遥操作提取刚度及平衡点，模仿生成运动的阻抗参数而非输出轨迹。 | 摘要级材料；作者报告，未独立复现。 |
| [When Listening Becomes Easier: Scrubbing Visual Cues for Shortcut-Free VLAs](https://arxiv.org/abs/2610.10912) | VLA易依赖视角背景捷径，补采多样示教昂贵且骨干纠偏能力不一致。 | 以免rollout动作裕量及逐层表示诊断视觉捷径早入与语言后修差异，再用task scrubbing域对抗训练削弱视觉捷径促进听指令。 | 摘要未展开对抗支路目标与训练构造，可选细节延后 |
| [SkillVLA: Tackling Combinatorial Diversity in Dual-Arm Manipulation via Skill Reuse](https://arxiv.org/html/2603.03836v1) | 双臂技能组合数爆炸，左右动作纠缠限制已学单臂技能新配对复用。 | 冻结高层VLM输出逐臂文字，独立低层动作专家以合作门控跨注意互通；通信开关BC误差差及先验平滑保守正则训练门控复用技能。 | 父核正文补逐臂高低层及合作门控训练；技能复用非固定完全禁双臂通信 |

### 感知状态

**子问题：**遮挡、低纹理、天气和水下退化；度量深度、位姿与三维结构；触觉、雷达等异构信号；历史信息和可控动力因素的表示。

**主要解决路线：**几何约束与基础模型先验融合；多传感器和时序融合；显式不确定性及后验估计；保留状态与动作差异的表征目标；按查询选择模态。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [OASIS-Map: Object-Level Change Detection in Multi-Session Mapping using Semantic Correspondence Matching](https://arxiv.org/abs/2607.14899) | 长期巡检中物体出现消失或移动使静态地图失效，遮挡和分割误差又妨碍跨访问对象关联。 | 通过时间观测间稠密图块语义对应检测场景变化，并在重访时增量关联对象维持时空一致地图。 | 摘要级材料；作者报告，未独立复现。 |
| [VIDAR: Visual-Inertial Dense Alignment and Reconstruction via a Geometric Foundation Model](https://arxiv.org/abs/2607.17171) | 单目几何基础模型缺少稳定度量尺度，多帧预测难可靠融合为全局三维场景。 | 以视觉惯性里程计提供位姿尺度和统一坐标，条件化或解耦对齐深度基础预测并融合稠密几何。 | 摘要级材料；作者报告，未独立复现。 |
| [PredTac: Learning Contact-Rich Manipulation with Predicted Touch](https://arxiv.org/abs/2609.15198) | 触觉传感硬件的标定同步和维护增加接触操作策略学习部署成本。 | 先用触觉监督训练由因果视觉及机器人状态预测接触的模型，再将预测触觉作为下游策略训练执行的显式接口。 | 触觉预测器训练仍需要触觉监督，下游不需要实测触觉输入。 |
| [Acting from Belief, Looking When Needed: A Bayesian Spatial World Model for Navigation under Intermittent Perception](https://arxiv.org/abs/2610.11591) | 共享传感器会被其他任务占用，导航需在间歇观察下行动且判断何时必须再看。 | ALONE用执行动作传播空间信念、学习结构先验补未知，再解码空间与可靠图给规划，仅当导航受可靠性不足阻碍且新证据有用时请求深度。 | 低观察比例为成功试次中位数，不等同所有失败条件也如此节约。 |

### 指令理解与任务推理

**子问题：**语言描述如何对应空间目标和可执行动作；复杂指令的分解；生成计划是否忠实于意图；任务规范如何学习与检查。目标、锚点和参考系歧义；缺少观察与意图不明确如何区分。

**主要解决路线：**视觉或空间工具辅助推理；显式抓放锚点；结构化任务、程序和时序逻辑；由感知、规划与执行模块共同落实指令。主动重观察和交互澄清；对多种指代解释保留联合后验后再决定是否执行。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [Logic-VLA: A Temporal Logic Conditioned Vision-Language-Action Model](https://arxiv.org/abs/2608.20556) | 自然语言VLA指令难精确表达安全时空要求，也难让同一策略适应不同形式规范。 | 用预训练STL语法图编码器条件化VLA，先以满足示范监督微调，再对匹配满足违反轨迹作流匹配偏好优化。 | 闭环仿真；满足率提升不等同形式安全保证。 |
| [GRAVA: Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving](https://arxiv.org/abs/2609.15169) | 驾驶VLA的中间推理与真实场景证据及可执行轨迹联系薄弱，易丢失动作相关物理信息。 | 将视觉区域和自车物理状态组织成轨迹锚定类型图并序列化推理，统一自回归生成可确定解码轨迹的规划动作。 | 长尾结果含内部基准，摘要未给道路实车部署。 |
| [Active Perception for Embodied Disambiguation](https://arxiv.org/abs/2608.13605) | 具身目标歧义不仅源于用户意图，也来自遮挡、视角限制和缺失物理证据。 | 主动改变观测获取目标证据，由视觉语言模型根据累计视觉和交互信息选择继续观察、请求澄清或确定目标。 | 摘要级材料；作者报告，未独立复现。 |
| [RefGuard: Identity-Aware Language-Guided Robot Manipulation via Joint Target-Anchor-Frame Grounding](https://arxiv.org/abs/2609.06221) | 重复物体、锚点歧义和参考系不确定会使机器人在几何正确时操作错误实例。 | 维护目标锚点参考系联合后验，从RGB-D构建条件化对象图，并按后验选择执行、澄清、重观察或中止。 | 摘要级材料；作者报告，未独立复现。 |
| [Rephrase Before You Act: Characterizing and Mitigating Language Sensitivity in Vision-Language-Action Models](https://arxiv.org/abs/2610.10526) | VLA对近义措辞也可能成功率剧变，语言增强训练仍未消除敏感性，部署逐步验证又增加开销。 | 先对少量训练任务评分多措辞，让语言模型把证据蒸馏为十至二十改写规则，部署只按规则改写一次输入并保持机器人策略冻结。 | 规则需预先任务评分；16–27%为相对提升非百分点，oracle措辞搜索上界非实际规则全部收益。 |

### 长任务与记忆

**子问题：**局部预测正确却不持续接近目标；长期观察不能形成一致记忆；动作段衔接；计划随环境变化失效。

**主要解决路线：**进度条件表示和规划目标；分层技能与边界建模；实体记忆和反思检索；执行中检测计划过期并重规划。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [ActSWM: Action-Sensitive World Models for Long-Horizon Planning in Open-World Games](https://arxiv.org/abs/2607.26712) | 潜世界模型长期预测可看似准确却对不同动作失去区分，削弱规划能力。 | ActSWM以转移分离原则约束潜展开保持替代动作未来差异并可恢复局部动作，在Minecraft闭环规划验证。 | 通用交互策略规划支撑技术，摘要未明确机器人实验。 |
| [TRACT: Temporally Routed Action Chunks with Chronological Phase Authority for Contact-Rich Manipulation](https://arxiv.org/abs/2607.29285) | 跨程序阶段的动作块若统一条件于当前阶段会产生时间结构错配。 | TRACT显式估计当前到下一阶段边界，以单调分布路由动作查询，并累计意图与实际响应缺额补偿受阻接触运动。 | 作者说明完整方案对照未隔离路由与其他生成器组件差异。 |
| [OCC4M: Object-Centric 4D Memory for Spatiotemporal Reasoning in Long-Horizon Manipulation](https://arxiv.org/abs/2609.28798) | 长任务操作需记住视野外物体、时间身份及容器内容，原始历史帧难稳定支持空间时间推理。 | OCC4M在统一世界坐标保持物体轨迹并显式记录时间、运动与容纳关系，由VLM查询结构记忆选择目标交给无历史低层执行器。 | 实机85%联合记忆准确率与45%完整两阶段任务完成率口径不同。 |
| [Long-WAM: Scaling the Context of World-Action Models](https://arxiv.org/abs/2610.10528) | WAM需要历史推运动进度，但长上下文算慢，提供历史也不保证模型真正利用。 | Long-WAM先从无动作机器人与自视视频自回归预训练，再保因果历史到未来结构做动作适配，配流式编码、异步执行及硬件加速支持长时记忆控制。 | 19.2秒历史收益依赖AR初始化；107.4毫秒为5090单块含潜未来预测；动态叠杯95%为所测任务，两基线各20次全失败，不推广所有动态操作。 |
| [PRISM: Primitive Routing via In-context Skill Mixing for Lifelong VLA](https://neurips.cc/virtual/2026/poster/154388) | 终身VLA共享更新遗忘技能且难在复合阶段调用已会原子能力。 | PRISM存各技能动作专家LoRA为原语库，冻结原语让按块VLM路由读进度混合，动作派生oracle标签训路由及视觉语言模块。 | 原语参数保留是构造保证，视觉语言仍更新，不等同所有行为理论绝不遗忘 |

### 动作与物理交互

**子问题：**接触精度、力和形变；抓取与灵巧操作；高速或受约束运动；动作表示与实际动力学不匹配。

**主要解决路线：**结合动力学、几何与接触先验的策略；最优控制和学习控制组合；力反馈与动作监督改造；物理仿真和可微模型；轨迹及抓取生成。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [Reconstructing Is Not Acting: Action-Centric Latent Dynamics Modeling](https://arxiv.org/abs/2609.15189) | 视频潜动作模型重建误差低不等于动作建模好，逆模型混入外观且前向模型可能绕过动作。 | 用可学习动作查询和门控聚合提取动作转变线索，再以逐层交互动作令牌持续调制前向状态表示。 | 摘要级材料；作者报告，未独立复现。 |
| [From Prediction to Decision: World-Model-Guided Action Selection for Continuous Pile Excavation](https://arxiv.org/abs/2609.15382) | 装载机连续挖掘每次改变地形，需实时预测并排序动作后果以减少完成料堆清理的铲次。 | 生成多候选铲挖并剔除几何不可行动作，联合预测带符号地形变化与装载量，执行最大预测载荷候选后重新观测规划。 | 决策收益来自仿真对照，实体实验支持完整闭环可行性，不能混作同一量化效果。 |
| [An Efficient Algorithm for Minimum-Pressure Growth Planning of Vine Robots](https://arxiv.org/abs/2609.18070) | 软生长机器人路径规划忽视所需压力，最短路径可能压力过高而爆裂。 | 推导任意形状生长压力，证明最优路径只在特定障碍点折转，将规划化为时变权重最短路并用改进Dijkstra求解。 | 二维全局最优，三维为离散渐近精确的近似，不统称三维全局最优。 |
| [Imp-ACT: Adaptive Impedance Control and Action Chunking with Transformers to Learn Contact-Rich Manipulation from Demonstrations](https://arxiv.org/abs/2609.31225) | 视觉动作策略通常固定柔顺性，接触任务难同时保持运动方向精度与正交方向低作用力。 | 遥操作时自调阻抗沿运动方向调刚度、正交方向保持柔顺，同步记录实际刚度与动作；ACT从视觉本体力旋量预测位姿夹爪和刚度。 | 摘要级材料；作者报告，未独立复现。 |

### 自主学习与持续改进

**子问题：**交互样本昂贵且奖励稀疏；知识如何随探索更新；模型偏差如何误导策略；少量经验如何产生可靠学习信号。 人工重置能否减少而不破坏持续采集。

**主要解决路线：**模型辅助或层次强化学习；增量知识与符号规划；观测和潜状态预测辅助学习；利用执行反馈、价值估计和失败经验改进策略。奖励模型与物理失败合成；执行动作空间的结构化探索；按价值调度有限想象；MPC提供初期引导。 前向技能与恢复动作配对形成自重置采集循环。 单次示教提供自动干预窗口与恢复参考，成熟后停止指导；需单独核算恢复点设置和实际交互成本。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [RoboClaw: An Agentic Framework for Scalable Long-Horizon Robotic Tasks](https://arxiv.org/abs/2603.11558) | 采集学习执行分离导致大量人工重置及长程多策略调度脆弱。 | RoboClaw用VLM统一生命周期，以前向操作和逆恢复成对耦合形成自主重置采集循环，迭代改策略并同语义上下文编排原语长任务。 | 人力显著减少但仍为最少人工介入，不称完全无人工。 |
| [One Demonstration Is Enough for Real-World Robotic Reinforcement Learning](https://arxiv.org/abs/2607.01651) | 真实机器人强化学习交互昂贵，少示教条件下探索还容易偏离安全轨迹并陷入局部最优。 | 以一次示教建立滑窗自动干预、预定义恢复点和自主完成后停止指导机制，在训练中自动纠错并保留后期探索。 | 一次示教不等于零人工设置，仍采用预定义恢复点；100%指所测插入任务。 |
| [From Pretraining to Proficiency: Real-World Subtask RL for Long-Horizon Manipulation with Minimal Human Intervention](https://arxiv.org/abs/2609.21788) | 长程基座策略仅少数子任务失败，整段补示范重复已有技能且全程稀疏RL难学习。 | PARTS冻结基座提供标称动作，代理生成选择器验证器给瓶颈残差局部奖励，交替在线RL和成功重加权重训以集中真实练习。 | 人类仍需初始化识别瓶颈并按需物理复位，不称全无人自主训练。 |
| [When Life Gives You BC, Make Q-functions: Extracting Q-values from Behavior Cloning for On-Robot Reinforcement Learning](https://arxiv.org/abs/2605.05172) | BC无法自导在线进步，离线转在线分布变化易覆盖原有好动作。 | Q2RL少交互从BC提Q，再据各Q选择BC或RL动作采样训练，保留好示教动作同时在线改精细接触控制。 | 1–2小时为所测在线任务，不泛化所有机器人任务成本 |
| [EmbodiedRSI: Active Continual Robot Learning Through Hypothesis-Guided Co-Evolution](https://arxiv.org/abs/2610.10498) | 机器人基础策略变化域下降，持续改代码技能的代理又常低效消耗实体试验。 | EmbodiedRSI以快慢系统保竞争代码技能假设图，按信息价值选可辨试验，用结果共同进化代码技能，并以奖励选择分层记忆帮助后续快系统改进。 | 模型本体预训练与代理程序学习区分；实体71.3%为所测任务，未称全系统无需更新或无试验。 |

### 安全、鲁棒与恢复

**子问题：**执行前能否识别失败；分布外观察与感知攻击；约束和碰撞风险；失败后的恢复；长期硬件退化。

**主要解决路线：**运行时监测与反事实探测；不确定性校准和弃权；约束规划与安全控制；显式恢复、重规划和分层监督；鲁棒传感与可信锚点。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [Learning Robust Execution in Robotic Manipulation with Agentic Reinforcement Learning](https://arxiv.org/abs/2607.13818) | 机器人操作误差累积易破坏执行，现有VLA缺少实时稳定性评估与恢复机制。 | 用两种运行质量指标评估近期历史，由代理强化学习选择执行模式，并在退化时恢复到先前名义状态继续任务。 | 摘要级材料；作者报告，未独立复现。 |
| [CheckVLA: Execution-Time Verification with Action-Conditioned World Model for Long-Horizon Mobile Manipulation](https://arxiv.org/abs/2607.26789) | VLA开放动作块执行后的偏差无法由提交时置信度发现，残余动作会继续放大错误。 | CheckVLA用冻结动作条件世界模型校验观测，保形阈值触发延迟可执行后缀重写，并用关键帧库保留已完成进度。 | 仅仿真评估执行期修复。 |
| [MAGMA-GEN: Validated Recovery Supervision from Ambiguous Failures via Counterfactual Re-Execution](https://arxiv.org/abs/2609.20056) | 长时层级操作失败难区分语义错误和低层随机执行失败，缺少可靠恢复状态监督。 | 特权教练提出早期局部纠正候选，仅保留同状态匹配条件反事实重执行确实改善后续进度的恢复示例。 | 教练诊断允许有误，可靠性来自后续重执行筛选而非假设诊断正确。 |
| [FoldBack: Self-Correcting Masked Generative Policy for Long-Horizon Garment Folding](https://arxiv.org/abs/2610.10462) | 长折衣轨迹在漏抓滑抓后仍继续执行会放大错误，需检测、退回可重试状态并局部修复。 | FoldBack按抓放事件对齐精化和核验，保留成功抓握退回失败前配置，再避开旧失败点仅重生成失败片段及部分未来动作，无需恢复示教或基策略重训。 | 75.2%为最终折叠成功，0.837为最终掩码IoU；回滚保持已成功抓握不等于任意物理状态可完全恢复。 |
| [Seeing is Believing: Certified Perception-Based Control from Learned Visual Representations via System Level Synthesis](https://roboticsconference.org/2026/program/papers/172/) | RGB输出反馈非线性部分可见带噪，学视觉控制难同时保鲁棒约束。 | 从预训视觉学低维观测与状态相关误差界，以SLS优化因果仿射时变输出反馈并顺序凸求解，主动取信息同时控约束。 | 认证依赖观测误差界及模型假设，零实测违规不能独立证明任意场景安全 |

### 人机与多机协作

**子问题：**异步代理的依赖与冲突；他人意图、信任和未知目标；遥操作及辅助交互；共享知识的适用范围。

**主要解决路线：**显式同步和死锁检测；博弈与意图推断；可解释交互和多模态反馈；按范围共享记忆；分布式协调与共享场景状态。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [Let the Body Follow: Coupled Egocentric Control for Whole-Body Robot Teleoperation](https://arxiv.org/abs/2607.16095) | 全身遥操作需同时协调头臂躯干和基座，显式按钮调整增加认知负担且易碰运动学边界。 | 耦合自我视角控制将头俯仰偏航和手部高度映射到躯干基座，并在手到工作区边界时自动平移基座。 | 摘要级材料；作者报告，未独立复现。 |
| [Learning Personalized Safety Interventions for Haptic Human-Robot Shared Control](https://arxiv.org/abs/2607.19534) | 触觉共享控制预定义安全介入方式难匹配不同用户及应用的安全偏好。 | 用稀疏用户触觉示范，通过可微屏障优化层学习安全参数，使介入反馈接近示范偏好。 | 摘要级材料；作者报告，未独立复现。 |
| [Belief-Aware Influence and Trust (BAIT): Shaping Human Belief During Repeated Human-Robot Interaction](https://arxiv.org/abs/2607.25327) | 重复人机交互中人类会持续更新对机器人的信念，孤立决策或不可预测的影响策略会损害长期信任。 | 分层粒子滤波同时估计快速策略和慢速信念变化，结合信念感知MPPI在硬性即时绩效下权衡长期影响与信任。 | 摘要级材料；作者报告，未独立复现。 |
| [RoboPrompt: Intuitive Robot Policy Steering with Sparse Human Input](https://arxiv.org/abs/2610.10534) | 人类纠正模仿策略常需专用遥操作或为引导重训策略，稀疏直观意图难直接进入现有生成动作先验。 | RoboPrompt把绘迹点和方向转可复用动作草稿，在基策略噪声空间按扩散或流动态精化，兼顾意图与先验，再用受引导回合DAgger提升后续策略。 | 初始引导不改基策略，后续DAgger会训练；各15.5%/21.3%未明确百分点，不擅自转换。 |
| [Trend formation with sparse global sampling](https://arxiv.org/abs/2610.10521) | 无中心与稠密通信的群体如何用极少非局部信息形成全局运动秩序尚不清楚。 | 自推进代理周期转向少量随机同伴子集的质心，以几何转弯过滤采样噪声打破对称，推采样规模频率有序条件并在最多十五台差动机器人上验证。 | 产生向列有序即共同轴向，不等于所有机器人同向航行；不直接观察单个邻居不等于完全无全局采样信息。 |

### 效率与部署

**子问题：**迭代策略、规划求解与感知开销；设备算力和能耗受限；模型与技能库规模增加；实时执行的系统瓶颈。

**主要解决路线：**少步采样、动态早停和压缩；轻量模型及蒸馏；并行求解和近似搜索；按需感知、缓存与异步执行；面向硬件的系统集成。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [Just-In-Time Scene Graph Growth: Combating Perceptual Saturation in Long-Horizon Robotics](https://arxiv.org/abs/2607.13245) | 先完整构建再筛选场景图带来观测冗余与语义计算饱和，难满足边缘长程机器人低延迟要求。 | JITOMA以任务热图过滤前端观测并保留休眠锚，查询时才唤醒局部子图进行昂贵语义操作，实现按需图增长。 | 摘要级材料；作者报告，未独立复现。 |
| [JustDepth: Real-Time Radar-Camera Depth Estimation with Single-Scan LiDAR Supervision](https://arxiv.org/abs/2607.22172) | 雷达相机深度估计的多阶段处理及辅助标注依赖增加延迟并限制部署。 | 将雷达聚合为定宽一维表示，以高度融合和轻量图网络预测深度，并仅在训练时使用置信解码器。 | 摘要级材料；作者报告，未独立复现。 |
| [Think Fast, Plan Selectively: Adaptive Deliberation for Efficient Data-Driven MPC](https://arxiv.org/html/2609.32591v1) | 数据驱动MPC逐步评估大量轨迹计算昂贵，熟悉状态也持续规划妨碍实时控制。 | Fast-TD-MPC用MLP模仿MPPI，并按潜状态Mahalanobis距离切换快策略或原规划器，切回规划时清掉过期热启动。 | 摘要级材料；作者报告，未独立复现。 |
| [Muninn: Your Trajectory Diffusion Model But Faster](https://roboticsconference.org/2026/program/papers/160/) | 轨迹扩散逐步重算去噪器慢，简单缓存可能损轨迹质量和下游安全。 | Muninn离线校准内部轨迹变化探针与采样器误差系数，累计最终偏差预算，在小预测偏差步复用去噪缓存否则重算。 | 证书针对相对全算轨迹距离及选定风险容差，不等同全算策略本身真实安全 |
| [PathTime-VLA: Path-Time Decoupling for Factorized Post-Training of Vision-Language-Action Policies](https://arxiv.org/abs/2610.11771) | 固定时步动作把几何路径与人类示教节奏耦合，难分开学习跨域路线和执行速度。 | PathTime-VLA以进度路径和正区间时间分离动作，示教DAgger建先验、Speed-DQN学速度、Path-AWR按回报修路径，再由条件动作专家实现。 | 时间改善统计于成功试次，完整方法和固定速基线成功分别58/60及57/60，不称大幅成功提升。 |

### 本体与传感设计

**子问题：**现有机构、材料或传感方式限制可达动作与可靠感知；成本、尺寸、载荷和适应性的取舍。

**主要解决路线：**软体、可变形与模块化机构；传感和机械结构联合设计；主动光学及新型触觉；低成本部件复用；设计优化与物理验证。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [Configuration-Induced Passive Self-Rotation for Perception-Enhanced Autonomous Flight](https://arxiv.org/abs/2607.17646) | 机载传感视场有限，旋转扩大覆盖又需平衡刷新率与飞行性能。 | 调节三旋翼后臂构型控制被动自转工作点，并以分层规划控制和引导点重规划实现持续旋转下自主飞行巡检。 | 摘要级材料；作者报告，未独立复现。 |
| [Distributed Acoustic Localization Array Deployed Using a Soft Everting Vine Robot](https://arxiv.org/abs/2607.20392) | 受限灾害环境中软生长机器人需在形变体上部署声学阵列并定位远近遇险声源。 | 沿藤蔓机器人布置麦克风，用动态SRP-PHAT兼顾远场方向与近场三维定位，并测试安装位置和形状对精度影响。 | 摘要级材料；作者报告，未独立复现。 |
| [Towards Ultrafast Depth Sensing Via Active Event-based Stereo Vision](https://arxiv.org/abs/2607.23684) | 传统主动立体相机在快速运动场景受帧式成像限制，难以低延迟获取稠密深度。 | 结合双目事件相机和红外图案投影器构建原型与数据，以轻量网络、动态代价体和时间一致性估计视差。 | 传感支撑技术，摘要未明确机器人实验。 |
| [TacGooseBumps (TacGB): Retrofitting Normal-Only Tactile Sensors with Shear Encoding for Learning Contact-Rich Manipulation](https://arxiv.org/abs/2609.34006) | 视觉与法向触觉分不清剪切相关接触状态，导致插入定位等任务选择错误动作。 | TacGB在原法向传感器上覆盖被动圆顶膜，让切向载荷倾斜圆顶并重排压力图，策略直接学习该可重复编码而无需重建剪切力。 | 摘要级材料；作者报告，未独立复现。 |

### 评测与机制理解

**子问题：**模型为何失败；离线指标是否对应任务能力；评测覆盖、泄漏和混杂因素；系统及安全主张的证据边界。

**主要解决路线：**受控对照、消融与失败归因；专门诊断基准；真实执行及用户研究；带条件的理论界；综述和分类框架。

| 代表论文 | 研究问题 | 解决思路 | 证据边界 |
|---|---|---|---|
| [REVERSAL-BENCH: A Reversibility Axis and Reset Oracle for Measuring the Reset-Free RL Cliff](https://arxiv.org/abs/2609.17745) | 无重置RL依赖可逆环境，真实操控不可恢复状态可能使学习永久困住，现有评测未分离原因。 | REVERSAL-BENCH连续控制可逆性并提供重置判定真值，在多引擎几何匹配任务比较策略和安全盾，揭示不可逆吸收导致的学习断崖。 | 源摘要结尾缺句号但核心机制完整；恢复预测准确不意味着能从物理不可逆陷阱恢复。 |
| [RESCUE: Is This Alarm Worth Acting On? Counterfactual Evaluation of VLA Runtime Monitors](https://arxiv.org/abs/2610.13516) | 失败检测准确不必然意味着恢复有用，误报造成的停机和恢复代价可能抵消收益。 | 恢复模拟器和策略检查点，对比分支继续执行与脚本恢复的反事实结果，定位可恢复时点并量化误报率与恢复成本的权衡。 | 恢复对部分案例有害；理想停机重置下的误报预算也很低，不能把检测性能直接当任务收益。 |
| [System Switch: When Should a Fast Decision Model Stop and Think?](https://arxiv.org/abs/2610.09683) | 快慢双过程实时代理何时交慢推理不应只看离线准确率，置信排序和闭环计划承诺可能分别失效。 | 作者在Doom比较快策略置信校准、选择性推理转交及持续计划，对照随机转交和固定探索，分离离线问答收益与实时到出口表现。 | 闭环所有变体均未到出口；开门进展或离线准确提升不等同任务完成。 |
| [RoboQuest: Generalist Physical Agents that Search, Inspect and Test](https://arxiv.org/abs/2610.10388) | 未知环境需搜索、操作检查或试工具获取证据，通用物理代理会过早结束且难修复探索自身扰动。 | RoboQuest以十移动操作任务分搜索、操作检查及交互测试三类隐藏信息，用共同视觉动作接口比较五代理和完整示教VLA，并以给定信息技能测试分离执行与探索。 | 最佳23%和微调策略近零为完整不确定任务；摘要未明确全部实仿层级，不补硬件。 |
| [OopsieVerse: A Safety Benchmark with Damage-Aware Simulation for Robot Manipulation](https://arxiv.org/abs/2606.31993) | 完成家务也可能损坏机器人环境，现仿真缺通用可测机械热液体伤害。 | OopsieVerse把接触温度液体映机械、热与流体损伤，跨引擎DamageSim配家务任务分成败与伤害，辅助采示教学安全策略并迁真实。 | 损伤映射依模型参数，不能等同覆盖全部真实损坏模式 |

## 交叉观察与后续阅读

从本轮摘要级研究版图观察，数据问题至少包含采集成本、动作或触觉标注缺失、外部数据的任务相关性三层。减少示教条数的证据需要同时检查已有预训练数据、仿真生成数据和人工筛选成本；这些路线不能直接换算为 Carrot 所需的采集小时数。
世界模型的表示质量、预测误差、规划代价和闭环成功是不同环节。部分诊断工作发现，预测能力充分时，目标距离的几何或可达性失真仍可能限制规划；也有工作把预训练特征与对象表示本身的贡献拆开。地图应保留这些可检验的问题，而不将世界模型归成单一解决路线。
运行时读出进度、检测失败和执行恢复是三个不同问题。[RESCUE](https://arxiv.org/abs/2610.13516)恢复同一模拟器与策略检查点，对比不同干预分支，说明高检测分数未必降低任务完成成本；恢复还可能破坏原本能成功的执行。其误报预算结论依特定停机重置成本模型，不能直接套用实机。离线轨迹误差下降同样不等于真实闭环更安全。
完成任务、遵守指令、真正利用记忆和避免损伤不能用同一个成功率代替。[FindingDory](https://eccv.ecva.net/virtual/2026/poster/4986)要求记忆对行动确实必要；[BeTTER](https://eccv.ecva.net/virtual/2026/poster/5364)隔离运动学因素后干预推理；[ICBench](https://eccv.ecva.net/virtual/2026/poster/5241)保持场景不变而改变指令；[OopsieVerse](https://arxiv.org/abs/2606.31993)把损伤作为显式物理信号。这些工作提供可区分失败原因的实验设计，而非直接证明某类架构优越。
部署效率也应区分计算时延与环境变化后的反应时延。[FASTER](https://neurips.cc/virtual/2026/poster/156001)优先生成近期动作，并分析首动作时延和执行时域；[Fast-WAM](https://neurips.cc/virtual/2026/poster/149922)用受控变体区分训练期视频建模与推断期未来生成的贡献。这提供了两个独立研究问题：策略何时吸收新观察，以及为当前动作到底需要计算哪些未来信息。所报收益限作者评测设置。

## 范围与证据限制

论文中的性能提升、理论保证及适用范围均来自作者材料，本轮没有独立复现实验。不能仅凭本地图判断方法优劣、复现成本或 Carrot 的最优路线。
主分类是检索和组织工具；一篇论文可能横跨多个研究问题。方法标签与问题分类分开，不能把使用 RL、VLA 或世界模型本身视为研究问题。
跨分类关键词补查、会议候选筛选以及摘要缺失均可能造成覆盖偏差。IROS含二手目录候选，官方目录访问受限；CoRL官方覆盖仍未闭合。关联到arXiv不能独立确认录用。RSS、ECCV、NeurIPS官方目录各有已记录的筛选范围；不得将本次语料称作所有顶会全量。未确认的近似标题重复仍保留。
范围包含通用策略学习、几何感知和规划控制支撑技术以及虚拟具身交互。与机器人不相关的判断针对本轮收录范围，并非认定该工作绝无机器人迁移价值。
保守去重仍保留6条带疑似重复标记的条目。全部9,073条有标题；其中880条IROS二手候选仅有集合目录链接，没有已确认的单篇论文URL和摘要。另有1条已读摘要但方法机制证据不足。共881条信息不足单列，不根据标题补写解决方案。

## 数据与复核记录

- [逐篇中文问题与方法 JSONL](paper-summaries.jsonl)
- [原始论文目录](new-paper.md)
- [输入来源到论文的映射](summaries/source-ledger.jsonl)
- [批次、分类规范和续跑方法](summaries/README.md)
- [最终覆盖校验](summaries/final-validation.json)
- [代表论文复核记录](summaries/reviews/representatives.jsonl)
