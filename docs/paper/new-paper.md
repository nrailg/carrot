# 近三个月机器人论文清单

快照：2026-10-12。主窗口：**2026-07-12—2026-10-12（UTC 首次提交日期）**。机器人学习为主，兼顾感知、规划、控制和系统。此版是文献盘点与规则初分，未逐篇精读，不给出实验结论或优先级排名。

## 逐篇问题与方法整理

逐篇摘要提取与复核已完成（2026-10-13）：[最终问题地图](problem-map.md) · [逐篇结果](paper-summaries.jsonl) · [执行与复核目录](summaries/README.md)。9,659 条来源记录映射为 9,073 个条目，其中 7,551 条摘要支持、641 条范围外、881 条信息不足。以下原始检索目录和规则标签保持独立，不能代替逐篇研究问题分类。

## 先看覆盖与缺口

- **arXiv cs.RO：4850 篇。** API 总数 = 分页实际条数 = 唯一 ID 数；主分类或交叉分类命中 cs.RO 均保留。最新返回首次提交为 2026-10-09，未以更新日期充当新论文。
- **跨分类：1692 条补充候选。** 来自标题检索和六组摘要检索的并集，均在时间窗内且不含 cs.RO。保留噪声以便后续筛选，不将其全部称为机器人论文。
- **官方会议目录：RSS 全表 210；ECCV 相关候选 291；NeurIPS 相关候选 368。** NeurIPS 为未来会议补充；综合会议候选靠规则筛选，既可能漏检也可能混入无关论文。
- **未解决缺口：IROS、CoRL 的完整官方目录。** IROS 保留 1933 个二手程序候选，CoRL 与 IROS 另保留作者会议备注。二手候选、作者自报、workshop 不算官方主会录用。

各来源之间存在重叠，**上述数字不能直接相加成唯一论文总数**。JSONL 保留不同来源记录，以 `arxiv_id` 关联已可靠匹配的条目；关联未完成不代表不同论文。

## 完整目录入口

| 目录 | 条目数 | 用途 |
| --- | --- | --- |
| [arXiv cs.RO · 2026-07](catalog/arxiv-2026-07.md) | 752 | 按首次提交时间完整浏览；从 12 日开始 |
| [arXiv cs.RO · 2026-08](catalog/arxiv-2026-08.md) | 1169 | 按首次提交时间完整浏览 |
| [arXiv cs.RO · 2026-09](catalog/arxiv-2026-09.md) | 2168 | 按首次提交时间完整浏览 |
| [arXiv cs.RO · 2026-10](catalog/arxiv-2026-10.md) | 761 | 按首次提交时间完整浏览 |
| [跨分类候选池](catalog/cross-category-candidates.md) | 1692 | 查看不在 cs.RO 中的补充线索，待人工筛选 |
| [结构化元数据](new-paper-metadata.jsonl) | 9659 条来源记录 | 含标题、作者、日期、分类、会议证据、链接；不是 9659 篇去重论文 |
| [查询与源文件指纹](new-paper-sources.json) | — | 分页计数、查询式、快照 SHA-256 |
| [检索与质量记录](collection-audit.md) | — | 方法、校验、限制、后续补洞事项 |
| [执行计划](collection-plan.md) | — | 范围与验收检查表 |

## 九个会议逐项覆盖

| 会议 | 举行时间与范围决定 | 获取结果 | 来源与清单 |
| --- | --- | --- | --- |
| RSS 2026 | 07-13—07-17，窗口内，全收官方目录 | 210 条 | [官方](https://roboticsconference.org/2026/program/papers/) · [清单](catalog/rss-2026.md) |
| IROS 2026 | 09-27—10-01，窗口内，目标全收 | 官方 Paper Index 403；1933 个二手程序条目；另有 494 条作者会议备注，不等于录用数 | [官方入口](https://2026.ieee-iros.org/program/paper_index/) · [日期旁证](https://robotics.gatech.edu/gt-iros-2026) · [候选](catalog/iros-2026-candidates.md) · [作者备注](catalog/iros-2026-author-reported.md) |
| ECCV 2026 | 09-08—09-12，窗口内，扫描整表后筛相关候选 | 2997 个展示事件 → 2834 个标题与作者组 → 291 个候选 | [官方目录](https://eccv.ecva.net/virtual/2026/papers.html) · [日期](https://eccv.ecva.net/Conferences/2026/Venues) · [清单](catalog/eccv-2026.md) |
| CoRL 2026 | 11-09—11-12，未来会议，单列已公开论文补充 | 本次未取得完整官方名单；115 条作者会议备注，不等于录用数 | [官网与日期](https://2026.corl.org/) · [Program](https://www.corl.org/program) · [作者备注](catalog/corl-2026-author-reported.md) |
| NeurIPS 2026 | 12 月，未来会议，单列已公开目录补充 | 9242 个事件 → 6347 个标题与作者组 → 368 个候选；367 条 decision 含 Accept，1 条未标明 | [官方目录](https://neurips.cc/virtual/2026/papers.html) · [日期](https://neurips.cc/Conferences/2026/Dates) · [清单](catalog/neurips-2026.md) |
| ICRA 2026 | 06-01—06-05，窗口外 | 已核日期，不整场纳入 | [官方日期](https://2026.ieee-icra.org/media/press/) |
| ICML 2026 | 07-06—07-11，截止在窗口前一天 | 已核日期，不整场纳入 | [官网与日期](https://icml.cc/Conferences/2026) |
| CVPR 2026 | 06-03—06-07，窗口外 | 已核日期，不整场纳入 | [官方日期](https://cvpr.thecvf.com/Conferences/2026/News/Media_Reg_Open) |
| ICLR 2026 | 04-23—04-27，窗口外 | 已核日期，不整场纳入 | [官网与日期](https://iclr.cc/Conferences/2026/index.html) |

对于窗口外的会议，不扫入完整旧论文集；若其中论文在窗口内首次提交 arXiv 并命中本轮查询，仍保留在 arXiv 主池或候选池。对于窗口内举行的会议，目录中论文的 arXiv 首次提交可能更早，不能把会议日期当成论文首次公开日期。

## 按主题浏览完整 cs.RO 主池

同一论文可有多个标签，各行不能相加。标题优先、标题无命中时回退摘要的规则分类，标签仅帮助导航，不代表已经判断论文贡献。下方主题目录没有按热度或 Carrot 当前问题删减。

| 主题 | 条目数 | 完整索引 |
| --- | --- | --- |
| VLA、基础模型与适配 | 604 | [打开](catalog/topic-vla.md) |
| 数据、示教、遥操作与数据效率 | 514 | [打开](catalog/topic-data.md) |
| 模仿学习、扩散策略与动作表示 | 309 | [打开](catalog/topic-policy.md) |
| 强化学习、在线适配与自主改进 | 331 | [打开](catalog/topic-rl.md) |
| 世界模型、仿真与 Sim-to-Real | 798 | [打开](catalog/topic-world.md) |
| 操作、灵巧手、抓取与触觉 | 1118 | [打开](catalog/topic-manipulation.md) |
| 人形、运动技能与足式运动 | 437 | [打开](catalog/topic-locomotion.md) |
| 导航、移动机器人与自动驾驶 | 672 | [打开](catalog/topic-navigation.md) |
| 感知、SLAM、三维与状态估计 | 577 | [打开](catalog/topic-perception.md) |
| 规划、控制、接触动力学与安全 | 1509 | [打开](catalog/topic-planning.md) |
| 评测、泛化、鲁棒性与失败检测 | 1225 | [打开](catalog/topic-evaluation.md) |
| 系统、部署、硬件与计算效率 | 921 | [打开](catalog/topic-systems.md) |
| 多机器人、人机交互与协作 | 352 | [打开](catalog/topic-interaction.md) |
| 其他机器人方向 | 34 | [打开](catalog/topic-other.md) |

## 分月份浏览样本

以下从各主题按月抽取少量条目，帮助开始浏览；不是精选榜单，也不取代上面的完整目录。采用固定、可复现的抽样：仅在标题直接命中主题词的条目中，每主题每月按首次提交时间排序，取中间和末尾的可用条目，跨主题避免重复。主题覆盖与月份分布比热度排序优先。所有论文都保留原始英文标题，不根据标题编写未经核验的研究结论。

### VLA、基础模型与适配

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-17 | [Think at 5 Hz, Act at 20 Hz: Asynchronous Fast-Slow Vision-Language-Action Inference for Closed-Loop Driving](https://arxiv.org/abs/2607.15621) |
| 2026-07-31 | [WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning](https://arxiv.org/abs/2607.29613) |
| 2026-08-14 | [Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation](https://arxiv.org/abs/2608.14379) |
| 2026-08-31 | [LightNav-0: Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation](https://arxiv.org/abs/2608.30935) |
| 2026-09-20 | [MaskVLA: Visual Masking Against Trajectory Overfitting of Vision-Language-Action Model](https://arxiv.org/abs/2609.23565) |
| 2026-09-30 | [MIKASA-Robo-VLA: Benchmarking Memory in VLA Models for Long-Horizon Manipulation](https://arxiv.org/abs/2610.00604) |
| 2026-10-06 | [Seeing the Invisible: Physics-Guided Visual Prompting for Temperature- and Radiation-Aware VLA Navigation](https://arxiv.org/abs/2610.07558) |
| 2026-10-09 | [Splat2Act: From Captured Scenes to Personalized VLA Manipulation](https://arxiv.org/abs/2610.13945) |

### 数据、示教、遥操作与数据效率

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-23 | [TableVerse: A Large-scale Tabletop Dataset with Real-world Grounded Layouts for Generalizable Manipulation](https://arxiv.org/abs/2607.21017) |
| 2026-07-31 | [Temporal Policy: History-Initialized Action Generation for Robotic Learning from Demonstration](https://arxiv.org/abs/2607.29482) |
| 2026-08-14 | [Demonstration of Space Robot Teleoperation over a Lossy and Delayed Network using ATMOS](https://arxiv.org/abs/2608.14031) |
| 2026-08-31 | [Scene Graph-based Driving Scenario Extraction for Automotive Egocentric Datasets](https://arxiv.org/abs/2609.00333) |
| 2026-09-21 | [Dexterous Robot Manipulation from Human Demonstrations via Contact-Anchored Retargeting and Residual Policy Learning](https://arxiv.org/abs/2609.24093) |
| 2026-09-30 | [DITTO-X: Forward and Reverse Teleoperation for Dexterous Manipulation and Human Intervention](https://arxiv.org/abs/2610.00781) |
| 2026-10-06 | [Beyond Retargeting: Low-Latency and Robust Humanoid Whole-Body Teleoperation with Learned Atomic Motion Primitives](https://arxiv.org/abs/2610.07891) |
| 2026-10-09 | [LAPT-Scale: Scaling Embodied Policies with Latent Action Pre-Training on Human Videos](https://arxiv.org/abs/2610.13732) |

### 模仿学习、扩散策略与动作表示

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-27 | [PAC-DP: PAC-Bayesian Diffusion Policy Learning](https://arxiv.org/abs/2607.24296) |
| 2026-07-31 | [Action Chunk Scheduling for Batched Robot Policy Serving](https://arxiv.org/abs/2608.00337) |
| 2026-08-17 | [SparkVLA: Stop-Aware Hierarchical VLA with Adaptive Action Chunking for Long-Horizon Manipulation](https://arxiv.org/abs/2608.16172) |
| 2026-08-29 | [AdaVLA: Adaptive Step Flow Matching for Training-free Acceleration of Vision-Language-Action Models](https://arxiv.org/abs/2608.29208) |
| 2026-09-18 | [Beyond Kinematics: Benchmarking Simulation Fidelity for Muscle-Driven Imitation Learning](https://arxiv.org/abs/2609.21909) |
| 2026-09-30 | [Does Continual Imitation Learning Remain Grounded? A Language-Perturbed Benchmark for Robotic Task Retention](https://arxiv.org/abs/2610.00542) |
| 2026-10-05 | [InterMimicGen: Scaling Humanoid Loco-Manipulation through Self-Evolving Motion Imitation](https://arxiv.org/abs/2610.06850) |
| 2026-10-09 | [ReAT: Residual Action Tokenization for Autoregressive Robot Policies](https://arxiv.org/abs/2610.13093) |

### 强化学习、在线适配与自主改进

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-21 | [Loco-Loco-RL: Low-Cost Terrain Mapping for Humanoid Locomotion with Reinforcement Learning](https://arxiv.org/abs/2609.19041) |
| 2026-07-31 | [LEMUR: Learning to Align with Multi-Objective Reinforcement Learning from Preference Feedback](https://arxiv.org/abs/2607.29559) |
| 2026-08-18 | [Iterative Grasp Pose Refinement: A Deep Reinforcement Learning Approach for 2D Vision](https://arxiv.org/abs/2608.17628) |
| 2026-08-31 | [CIG-RL: Curiosity-Driven Information-Guided Reinforcement Learning for Source Term Estimation in Uncertain Environments](https://arxiv.org/abs/2608.30673) |
| 2026-09-18 | [From Pretraining to Proficiency: Real-World Subtask RL for Long-Horizon Manipulation with Minimal Human Intervention](https://arxiv.org/abs/2609.21788) |
| 2026-09-30 | [RL-Guided PAC-NMPC for Probabilistically-Safe Perception-Based Navigation in Unknown Environments](https://arxiv.org/abs/2609.39854) |
| 2026-10-05 | [Task-Space Imitation Guidance for Efficient Reinforcement Learning](https://arxiv.org/abs/2610.07527) |
| 2026-10-09 | [MC-EvoBench: Auditing Experience-Driven Self-Improvement in Minecraft Agents](https://arxiv.org/abs/2610.13829) |

### 世界模型、仿真与 Sim-to-Real

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-22 | [KineBench: Benchmarking Embodied World Models via IDM-Free Kinematic Grounding](https://arxiv.org/abs/2607.19876) |
| 2026-07-31 | [BWM: A Low-Cost High-Fidelity World Simulator for Robot Learning](https://arxiv.org/abs/2607.29302) |
| 2026-08-13 | [Simulation-Aware In-Context Policy Improvement for LLM-Aided Analog Layout Refinement](https://arxiv.org/abs/2608.13767) |
| 2026-08-31 | [IMPACT: Attention Is the Interaction Map for Scalable Interaction-Aware World Model Training](https://arxiv.org/abs/2609.00161) |
| 2026-09-21 | [DexTacWAM: A Visuo-Tactile World-Action Model for Dexterous Manipulation](https://arxiv.org/abs/2609.24976) |
| 2026-09-30 | [Getting Out and Getting Back: World and Behavior Grounding in Real2Sim2Real Co-Training](https://arxiv.org/abs/2610.00821) |
| 2026-10-05 | [Sim-to-Real Transfer of Vision-Language Navigation in Continuous Environments Using an Ackermann-Steered Mobile Robot](https://arxiv.org/abs/2610.07192) |
| 2026-10-09 | [Dream to Remember: Continual Steering of Generalist Robot Policies with World Models](https://arxiv.org/abs/2610.13816) |

### 操作、灵巧手、抓取与触觉

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-21 | [MVP-Tac: A Miniaturized Dual-Modal Vision and Photoelastic Tactile Sensor for Robot-Assisted Minimally Invasive Surgery](https://arxiv.org/abs/2607.18660) |
| 2026-07-31 | [Developing Combined Manipulation and Locomotion Skills with Interaction Representation and Skill Composition](https://arxiv.org/abs/2608.00208) |
| 2026-08-16 | [Tabletop Pen Manipulation With a Vision-Guided 4-DoF Arm](https://arxiv.org/abs/2608.15968) |
| 2026-08-31 | [Autonomously Acquiring Robot Manipulation Skills with Language-Driven Quality-Diversity](https://arxiv.org/abs/2608.30983) |
| 2026-09-20 | [G6D: Geometric Learning-Free RGB-D 6D Pose Solver for Robotic Manipulation](https://arxiv.org/abs/2609.23566) |
| 2026-09-30 | [TacDyn-WAM: Learning Implicit Tactile Dynamics in a Heterogeneous Visuo-Tactile World Action Model](https://arxiv.org/abs/2610.00638) |
| 2026-10-06 | [IronMan: Information-Constrained Video-Action Learning for Robot Manipulation](https://arxiv.org/abs/2610.07961) |
| 2026-10-09 | [SmolWAM: A Latent World-Action Model for Robotic Manipulation on Edge Devices](https://arxiv.org/abs/2610.13740) |

### 人形、运动技能与足式运动

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-20 | [Towards Torque-Driven Reinforcement Learning for Quadruped Locomotion](https://arxiv.org/abs/2607.18365) |
| 2026-07-31 | [Balancing of Humanoid with Object Mass: Trade-off Analyses and Lifting Control](https://arxiv.org/abs/2607.29625) |
| 2026-08-17 | [Trajectory-Level Automatic Curriculum Learning for Legged Locomotion on Unstructured Terrain](https://arxiv.org/abs/2608.16164) |
| 2026-08-31 | [SleepWalking: Privileged Representation Shaping for End-to-End Blind Locomotion in Legged Robots](https://arxiv.org/abs/2608.30883) |
| 2026-09-18 | [Duty Factor Predicts Robust Constrained Quadrupedal Locomotion Across Gait Types](https://arxiv.org/abs/2609.22073) |
| 2026-09-30 | [Reactive Humanoid Multi-Contact Using Learned Stability Models](https://arxiv.org/abs/2610.00823) |
| 2026-10-06 | [Model-Based Geometry-Aware Generative Optimization for Constrained Locomotion Planning](https://arxiv.org/abs/2610.07772) |
| 2026-10-09 | [ATHLETE: Learning Reactive Humanoid Tennis via Trajectory-Guided Motion Matching](https://arxiv.org/abs/2610.13572) |

### 导航、移动机器人与自动驾驶

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-21 | [Beyond Transformers: Linear Attention Policy for Open-Vocabulary Object Goal Navigation](https://arxiv.org/abs/2607.18794) |
| 2026-07-31 | [Latent-Centroid Steering: Single-Pass Classifier-Free Guidance for Command-Aligned Autonomous Driving](https://arxiv.org/abs/2608.00237) |
| 2026-08-13 | [HumanoidVLN: A Physics-Grounded Simulator and Benchmark for Vision-Language Navigation Across Diverse Humanoid Embodiments](https://arxiv.org/abs/2608.12860) |
| 2026-08-31 | [Beyond Textual Chain-of-Thought: A Survey on Action-Grounded Reasoning in Autonomous Driving](https://arxiv.org/abs/2609.01659) |
| 2026-09-20 | [Marginal Calibration Does Not Compose: Hidden Dependence in Modular Robot Navigation](https://arxiv.org/abs/2609.23731) |
| 2026-09-30 | [Social-WM: Safety-Aware Latent World Models for Robot Social Navigation](https://arxiv.org/abs/2609.40177) |
| 2026-10-06 | [Mitigating Concept Drift in QoS Prediction for Teleoperation of Autonomous Vehicles Using Historic Data](https://arxiv.org/abs/2610.08297) |
| 2026-10-09 | [HorizonNav: Geometry-Guided Video Diffusion for Anticipatory 3D Robot Navigation](https://arxiv.org/abs/2610.13849) |

### 感知、SLAM、三维与状态估计

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-20 | [SLAM in Low-Light Environments: Project Report](https://arxiv.org/abs/2607.17699) |
| 2026-07-31 | [Belief-Space Perception Routing under Coupled Sensor Faults and Compute Contention](https://arxiv.org/abs/2608.00322) |
| 2026-08-14 | [Geometry-Aware Online Mapping for 3D Gaussian Splatting SLAM](https://arxiv.org/abs/2608.14902) |
| 2026-08-31 | [DARP: A Calibrated Dual-Arm RGB-D-IR Dataset for Multi-View Robotic Perception](https://arxiv.org/abs/2608.31002) |
| 2026-09-18 | [NeuRIO: A Streaming Neural Estimator for Zero-Shot Sim-to-Real Multi-Robot Relative Inertial Odometry](https://arxiv.org/abs/2609.21707) |
| 2026-09-30 | [Measuring Asset and Scene Reconstruction Effects in Real-to-Sim Robot Evaluation](https://arxiv.org/abs/2610.00731) |
| 2026-10-06 | [Learning Grasp Targeting from Point Clouds for Log Pile Clearing on a Hydraulic Crane](https://arxiv.org/abs/2610.07613) |
| 2026-10-09 | [Beyond the Snowflakes: Disentangling Weather Effects from Confounding Meta-Factors in LiDAR Object Detection](https://arxiv.org/abs/2610.13891) |

### 规划、控制、接触动力学与安全

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-21 | [On the Limits of Sampling-Based Reachability: Geometry, Dynamics, and Sample Complexity](https://arxiv.org/abs/2607.18606) |
| 2026-07-31 | [Towards General Language-Conditioned Latent Safety Filters](https://arxiv.org/abs/2608.00315) |
| 2026-08-16 | [Correcting Learning-based Perception for Safety](https://arxiv.org/abs/2609.22108) |
| 2026-08-31 | [Multi-robot Learning-based Informative Path Planning Using Spatio-Temporal Gaussian Process Kalman Filter](https://arxiv.org/abs/2609.05515) |
| 2026-09-18 | [VLPSA: Vision-Language-Poisson-Safe Actions for Full-Body Safety of Learned Policies](https://arxiv.org/abs/2609.22462) |
| 2026-09-30 | [ECoMEM: Explicit Concept Memory for Memory-Dependent Robot Control](https://arxiv.org/abs/2610.00801) |
| 2026-10-06 | [From Model to Prototype: Design and Motor-Flap Propulsion Control of a Twin-Wing Metamorphic UAV](https://arxiv.org/abs/2610.07896) |
| 2026-10-09 | [Mapless, Stabilizing, and Anti-Stuck Flipper Control for Articulated Tracked Robots](https://arxiv.org/abs/2610.13838) |

### 评测、泛化、鲁棒性与失败检测

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-20 | [RynnBrain 1.1: Towards More Capable and Generalizable Embodied Foundation Model](https://arxiv.org/abs/2607.17977) |
| 2026-07-31 | [Diagnosing Compositional Generalization in Sequential Robot Tasks](https://arxiv.org/abs/2607.29687) |
| 2026-08-15 | [HP2-SLAM: Adaptive Hybrid ICP for Robust and Efficient LiDAR SLAM](https://arxiv.org/abs/2608.14996) |
| 2026-08-31 | [CoLT-Drive: Counterfactual Long-Tail Benchmarking and Knowledge-Preserving Adaptation for Driving Affordance Prediction](https://arxiv.org/abs/2609.00242) |
| 2026-09-19 | [M3GA-Wild: A Large-Scale Dataset and Benchmark for Multi-Modal Multi-session Ground-to-Aerial Place Recognition in Forests](https://arxiv.org/abs/2609.23003) |
| 2026-09-30 | [CF-JEPA: Improving Robustness of JEPA World Models via Controllability Factorization](https://arxiv.org/abs/2610.00727) |
| 2026-10-06 | [Nine Trials to Recover: A Reproducible Benchmark for Repertoire-Free Soft-Robot Damage Adaptation](https://arxiv.org/abs/2610.07616) |
| 2026-10-09 | [BT-Agent: Deploying Onboard Robot Agents with Behavior Trees and Multi-Fidelity Evaluation](https://arxiv.org/abs/2610.13862) |

### 系统、部署、硬件与计算效率

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-18 | [GLidE-SLAM: GL-Accelerated Indirect-Direct Embedded SLAM](https://arxiv.org/abs/2607.16897) |
| 2026-07-31 | [FibVLA: An Efficient Temporal Vision-Language-Action Model with Fibonacci Sampling](https://arxiv.org/abs/2607.29596) |
| 2026-08-11 | [BooST: Bridging Semantics and Motions for Efficient Skill Transfer](https://arxiv.org/abs/2608.10600) |
| 2026-08-31 | [Deploying and Evaluating a Smart-Agriculture Agentic Engine for Full-Season Soybean Farm Operations](https://arxiv.org/abs/2609.00106) |
| 2026-09-21 | [VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2609.25376) |
| 2026-09-30 | [Spatial Strategies, Not Actions: Vector-Quantized Geodesics as Tools for LLM-Driven Agents](https://arxiv.org/abs/2610.00613) |
| 2026-10-06 | [Seeing Through the Displaced Frame: Privileged Noise Distillation for Vision-Force Precision Assembly](https://arxiv.org/abs/2610.07745) |
| 2026-10-09 | [A Local Agentic AI Framework to Generate Edge Case Scenarios for Safety Assurance](https://arxiv.org/abs/2610.13479) |

### 多机器人、人机交互与协作

| 首次提交 | 论文 |
| --- | --- |
| 2026-07-17 | [PACE: Persona Adaptation through Conversational Elicitation in Human-Robot Interaction](https://arxiv.org/abs/2607.15579) |
| 2026-07-31 | [MROPE: A Multi-Robot Safe Cooperative Strategy via combined Predictive Safety Filters and Ellipse-based Constraint Compression](https://arxiv.org/abs/2607.29203) |
| 2026-08-17 | [Pluralistic Human-Robot Interaction: Designing for Robot Interaction with Diverse Communities](https://arxiv.org/abs/2608.16049) |
| 2026-08-31 | [Not All Agreement Counts as Corroboration: Provenance-Conserving Multi-View Fusion for Typed Action Admission in Human-Robot Collaboration](https://arxiv.org/abs/2609.01662) |
| 2026-09-18 | [AgenticSwarm: Semantic Perception and Adaptive Task Allocation for Heterogeneous Multi-UAV Missions](https://arxiv.org/abs/2609.21716) |
| 2026-09-30 | [STARS: From Spatiotemporal Dynamics to Social Representations in Human-Robot Interaction](https://arxiv.org/abs/2609.40245) |
| 2026-10-06 | [Safe Multi-Robot Collaborative Transport Using Density Functions](https://arxiv.org/abs/2610.08441) |
| 2026-10-09 | [Cooperative Aerial Transportation of a Payload with a Sliding Cable Attachment](https://arxiv.org/abs/2610.13665) |

## 留给下一轮分析的字段

下一轮逐篇筛选时再填写：问题与方法、机器人平台、真实/仿真实验、示教数量与采集人时、预训练数据需求、跨场景泛化、干预与失败恢复、代码/数据/权重可用性、单人复现成本、对 Carrot 的关联。当前的单人调研与采集成本约束是分析维度，不作为本轮提前排除文献的条件。
