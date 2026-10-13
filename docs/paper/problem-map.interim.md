# 机器人论文研究问题地图：中间版

> 全量提取和最终验收仍在进行。3 个 GPT‑6.1 Sol High 子代理继续处理，主代理逐批抽查；本文件只收录完整且通过结构校验的批次。

更新时间：2026-10-12T12:46:13+00:00。

当前收集 9,659 条来源记录，保守去重为 9,073 篇。完成 **145/205 批**，另含 27 篇校准结果；已有 **5,827** 篇摘要或相关性判断，880 篇缺摘要单列，剩余 **2,366** 篇待提取。

状态：摘要支持 5,302；不相关 525；信息不足 880。抽查要求已完成的批次 142/145，还有 58 项待复核；并非每篇都由主模型逐篇复核。

当前批次按输入顺序推进，以下数量受已处理子集影响，不能比较全领域热度或最终占比。

| 主问题 | 已提取数量 |
|---|---:|
| 数据与监督成本 | 300 |
| 泛化迁移 | 319 |
| 感知状态 | 792 |
| 指令理解与任务推理 | 186 |
| 长任务与记忆 | 163 |
| 动作与物理交互 | 720 |
| 自主学习与持续改进 | 230 |
| 安全、鲁棒与恢复 | 603 |
| 人机与多机协作 | 378 |
| 效率与部署 | 579 |
| 本体与传感设计 | 267 |
| 评测与机制理解 | 765 |

## 初步观察



从当前已复核子集观察，数据问题至少包含采集成本、动作或触觉标注缺失、外部数据的任务相关性三层。减少示教条数的证据需要同时检查已有预训练数据、仿真生成数据和人工筛选成本；这些路线不能直接换算为 Carrot 所需的采集小时数。

世界模型的表示质量、预测误差、规划代价和闭环成功是不同环节。部分诊断工作发现，预测能力充分时，目标距离的几何或可达性失真仍可能限制规划；也有工作把预训练特征与对象表示本身的贡献拆开。地图应保留这些可检验的问题，而不将世界模型归成单一解决路线。

运行时读出进度、检测失败和执行恢复是三个不同问题。能提前报警不意味着能恢复，离线轨迹误差下降也不意味着真实闭环更安全。代表论文需要说明验证层级。



## 各类问题与代表论文（暂选）



### 数据与监督成本



研究问题：示教与标注昂贵；人类视频缺少机器人动作标签；海量外部数据中哪些样本有用；合成数据如何满足任务需求。



解决路线：筛选与目标示教相似的人类数据；视角变换和逆动力学补动作；仿真及受任务约束的场景生成；弱监督、自监督与少量本地数据适配。



- [EgoHTR: Egocentric 4D Demonstrations of Human Terrain Traversal](https://arxiv.org/abs/2607.13472)：人形机器人借助人体先验穿越复杂地形时缺少场景感知示范，现有数据难对齐人体与挑战环境。 融合自我视角可穿戴设备和便携扫描器重建场景对齐四维人体轨迹，构建数据集并训练感知运动策略。

- [DINS-IO: Learned Inertial Odometry via Differentiable INS Consistency](https://arxiv.org/abs/2607.20232)：学习惯性里程计依赖昂贵稠密位置真值，难规模化获取监督。 将捷联惯导速度递推写为滑窗共享偏置最小二乘闭式一致损失自监督预训练，再以少量轨迹LoRA校准度量速度。 证据边界：感知、控制或系统支撑技术，摘要未明确机器人实验。

- [DISEIL: Demonstration Distillation for Sample-Efficient Imitation Learning](https://arxiv.org/abs/2609.08123)：交互模仿虽能中断失败，却未主动选择需纠正的失败类型及示范起点。 定位首次不可靠步骤以几何描述聚类失败模式，由视觉语言及语言模型提出示范请求并经任务约束核验。 证据边界：单轮仿真且专家多为脚本；语言模型仅产生示范请求，不产生机器人动作。

- [Universal Navigation Interface: Robot-Free Data for Wheeled Robot Navigation](https://arxiv.org/abs/2609.20114)：轮式导航实机遥操作数据昂贵且依赖平台，人类自由行走路径又未必适合轮式运动。 以四轮助行器和手机采集受物理约束人类轨迹，恢复度量路径监督目标导航并迁移到电动轮椅。 证据边界：其他数据集收益依指标变化，不宣称普遍迁移优势。



### 泛化迁移



研究问题：环境外观、物体与任务变化；跨机器人形态和控制接口；仿真与真实动力学差异；跨传感器或数据域适配。



解决路线：分离任务相关因素与背景；学习共享或形态无关表示；测试时适配与分布校准；动力学翻译及仿真RL对齐；按适用范围转移知识。



- [World Translation: Minimizing Sim-to-Real Gap with Backward Dynamics Extraction and Unpaired Domain Translation](https://arxiv.org/abs/2607.18154)：突然接触等隐藏动力学无法从历史恢复，前向真实模型难迁移仿真策略。 World Translation从已观察转移反向提取隐藏动力学，以无配对域翻译保留动力学内容并转换仿真实际风格。

- [EA-Nav: Learning Safe Visual Navigation Policies with Embodiment Awareness](https://arxiv.org/abs/2607.19880)：同一视觉场景对不同机器人需不同动作，纯视觉导航歧义及强化学习成本限制跨本体扩展。 从互联网视频构建跨本体数据，以本体几何token条件化预训练，再通过高风险轨迹增广分别训练空间感知与风险纠正。

- [One Demonstration, Many Objects: Generalizing Manipulation via Local Contact Geometry](https://arxiv.org/abs/2609.01938)：人类示教灵巧操控缺少精准接触激励，且易对新物体和实机迁移失效。 以接触点局部几何定义操控策略和接触中心奖励，使局部结构保留时跨形状尺寸质量摩擦迁移。 证据边界：泛化依赖局部接触结构保留。



### 感知状态



研究问题：遮挡、低纹理、天气和水下退化；度量深度、位姿与三维结构；触觉、雷达等异构信号；历史信息和可控动力因素的表示。



解决路线：几何约束与基础模型先验融合；多传感器和时序融合；显式不确定性及后验估计；保留状态与动作差异的表征目标；按查询选择模态。



- [OASIS-Map: Object-Level Change Detection in Multi-Session Mapping using Semantic Correspondence Matching](https://arxiv.org/abs/2607.14899)：长期巡检中物体出现消失或移动使静态地图失效，遮挡和分割误差又妨碍跨访问对象关联。 通过时间观测间稠密图块语义对应检测场景变化，并在重访时增量关联对象维持时空一致地图。

- [VIDAR: Visual-Inertial Dense Alignment and Reconstruction via a Geometric Foundation Model](https://arxiv.org/abs/2607.17171)：单目几何基础模型缺少稳定度量尺度，多帧预测难可靠融合为全局三维场景。 以视觉惯性里程计提供位姿尺度和统一坐标，条件化或解耦对齐深度基础预测并融合稠密几何。

- [PredTac: Learning Contact-Rich Manipulation with Predicted Touch](https://arxiv.org/abs/2609.15198)：触觉传感硬件的标定同步和维护增加接触操作策略学习部署成本。 先用触觉监督训练由因果视觉及机器人状态预测接触的模型，再将预测触觉作为下游策略训练执行的显式接口。 证据边界：触觉预测器训练仍需要触觉监督，下游不需要实测触觉输入。



### 指令理解与任务推理



研究问题：语言描述如何对应空间目标和可执行动作；复杂指令的分解；生成计划是否忠实于意图；任务规范如何学习与检查。目标、锚点和参考系歧义；缺少观察与意图不明确如何区分。



解决路线：视觉或空间工具辅助推理；显式抓放锚点；结构化任务、程序和时序逻辑；由感知、规划与执行模块共同落实指令。主动重观察和交互澄清；对多种指代解释保留联合后验后再决定是否执行。



- [Logic-VLA: A Temporal Logic Conditioned Vision-Language-Action Model](https://arxiv.org/abs/2608.20556)：自然语言VLA指令难精确表达安全时空要求，也难让同一策略适应不同形式规范。 用预训练STL语法图编码器条件化VLA，先以满足示范监督微调，再对匹配满足违反轨迹作流匹配偏好优化。 证据边界：闭环仿真；满足率提升不等同形式安全保证。

- [GRAVA: Grounded Reasoning-to-Action Representation and Learning for Autonomous Driving](https://arxiv.org/abs/2609.15169)：驾驶VLA的中间推理与真实场景证据及可执行轨迹联系薄弱，易丢失动作相关物理信息。 将视觉区域和自车物理状态组织成轨迹锚定类型图并序列化推理，统一自回归生成可确定解码轨迹的规划动作。 证据边界：长尾结果含内部基准，摘要未给道路实车部署。

- [Active Perception for Embodied Disambiguation](https://arxiv.org/abs/2608.13605)：具身目标歧义不仅源于用户意图，也来自遮挡、视角限制和缺失物理证据。 主动改变观测获取目标证据，由视觉语言模型根据累计视觉和交互信息选择继续观察、请求澄清或确定目标。

- [RefGuard: Identity-Aware Language-Guided Robot Manipulation via Joint Target-Anchor-Frame Grounding](https://arxiv.org/abs/2609.06221)：重复物体、锚点歧义和参考系不确定会使机器人在几何正确时操作错误实例。 维护目标锚点参考系联合后验，从RGB-D构建条件化对象图，并按后验选择执行、澄清、重观察或中止。



### 长任务与记忆



研究问题：局部预测正确却不持续接近目标；长期观察不能形成一致记忆；动作段衔接；计划随环境变化失效。



解决路线：进度条件表示和规划目标；分层技能与边界建模；实体记忆和反思检索；执行中检测计划过期并重规划。



- [ActSWM: Action-Sensitive World Models for Long-Horizon Planning in Open-World Games](https://arxiv.org/abs/2607.26712)：潜世界模型长期预测可看似准确却对不同动作失去区分，削弱规划能力。 ActSWM以转移分离原则约束潜展开保持替代动作未来差异并可恢复局部动作，在Minecraft闭环规划验证。 证据边界：通用交互策略规划支撑技术，摘要未明确机器人实验。

- [TRACT: Temporally Routed Action Chunks with Chronological Phase Authority for Contact-Rich Manipulation](https://arxiv.org/abs/2607.29285)：跨程序阶段的动作块若统一条件于当前阶段会产生时间结构错配。 TRACT显式估计当前到下一阶段边界，以单调分布路由动作查询，并累计意图与实际响应缺额补偿受阻接触运动。 证据边界：作者说明完整方案对照未隔离路由与其他生成器组件差异。

- [LT-Mem: Volatility-Aware Spatio-Temporal Memory for Lifelong Scene Understanding](https://arxiv.org/abs/2608.19059)：长期机器人地图覆盖历史或缺少跨会话身份一致性，难回答物体过去位置和变化过程。 LT-Mem用跨会话SLAM对齐实例，以证据评分保留身份并按易变性选择覆盖保留或多假设，三层记忆存状态与事件。



### 动作与物理交互



研究问题：接触精度、力和形变；抓取与灵巧操作；高速或受约束运动；动作表示与实际动力学不匹配。



解决路线：结合动力学、几何与接触先验的策略；最优控制和学习控制组合；力反馈与动作监督改造；物理仿真和可微模型；轨迹及抓取生成。



- [Reconstructing Is Not Acting: Action-Centric Latent Dynamics Modeling](https://arxiv.org/abs/2609.15189)：视频潜动作模型重建误差低不等于动作建模好，逆模型混入外观且前向模型可能绕过动作。 用可学习动作查询和门控聚合提取动作转变线索，再以逐层交互动作令牌持续调制前向状态表示。

- [From Prediction to Decision: World-Model-Guided Action Selection for Continuous Pile Excavation](https://arxiv.org/abs/2609.15382)：装载机连续挖掘每次改变地形，需实时预测并排序动作后果以减少完成料堆清理的铲次。 生成多候选铲挖并剔除几何不可行动作，联合预测带符号地形变化与装载量，执行最大预测载荷候选后重新观测规划。 证据边界：决策收益来自仿真对照，实体实验支持完整闭环可行性，不能混作同一量化效果。

- [An Efficient Algorithm for Minimum-Pressure Growth Planning of Vine Robots](https://arxiv.org/abs/2609.18070)：软生长机器人路径规划忽视所需压力，最短路径可能压力过高而爆裂。 推导任意形状生长压力，证明最优路径只在特定障碍点折转，将规划化为时变权重最短路并用改进Dijkstra求解。 证据边界：二维全局最优，三维为离散渐近精确的近似，不统称三维全局最优。



### 自主学习与持续改进



研究问题：交互样本昂贵且奖励稀疏；知识如何随探索更新；模型偏差如何误导策略；少量经验如何产生可靠学习信号。



解决路线：模型辅助或层次强化学习；增量知识与符号规划；观测和潜状态预测辅助学习；利用执行反馈、价值估计和失败经验改进策略。奖励模型与物理失败合成；执行动作空间的结构化探索；按价值调度有限想象；MPC提供初期引导。



- [DenseReward: Dense Reward Learning via Failure Synthesis for Robotic Manipulation](https://arxiv.org/abs/2607.13033)：机器人稠密奖励缺少多样失败和逐帧监督，人工收集昂贵且伪失败不能反映物理错误。 DenseReward自动在仿真合成碰撞漏抓掉落恢复轨迹，训练视觉语言逐帧奖励模型估计任务进展并指导MPC和强化学习。

- [StructRL: Structured Action-Space Exploration for Flow-Based VLAs](https://arxiv.org/abs/2608.15139)：流VLA去噪链内结构噪声被后续步骤稀释，难保证执行动作平滑分组探索。 StructRL以确定ODE解码，直接动作空间注入结构噪声，并末步重放梯度避免中间状态似然，连接探索与实际执行。

- [WISE: World-model-guided Imagination Scheduling for Efficient Post-training of Vision-Language-Action Models](https://arxiv.org/abs/2609.03681)：VLA世界模型后训练均匀想象浪费计算，长展开又产生不可靠策略监督。 以弱监督视觉相关性阈值选择交互状态，有限多视角想象评分筛选相对结果，仅更新真实上下文的首动作块。 证据边界：官方HTML3.3–3.4补证据已登记；想象只供候选反事实评估，首动作监督来自真实上下文，非直接学习合成状态动作对。

- [Real-World Reinforcement Learning with MPC Scaffolding for Dexterous Manipulation](https://arxiv.org/abs/2609.14878)：实体灵巧操作强化学习的早期探索低效且失败代价高，难以快速形成可靠操作策略。 用少量MPC实机轨迹预填经验并预训练策略与价值网络，在线间歇MPC指导配合SAC更新并逐步转交控制。

- [From Pretraining to Proficiency: Real-World Subtask RL for Long-Horizon Manipulation with Minimal Human Intervention](https://arxiv.org/abs/2609.21788)：长程基座策略仅少数子任务失败，整段补示范重复已有技能且全程稀疏RL难学习。 PARTS冻结基座提供标称动作，代理生成选择器验证器给瓶颈残差局部奖励，交替在线RL和成功重加权重训以集中真实练习。 证据边界：人类仍需初始化识别瓶颈并按需物理复位，不称全无人自主训练。



### 安全、鲁棒与恢复



研究问题：执行前能否识别失败；分布外观察与感知攻击；约束和碰撞风险；失败后的恢复；长期硬件退化。



解决路线：运行时监测与反事实探测；不确定性校准和弃权；约束规划与安全控制；显式恢复、重规划和分层监督；鲁棒传感与可信锚点。



- [Learning Robust Execution in Robotic Manipulation with Agentic Reinforcement Learning](https://arxiv.org/abs/2607.13818)：机器人操作误差累积易破坏执行，现有VLA缺少实时稳定性评估与恢复机制。 用两种运行质量指标评估近期历史，由代理强化学习选择执行模式，并在退化时恢复到先前名义状态继续任务。

- [MemoGuard: An Adaptive Runtime for Guarding Against Memory Traps in Communication-Limited Robot Navigation](https://arxiv.org/abs/2607.15589)：通信受限机器人复用相似记忆时，拓扑、电量和历史结果变化可能使动作失效或不安全。 重用前以拓扑、资源和结果契约验证情节记忆，仅验证失败时调用本地推理回退，以降低安全风险和计算开销。

- [CheckVLA: Execution-Time Verification with Action-Conditioned World Model for Long-Horizon Mobile Manipulation](https://arxiv.org/abs/2607.26789)：VLA开放动作块执行后的偏差无法由提交时置信度发现，残余动作会继续放大错误。 CheckVLA用冻结动作条件世界模型校验观测，保形阈值触发延迟可执行后缀重写，并用关键帧库保留已完成进度。 证据边界：仅仿真评估执行期修复。

- [MAGMA-GEN: Validated Recovery Supervision from Ambiguous Failures via Counterfactual Re-Execution](https://arxiv.org/abs/2609.20056)：长时层级操作失败难区分语义错误和低层随机执行失败，缺少可靠恢复状态监督。 特权教练提出早期局部纠正候选，仅保留同状态匹配条件反事实重执行确实改善后续进度的恢复示例。 证据边界：教练诊断允许有误，可靠性来自后续重执行筛选而非假设诊断正确。



### 人机与多机协作



研究问题：异步代理的依赖与冲突；他人意图、信任和未知目标；遥操作及辅助交互；共享知识的适用范围。



解决路线：显式同步和死锁检测；博弈与意图推断；可解释交互和多模态反馈；按范围共享记忆；分布式协调与共享场景状态。



- [Let the Body Follow: Coupled Egocentric Control for Whole-Body Robot Teleoperation](https://arxiv.org/abs/2607.16095)：全身遥操作需同时协调头臂躯干和基座，显式按钮调整增加认知负担且易碰运动学边界。 耦合自我视角控制将头俯仰偏航和手部高度映射到躯干基座，并在手到工作区边界时自动平移基座。

- [Learning Personalized Safety Interventions for Haptic Human-Robot Shared Control](https://arxiv.org/abs/2607.19534)：触觉共享控制预定义安全介入方式难匹配不同用户及应用的安全偏好。 用稀疏用户触觉示范，通过可微屏障优化层学习安全参数，使介入反馈接近示范偏好。

- [Belief-Aware Influence and Trust (BAIT): Shaping Human Belief During Repeated Human-Robot Interaction](https://arxiv.org/abs/2607.25327)：重复人机交互中人类会持续更新对机器人的信念，孤立决策或不可预测的影响策略会损害长期信任。 分层粒子滤波同时估计快速策略和慢速信念变化，结合信念感知MPPI在硬性即时绩效下权衡长期影响与信任。



### 效率与部署



研究问题：迭代策略、规划求解与感知开销；设备算力和能耗受限；模型与技能库规模增加；实时执行的系统瓶颈。



解决路线：少步采样、动态早停和压缩；轻量模型及蒸馏；并行求解和近似搜索；按需感知、缓存与异步执行；面向硬件的系统集成。



- [Just-In-Time Scene Graph Growth: Combating Perceptual Saturation in Long-Horizon Robotics](https://arxiv.org/abs/2607.13245)：先完整构建再筛选场景图带来观测冗余与语义计算饱和，难满足边缘长程机器人低延迟要求。 JITOMA以任务热图过滤前端观测并保留休眠锚，查询时才唤醒局部子图进行昂贵语义操作，实现按需图增长。

- [GLidE-SLAM: GL-Accelerated Indirect-Direct Embedded SLAM](https://arxiv.org/abs/2607.16897)：嵌入式视觉 SLAM 需高频跟踪同时为建图维护保留计算，CPU 资源不足限制部署。 中间帧用 OpenGL ES 计算着色器并行直接图像对齐估位姿，保留完整间接管线扩图及全局一致性。

- [JustDepth: Real-Time Radar-Camera Depth Estimation with Single-Scan LiDAR Supervision](https://arxiv.org/abs/2607.22172)：雷达相机深度估计的多阶段处理及辅助标注依赖增加延迟并限制部署。 将雷达聚合为定宽一维表示，以高度融合和轻量图网络预测深度，并仅在训练时使用置信解码器。



### 本体与传感设计



研究问题：现有机构、材料或传感方式限制可达动作与可靠感知；成本、尺寸、载荷和适应性的取舍。



解决路线：软体、可变形与模块化机构；传感和机械结构联合设计；主动光学及新型触觉；低成本部件复用；设计优化与物理验证。



- [Configuration-Induced Passive Self-Rotation for Perception-Enhanced Autonomous Flight](https://arxiv.org/abs/2607.17646)：机载传感视场有限，旋转扩大覆盖又需平衡刷新率与飞行性能。 调节三旋翼后臂构型控制被动自转工作点，并以分层规划控制和引导点重规划实现持续旋转下自主飞行巡检。

- [Distributed Acoustic Localization Array Deployed Using a Soft Everting Vine Robot](https://arxiv.org/abs/2607.20392)：受限灾害环境中软生长机器人需在形变体上部署声学阵列并定位远近遇险声源。 沿藤蔓机器人布置麦克风，用动态SRP-PHAT兼顾远场方向与近场三维定位，并测试安装位置和形状对精度影响。

- [Towards Ultrafast Depth Sensing Via Active Event-based Stereo Vision](https://arxiv.org/abs/2607.23684)：传统主动立体相机在快速运动场景受帧式成像限制，难以低延迟获取稠密深度。 结合双目事件相机和红外图案投影器构建原型与数据，以轻量网络、动态代价体和时间一致性估计视差。 证据边界：传感支撑技术，摘要未明确机器人实验。



### 评测与机制理解



研究问题：模型为何失败；离线指标是否对应任务能力；评测覆盖、泄漏和混杂因素；系统及安全主张的证据边界。



解决路线：受控对照、消融与失败归因；专门诊断基准；真实执行及用户研究；带条件的理论界；综述和分类框架。



- [When Does On-Policy Interaction Help? Representational Tradeoffs in Value-Based Imitation Learning](https://arxiv.org/abs/2607.29617)：模仿学习者无法表达专家策略时，交互与价值估计为何有益尚不清楚。 OVI在自身轨迹交互查询专家，只要求表达专家价值函数即可高效学习，并证明离线条件下需额外策略类复杂度。 证据边界：支撑技术，摘要未明确机器人实验。

- [REVERSAL-BENCH: A Reversibility Axis and Reset Oracle for Measuring the Reset-Free RL Cliff](https://arxiv.org/abs/2609.17745)：无重置RL依赖可逆环境，真实操控不可恢复状态可能使学习永久困住，现有评测未分离原因。 REVERSAL-BENCH连续控制可逆性并提供重置判定真值，在多引擎几何匹配任务比较策略和安全盾，揭示不可逆吸收导致的学习断崖。 证据边界：源摘要结尾缺句号但核心机制完整；恢复预测准确不意味着能从物理不可逆陷阱恢复。

- [How Long Until Your Robot Ignores You? A Safety Benchmark for LLM Orchestrators in Human-Humanoid Collaboration](https://arxiv.org/abs/2609.07288)：语言模型编排人形机器人时的拒绝、欠合规和安全违规缺少长期交互评测标准。 基准以五项安全不变量、四级合规分类及文本到实机三层架构评测长对话，比较模型与上下文预算造成的不同失败轴。 证据边界：主要报告文本层；仿真层仅初步结果，物理验证仍进行中，未将Unitree G1计划写成已验证。

- [Characterizing Replay Retention Under Dynamics Shift in Model-Based Reinforcement Learning](https://arxiv.org/abs/2609.18167)：动力学变化后旧回放可能拖慢适配或在旧状态回归时再次有用。 以变化幅度和年龄区分陈旧数据的AUC分析保留策略，跨形态算法及扰动比较永久和复现变化条件。 证据边界：回放遗忘并非普适有益，永久大变化与动力学重现需区别。

- [Outcome-Conditioned End-Effector Geometry Across Vision-Language-Action Policies](https://arxiv.org/abs/2609.21659)：不同VLA即使完成同任务也未必物理轨迹一致，成功率掩盖执行几何差异。 在大量配置匹配闭环轨迹中按成功结果分组，用多种DTW表示及窗口端点时长控制分析跨策略距离，分离共有任务几何与残余策略差异。 证据边界：配置匹配策略对相关且失败对样本薄；几何相近不等于可互换，数据重叠与任务约束未分离。



## 范围与限制



本报告把已收集条目整理成研究问题地图，以机器人学习为主，兼顾感知、规划、控制、硬件与系统。它是摘要级研究版图；少量摘要未交代机制的条目补读了官方正文方法概述。

检索主窗口为 2026-07-12 至 2026-10-12，但语料也包含会议补充的更早预印本和未来会议候选。下述统计针对当前收集语料，不能解读为这三个月整个机器人领域的完整发表统计。



论文中的性能提升、理论保证及适用范围均来自作者材料，本轮没有独立复现实验。不能仅凭本地图判断方法优劣、复现成本或 Carrot 的最优路线。

主分类是检索和组织工具；一篇论文可能横跨多个研究问题。方法标签与问题分类分开，不能把使用 RL、VLA 或世界模型本身视为研究问题。

跨分类关键词补查、会议候选筛选以及摘要缺失均可能造成覆盖偏差。IROS含二手目录候选，官方目录访问受限；CoRL官方覆盖仍未闭合。关联到arXiv不能独立确认录用。RSS、ECCV、NeurIPS官方目录各有已记录的筛选范围；不得将本次语料称作所有顶会全量。未确认的近似标题重复仍保留。

范围包含通用策略学习、几何感知和规划控制支撑技术以及虚拟具身交互。与机器人不相关的判断针对本轮收录范围，并非认定该工作绝无机器人迁移价值。



## 文件与续跑



- [中间版逐篇结果](paper-summaries.interim.jsonl)

- 原始批次、来源关联、证据与复核记录：`summaries/`。

- 最终 `paper-summaries.jsonl` 与 `problem-map.md` 尚未发布。

- 续跑：先处理 `summaries/resume-state.json` 的未完成批次、抽查项及待登记正文；沿用现有模板、范围和抽查标准，不重新采集或改写已验收批次。

