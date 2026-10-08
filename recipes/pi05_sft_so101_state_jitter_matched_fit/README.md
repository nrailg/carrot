# SO101 state增强：对齐拟合程度再比较反馈

**结论状态：探索性观察，尚不能判断 state 数据增强对反馈鲁棒性的独立收益。** 本轮结果用于记录当前简化条件下的拟合与反馈现象，不据此决定保留或放弃增强。原始指标、预注册选择规则及核验记录保持不变。

2026-10-04用户要求增加增强轮训练步数，先达到与无增强500step模型相同拟合级别，再比较连续反馈。此前同训练预算不等于同拟合程度；增强轮真实state MAE1.656161°，对照1.346750°，所以不能只凭反馈14.705301° vs12.450586°判断增强的独立作用。

2026-10-04 15:40:51（Asia/Shanghai）已真实启动。复用本对话历史连接凭证成功重连；原MCP会话不存在不代表凭证不可用。总2000step，从同官方base新起，不resume、不覆盖原500step结果。

| 阶段 | 最终证据（2026-10-04 17:16 Asia/Shanghai） |
|---|---|
| 预检 | 78源码/配置SHA一致，生产源码同baseline；base/stats/264frame/±3°/视觉屏蔽PASS |
| 训练 | 2000step完成，无NaN/Inf；2000条连续有限指标及warmup100/constant1e-6核验通过，8rank更新有限非零 |
| checkpoint | 500/1000/1500/2000均完整：trainer.step/h10/812参数全BF16/stats/tokenizer/8optimizer shards；eval前后hash不变 |
| 评估 | 四组各424NPZ/848infer，每模式2112有效采用action行；原Parquet核验、1696NPZ独立逐元素配对与指标复算、Mac NumPy复核PASS |
| 选择与对比 | 按预注册±5%规则选择step1000：真实state MAE1.391723°（基线+3.34%），反馈14.209092°（+14.12%）；step1500近边界，step2000更好，见下表 |
| 收尾 | 总task f5302c99-0399 exit0/5055s，已dump释放；dguard watch1/guard与run.py运行/无恢复计划，Ray224CPU与8GPU全部可用 |
| 后续 | 本轮预算已完成；暂停30分钟检查。真机待休假结束；本轮改动未commit/push，图仅在Ceph/Mac cache |

| 项目 | 计划 |
|---|---|
| 初始化 | 与前两轮相同官方pi05 base h10独立副本，从step0新训，不resume |
| 条件 | 同264frame完整episode，无视觉，五轴raw U(-3°,3°)，action/gripper/stats保持；随机训练noise/t |
| 优化 | 8GPU BF16 FSDP、micro4/global64/GAS2、warmup100后constant1e-6、seed1000 |
| 预算 | 总2000step，每500保存；只是增加步数，其余训练设置不变 |
| 评估 | 复用state_only_rollout/evaluate.py，新增--step；NFE10/h10采用5帧/8noise序列，原协议不变 |
| 拟合指标 | recorded_state模式的五轴平均MAE，原对照1.3467496604°；同53chunk起点×8noise/2112有效action行 |
| 选择规则 | 500/1000/1500/2000里MAE落在对照±5%的checkpoint中，选与对照最接近者，同差时优先早步数 |
| 公平比较 | 选择只用真实state拟合指标；随后比较选定checkpoint与原baseline反馈MAE/P95/max/曲线及各轴 |
| 未匹配 | 若所有checkpoint都未落在±5%，明确报告未达匹配条件，不选反馈最好者、不宣称同拟合比较、不自动超预算 |
| 额外判断 | 同时列各轴真实state MAE/P95/max，平均值近似不代表逐轴能力相同；8noise序列不是8训练seed |
| 输出 | 当前实测$MY_DFS/experiments/carrot/pi05_so101_state_jitter_matched_fit/20261004_2000step，MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu |

运行前确认当前DFS/同步SHA/官方base权重h10/stats/264frame/视觉全false与增强、解释器offline/GPU/Ray/dguard。SOURCE_COMMIT传入Mac实际提交标签，快照及文件SHA另存；远端Git元数据不能代替执行源码证据。Docker tag与上游commit未记录。

run.sh用同一新训练产生四个checkpoint，再依次评估及原Parquet独立核验，禁止GPU并发；全程NaN/Inf或实际异常停止并归档，不自动重跑。保存全步loss/grad/LR、checkpoint完整性及评估NPZ；原500step结果只作为固定对照保留。完成后更新本表、作配对图、回填canonical memory；图片不入Git，真机仍待用户休假结束。

不能要求增强轮training loss等于无增强轮：它们的输入分布不同。以真实state上的同协议动作误差对齐，能减少基础拟合差异这个混杂因素，但单训练seed与离线理想action→state假设仍限制因果结论。之前关于纠正监督的建议只是待验证方案，此次先检验延长训练能否改变结果。


实际session f5302c99/container mpi-launcher@mpi-1784764303-launcher，Ray29.209.160.111:6379；启动命令为激活Carrot/offline/PYTHONPATH后设置RAY_ADDRESS和SOURCE_COMMIT，执行本recipe run.sh。Mac HEAD abd265ee5b49be99f5e91e3b0d7fd3c108365754加本轮未提交配置/run/evaluate --step扩展；执行源码由expected_source_hashes.json和source_snapshot精确绑定，镜像tag未记录。完整证据preflight/launch/ray_logs/training_metrics.csv/latest_training_status.json在新输出根。rank0日志为/tmp/ray/session_latest/logs/worker-93f2937340ac6f7bfebcc1d210dcd38b84b71528af267356daed356b-0b000000-363535.out；monitor.py只读该job源日志并归档，不改变训练代码/配置。

2026-10-04 16:15（Asia/Shanghai）检查：后台task0399仍运行，无driver异常；本job所有1113条已记录指标连续有限，constant LR1e-6。worker日志/CSV/status及checkpoint_progress.json持续归档。dguard实际watch0、run.py未运行、恢复计划仍在（启动stop120，约17:40恢复），暂不需延长。

2026-10-04 16:48（Asia/Shanghai）：training_completion.json与training_audit_backend.log确认2000条连续有限、四checkpoint全BF16及stats/optimizer完整、8rank更新有限非零、78执行文件SHA未变。新的step500权重SHA与前轮增强500完全相同；其424NPZ每个key数组也逐元素一致（step500_replication.json），真实state MAE1.656161°/反馈14.705301°，仍未满足±5%匹配区间。step1000评估进行中，其余未评估，不能提前选checkpoint或宣称匹配。

完整训练曲线已保存training_curves.png/svg/pdf及CSV（淡线逐step，粗线20step尾随均值），Mac缓存~/.cache/carrot/so101_state_jitter_matched_fit_20261004_2000step/已回拉并亲看。dguard仍暂停，约17:40恢复，当前剩余约52分钟，预计足够余下串行评估，继续核对。后台0399仍执行eval，未dump释放；audit task0408已exit0/28s并dump。


## 2026-10-04 最终结果与解释

增强2000step训练和四个checkpoint评估均完成，整体task退出0。以下是五个运动轴平均MAE，均为反归一化后的degrees；gripper不计入五轴平均，原始单位单独留在CSV。每次horizon10预测只采用前5个action，下一chunk的state精确等于上一chunk预测的index4；NFE10、8配对noise序列、53个chunk起点。所有训练/推理视觉屏蔽，训练noise/t随机。

| checkpoint | 真实state输入 MAE° | 连续预测反馈 MAE° | 真实state相对基线 | 反馈相对基线 |
|---|---:|---:|---:|---:|
| 无增强500（固定基线） | 1.346750 | 12.450586 | — | — |
| 增强500 | 1.656162 | 14.705298 | +22.97% | +18.11% |
| 增强1000（预注册匹配选择） | 1.391723 | 14.209092 | +3.34% | +14.12% |
| 增强1500 | 1.278878 | 11.658270 | -5.04% | -6.36% |
| 增强2000 | 1.191509 | 10.313211 | -11.53% | -17.17% |

本轮观察到：同一增强训练过程的500/1000/1500/2000四个保存点，两种输入的平均误差随训练步数增加而下降；2000step的反馈平均误差比旧无增强500小17.17%，基础拟合也更好11.53%。这是当前训练与评测条件下的结果，尚未分离训练预算、基础拟合程度和数据增强各自的作用。

严格按预注册规则，真实state匹配区间[1.2794121774,1.4140871434]°里仅step1000合格，因此选择1000，完全没有按反馈指标选。它在平均基础拟合相近时，反馈误差仍比基线高14.12%；逐轴基础拟合并不相同，见下表。这个结果只描述当前选择的checkpoint，不能推出“增强一定伤害鲁棒性”：step1500真实state1.2788776°，只比下界低约0.000535°（基线−5.04%），反馈却已改善6.36%。因此“相近拟合”的阈值非常敏感，结果不支持简单的全局正/负结论。

| 轴（数据顺序） | 基线真实state MAE° | 匹配1000真实state MAE° | 基线反馈 MAE° | 匹配1000反馈 MAE° | 更长2000反馈 MAE° |
|---|---:|---:|---:|---:|---:|
| shoulder_pan | 1.0900 | 1.2440 | 5.6243 | 7.3467 | 6.2033 |
| shoulder_lift | 2.4957 | 2.3358 | 31.8833 | 33.0465 | 21.2750 |
| elbow_flex | 1.4089 | 1.4463 | 8.9664 | 14.7043 | 8.1197 |
| wrist_flex | 1.5908 | 1.7444 | 15.2730 | 15.5050 | 15.5374 |
| wrist_roll | 0.1484 | 0.1881 | 0.5059 | 0.4430 | 0.4308 |

| 轴 | 基线反馈 P95° / max° | 匹配1000反馈 P95° / max° | 更长2000反馈 P95° / max° |
|---|---:|---:|---:|
| shoulder_pan | 25.749 / 37.776 | 22.115 / 36.802 | 23.146 / 34.812 |
| shoulder_lift | 79.219 / 93.270 | 84.128 / 91.045 | 67.460 / 90.218 |
| elbow_flex | 22.914 / 35.978 | 35.795 / 40.683 | 23.823 / 38.949 |
| wrist_flex | 57.191 / 64.129 | 62.208 / 65.194 | 55.307 / 63.270 |
| wrist_roll | 2.364 / 4.073 | 1.024 / 4.249 | 0.993 / 4.204 |

更长2000模型仍有显著反馈误差及尾部大误差，尤其shoulder_lift反馈MAE21.275°/P95 67.460°/max90.218°，不能宣称反馈问题解决。曲线中反馈误差会升降，并不是每次diffusion process单调增加。连续状态反馈、输入分布变化与模型预测可能共同影响结果；本轮未分别定位这些因素，也不能把结果当作去噪数值误差的直接测量。

### 2026-10-04 保守结论（按用户要求更新）

本轮能确认训练与评测按记录的协议完成，且当前增强模型的较晚checkpoint在这条episode上的拟合及连续反馈平均误差较低。**数据增强是否改善鲁棒性，仍未得出结论**；“增强有效”“增强无效”和“增强损害鲁棒性”均超出本轮证据。

限制在于：只使用一条episode、一个训练seed、无视觉和直接action→state的简化反馈；无增强对照仅训练500step，缺少覆盖相同训练预算的对照；真实state平均MAE相近不代表逐轴误差或局部反馈稳定性相同，±5%选择在1500step处也表现出边界敏感性。8条推理noise序列只能反映这两个固定模型对noise的变化，不能替代独立训练重复。此外，本轮没有单独检验受扰state下的恢复能力、其他episode或真实closed-loop表现。

将本实验归档为初步诊断，保留全部正负结果及限制，不把单个匹配checkpoint推广为数据增强的一般结论。更充分的增强比较留待后续明确实验范围时设计；本次记录更新不追加训练或评估。

当前只训练一个seed，8条序列是推理noise重复，不是8次独立训练。没有额外无增强长训对照，无法隔离增加训练预算与增强收益。用户认可先做模型自身鲁棒性验证；本次没有追加预算/消融、机器人动作或服务启动。后续仍为模型拟合/鲁棒性→数据输入真机执行检查→简单同步closed-loop replay→泛化，真机待休假结束。

证据与产物全部在本轮输出根：training_completion.json（训练审计当时wrapper仍运行的历史快照）、final_remote_checks.json（最终wrapper退出0/dguard恢复/Ray资源全释放）、training_evaluation_backend.log、step500_replication.json、independent_comparison.json、mac_independent_comparison.json、result_summary.json。各evaluation/stepSTEP包含原Parquet proof、checkpoint/source provenance和424NPZ。finish_comparison.py在输出根/Mac cache，CPU-only独立复核1696NPZ的原reference/padding/BF16随机noise/配对clean输入/index4反馈以及whole/chunk/endpoint/input MAE/P95/max；FP64均值与原FP32汇总仅舍入差（rtol1e-5/atol1e-6），源数据和反馈数组要求精确相等。

全四checkpoint六维MAE/P95/max保留checkpoint_metrics.csv；逐chunk逐noise指标在paired_chunk_metrics.csv。training_metrics.csv保留2000逐步原始loss/clip前grad/LR。training_curves、fit_vs_budget、paired_feedback均保存PNG/SVG/PDF；训练图粗线是尾随20step均值，反馈图阴影为8条noise的P10–P90（不是置信区间）。预算图x轴是训练step；反馈图x轴是录制frame的chunk起点0/5/.../260，15fps。图片不入Git。

Mac cache为/Users/wujunyu/.cache/carrot/so101_state_jitter_matched_fit_20261004_2000step/，无大权重/optimizer；已独立复算并展示图。最终源码仍为abd265ee5b49be99f5e91e3b0d7fd3c108365754加本轮未提交recipe/evaluate扩展，Docker tag/上游commit未记录；用户Isaac文件保持原状。

> 2026-10-08：当前共享augmentation/config已增加五轴标定bounds，raw state加噪后clip；
> action/gripper/stats不变。CPU契约测试8 passed。本文历史运行使用未clamp的旧版本，未重训。
