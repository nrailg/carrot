# SO101无视觉模型：连续action chunk反馈测试

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
