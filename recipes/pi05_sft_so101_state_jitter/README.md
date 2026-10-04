# SO101无视觉500step：输入state ±3°增强

用户2026-10-04明确要求启动。对照是已完成的无增强无视觉500step实验；目标检验训练
state增强能否降低连续action chunk反馈误差，同时检查干净state单次预测是否退化。

| 条件 | 本轮设置 |
|---|---|
| 初始化/数据 | 同官方h10 base，step0，不resume；同完整1episode/264帧 |
| 训练 | 500step/warmup100/constant1e-6，8GPU BF16 FSDP，micro4/global64/GAS2，seed1000 |
| 唯一训练变化 | 每次读取每个运动轴独立U(-3°,3°)，原始degrees；Normalize/TokenizePrompt之前 |
| 不扰动的字段 | gripper、示教action、padding、图像、任务；norm_stats沿用原数据 |
| 视觉/noise | 训练和推理全部视觉mask=false/pixels0；训练noise/t仍随机 |
| 入口 | 既有fit_validation/train.py、FitTrainWorker；增强仅recipe dataset wrapper，不新增train.py |
| 评估 | 复用state_only_rollout/evaluate.py与verify.py；干净输入，生产BF16/NFE10/h10，只采用前5帧 |
| 反馈 | 下一state精确取上一段采用的index4动作；起点真实state，后续不重置 |
| 对照与配对 | 每段真实state vs连续预测state，8noise序列×53段=424NPZ/848推理；与旧轮同seed/noise/reference/padding |
| 统计 | 每模式2112有效采用action行；五轴反归一化degrees的MAE/P95/max，gripper单列 |
| 输出 | $MY_DFS/experiments/carrot/pi05_so101_state_jitter/20261004T033515Z |

旧无增强对照：真实state五轴均值1.346750°、反馈12.450586°（9.244915倍）。
新结果须亲读NPZ与原Parquet、复算所有统计；不得沿用旧值冒充本轮、不得提前断言改善。
先跑数据契约测试及preflight，核验实际数据增强/统计/视觉/源码/GPU/Ray，再运行`run.sh`。
监控所有训练loss/grad/LR有限性，NaN/实际崩溃停止本任务，不自动重训或追加预算。
完成后对齐两轮逐段误差与轨迹，保存CSV/PNG/SVG/PDF；图片在Ceph/Mac缓存，不入Git。

这是理想位置跟踪下离线state反馈，无视觉模型的局部扰动鲁棒性实验；不操作机器人。
参考robomimic官方观测noise randomizer与DART噪声/分布偏移思路；DART重新采集纠正动作，
与本轮仅扰动离线state且保持target不同，不能用其收益证明本轮有效。
源码71b1144加本轮未提交改动，镜像tag未记录。75文件SHA同步、2个契约测试、Ruff/语法、
base/stats/视觉与真实增强preflight均PASS，证据在输出根。

2026-10-04 11:40 UTC+8真实启动，session cb79ec05/task cb79ec05-0367，Ray29.209.160.111:6379。
11:55亲核step1–443连续有限，LR/warmup一致；最近20步loss均值0.0113967。
Ray job07000000，rank0 worker-0034607...-07000000-344850.out，CSV/status/worker日志持久化。
dguard已暂停60分钟并设置EXIT恢复，退出状态/新指标待完成回填。
用户训练中要求先commit：代码/配置/测试已提交654a74b，无图片；初始运行快照记录71b1144加working tree，
源码/配置未改，committed_source.json绑定当前提交，后续进度记录为Markdown更新。
