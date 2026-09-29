# SO101 wipe：两条 episode 的 100-step 拟合实验

## 目的与配置

在用户录制的两条 episode 上做小数据拟合。100 step 是本轮预算，不预先宣称足够过拟合。

- 本地数据 `${MY_DFS}/hf-hub/nrailg/so101_wipe_down_the_cylinder`，2 episodes，354 帧（198/156），15 FPS。
- 唯一相机 `observation.images.wrist`，映射模型左腕槽；缺失 base/right_wrist 的 mask=false。
- robot_type=so_follower；沿用数据任务文本 `Move an object`，不改标注或单位。
- 用户确认录制日志 robot.use_degrees=True、teleop.use_degrees=True：前五轴度数，第六轴夹爪开合百分比。训练保留原单位并用本数据统计量；当前部署客户端固定false，不能直接下发此模型。
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
bash "$MY_DFS/work/carrot/recipes/pi05_sft_so101_wipe_overfit/run.sh"
```

新输出目录 `${MY_DFS}/experiments/carrot/pi05_sft_so101_wipe_overfit_100`；不覆盖已有目录。
预期 checkpoints/step-00000025、50、75、100，保存权重、统计量、tokenizer与训练状态。

## 验收

- 退出0、有SFT finished/step100、loss及梯度有限、四个checkpoint文件完整。
- 训练前后对每episode均匀抽取16帧（共32个固定训练观测）、相同噪声做50步动作预测，排除episode padding，记录六关节MAE、归一化RMSE及state保持基线。
- 样本拟合、训练完成、模型可重载分别记录；无留出集，不据此宣称泛化或真机任务成功。

## 2026-09-29

状态：准备中。数据已只读预检，单相机兼容改动等待回归，尚未开始训练。
实际版本、任务ID、日志与结果执行后补充。

### 启动记录

- 2026-09-29 11:30:38 UTC（北京时间19:30:38）通过现有空闲Ray启动；后台task `2f6f5af6-0264`。
- 预留40 CPU/8 H20；dguard暂停30分钟，未重启Ray或停止其他实验。
- 源码基于 `d0af86db8e93c400d1170b5a30828eaa6971d246` 加本次单相机/robot_type兼容与recipe改动，启动前同步文件SHA256一致。
- 前置SO101 CPU测试5 passed；真实factory首尾样本和视角mask验证通过。
- 持久监控目录 `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_sft_so101_wipe_overfit_monitor/20260929T1129Z`，CPU日志 `cpu_tests.log`。
- 此记录仅证明任务启动，训练完成和拟合效果待下节实际结果。
