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

状态：100-step训练及固定训练样本推理评估已有完整日志和产物；详见下方结果。

### 启动记录

- 2026-09-29 11:30:38 UTC（北京时间19:30:38）通过现有空闲Ray启动；后台task `2f6f5af6-0264`。
- 预留40 CPU/8 H20；dguard暂停30分钟，未重启Ray或停止其他实验。
- 源码基于 `d0af86db8e93c400d1170b5a30828eaa6971d246` 加本次单相机/robot_type兼容与recipe改动，启动前同步文件SHA256一致。
- 前置SO101 CPU测试5 passed；真实factory首尾样本和视角mask验证通过。
- 持久监控目录 `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_sft_so101_wipe_overfit_monitor/20260929T1129Z`，CPU日志 `cpu_tests.log`。
- 此记录仅证明任务启动，训练完成和拟合效果待下节实际结果。


### 2026-09-29 20:31 北京时间结果核查

- 持久训练日志 `train_backend.log` 有 `SFT finished: step=100`；rank0记录step1..100共100条。
- 首步loss=0.137848，末步=0.009677887；前10步均值0.0777427，后10步均值0.0115624。
- step25/50/75/100四个checkpoint均有权重、config、norm_stats、tokenizer_config和8份optimizer shard；step100已由独立评估成功加载。
- 训练前后使用同一32个训练观测、固定噪声，排除padding后1345行有效动作。

| 指标 | base | step100 |
|---|---:|---:|
| 动作范围归一化RMSE | 0.4661695 | 0.1795024 |
| shoulder_pan MAE（度） | 7.8626 | 2.2565 |
| shoulder_lift MAE（度） | 30.6453 | 8.4070 |
| elbow_flex MAE（度） | 22.9190 | 5.8599 |
| wrist_flex MAE（度） | 9.8191 | 2.6059 |
| wrist_roll MAE（度） | 2.2817 | 0.7456 |
| gripper MAE（数据中的百分比单位） | 0.10930 | 0.01785 |

- RMSE下降约61.5%；保持当前state基线RMSE=0.6059362。拟合明显改善，但未达到可宣称充分过拟合的证据；未延长超过用户指定的100step。
- 这是训练样本子集评估，不是留出集或真机任务成功率。
- 评估证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_sft_so101_wipe_overfit_100/evaluation/{base,step100}/metrics.json`；对应NPZ在相同目录。
- 完整训练、评估stdout见上述monitor目录的 `train_backend.log`、`rank0_worker_stdout.log`、`eval_base.log`、`eval_step100.log`。

### 2026-09-29 MacBook 联调服务

- 启动入口 `serve.sh`；`MY_DFS=<个人DFS> CUDA_VISIBLE_DEVICES=0 bash recipes/pi05_sft_so101_wipe_overfit/serve.sh`。
- 2026-09-29约20:45北京时间最初启动8000端口task `1d7252fc-0307`；用户确认Mac仅可访问8080/8081后，该任务已停止。
- 20:49切换到8080，新后台task `1d7252fc-0313`，GPU0、host=0.0.0.0；serve.sh默认端口已改8080。
- launcher地址 `29.209.160.111`；WebSocket `ws://29.209.160.111:8080`，健康检查 `/healthz`。
- dguard暂停120分钟，计划22:45:48北京时间自动恢复；服务持续运行，恢复后可能竞争GPU资源。
- 模型step100；请求字段 observation/state（6维，前5轴度数、夹爪百分比）、observation/wrist_image（RGB uint8）、prompt（训练文本 Move an object）。省略不存在的外部相机，不伪造双相机。
- 输出绝对目标float32[50,6]，对应15FPS训练时间尺度。
- 旧examples/so101_real配置仍强制30FPS/use_degrees=false/双相机，不能直接用于本模型控制；PolicyClient网络接口可复用。
- 主代理从devcloud访问该IP被IDC网关拒绝：HTTP403、acl.14001/acl_denied，说明需要igate权限；此结果不能判定MacBook的办公网路由是否可达。未调整网络权限。

- 20:51北京时间新服务加载完成，远端localhost和launcher IP的8080 `/healthz` 均返回200 OK；8000无监听。
- 20:51:50北京时间WebSocket真实单腕数据推理通过：metadata horizon50/dim6/num_steps10/so101，返回float32[50,6]全部有限，首次客户端往返534ms。证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_sft_so101_wipe_overfit_100/serving/20260929T1249Z/health_websocket_infer.log`。主代理已独立读取日志；服务保持运行。
