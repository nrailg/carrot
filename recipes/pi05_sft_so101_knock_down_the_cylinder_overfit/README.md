# SO101 knock_down_the_cylinder：两条 episode 的拟合实验

## 2026-09-30：单位与配置清理（尚未启动新训练）

- 任务名为 knock_down_the_cylinder（推倒圆柱体），录制prompt `Move an object` 保留。
- 数据、训练统计量、策略请求/响应、日志及驱动统一使用degree/夹爪百分点。
  不做单位转换或额外范围归一化；模型原有q01/q99 Normalize/Unnormalize保留。
- `train.yaml` 显式单腕相机字段，factory相机key默认None；统计量直接读取dataset metadata。
- 新输出目录为 `pi05_sft_so101_knock_down_the_cylinder_overfit_degrees_100`，尚未运行。
  历史step100、诊断路径和指标仍保留原名，不覆盖旧模型或数据。
- `serve.sh` 加载历史degree/百分点step100，不传单位参数、不转换stats。
  `diagnose.sh`、`evaluate.py` 也直接比较degree/百分点；本轮未运行GPU推理。
- 本地数据路径 `$MY_DFS/hf-hub/nrailg/so101_knock_down_the_cylinder`；Mac缓存同名。
  之前仅重命名本地路径，6个文件SHA256不变，未重命名Hub仓库。
- 旧公共数据recipe已删除。转折预测不可靠的诊断结论仍待排查，单位清理不代表效果修复。

## 2026-09-30：训练输入与前 5 帧动作诊断（完成）

- 目的：确认同一录制观测能否复现示教动作，区分预测误差与部署输入处理问题。
- 模型：已有 step100 policy server，`ws://127.0.0.1:8080`；不启动训练、不操作机器人。
- 数据：原始推倒圆柱两条 episode、354 帧、单腕 RGB、15 FPS、六维绝对目标。
- 审计：checkpoint/dataset 全部统计量、关节顺序、帧时间戳、50步动作对齐及
  episode padding；逐帧比较训练与客户端入口处理后的 state、图像、mask、tokens。
- 推理：354个训练观测每个请求一次，服务使用随机噪声；统计首1/5/50帧的六轴
  MAE、P95和bias，并重算原有32个固定噪声观测的相同窗口指标。
- 入口：`diagnose.py`、`diagnose.sh`；独立输出，不覆盖此前评估。
- 命令：确认 `MY_DFS` 后设置
  `SO101_DIAG_OUTPUT="$MY_DFS/experiments/carrot/pi05_sft_so101_wipe_overfit_100/diagnostics/training_inputs_20260930T111750Z"`，
  运行 `bash recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/diagnose.sh`。
- 验收：退出0，354个唯一观测与预测、两条episode、有效padding统计正确；input_audit.json、
  metrics.json、all_predictions.npz与逐帧NPZ齐全。这里只评估训练观测，不验收闭环任务成功率。
- 本地源码 d16d29a 加未提交诊断脚本；实际镜像tag尚未记录。运行结果在本节续写。

### 实际结果

- 成功任务 `56253f33-0034`，退出0、耗时166秒；补充重复抽样任务 `56253f33-0037`，
  退出0、耗时56秒。共354个唯一训练观测（198/156）、首5步1750个有效目标、
  全50步15250个有效目标；另选两条episode首帧和四个分散异常帧，各重复8次，共48次推理。
- 输出在以上目录加 `_retry1`。首轮 `56253f33-0032` 因诊断脚本误用
  `pyarrow.parquet.concat_tables` 在推理前退出，已修正为 `pyarrow.concat_tables`；
  重复抽样初轮 `56253f33-0036` 因直接序列化Tensor在请求发送前失败，改为显式NumPy请求。
  两轮失败日志保留，不计入成功样本。
- checkpoint/dataset 20组统计量逐项最大差为0；全部354帧训练与客户端处理后的
  图像、state、mask、tokens完全一致；动作时间窗口/padding/关节顺序通过，
  timestamp最大偏差4.45e-7秒。OpenPI loader明确pi05=True、离散state输入开启；
  导出config只存架构字段，不直接读取不存在的discrete_state_input配置项。

| 关节 | 首1步MAE | 首5步MAE | 首5步P95绝对误差 |
|---|---:|---:|---:|
| shoulder_pan | 1.926° | 2.036° | 9.408° |
| shoulder_lift | 3.761° | 3.970° | 11.942° |
| elbow_flex | 2.795° | 2.990° | 9.133° |
| wrist_flex | 1.607° | 1.717° | 6.580° |
| wrist_roll | 0.445° | 0.465° | 2.134° |
| gripper | 0.00994百分点 | 0.01000百分点 | 0.06390百分点 |

- 首5步最大单点肩/肘误差43.310°/28.001°；超过10°的比例分别8.743%/4.114%。
  总体轨迹接近示教，但转折处预测仍明显滞后且存在反向首目标。
- 各重复8次：episode0 frame46肩/肘首5步MAE均值17.308°/10.731°；
  episode1 frame32为15.674°/15.119°；两条episode首帧肩约2.0°/2.3°。
  episode1 frame103示教要求肩-59.824→-69.099°、肘46.549→52.264°，
  重复首目标肩7/8、肘8/8方向相反。随机性影响误差，但异常并非仅一个随机样本。
- 结论范围：同训练输入的统计/预处理错位未发现；模型在训练数据转折处仍不能可靠复现，
  100step尚不能验收充分拟合。不是闭环评测，也不能由此确定唯一根因。
  建议固定失败帧/噪声作为回归集，诊断拟合与条件歧义后再决定训练预算；节拍问题另行修复。
- 独立读回核对354份逐帧NPZ、汇总shape、唯一键、有效目标数，重算MAE/P95一致；
  11份本地/远端相关源码SHA256一致。Ruff、bash语法、py_compile、diff检查通过，
  另有窗口/padding及Parquet API轻量检查通过；没有重跑无关测试。
- provenance.json记录Python3.12.13、Torch2.11.0+cu128、NumPy2.3.1、LeRobot0.6.1、
  PyArrow23.0.1、Transformers5.5.4、OpenPI client0.1.0；实际Docker tag未记录。
- 本地结果/曲线副本：`/Users/wujunyu/.cache/carrot/so101_model_diagnosis_20260930T111750Z/`。
  包含metrics/input_audit、all_predictions/repeat_probes及肩肘/六轴/异常块曲线。
- 本轮无串口/相机访问、无运动、无追加训练、无commit。GPU0 policy服务健康，
  GPU1–7原有matmul保持；dguard短暂停10分钟后已自动恢复，收尾DGUARD_WATCH=1。

## 目的与配置

在用户录制的两条 episode 上做小数据拟合。100 step 是本轮预算，不预先宣称足够过拟合。

- 当前本地数据 `${MY_DFS}/hf-hub/nrailg/so101_knock_down_the_cylinder`，2 episodes，354 帧（198/156），15 FPS。
- 唯一相机 `observation.images.wrist`，映射模型左腕槽；缺失 base/right_wrist 的 mask=false。
- robot_type=so_follower；沿用数据任务文本 `Move an object`，不改标注或单位。
- 用户确认录制日志 robot.use_degrees=True、teleop.use_degrees=True：前五轴度数，第六轴夹爪开合百分比。样本、统计量和部署均保留这些单位。下方历史100step训练仍使用原单位。
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
bash "$MY_DFS/work/carrot/recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/run.sh"
```

当前新输出目录 `${MY_DFS}/experiments/carrot/pi05_sft_so101_knock_down_the_cylinder_overfit_degrees_100`；不覆盖已有目录。
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

- 启动入口 `serve.sh`；`MY_DFS=<个人DFS> CUDA_VISIBLE_DEVICES=0 bash recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/serve.sh`。
- 2026-09-29约20:45北京时间最初启动8000端口task `1d7252fc-0307`；用户确认Mac仅可访问8080/8081后，该任务已停止。
- 20:49切换到8080，新后台task `1d7252fc-0313`，GPU0、host=0.0.0.0；serve.sh默认端口已改8080。
- launcher地址 `29.209.160.111`；WebSocket `ws://29.209.160.111:8080`，健康检查 `/healthz`。
- dguard暂停120分钟，计划22:45:48北京时间自动恢复；服务持续运行，恢复后可能竞争GPU资源。
- 模型step100；请求字段 observation/state（6维，前5轴度数、夹爪百分比）、observation/wrist_image（RGB uint8）、prompt（训练文本 Move an object）。省略不存在的外部相机，不伪造双相机。
- 输出绝对目标float32[50,6]，对应15FPS训练时间尺度。
- 当前部署使用单腕15FPS、use_degrees=true配置；历史诊断过程保留原结果。
- 主代理从devcloud访问该IP被IDC网关拒绝：HTTP403、acl.14001/acl_denied，说明需要igate权限；此结果不能判定MacBook的办公网路由是否可达。未调整网络权限。

- 20:51北京时间新服务加载完成，远端localhost和launcher IP的8080 `/healthz` 均返回200 OK；8000无监听。
- 20:51:50北京时间WebSocket真实单腕数据推理通过：metadata horizon50/dim6/num_steps10/so101，返回float32[50,6]全部有限，首次客户端往返534ms。证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_sft_so101_wipe_overfit_100/serving/20260929T1249Z/health_websocket_infer.log`。主代理已独立读取日志；服务保持运行。


## 2026-09-30：review清理验收

94项CPU/真实数据契约回归通过，Ruff/语法/diff检查通过，31个同步文件hash一致。部署字段dataset_revision移除；相机配置通过config对象传递；camera key默认None并在recipe/test YAML显式指定。单位转换包含状态、动作、统计量、起点比较和发送日志，旧checkpoint必须显式声明统计量单位，新checkpoint导出自带单位。数据目录两端真实重命名，文件内容未改。未commit、未新增训练/模型评估/真机动作，旧服务未重启。测试日志在个人DFS `test-runs/so101_review_cleanup_20260930/final_tests.log`。
