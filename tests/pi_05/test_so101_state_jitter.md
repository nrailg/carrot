# SO101 state jitter数据契约

## 2026-10-08：增强后按标定边界clamp

同日后续：用户改用扩大肩抬标定范围的方案。bounds更新为±106.24175824175825°，其余四轴不变；
原始数据保留，未采用清洗副本。新bounds版本Gemini CPU回归8 passed in7.20s，Ruff/7文件同步SHA通过；
session b1c4b5e1，当前MY_DFS重新核验不变，Ceph原Parquet SHA未变。下方7.29s是扩大范围前的测试记录。

状态：PASS。Gemini CPU测试8 passed in 7.29s，Ruff/py_compile通过，7个执行源码/配置/runner文件Mac与GPU SHA一致。
新增上/下边界定向扰动检查、非法bounds拒绝及三个recipe配置检查；
保留随机重采样、raw degrees、action/gripper/padding/底层sample及干净stats不变的断言。
五轴limits显式取当前follower标定，float32计算允许约1e-5°的表示精度；不依赖GPU读取Mac标定文件。
本修改没有重训，旧checkpoint仍对应未clamp的增强。
执行：session `b1c4b5e1`，实测唯一Ceph mount与`__SYS_USER_NAME__=nrwu`，
`MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu bash tests/pi_05/test_so101_state_jitter.sh`，exit0。
解释器`/opt/venvs/carrot/bin/python`；最终记录时Mac HEAD `fc54e52`（用户另提交Isaac文件）加本次未提交改动；
测试执行版本以7文件SHA一致为准，Docker image tag/上游commit未记录。
仅CPU契约测试，无GPU任务、Ray重启、dguard暂停、机器人操作或新训练。

2026-10-04：Gemini CPU契约测试 PASS（2 passed in 7.39s），Ruff/语法/同步SHA通过。训练增强仅作用于五角度输入，单位degrees、范围±3°，不改变gripper、
action、padding或底层sample；重复访问重新采样。Factory保留干净state/action统计，
确定3°端点经过Normalize后为0.06（q01=0/q99=100），用于防止在normalized空间误加3单位。

运行：`MY_DFS=<当前实测DFS> bash tests/pi_05/test_so101_state_jitter.sh`。
CPU契约测试在Gemini既有`/opt/venvs/carrot`执行，不占GPU、不下载或安装依赖。
源码commit71b1144加本轮未提交改动；Docker image tag未记录。实际75文件SHA见本轮source_hashes.json；证据位于当前MY_DFS/experiments/carrot/pi05_so101_state_jitter/20261004T033515Z/contract_test.log。

## 2026-10-08：移除重复 bounds，统一 LeRobot 标定来源（准备）

状态：NOT RUN；HEAD600acb0加本轮未提交修改；Docker tag/上游commit未记录。
三个recipe改为同一个state_jitter_calibration_path，增强通过LeRobot离线读取标定，
与robot执行共用action_limits。修改测试fixture为raw标定JSON，保留不污染label/gripper/stats、
逐访问采样、归一化前抖动、上下界clip及非法bounds断言。新增：文件更新立即改变工厂限位、
禁止SDK连接硬件、缺文件先于数据I/O失败、三个配置无重复角度表。
本轮同步实际LeRobot JSON到$MY_DFS/robots/so101/my_awesome_follower_arm.json，需核对Mac/GPU SHA。
命令：`bash tests/pi_05/test_so101_state_jitter.sh`；相关回归另跑
`bash tests/pi_05/test_so101_sft.sh tests/pi_05/test_pi05_inference.py tests/so101_real -q`。
CPU测试，无权重加载/GPU/训练/机器人动作；历史实验不重跑。

### 实际结果 PASS

Gemini session b1c4b5e1，当前CephFS/mm-base-plt2/用户nrwu重新核验。
Mac/GPU 18改动文件与共享标定JSON合计19文件SHA一致；共享JSON与Mac现有标定SHA均为
68ba463158a5bfde211bb2633d552ba13790c17be378d2e657338990e3bf1243。
Ruff 7相关Python文件PASS；同名shell增强契约10 passed in7.32s，其他SO101 SFT/共享inference/
全部so101_real回归86 passed in8.57s，合计96通过。
实际264frame原数据工厂通过新YAML加载、bounds由共享SDK解析且float32向内取整，未读取视频帧或执行模型。
/opt/venvs/carrot、源码PYTHONPATH、离线、CUDA_VISIBLE_DEVICES为空；没有机器人connect、GPU占用、
训练/模型评估/服务启动，dguard/Ray未操作。源码600acb0加工作区修改（之前用户stage的执行端改动保留），
Python3.12.13/pytest9.1.1，Docker tag/上游commit未记录。未commit/push，标定JSON不入Git。
证据：`$MY_DFS/test-runs/so101_calibration_source_20261008/source_hashes.json`、`preflight.json`。

### 2026-10-08：标定目录统一为 LeRobot cache 布局

按用户要求将Mac ~/.cache/huggingface/lerobot/calibration/同步到
$MY_DFS/.cache/huggingface/lerobot/calibration/，保留robots/teleoperators/backups与备份文件。
rsync排除.DS_Store、未使用--delete；7个标定/备份文件逐文件SHA一致，三份recipe均指向
新目录robots/so_follower/my_awesome_follower_arm.json。旧robots/so101副本不再被recipe引用，未删除。
首次因新目录root属主导致SSH写入失败；仅修正新建lerobot与calibration两个目录属主为
实际DevCloud用户1001:100后重传exit0，未改其他cache目录权限或绕过保护。
当前Gemini会话已核验MY_DFS；新路径真实264帧工厂PASS、既有增强契约10 passed in7.39s。
仅路径/文档修改，无新增训练/评估/硬件动作；源码600acb0加工作区修改，Docker tag未记录。
证据：$MY_DFS/test-runs/so101_calibration_source_20261008/cache_layout_upload.json。
