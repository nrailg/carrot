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
