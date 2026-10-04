# SO101 state jitter数据契约

2026-10-04：Gemini CPU契约测试 PASS（2 passed in 7.39s），Ruff/语法/同步SHA通过。训练增强仅作用于五角度输入，单位degrees、范围±3°，不改变gripper、
action、padding或底层sample；重复访问重新采样。Factory保留干净state/action统计，
确定3°端点经过Normalize后为0.06（q01=0/q99=100），用于防止在normalized空间误加3单位。

运行：`MY_DFS=<当前实测DFS> bash tests/pi_05/test_so101_state_jitter.sh`。
CPU契约测试在Gemini既有`/opt/venvs/carrot`执行，不占GPU、不下载或安装依赖。
源码commit71b1144加本轮未提交改动；Docker image tag未记录。实际75文件SHA见本轮source_hashes.json；证据位于当前MY_DFS/experiments/carrot/pi05_so101_state_jitter/20261004T033515Z/contract_test.log。
