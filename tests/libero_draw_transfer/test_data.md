# LIBERO/draw adapter 与生产 loss 契约

目标：draw 的 horizon 不跨 phase；采样清单与实际 mask 不一致时报错；直接调用生产 Pi05SFTLossFn，检查两个域的有效步数归一化及参数梯度。

命令：`bash tests/libero_draw_transfer/test_data.sh`。

2026-10-24 H20：PASS。与 test_schedule.py 合并执行共7项通过，耗时6.93秒；MCP任务9f2afe6d-0122退出0。fixture只验证数据与loss契约，不代表真实物理任务或大模型训练成功。

实际Docker image及上游commit：未记录。Carrot基线6be3d5a，加本实验未提交改动；测试源码已按SHA核验。冻结源码位于H20个人DFS的 `experiments/carrot/pi05_libero_draw_sft_transfer/sources/sft-mix-20261024-01`，其中source-manifest.json记录逐文件SHA。
