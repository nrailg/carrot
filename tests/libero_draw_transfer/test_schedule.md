# LIBERO/draw 混合 SFT 数据契约

目标：精确128/128样本配额，L100和Mix200的LIBERO曝光一致，变长chunk在DP/micro切分下loss归一化一致，动作目标不跨episode。

命令：`bash tests/libero_draw_transfer/test_schedule.sh`。

预期：4项通过；常数loss不足以验证权重，测试使用变长mask与非恒定误差独立计算global oracle。

状态：NOT RUN（远端）。本地检查记录另列，不替代真实GPU梯度与物理预检。

实际Docker image：未记录；Carrot基线：6be3d5a，加本实验未提交改动；上游版本：此CPU检查不使用上游模型。

2026-10-24 本地CPU检查：4 passed；通过uv临时依赖环境运行pytest/numpy/ray。ruff检查通过。远端与GPU状态仍NOT RUN。

2026-10-24 远端更新：H20任务9f2afe6d-0122退出0；与test_data.py合并7项PASS，耗时6.93秒。此项CPU契约状态更新为PASS，真实GPU梯度与物理预检分别留档。

## 2026-10-25 无放回遍历修正

用户要求每个来源先打乱并无放回遍历，用完该来源全体样本后再重新打乱。
固定每步128/128配额与有效动作loss归一化不变；来源各自维护遍历轮次。
本次代码用于未来新清单，不修改当前实验已冻结的源码、采样清单或checkpoint。

新增断言覆盖不同来源大小、多轮及尾部不足一轮、确定性重现、真实帧数下零重复，
并保留L100/Mix200前缀配对及DP/micro归一化检查。预期7项PASS。
运行：`bash tests/libero_draw_transfer/test_schedule.sh`；状态：NOT RUN。
Carrot基线30392d7加本次未提交修改；Docker image未记录；无上游模型依赖。

2026-10-25 远端验证：H20 session98b2ba23，在已SHA核对的work/carrot源码上运行上述runner，7 passed（0.60s）；test_data.py关联回归8 passed（9.21s），ruff检查通过。合计15项PASS，未启动训练或仿真。源码SHA ae7f7e5ff9eab7ff04abb0868cd7f3601b9a5ad090178f2ba339dbeeb533f7cc；测试SHA c68965c5b4cb40282560a1b85960ae64f5ff1d06265ea39f3953629087b43956。
