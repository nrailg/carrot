# LIBERO/draw 混合 SFT 数据契约

目标：精确128/128样本配额，L100和Mix200的LIBERO曝光一致，变长chunk在DP/micro切分下loss归一化一致，动作目标不跨episode。

命令：`bash tests/libero_draw_transfer/test_schedule.sh`。

预期：4项通过；常数loss不足以验证权重，测试使用变长mask与非恒定误差独立计算global oracle。

状态：NOT RUN（远端）。本地检查记录另列，不替代真实GPU梯度与物理预检。

实际Docker image：未记录；Carrot基线：6be3d5a，加本实验未提交改动；上游版本：此CPU检查不使用上游模型。

2026-10-24 本地CPU检查：4 passed；通过uv临时依赖环境运行pytest/numpy/ray。ruff检查通过。远端与GPU状态仍NOT RUN。

2026-10-24 远端更新：H20任务9f2afe6d-0122退出0；与test_data.py合并7项PASS，耗时6.93秒。此项CPU契约状态更新为PASS，真实GPU梯度与物理预检分别留档。
