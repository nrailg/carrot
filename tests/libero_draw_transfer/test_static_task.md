# 静态整图任务契约

目的：三条线全程同时显示，策略只收到固定总任务指令；环境允许任意颜色顺序完成，teacher阶段不截断SFT动作监督。

命令：`bash tests/libero_draw_transfer/test_static_task.sh`。

断言：六种颜色顺序全部有效；每种颜色的抓取/完成归还奖励只给一次；未抓起、未画完或非任务颜色不能算完成；动作监督跨teacher阶段但不跨episode；旧动态子任务prompt数据必须拒绝。

2026-10-24 H20：PASS。任务b434d471-0153退出0；static_task与data共11项通过（7.36秒），schedule回归4项通过（0.48秒），共15项。日志位于H20个人DFS的`experiments/carrot/pi05_libero_draw_sft_transfer/sources/static-reference-20261024-01/static-task-tests.log`。

真实Isaac任务6b56755c-0428退出0，309秒；seed300000完整三色示范成功，2345controls、IoU0.7988097、抓取/归还各3次、掉笔0。每步检查prompt恒定；起始及中后段截图确认同时显示同一完整三线图。输出为PRO个人DFS下`experiments/carrot/pi05_libero_draw_sft_transfer/static-reference-preview-20261024-01`。这是teacher示范与任务契约验证，不证明策略已经学会规划。

Carrot基线975fc31；实际Docker image与上游commit未记录。
