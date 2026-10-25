# LIBERO/draw adapter 与生产 loss 契约

当前目标：静态整图任务的horizon允许跨teacher内部phase，只在episode末尾截断；动态子任务prompt数据必须拒绝；采样清单与实际mask不一致时报错；生产Pi05SFTLossFn的有效步数归一化及梯度正确。

命令：`bash tests/libero_draw_transfer/test_data.sh`。

2026-10-24 H20：PASS。与 test_schedule.py 合并执行共7项通过，耗时6.93秒；MCP任务9f2afe6d-0122退出0。fixture只验证数据与loss契约，不代表真实物理任务或大模型训练成功。

实际Docker image及上游commit：未记录。Carrot基线6be3d5a，加本实验未提交改动；测试源码已按SHA核验。冻结源码位于H20个人DFS的 `experiments/carrot/pi05_libero_draw_sft_transfer/sources/sft-mix-20261024-01`，其中source-manifest.json记录逐文件SHA。

2026-10-24任务更正：上方PASS对应旧逐阶段prompt契约；新静态整图契约已在H20通过，执行记录见test_static_task.md。

2026-10-25真实混合检查发现LIBERO的Tensor mask与draw的NumPy mask在默认collate中冲突，正式训练尚未开始。生产transform adapter统一为bool Tensor，新增两种样本顺序×单/多进程加载四项回归。H20任务b434d471-0162退出0，data/schedule/static_task合计19项PASS，7.49秒；113文件SHA冻结于sources/train-20261025-01。复用已采集并去重的64条数据，不重采。
