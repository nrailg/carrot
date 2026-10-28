# 跨资源池真实闭环

状态：PASS（2026-10-23），H20 与 Pro5000 两个显式 Ray 节点，渲染只落在 Pro5000。

验证同一正式启动路径下的节点/GPU 放置、Isaac partial reset、双相机观测跨池传输、
文件分块传输 SHA、真实动作块与一次 PPO 更新，以及 Isaac → LIBERO 子进程切换。
四个 LIBERO suite 各验证 task0 的 reset/infer/step 和同 seed/初态重置一致性。
本项不衡量策略成功率，物理示范验收另见 test_physics.md。

先按 recipe start_ray.sh 启动两节点，再在 H20 设置 train.yaml 所需环境变量及
`MY_DFS`、`TEST_OUTPUT`、`SIMULATOR_TEST_OUTPUT`、`RAY_ADDRESS`，执行
`bash tests/multicolor_transfer/test_cross_pool.sh`。输出目录必须为新目录。

预期：exit0，result.json 的 status=PASS；renderer 输出含 simulator.log，
learner 输出含已校验快照及计数/概率重算误差、4 个 LIBERO case 初态 SHA。
Carrot 基线 ff9e882 加未提交改动；实际 image tag 未记录；运行 commit/SHA 和日志待补。

首轮 `8ec98b1f-0021`，exit0，168s：10 个实际控制转移、一次 PPO 更新，
old logprob 最大重算误差0；partial reset通过，7文件124899bytes跨池SHA核对通过。
四suite的task0均完成reset/infer/step与相同seed重置后图像、状态逐元素一致检查。
H20输出：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/tests/carrot/multicolor_transfer/cross-pool-20261023-01`，
含 `result.json` 与 `run.log`；Pro5000同后缀目录在zjk1，含simulator.log。
test_cross_pool.py SHA `25ed5b252ce6f9a233fd7a8f04a37426036dd8b8d3dcc90188f0312c6462dcb6`；
simulator.py SHA `44046931b4ec972ffb13b108e2f6db18a361bf166287e10cc3e6662817cd26fd`。
日志另发现 CVD=1 时 PhysX CUDA ordinal 应为0（Vulkan renderer仍为物理1）；已修正，待回归。

第二轮 `8ec98b1f-0025`：PASS，exit0；CUDA ordinal错误消失，两个相机与四suite契约再次通过。
10转移/1更新，old logprob误差4.768e-7，7文件124226bytes校验通过。
证据同测试根目录的 `cross-pool-20261023-02/result.json`、`run.log` 和Pro5000的simulator.log。
server SHA `bb17d4139a86835306c061b7075077a726fb38702134b405ca4cb8b2dc1488d3`，
simulator SHA `dfd59439cd0e11f711fac7426a4c8d43d2b889d0f9031ec4597565e6ee2a91ac`，
learner SHA `4bbcc9b59cc8eeef3a50c602c9048bf7a208a658cdb508cfa1c3eb7c67037fdf`。

前两轮为修改前的第三参考图版本。用户要求画稿仅在场景中显示后，
测试增加观测键必须与LIBERO一致的断言。
第三轮`8ec98b1f-0029`：PASS，exit0；双相机观测、10转移/1PPO更新、四suite重置/推理均通过。
old logprob误差0，peak19.97GiB，7文件111929bytes跨池SHA校验通过。
两池各自同根目录`cross-pool-20261023-03`中保留result/run.log和simulator.log。

批量配置改动待复测：真实 Pro5000 扩容到 32 个 Isaac 环境，31 个未重置 slot 保持不变；双相机跨池采样、GBS=32/micro=8 的小规模闭环和四个 LIBERO suite。正式 recipe 的 GBS=256 由 test_policy 单独验收。

第四轮 `8ec98b1f-0041`：PASS/exit0，227s。32环境/64相机均有效，partial reset保持其余31slot不变。
160控制转移、一次GBS32/micro8 PPO更新（4次累积），概率重算误差3.8147e-6。
97文件/1787399bytes跨池SHA通过，四个LIBERO suite的task0重置/推理/步进及同seed初态一致性通过。
证据：两池同测试根目录cross-pool-20261023-04，H20含result.json/run.log，Pro5000含simulator.log。
