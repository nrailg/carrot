# SO101 SFT checkpoint 独立推理验收

## 目的与契约

重载 step-00005000 的完整模型、tokenizer 和 SO101 quantile stats，以
orange_cube_merged 原始观测执行两次固定噪声推理，要求 float32[50,6]、有限且可重复。
真实示教动作仅是数据加载窗口，不送入模型推理请求。

## 运行

确认当前 Gemini MY_DFS、同步源码、检查空闲 GPU 与 dguard 后运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash tests/pi_05/test_so101_inference_checkpoint.sh
```

可通过 CARROT_SO101_CHECKPOINT/CARROT_SO101_DATASET 覆盖下载好的本地路径。
运行环境为 /opt/venvs/carrot，runner 使用源码 PYTHONPATH，不下载资源或安装项目。

## 2026-09-26

状态：BLOCKED，未运行。Gemini MCP 连接返回缺少 token 且未配置自动认证。
目标 checkpoint precision=float32、horizon=50；这不构成实际模型重载证据。
Docker image tag / 实际运行的源码和依赖版本：未记录，等待 GPU 运行时填写。
CPU 客户端与协议验证见 tests/so101_real/test_so101_runtime.md。


## 2026-09-27：Gemini 验证

状态：**PASS**；`1 passed in 71.43s`，runner exit=0。

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
cd "${MY_DFS}/work/carrot"
CUDA_VISIBLE_DEVICES=0 bash tests/pi_05/test_so101_inference_checkpoint.sh
```

证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101-20260927T/test_so101_inference_checkpoint.log`。
使用 H20 GPU 0；dguard 暂停 10 分钟后执行，安排自动恢复。
真实 step-00005000 checkpoint 重载及固定噪声双次推理通过；不代表真机任务成功率。

本轮镜像、源码和环境记录见 [共同环境](../so101_real/test_so101_runtime.md#本轮共同环境与源码)。


## 2026-09-30：旧checkpoint单位声明

公开orange-cube重载入口显式声明legacy统计量 `joint_units="normalized"`；原始关节仍归一化位置，夹爪统计量及client输入转换为[0,1]。原真GPU重载测试本轮NOT RUN；显式degree迁移/新export不重复转换已在CPU mock覆盖。源码d16d29a加未提交改动，Docker tag未记录。


## 2026-09-30：统一degree/夹爪百分点

当前契约取代此前单位方案：所有SO101样本、stats、网络接口、日志和驱动使用degree与[0,100]夹爪。
删除单位换算及额外裁剪，保留模型已有q01/q99 Normalize/Unnormalize。
CPU验证范围：样本/stats原值、训练/推理一致性、导出/加载、握手、mock驱动与报告。
真实数据入口：`bash tests/pi_05/test_so101_dataset.sh`，逐帧核对单腕录制与client数据。
状态：NOT RUN（本轮未执行GPU checkpoint/WebSocket测试）；相关CPU契约回归已通过。
未操作真机、启动训练或重启policy server。
源码：c4593f2加工作区改动；Docker image tag未记录，Python 3.12.13，LeRobot 0.6.1。

证据：`$MY_DFS/test-runs/so101_degrees_20260930/final_tests.log`、`source_hashes.json`。
