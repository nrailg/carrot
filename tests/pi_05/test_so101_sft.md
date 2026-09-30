# SO101 SFT 数据与推理契约

## 目的与关键断言

使用模拟数据、tokenizer 和小模型验证四项契约，不加载真实 checkpoint 或连接机械臂。

- 数据工厂保留 revision、30 FPS 动作窗口、双相机字段及 padding mask。
- 训练和推理共用图像、状态、任务 token 变换；第三视角无效，padding 不参与 loss。
- 六维绝对位置动作正确归一化和还原，不误作相对状态的增量。
- 配置指定数据集统计量时不误读基座的 14 维统计量；导出布局可装配六关节 policy。

## 运行

只需提供个人 DFS 根目录；公共 runner 激活 `/opt/venvs/carrot`，
从 `${MY_DFS}/work/carrot` 加载源码并设置 `PYTHONPATH`。测试无需 GPU 或下载资源。

```bash
export MY_DFS=/absolute/path/to/personal/dfs
bash "${MY_DFS}/work/carrot/tests/pi_05/test_so101_sft.sh"
```

## 2026-09-27

补齐同名 runner 和档案；Shell 语法及 diff 检查通过。


## 2026-09-27：Gemini 验证

状态：**PASS**；`4 passed in 6.87s`，runner exit=0。

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
cd "${MY_DFS}/work/carrot"
bash tests/pi_05/test_so101_sft.sh
```

证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101-20260927T/test_so101_sft.log`。

本轮镜像、源码和环境记录见 [共同环境](../so101_real/test_so101_runtime.md#本轮共同环境与源码)。

## 2026-09-29：15 FPS 单腕视角

新增单腕相机数据配置与缺失视角mask契约回归，同时接纳LeRobot的so_follower类型名。
Gemini既定venv运行同名runner：**5 passed**（含此前双相机4项）。
真实两episode数据首尾样本50×6窗口、末尾49帧padding与base/right mask=false也已预检。
证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/experiments/carrot/pi05_sft_so101_wipe_overfit_monitor/20260929T1129Z/cpu_tests.log`。
源码与环境及实验最终结论见 `recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/README.md`。


## 2026-09-30：review清理与模型单位边界回归（准备）

- 范围：显式YAML相机字段、config传递、degree→radian及夹爪百分点→[0,1]，样本/stats同时换算；反向驱动换算、起点容差、导出与握手单位契约。
- 命令：既定Gemini venv中从当前同步源码运行 `python -m pytest -q tests/pi_05/test_so101_sft.py tests/so101_real`。均使用CPU及fake设备，不接触真实串口或相机，不启动训练/模型评估。
- 预期：新单位回归与已有客户端/日志/调度测试通过，旧服务缺单位声明时在硬件连接前拒绝。
- 状态：NOT RUN；源码d16d29a加本轮未提交改动，Docker image tag未记录；实际结果运行后补充。


### 2026-09-30 20:38北京时间：实际结果 PASS

- Gemini task56253f33-0052 exit0：`tests/pi_05/test_so101_sft.py tests/so101_real tests/pi_05/test_pi05_inference.py tests/sft/test_sft_checkpoint.py` 合计92 passed；真实数据同名shell另2 passed，共94项。CPU-only，CUDA_VISIBLE_DEVICES为空；未接触机器人或新模型推理。
- Ruff通过；31个改动源码/配置/shell的Mac与hz1 SHA256一致，shell语法、py_compile和git diff --check通过。实际源码d16d29a加未提交改动，Docker tag未记录。
- 初轮夹爪缩放后固定归一化epsilon造成约1e-5偏差，测试原1e-6零点容差过严；改成有解释的2e-5。随后wrapper子shell因MY_DFS未export失败，已修正执行环境；风格问题已修正。最终测试正常完成，未跳过失败项。
- 完整最终日志及初轮失败日志在 `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101_review_cleanup_20260930/`，含final_tests.log和source_hashes.json。
- 新YAML配置/新单位接口准备完成；未启动新训练或真机任务，未commit。旧服务没有单位metadata，后续联调须按当前代码重启。


## 2026-09-30：录制单位命名与显式repo配置

状态：**NOT RUN**；本轮没有可用Gemini会话，未执行pytest、训练或推理。

- 数据集配置改用 `recorded_in_degrees`；已有degree样本、统计量与推理一致性用例沿用该字段。
- repo默认留空；新增断言要求缺少repo时在LeRobot元数据I/O之前失败。
- 真实数据、checkpoint与WebSocket测试显式填写其测试对象的repo，不再依赖通用默认值。
- 静态检查PASS：Python AST语法与100字符行长、recipe/test YAML字段与factory签名匹配、runner `bash -n`、`git diff --check`。

复现：`bash tests/pi_05/test_so101_sft.sh`（按runner要求先设置当前Gemini的 `MY_DFS`）。
相关三件套：`test_so101_sft.py`、`test_so101_sft.sh`、本文件。
源码：`d16d29a7245c56f17d9eec07e42234c11b831200` 加当前未提交改动；Docker image tag及远端依赖commit未记录，本轮未连接远端。


## 2026-09-30：统一degree/夹爪百分点

当前契约取代此前单位方案：所有SO101样本、stats、网络接口、日志和驱动使用degree与[0,100]夹爪。
删除单位换算及额外裁剪，保留模型已有q01/q99 Normalize/Unnormalize。
CPU验证范围：样本/stats原值、训练/推理一致性、导出/加载、握手、mock驱动与报告。
真实数据入口：`bash tests/pi_05/test_so101_dataset.sh`，逐帧核对单腕录制与client数据。
状态：PASS，包含本文件的CPU回归共89项通过，耗时9.88s；Ruff通过。
未操作真机、启动训练或重启policy server。
源码：c4593f2加工作区改动；Docker image tag未记录，Python 3.12.13，LeRobot 0.6.1。

证据：`$MY_DFS/test-runs/so101_degrees_20260930/final_tests.log`、`source_hashes.json`。

实际CPU命令（/opt/venvs/carrot，当前源码PYTHONPATH，CUDA_VISIBLE_DEVICES为空）：

```bash
python -m pytest -q tests/pi_05/test_so101_sft.py tests/so101_real tests/pi_05/test_pi05_inference.py tests/sft/test_sft_checkpoint.py
bash tests/pi_05/test_so101_dataset.sh
```

首轮87通过、2失败：握手报错匹配和反馈夹爪期望仍使用旧值；修正后89通过。
首轮证据保存在同目录 `initial_tests.log`，最终90项通过包含1项真实数据检查。


## 2026-09-30：删除数据语义字段

范围：删除dataset spec、训练导出、policy与握手、报告中的额外声明和校验。
样本与stats保留源数值，已有Normalize/Unnormalize与标定限幅保留。
预期：无额外metadata的统计文件可以加载、日志可以绘图；握手仍核对动作维度等推理结构。
命令：合跑SO101 SFT、全部so101_real、共享PI05 inference与SFT checkpoint回归，
另执行 `bash tests/pi_05/test_so101_dataset.sh` 验证真实录制数据。
状态：PASS，含本文件的CPU回归共88项通过（10.44s），Ruff通过；未进行GPU推理、训练或真机操作。
源码：bf11592加工作区改动；Docker tag未记录。

环境：Python 3.12.13、LeRobot 0.6.1；证据：
`$MY_DFS/test-runs/so101_source_values_20260930/final_tests.log` 和 `source_hashes.json`。
