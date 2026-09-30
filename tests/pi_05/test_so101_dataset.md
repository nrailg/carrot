# SO101 真数据读取与 episode 边界

读取固定 revision c021b3c22a3de4e70e81010e54fb250a5dde348b 的完整本地数据，
验证双相机视频解码、六维关节、非空任务文本和末尾 padding。

## 运行

```bash
export MY_DFS=/absolute/path/to/personal/dfs
bash "${MY_DFS}/work/carrot/tests/pi_05/test_so101_dataset.sh"
```

只需提供 `MY_DFS`；脚本从 `${MY_DFS}/hf-hub/felixmayor/orange_cube_merged`
读取数据，通过 `${MY_DFS}/work/carrot` 解析源码并激活 `/opt/venvs/carrot`。
测试不下载数据、不加载模型、不连接机械臂。

## 结果

2026-09-26：**PASS**。与客户端测试合跑，30 passed in 6.42s，其中本文件 1 项。
命令、版本和源码证据见 `tests/so101_real/test_so101_runtime.md` 同日记录。

- 实际输入：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/felixmayor/orange_cube_merged`，
  选中 episode 0，action horizon 50。
- top 解码为 `(540,960,3)`、fpv 为 `(640,480,3)`，均 RGB uint8；六维 state 和任务字段有效。
- episode 最后两帧 valid_steps 分别为 2、1，下一次读取返回 None，未跨 episode。
- 环境为 devcloud CPU 临时 venv，LeRobot 0.6.1、PyTorch 2.11.0+cpu、NumPy 2.2.6；
  Docker image tag 不适用。未加载模型或连接机器人。


## 2026-09-27：Gemini 验证

状态：**PASS**；`1 passed in 6.79s`，runner exit=0。

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
cd "${MY_DFS}/work/carrot"
bash tests/pi_05/test_so101_dataset.sh
```

证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101-20260927T/test_so101_dataset.log`。

本轮镜像、源码和环境记录见 [共同环境](../so101_real/test_so101_runtime.md#本轮共同环境与源码)。


## 2026-09-30：真实数据单位/输入契约（准备）

- 运行同名shell，读取公开orange-cube和重命名后的knock_down_the_cylinder本地数据；不访问Hub。
- 新用例读取recipe YAML单腕字段与degree开关，比较真实首帧/episode0末帧训练与client状态、图像、padding和reference，以及所有统计量比例；无GPU推理/硬件。
- 预期：两用例PASS。状态NOT RUN；源码d16d29a加本轮改动，Docker tag未记录。


### 2026-09-30 20:38北京时间：实际结果 PASS

- Gemini task56253f33-0052 exit0：`tests/pi_05/test_so101_sft.py tests/so101_real tests/pi_05/test_pi05_inference.py tests/sft/test_sft_checkpoint.py` 合计92 passed；真实数据同名shell另2 passed，共94项。CPU-only，CUDA_VISIBLE_DEVICES为空；未接触机器人或新模型推理。
- Ruff通过；31个改动源码/配置/shell的Mac与hz1 SHA256一致，shell语法、py_compile和git diff --check通过。实际源码d16d29a加未提交改动，Docker tag未记录。
- 初轮夹爪缩放后固定归一化epsilon造成约1e-5偏差，测试原1e-6零点容差过严；改成有解释的2e-5。随后wrapper子shell因MY_DFS未export失败，已修正执行环境；风格问题已修正。最终测试正常完成，未跳过失败项。
- 完整最终日志及初轮失败日志在 `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101_review_cleanup_20260930/`，含final_tests.log和source_hashes.json。
- 新YAML配置/新单位接口准备完成；未启动新训练或真机任务，未commit。旧服务没有单位metadata，后续联调须按当前代码重启。
