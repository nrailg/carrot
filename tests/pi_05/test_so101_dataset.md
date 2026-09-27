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
