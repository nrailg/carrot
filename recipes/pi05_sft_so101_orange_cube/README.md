# PI0.5 + SO101 orange cube SFT

## 目的与配置

从 `pi05_base_pytorch` 初始化 PI0.5，使用
[`felixmayor/orange_cube_merged`](https://huggingface.co/datasets/felixmayor/orange_cube_merged)
示教数据训练 SO101。参数以
[`pi05_sft_so101_orange_cube.yaml`](pi05_sft_so101_orange_cube.yaml) 为准：
8 GPU 单节点、计划 5000 step、global batch 256、micro batch 4、梯度累积 8、
action horizon 50、学习率 `1e-5`，每 500 step 保存。W&B 未启用。

数据集 revision 固定为 `c021b3c22a3de4e70e81010e54fb250a5dde348b`。
配置要求本地副本位于 `${MY_DFS}/hf-hub/felixmayor/orange_cube_merged`；
SO101 数据适配器从该副本的元数据读取 state/action 统计量。
运行前确认本地副本对应上述 revision；脚本只检查 `meta/info.json` 是否存在。
训练结果写入 `${MY_DFS}/experiments/carrot/pi05_sft_so101_orange_cube`。

## 运行

在 Gemini 会话确认 `MY_DFS`、同步源码、检查 8 张 GPU 和 dguard，并启动
Ray 集群，设置 `RAY_ADDRESS`。确认上述模型、tokenizer 和数据集已下载后运行：

```bash
bash recipes/pi05_sft_so101_orange_cube/run.sh
```

脚本激活 `/opt/venvs/carrot`、设置源码 `PYTHONPATH`、强制离线加载，
并在输出目录已存在时停止。若换用其他 DFS 或输出目录，先复制并调整
本目录中的配置和脚本；续训需单独确认 checkpoint 格式并使用 `--resume`。

## 状态

2026-09-25 开跑前为 **NOT RUN**。本次运行记录如下；后续检查追加，保留历史状态。

## 2026-09-25 首次运行（进行中）

- Gemini 容器：`mpi-launcher@mpi-1759754893-launcher`，单节点 8 张 H20；`MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu`。2026-09-25 15:25 UTC 左右通过 `bash recipes/pi05_sft_so101_orange_cube/run.sh` 启动，后台任务 ID `1a17e72c-0036`。Ray 已连接，资源占用 8/8 GPU、40/224 CPU；dguard 暂停 2880 分钟。W&B 关闭。
- 数据下载到配置指定的本地目录，Hugging Face 本地 cache 的 tree 和 `meta/info.json.metadata` 均标识固定 revision `c021b3c22a3de4e70e81010e54fb250a5dde348b`。tree 中 13 个文件全部存在且大小匹配，合计 1,130,873,263 字节；`meta/info.json` 为 `so101_follower`、30 FPS、154 episodes、68,468 frames。base checkpoint 的远端 `config.json` 实测 `action_horizon=50`。
- 本地 Carrot HEAD 为 `d4d78ef`；远端同步副本不含 `.git`，已逐字节核对 SO101 adapter、recipe YAML 与 `run.sh` 的 SHA-256 与本地相同，不能据此宣称远端完整 checkout commit。实际容器镜像 tag 未核实。
- 2026-09-25 15:31 UTC 只读检查：后台任务仍为 running；Ray rank-0 worker 输出 `first forward ok loss=0.072266`，step 10 `loss=0.040472, grad_norm=0.198651`，step 20 `loss=0.030198, grad_norm=0.082005`，step 30 `loss=0.028381, grad_norm=0.079973`，LR 均为 `1e-5`。这些是早期有限数值，不代表 5000-step 完成或效果验收。
- 持久化日志快照：`${MY_DFS}/experiments/carrot/pi05_sft_so101_orange_cube_monitor/20260925T153144Z/rank0_worker_stdout.txt` 与同目录 `backend_task_1a17e72c-0036.txt`。Ray 的实时 worker 日志仍在 launcher 的 `/tmp/ray/session_latest/logs/`，属于临时盘；后台任务状态可用 task ID 查询。计划训练输出目录为 `${MY_DFS}/experiments/carrot/pi05_sft_so101_orange_cube`，截至该次检查尚不存在；首个 checkpoint 计划在 step 500。

待验收：持续核对有限 loss/grad、任务退出状态、step 500 及最终 checkpoint 的完整性，再分别验证模型加载、resume 和效果。不得把早期 step 或存在 checkpoint 当作 5000-step 训练完成。

### 2026-09-25 16:09 UTC：按用户要求延后约 30 分钟的只读检查

后台任务 `1a17e72c-0036` 仍为 running；Ray 单节点占用 8/8 GPU、40/224 CPU。rank-0 worker 已到 step 370，`loss=0.011971`、`grad_norm=0.063847`、LR `1e-5`，最近 step 300–370 的记录均为有限数值。训练输出目录仍不存在，故尚无 step-500 checkpoint；不宣称训练完成。证据为该任务状态、`/tmp/ray/session_latest/logs/worker-b61ac2308fa1506fc82f1110f213f8d67dd5d2d2d7a439e66390b74d-01000000-17659.out` 与 `ray status` 的 2026-09-25 16:09 UTC 只读输出；Ray 日志位于临时盘，早期持久化快照路径见上文。
