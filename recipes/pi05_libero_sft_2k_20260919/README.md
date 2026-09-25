# PI0.5 + LIBERO SFT，2026-09-19

## 目的与配置

从 `pi05_base_pytorch` 初始化 PI0.5，在 `lerobot/libero` 上做正式 SFT，观察
训练 checkpoint 的 LIBERO 闭环表现。参数以
[`pi05_sft_libero_2k_20260919.yaml`](pi05_sft_libero_2k_20260919.yaml)
为准：8 GPU 单节点、2000 个计划 step、global batch 256、micro batch 4、
梯度累积 8、action horizon 10、恒定学习率 `1e-5`，每 100 step 保存。
模型由 run 目录 `init/` 中的 base 权重初始化；数据为本地 `lerobot/libero`，
归一化统计来自 `pi05_libero_pytorch`。W&B 配置已启用，凭据由运行环境提供。

## 运行方式

先在 Gemini 会话确认 `MY_DFS`、同步源码、检查 8 张 GPU 和 dguard，并启动
Ray 集群，使 `RAY_ADDRESS` 指向该集群。确认配置中的绝对路径与本次 DFS 一致后运行：

```bash
bash recipes/pi05_libero_sft_2k_20260919/run.sh
```

脚本激活 `/opt/venvs/carrot`、设置源码 `PYTHONPATH`、离线读取本地模型和数据，
并拒绝覆盖已有 `checkpoints/`。它是这次历史配置的运行入口；当前输出目录已有
checkpoint，因此直接重跑会被拒绝。新实验应复制配置并更换 `output_dir`、W&B 名称和
运行脚本中的目标目录；续训需单独使用 `--resume` 并验证 checkpoint 格式。

## 历史结果与证据

状态：**提前停止，未完成 2000 step**。历史记录显示训练在 log step 1470 后停止。
持久化目录 `${MY_DFS}/experiments/carrot/pi05_libero_sft_2k_20260919/checkpoints/`
存在 step 100–1400 的 checkpoint；step 1400 的 `trainer_state.json` 记录
`"step": 1400`，并有模型权重、优化器状态和 `train_config.yaml`。
本次没有 step 2000 checkpoint，也没有已记录的 resume 验收。

step 100、300、1000 的四套 LIBERO 闭环评测已完成，汇总成功数分别为
745/2000、1513/2000、1843/2000；这些是 checkpoint 评测结果，
不代表 2k 训练完成。评测命令、逐套结果与有效性检查保存在
[`tests/pi_05/test_libero.md`](../../tests/pi_05/test_libero.md)。评测产物位于
`${MY_DFS}/benchmarks/carrot-pi05-libero-sft-four-suites-20260919/`。

历史训练的实际启动命令、退出码、Docker image tag 和完整远端源码 commit 未记录，
上面的 `run.sh` 是根据现存配置整理的可重复入口，不冒充当时的原始命令。
