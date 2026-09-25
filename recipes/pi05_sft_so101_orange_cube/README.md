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

**NOT RUN**：此 recipe 仅完成配置和运行入口整理，没有训练任务 ID、
退出码或 checkpoint；正式运行后的环境、进度和结果应追加到本档案。
