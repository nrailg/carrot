# SO101 arm dance：no vision + state jitter

## 2026-10-08：2000-step 训练配置（待开跑）

| 项目 | 设置 / 状态 |
|---|---|
| 目标 | 在新 arm dance 示教上验证无视觉、输入 state 增强的拟合能力 |
| 数据 | `nrailg/arm_dance_20261008_225311_20261008_225313`，1 episode / 144 frames / 15 FPS |
| 任务文本 | `Dance with the arm`，训练从数据集读取 |
| 初始化 | 官方 `pi05_base_pytorch_h10` 独立副本，从 step0 新训，不 resume |
| 输入增强 | 五运动轴 raw state 独立 U(-3°, 3°)，按共享 LeRobot 标定限幅；gripper/action 不改 |
| 视觉 | 所有图像 pixels=0、attention mask=false；复用 `DropVision` |
| 动作 / 随机性 | action horizon 10；每个 batch 随机 noise 和 t |
| 训练 | 2000 steps；warmup 100 后 constant LR 1e-6；seed 1000 |
| 分布式 | 8 GPU BF16 FSDP，micro batch 4 / global batch 64 / GAS 2 |
| 统计量 | 从这份 arm dance 原始数据计算；不复用 cylinder 的 norm_stats |
| 保存 | 每 500 steps，预期 step500/1000/1500/2000 |
| 输出 | `$MY_DFS/experiments/carrot/pi05_so101_arm_dance/arm_dance_20261008_225311_20261008_225313/` |
| 状态 | 已核对 Mac 原始数据与配置；尚未同步新数据、启动训练或运行评估 |
| 实际任务 / 环境 | 启动时间、Gemini/Ray task、运行源码 commit、Docker image tag / 上游 commit 均待实际运行记录 |

Mac 原始数据：
`~/.cache/huggingface/lerobot/local/arm_dance_20261008_225311_20261008_225313`。
远端预期数据根：
`$MY_DFS/hf-hub/nrailg/arm_dance_20261008_225311_20261008_225313`。
共享标定：
`$MY_DFS/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json`。

2026-10-08 本地核验：Parquet 共 144 行，frame_index 连续 0–143、episode_index 全为 0，
state/action 均为有限 float32[6]；关节顺序与 SO101 一致，腕部视频文件存在。
原始 `data/chunk-000/file-000.parquet` SHA256：
`752fb901eab1ddb0892ceee2ed4b68afe36fabdb971a805db40a28c3424f8629`。

配置核验 PASS：与旧 no-vision+jitter2000 的 YAML 仅数据 repo/root 和 output_dir 不同；
新数据元信息与相机字段匹配，`bash -n run.sh` 通过。尚未运行 GPU 训练或验证视频解码。

确认当前 MY_DFS、同步源码/完整数据/标定并核对 GPU、Ray、base 与 tokenizer 后运行：

```bash
export SOURCE_COMMIT=<已核验的本轮Mac源码commit>
bash recipes/pi05_sft_so101_arm_dance/run.sh
```

`MY_DFS` 和 `RAY_ADDRESS` 必须由运行环境明确设置。train.yaml 使用当前已有资源布局的
固定路径；若实际 MY_DFS 不同，先更新配置和 runner 的路径检查。runner 拒绝覆盖已有输出，
保存实际 train.yaml、calibration.json 和源码提交标识；暂停 dguard 120 分钟，退出恢复。
训练复用现有 fit-validation runner 与 state-jitter 数据封装，不新增 train.py 或模型改动。

验收以实际退出码、`SFT finished`、2000 条连续有限 loss/grad/LR、warmup/constant 曲线、
四个 checkpoint 的 step/horizon/权重/统计量/tokenizer/optimizer 完整性与参数更新为准。
效果评估另行执行；旧 cylinder rollout 评估写死 264 帧，不能直接用于本数据。
训练完成不等同于已拟合或真机成功。本 recipe 不操作机器人；生成的图像、数据和权重不入 Git。
