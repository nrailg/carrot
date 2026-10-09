# 实验 recipes

`recipes/` 记录正式训练、评测和真机实验；`tests/` 记录代码与数据契约测试。
每个实验目录保留配置、运行入口和 README，详细参数、版本、任务 ID 与证据路径以各自档案为准。
本页是实验导航和阶段摘要，状态整理于 **2026-10-09**，不替代实时进程检查。

## 当前重点

目标顺序：拟合能力 → 简单条件下的真机 closed-loop replay → 泛化。
当前重点是 demo 附近的稳定推进和末端抖动；能够完成动作与稳定复现示教是不同结论。

| 实验 | 要验证什么 | 已记录进展 / 待解决问题 | 配置与入口 | 实验档案 |
| --- | --- | --- | --- | --- |
| Arm Dance：no vision + ±3° state jitter | 新单 episode 的拟合与真实 state 反馈 | 2000-step 训练完成；三轮 K=1 真机均 144/144、0 超时。录制 state 输入首动作 MAE 0.557°；最新执行跟踪 MAE 0.551°，模型对同序号示教 action MAE 6.616°，提前进入末态。稳定重放及末端抖动尚未解决 | [train.yaml](pi05_sft_so101_arm_dance/train.yaml) · [run.sh](pi05_sft_so101_arm_dance/run.sh) | [README](pi05_sft_so101_arm_dance/README.md) |
| Cylinder：frame prompt + no vision + ±5° state jitter | 给原 cylinder demo 增加显式帧进度，检验阶段条件能否改善推进 | 39 项 CPU 契约与真实 tokenizer 预检 PASS；2000-step 训练已启动。此档案最后检查为 1545/2000，最终训练验收与效果评估待补；不据旧进度宣称仍在运行 | [train.yaml](pi05_sft_so101_knock_down_the_cylinder_frame_prompt/train.yaml) · [run.sh](pi05_sft_so101_knock_down_the_cylinder_frame_prompt/run.sh) | [README](pi05_sft_so101_knock_down_the_cylinder_frame_prompt/README.md) |
| SO101 原始 action 回放与执行诊断 | 区分示教回放、模型目标偏差和舵机跟踪误差 | 原始 action 整段及 h10/K5 分块回放完成；cylinder 模型 h10/K1、wait 3°、超时继续已执行 264/264。等待后五轴跟踪 MAE 约 0.65°；末段模型腕屈目标往复已确认，现场抖动的各因素尚未隔离 | [固定目标诊断脚本](so101_action_replay/diagnose_tracking.py)；各轮实际执行协议见档案 | [README](so101_action_replay/README.md) |

Arm Dance 与 Cylinder 使用不同数据、prompt、统计量和 checkpoint，不能互换模型或合并结论。
当前 Arm Dance 模型没有 frame prompt；Cylinder 的帧条件实验不能当作 Arm Dance 的已验证解法。

## Cylinder 拟合与反馈实验

以下保留实验推进过程和对照结果，不表示需要重新启动全部实验。

| 实验 | 条件与用途 | 已记录结果 / 限制 | 配置与入口 | 实验档案 |
| --- | --- | --- | --- | --- |
| 完整单 episode overfit | 264 帧、腕部视觉、随机 noise/t、h10；官方 base 从 step0 训练 5000 步 | 训练及五个 checkpoint 的配对评估完成；用于完整条件下的拟合基线，不能据离线拟合宣称真机闭环成功 | [train.yaml](pi05_sft_so101_knock_down_the_cylinder_overfit/train.yaml) · [run.sh](pi05_sft_so101_knock_down_the_cylinder_overfit/run.sh) | [README](pi05_sft_so101_knock_down_the_cylinder_overfit/README.md) |
| 逐步拟合与推理诊断 | h10 / 无视觉 / 单帧 / 固定 noise，以及精度、缓存和 NFE 对照 | 保存独立核验、推理路径修复与 NFE 扫描记录；NFE 扫描只回答单次 diffusion process 内的去噪步数，后续连续 chunk 问题见下一项。简化消融不再追加 | [cases.yaml](pi05_sft_so101_fit_validation/cases.yaml) · [run.sh](pi05_sft_so101_fit_validation/run.sh) · [reevaluate.sh](pi05_sft_so101_fit_validation/reevaluate.sh) | [README](pi05_sft_so101_fit_validation/README.md) |
| 无视觉连续 action chunk 反馈 | h10/K5/NFE10；采用第 5 个预测 action 作为下一 state | 历史无增强 500-step 对照：真实 state 输入 MAE 1.347°，反馈 12.451°；后续 ±3° 增强 1000-step 复跑完成。当前配置已加入 jitter/标定 clamp，不代表旧基线使用了这些设置 | [train.yaml](pi05_sft_so101_state_only_rollout/train.yaml) · [run.sh](pi05_sft_so101_state_only_rollout/run.sh) | [README](pi05_sft_so101_state_only_rollout/README.md) |
| 无视觉 state jitter | 五轴 raw state ±3°，保留 action/gripper/stats；后续统一 SDK 标定 clamp | 旧 500-step 增强比较及新 2000-step 重训/评估完成。新 2000-step 真实 state / 反馈 MAE 为 1.192° / 10.313°；权重与预测复现旧未 clamp 的 2000-step 结果，未观察到 clamp 改变模型效果 | [train.yaml](pi05_sft_so101_state_jitter/train.yaml) · [run.sh](pi05_sft_so101_state_jitter/run.sh) | [README](pi05_sft_so101_state_jitter/README.md) |
| 匹配基础拟合后比较 jitter | 增强训练 2000 步、每 500 保存；按预先固定的基础拟合规则选 checkpoint | 四组评估完成；选中 step1000。数据增强的独立鲁棒性收益尚不能确认，预算、基础拟合和增强作用未完全分离 | [train.yaml](pi05_sft_so101_state_jitter_matched_fit/train.yaml) · [run.sh](pi05_sft_so101_state_jitter_matched_fit/run.sh) | [README](pi05_sft_so101_state_jitter_matched_fit/README.md) |

## LIBERO

| 实验 | 已记录状态 | 配置与入口 | 实验档案 |
| --- | --- | --- | --- |
| PI0.5 + LIBERO SFT，计划 2000 步 | 历史 log step1470 后提前停止，未完成 2000 步；step100/300/1000 的四套评测成功数分别为 745/1513/1843（各 2000 episodes） | [训练配置](pi05_libero_sft_2k_20260919/pi05_sft_libero_2k_20260919.yaml) · [run.sh](pi05_libero_sft_2k_20260919/run.sh) | [README](pi05_libero_sft_2k_20260919/README.md) |

## 记录与比较约定

- 新增实验或阶段变化时，同步维护本索引与对应 README；历史重跑追加记录，保留原始结果。
- 计划预算、实际训练完成、checkpoint 可加载、离线效果和真机执行分别验收，不能互相代替。
- 比较动作误差时明确输入 state 来源、参考是 action 还是 observation.state，以及 h10/K/NFE、noise 和 padding 口径；五运动轴用反归一化 degrees，gripper 原单位另列。除平均值外保留 per-servo P50/P90/max。
- frame/inference 序号相同不代表录制物理时间或任务进度相同；真机跟踪使用发送目标与等待后实测位置配对，不能用推理前 state 直接代替。
- 当前配置可能已更新；复现历史实验以当时快照、命令和版本为准。未知版本写“未记录”。
- checkpoint、日志、NPZ/CSV、视频和曲线保存在 Ceph 与 Mac 缓存；图片、数据和权重不提交 Git。
