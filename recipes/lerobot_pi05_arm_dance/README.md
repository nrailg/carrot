# LeRobot PI0.5 / Arm Dance 官方实现对照

2026-10-09：使用独立 venv 和官方 LeRobot 0.6.1，复查之前 Carrot Arm Dance 的训练与
离线拟合。全部新增代码局限于本 recipe；不 import Carrot，不修改已有代码、共享模型或数据。

数据 `nrailg/arm_dance_20261008_225311_20261008_225313`：单 episode、144帧、15FPS，
任务 `Dance with the arm`；五运动轴为角度，gripper 保持录制单位。
初始化 `$MY_DFS/hf-hub/lerobot/pi05_base`，tokenizer 使用已有本地
`$MY_DFS/hf-hub/google/paligemma-3b-pt-224`，全程离线。

## 对照条件与边界

- 无视觉：三个 image tensor 置零，所有 image mask 置 false。
- 训练 raw state 的五轴独立 uniform ±3°，按原 SO101 标定范围 clamp；gripper、action、统计量不变。
- h10、每次执行一步、seed1000、global batch64、2000step、每500step保存。
- AdamW LR1e-6、WD1e-10、betas[0.9,0.95]、eps1e-8、grad clip1；warmup100后LR恒定1e-6。
- 离线 play 用录制 state，不加 jitter，同样屏蔽视觉。

官方 PI0.5 没有关闭全部 image mask 的配置，因此通过官方 policy/processor 注册接口加载
本 recipe 的最小适配：`PI05ControlPolicy` 只覆盖图像输入预处理，`StateJitterStep` 只处理
归一化前 state。官方 transformer、attention、flow matching、loss、optimizer、训练循环及
动作推理方法未修改；不是不加适配的原生 CLI。每次预检核验 wheel 中490个Python文件的SHA。

仍有实现差异：原 Carrot 是8卡FSDP/micro4/GAS2，本次为官方8卡DDP/batch8；原版本有FP32
优化器主参数和FP32归约，本次遵循官方混合精度/优化器行为。原loss按32维并屏蔽尾部padding，
官方loss按真实6维、保留尾部重复动作；原state prompt的padding与官方可能不同。
本次初始化用LeRobot官方base，原用OpenPI h10转换版，尚未证明两份权重逐项一致。
上述差异均保留，不能仅凭loss数值判断Carrot是否有BUG。

## 环境、命令与验收

独立 venv `/opt/venvs/lerobot-official-arm-dance`，无 system-site-packages，未安装Carrot。
官方PyPI wheel `lerobot-0.6.1-py3-none-any.whl` SHA256：
`1894516040c65f80a45bd9741f8174aae90ed5d93da0627ab4f1a85fd8d75e90`。
依赖从已有环境按METADATA闭包独立复制，缺少/版本不符的包用官方wheel补齐；106包依赖检查通过。
包来源及安装记录在 `$MY_DFS/experiments/carrot/lerobot_official_arm_dance_packages/`。
Docker image tag及上游源码commit未记录；PyPI版本和wheel SHA可溯源。

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
export RUN_ID=<unique_run_id>
export SOURCE_COMMIT=<verified_commit>
bash "$MY_DFS/work/carrot/recipes/lerobot_pi05_arm_dance/run.sh"
```

独立输出 `$MY_DFS/experiments/carrot/lerobot_pi05_arm_dance/$RUN_ID/`：保存源码快照、
预检环境/数据/权重SHA、训练命令、exit code、官方checkpoint及离线预测/逐轴指标。
开始前核对GPU仅有dguard，runner暂停dguard并在退出时恢复；不覆盖任何旧run。
预检先检查真实数据、视频、quantiles、tokenizer与输入适配，然后用3帧base离线推理验证
812个权重完整加载与全false视觉mask，再进入2000step训练，最后对144帧做离线play。

训练成功需exit0/2000step/4个完整checkpoint/有限loss及gradient；checkpoint可加载和
离线拟合另行验收。离线报告首动作和有效chunk的逐轴MAE/P50/P90/max，排除尾部padding，
gripper独立统计。离线拟合不代表真机闭环成功；本runner不操作机器人。

## 历史与运行记录

原Carrot训练重新核验：exit0/2000step/四个完整checkpoint/8rank有限非零更新。
原tokenizer本地五文件无需模型config.json，两venv词表SHA与示例token一致。
之前使用Hub ID+空cache引发配置请求失败，属于本轮路径偏移，不能据此说旧tokenizer缺失。
之前本recipe草稿还错误使用了视觉与官方默认LR/WD，现已对齐上述条件。

- `smoke_20261009T122700Z`：预检失败（Hub tokenizer路径）。
- `smoke_20261009T123900Z`：预检失败（numpy统计量类型检查），已修正。
- `smoke_20261009T124300Z` / task816a3192-0695：CPU预检通过，base加载中按用户要求停止，
  exit-15；未进入训练、未操作机器人，dguard已恢复。
- `controls_cpu_20261009T134100Z` / task816a3192-0711：CPU预检exit0；490源码SHA、
  真实144帧数据、tokenizer、3路图像全零/全false mask、jitter/gripper不变性、106包依赖通过；五轴标定范围与原SDK逐项一致。
- 正式运行计划 `matched_20261009T123800Z`：branch/commit及同步SHA核验后启动2000step；
  本runner从run目录中的源码快照执行，避免运行中工作区变化影响实验。

参考：[官方训练/推理](https://huggingface.co/docs/lerobot/il_robots)、
[PI0.5](https://huggingface.co/docs/lerobot/pi05)。

- `matched_20261009T123800Z` / task816a3192-0714：12:37:00Z启动，CPU预检通过，
  官方加载全部key成功；base-play权重检查错误要求812归档张量=813个state-dict键，exit1。
  官方loader从lm_head恢复共享embedding别名，修正检查为逐张量相等+别名data_ptr相同，
  未进入训练，dguard恢复。补充完整CPU processor加载/归一化逆变换检查与显式PI0.5注册。
- `processor_cpu_20261009T124100Z` / task816a3192-0718：exit0；完整官方processor加载、
  state tokenization、action归一化逆变换通过。
- `matched_20261009T124200Z` / task816a3192-0720：144s后exit1，官方812张量加载成功，
  检查进一步发现embedding恢复为独立clone，非共享存储；改为缺省embedding从源lm_head逐值精确核验。
  CPU预检增加独立副本可通过、篡改副本必失败检查。仍未进入训练，dguard恢复。
- `weight_cpu_20261009T124500Z` / task816a3192-0721：exit0，embedding独立副本及故意篡改
  检查通过，完整processor/数据/源码预检仍通过。
