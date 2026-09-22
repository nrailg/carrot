# Isaac Lightwheel LIBERO RL 环境

首个任务：`L90K1PutTheBlackBowlOnThePlate`，`libero-1-1` 场景、Panda。
复用 LW-BenchHub 场景和成功判定；不属于原版 MuJoCo LIBERO benchmark。

## 使用和接口

使用独立 `/opt/venvs/lightwheel-libero`，`carrot_sim` 通过 `PYTHONPATH=src:tests` 加载。
它是独立源码包：不导入需要 Python 3.12 和 Ray 的 `carrot` 核心，兼容现有 Isaac Python 3.11。
Carrot 主包的安装要求仍是 Python 3.12，不要在 Isaac venv 内安装 Carrot。
不要使用 Carrot 的 `isaac-arena` extra：该 extra 对应另一套 Isaac Sim 6 环境。
先创建启用相机的 `AppLauncher`，然后在模块顶层 import：

```python
from carrot_sim.isaac_libero import make_isaac_libero_env
from carrot_sim.libero import IsaacLiberoConfig

env = make_isaac_libero_env(IsaacLiberoConfig(num_envs=4))
try:
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(action)
finally:
    env.close()
    simulation_app.close()
```

| 内容 | 契约 |
|---|---|
| action | float32 `[N,7]`，范围 `[-1,1]`；越界显式报错 |
| 前 6 维 | 机器人基座坐标系下的位置/旋转向量增量；默认每分量 0.02 m / 0.1 rad |
| 夹爪 | 负数张开，零及正数闭合 |
| controller | 原生 differential IK + joint PD；每步相对当前 TCP 设目标，不等价于 OSC |
| 控制频率 | 沿用 pinned LW 配置，物理 100 Hz / decimation 2，即控制 50 Hz |
| `obs['policy']['state']` | float32 `[N,8]`：基座系 TCP xyz、rotvec、两个 finger joint position |
| `image` / `wrist_image` | `obs['policy']` 中的 uint8 `[N,256,256,3]`，外部/腕部 RGB |
| `obs['critic']` | float32 `[N,52]`：state 8、joint pos/vel 18、碗/盘各 13D 位姿/速度；特权信息 |
| `info['prompt']` | N 个任务文本；接 PI0.5 时显式映射到它的单样本输入 |
| reward | float32 `[N]`，Lightwheel 判定成功为 1，否则 0 |
| terminated / truncated | bool `[N]`，成功终止 / 时间上限；可能同时为真 |
| 自动重置 | same-step：返回 obs 的 done 槽已经属于下一回合 |
| final observation | `info['final_observation']` 是重置前完整 batch 的快照，只有 `_final_observation` 为真的槽有效；无 done 时为 None |
| episode stats | `info['episode']['r'/'l']`，仅 `_episode` mask 为真的槽已结束 |

外部和腕部相机、state 都是本任务配置，不保证与原版 LIBERO 的预训练策略直接匹配。
返回张量拥有独立存储，可安全保存到 rollout buffer。`reset(env_ids=int64_tensor)` 只重置指定槽；
首次必须全量 reset。seed 在创建时传给上游；不提供会假装重置所有上游 RNG 的 reset-time seed。
LW 上游在一次 reset 中可能为多个槽采样同一个物体布局，不宣称每槽随机流独立。
Panda 基座固定安装于 0.75 m 高度；上游通用的地面安装位置会使初始机械臂与台面冲突。
腕部相机固定于 hand 局部 `(0.1, 0, 0.05)`，ROS 四元数
`(0.90777199, -0.00438587, -0.37390275, 0.19007240)`，不会随物体位置动态瞄准。

## RL 边界

`delta = reward + gamma * (~terminated) * V(actual_next_obs) - V(obs)`。
done 槽的 `actual_next_obs` 必须取 final observation；GAE 递推在
`terminated | truncated` 处断开。`info['time_outs'] = truncated & ~terminated`。
不能既手工 bootstrap 又让训练框架重复补偿 timeout。

稀疏 reward 可用于 RL，但 smoke 不证明随机初始化策略可以快速学会任务。
验证用小型视觉特征 actor + 特权 critic 执行两轮 PPO 更新，不接入分布式训练或宣称 PI0.5 RL 完成。
本任务沿用上游成功定义：碗/盘水平距离小于阈值且夹爪离碗足够远；它不检查稳定接触，
所以最终学得行为仍须视频验收，不能把该 reward 当作严格物理放置的证明。

## 执行与验收

先动态确认 `MY_DFS`、同步源码、检查 dguard 和 GPU。H20 每张卡最多一个 renderer；
本 runner 用单个 Isaac 进程承载 4 个环境和两组 TiledCamera。
资产必须已经准备好，runner 不安装依赖或下载权重。

```bash
export ISAAC_LIBERO_OUTPUT="${MY_DFS}/benchmarks/isaac-libero-rl/<unique-run>"
bash tests/isaac_libero/test_isaac_libero.sh
```

- CPU：动作/重置参数校验、夹爪方向、张量快照、成功与超时、手算 GAE、PPO 更新。
- GPU：零动作 TCP 稳定性、双相机形状/像素、非目标槽 reset 隔离、真实超时与 final observation、真实 rollout 更新。
  另用状态注入 fixture 触发上游成功 term，验证 reward/终止/缓存清理；它不是策略成功率。
- 检查退出码、`result.json`、`image.png` / `wrist_image.png`，人工确认视角和物体入画。
- 仿真结束后检查无残留 renderer、无新增 GPU Xid，并恢复测试前 dguard 状态。

## 2026-09-22

- CPU：PASS，19 tests，Python 3.12.13 / torch 2.11.0+cu130，使用 CPU 张量。
  19 项也在实际 Lightwheel Python 3.11.16 / torch 2.7.0+cu126 环境通过（2.46 s）；
  第 19 项覆盖 Isaac 历史 success 不应重复发放 reward。
  远端证据：`${MY_DFS}/benchmarks/isaac-libero-rl/unique/cpu_pytest.stdout`。
- GPU 首轮：未完成。任务 `4c8a6a15-0461` 返回 exit 0，但无 result.json；SDK 查询
  `Microwave039` 资产 registry 时直连 Lightwheel API 超时，未进入环境 rollout。
  证据：`${MY_DFS}/benchmarks/isaac-libero-rl/unique/runner.stdout`。
  runner 已补独立产物校验及旧完成标记拒绝，防止 Kit 的 exit 0 掩盖失败。
- GPU0 独立 CUDA 张量测试通过；首轮后无残留 renderer，dmesg 检索无 NVIDIA Xid，
  dguard 恢复 `DGUARD_WATCH=1` 且取消临时恢复计划。
- 当前 H20 环境使用 Isaac Sim 5.0.0.0、Isaac Lab 0.47.3、Arena 1.0.0、SDK 1.0.1、
  Gymnasium 1.2.1。P5000 没有该独立环境，未改动其已有 Isaac Sim 6 进程或依赖。
- 目标上游版本沿用已有 Lightwheel 示例：LW `b2bcb2d00edef691f9fcc49039cbf0bcc7464605`、
  Arena `c7b70779f103e10d690d1a13863e8d77da7fc782`、
  Isaac Lab `6acdd82a1633732d32bb575e3d792e34fdeb437e`；运行时版本另行记录。
- 当前修改基于 Carrot `3d0603ad10df4d0e6cd6611083966a73248d049b`，尚未提交。
  远端没有 Git metadata；已独立验证 step/reset、Panda、任务基类源码与上述目标版本的哈希一致。
  Docker image tag 未记录。

### 获得下载授权后的真实 RL 验证

- 使用已有 `/tmp/set_proxy.sh` 后，SDK 元数据访问成功。
- 前几轮 RL 接口和参数更新检查通过，但人工看图发现腕部视角异常。
  进一步记录关节位姿，定位到上游将 Panda 基座放在地面，初始机械臂与台面碰撞。
  修正为固定 0.75 m 安装高度，并增加 10 步零动作 TCP 漂移小于 2 cm 的检查。
- `20260922-proxy-05`：修正后姿态稳定性、partial reset、final observation、成功状态 fixture
  和 PPO 更新均 PASS；零动作下关节最大变化约 1e-7 rad。随后按真实位姿校准固定腕部相机朝向。
- 最终配置 `20260922-proxy-06`（task `748af0bb-0523`）：真实仿真及 PPO 更新 PASS。
  H20 GPU0 单 renderer，4 env × 32 steps = 128 transitions，8 次 timeout。
  actor / critic 最大参数变化分别为 `0.0006004050374031067` / `0.0005995705723762512`，
  loss 为 `-0.01189767848700285`。partial reset、final observation、success fixture、
  零动作稳定性检查均通过。人工确认外部与腕部画面都有碗盘，腕部还能看见夹爪。
- 随机 rollout 的 reward_sum=0；本验收证明仿真到 PPO 更新的数据通路，不证明任务收敛。
- 最终证据：`${MY_DFS}/benchmarks/isaac-libero-rl/20260922-proxy-06/`，
  包括 result.json、runner.stdout、camera_poses.json、image.png、wrist_image.png。
  相机世界位姿由物理 hand 位姿与固定外参计算，避免读取 Fabric 下滞后的 USD 位姿缓存。
- 已核对本地/远端最终 backend SHA256：
  `6a7579074258147b65a701f23c22debde57752ad2acf43e15e88f0dfb6f1bea1`。

最终进程 exit 0，独立产物 validator PASS，同轮 CPU 19 项通过（2.68 s）。
复核无残留 renderer、dmesg 无 Xid；dguard 已恢复 `DGUARD_WATCH=1`，无待执行恢复计划。

## Python 3.12 兼容性验证（2026-09-22）

目标：在现有 Carrot Python 3.12 环境运行相同的真实 RL 验收，随后检查 Ray worker 集成。
候选环境 `/opt/venvs/carrot`：Python 3.12.13、Isaac Sim 6.0.1.0、
Isaac Lab 3.0.0b2.post1、Arena 0.3.0、torch 2.11.0+cu128、Ray 2.58.0、SDK 1.0.3。
当前状态：STOPPED，按用户要求停止兼容尝试；Python 3.12 / Isaac Sim 6 尚未通过 GPU RL 验证，不能沿用此前 Python 3.11 的 GPU PASS。

```bash
export ISAAC_LIBERO_VENV=/opt/venvs/carrot
export ISAAC_LIBERO_ARENA_ROOT="${MY_DFS}/work/LW-BenchHub/third_party/IsaacLab-Arena"
export ISAAC_LIBERO_OUTPUT="${MY_DFS}/benchmarks/isaac-libero-rl/<unique-py312-run>"
bash tests/isaac_libero/test_isaac_libero.sh
```

仍需先确认 GPU/dguard 与资产元数据网络，单个 H20 GPU 仅一个 renderer。

Ray 验证入口为 `tests/isaac_libero/ray_check.py`：在同一 Python 3.12 / Ray 版本的
Carrot `Worker` 进程内部启动 AppLauncher、执行完整仿真/PPO 验收，再次调用 worker
确认进程仍存活，正常关闭后才写 `ray_result.json`。当前尚未运行。

Docker 评估：`docker-images/gpu/carrot` 已锁定 Isaac 6 / Python 3.12 基础栈。
LW-BenchHub 的 `pin-pink==3.1.0` 与现有锁的 3.3.0 不一致，不能无约束安装上游依赖。
本轮只安装其源码 entry points（`--no-deps`），先验证已有依赖；不构建或发布镜像。

兼容性阻塞和修复记录：

- `20260922-py312-01/02`：Lab 3 已删除 teleop `DEVICE_MAP`，LW 导入时执行的旧补丁失败。
  runner 现会在 Kit close 前打印异常，避免 fast shutdown 掩盖根因。
- `20260922-py312-03`：`isaaclab.utils.dataclass` 不再导出，改用标准库 `dataclasses.dataclass`。
- 试验补丁见 `lightwheel_lab3.patch`（作用于 LW-BenchHub 上述固定 commit）。
  Lab 3 保留其原生 reset/step/termination/device 实现，仅保留 LW configclass 循环校验补丁；
  PhysX 配置由 `sim.physx` 适配到 `sim.physics`。
- LIBERO 进程的 `PYTHONPATH` 前置 LW 的 `third_party/IsaacLab-Arena` 源码根，
  **不添加其中的 IsaacLab 子模块**。环境内安装的 Arena wheel 仍是 0.3.0，但此 worker
  实际加载固定 commit `c7b70779f103e10d690d1a13863e8d77da7fc782` 的源码；二者不可混称。
