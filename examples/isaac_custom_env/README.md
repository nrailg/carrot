# 从零搭一个自己的机器人任务环境

先读这个文件，再读 [reach_env_cfg.py](reach_env_cfg.py) 和 [run.py](run.py)。
**最小起点是一个 Isaac Lab 环境；想组织多个机器人、场景和任务时，再用 Arena。**
本目录独立编写，只参考 NVIDIA 官方教程/上游源码，不依赖 Carrot 的环境封装或旧示例。

本例的任务是：**Franka Panda 把 `panda_hand` 刚体原点移动到随机目标点附近**。
它包含环境闭环，但没有训练算法、抓取策略或 LIBERO 数据集。
SO101 的替换步骤和官方完整项目入口见 [SO101.md](SO101.md)。

## 1. 三层分别负责什么

| 层 | 在本例里做什么 | 你需要写什么 |
| --- | --- | --- |
| Isaac Sim / PhysX | 加载 USD，处理关节、刚体、碰撞，按需渲染 | 资产、物理参数；Lab 会调用底层 |
| Isaac Lab | 并行场景、控制动作、观测、reset、reward、termination | `ReachEnvCfg` 中的各组配置 |
| Isaac Lab-Arena | 组合 Scene、Embodiment、Task，组织评测 | 可选的 `arena_adapter.py` |

这是本例采用的 Sim/PhysX 路径；较新的 Lab 还支持其他物理后端。
官方入门：[Lab 环境教程 S1/S2](SOURCES.md#s1)、[Arena 自定义组件 S8](SOURCES.md#s8)。

几个词先认识：USD 是场景/资产描述；prim 是场景树上的一个节点；articulation 是由
关节连接起来的机器人；cfg 是描述“要建立什么”的配置；manager 负责调用一组功能函数。
`@configclass` 让这些 Python 类成为可复制、可嵌套的配置对象。

```text
你的策略输出 action [N, J]
         ↓
JointPositionAction → 关节目标 → PD 驱动 → 4 个物理步
         ↓
读取状态 → 判成功/超时 → 算奖励 → 只重置结束的槽位 → 新 observation
```

这个时序来自官方 `ManagerBasedRLEnv.step()`，见 [S5](SOURCES.md#s5)。
`N=4` 表示一个进程内四份场景，不是四个独立 Python 仿真进程。

## 2. 先准备官方运行环境

需要支持 Isaac Sim 的 Linux/NVIDIA GPU 环境。依赖安装和 GPU 仿真没有在本次写作中执行。
本例查阅的 API 基线为 Isaac Lab `v3.0.0-beta2.patch1`，可选 Arena 为
`8737b4ceb25f99f81a81786b7fde73139b52f324`（`release/0.3.0` 的一个固定提交）。
这是**源码核对基线，不是实机通过声明**。旧版 Lab 2.x 的参数、四元数约定可能不同。

如果已经有匹配的 Isaac Python 环境，激活后直接进入下一节。否则：

- 只学 Lab：按 [官方 Lab 安装教程](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/setup/installation/pip_installation.html)
  准备 Isaac Sim 与 Lab，包含 `isaaclab_assets`、`isaaclab_tasks`。
- 同时学 Arena：可以使用下面的官方源码安装路线。它会安装较大的完整依赖集。
  命令依据 [Arena 安装教程](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/quickstart/installation.html)，
  加入固定提交是本例为便于复现作的调整。

在**独立目录**执行，不要把依赖装进已有项目环境：

```bash
mkdir -p ~/isaac-learning
cd ~/isaac-learning
git clone https://github.com/isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena
git checkout 8737b4ceb25f99f81a81786b7fde73139b52f324
git submodule update --init --recursive
uv sync --frozen
source .venv/bin/activate
# 阅读并同意 NVIDIA EULA 后设置；出处：上面的官方安装页。
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
```

这一路线使用 Arena 锁文件及其 Lab 子模块配套版本，不要另外覆盖安装另一个 Lab。
本目录没有自己的依赖项目，也无需 `pip install -e` 安装 Carrot。

## 3. 运行最小例子

下面命令里的相对路径从包含 `examples/` 的目录执行；也可以把本目录整体复制到别处。
使用上一步已激活环境中的 `python`。参数和启动顺序出处：[S0](SOURCES.md#s0)。

```bash
# 无窗口；默认 300 个控制步，约 10 秒仿真时间，实际耗时还包含启动。
python examples/isaac_custom_env/run.py --viz none --num_envs 1 --steps 300

# 有桌面/显示环境时，用 Kit 窗口看机械臂和目标坐标轴。
python examples/isaac_custom_env/run.py --viz kit --num_envs 1 --steps 600

# 理解批量接口后，再试一进程四份环境。
python examples/isaac_custom_env/run.py --viz none --num_envs 4 --steps 300
```

Panda USD 默认从官方资产地址获取，首次使用需要能够访问该地址。
已有同款完整本地资产时，可传 `--robot_usd /absolute/path/panda_instanceable.usd`。
USD 引用的子文件/材质也必须齐全；这个参数仅替换资产路径，不会转换机器人型号。
资产加载方式出处：[S3](SOURCES.md#s3)、[S6](SOURCES.md#s6)。

程序会打印关节名、刚体名、观测/动作 shape、奖励，以及成功/超时事件数量。
默认 Panda 应为 `policy=(N, 28)`、`action=(N, 7)`，控制间隔 `1/30 s`，每回合最多 150 步。
这些是配置推导的预期值，不是本次执行日志。

`wiggle` 只让第一关节缓慢摆动；`--policy hold` 给零动作。
**零动作意味着回到默认关节目标，不是零力矩，也不一定是保持当前关节位置。**
成功计数为零很正常；例子演示环境接口，尚未实现完成任务的控制器。
`DEMO_FINISHED` 仅表示循环结束。输出 NaN/Inf 会直接报错。

## 4. 按什么顺序读代码

每个主要代码块旁都有 `[S数字]`，对应 [SOURCES.md](SOURCES.md) 中的教程标题、链接和
具体源码符号。阈值、采样范围、摆动输入是本例设计，注释不会把它们说成官方推荐参数。

| 阅读顺序 | 代码 | 你要回答的问题 |
| --- | --- | --- |
| 1 | `ReachSceneCfg` | 场景里有什么？机器人从哪个 USD 加载？ |
| 2 | `ActionsCfg` | 策略输出的数字最终变成什么物理控制量？ |
| 3 | `CommandsCfg` | 这一回合让机器人到哪里？坐标属于哪个参考系？ |
| 4 | `ObservationsCfg` | 策略每一步能看到哪些状态？按什么顺序拼接？ |
| 5 | `EventsCfg` | 开始新回合时，哪些状态需要恢复？ |
| 6 | `RewardsCfg` / `TerminationsCfg` | 怎样评价行为？什么时候算成功或超时？ |
| 7 | `ReachEnvCfg` / `run.py` | 把这些组件装起来并调用 reset/step |

本例定义的合同：

| 项目 | 含义 |
| --- | --- |
| 动作 | 7 个 Panda 手臂关节，`q_target = q_default + 0.25 × a`；夹爪不受策略控制 |
| 观测 | 相对默认关节位置 7、相对默认速度 7、基座系目标位姿 7、上次动作 7 |
| 目标 | 基座系 x ∈ [0.4, 0.5]、y ∈ [-0.1, 0.1]、z ∈ [0.4, 0.5] 米 |
| 奖励 | `1 - tanh(distance / 0.1)`，RewardManager 还会乘权重和控制步长 dt |
| 成功 | hand **刚体原点**到目标距离小于 0.03 m，立即结束 |
| 超时 | 到达 5 秒仿真时间，返回 truncated |
| reset | 关节默认姿态加少量扰动、速度归零、重新采样目标 |

本例没有实现动作 `[-1, 1]` 的强制裁剪；演示输入自身有界，接策略时应明确动作范围。
目标中的四元数不参与奖励/成功判定。本例的 hand 原点也不等同于真实夹持点 TCP。
若要控制两指中间的点，需要定义 frame offset 或 FrameTransformer，见 [S7](SOURCES.md#s7)。

## 5. 再看 Arena 怎么组织同一个环境

已有 Arena 配套环境后运行：

```bash
# 参数 --arena 属于本例；底层组合方法出处为 S8/S9。
python examples/isaac_custom_env/run.py --arena --viz kit --num_envs 1 --steps 300
```

[arena_adapter.py](arena_adapter.py) 将同一组配置拆给三个对象：

- `Scene`：地面和灯光。
- `JointArmEmbodiment`：机器人资产、关节动作、机器人 reset。
- `ReachTask`：目标、观测、奖励、成功/超时、成功率指标。

`ArenaEnvBuilder` 合并它们，再建立 Lab 环境。先读懂 Lab 配置后，这一层更容易理解。
本例通过构造函数直接传组件，没有要求你注册全局资产或修改 Arena 上游仓库。
任务观测仍使用传入 cfg 的机器人绑定；它不是任意机器人自动兼容的策略接口。
布局是固定的，因此关闭关系求解；遥操作、Mimic 和相机尚未配置。
官方对照：[自定义 Task/Embodiment](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/arena_in_your_repo/external_tasks_and_embodiments.html)。

## 6. 从这个例子扩成“自己的 LIBERO”

先把需求写成：**某机器人在某场景中，依据某观测，接受某动作，完成某个可判定目标。**
例如 SO101 把红色方块放入托盘：

1. 准备机器人 USD；确认基座固定、关节名/限制、碰撞、惯量和驱动参数。
2. 创建桌面、方块和托盘；方块需要 `RigidObjectCfg`、质量和碰撞，外观 mesh 不够。
3. 先用关节控制确认机械臂能稳定运动，再加夹爪开合。其后才考虑位置 IK。
4. 定义观察：关节状态、物体位置，或外部/腕部相机 RGB。状态与图像策略合同不同。
5. reset 随机化物体位置，保证没有穿透并处于机器人可达区域。
6. 将成功定义成“方块在托盘内部、机器人已释放、速度足够低并持续若干步”。仅靠中心点
   接近，会把悬空或仍被夹着的状态算成功。
7. 接入脚本控制器/遥操作验证能完成一次，再接模仿学习或 RL；最后扩充任务集和评测种子。

这是搭建任务的建议顺序；API/抓取与感知参考见 [S6/S7/S8](SOURCES.md#s6)。
“任务不同”主要改物体、reset、目标和判据；“机器人不同”还必须改动作、观测、工作空间和驱动。
策略不会因为 Arena 能换模型，就自然跨机器人工作。

RL 接入还有一个边界：官方 `step()` 会自动 reset，done 槽位返回的是下一回合的观测。
本例没有提供 `final_observation` 适配器，也不声称已完成 PPO 训练接口。
如果学习器要对超时状态 bootstrap，必须在 reset **之前**保存最终观测，或使用已核对过
语义的官方学习器 wrapper，不能拿下一回合初始观测代替。[S5](SOURCES.md#s5)

## 7. 本次验证边界

运行环境中没有安装 `isaaclab`，所以本次只做 Python 语法、ruff 和上游接口静态核对。
没有运行 GPU 仿真、验证画面、验证成功可达性或训练策略；SO101 资产也未导入测试。
具体检查命令与结果见 [VALIDATION.md](VALIDATION.md)。
