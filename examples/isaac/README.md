# Isaac Sim、Isaac Lab 和 Isaac Lab-Arena 入门示例

这组示例按从底层到上层的顺序介绍 NVIDIA Isaac 生态：

`04_lightwheel_libero.py` 使用下文单独说明的 Lightwheel venv，不使用 01–03 的环境。

```text
Isaac Lab-Arena：组合机器人、场景和任务，并统一评测策略
        ↓
Isaac Lab：把仿真包装成适合机器人学习的批量环境
        ↓
Isaac Sim：创建三维世界，运行物理、渲染和传感器
```

这些文件是学习材料，不属于 Carrot 的运行依赖。项目支持两种互斥的运行环境：

- Gemini/Carrot 容器：使用容器预装的 `/opt/venvs/carrot`，这是远程服务器上的推荐方式。
- 独立 uv 环境：使用目录里的 `pyproject.toml` 和 `.venv`，适合本地或已准备好依赖缓存的
  Linux x86_64 机器。

不要在激活 `/opt/venvs/carrot` 后再执行 `uv run`。uv 会发现当前激活环境与项目的
`.venv` 不同，忽略已激活的环境并切换到 `.venv`；如果 `.venv` 尚未完整安装，运行就会
失败。两种环境请选择一种，不要混用。

在 Gemini/Carrot 容器中运行：

```bash
cd ~/work/carrot/examples/isaac
source /opt/venvs/carrot/bin/activate
test "$(command -v python)" = /opt/venvs/carrot/bin/python
```

在独立 uv 环境中运行：

```bash
cd ~/work/carrot/examples/isaac
uv sync --frozen --no-group arena
```

Isaac Sim 首次启动会要求接受 NVIDIA EULA：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

示例默认接受 EULA、使用 GPU 0，并从以下本地目录读取完整资产包：

```bash
/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/isaacsim_assets/Assets/Isaac/6.0
```

代码使用环境变量作为默认值；仍可在启动前设置 `CUDA_VISIBLE_DEVICES` 或
`ISAACSIM_ASSET_ROOT` 来选择其他 GPU 或资产目录。

## 01：Isaac Sim 让方块落下

文件：[01_isaac_sim_falling_cube.py](01_isaac_sim_falling_cube.py)

这个例子直接操作仿真器：创建场景、加入地面和刚体方块，然后逐帧推进仿真。

```bash
cd ~/work/carrot/examples/isaac
source /opt/venvs/carrot/bin/activate
python 01_isaac_sim_falling_cube.py --headless
```

使用独立 uv 环境时，将上面的运行命令替换为：

```bash
uv run --frozen --no-group arena python 01_isaac_sim_falling_cube.py --headless
```

远程服务器没有桌面窗口时，追加 `--headless`。

观察重点：

- `SimulationApp` 启动 Isaac Sim 应用。
- `Cube` 只创建方块的几何外形。
- `RigidPrim` 和 `GeomPrim` 分别为方块加入刚体和碰撞属性。
- `simulation_app.update()` 推进仿真和窗口渲染。

Isaac Sim 的插件在应用启动后才可用，因此必须先创建 `SimulationApp`，再导入其他
`isaacsim` 模块。这种看起来不常见的导入顺序是有意的。

## 02：Isaac Lab 同时运行多个 Cartpole

文件：[02_isaac_lab_parallel_cartpole.py](02_isaac_lab_parallel_cartpole.py)

这个例子创建多个 Cartpole 环境，并一次性向所有环境发送一批随机动作。

```bash
cd ~/work/carrot/examples/isaac
source /opt/venvs/carrot/bin/activate
python 02_isaac_lab_parallel_cartpole.py --num_envs 32 --viz none
```

使用独立 uv 环境时，将上面的运行命令替换为：

```bash
uv run --frozen --no-group arena python 02_isaac_lab_parallel_cartpole.py --num_envs 32 --viz none
```

观察重点：

- `num_envs=32` 表示一个进程中的 32 个并行环境，不是 32 个 Python 进程。
- `actions` 的第 0 维对应环境编号，所以一次 `env.step(actions)` 推进全部环境。
- Isaac Lab 负责 observation、action、reset、termination 等机器人学习接口。

Isaac Lab 3.0 默认无窗口运行；想在桌面观察时追加 `--viz kit`。环境数量可以逐渐改为
128、1024，观察 GPU 显存和仿真吞吐的变化。

## 03：Isaac Lab-Arena 组合场景和机器人

文件：[03_isaac_lab_arena_composition.py](03_isaac_lab_arena_composition.py)

这个例子从注册表选择厨房、Franka 和两个物体，再把它们组合成一个可运行的 Isaac Lab
环境。它特意不定义任务，方便先看清 Arena 的组合边界。

在 Gemini/Carrot 容器中，Arena 已包含在 `/opt/venvs/carrot`。直接运行：

```bash
cd ~/work/carrot/examples/isaac
source /opt/venvs/carrot/bin/activate
python 03_isaac_lab_arena_composition.py
```

使用独立 uv 环境时，先安装并启用 `arena` 依赖组，再运行：

```bash
uv sync --frozen --group arena
uv run --frozen --group arena python 03_isaac_lab_arena_composition.py
```

Isaac Lab-Arena 3.0 默认无窗口运行；想观察画面时追加 `--viz kit`。可以用
`--num_envs 4` 创建四份并行场景。

观察重点：

- `Embodiment` 是机器人以及它的动作、观测、控制器和传感器。
- `Scene` 是背景、物体和灯光的集合。
- `Task` 再描述目标、成功条件、终止条件和指标。
- `ArenaEnvBuilder` 最终把这些组件编译成 Isaac Lab 的环境。

实际的抓取任务会在组合时再传入 `task`：

```python
arena_environment = IsaacLabArenaEnvironment(
    name="pick_and_place",
    embodiment=robot,
    scene=scene,
    task=pick_and_place_task,
)
```

Arena 当前仍处于快速开发阶段，不同版本的资产名称和构造参数可能变化。遇到导入或参数
错误时，应首先确认 Isaac Lab-Arena、Isaac Lab 和 Isaac Sim 是否使用项目指定的配套版本。

## 推荐学习顺序

1. 修改方块的高度、大小和质量，理解 Isaac Sim 的场景与物理。
2. 修改 Cartpole 的 `num_envs`，理解 Isaac Lab 的批量 Tensor 接口。
3. 替换 Arena 示例里的机器人或物体，理解组合式环境的价值。
4. 最后再加入相机、任务、真实策略和指标，避免一开始同时面对所有抽象。

## 04：Lightwheel-LIBERO 任务

文件：[04_lightwheel_libero.py](04_lightwheel_libero.py)

[LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) 是 Lightwheel 基于
Isaac Lab-Arena 构建的任务与模型评测平台。Lightwheel-LIBERO 是其中的 130-task
套件，源码在 `lw_benchhub_tasks/lightwheel_libero_tasks/`。它与原始 MuJoCo LIBERO、
Rebecca 的 Arena LIBERO fork 是不同实现。

本例使用官方任务 `L90K1PutTheBlackBowlOnThePlate`、`libero-1-1` 布局和 `Panda`
机器人：创建环境、reset、读取状态、批量发送动作、读取任务成功信号，并保存 RGB。
上游 Panda 未预置 RGB 相机，因此本例显式添加一台 224×224 TiledCamera，
reset 后让它朝向本次采样的碗和盘子。

动作是 absolute IK 的位置、wxyz 四元数和夹爪，共 8 维。本例保持当前末端位姿并打开
夹爪，不加载模型，不会主动完成任务；输出的 `success_seen` 仅演示任务判定接口，
不是模型 benchmark 成绩。替换循环中的 `action` 即可接入匹配该动作定义的策略。

### 独立依赖环境

LW-BenchHub 当前固定栈使用 Isaac Sim 5.0；不要直接安装进 01–03 的 Isaac 6 / Carrot
环境。本次试运行使用独立 `/opt/venvs/lightwheel-libero`（Python 3.11.16），
Carrot 保持源码加载。上游代码和子模块固定在验证记录中的提交。Lightwheel SDK
固定为 1.0.1；更新的 1.0.2/1.0.3 与该提交的 `lightwheel_sdk.loader.ENDPOINT`
导入不兼容。Lightwheel 场景与物体通过 SDK 获取，NVIDIA 机器人资产也必须可访问。
远端访问 PyPI、NVIDIA 包索引和 Lightwheel API 时，需要为命令设置可用的 HTTPS 代理。

准备好依赖和资产后运行：

```bash
source /opt/venvs/lightwheel-libero/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES=0
python examples/isaac/04_lightwheel_libero.py \
  --headless --enable_cameras --num_envs 1 --steps 60 \
  --output "${MY_DFS:?Set verified Gemini DFS}/benchmarks/lightwheel-libero/example"
```

从 Carrot 根目录运行；输出为 `camera.png` 和 `result.json`。去掉 `--enable_cameras`
可只检查物理与状态接口。H20 上每张卡仅运行一个独立 renderer 进程。
本例的 `--num_envs` 是同一进程内的环境 batch；增大该值时动作的第 0 维也同步增大。
安装版本与本次实际验证状态见
[验证记录](../../tests/isaac/test_lightwheel_libero.md)。

### 用于 RL 的环境接口

`carrot_sim.isaac_libero.make_isaac_libero_env` 提供同任务的批量 RL 环境：
7D 增量动作、外部/腕部 RGB、稀疏 reward、独立 reset 和重置前最终观测。
`carrot_sim` 是可在上述 Python 3.11 环境直接加载的独立源码包，不导入 Carrot/Ray。
接口约定、timeout bootstrap 和真实 PPO smoke 的运行方式见
[RL 环境验证记录](../../tests/isaac_libero/test_isaac_libero.md)。
