# 每个代码块的出处

核对日期：2026-09-23。代码是根据下面 API 独立编写的教学组合，并非官方教程原文拷贝。
`[S0]` 等编号用于覆盖一个逻辑块，包括该块使用的 import；Python 参数解析、打印、
`try/finally` 和演示摆动输入是本例的普通 Python 组织代码。

Lab 固定源码基线：`v3.0.0-beta2.patch1`，commit
`ffff603eafc6b74264a5261cc0183d6a65390d78`。
Arena 固定源码基线：`8737b4ceb25f99f81a81786b7fde73139b52f324`。
文档链接可能随网站更新；有疑问时以这里的固定源码符号为准。

## S0

**启动 AppLauncher、CLI、导入顺序、应用关闭** → `run.py`。

- 教程：[Deep-dive into AppLauncher](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/00_sim/launch_app.html)。
- 示例：[run_cartpole_rl_env.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/scripts/tutorials/03_envs/run_cartpole_rl_env.py)，`AppLauncher`、`main`。
- API：[app_launcher.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/app/app_launcher.py)，`add_app_launcher_args`、`app`。

## S1

**Scene / actions / observations / reset events / configclass** → `reach_env_cfg.py`。

- 教程：[Creating a Manager-Based Base Environment](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/03_envs/create_manager_base_env.html)。
- 示例：[create_cartpole_base_env.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/scripts/tutorials/03_envs/create_cartpole_base_env.py)，`ActionsCfg`、`ObservationsCfg`、`EventCfg`。
- 场景教程：[Using the Interactive Scene](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/02_scene/create_scene.html)。

## S2

**奖励、终止、RL 配置与五元组返回值** → `RewardsCfg`、`TerminationsCfg`、`ReachEnvCfg`、主循环。

- 教程：[Creating a Manager-Based RL Environment](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/03_envs/create_manager_rl_env.html)，Defining rewards / terminations、Tying it all together。
- 配置源码：[manager_based_rl_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py)。
- 奖励乘 dt：[reward_manager.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/managers/reward_manager.py)，`RewardManager.compute`。
- 本例自定义：3 cm 成功阈值、5 秒回合、120/30 Hz、奖励权重、随机范围。

## S3

**机器人 ArticulationCfg / USD / 默认关节 / 驱动** → `ReachSceneCfg`、`custom_arm.py`。

- 教程：[Writing an Asset Configuration](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/how-to/write_articulation_cfg.html)。
- 教程：[Interacting with an articulation](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/01_assets/run_articulation.html)。
- Panda 定义：[franka.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab_assets/isaaclab_assets/robots/franka.py)，`FRANKA_PANDA_CFG`。
- USD 驱动参数继承：[actuator_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/actuators/actuator_cfg.py)，`stiffness`、`damping` 的 `None` 语义。

## S4

**机械臂 reach 的目标生成、观测、距离奖励与关节控制** → `reach_env_cfg.py`。

- 官方任务配置：[reach_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py)。这是任务源码，不是单独的入门教程。
- 距离函数：[mdp/rewards.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/mdp/rewards.py)，`position_command_error`、`position_command_error_tanh`。
- 动作公式：[joint_actions.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py)，`JointAction.process_actions`、`JointPositionAction`。
- 命令：[pose_command.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/envs/mdp/commands/pose_command.py)，`UniformPoseCommand`。
- 本例删去了姿态奖励，增加距离成功终止；`configure_robot` 是本例编写的集中替换函数。

## S5

**step 内部顺序、自动 reset、terminated/truncated** → `run.py` 与 README 的 RL 边界。

- 官方源码：[manager_based_rl_env.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)，`step`、`_reset_idx`。
- 终止项：[terminations.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/envs/mdp/terminations.py)，`time_out`。
- 学习器包装：[Wrapping environments](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/how-to/wrap_rl_env.html)。

## S6

**自己的机器人/物体从哪里来，如何导入** → `custom_arm.py`、`SO101.md`。

- 教程：[Importing a New Asset](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/how-to/import_new_asset.html)，URDF/MJCF/USD。
- 固定基座源码：[schemas_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py)，`ArticulationRootPropertiesCfg.fix_root_link`。
- 原机器人项目：[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)。必须辨别 SO100/SO101、leader/follower 和具体资产版本。

## S7

**下一课：TCP、IK、相机、接策略**，本例未实现这些模块。

- 教程：[Using a task-space controller](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/05_controllers/run_diff_ik.html)。
- 教程：[Adding sensors on a robot](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/04_sensors/add_sensors_on_robot.html)。
- 传感器：[Frame Transformer](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/core-concepts/sensors/frame_transformer.html)。
- 教程：[Registering an Environment](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/03_envs/register_rl_env_gym.html)、[Training with an RL Agent](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/03_envs/run_rl_training.html)。

## S8

**Arena 的 Scene / Embodiment / Task 三组件** → `arena_adapter.py`。

- 教程：[First Arena Environment](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/quickstart/first_arena_env.html)。
- 教程：[Your Own Tasks and Embodiments](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/arena_in_your_repo/external_tasks_and_embodiments.html)。
- 概念：[RL Tasks](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/concepts/task/concept_rl_tasks_design.html)。
- 本例不是拷贝教程里的“步数到了就成功”；它把本例真实距离判据交给 `TaskBase`。

## S9

**Arena 适配器具体方法与类型** → `arena_adapter.py` 的所有类/方法。

- [task_base.py](https://github.com/isaac-sim/IsaacLab-Arena/blob/8737b4ceb25f99f81a81786b7fde73139b52f324/isaaclab_arena/tasks/task_base.py)：需实现的 getter。
- [embodiment_base.py](https://github.com/isaac-sim/IsaacLab-Arena/blob/8737b4ceb25f99f81a81786b7fde73139b52f324/isaaclab_arena/embodiments/embodiment_base.py)：scene/action/event_config。
- [object.py](https://github.com/isaac-sim/IsaacLab-Arena/blob/8737b4ceb25f99f81a81786b7fde73139b52f324/isaaclab_arena/assets/object.py)：`Object` 的 `object_type` / `spawner_cfg`。
- [arena_env_builder.py](https://github.com/isaac-sim/IsaacLab-Arena/blob/8737b4ceb25f99f81a81786b7fde73139b52f324/isaaclab_arena/environments/arena_env_builder.py)：`compose_manager_cfg`、`make_registered`、`env_cfg_callback`。
- [arena_env_builder_cfg.py](https://github.com/isaac-sim/IsaacLab-Arena/blob/8737b4ceb25f99f81a81786b7fde73139b52f324/isaaclab_arena/environments/arena_env_builder_cfg.py)：布局/数量/设备参数。
- [success_rate.py](https://github.com/isaac-sim/IsaacLab-Arena/blob/8737b4ceb25f99f81a81786b7fde73139b52f324/isaaclab_arena/metrics/success_rate.py)：reset 前读取名为 `success` 的终止项。

## S10

**SO101 的官方完整教学项目** → `SO101.md`。

- [SO101 教程](https://isaac-sim.github.io/IsaacLab/develop/source/setup/tutorial.html)。
- [IsaacLabTutorial 项目](https://github.com/isaac-sim/IsaacLabTutorial)。
- [官方 SO101_CFG](https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab_assets/isaaclab_assets/robots/so101.py)。
- [完整放置任务配置](https://github.com/isaac-sim/IsaacLabTutorial/blob/main/src/isaaclab_tutorial/tasks/place_vial/config/so101/env_cfg.py)。

S10 是核对当天的 `develop/main` 资料，使用的运行栈与本目录固定的 Beta 2 API 不同。
它是后续学习路线，不代表能直接把最新 `SO101_CFG` 导入本例的旧版安装。
