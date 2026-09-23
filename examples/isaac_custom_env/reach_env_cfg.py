"""先启动 AppLauncher，再导入本模块。代码块出处见 SOURCES.md 中的编号。

官方教程（基础配置、奖励/终止、机器人配置）：
https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/03_envs/create_manager_base_env.html
https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/tutorials/03_envs/create_manager_rl_env.html
https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/how-to/write_articulation_cfg.html
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_tasks.manager_based.manipulation.reach.mdp import (
    position_command_error,
    position_command_error_tanh,
)

# [S3, S4] 这些名字属于官方 Panda 资产；换机器人时由 configure_robot 一起替换。
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
EE_BODY = "panda_hand"


# [S2] TerminationTermCfg 接受返回 [num_envs] bool 的函数。
# [S4] 距离算法直接使用官方 reach 实现；3 cm 成功阈值是本例的任务设计。
def reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    return position_command_error(env, command_name, asset_cfg) < threshold


# [S1, S3] InteractiveSceneCfg 描述物理世界；{ENV_REGEX_NS} 为每份环境创建机器人。
# 地面是公共静态平面；机器人基座在地面上。这里没有桌子、抓取物和相机。
@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# [S1, S4] 关节位置动作：q_target = q_default + 0.25 * action，单位 rad。
# 不是关节力矩，也不是末端 xyz。preserve_order 保证输入顺序与 ARM_JOINTS 相同。
@configclass
class ActionsCfg:
    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
    )


# [S4] command 是“期望目标”，action 是“控制输入”，不要把二者混淆。
# 目标相对机器人基座采样。重采样间隔大于 episode，因此一回合内目标保持不变。
# debug_vis 画出目标/当前 body 的坐标轴；它们不是具有碰撞的实体。
@configclass
class CommandsCfg:
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_BODY,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.40, 0.50),
            pos_y=(-0.10, 0.10),
            pos_z=(0.40, 0.50),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# [S1, S4] 每个 ObservationTermCfg 是一个观测函数；policy 组按下面顺序拼接。
# 当前为 7 + 7 + 7 + 7 = 28 维；目标姿态中的四元数保留，但任务只评价位置。
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
        )
        goal = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        last_action = ObservationTermCfg(func=mdp.last_action)
        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# [S1, S4] reset 只作用于结束的环境槽位；关节回到默认姿态附近，速度归零。
# 新目标由 CommandManager 的 reset 生成，不需要在主循环里手动改目标。
@configclass
class EventsCfg:
    reset_robot = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.02, 0.02),
            "velocity_range": (0.0, 0.0),
        },
    )


# [S2, S4] reward 是学习信号；成功判据单独写在 TerminationsCfg。
# 本例使用官方距离奖励，std=0.1 m 和权重=1 是教学选择，并非调优结果。
@configclass
class RewardsCfg:
    reach = RewardTermCfg(
        func=position_command_error_tanh,
        weight=1.0,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]),
            "std": 0.1,
        },
    )


# [S2, S5] success -> terminated；超过时限 -> truncated；两者可能同一步为 True。
@configclass
class TerminationsCfg:
    success = TerminationTermCfg(
        func=reached_goal,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=[EE_BODY]),
            "threshold": 0.03,
        },
    )
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


# [S2] 将前面的配置交给官方 ManagerBasedRLEnv；无需重写 step 或物理循环。
@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=1, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4  # 120 Hz 物理 / 4 = 30 Hz 控制。
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0  # 150 次 env.step，目标只在 reset 时更新。
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.3, 0.0, 0.4)


# [S3, S4] 本例的替换接口：机器人相关的名字必须同步修改，不能只换 USD。
def configure_robot(
    cfg: ReachEnvCfg,
    robot: ArticulationCfg,
    arm_joint_names: list[str],
    ee_body: str,
    goal_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> None:
    """Replace the robot and its task bindings before constructing an environment.

    Parameters
    ----------
    cfg : ReachEnvCfg
        Modified in place, before managers resolve scene entities.
    robot : ArticulationCfg
        Must define a fixed base, valid initial joints and all required actuators.
    arm_joint_names : list[str]
        Ordered, exact names of controlled joints; excludes passive/mimic joints.
    ee_body : str
        Rigid body origin used as the target point, not an arbitrary USD Xform.
    goal_ranges : tuple
        Reachable x/y/z intervals in meters, relative to the robot base.
    """
    cfg.scene.robot = robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.arm.joint_names = arm_joint_names
    for term in (cfg.observations.policy.joint_pos, cfg.observations.policy.joint_vel):
        term.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=arm_joint_names, preserve_order=True
        )
    cfg.commands.ee_pose.body_name = ee_body
    ranges = cfg.commands.ee_pose.ranges
    ranges.pos_x, ranges.pos_y, ranges.pos_z = goal_ranges
    for term in (cfg.rewards.reach, cfg.terminations.success):
        term.params["asset_cfg"] = SceneEntityCfg("robot", body_names=[ee_body])
