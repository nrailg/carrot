"""可选的 Arena 组合课：同一组 Lab 配置如何交给 Scene / Embodiment / Task。"""

from typing import Any, override

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.task_base import TaskBase

from reach_env_cfg import ReachEnvCfg


# [S8, S9] EmbodimentBase 通过 scene/action/event_config 向 Builder 提供机器人配置。
@configclass
class RobotSceneCfg:
    robot: ArticulationCfg | None = None


class JointArmEmbodiment(EmbodimentBase):
    name = "tutorial_joint_arm"

    def __init__(self, cfg: ReachEnvCfg) -> None:
        super().__init__()
        self.scene_config = RobotSceneCfg(robot=cfg.scene.robot.copy())
        self.action_config = cfg.actions.copy()
        self.event_config = cfg.events.copy()


# [S8, S9] 完整实现当前 TaskBase 的必需方法；具体奖励和判据仍复用本例的 Lab 配置。
# 这里用已经绑定机器人名字的 cfg 传入任务观测，避免教学示例再引入观测适配层。
class ReachTask(TaskBase):
    def __init__(self, cfg: ReachEnvCfg) -> None:
        super().__init__(
            episode_length_s=cfg.episode_length_s,
            task_description="Move the hand body origin to the target position.",
        )
        self.cfg = cfg

    @override
    def get_scene_cfg(self) -> None:
        return None

    @override
    def get_events_cfg(self) -> None:
        return None

    @override
    def get_termination_cfg(self) -> Any:
        return self.cfg.terminations.copy()

    @override
    def get_observation_cfg(self) -> Any:
        return self.cfg.observations.copy()

    @override
    def get_rewards_cfg(self) -> Any:
        return self.cfg.rewards.copy()

    @override
    def get_commands_cfg(self) -> Any:
        return self.cfg.commands.copy()

    @override
    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    @override
    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> None:
        raise NotImplementedError("本例仅演示环境组合，没有实现 Mimic 数据生成。")


# [S8, S9] 对象用程序几何生成，不依赖额外的厨房或桌子资产下载。
def make_arena_env(cfg: ReachEnvCfg) -> ManagerBasedRLEnv:
    """Compose the tutorial configuration through Arena after AppLauncher has started.

    Parameters
    ----------
    cfg : ReachEnvCfg
        Fully bound robot/task configuration, before construction of managers.

    Returns
    -------
    ManagerBasedRLEnv
        Unwrapped Arena environment; the caller owns its close lifecycle.
    """
    ground = Object(
        name="ground", prim_path="/World/Ground", object_type=ObjectType.BASE,
        spawner_cfg=cfg.scene.ground.spawn.copy(),
    )
    light = Object(
        name="light", prim_path="/World/Light", object_type=ObjectType.BASE,
        spawner_cfg=cfg.scene.light.spawn.copy(),
    )

    # [S9] Builder 有自己的默认时步；显式复制，保证两个入口的控制频率相同。
    def configure_timing(arena_cfg: Any) -> Any:
        arena_cfg.sim = cfg.sim.copy()
        arena_cfg.decimation = cfg.decimation
        arena_cfg.viewer = cfg.viewer.copy()
        return arena_cfg

    description = IsaacLabArenaEnvironment(
        name="Tutorial-Reach-JointArm-v0",
        scene=Scene(assets=[ground, light]),
        embodiment=JointArmEmbodiment(cfg),
        task=ReachTask(cfg),
        env_cfg_callback=configure_timing,
    )
    builder = ArenaEnvBuilder(
        description,
        ArenaEnvBuilderCfg(
            num_envs=cfg.scene.num_envs,
            env_spacing=cfg.scene.env_spacing,
            device=cfg.sim.device,
            seed=cfg.seed,
            solve_relations=False,  # 固定布局，没有待求解的物体摆放关系。
        ),
    )
    return builder.make_registered().unwrapped
