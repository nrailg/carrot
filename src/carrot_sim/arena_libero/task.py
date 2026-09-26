from typing import Never, override

from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.task_base import TaskBase

from carrot_sim.arena_libero import mdp


@configclass
class ResetCfg:
    defaults = EventTermCfg(func=lab_mdp.reset_scene_to_default, mode="reset")
    bowl = EventTermCfg(
        func=lab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("bowl"),
            "pose_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025)},
            "velocity_range": {},
        },
    )
    plate = EventTermCfg(
        func=lab_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("plate"),
            "pose_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015)},
            "velocity_range": {},
        },
    )


@configclass
class TerminationsCfg:
    success = TerminationTermCfg(func=mdp.placed)
    dropped = TerminationTermCfg(func=mdp.dropped)
    time_out = TerminationTermCfg(func=lab_mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    success = RewardTermCfg(func=mdp.success_reward, weight=50.0)


class BowlOnPlateTask(TaskBase):
    def __init__(self, max_episode_steps: int) -> None:
        super().__init__(
            episode_length_s=max_episode_steps / 50,
            task_description="put the black bowl on the plate",
        )

    @override
    def get_scene_cfg(self) -> None:
        return None

    @override
    def get_termination_cfg(self) -> TerminationsCfg:
        return TerminationsCfg()

    @override
    def get_events_cfg(self) -> ResetCfg:
        return ResetCfg()

    @override
    def get_rewards_cfg(self) -> RewardsCfg:
        return RewardsCfg()

    @override
    def get_metrics(self) -> list[MetricBase]:
        return []

    @override
    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Never:
        raise NotImplementedError(
            "This environment provides RL transitions, not Mimic configuration"
        )
